"""
Pipeline orchestrator (T6.3).

Connects ingestion → extraction → normalization → verification → reporting.

Each stage is wrapped in the {status, data, error} contract: a failed stage
returns a partial report with a HIGH-severity flag rather than crashing.

Public API:
    run(claim_id, transcript_path, document_path, backend, judge) -> PipelineResult
    run_from_text(claim_id, transcript_text, document_text, backend, judge) -> PipelineResult
    PipelineResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from claim_verifier.backends import LLMBackend
from claim_verifier.judge import DiagnosisJudge
from claim_verifier.models import Flag, VerificationResult
from claim_verifier.redaction import redact
from claim_verifier.stages.extraction import ExtractionError, extract
from claim_verifier.stages.ingestion import IngestionError, ingest_document, ingest_transcript
from claim_verifier.stages.normalization import normalize
from claim_verifier.stages.reporting import render
from claim_verifier.stages.verification import verify

_SEVERITY_RANK: dict[str, int] = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run."""

    claim_id: str
    report: str  # Markdown report — always present, even on failure
    errors: list[str] = field(default_factory=list)
    verification_result: VerificationResult | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_result(claim_id: str, flag_type: str, message: str) -> VerificationResult:
    """Build a minimal INSUFFICIENT_DATA VerificationResult for stage failures."""
    return VerificationResult(
        claim_id=claim_id,
        verified_at=_now(),
        consistency_score=None,
        risk_level="INSUFFICIENT_DATA",
        field_verdicts={},
        flags=[Flag(type=flag_type, severity="HIGH", message=message)],
        missing_fields=[],
        low_quality_sources=[],
    )


def _sort_flags(flags: list[Flag]) -> list[Flag]:
    return sorted(flags, key=lambda f: _SEVERITY_RANK.get(f.severity, 3))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    claim_id: str,
    transcript_path: str | Path,
    document_path: str | Path,
    backend: LLMBackend,
    judge: DiagnosisJudge,
) -> PipelineResult:
    """Full pipeline from file paths (stages 1–5)."""
    # Stage 1: Ingestion
    try:
        transcript_text = ingest_transcript(transcript_path)
    except IngestionError as exc:
        error_msg = f"Transcript ingestion failed: {exc}"
        vr = _error_result(claim_id, "INGESTION_FAILURE", error_msg)
        return PipelineResult(
            claim_id=claim_id,
            report=render(vr),
            errors=[error_msg],
            verification_result=vr,
        )

    try:
        document_text = ingest_document(document_path)
    except IngestionError as exc:
        error_msg = f"Document ingestion failed: {exc}"
        vr = _error_result(claim_id, "INGESTION_FAILURE", error_msg)
        return PipelineResult(
            claim_id=claim_id,
            report=render(vr),
            errors=[error_msg],
            verification_result=vr,
        )

    return run_from_text(claim_id, transcript_text, document_text, backend, judge)


def run_from_text(
    claim_id: str,
    transcript_text: str,
    document_text: str,
    backend: LLMBackend,
    judge: DiagnosisJudge,
) -> PipelineResult:
    """
    Pipeline starting from already-ingested text strings (stages 2–5).

    Bypasses the ingestion stage — useful for testing and for callers that
    obtain text through means other than file I/O.
    """
    # Stage 2: PII redaction before LLM calls
    redacted_transcript = redact(transcript_text)
    redacted_document = redact(document_text)

    # Stage 2: Extraction
    try:
        transcript_fs = extract("transcript", claim_id, redacted_transcript, backend)
    except ExtractionError as exc:
        error_msg = f"Transcript extraction failed: {exc}"
        vr = _error_result(claim_id, "EXTRACTION_FAILURE", error_msg)
        return PipelineResult(
            claim_id=claim_id,
            report=render(vr),
            errors=[error_msg],
            verification_result=vr,
        )

    try:
        document_fs = extract("document", claim_id, redacted_document, backend)
    except ExtractionError as exc:
        error_msg = f"Document extraction failed: {exc}"
        vr = _error_result(claim_id, "EXTRACTION_FAILURE", error_msg)
        return PipelineResult(
            claim_id=claim_id,
            report=render(vr),
            errors=[error_msg],
            verification_result=vr,
        )

    # Stage 3: Normalization
    norm_t, t_flags = normalize(transcript_fs)
    norm_d, d_flags = normalize(document_fs)

    # Stage 4: Verification
    vr = verify(claim_id, norm_t, norm_d, judge)

    # Merge normalization flags (re-sort by severity)
    extra = t_flags + d_flags
    if extra:
        vr = vr.model_copy(update={"flags": _sort_flags(vr.flags + extra)})

    # Stage 5: Report
    report = render(vr)

    return PipelineResult(
        claim_id=claim_id,
        report=report,
        errors=[],
        verification_result=vr,
    )
