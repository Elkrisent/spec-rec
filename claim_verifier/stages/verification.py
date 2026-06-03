"""
Stage 4 — Verification + scoring.

Compares two normalized FactSets field-by-field using typed matchers,
produces binary per-field scores, and returns a VerificationResult.

Scoring formula:
    field_score(f) = 1.0 if MATCH else 0.0
    consistency_score = Σ weight(f)·field_score(f) / Σ weight(included fields)
    "included" = value present in both sources; MISSING ⇒ flagged + excluded
    no included fields ⇒ INSUFFICIENT_DATA (consistency_score is None)

Risk bands (from config): score ≥ 0.90 → LOW; 0.70–0.89 → MEDIUM; < 0.70 → HIGH.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import datetime, timezone
from typing import Union

from rapidfuzz import fuzz

from claim_verifier.config import (
    AMOUNT_TOLERANCE_PCT,
    DATE_TOLERANCE_DAYS,
    FUZZY_HOSPITAL_THRESHOLD,
    LOS_TOLERANCE_ABS,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    load_schema,
)
from claim_verifier.judge import DiagnosisJudge
from claim_verifier.models import FactSet, FactValue, FieldVerdict, Flag, VerificationResult

# ---------------------------------------------------------------------------
# Internal: matchers
# ---------------------------------------------------------------------------

def _match_fuzzy_string(a: str, b: str) -> tuple[str, str]:
    ratio = fuzz.token_sort_ratio(a, b) / 100.0
    if ratio >= FUZZY_HOSPITAL_THRESHOLD:
        return "MATCH", f"fuzzy match, ratio={ratio:.2f}"
    return "MISMATCH", f"fuzzy mismatch, ratio={ratio:.2f}"


def _parse_for_date_cmp(s: str) -> _date:
    """Parse ISO YYYY-MM-DD or year-absent --MM-DD (using year 2000 as anchor)."""
    if s.startswith("--"):
        mm, dd = s[2:].split("-")
        return _date(2000, int(mm), int(dd))
    return _date.fromisoformat(s)


def _match_date(a_str: str, b_str: str) -> tuple[str, str]:
    a_absent = a_str.startswith("--")
    b_absent = b_str.startswith("--")
    year_note = " (year absent in one source)" if (a_absent or b_absent) else ""

    try:
        if a_absent or b_absent:
            # Strip both to --MM-DD so both use the same year-2000 anchor.
            # For YYYY-MM-DD: [5:] gives "MM-DD"; prepend "--".
            a_md = a_str if a_absent else "--" + a_str[5:]
            b_md = b_str if b_absent else "--" + b_str[5:]
            a = _parse_for_date_cmp(a_md)
            b = _parse_for_date_cmp(b_md)
        else:
            a = _parse_for_date_cmp(a_str)
            b = _parse_for_date_cmp(b_str)
    except (ValueError, IndexError) as exc:
        return "MISMATCH", f"date parse error: {exc}"

    delta = abs((a - b).days)
    if delta <= DATE_TOLERANCE_DAYS:
        return "MATCH", f"date match{year_note}, delta={delta}d"
    return "MISMATCH", f"date mismatch{year_note}, delta={delta}d"


def _match_numeric(
    a: Union[str, int, float],
    b: Union[str, int, float],
    field_id: str,
) -> tuple[str, str]:
    if field_id == "length_of_stay":
        delta = abs(int(a) - int(b))
        if delta <= LOS_TOLERANCE_ABS:
            return "MATCH", f"LOS match, delta={delta}d"
        return "MISMATCH", f"LOS mismatch, delta={delta}d"
    # billed_amount — percentage tolerance
    fa, fb = float(a), float(b)
    max_v = max(fa, fb)
    if max_v == 0.0:
        return "MATCH", "both zero"
    pct = abs(fa - fb) / max_v
    if pct <= AMOUNT_TOLERANCE_PCT:
        return "MATCH", f"amount match, delta={pct:.1%}"
    return "MISMATCH", f"amount mismatch, delta={pct:.1%}"


# ---------------------------------------------------------------------------
# Internal: flag helpers
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# Maps field_id → (flag_type, severity) for mismatch flags.
_MISMATCH_FLAG: dict[str, tuple[str, str]] = {
    "billed_amount":  ("AMOUNT_MISMATCH",    "HIGH"),
    "diagnosis":      ("DIAGNOSIS_MISMATCH", "HIGH"),
    "admission_date": ("DATE_MISMATCH",      "MEDIUM"),
    "discharge_date": ("DATE_MISMATCH",      "MEDIUM"),
    "hospital_name":  ("HOSPITAL_MISMATCH",  "MEDIUM"),
    "length_of_stay": ("LOS_MISMATCH",       "MEDIUM"),
}


def _mismatch_flag(field_id: str, t_fv: FactValue, d_fv: FactValue, note: str) -> Flag:
    flag_type, severity = _MISMATCH_FLAG.get(field_id, ("FIELD_MISMATCH", "MEDIUM"))
    return Flag(
        type=flag_type,
        severity=severity,
        message=(
            f"field '{field_id}': "
            f"transcript={t_fv.value!r}, document={d_fv.value!r} ({note})"
        ),
    )


def _sort_flags(flags: list[Flag]) -> list[Flag]:
    return sorted(flags, key=lambda f: _SEVERITY_ORDER[f.severity])


def _compute_risk(score: float) -> str:
    if score >= RISK_LOW_THRESHOLD:
        return "LOW"
    if score < RISK_HIGH_THRESHOLD:
        return "HIGH"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# Internal: dispatcher
# ---------------------------------------------------------------------------

def _dispatch(
    match_type: str,
    field_id: str,
    t_fv: FactValue,
    d_fv: FactValue,
    judge: DiagnosisJudge,
) -> tuple[str, str]:
    if match_type == "fuzzy_string":
        return _match_fuzzy_string(str(t_fv.value), str(d_fv.value))
    if match_type == "date":
        return _match_date(str(t_fv.value), str(d_fv.value))
    if match_type == "numeric":
        return _match_numeric(t_fv.value, d_fv.value, field_id)
    if match_type == "medical_semantic":
        return judge.compare(t_fv, d_fv)
    return "MISMATCH", f"unknown match_type '{match_type}'"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(
    claim_id: str,
    transcript: FactSet,
    document: FactSet,
    judge: DiagnosisJudge,
) -> VerificationResult:
    """
    Cross-verify two normalized FactSets and return a VerificationResult.

    Fields present in both sources are compared via typed matchers and scored
    binary (1.0 MATCH / 0.0 MISMATCH). Fields absent from either source are
    flagged as MISSING and excluded from the denominator. If no fields are
    included (all missing), risk_level is INSUFFICIENT_DATA.
    """
    schema = load_schema()

    field_verdicts: dict[str, FieldVerdict] = {}
    flags: list[Flag] = []
    missing_fields: list[str] = []
    low_quality_sources: list[str] = []

    included_weight_total = 0.0
    included_score_total = 0.0

    for field_def in schema["fields"]:
        field_id: str = field_def["id"]
        weight: float = field_def["weight"]
        match_type: str = field_def["match_type"]

        t_fv = transcript.facts.get(field_id)
        d_fv = document.facts.get(field_id)

        t_missing = t_fv is None or t_fv.value is None
        d_missing = d_fv is None or d_fv.value is None

        if t_missing or d_missing:
            missing_fields.append(field_id)
            missing_from = "transcript" if t_missing else "document"
            flags.append(Flag(
                type="MISSING_FIELD",
                severity="LOW",
                message=f"field '{field_id}' absent from {missing_from}",
            ))
            field_verdicts[field_id] = FieldVerdict(
                status="MISSING",
                score=0.0,
                transcript_value=t_fv.value if t_fv else None,
                document_value=d_fv.value if d_fv else None,
                transcript_quote=t_fv.source_quote if t_fv else None,
                document_quote=d_fv.source_quote if d_fv else None,
                note=f"absent from {missing_from}",
            )
            continue

        # Both values present — run typed matcher.
        verdict_str, note = _dispatch(match_type, field_id, t_fv, d_fv, judge)
        field_score = 1.0 if verdict_str == "MATCH" else 0.0
        included_weight_total += weight
        included_score_total += weight * field_score

        if verdict_str == "MISMATCH":
            flags.append(_mismatch_flag(field_id, t_fv, d_fv, note))

        # Flag low-confidence extractions (threshold mirrors RISK_HIGH_THRESHOLD).
        for fv, src_id in [(t_fv, transcript.source_id), (d_fv, document.source_id)]:
            if fv.confidence < RISK_HIGH_THRESHOLD:
                flags.append(Flag(
                    type="LOW_CONFIDENCE_EXTRACTION",
                    severity="LOW",
                    message=(
                        f"source '{src_id}' field '{field_id}' "
                        f"confidence={fv.confidence:.2f}"
                    ),
                ))
                if src_id not in low_quality_sources:
                    low_quality_sources.append(src_id)

        field_verdicts[field_id] = FieldVerdict(
            status=verdict_str,
            score=field_score,
            transcript_value=t_fv.value,
            document_value=d_fv.value,
            transcript_quote=t_fv.source_quote,
            document_quote=d_fv.source_quote,
            note=note,
        )

    verified_at = datetime.now(timezone.utc).isoformat()

    if included_weight_total == 0.0:
        return VerificationResult(
            claim_id=claim_id,
            verified_at=verified_at,
            consistency_score=None,
            risk_level="INSUFFICIENT_DATA",
            field_verdicts=field_verdicts,
            flags=_sort_flags(flags),
            missing_fields=missing_fields,
            low_quality_sources=low_quality_sources,
        )

    consistency_score = round(included_score_total / included_weight_total, 6)
    return VerificationResult(
        claim_id=claim_id,
        verified_at=verified_at,
        consistency_score=consistency_score,
        risk_level=_compute_risk(consistency_score),
        field_verdicts=field_verdicts,
        flags=_sort_flags(flags),
        missing_fields=missing_fields,
        low_quality_sources=low_quality_sources,
    )
