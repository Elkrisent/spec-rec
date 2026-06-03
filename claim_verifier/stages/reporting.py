"""
Stage 5 — Reporting.

Renders a VerificationResult as a Markdown document using a Jinja2 template.

Public API:
    render(result: VerificationResult) -> str
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from jinja2 import Environment, FileSystemLoader

from claim_verifier.config import load_schema
from claim_verifier.models import VerificationResult

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_RISK_BADGE: dict[str, str] = {
    "LOW": "LOW RISK",
    "MEDIUM": "MEDIUM RISK",
    "HIGH": "HIGH RISK",
    "INSUFFICIENT_DATA": "INSUFFICIENT DATA",
}

_REVIEWER_GUIDANCE: dict[str, str] = {
    "LOW": (
        "All verified fields are consistent between the transcript and the submitted document. "
        "Standard processing is appropriate. No further action required."
    ),
    "MEDIUM": (
        "One or more discrepancies were found between the claimant's statements and the document. "
        "Review flagged fields before approval. Minor variations may be acceptable; "
        "escalate any unexplained mismatch to a senior reviewer."
    ),
    "HIGH": (
        "Significant discrepancies detected. Do not approve this claim without further "
        "investigation. Contact the claimant and/or the provider to resolve the flagged "
        "mismatches. Consider referring to the Special Investigations Unit if discrepancies "
        "cannot be explained."
    ),
    "INSUFFICIENT_DATA": (
        "Verification could not be completed — no fields were present in both sources. "
        "Manual review of the original documents is required before any decision is made. "
        "Ensure the transcript and document are complete and correctly associated with "
        "this claim."
    ),
}


def _citation(value: Optional[Union[str, int, float]], quote: Optional[str]) -> str:
    """Format a cell value with its source citation.

    Returns:
        "—"                          if value is None
        "value"                      if value is set but quote is absent
        'value [source: "quote"]'    if both are present
    """
    if value is None:
        return "—"  # em dash
    val_str = str(value).replace("|", "\\|")
    if quote:
        safe_quote = quote.replace('"', '\\"').replace("|", "\\|").replace("\n", " ")
        return f'{val_str} [source: "{safe_quote}"]'
    return val_str


def _build_rows(result: VerificationResult, field_labels: dict[str, str]) -> list[dict]:
    rows = []
    for field_id, label in field_labels.items():
        fv = result.field_verdicts.get(field_id)
        if fv is None:
            continue
        rows.append({
            "label": label,
            "transcript": _citation(fv.transcript_value, fv.transcript_quote),
            "document": _citation(fv.document_value, fv.document_quote),
            "status": fv.status,
            "note": (fv.note or "").replace("|", "\\|"),
        })
    return rows


def render(result: VerificationResult) -> str:
    """Render a VerificationResult as a Markdown report string."""
    schema = load_schema()
    field_labels = {f["id"]: f["label"] for f in schema["fields"]}

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )

    score_pct = (
        f"{result.consistency_score:.1%}"
        if result.consistency_score is not None
        else "N/A"
    )

    template = env.get_template("report.md.j2")
    return template.render(
        result=result,
        rows=_build_rows(result, field_labels),
        risk_badge=_RISK_BADGE[result.risk_level],
        score_pct=score_pct,
        guidance=_REVIEWER_GUIDANCE[result.risk_level],
    )
