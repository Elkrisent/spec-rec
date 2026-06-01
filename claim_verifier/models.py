from __future__ import annotations

from typing import Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Fact extraction models
# ---------------------------------------------------------------------------

class FactValue(BaseModel):
    """One extracted field from a single source (transcript or document)."""

    value: Optional[Union[str, int, float]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_quote: Optional[str] = None
    # Entity type tagged by the extraction LLM (replaces scispacy): DISEASE, PROCEDURE, etc.
    entity_type: Optional[str] = None
    # Set to False when source_quote is not a substring of the source text (paraphrase detected).
    quote_verified: bool = True


class FactSet(BaseModel):
    """All extracted fields from one source, pre-normalization."""

    source_type: Literal["transcript", "document"]
    source_id: str  # filename or "call_recording"
    extraction_timestamp: str  # ISO 8601
    facts: dict[str, FactValue]


# ---------------------------------------------------------------------------
# Verification result models
# ---------------------------------------------------------------------------

class FieldVerdict(BaseModel):
    """Result of comparing one field across both sources."""

    status: Literal["MATCH", "MISMATCH", "MISSING"]
    # Binary per-field score: 1.0 on MATCH, 0.0 on MISMATCH or MISSING.
    score: float = Field(ge=0.0, le=1.0)
    transcript_value: Optional[Union[str, int, float]] = None
    document_value: Optional[Union[str, int, float]] = None
    transcript_quote: Optional[str] = None
    document_quote: Optional[str] = None
    note: Optional[str] = None  # delta info, ICD codes, judge rationale, etc.


class Flag(BaseModel):
    """A noteworthy finding surfaced to the human reviewer."""

    type: str  # e.g. AMOUNT_MISMATCH, DATE_MISMATCH, LOW_CONFIDENCE_EXTRACTION
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    message: str


class VerificationResult(BaseModel):
    """
    Full output of the cross-verification pipeline.

    Scoring:
        field_score(f) = 1.0 if MATCH else 0.0
        consistency_score = Σ weight(f)·field_score(f) / Σ weight(f for included fields)
        "included" = present in both sources; absent-from-one fields are flagged and excluded.

    INSUFFICIENT_DATA: when no fields are included (all missing from at least one source).
    In this case consistency_score is None and risk_level is INSUFFICIENT_DATA.
    """

    claim_id: str
    verified_at: str  # ISO 8601
    consistency_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "INSUFFICIENT_DATA"]
    field_verdicts: dict[str, FieldVerdict]
    flags: list[Flag]
    missing_fields: list[str]
    low_quality_sources: list[str]

    @model_validator(mode="after")
    def _check_score_and_risk_consistent(self) -> VerificationResult:
        if self.risk_level == "INSUFFICIENT_DATA":
            if self.consistency_score is not None:
                raise ValueError(
                    "consistency_score must be None when risk_level is INSUFFICIENT_DATA"
                )
        else:
            if self.consistency_score is None:
                raise ValueError(
                    "consistency_score is required when risk_level is not INSUFFICIENT_DATA"
                )
        return self
