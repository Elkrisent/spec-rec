"""
Stage 2 — LLM extraction (T5.3, T5.4, T5.5).

Calls an LLMBackend to extract a FactSet from raw source text.
Validates against a Pydantic model; performs one repair retry on failure.
Checks that source_quote values are substrings of the text (T5.5).

Public API:
    extract(source_type, source_id, text, backend) -> FactSet
    EXTRACTION_SCHEMA  -- JSON schema for the Ollama format param
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

from claim_verifier.backends import LLMBackend
from claim_verifier.models import FactSet, FactValue

# ---------------------------------------------------------------------------
# Extraction JSON schema (T5.3)
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "hospital_name":  {"$ref": "#/$defs/field"},
        "admission_date": {"$ref": "#/$defs/field"},
        "discharge_date": {"$ref": "#/$defs/field"},
        "diagnosis":      {"$ref": "#/$defs/field"},
        "billed_amount":  {"$ref": "#/$defs/field"},
        "length_of_stay": {"$ref": "#/$defs/field"},
    },
    "required": [
        "hospital_name", "admission_date", "discharge_date",
        "diagnosis", "billed_amount", "length_of_stay",
    ],
    "$defs": {
        "field": {
            "type": "object",
            "properties": {
                "value":        {"type": ["string", "number", "null"]},
                "confidence":   {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "source_quote": {"type": ["string", "null"]},
                "entity_type":  {"type": ["string", "null"]},
            },
            "required": ["value", "confidence", "source_quote", "entity_type"],
        },
    },
}

_FIELD_IDS: list[str] = [
    "hospital_name", "admission_date", "discharge_date",
    "diagnosis", "billed_amount", "length_of_stay",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a medical claims extraction assistant. Extract facts from the source text.

For each field, return an object with exactly these keys:
  "value"        - the extracted value (string, number, or null if the field is not mentioned)
  "confidence"   - float 0.0-1.0 (0.9+ explicit statement; 0.6-0.8 clearly implied; <0.6 uncertain)
  "source_quote" - a verbatim substring copied from the source text (null when value is null)
  "entity_type"  - for the diagnosis field only: "DISEASE" or "PROCEDURE"; null for all other fields

Fields to extract:
  hospital_name   - name of the hospital or medical facility
  admission_date  - date of admission, preserved in the original text format
  discharge_date  - date of discharge, preserved in original format; null if not mentioned
  diagnosis       - primary medical diagnosis or condition
  billed_amount   - total billed amount as stated (preserve currency symbol and formatting)
  length_of_stay  - duration of hospital stay in whole days (integer; null if not explicitly stated)

Return only valid JSON. No explanation, no markdown fences.\
"""

# ---------------------------------------------------------------------------
# Internal Pydantic models
# ---------------------------------------------------------------------------


class ExtractedField(BaseModel):
    model_config = {"extra": "ignore"}

    value: Optional[Union[str, int, float]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_quote: Optional[str] = None
    entity_type: Optional[str] = None
    quote_verified: bool = True

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> float:
        if v is None:
            return 0.0
        return max(0.0, min(1.0, float(v)))


class ExtractionOutput(BaseModel):
    model_config = {"extra": "ignore"}

    hospital_name:  ExtractedField = Field(default_factory=ExtractedField)
    admission_date: ExtractedField = Field(default_factory=ExtractedField)
    discharge_date: ExtractedField = Field(default_factory=ExtractedField)
    diagnosis:      ExtractedField = Field(default_factory=ExtractedField)
    billed_amount:  ExtractedField = Field(default_factory=ExtractedField)
    length_of_stay: ExtractedField = Field(default_factory=ExtractedField)


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------


class ExtractionError(Exception):
    """Raised when LLM extraction fails after all retries."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_messages(source_type: str, source_id: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Source type: {source_type}\n"
                f"Source ID: {source_id}\n\n"
                f"Text:\n{text}"
            ),
        },
    ]


def _validate_output(raw: object) -> ExtractionOutput:
    if not isinstance(raw, dict):
        raise ValueError(f"LLM returned {type(raw).__name__}, expected dict")
    missing = [f for f in _FIELD_IDS if f not in raw]
    if missing:
        raise ValueError(f"LLM response missing required fields: {missing}")
    return ExtractionOutput.model_validate(raw)


def _verify_quotes(output: ExtractionOutput, text: str) -> ExtractionOutput:
    """
    Return a (possibly new) ExtractionOutput with quote_verified=False for
    any field whose source_quote is not a substring of the original text.
    Confidence is left unchanged (T5.5).
    """
    updates: dict[str, ExtractedField] = {}
    for field_id in _FIELD_IDS:
        ef: ExtractedField = getattr(output, field_id)
        if ef.source_quote is not None and ef.source_quote not in text:
            updates[field_id] = ef.model_copy(update={"quote_verified": False})
    return output if not updates else output.model_copy(update=updates)


def _to_factset(
    output: ExtractionOutput,
    source_type: str,
    source_id: str,
) -> FactSet:
    facts: dict[str, FactValue] = {}
    for field_id in _FIELD_IDS:
        ef: ExtractedField = getattr(output, field_id)
        facts[field_id] = FactValue(
            value=ef.value,
            confidence=ef.confidence,
            source_quote=ef.source_quote,
            entity_type=ef.entity_type,
            quote_verified=ef.quote_verified,
        )
    return FactSet(
        source_type=source_type,
        source_id=source_id,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        facts=facts,
    )


# ---------------------------------------------------------------------------
# Public API (T5.4)
# ---------------------------------------------------------------------------


def extract(
    source_type: str,
    source_id: str,
    text: str,
    backend: LLMBackend,
) -> FactSet:
    """
    Extract a FactSet from source text using the provided LLM backend.

    On Pydantic validation failure, performs one repair retry by appending
    the bad response and error to the conversation. Raises ExtractionError
    if the retry also fails.

    source_type : "transcript" or "document"
    source_id   : filename or call identifier
    text        : raw text to extract from
    backend     : LLMBackend implementation (OllamaBackend or StubBackend)
    """
    messages = _build_messages(source_type, source_id, text)
    raw = backend.complete(messages, schema=EXTRACTION_SCHEMA)

    try:
        output = _validate_output(raw)
    except (ValueError, ValidationError) as first_err:
        repair_messages = messages + [
            {"role": "assistant", "content": json.dumps(raw)},
            {
                "role": "user",
                "content": (
                    f"The previous response was invalid: {first_err}\n"
                    "Please return valid JSON that exactly matches the required schema."
                ),
            },
        ]
        raw2 = backend.complete(repair_messages, schema=EXTRACTION_SCHEMA)
        try:
            output = _validate_output(raw2)
        except (ValueError, ValidationError) as second_err:
            raise ExtractionError(
                f"Extraction failed after repair retry: {second_err}"
            ) from second_err

    output = _verify_quotes(output, text)
    return _to_factset(output, source_type, source_id)
