"""
T2.1–T2.6 — Normalization tests.

All tests are deterministic (no LLM, no I/O beyond in-memory FactSets).
Table-driven cases are collected below the unit helpers.
"""

from __future__ import annotations

import pytest

from claim_verifier.models import FactSet, FactValue
from claim_verifier.stages.normalization import (
    YEAR_ABSENT_PREFIX,
    _normalize_amount,
    _normalize_date,
    _normalize_hospital,
    _normalize_los,
    normalize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factset(**field_overrides) -> FactSet:
    """Build a minimal transcript FactSet for a single field under test."""
    defaults = {
        "hospital_name": FactValue(value=None, confidence=0.0),
        "admission_date": FactValue(value=None, confidence=0.0),
        "discharge_date": FactValue(value=None, confidence=0.0),
        "diagnosis": FactValue(value=None, confidence=0.0),
        "billed_amount": FactValue(value=None, confidence=0.0),
        "length_of_stay": FactValue(value=None, confidence=0.0),
    }
    defaults.update(field_overrides)
    return FactSet(
        source_type="transcript",
        source_id="test_source",
        extraction_timestamp="2025-01-01T00:00:00",
        facts=defaults,
    )


# ---------------------------------------------------------------------------
# T2.1 — Date normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    # Year-absent: month + day only → "--MM-DD" marker
    ("March 12", "--03-12"),
    ("Jan 5", "--01-05"),
    ("15 February", "--02-15"),
    ("1 Dec", "--12-01"),
    # Year-present: full ISO date (DMY order respected)
    ("12/03/2025", "2025-03-12"),
    ("16/03/2025", "2025-03-16"),
    ("March 12, 2025", "2025-03-12"),
    ("1st January 2024", "2024-01-01"),
    ("2025-03-12", "2025-03-12"),      # already ISO
    ("01-01-2023", "2023-01-01"),      # DMY
])
def test_normalize_date_valid(raw: str, expected: str):
    result, flag = _normalize_date(raw, "admission_date", "src")
    assert flag is None, f"unexpected flag for '{raw}': {flag}"
    assert result == expected, f"'{raw}' → '{result}', want '{expected}'"


def test_normalize_date_year_absent_uses_marker():
    result, _ = _normalize_date("March 12", "admission_date", "src")
    assert result is not None
    assert result.startswith(YEAR_ABSENT_PREFIX)
    assert len(result) == 7  # "--MM-DD"


def test_normalize_date_unparseable_emits_flag():
    result, flag = _normalize_date("not-a-date-xyz", "admission_date", "src")
    assert result is None
    assert flag is not None
    assert flag.type == "UNPARSEABLE_DATE"
    assert flag.severity == "MEDIUM"


def test_normalize_date_unparseable_flag_contains_field_and_source():
    _, flag = _normalize_date("garbage", "admission_date", "bill_C001.pdf")
    assert flag is not None
    assert "admission_date" in flag.message
    assert "bill_C001.pdf" in flag.message


# ---------------------------------------------------------------------------
# T2.2 — Amount normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    # Indian grouping (lakh system)
    ("1,20,000", 120000),
    ("50,000", 50000),
    ("1,50,000", 150000),
    # Currency prefix variants
    ("₹50,000", 50000),
    ("Rs.62,000", 62000),
    ("Rs. 62,000", 62000),
    ("Rs62,000", 62000),
    ("INR 75000", 75000),
    ("INR75000", 75000),
    ("$50,000", 50000),
    # Plain integers / floats
    (50000, 50000),
    (62000.0, 62000),
    # Decimal paise (truncated to int)
    ("1,20,000.50", 120000),
    ("50000.00", 50000),
])
def test_normalize_amount_valid(raw, expected: int):
    result, flag = _normalize_amount(raw, "billed_amount", "src")
    assert flag is None, f"unexpected flag for '{raw}': {flag}"
    assert result == expected


def test_normalize_amount_unparseable_emits_flag():
    result, flag = _normalize_amount("fifty thousand", "billed_amount", "src")
    assert result is None
    assert flag is not None
    assert flag.type == "UNPARSEABLE_AMOUNT"
    assert flag.severity == "MEDIUM"


def test_normalize_amount_empty_string_emits_flag():
    result, flag = _normalize_amount("", "billed_amount", "src")
    assert result is None
    assert flag is not None


# ---------------------------------------------------------------------------
# T2.3 — Hospital canonicalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Apollo Hosp.", "apollo hospital"),
    ("Apollo Hospitals", "apollo hospitals"),
    ("APOLLO HOSPITAL", "apollo hospital"),
    ("City Gen. Hosp.", "city general hospital"),
    ("St. Mary's Medical Ctr.", "saint mary's medical center"),
    ("Dr. Reddy's Pvt. Hosp.", "doctor reddy's private hospital"),
    ("  Metro  Hosp  ", "metro hospital"),
])
def test_normalize_hospital(raw: str, expected: str):
    assert _normalize_hospital(raw) == expected


# ---------------------------------------------------------------------------
# T2.4 — Diagnosis passthrough
# ---------------------------------------------------------------------------

def test_diagnosis_value_unchanged():
    fs = _make_factset(
        diagnosis=FactValue(
            value="Acute Appendicitis",
            confidence=0.9,
            entity_type="DISEASE",
        )
    )
    normalized, flags = normalize(fs)
    assert normalized.facts["diagnosis"].value == "Acute Appendicitis"
    assert normalized.facts["diagnosis"].entity_type == "DISEASE"
    assert not flags


def test_diagnosis_entity_type_carried_through():
    fs = _make_factset(
        diagnosis=FactValue(
            value="appendectomy",
            confidence=0.85,
            entity_type="PROCEDURE",
        )
    )
    normalized, _ = normalize(fs)
    assert normalized.facts["diagnosis"].entity_type == "PROCEDURE"


# ---------------------------------------------------------------------------
# T2.5 — Unparseable-value handling via normalize()
# ---------------------------------------------------------------------------

def test_bad_date_sets_confidence_zero_and_emits_flag():
    fs = _make_factset(
        admission_date=FactValue(value="notadate", confidence=0.8)
    )
    normalized, flags = normalize(fs)
    fv = normalized.facts["admission_date"]
    assert fv.value is None
    assert fv.confidence == 0.0
    assert any(f.type == "UNPARSEABLE_DATE" for f in flags)


def test_bad_amount_sets_confidence_zero_and_emits_flag():
    fs = _make_factset(
        billed_amount=FactValue(value="fifty thousand rupees", confidence=0.7)
    )
    normalized, flags = normalize(fs)
    fv = normalized.facts["billed_amount"]
    assert fv.value is None
    assert fv.confidence == 0.0
    assert any(f.type == "UNPARSEABLE_AMOUNT" for f in flags)


def test_null_field_passes_through_unchanged():
    fs = _make_factset(
        discharge_date=FactValue(value=None, confidence=0.0)
    )
    normalized, flags = normalize(fs)
    assert normalized.facts["discharge_date"].value is None
    assert normalized.facts["discharge_date"].confidence == 0.0
    assert not flags


# ---------------------------------------------------------------------------
# T2.6 — Full normalize() integration — C001 fixtures
# ---------------------------------------------------------------------------

def test_normalize_transcript_fixture(factset_transcript):
    fs = FactSet(**factset_transcript)
    normalized, flags = normalize(fs)

    # Hospital canonicalized
    assert normalized.facts["hospital_name"].value == "apollo hospital"

    # Admission date year-absent: "March 12" → "--03-12"
    assert normalized.facts["admission_date"].value == "--03-12"

    # Diagnosis unchanged (passthrough)
    assert normalized.facts["diagnosis"].value == "appendicitis"
    assert normalized.facts["diagnosis"].entity_type == "DISEASE"

    # Amount already int — preserved
    assert normalized.facts["billed_amount"].value == 50000

    # No parse errors expected
    assert not flags


def test_normalize_document_fixture(factset_document):
    fs = FactSet(**factset_document)
    normalized, flags = normalize(fs)

    # Hospital canonicalized (already plural "hospitals")
    assert normalized.facts["hospital_name"].value == "apollo hospitals"

    # Admission date with year: "12/03/2025" DMY → "2025-03-12"
    assert normalized.facts["admission_date"].value == "2025-03-12"

    # Discharge date: "16/03/2025" → "2025-03-16"
    assert normalized.facts["discharge_date"].value == "2025-03-16"

    # Amount already int
    assert normalized.facts["billed_amount"].value == 62000

    # LOS already int
    assert normalized.facts["length_of_stay"].value == 4

    assert not flags


def test_normalize_preserves_source_quote():
    fs = _make_factset(
        billed_amount=FactValue(
            value="₹50,000",
            confidence=0.9,
            source_quote="around fifty thousand rupees",
        )
    )
    normalized, _ = normalize(fs)
    assert normalized.facts["billed_amount"].source_quote == "around fifty thousand rupees"


def test_normalize_preserves_metadata():
    fs = _make_factset()
    normalized, _ = normalize(fs)
    assert normalized.source_type == fs.source_type
    assert normalized.source_id == fs.source_id
    assert normalized.extraction_timestamp == fs.extraction_timestamp


def test_normalize_los_string_days():
    result, flag = _normalize_los("4 days", "length_of_stay", "src")
    assert flag is None
    assert result == 4


def test_normalize_los_string_integer():
    result, flag = _normalize_los("7", "length_of_stay", "src")
    assert flag is None
    assert result == 7


def test_normalize_los_unparseable_emits_low_flag():
    result, flag = _normalize_los("one week", "length_of_stay", "src")
    assert result is None
    assert flag is not None
    assert flag.type == "UNPARSEABLE_LOS"
    assert flag.severity == "LOW"


def test_multiple_parse_errors_emit_multiple_flags():
    fs = _make_factset(
        admission_date=FactValue(value="baddate", confidence=0.5),
        billed_amount=FactValue(value="not-a-number", confidence=0.5),
    )
    _, flags = normalize(fs)
    types = {f.type for f in flags}
    assert "UNPARSEABLE_DATE" in types
    assert "UNPARSEABLE_AMOUNT" in types
    assert len(flags) == 2
