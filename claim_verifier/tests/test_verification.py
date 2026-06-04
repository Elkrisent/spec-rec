"""
T3.1–T3.8 — Verification engine + scoring tests.

All tests are deterministic: no LLM calls, no file I/O.
The StubJudge stands in for the real diagnosis judge (same Protocol).

Score arithmetic (hand-verified) is annotated inline on each scenario.
Field weights from verification_schema.json:
    hospital_name  = 0.10   (fuzzy_string)
    admission_date = 0.20   (date)
    discharge_date = 0.10   (date)
    diagnosis      = 0.25   (medical_semantic)
    billed_amount  = 0.25   (numeric ±5%)
    length_of_stay = 0.10   (numeric ±1 abs)
    total          = 1.00
"""

from __future__ import annotations

import pytest

from claim_verifier.judge import DiagnosisJudge, StubJudge
from claim_verifier.models import FactSet, FactValue, VerificationResult
from claim_verifier.stages.verification import (
    _match_date,
    _match_fuzzy_string,
    _match_numeric,
    verify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fv(value, confidence: float = 0.95, source_quote: str | None = None, entity_type: str | None = None) -> FactValue:
    return FactValue(value=value, confidence=confidence, source_quote=source_quote, entity_type=entity_type)


def _fv_none() -> FactValue:
    return FactValue(value=None, confidence=0.0)


def _make_factset(source_type: str, source_id: str, **facts) -> FactSet:
    return FactSet(
        source_type=source_type,
        source_id=source_id,
        extraction_timestamp="2025-03-20T10:00:00",
        facts=facts,
    )


def _all_match_transcript() -> FactSet:
    return _make_factset(
        "transcript", "transcript_C001",
        hospital_name=_fv("apollo hospital"),
        admission_date=_fv("2025-03-12"),
        discharge_date=_fv("2025-03-16"),
        diagnosis=_fv("appendicitis", entity_type="DISEASE"),
        billed_amount=_fv(62000),
        length_of_stay=_fv(4),
    )


def _all_match_document() -> FactSet:
    return _make_factset(
        "document", "bill_C001.pdf",
        hospital_name=_fv("apollo hospital"),
        admission_date=_fv("2025-03-12"),
        discharge_date=_fv("2025-03-16"),
        diagnosis=_fv("appendicitis", entity_type="DISEASE"),
        billed_amount=_fv(62000),
        length_of_stay=_fv(4),
    )


# ---------------------------------------------------------------------------
# T3.1 — DiagnosisJudge Protocol
# ---------------------------------------------------------------------------

def test_stub_judge_implements_protocol():
    stub = StubJudge()
    assert isinstance(stub, DiagnosisJudge)


def test_stub_judge_default_mismatch():
    stub = StubJudge()
    verdict, note = stub.compare(_fv("appendicitis"), _fv("appendectomy"))
    assert verdict == "MISMATCH"
    assert note


def test_stub_judge_configurable_match():
    stub = StubJudge(verdict="MATCH", note="same disease")
    verdict, note = stub.compare(_fv("appendicitis"), _fv("acute appendicitis"))
    assert verdict == "MATCH"


# ---------------------------------------------------------------------------
# T3.2 — Fuzzy string matcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("apollo hospital",   "apollo hospital",   "MATCH"),
    ("apollo hospital",   "apollo hospitals",  "MATCH"),   # pluralisation noise
    ("city general hosp", "city general hospital", "MATCH"),
    ("apollo hospital",   "fortis hospital",   "MISMATCH"),
    ("star hospital",     "moon clinic",       "MISMATCH"),
    # multi-token subset: shorter ⊆ longer with extra non-generic tokens
    ("riverside shore memorial", "riverside shore memorial hospital", "MATCH"),  # dropped "hospital"
    ("riverside shore memorial hospital", "riverside shore memorial", "MATCH"),  # order-independent
    ("city hospital",    "city medical center",  "MISMATCH"),  # different substantive tokens
    # single-token subset: allowed only when extras are all generic facility words
    ("hosmat",  "hosmat hospital",          "MATCH"),   # brand name + generic suffix
    ("hosmat",  "hosmat clinic",            "MATCH"),   # brand name + generic suffix
    ("hosmat",  "hosmat hospitals bangalore","MISMATCH"),# extra "bangalore" is not generic
    ("memorial","memorial children's hospital","MISMATCH"),# "children's" is substantive
])
def test_fuzzy_string_matcher(a: str, b: str, expected: str):
    verdict, _ = _match_fuzzy_string(a, b)
    assert verdict == expected, f"_match_fuzzy_string({a!r}, {b!r}) → {verdict}, want {expected}"


# ---------------------------------------------------------------------------
# T3.3 — Date matcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    # Exact match
    ("2025-03-12", "2025-03-12", "MATCH"),
    # Within tolerance (±2 days)
    ("2025-03-12", "2025-03-13", "MATCH"),       # delta=1
    ("2025-03-13", "2025-03-12", "MATCH"),       # delta=1 reversed
    ("2025-03-12", "2025-03-14", "MATCH"),       # delta=2 — verbal testimony boundary
    ("2025-03-14", "2025-03-12", "MATCH"),       # delta=2 reversed
    # Beyond tolerance
    ("2025-03-12", "2025-03-15", "MISMATCH"),   # delta=3
    ("2025-03-12", "2025-03-17", "MISMATCH"),   # delta=5
    ("2025-03-12", "2025-03-20", "MISMATCH"),   # delta=8
    # Year-absent vs year-present: compare month/day only
    ("--03-12",    "2025-03-12", "MATCH"),
    ("--03-12",    "2025-03-13", "MATCH"),       # delta=1 by M/D
    ("--03-12",    "2025-03-14", "MATCH"),       # delta=2 by M/D
    ("--03-12",    "2025-03-17", "MISMATCH"),    # delta=5 by M/D
    # Both year-absent
    ("--03-12",    "--03-12",    "MATCH"),
    ("--03-12",    "--03-13",    "MATCH"),
    ("--03-12",    "--03-14",    "MATCH"),       # delta=2
    ("--03-12",    "--04-01",    "MISMATCH"),    # delta=20
])
def test_date_matcher(a: str, b: str, expected: str):
    verdict, _ = _match_date(a, b)
    assert verdict == expected, f"_match_date({a!r}, {b!r}) → {verdict}, want {expected}"


def test_date_matcher_note_mentions_year_absent():
    _, note = _match_date("--03-12", "2025-03-12")
    assert "year absent" in note


# ---------------------------------------------------------------------------
# T3.4 — Numeric matcher
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,field,expected", [
    # Amount ±5%
    (50000, 51000, "billed_amount", "MATCH"),    # delta=2.0%  < 5%
    (50000, 52000, "billed_amount", "MATCH"),    # delta=3.85% < 5%
    (50000, 62000, "billed_amount", "MISMATCH"), # delta=19.4% > 5%  (C001 scenario)
    (0,     0,     "billed_amount", "MATCH"),    # both zero
    # LOS ±1 absolute
    (4,     4,     "length_of_stay", "MATCH"),
    (4,     5,     "length_of_stay", "MATCH"),   # delta=1
    (4,     6,     "length_of_stay", "MISMATCH"),# delta=2
])
def test_numeric_matcher(a, b, field: str, expected: str):
    verdict, _ = _match_numeric(a, b, field)
    assert verdict == expected, f"_match_numeric({a}, {b}, {field!r}) → {verdict}"


# ---------------------------------------------------------------------------
# T3.5 — Medical semantic via stub (disease ≠ procedure guardrail)
# ---------------------------------------------------------------------------

def test_diagnosis_mismatch_disease_vs_procedure():
    stub = StubJudge(verdict="MISMATCH", note="disease vs procedure")
    t = _all_match_transcript()
    d = _all_match_document()
    # Override diagnosis to procedure
    d_facts = dict(d.facts)
    d_facts["diagnosis"] = _fv("appendectomy", entity_type="PROCEDURE")
    d = d.model_copy(update={"facts": d_facts})
    result = verify("C001", t, d, stub)
    fv = result.field_verdicts["diagnosis"]
    assert fv.status == "MISMATCH"
    assert any(f.type == "DIAGNOSIS_MISMATCH" for f in result.flags)


# ---------------------------------------------------------------------------
# T3.6 — Scoring: golden scenarios with hand-checked arithmetic
# ---------------------------------------------------------------------------

def test_scenario_all_match_score_1():
    """
    All 6 fields present and MATCH.
    score = (0.10+0.20+0.10+0.25+0.25+0.10) / 1.00 = 1.0
    risk  = LOW (≥0.90)
    """
    result = verify("C001", _all_match_transcript(), _all_match_document(), StubJudge(verdict="MATCH"))
    assert result.consistency_score == pytest.approx(1.0)
    assert result.risk_level == "LOW"
    assert all(v.status == "MATCH" for v in result.field_verdicts.values())


def test_scenario_amount_mismatch_50k_vs_62k():
    """
    billed_amount: 50000 vs 62000 → MISMATCH (delta≈19.4% > 5%)
    All other 5 fields MATCH.
    included weights = 1.00
    score = (0.10+0.20+0.10+0.25+0.00+0.10) / 1.00 = 0.75
    risk  = MEDIUM (0.70 ≤ 0.75 < 0.90)
    """
    t = _all_match_transcript()
    d = _all_match_document()
    d_facts = dict(d.facts)
    d_facts["billed_amount"] = _fv(50000)      # transcript also 62000 originally; make them differ
    # Reset transcript amount to 62000 to keep difference clear
    t_facts = dict(t.facts)
    t_facts["billed_amount"] = _fv(50000)
    d_facts["billed_amount"] = _fv(62000)
    t = t.model_copy(update={"facts": t_facts})
    d = d.model_copy(update={"facts": d_facts})

    result = verify("C001", t, d, StubJudge(verdict="MATCH"))
    assert result.consistency_score == pytest.approx(0.75)
    assert result.risk_level == "MEDIUM"
    assert result.field_verdicts["billed_amount"].status == "MISMATCH"
    assert any(f.type == "AMOUNT_MISMATCH" and f.severity == "HIGH" for f in result.flags)


def test_scenario_date_off_by_1_match():
    """
    admission_date: 2025-03-12 vs 2025-03-13 → delta=1 → MATCH (within ±1 day).
    score = 1.0  →  risk = LOW
    """
    t = _all_match_transcript()
    d = _all_match_document()
    d_facts = dict(d.facts)
    d_facts["admission_date"] = _fv("2025-03-13")
    d = d.model_copy(update={"facts": d_facts})

    result = verify("C001", t, d, StubJudge(verdict="MATCH"))
    assert result.field_verdicts["admission_date"].status == "MATCH"
    assert result.consistency_score == pytest.approx(1.0)
    assert result.risk_level == "LOW"


def test_scenario_date_off_by_5_mismatch():
    """
    admission_date: 2025-03-12 vs 2025-03-17 → delta=5 → MISMATCH.
    All other fields MATCH.
    score = (0.10 + 0.00×0.20 + 0.10 + 0.25 + 0.25 + 0.10) / 1.00 = 0.80
    risk  = MEDIUM
    """
    t = _all_match_transcript()
    d = _all_match_document()
    d_facts = dict(d.facts)
    d_facts["admission_date"] = _fv("2025-03-17")
    d = d.model_copy(update={"facts": d_facts})

    result = verify("C001", t, d, StubJudge(verdict="MATCH"))
    assert result.consistency_score == pytest.approx(0.80)
    assert result.risk_level == "MEDIUM"
    assert result.field_verdicts["admission_date"].status == "MISMATCH"
    assert any(f.type == "DATE_MISMATCH" for f in result.flags)


def test_scenario_missing_field_excluded_from_denominator():
    """
    billed_amount absent from document → MISSING → excluded from scoring.
    Included fields: hospital(0.10) + admission(0.20) + discharge(0.10) + diagnosis(0.25) + LOS(0.10) = 0.75
    All included MATCH.
    score = 0.75 / 0.75 = 1.0  →  risk = LOW
    """
    t = _all_match_transcript()
    d = _all_match_document()
    d_facts = dict(d.facts)
    d_facts["billed_amount"] = _fv_none()
    d = d.model_copy(update={"facts": d_facts})

    result = verify("C001", t, d, StubJudge(verdict="MATCH"))
    assert result.consistency_score == pytest.approx(1.0)
    assert result.risk_level == "LOW"
    assert "billed_amount" in result.missing_fields
    assert result.field_verdicts["billed_amount"].status == "MISSING"
    assert any(f.type == "MISSING_FIELD" for f in result.flags)


def test_scenario_all_fields_missing_insufficient_data():
    """
    All fields absent from transcript → no included fields → INSUFFICIENT_DATA.
    consistency_score must be None.
    """
    t = _make_factset(
        "transcript", "transcript_C001",
        hospital_name=_fv_none(),
        admission_date=_fv_none(),
        discharge_date=_fv_none(),
        diagnosis=_fv_none(),
        billed_amount=_fv_none(),
        length_of_stay=_fv_none(),
    )
    d = _all_match_document()

    result = verify("C001", t, d, StubJudge())
    assert result.consistency_score is None
    assert result.risk_level == "INSUFFICIENT_DATA"
    assert len(result.missing_fields) == 6


def test_scenario_diagnosis_mismatch():
    """
    diagnosis MISMATCH via stub; all other 5 fields MATCH.
    score = (0.10+0.20+0.10+0.00×0.25+0.25+0.10) / 1.00 = 0.75
    risk  = MEDIUM
    """
    result = verify("C001", _all_match_transcript(), _all_match_document(), StubJudge(verdict="MISMATCH"))
    assert result.consistency_score == pytest.approx(0.75)
    assert result.risk_level == "MEDIUM"
    assert result.field_verdicts["diagnosis"].status == "MISMATCH"
    assert any(f.type == "DIAGNOSIS_MISMATCH" and f.severity == "HIGH" for f in result.flags)


def test_scenario_year_absent_matches_year_present():
    """
    Transcript admission_date year-absent (--03-12) matches document 2025-03-12.
    Month/day comparison → delta=0 → MATCH.
    score = 1.0
    """
    t = _all_match_transcript()
    t_facts = dict(t.facts)
    t_facts["admission_date"] = _fv("--03-12")
    t = t.model_copy(update={"facts": t_facts})

    result = verify("C001", t, _all_match_document(), StubJudge(verdict="MATCH"))
    assert result.field_verdicts["admission_date"].status == "MATCH"
    assert result.consistency_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T3.7 — Flag ordering (HIGH → MEDIUM → LOW)
# ---------------------------------------------------------------------------

def test_flags_ordered_high_before_medium_before_low():
    """
    Force both an AMOUNT_MISMATCH (HIGH) and a DATE_MISMATCH (MEDIUM).
    Resulting flags must be sorted HIGH → MEDIUM → LOW.
    """
    t = _all_match_transcript()
    d = _all_match_document()
    t_facts = dict(t.facts)
    d_facts = dict(d.facts)
    t_facts["billed_amount"] = _fv(50000)
    d_facts["billed_amount"] = _fv(62000)           # AMOUNT_MISMATCH (HIGH)
    d_facts["admission_date"] = _fv("2025-03-17")   # DATE_MISMATCH (MEDIUM)
    t = t.model_copy(update={"facts": t_facts})
    d = d.model_copy(update={"facts": d_facts})

    result = verify("C001", t, d, StubJudge(verdict="MATCH"))
    severities = [f.severity for f in result.flags if f.severity != "LOW"]
    # HIGH must appear before MEDIUM
    high_idx = next(i for i, s in enumerate(severities) if s == "HIGH")
    med_idx = next(i for i, s in enumerate(severities) if s == "MEDIUM")
    assert high_idx < med_idx


# ---------------------------------------------------------------------------
# T3.8 — VerificationResult model compliance
# ---------------------------------------------------------------------------

def test_result_fields_present():
    result = verify("C001", _all_match_transcript(), _all_match_document(), StubJudge(verdict="MATCH"))
    assert result.claim_id == "C001"
    assert result.verified_at
    assert isinstance(result.field_verdicts, dict)
    assert isinstance(result.flags, list)
    assert isinstance(result.missing_fields, list)
    assert isinstance(result.low_quality_sources, list)


def test_result_validates_against_pydantic_model():
    result = verify("C001", _all_match_transcript(), _all_match_document(), StubJudge(verdict="MATCH"))
    # Round-trip through model to confirm schema compliance.
    VerificationResult.model_validate(result.model_dump())


def test_c001_normalized_fixture_amount_mismatch(factset_transcript, factset_document):
    """
    Integration: use W2-normalized C001 fixtures.
    After normalization:
      transcript: amount=50000 (year-absent admission), hospital="apollo hospital"
      document:   amount=62000 (full ISO dates),        hospital="apollo hospitals"

    Fields included:
      hospital_name(0.10)   MATCH  (fuzzy "apollo hospital" ≈ "apollo hospitals")
      admission_date(0.20)  MATCH  (--03-12 vs 2025-03-12, same M/D)
      discharge_date: absent from transcript → MISSING → excluded
      diagnosis(0.25)       MATCH  (stub)
      billed_amount(0.25)   MISMATCH (50k vs 62k, ≈19.4%)
      length_of_stay: absent from transcript → MISSING → excluded

    Included weight = 0.10+0.20+0.25+0.25 = 0.80
    score = (0.10+0.20+0.25+0.00) / 0.80 = 0.55/0.80 = 0.6875
    risk  = HIGH (<0.70)
    """
    from claim_verifier.stages.normalization import normalize

    t_raw = FactSet(**factset_transcript)
    d_raw = FactSet(**factset_document)
    t_norm, _ = normalize(t_raw)
    d_norm, _ = normalize(d_raw)

    result = verify("C001", t_norm, d_norm, StubJudge(verdict="MATCH"))

    assert result.consistency_score == pytest.approx(0.6875, rel=1e-4)
    assert result.risk_level == "HIGH"
    assert result.field_verdicts["billed_amount"].status == "MISMATCH"
    assert any(f.type == "AMOUNT_MISMATCH" for f in result.flags)
    assert "discharge_date" in result.missing_fields
    assert "length_of_stay" in result.missing_fields
