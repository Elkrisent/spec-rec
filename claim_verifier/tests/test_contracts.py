"""
T1.2–T1.5 / T1.8 — Contract tests.
Verifies: schema integrity, Pydantic model validation, LLMCache behaviour.
No LLM, no network, no file I/O beyond fixtures.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from claim_verifier.config import (
    FUZZY_HOSPITAL_THRESHOLD,
    LLM_CACHE_DIR,
    RISK_HIGH_THRESHOLD,
    RISK_LOW_THRESHOLD,
    load_schema,
)
from claim_verifier.llm_cache import LLMCache
from claim_verifier.models import (
    FactSet,
    FactValue,
    FieldVerdict,
    Flag,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# T1.2 — Schema integrity
# ---------------------------------------------------------------------------

def test_schema_loads():
    schema = load_schema()
    assert "fields" in schema
    assert "schema_version" in schema


def test_schema_has_exactly_six_fields():
    schema = load_schema()
    assert len(schema["fields"]) == 6


def test_schema_weights_sum_to_one():
    schema = load_schema()
    total = sum(f["weight"] for f in schema["fields"])
    assert total == pytest.approx(1.0, abs=1e-9)


def test_schema_has_required_field_ids():
    schema = load_schema()
    ids = {f["id"] for f in schema["fields"]}
    expected = {
        "hospital_name",
        "admission_date",
        "discharge_date",
        "diagnosis",
        "billed_amount",
        "length_of_stay",
    }
    assert ids == expected


def test_schema_every_field_has_weight_and_match_type():
    schema = load_schema()
    for field in schema["fields"]:
        assert "weight" in field, f"missing weight: {field['id']}"
        assert "match_type" in field, f"missing match_type: {field['id']}"
        assert field["weight"] > 0, f"zero weight: {field['id']}"


# ---------------------------------------------------------------------------
# T1.3 — Config thresholds
# ---------------------------------------------------------------------------

def test_risk_thresholds_are_ordered():
    assert 0.0 < RISK_HIGH_THRESHOLD < RISK_LOW_THRESHOLD < 1.0


def test_fuzzy_threshold_in_range():
    assert 0.0 < FUZZY_HOSPITAL_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# T1.4 — FactValue model
# ---------------------------------------------------------------------------

def test_factvalue_accepts_valid_confidence():
    fv = FactValue(value="Apollo Hospital", confidence=0.95, source_quote="Apollo Hospital")
    assert fv.confidence == 0.95


def test_factvalue_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        FactValue(value="x", confidence=1.1)


def test_factvalue_rejects_negative_confidence():
    with pytest.raises(ValidationError):
        FactValue(value="x", confidence=-0.1)


def test_factvalue_allows_null_value():
    fv = FactValue(value=None, confidence=0.0)
    assert fv.value is None
    assert not fv.quote_verified or fv.source_quote is None  # both null — fine


def test_factvalue_accepts_int_value():
    fv = FactValue(value=50000, confidence=0.75)
    assert fv.value == 50000


# ---------------------------------------------------------------------------
# T1.4 — FactSet model
# ---------------------------------------------------------------------------

def test_factset_parses_transcript(factset_transcript):
    fs = FactSet(**factset_transcript)
    assert fs.source_type == "transcript"
    assert "hospital_name" in fs.facts
    assert fs.facts["hospital_name"].confidence == pytest.approx(0.95)


def test_factset_parses_document(factset_document):
    fs = FactSet(**factset_document)
    assert fs.source_type == "document"
    assert fs.facts["billed_amount"].value == 62000
    assert fs.facts["billed_amount"].confidence == 1.0


def test_factset_rejects_invalid_source_type():
    with pytest.raises(ValidationError):
        FactSet(
            source_type="audio",
            source_id="test",
            extraction_timestamp="2025-01-01T00:00:00",
            facts={},
        )


def test_factset_transcript_has_diagnosis_entity_type(factset_transcript):
    fs = FactSet(**factset_transcript)
    assert fs.facts["diagnosis"].entity_type == "DISEASE"


def test_factset_missing_field_has_zero_confidence(factset_transcript):
    fs = FactSet(**factset_transcript)
    assert fs.facts["discharge_date"].value is None
    assert fs.facts["discharge_date"].confidence == 0.0


# ---------------------------------------------------------------------------
# T1.4 — VerificationResult model
# ---------------------------------------------------------------------------

def test_verification_result_parses(verification_result):
    vr = VerificationResult(**verification_result)
    assert vr.claim_id == "C001"
    assert vr.risk_level == "HIGH"
    assert vr.consistency_score == pytest.approx(0.6875, rel=1e-4)


def test_verification_result_flags_ordered_by_severity(verification_result):
    vr = VerificationResult(**verification_result)
    severity_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    ranks = [severity_rank[f.severity] for f in vr.flags]
    assert ranks == sorted(ranks), "flags are not sorted HIGH→MEDIUM→LOW"


def test_verification_result_insufficient_data_no_score():
    vr = VerificationResult(
        claim_id="C002",
        verified_at="2025-01-01T00:00:00",
        consistency_score=None,
        risk_level="INSUFFICIENT_DATA",
        field_verdicts={},
        flags=[],
        missing_fields=["hospital_name", "admission_date", "diagnosis", "billed_amount"],
        low_quality_sources=[],
    )
    assert vr.consistency_score is None
    assert vr.risk_level == "INSUFFICIENT_DATA"


def test_verification_result_rejects_score_with_insufficient_data():
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C003",
            verified_at="2025-01-01T00:00:00",
            consistency_score=0.5,  # must be None for INSUFFICIENT_DATA
            risk_level="INSUFFICIENT_DATA",
            field_verdicts={},
            flags=[],
            missing_fields=[],
            low_quality_sources=[],
        )


def test_verification_result_rejects_none_score_with_real_risk():
    with pytest.raises(ValidationError):
        VerificationResult(
            claim_id="C004",
            verified_at="2025-01-01T00:00:00",
            consistency_score=None,  # must have a value when risk_level is LOW/MEDIUM/HIGH
            risk_level="HIGH",
            field_verdicts={},
            flags=[],
            missing_fields=[],
            low_quality_sources=[],
        )


# ---------------------------------------------------------------------------
# T1.4 — Flag model
# ---------------------------------------------------------------------------

def test_flag_accepts_valid_severity():
    f = Flag(type="AMOUNT_MISMATCH", severity="HIGH", message="difference: 24%")
    assert f.severity == "HIGH"


def test_flag_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        Flag(type="X", severity="CRITICAL", message="x")


def test_flag_rejects_invalid_field_verdict_status():
    with pytest.raises(ValidationError):
        FieldVerdict(status="UNKNOWN", score=0.5)


# ---------------------------------------------------------------------------
# T1.8 — LLMCache
# ---------------------------------------------------------------------------

def test_llm_cache_miss_returns_none(tmp_path):
    cache = LLMCache(tmp_path)
    result = cache.get("qwen2.5:7b", [{"role": "user", "content": "hello"}])
    assert result is None


def test_llm_cache_hit_returns_identical_dict(tmp_path):
    cache = LLMCache(tmp_path)
    model = "qwen2.5:7b"
    messages = [{"role": "user", "content": "extract facts"}]
    response = {"hospital": "Apollo", "amount": 50000}

    cache.put(model, messages, response)
    retrieved = cache.get(model, messages)

    assert retrieved == response


def test_llm_cache_different_messages_do_not_collide(tmp_path):
    cache = LLMCache(tmp_path)
    model = "qwen2.5:7b"
    msgs_a = [{"role": "user", "content": "message A"}]
    msgs_b = [{"role": "user", "content": "message B"}]

    cache.put(model, msgs_a, {"result": "A"})
    cache.put(model, msgs_b, {"result": "B"})

    assert cache.get(model, msgs_a) == {"result": "A"}
    assert cache.get(model, msgs_b) == {"result": "B"}


def test_llm_cache_different_models_do_not_collide(tmp_path):
    cache = LLMCache(tmp_path)
    messages = [{"role": "user", "content": "same prompt"}]

    cache.put("model-A", messages, {"from": "A"})
    cache.put("model-B", messages, {"from": "B"})

    assert cache.get("model-A", messages) == {"from": "A"}
    assert cache.get("model-B", messages) == {"from": "B"}


def test_llm_cache_has_method(tmp_path):
    cache = LLMCache(tmp_path)
    model = "qwen2.5:7b"
    messages = [{"role": "user", "content": "test"}]

    assert not cache.has(model, messages)
    cache.put(model, messages, {"ok": True})
    assert cache.has(model, messages)


def test_llm_cache_kwargs_affect_key(tmp_path):
    cache = LLMCache(tmp_path)
    model = "qwen2.5:7b"
    messages = [{"role": "user", "content": "same"}]

    cache.put(model, messages, {"schema": "v1"}, schema="v1")
    cache.put(model, messages, {"schema": "v2"}, schema="v2")

    assert cache.get(model, messages, schema="v1") == {"schema": "v1"}
    assert cache.get(model, messages, schema="v2") == {"schema": "v2"}
