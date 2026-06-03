"""
T5.1–T5.7 — Extraction tests.

T5.1 : LLMBackend Protocol + StubBackend type-checks
T5.2 : OllamaBackend cache hit path (no live Ollama needed)
T5.3 : EXTRACTION_SCHEMA structure
T5.4 : extract() validation + one repair retry
T5.5 : quote substring-validation → quote_verified=False, confidence unchanged
T5.6 : LLMJudge verdict + rationale; satisfies DiagnosisJudge Protocol
T5.7 : ≥5 texts → valid FactSets, fully offline via pre-seeded LLMCache
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from claim_verifier.backends import LLMBackend, StubBackend
from claim_verifier.backends.ollama import OllamaBackend, _parse_content
from claim_verifier.judge import DiagnosisJudge, LLMJudge, StubJudge
from claim_verifier.llm_cache import LLMCache
from claim_verifier.models import FactSet, FactValue
from claim_verifier.stages.extraction import (
    EXTRACTION_SCHEMA,
    ExtractionError,
    ExtractionOutput,
    _build_messages,
    _verify_quotes,
    extract,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ALL_FIELDS = [
    "hospital_name", "admission_date", "discharge_date",
    "diagnosis", "billed_amount", "length_of_stay",
]


def _good_raw() -> dict:
    """A minimal valid LLM response dict (all fields present)."""
    return {
        "hospital_name":  {"value": "Apollo Hospital",    "confidence": 0.95, "source_quote": "Apollo Hospital",    "entity_type": None},
        "admission_date": {"value": "March 12, 2025",     "confidence": 0.95, "source_quote": "March 12, 2025",     "entity_type": None},
        "discharge_date": {"value": None,                 "confidence": 0.0,  "source_quote": None,                "entity_type": None},
        "diagnosis":      {"value": "appendicitis",       "confidence": 0.95, "source_quote": "appendicitis",       "entity_type": "DISEASE"},
        "billed_amount":  {"value": "Rs. 50,000",         "confidence": 0.90, "source_quote": "Rs. 50,000",         "entity_type": None},
        "length_of_stay": {"value": 4,                    "confidence": 0.95, "source_quote": "4 days",             "entity_type": None},
    }


def _ollama_response(content_dict: dict) -> dict:
    """Wrap a content dict in Ollama's /api/chat response envelope."""
    return {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "message": {"role": "assistant", "content": json.dumps(content_dict)},
        "done": True,
    }


_SAMPLE_TEXT = (
    "I was admitted to Apollo Hospital on March 12, 2025. "
    "I was diagnosed with appendicitis and stayed for 4 days. "
    "Total bill was Rs. 50,000."
)


# Stateful stub: different responses per call (for retry tests).
class _SeqStub:
    def __init__(self, *responses: dict) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0

    def complete(self, messages: list[dict], schema: dict | None = None) -> dict:
        r = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        self.call_count += 1
        return r


# ---------------------------------------------------------------------------
# T5.1 — LLMBackend Protocol + StubBackend
# ---------------------------------------------------------------------------

class TestLLMBackendProtocol:
    def test_stub_satisfies_protocol(self):
        stub = StubBackend(response={})
        assert isinstance(stub, LLMBackend)

    def test_ollama_satisfies_protocol(self, tmp_path):
        backend = OllamaBackend(cache=LLMCache(tmp_path))
        assert isinstance(backend, LLMBackend)

    def test_stub_returns_configured_response(self):
        expected = {"key": "value"}
        stub = StubBackend(response=expected)
        assert stub.complete([]) == expected

    def test_stub_tracks_call_count(self):
        stub = StubBackend(response={})
        stub.complete([{"role": "user", "content": "hi"}])
        stub.complete([{"role": "user", "content": "bye"}])
        assert stub.call_count == 2

    def test_stub_records_messages(self):
        msgs = [{"role": "user", "content": "test"}]
        stub = StubBackend(response={})
        stub.complete(msgs, schema={"type": "object"})
        assert stub.calls[0][0] == msgs

    def test_stub_records_schema(self):
        schema = {"type": "object"}
        stub = StubBackend(response={})
        stub.complete([], schema=schema)
        assert stub.calls[0][1] == schema


# ---------------------------------------------------------------------------
# T5.2 — OllamaBackend cache hit (no live Ollama required)
# ---------------------------------------------------------------------------

class TestOllamaBackendCache:
    def _seeded_backend(self, tmp_path: Path, messages: list[dict], content: dict) -> OllamaBackend:
        cache = LLMCache(tmp_path)
        cache.put(
            "qwen2.5:7b-instruct-q4_K_M",
            messages,
            _ollama_response(content),
            format=EXTRACTION_SCHEMA,
        )
        return OllamaBackend(
            model="qwen2.5:7b-instruct-q4_K_M",
            cache=cache,
        )

    def test_cache_hit_returns_parsed_content(self, tmp_path):
        msgs = _build_messages("transcript", "t1", _SAMPLE_TEXT)
        backend = self._seeded_backend(tmp_path, msgs, _good_raw())
        result = backend.complete(msgs, schema=EXTRACTION_SCHEMA)
        assert result == _good_raw()

    def test_cache_hit_skips_ollama(self, tmp_path, monkeypatch):
        msgs = _build_messages("transcript", "t1", _SAMPLE_TEXT)
        backend = self._seeded_backend(tmp_path, msgs, _good_raw())

        def _no_network(*args, **kwargs):
            raise AssertionError("Should not call Ollama on cache hit")

        monkeypatch.setattr(backend, "_call_ollama", _no_network)
        result = backend.complete(msgs, schema=EXTRACTION_SCHEMA)
        assert "hospital_name" in result

    def test_parse_content_extracts_inner_dict(self):
        content = {"hospital_name": {"value": "X"}}
        response = _ollama_response(content)
        assert _parse_content(response) == content

    def test_different_schema_is_separate_cache_entry(self, tmp_path):
        msgs = [{"role": "user", "content": "hi"}]
        cache = LLMCache(tmp_path)
        content_a = {"a": 1}
        content_b = {"b": 2}
        cache.put("model", msgs, _ollama_response(content_a), format={"type": "object"})
        cache.put("model", msgs, _ollama_response(content_b), format={"type": "string"})

        backend = OllamaBackend(model="model", cache=cache)
        r_a = backend.complete(msgs, schema={"type": "object"})
        r_b = backend.complete(msgs, schema={"type": "string"})
        assert r_a != r_b


# ---------------------------------------------------------------------------
# T5.3 — EXTRACTION_SCHEMA structure
# ---------------------------------------------------------------------------

class TestExtractionSchema:
    def test_schema_is_dict(self):
        assert isinstance(EXTRACTION_SCHEMA, dict)

    def test_schema_type_is_object(self):
        assert EXTRACTION_SCHEMA["type"] == "object"

    def test_schema_has_all_six_fields(self):
        for field_id in _ALL_FIELDS:
            assert field_id in EXTRACTION_SCHEMA["properties"], (
                f"EXTRACTION_SCHEMA missing field '{field_id}'"
            )

    def test_schema_requires_all_fields(self):
        required = set(EXTRACTION_SCHEMA.get("required", []))
        assert required == set(_ALL_FIELDS)

    def test_field_def_has_value_confidence_quote_entitytype(self):
        field_def = EXTRACTION_SCHEMA["$defs"]["field"]
        for key in ("value", "confidence", "source_quote", "entity_type"):
            assert key in field_def["properties"]


# ---------------------------------------------------------------------------
# T5.4 — extract(): validation + repair retry
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_factset(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert isinstance(result, FactSet)

    def test_source_type_preserved(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.source_type == "transcript"

    def test_source_id_preserved(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "MY_SOURCE", _SAMPLE_TEXT, stub)
        assert result.source_id == "MY_SOURCE"

    def test_all_six_fields_in_factset(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        for field_id in _ALL_FIELDS:
            assert field_id in result.facts, f"Missing field '{field_id}' in FactSet"

    def test_extraction_timestamp_is_iso(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert "T" in result.extraction_timestamp  # ISO 8601

    def test_integer_los_value_preserved(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["length_of_stay"].value == 4

    def test_null_value_is_none(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["discharge_date"].value is None

    def test_confidence_preserved(self):
        stub = StubBackend(response=_good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["hospital_name"].confidence == pytest.approx(0.95)

    def test_null_confidence_coerced_to_zero(self):
        raw = _good_raw()
        raw["discharge_date"]["confidence"] = None
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["discharge_date"].confidence == 0.0

    def test_repair_retry_on_invalid_first_response(self):
        """First response is invalid; second (retry) is valid."""
        invalid = {"garbage": "data"}
        seq = _SeqStub(invalid, _good_raw())
        result = extract("transcript", "t1", _SAMPLE_TEXT, seq)
        assert isinstance(result, FactSet)
        assert seq.call_count == 2

    def test_retry_messages_include_error_info(self):
        """Verify the repair retry appends the bad response and error to messages."""
        invalid = {"wrong": True}
        captured_messages: list[list[dict]] = []

        class _CapturingSeqStub:
            def __init__(self) -> None:
                self._call = 0

            def complete(self, messages, schema=None):
                captured_messages.append(list(messages))
                self._call += 1
                return invalid if self._call == 1 else _good_raw()

        extract("transcript", "t1", _SAMPLE_TEXT, _CapturingSeqStub())
        # Second call should have more messages (original + bad response + repair request)
        assert len(captured_messages[1]) > len(captured_messages[0])

    def test_raises_extraction_error_after_two_failures(self):
        """Both attempts return invalid JSON → ExtractionError raised."""
        seq = _SeqStub({"bad": 1}, {"still_bad": 2})
        with pytest.raises(ExtractionError):
            extract("transcript", "t1", _SAMPLE_TEXT, seq)

    def test_document_source_type_accepted(self):
        stub = StubBackend(response=_good_raw())
        result = extract("document", "bill.pdf", "Total: Rs.50,000", stub)
        assert result.source_type == "document"

    def test_extra_keys_in_response_ignored(self):
        """LLM may return extra keys; ExtractionOutput ignores them."""
        raw = _good_raw()
        raw["unexpected_key"] = "ignored"
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert isinstance(result, FactSet)


# ---------------------------------------------------------------------------
# T5.5 — Quote substring-validation
# ---------------------------------------------------------------------------

class TestQuoteVerification:
    def test_valid_quote_is_verified(self):
        raw = _good_raw()
        raw["hospital_name"]["source_quote"] = "Apollo Hospital"  # substring of _SAMPLE_TEXT
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["hospital_name"].quote_verified is True

    def test_invalid_quote_marked_unverified(self):
        raw = _good_raw()
        raw["hospital_name"]["source_quote"] = "NOT IN TEXT"
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["hospital_name"].quote_verified is False

    def test_unverified_quote_does_not_zero_confidence(self):
        """T5.5 spec: quote_verified=False, confidence unchanged."""
        raw = _good_raw()
        raw["hospital_name"]["source_quote"] = "NOT IN TEXT"
        original_confidence = raw["hospital_name"]["confidence"]
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["hospital_name"].confidence == pytest.approx(original_confidence)

    def test_null_quote_does_not_trigger_check(self):
        raw = _good_raw()
        raw["discharge_date"]["source_quote"] = None
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["discharge_date"].quote_verified is True

    def test_multiple_invalid_quotes_all_marked(self):
        raw = _good_raw()
        raw["hospital_name"]["source_quote"] = "PHANTOM HOSPITAL"
        raw["diagnosis"]["source_quote"] = "PHANTOM DIAGNOSIS"
        stub = StubBackend(response=raw)
        result = extract("transcript", "t1", _SAMPLE_TEXT, stub)
        assert result.facts["hospital_name"].quote_verified is False
        assert result.facts["diagnosis"].quote_verified is False

    def test_verify_quotes_helper_directly(self):
        output = ExtractionOutput.model_validate(_good_raw())
        modified = _verify_quotes(output, "short text")
        # "Apollo Hospital" is not in "short text"
        assert modified.hospital_name.quote_verified is False
        # confidence unchanged
        assert modified.hospital_name.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# T5.6 — LLMJudge
# ---------------------------------------------------------------------------

def _fv(value: str, entity_type: str | None = None) -> FactValue:
    return FactValue(value=value, confidence=0.9, entity_type=entity_type)


class TestLLMJudge:
    def test_satisfies_protocol(self):
        backend = StubBackend(response={"verdict": "MATCH", "rationale": "same disease"})
        judge = LLMJudge(backend)
        assert isinstance(judge, DiagnosisJudge)

    def test_match_verdict_returned(self):
        backend = StubBackend(response={"verdict": "MATCH", "rationale": "identical"})
        judge = LLMJudge(backend)
        verdict, _ = judge.compare(_fv("appendicitis"), _fv("acute appendicitis"))
        assert verdict == "MATCH"

    def test_mismatch_verdict_returned(self):
        backend = StubBackend(response={"verdict": "MISMATCH", "rationale": "disease vs procedure"})
        judge = LLMJudge(backend)
        verdict, _ = judge.compare(_fv("appendicitis"), _fv("appendectomy"))
        assert verdict == "MISMATCH"

    def test_rationale_included_in_note(self):
        backend = StubBackend(response={"verdict": "MATCH", "rationale": "synonym for MI"})
        judge = LLMJudge(backend)
        _, note = judge.compare(_fv("heart attack"), _fv("myocardial infarction"))
        assert "synonym for MI" in note

    def test_invalid_verdict_defaults_to_mismatch(self):
        """Guard against LLM returning garbage verdict."""
        backend = StubBackend(response={"verdict": "UNSURE", "rationale": "unclear"})
        judge = LLMJudge(backend)
        verdict, _ = judge.compare(_fv("flu"), _fv("influenza"))
        assert verdict == "MISMATCH"

    def test_judge_passes_correct_schema(self):
        backend = StubBackend(response={"verdict": "MATCH", "rationale": "ok"})
        judge = LLMJudge(backend)
        from claim_verifier.judge import JUDGE_SCHEMA
        judge.compare(_fv("diabetes"), _fv("type 2 diabetes mellitus"))
        _, schema = backend.calls[0]
        assert schema == JUDGE_SCHEMA

    def test_stub_judge_still_works(self):
        judge = StubJudge(verdict="MATCH", note="stub match")
        verdict, note = judge.compare(_fv("x"), _fv("y"))
        assert verdict == "MATCH"
        assert note == "stub match"


# ---------------------------------------------------------------------------
# T5.7 — ≥5 texts → valid FactSets, fully offline (pre-seeded LLMCache)
# ---------------------------------------------------------------------------

# Five diverse test cases: (source_type, source_id, text, llm_response_dict)
_OFFLINE_CASES: list[tuple[str, str, str, dict]] = [
    (
        "transcript", "call_001",
        (
            "I was admitted to Apollo Hospital on March 12, 2025. "
            "I was diagnosed with appendicitis and stayed for 4 days. "
            "Total bill was Rs. 50,000."
        ),
        {
            "hospital_name":  {"value": "Apollo Hospital",   "confidence": 0.95, "source_quote": "Apollo Hospital",   "entity_type": None},
            "admission_date": {"value": "March 12, 2025",    "confidence": 0.95, "source_quote": "March 12, 2025",    "entity_type": None},
            "discharge_date": {"value": None,                "confidence": 0.0,  "source_quote": None,               "entity_type": None},
            "diagnosis":      {"value": "appendicitis",      "confidence": 0.95, "source_quote": "appendicitis",      "entity_type": "DISEASE"},
            "billed_amount":  {"value": "Rs. 50,000",        "confidence": 0.90, "source_quote": "Rs. 50,000",        "entity_type": None},
            "length_of_stay": {"value": 4,                   "confidence": 0.95, "source_quote": "4 days",            "entity_type": None},
        },
    ),
    (
        "document", "bill_001.pdf",
        (
            "Patient: John Doe\n"
            "Hospital: National General Hospital\n"
            "Date of Admission: 15/02/2025\n"
            "Date of Discharge: 20/02/2025\n"
            "Primary Diagnosis: Type 2 Diabetes Mellitus\n"
            "Total Bill Amount: INR 85,000\n"
            "Length of Stay: 5 days"
        ),
        {
            "hospital_name":  {"value": "National General Hospital",  "confidence": 0.95, "source_quote": "National General Hospital",  "entity_type": None},
            "admission_date": {"value": "15/02/2025",                 "confidence": 0.95, "source_quote": "15/02/2025",                 "entity_type": None},
            "discharge_date": {"value": "20/02/2025",                 "confidence": 0.95, "source_quote": "20/02/2025",                 "entity_type": None},
            "diagnosis":      {"value": "Type 2 Diabetes Mellitus",   "confidence": 0.95, "source_quote": "Type 2 Diabetes Mellitus",   "entity_type": "DISEASE"},
            "billed_amount":  {"value": "INR 85,000",                 "confidence": 0.95, "source_quote": "INR 85,000",                 "entity_type": None},
            "length_of_stay": {"value": 5,                            "confidence": 0.95, "source_quote": "5 days",                     "entity_type": None},
        },
    ),
    (
        "transcript", "call_002",
        (
            "I went to Fortis Hospital on March 5. "
            "The diagnosis was acute bronchitis. "
            "I stayed for 3 nights. Total cost was around 35,000 rupees."
        ),
        {
            "hospital_name":  {"value": "Fortis Hospital",     "confidence": 0.95, "source_quote": "Fortis Hospital",     "entity_type": None},
            "admission_date": {"value": "March 5",             "confidence": 0.90, "source_quote": "March 5",             "entity_type": None},
            "discharge_date": {"value": None,                  "confidence": 0.0,  "source_quote": None,                 "entity_type": None},
            "diagnosis":      {"value": "acute bronchitis",    "confidence": 0.95, "source_quote": "acute bronchitis",    "entity_type": "DISEASE"},
            "billed_amount":  {"value": "35,000 rupees",       "confidence": 0.70, "source_quote": "35,000 rupees",       "entity_type": None},
            "length_of_stay": {"value": 3,                     "confidence": 0.85, "source_quote": "3 nights",            "entity_type": None},
        },
    ),
    (
        "transcript", "call_003",
        (
            "I was in City Medical Center for about a week. "
            "They said I had a cardiac issue. "
            "I don't remember the exact bill."
        ),
        {
            "hospital_name":  {"value": "City Medical Center", "confidence": 0.95, "source_quote": "City Medical Center", "entity_type": None},
            "admission_date": {"value": None,                  "confidence": 0.0,  "source_quote": None,                 "entity_type": None},
            "discharge_date": {"value": None,                  "confidence": 0.0,  "source_quote": None,                 "entity_type": None},
            "diagnosis":      {"value": "cardiac issue",       "confidence": 0.60, "source_quote": "cardiac issue",       "entity_type": "DISEASE"},
            "billed_amount":  {"value": None,                  "confidence": 0.0,  "source_quote": None,                 "entity_type": None},
            "length_of_stay": {"value": 7,                     "confidence": 0.50, "source_quote": "about a week",       "entity_type": None},
        },
    ),
    (
        "document", "bill_002.pdf",
        (
            "DISCHARGE SUMMARY\n"
            "Facility: Sunshine Children's Hospital\n"
            "Admitted: 2025-01-10\n"
            "Discharged: 2025-01-14\n"
            "Diagnosis: Acute Otitis Media\n"
            "Total Charges: ₹42,500\n"
            "Duration: 4 days"
        ),
        {
            "hospital_name":  {"value": "Sunshine Children's Hospital", "confidence": 0.95, "source_quote": "Sunshine Children's Hospital", "entity_type": None},
            "admission_date": {"value": "2025-01-10",                   "confidence": 0.95, "source_quote": "2025-01-10",                   "entity_type": None},
            "discharge_date": {"value": "2025-01-14",                   "confidence": 0.95, "source_quote": "2025-01-14",                   "entity_type": None},
            "diagnosis":      {"value": "Acute Otitis Media",           "confidence": 0.95, "source_quote": "Acute Otitis Media",           "entity_type": "DISEASE"},
            "billed_amount":  {"value": "₹42,500",                      "confidence": 0.95, "source_quote": "₹42,500",                      "entity_type": None},
            "length_of_stay": {"value": 4,                              "confidence": 0.95, "source_quote": "4 days",                        "entity_type": None},
        },
    ),
]

_MODEL = "qwen2.5:7b-instruct-q4_K_M"


@pytest.fixture
def seeded_cache(tmp_path: Path) -> LLMCache:
    """Create a LLMCache pre-seeded with all five offline test responses."""
    cache = LLMCache(tmp_path / "llm_cache")
    for source_type, source_id, text, response_dict in _OFFLINE_CASES:
        messages = _build_messages(source_type, source_id, text)
        cache.put(_MODEL, messages, _ollama_response(response_dict), format=EXTRACTION_SCHEMA)
    return cache


@pytest.mark.parametrize(
    "source_type,source_id,text,expected",
    _OFFLINE_CASES,
    ids=[c[1] for c in _OFFLINE_CASES],
)
def test_offline_extraction_returns_valid_factset(
    source_type: str,
    source_id: str,
    text: str,
    expected: dict,
    seeded_cache: LLMCache,
) -> None:
    """T5.7: each of the 5 fixture texts produces a valid FactSet offline via cache."""
    backend = OllamaBackend(model=_MODEL, cache=seeded_cache)
    result = extract(source_type, source_id, text, backend)

    assert isinstance(result, FactSet), "extract() must return a FactSet"
    assert result.source_type == source_type
    assert result.source_id == source_id
    assert result.extraction_timestamp  # non-empty ISO timestamp

    for field_id in _ALL_FIELDS:
        assert field_id in result.facts, f"FactSet missing field '{field_id}'"
        fv = result.facts[field_id]
        assert 0.0 <= fv.confidence <= 1.0, f"Confidence out of range for '{field_id}'"

    # Verify no live Ollama call was made: re-running the same extraction returns identical values
    result2 = extract(source_type, source_id, text, backend)
    for field_id in _ALL_FIELDS:
        assert result.facts[field_id].value == result2.facts[field_id].value
