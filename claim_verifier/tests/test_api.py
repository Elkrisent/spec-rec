"""
W10 tests — FastAPI web API (T10.3).

TestHealth       : GET /health
TestRoot         : GET / serves HTML
TestVerify       : POST /verify happy-path and input validation
TestAuth         : API key enforcement
"""

from __future__ import annotations

import sys
from io import BytesIO
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from claim_verifier.api.app import create_app
from claim_verifier.judge import StubJudge

# ---------------------------------------------------------------------------
# Shared stubs and helpers
# ---------------------------------------------------------------------------

_TRANSCRIPT_TEXT = (
    "The patient was admitted to Apollo Hospital on March 12 for appendicitis. "
    "The total bill was approximately fifty thousand rupees."
)

_TRANSCRIPT_RAW: dict[str, Any] = {
    "hospital_name":  {"value": "Apollo Hospital",  "confidence": 0.95, "source_quote": "Apollo Hospital",        "entity_type": None},
    "admission_date": {"value": "March 12",          "confidence": 0.90, "source_quote": "March 12",               "entity_type": None},
    "discharge_date": {"value": None,                "confidence": 0.0,  "source_quote": None,                     "entity_type": None},
    "diagnosis":      {"value": "appendicitis",      "confidence": 0.85, "source_quote": "appendicitis",           "entity_type": "DISEASE"},
    "billed_amount":  {"value": 50000,               "confidence": 0.75, "source_quote": "fifty thousand rupees",  "entity_type": None},
    "length_of_stay": {"value": None,                "confidence": 0.0,  "source_quote": None,                     "entity_type": None},
}

_DOCUMENT_RAW: dict[str, Any] = {
    "hospital_name":  {"value": "Apollo Hospitals",  "confidence": 1.0, "source_quote": "Apollo Hospitals",         "entity_type": None},
    "admission_date": {"value": "12/03/2025",         "confidence": 1.0, "source_quote": "Date of Admission: 12/03/2025", "entity_type": None},
    "discharge_date": {"value": "16/03/2025",         "confidence": 1.0, "source_quote": "Date of Discharge: 16/03/2025", "entity_type": None},
    "diagnosis":      {"value": "Acute Appendicitis", "confidence": 1.0, "source_quote": "Acute Appendicitis",       "entity_type": "DISEASE"},
    "billed_amount":  {"value": 62000,                "confidence": 1.0, "source_quote": "Rs.62,000",                "entity_type": None},
    "length_of_stay": {"value": 4,                    "confidence": 1.0, "source_quote": "4 days",                   "entity_type": None},
}


class _SeqStub:
    """Returns one response per call in sequence."""

    def __init__(self, *responses: dict) -> None:
        self._iter = iter(responses)

    def complete(self, messages: list[dict], schema: dict | None = None) -> dict:
        return next(self._iter)


def _make_pdf(text: str = "") -> bytes:
    """Generate a minimal text-layer PDF using reportlab."""
    pytest.importorskip("reportlab.pdfgen.canvas")
    from reportlab.pdfgen.canvas import Canvas

    full_text = text or (
        "Apollo Hospitals Jubilee Hills. "
        "Date of Admission: 12/03/2025. Date of Discharge: 16/03/2025. "
        "Primary Diagnosis: Acute Appendicitis. Total Bill Amount: Rs.62,000. "
        "Length of Stay: 4 days. Patient covered under Group Health Insurance."
    )
    buf = BytesIO()
    c = Canvas(buf)
    c.drawString(50, 700, full_text)
    c.save()
    return buf.getvalue()


@pytest.fixture
def pdf_bytes():
    return _make_pdf()


@pytest.fixture
def client(pdf_bytes):
    """TestClient with SeqStub backend — no Ollama needed."""
    backend = _SeqStub(_TRANSCRIPT_RAW, _DOCUMENT_RAW)
    app = create_app(backend=backend, judge=StubJudge("MATCH"))
    return TestClient(app)


@pytest.fixture
def client_factory():
    """Factory for creating clients with custom backends."""
    def _make(backend=None, judge=None, api_key=None):
        bk = backend or _SeqStub(_TRANSCRIPT_RAW, _DOCUMENT_RAW)
        jg = judge or StubJudge("MATCH")
        return TestClient(create_app(backend=bk, judge=jg, api_key=api_key))
    return _make


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, client) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client) -> None:
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_has_backend_key(self, client) -> None:
        resp = client.get("/health")
        assert "backend" in resp.json()


# ---------------------------------------------------------------------------
# TestRoot
# ---------------------------------------------------------------------------


class TestRoot:
    def test_root_returns_200(self, client) -> None:
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_content_type_html(self, client) -> None:
        resp = client.get("/")
        assert "text/html" in resp.headers["content-type"]

    def test_root_contains_form(self, client) -> None:
        resp = client.get("/")
        assert "<form" in resp.text.lower()


# ---------------------------------------------------------------------------
# TestVerify
# ---------------------------------------------------------------------------


class TestVerify:
    def _post(self, client, pdf_bytes, *, claim_id="TEST001", transcript=True):
        files: dict = {"document": ("bill.pdf", pdf_bytes, "application/pdf")}
        if transcript:
            files["transcript"] = ("t.txt", _TRANSCRIPT_TEXT.encode(), "text/plain")
        return client.post(
            "/verify",
            data={"claim_id": claim_id},
            files=files,
        )

    def test_verify_returns_200(self, client, pdf_bytes) -> None:
        resp = self._post(client, pdf_bytes)
        assert resp.status_code == 200

    def test_verify_response_has_report(self, client, pdf_bytes) -> None:
        resp = self._post(client, pdf_bytes)
        assert "report" in resp.json()

    def test_verify_report_is_markdown(self, client, pdf_bytes) -> None:
        resp = self._post(client, pdf_bytes)
        report = resp.json()["report"]
        assert isinstance(report, str) and len(report) > 0
        assert "#" in report  # markdown heading

    def test_verify_claim_id_in_response(self, client, pdf_bytes) -> None:
        resp = self._post(client, pdf_bytes, claim_id="CLM-999")
        assert resp.json()["claim_id"] == "CLM-999"

    def test_verify_risk_level_valid(self, client, pdf_bytes) -> None:
        resp = self._post(client, pdf_bytes)
        valid = {"LOW", "MEDIUM", "HIGH", "INSUFFICIENT_DATA"}
        assert resp.json()["risk_level"] in valid

    def test_verify_errors_list_present(self, client, pdf_bytes) -> None:
        resp = self._post(client, pdf_bytes)
        assert "errors" in resp.json()
        assert isinstance(resp.json()["errors"], list)

    def test_verify_missing_claim_id_returns_422(self, client, pdf_bytes) -> None:
        resp = client.post(
            "/verify",
            files={
                "transcript": ("t.txt", b"text", "text/plain"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
        )
        assert resp.status_code == 422

    def test_verify_missing_document_returns_422(self, client) -> None:
        resp = client.post(
            "/verify",
            data={"claim_id": "X"},
            files={"transcript": ("t.txt", b"text", "text/plain")},
        )
        assert resp.status_code == 422

    def test_verify_neither_transcript_nor_audio_returns_400(self, client, pdf_bytes) -> None:
        resp = client.post(
            "/verify",
            data={"claim_id": "X"},
            files={"document": ("bill.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 400

    def test_verify_both_transcript_and_audio_returns_400(self, client, pdf_bytes) -> None:
        resp = client.post(
            "/verify",
            data={"claim_id": "X"},
            files={
                "transcript": ("t.txt", b"text", "text/plain"),
                "audio": ("call.wav", b"data", "audio/wav"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
        )
        assert resp.status_code == 400

    def test_verify_extraction_failure_still_200(self, pdf_bytes, client_factory) -> None:
        # Backend returns {} → ExtractionError → pipeline returns partial result
        class _AlwaysEmpty:
            def complete(self, messages, schema=None):
                return {}

        cl = client_factory(backend=_AlwaysEmpty())
        resp = cl.post(
            "/verify",
            data={"claim_id": "FAIL"},
            files={
                "transcript": ("t.txt", _TRANSCRIPT_TEXT.encode(), "text/plain"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["risk_level"] == "INSUFFICIENT_DATA"
        assert len(data["errors"]) > 0

    def test_verify_audio_path_calls_ingest_audio(self, pdf_bytes, client_factory) -> None:
        """Audio file → ingest_audio is called → run_from_text produces a report."""
        backend = _SeqStub(_TRANSCRIPT_RAW, _DOCUMENT_RAW)
        cl = client_factory(backend=backend)

        with patch(
            "claim_verifier.api.app.ingest_audio",
            return_value=_TRANSCRIPT_TEXT,
        ):
            resp = cl.post(
                "/verify",
                data={"claim_id": "AUDIO01"},
                files={
                    "audio": ("call.wav", b"fake-audio", "audio/wav"),
                    "document": ("bill.pdf", pdf_bytes, "application/pdf"),
                },
            )
        assert resp.status_code == 200
        assert resp.json()["claim_id"] == "AUDIO01"

    def test_verify_audio_ingestion_error_returns_200(self, pdf_bytes, client_factory) -> None:
        """IngestionError from ingest_audio → 200 with error info (graceful)."""
        cl = client_factory()
        from claim_verifier.stages.ingestion import IngestionError

        with patch(
            "claim_verifier.api.app.ingest_audio",
            side_effect=IngestionError("Non-English audio"),
        ):
            resp = cl.post(
                "/verify",
                data={"claim_id": "BAUDIO"},
                files={
                    "audio": ("call.wav", b"fake-audio", "audio/wav"),
                    "document": ("bill.pdf", pdf_bytes, "application/pdf"),
                },
            )
        assert resp.status_code == 200
        assert resp.json()["risk_level"] == "INSUFFICIENT_DATA"
        assert resp.json()["errors"]


# ---------------------------------------------------------------------------
# TestAuth
# ---------------------------------------------------------------------------


class TestAuth:
    def test_no_key_required_when_not_configured(self, pdf_bytes, client_factory) -> None:
        cl = client_factory(api_key=None)
        resp = cl.post(
            "/verify",
            data={"claim_id": "X"},
            files={
                "transcript": ("t.txt", _TRANSCRIPT_TEXT.encode(), "text/plain"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
        )
        assert resp.status_code == 200

    def test_correct_key_passes(self, pdf_bytes, client_factory) -> None:
        cl = client_factory(api_key="secret123")
        resp = cl.post(
            "/verify",
            data={"claim_id": "X"},
            files={
                "transcript": ("t.txt", _TRANSCRIPT_TEXT.encode(), "text/plain"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
            headers={"X-API-Key": "secret123"},
        )
        assert resp.status_code == 200

    def test_missing_key_returns_403(self, pdf_bytes, client_factory) -> None:
        cl = client_factory(api_key="secret123")
        resp = cl.post(
            "/verify",
            data={"claim_id": "X"},
            files={
                "transcript": ("t.txt", _TRANSCRIPT_TEXT.encode(), "text/plain"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
        )
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self, pdf_bytes, client_factory) -> None:
        cl = client_factory(api_key="secret123")
        resp = cl.post(
            "/verify",
            data={"claim_id": "X"},
            files={
                "transcript": ("t.txt", _TRANSCRIPT_TEXT.encode(), "text/plain"),
                "document": ("bill.pdf", pdf_bytes, "application/pdf"),
            },
            headers={"X-API-Key": "wrongkey"},
        )
        assert resp.status_code == 403

    def test_health_accessible_without_key(self, client_factory) -> None:
        cl = client_factory(api_key="secret")
        resp = cl.get("/health")
        assert resp.status_code == 200
