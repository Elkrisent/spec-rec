"""
W6 tests — ingestion, redaction, pipeline (T6.1–T6.5).

TestIngestion  : ingest_transcript / ingest_document happy-path and error cases
TestRedaction  : redact() covers all four PII pattern families
TestPipeline   : run_from_text() E2E + failure injection; run() ingestion-failure
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from claim_verifier.backends import StubBackend
from claim_verifier.judge import StubJudge
from claim_verifier.pipeline import PipelineResult, run, run_from_text
from claim_verifier.redaction import redact
from claim_verifier.stages.ingestion import IngestionError, ingest_document, ingest_transcript

# ---------------------------------------------------------------------------
# Shared fixtures & helpers
# ---------------------------------------------------------------------------

# Canned transcript text — all quotes used in _TRANSCRIPT_RAW appear verbatim here.
_TRANSCRIPT_TEXT = (
    "The patient was admitted to Apollo Hospital on March 12 for appendicitis. "
    "The total bill was approximately fifty thousand rupees."
)

# Canned document text — all quotes used in _DOCUMENT_RAW appear verbatim here.
_DOCUMENT_TEXT = (
    "Apollo Hospitals, Jubilee Hills\n"
    "Date of Admission: 12/03/2025\n"
    "Date of Discharge: 16/03/2025\n"
    "Primary Diagnosis: Acute Appendicitis\n"
    "Total Bill Amount: Rs.62,000\n"
    "Length of Stay: 4 days"
)

# Extraction-schema-shaped responses for StubBackend / SeqStub.
_TRANSCRIPT_RAW: dict[str, Any] = {
    "hospital_name":  {"value": "Apollo Hospital",    "confidence": 0.95, "source_quote": "Apollo Hospital",        "entity_type": None},
    "admission_date": {"value": "March 12",            "confidence": 0.90, "source_quote": "March 12",               "entity_type": None},
    "discharge_date": {"value": None,                  "confidence": 0.0,  "source_quote": None,                     "entity_type": None},
    "diagnosis":      {"value": "appendicitis",        "confidence": 0.85, "source_quote": "appendicitis",           "entity_type": "DISEASE"},
    "billed_amount":  {"value": 50000,                 "confidence": 0.75, "source_quote": "fifty thousand rupees",  "entity_type": None},
    "length_of_stay": {"value": None,                  "confidence": 0.0,  "source_quote": None,                     "entity_type": None},
}

_DOCUMENT_RAW: dict[str, Any] = {
    "hospital_name":  {"value": "Apollo Hospitals",    "confidence": 1.0,  "source_quote": "Apollo Hospitals, Jubilee Hills",     "entity_type": None},
    "admission_date": {"value": "12/03/2025",           "confidence": 1.0,  "source_quote": "Date of Admission: 12/03/2025",       "entity_type": None},
    "discharge_date": {"value": "16/03/2025",           "confidence": 1.0,  "source_quote": "Date of Discharge: 16/03/2025",       "entity_type": None},
    "diagnosis":      {"value": "Acute Appendicitis",   "confidence": 1.0,  "source_quote": "Primary Diagnosis: Acute Appendicitis","entity_type": "DISEASE"},
    "billed_amount":  {"value": 62000,                  "confidence": 1.0,  "source_quote": "Total Bill Amount: Rs.62,000",         "entity_type": None},
    "length_of_stay": {"value": 4,                      "confidence": 1.0,  "source_quote": "Length of Stay: 4 days",               "entity_type": None},
}

_NON_ENGLISH_TEXT = (
    "Dies ist ein medizinischer Bericht auf Deutsch. Der Patient wurde ins Krankenhaus "
    "eingeliefert. Diagnose: Blinddarmentzündung. Aufnahmedatum: 12. März 2025."
)


class _SeqStub:
    """Stateful stub returning one response per call in sequence."""

    def __init__(self, responses: list[dict]) -> None:
        self._iter = iter(responses)
        self.call_count = 0

    def complete(self, messages: list[dict], schema: dict | None = None) -> dict:
        self.call_count += 1
        return next(self._iter)


class _AlwaysEmptyStub:
    """Returns {} every time — triggers ExtractionError after retry."""

    def complete(self, messages: list[dict], schema: dict | None = None) -> dict:
        return {}


# ---------------------------------------------------------------------------
# TestIngestion
# ---------------------------------------------------------------------------


class TestIngestion:
    def test_transcript_happy_path(self, tmp_path: Path) -> None:
        f = tmp_path / "t.txt"
        f.write_text(_TRANSCRIPT_TEXT, encoding="utf-8")
        text = ingest_transcript(f)
        assert "Apollo Hospital" in text

    def test_transcript_strips_leading_trailing_whitespace(self, tmp_path: Path) -> None:
        f = tmp_path / "t.txt"
        f.write_text("  " + _TRANSCRIPT_TEXT + "\n\n", encoding="utf-8")
        text = ingest_transcript(f)
        assert text == _TRANSCRIPT_TEXT

    def test_transcript_returns_full_content(self, tmp_path: Path) -> None:
        f = tmp_path / "t.txt"
        f.write_text(_TRANSCRIPT_TEXT, encoding="utf-8")
        text = ingest_transcript(f)
        assert text == _TRANSCRIPT_TEXT

    def test_transcript_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="Cannot read transcript"):
            ingest_transcript(tmp_path / "missing.txt")

    def test_transcript_empty_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        with pytest.raises(IngestionError, match="empty"):
            ingest_transcript(f)

    def test_transcript_whitespace_only_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "ws.txt"
        f.write_text("   \n\t  ", encoding="utf-8")
        with pytest.raises(IngestionError, match="empty"):
            ingest_transcript(f)

    def test_transcript_non_english_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "de.txt"
        f.write_text(_NON_ENGLISH_TEXT, encoding="utf-8")
        with pytest.raises(IngestionError, match="non-English"):
            ingest_transcript(f)

    def test_transcript_non_english_error_mentions_language(self, tmp_path: Path) -> None:
        f = tmp_path / "de.txt"
        f.write_text(_NON_ENGLISH_TEXT, encoding="utf-8")
        with pytest.raises(IngestionError) as exc_info:
            ingest_transcript(f)
        assert "language=" in str(exc_info.value)

    def test_transcript_error_mentions_path(self, tmp_path: Path) -> None:
        p = tmp_path / "missing_path.txt"
        with pytest.raises(IngestionError) as exc_info:
            ingest_transcript(p)
        assert "missing_path.txt" in str(exc_info.value)

    def test_document_non_pdf_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "notapdf.pdf"
        f.write_text("this is not a pdf", encoding="utf-8")
        with pytest.raises(IngestionError, match="Cannot read document"):
            ingest_document(f)

    def test_document_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestionError, match="Cannot read document"):
            ingest_document(tmp_path / "no_such.pdf")

    def test_ingestion_error_is_exception(self) -> None:
        assert issubclass(IngestionError, Exception)

    def test_document_scanned_rejection(self, tmp_path: Path) -> None:
        """PDF with < 100 chars from pdfplumber where OCR also yields < 100 chars → IngestionError."""
        rl = pytest.importorskip("reportlab.pdfgen.canvas")
        pdf_path = tmp_path / "scanned.pdf"
        c = rl.Canvas(str(pdf_path))
        c.drawString(100, 700, "Hi")  # deliberately short
        c.save()
        with patch("claim_verifier.stages.ingestion._ocr_pdf", return_value="Hi"):
            with pytest.raises(IngestionError, match="characters"):
                ingest_document(pdf_path)

    def test_document_ocr_fallback_success(self, tmp_path: Path) -> None:
        """Scanned PDF where OCR yields sufficient English text → returns OCR text."""
        rl = pytest.importorskip("reportlab.pdfgen.canvas")
        pdf_path = tmp_path / "scanned.pdf"
        c = rl.Canvas(str(pdf_path))
        c.drawString(100, 700, "Hi")  # pdfplumber yields < 100 chars
        c.save()
        ocr_text = (
            "Apollo Hospitals Jubilee Hills. Date of Admission: 12/03/2025. "
            "Primary Diagnosis: Acute Appendicitis. Total Bill Amount: Rs 62000. "
            "Length of Stay: 4 days."
        )
        with patch("claim_verifier.stages.ingestion._ocr_pdf", return_value=ocr_text):
            text = ingest_document(pdf_path)
        assert "Apollo" in text
        assert len(text) >= 100

    def test_document_ocr_not_installed(self, tmp_path: Path) -> None:
        """Scanned PDF when OCR dependencies are absent → IngestionError propagated."""
        rl = pytest.importorskip("reportlab.pdfgen.canvas")
        pdf_path = tmp_path / "scanned.pdf"
        c = rl.Canvas(str(pdf_path))
        c.drawString(100, 700, "Hi")  # pdfplumber yields < 100 chars
        c.save()
        with patch(
            "claim_verifier.stages.ingestion._ocr_pdf",
            side_effect=IngestionError("OCR dependencies are not installed"),
        ):
            with pytest.raises(IngestionError, match="OCR"):
                ingest_document(pdf_path)

    def test_document_happy_path(self, tmp_path: Path) -> None:
        """Valid English PDF with enough text → returns extracted text."""
        rl = pytest.importorskip("reportlab.pdfgen.canvas")
        long_text = (
            "Apollo Hospitals Jubilee Hills. "
            "Date of Admission: 12/03/2025. "
            "Primary Diagnosis: Acute Appendicitis. "
            "Total Bill Amount: Rs 62000. "
            "Length of Stay: 4 days. "
            "All charges are as per standard hospital rates."
        )
        pdf_path = tmp_path / "bill.pdf"
        c = rl.Canvas(str(pdf_path))
        c.drawString(50, 700, long_text)
        c.save()
        text = ingest_document(pdf_path)
        assert len(text) >= 100
        assert "Apollo" in text

    def test_document_non_english_raises(self, tmp_path: Path) -> None:
        """PDF with non-English text → IngestionError."""
        rl = pytest.importorskip("reportlab.pdfgen.canvas")
        # Use ASCII-safe German text to avoid font encoding issues
        german_text = (
            "Dies ist ein medizinischer Bericht. Der Patient wurde ins Krankenhaus "
            "eingeliefert. Diagnose: Blinddarmentzuendung. Aufnahmedatum: 12. Maerz 2025. "
            "Entlassungsdatum: 16. Maerz 2025. Gesamtbetrag: 62000 Rupien. Aufenthalt: 4 Tage."
        )
        pdf_path = tmp_path / "german.pdf"
        c = rl.Canvas(str(pdf_path))
        c.drawString(50, 700, german_text)
        c.save()
        with pytest.raises(IngestionError, match="non-English"):
            ingest_document(pdf_path)


# ---------------------------------------------------------------------------
# TestRedaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_redacts_email(self) -> None:
        assert "<EMAIL>" in redact("Contact: john.doe@example.com for details")

    def test_redacts_pan(self) -> None:
        assert "<PAN>" in redact("PAN: ABCDE1234F was submitted")

    def test_redacts_aadhaar_space_delimited(self) -> None:
        assert "<AADHAAR>" in redact("Aadhaar: 1234 5678 9012")

    def test_redacts_aadhaar_hyphen_delimited(self) -> None:
        assert "<AADHAAR>" in redact("ID: 1234-5678-9012")

    def test_redacts_aadhaar_plain_12_digits(self) -> None:
        assert "<AADHAAR>" in redact("Aadhaar number: 234567890123")

    def test_redacts_phone_10_digit(self) -> None:
        assert "<PHONE>" in redact("Call us at 9876543210")

    def test_redacts_phone_with_plus91(self) -> None:
        assert "<PHONE>" in redact("Mobile: +919876543210")

    def test_redacts_phone_with_0_prefix(self) -> None:
        assert "<PHONE>" in redact("Phone: 09876543210")

    def test_clean_text_unchanged(self) -> None:
        clean = "Patient admitted to Apollo Hospital on March 12 for appendicitis."
        assert redact(clean) == clean

    def test_multiple_pii_types_all_redacted(self) -> None:
        text = "Email: patient@clinic.in, Phone: 9876543210, PAN: ABCDE1234F"
        result = redact(text)
        assert "<EMAIL>" in result
        assert "<PHONE>" in result
        assert "<PAN>" in result
        assert "patient@clinic.in" not in result
        assert "9876543210" not in result
        assert "ABCDE1234F" not in result

    def test_medical_amounts_not_redacted(self) -> None:
        text = "Total bill: Rs.62,000. Bill amount: 50000."
        assert redact(text) == text

    def test_dates_not_redacted(self) -> None:
        text = "Admission: 12/03/2025. Discharge: 16/03/2025."
        assert redact(text) == text

    def test_redact_does_not_mutate_input(self) -> None:
        original = "Email: test@example.com"
        original_copy = original
        _ = redact(original)
        assert original == original_copy

    @pytest.mark.parametrize("pii,label", [
        ("user@domain.org",   "<EMAIL>"),
        ("PQRST5678Z",        "<PAN>"),
        ("9988 7766 5544",    "<AADHAAR>"),
        ("8765432109",        "<PHONE>"),
    ])
    def test_pii_replaced_with_correct_label(self, pii: str, label: str) -> None:
        result = redact(f"value is {pii} here")
        assert label in result
        assert pii not in result


# ---------------------------------------------------------------------------
# TestPipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    # ---- happy-path E2E ----

    def test_run_from_text_returns_pipeline_result(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert isinstance(result, PipelineResult)

    def test_run_from_text_claim_id_preserved(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C999", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert result.claim_id == "C999"

    def test_run_from_text_no_errors_on_success(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert result.errors == []

    def test_run_from_text_report_is_nonempty_string(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert isinstance(result.report, str) and len(result.report) > 0

    def test_run_from_text_report_contains_claim_id(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C-REPORT-ID", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert "C-REPORT-ID" in result.report

    def test_run_from_text_report_is_markdown(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert result.report.startswith("# Claim Verification Report")

    def test_run_from_text_verification_result_present(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert result.verification_result is not None

    def test_run_from_text_risk_level_is_valid(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        result = run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        vr = result.verification_result
        assert vr is not None
        assert vr.risk_level in ("LOW", "MEDIUM", "HIGH", "INSUFFICIENT_DATA")

    def test_run_from_text_backend_called_twice(self) -> None:
        backend = _SeqStub([_TRANSCRIPT_RAW, _DOCUMENT_RAW])
        run_from_text("C001", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, backend, StubJudge("MATCH"))
        assert backend.call_count == 2

    # ---- extraction failure → partial report ----

    def test_extraction_failure_partial_report_nonempty(self) -> None:
        result = run_from_text("C-FAIL", "some text", "some doc", _AlwaysEmptyStub(), StubJudge())
        assert isinstance(result.report, str) and len(result.report) > 0

    def test_extraction_failure_errors_populated(self) -> None:
        result = run_from_text("C-FAIL", "some text", "some doc", _AlwaysEmptyStub(), StubJudge())
        assert len(result.errors) > 0

    def test_extraction_failure_risk_insufficient_data(self) -> None:
        result = run_from_text("C-FAIL", "some text", "some doc", _AlwaysEmptyStub(), StubJudge())
        assert result.verification_result is not None
        assert result.verification_result.risk_level == "INSUFFICIENT_DATA"

    def test_extraction_failure_has_extraction_failure_flag(self) -> None:
        result = run_from_text("C-FAIL", "some text", "some doc", _AlwaysEmptyStub(), StubJudge())
        vr = result.verification_result
        assert vr is not None
        flag_types = [f.type for f in vr.flags]
        assert "EXTRACTION_FAILURE" in flag_types

    def test_extraction_failure_flag_is_high_severity(self) -> None:
        result = run_from_text("C-FAIL", "some text", "some doc", _AlwaysEmptyStub(), StubJudge())
        vr = result.verification_result
        assert vr is not None
        error_flags = [f for f in vr.flags if f.type == "EXTRACTION_FAILURE"]
        assert all(f.severity == "HIGH" for f in error_flags)

    def test_extraction_failure_report_contains_insufficient_data(self) -> None:
        result = run_from_text("C-FAIL", "some text", "some doc", _AlwaysEmptyStub(), StubJudge())
        assert "INSUFFICIENT" in result.report

    # ---- document extraction failure ----

    def test_document_extraction_failure_partial_report(self) -> None:
        # First call (transcript) succeeds, second call (document) fails
        class _FirstOkThenEmpty:
            def __init__(self) -> None:
                self._call = 0

            def complete(self, messages: list[dict], schema: dict | None = None) -> dict:
                self._call += 1
                return _TRANSCRIPT_RAW if self._call == 1 else {}

        result = run_from_text("C-DOCFAIL", _TRANSCRIPT_TEXT, _DOCUMENT_TEXT, _FirstOkThenEmpty(), StubJudge())
        assert result.verification_result is not None
        assert result.verification_result.risk_level == "INSUFFICIENT_DATA"
        assert len(result.errors) > 0

    # ---- run() ingestion-failure path ----

    def test_run_ingestion_failure_returns_pipeline_result(self, tmp_path: Path) -> None:
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("", encoding="utf-8")
        doc_placeholder = tmp_path / "doc.pdf"
        result = run("C-INGEST", empty_txt, doc_placeholder, _AlwaysEmptyStub(), StubJudge())
        assert isinstance(result, PipelineResult)

    def test_run_ingestion_failure_errors_populated(self, tmp_path: Path) -> None:
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("", encoding="utf-8")
        doc_placeholder = tmp_path / "doc.pdf"
        result = run("C-INGEST", empty_txt, doc_placeholder, _AlwaysEmptyStub(), StubJudge())
        assert len(result.errors) > 0

    def test_run_ingestion_failure_partial_report(self, tmp_path: Path) -> None:
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("", encoding="utf-8")
        doc_placeholder = tmp_path / "doc.pdf"
        result = run("C-INGEST", empty_txt, doc_placeholder, _AlwaysEmptyStub(), StubJudge())
        assert isinstance(result.report, str) and len(result.report) > 0

    def test_run_ingestion_failure_has_ingestion_failure_flag(self, tmp_path: Path) -> None:
        empty_txt = tmp_path / "empty.txt"
        empty_txt.write_text("", encoding="utf-8")
        doc_placeholder = tmp_path / "doc.pdf"
        result = run("C-INGEST", empty_txt, doc_placeholder, _AlwaysEmptyStub(), StubJudge())
        vr = result.verification_result
        assert vr is not None
        assert any(f.type == "INGESTION_FAILURE" for f in vr.flags)

    def test_run_ingestion_failure_does_not_crash(self, tmp_path: Path) -> None:
        non_english_txt = tmp_path / "de.txt"
        non_english_txt.write_text(_NON_ENGLISH_TEXT, encoding="utf-8")
        doc_placeholder = tmp_path / "doc.pdf"
        # Must not raise — errors are surfaced via PipelineResult
        result = run("C-LANG", non_english_txt, doc_placeholder, _AlwaysEmptyStub(), StubJudge())
        assert result.verification_result is not None
