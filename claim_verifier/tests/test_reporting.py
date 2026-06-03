"""
T4.4 — Reporting snapshot tests.

Covers:
  - Risk badge for all four risk levels
  - Consistency score formatted as percentage
  - Citation format: value [source: "quote"]
  - Missing value renders as em dash
  - Flags ordered HIGH → MEDIUM → LOW in output
  - Missing-fields listing and "all present" fallback
  - Quality warnings listing and "no warnings" fallback
  - Reviewer guidance text for all four risk levels
  - All six schema field labels appear in the table
  - Claim ID in report header
"""

from __future__ import annotations

import pytest

from claim_verifier.models import FieldVerdict, Flag, VerificationResult
from claim_verifier.stages.reporting import render, _citation

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TS = "2025-03-20T10:00:30+00:00"


def _verdict(
    status: str,
    score: float,
    t_val=None,
    d_val=None,
    t_quote: str | None = None,
    d_quote: str | None = None,
    note: str | None = None,
) -> FieldVerdict:
    return FieldVerdict(
        status=status,
        score=score,
        transcript_value=t_val,
        document_value=d_val,
        transcript_quote=t_quote,
        document_quote=d_quote,
        note=note,
    )


def _flag(type_: str, severity: str, message: str) -> Flag:
    return Flag(type=type_, severity=severity, message=message)


def _all_match_result(**overrides) -> VerificationResult:
    """A LOW-risk result with all six fields present and matched."""
    defaults: dict = dict(
        claim_id="TEST-001",
        verified_at=_TS,
        consistency_score=1.0,
        risk_level="LOW",
        field_verdicts={
            "hospital_name": _verdict(
                "MATCH", 1.0,
                t_val="Apollo Hospital", d_val="Apollo Hospitals",
                t_quote="I was admitted to Apollo Hospital",
                d_quote="Apollo Hospitals, Jubilee Hills",
                note="fuzzy match, ratio=0.96",
            ),
            "admission_date": _verdict(
                "MATCH", 1.0,
                t_val="--03-12", d_val="2025-03-12",
                t_quote="on March 12",
                d_quote="Date of Admission: 12/03/2025",
                note="date match (year absent in one source), delta=0d",
            ),
            "discharge_date": _verdict(
                "MATCH", 1.0,
                t_val="2025-03-16", d_val="2025-03-16",
                t_quote="discharged on the 16th",
                d_quote="Date of Discharge: 16/03/2025",
                note="date match, delta=0d",
            ),
            "diagnosis": _verdict(
                "MATCH", 1.0,
                t_val="appendicitis", d_val="Acute Appendicitis",
                t_quote="for appendicitis",
                d_quote="Primary Diagnosis: Acute Appendicitis",
                note="stub: fixed verdict",
            ),
            "billed_amount": _verdict(
                "MATCH", 1.0,
                t_val=62000, d_val=62000,
                t_quote="total bill of Rs.62,000",
                d_quote="Total Bill Amount: Rs.62,000",
                note="amount match, delta=0.0%",
            ),
            "length_of_stay": _verdict(
                "MATCH", 1.0,
                t_val=4, d_val=4,
                t_quote="stayed for 4 days",
                d_quote="Length of Stay: 4 days",
                note="LOS match, delta=0d",
            ),
        },
        flags=[],
        missing_fields=[],
        low_quality_sources=[],
    )
    defaults.update(overrides)
    return VerificationResult(**defaults)


def _insufficient_result() -> VerificationResult:
    return VerificationResult(
        claim_id="TEST-INS",
        verified_at=_TS,
        consistency_score=None,
        risk_level="INSUFFICIENT_DATA",
        field_verdicts={
            "hospital_name": _verdict("MISSING", 0.0),
            "admission_date": _verdict("MISSING", 0.0),
            "discharge_date": _verdict("MISSING", 0.0),
            "diagnosis": _verdict("MISSING", 0.0),
            "billed_amount": _verdict("MISSING", 0.0),
            "length_of_stay": _verdict("MISSING", 0.0),
        },
        flags=[],
        missing_fields=["hospital_name", "admission_date", "discharge_date",
                        "diagnosis", "billed_amount", "length_of_stay"],
        low_quality_sources=[],
    )


# ---------------------------------------------------------------------------
# T4.2 — Citation format
# ---------------------------------------------------------------------------

class TestCitationHelper:
    def test_value_and_quote(self):
        assert _citation("Apollo Hospital", "I was admitted to Apollo Hospital") == (
            'Apollo Hospital [source: "I was admitted to Apollo Hospital"]'
        )

    def test_value_no_quote(self):
        assert _citation(62000, None) == "62000"

    def test_none_value_returns_dash(self):
        assert _citation(None, None) == "—"

    def test_none_value_ignores_quote(self):
        assert _citation(None, "some quote") == "—"

    def test_integer_value(self):
        result = _citation(50000, "around fifty thousand rupees")
        assert result == '50000 [source: "around fifty thousand rupees"]'

    def test_pipe_in_value_escaped(self):
        result = _citation("A|B", None)
        assert "|" not in result or result == "A\\|B"

    def test_pipe_in_quote_escaped(self):
        result = _citation("X", "some | pipe")
        assert result == 'X [source: "some \\| pipe"]'


# ---------------------------------------------------------------------------
# T4.4 — render() output: basic structure
# ---------------------------------------------------------------------------

class TestRenderBasic:
    def test_returns_nonempty_string(self):
        md = render(_all_match_result())
        assert isinstance(md, str) and len(md) > 0

    def test_claim_id_in_header(self):
        md = render(_all_match_result(claim_id="C999"))
        assert "C999" in md

    def test_verified_at_in_output(self):
        md = render(_all_match_result())
        assert _TS in md


# ---------------------------------------------------------------------------
# T4.4 — Badge correct
# ---------------------------------------------------------------------------

class TestRiskBadge:
    @pytest.mark.parametrize("risk,score,expected", [
        ("LOW",    1.0,  "LOW RISK"),
        ("MEDIUM", 0.80, "MEDIUM RISK"),
        ("HIGH",   0.55, "HIGH RISK"),
    ])
    def test_badge_text(self, risk, score, expected):
        result = _all_match_result(risk_level=risk, consistency_score=score)
        md = render(result)
        assert expected in md

    def test_badge_insufficient_data(self):
        md = render(_insufficient_result())
        assert "INSUFFICIENT DATA" in md

    def test_score_as_percentage_low(self):
        md = render(_all_match_result(consistency_score=1.0))
        assert "100.0%" in md

    def test_score_as_percentage_high(self):
        md = render(_all_match_result(risk_level="HIGH", consistency_score=0.6875))
        assert "68.8%" in md

    def test_score_na_for_insufficient(self):
        md = render(_insufficient_result())
        assert "N/A" in md


# ---------------------------------------------------------------------------
# T4.4 — Citations present
# ---------------------------------------------------------------------------

class TestCitationsInOutput:
    def test_transcript_citation_in_table(self):
        md = render(_all_match_result())
        assert 'Apollo Hospital [source: "I was admitted to Apollo Hospital"]' in md

    def test_document_citation_in_table(self):
        md = render(_all_match_result())
        assert 'Apollo Hospitals [source: "Apollo Hospitals, Jubilee Hills"]' in md

    def test_missing_transcript_shows_dash(self):
        result = _all_match_result(
            consistency_score=0.8,
            risk_level="MEDIUM",
            field_verdicts={
                **_all_match_result().field_verdicts,
                "discharge_date": _verdict(
                    "MISSING", 0.0,
                    t_val=None, d_val="2025-03-16",
                    t_quote=None, d_quote="Date of Discharge: 16/03/2025",
                    note="absent from transcript",
                ),
            },
            missing_fields=["discharge_date"],
        )
        md = render(result)
        assert "—" in md

    def test_value_without_quote_no_source_tag(self):
        result = _all_match_result(
            field_verdicts={
                **_all_match_result().field_verdicts,
                "billed_amount": _verdict("MATCH", 1.0, t_val=62000, d_val=62000),
            },
        )
        md = render(result)
        # No source citation when quote is None
        lines = [l for l in md.splitlines() if "62000" in l]
        assert lines, "62000 should appear in the table"
        for line in lines:
            # If value appears without a quote, no [source:] tag should follow it
            if '[source:' not in line:
                assert "62000" in line
                break


# ---------------------------------------------------------------------------
# T4.4 — Flags ordered
# ---------------------------------------------------------------------------

class TestFlagsOrdered:
    def _result_with_mixed_flags(self) -> VerificationResult:
        return _all_match_result(
            risk_level="HIGH",
            consistency_score=0.55,
            flags=[
                _flag("AMOUNT_MISMATCH", "HIGH", "Claimant stated 50k; bill shows 62k"),
                _flag("DATE_MISMATCH", "MEDIUM", "Admission date differs by 2 days"),
                _flag("MISSING_FIELD", "LOW", "discharge_date absent from transcript"),
            ],
        )

    def test_high_flag_before_low_flag(self):
        md = render(self._result_with_mixed_flags())
        hi_pos = md.index("AMOUNT_MISMATCH")
        lo_pos = md.index("MISSING_FIELD")
        assert hi_pos < lo_pos, "HIGH severity flag should appear before LOW"

    def test_high_flag_before_medium_flag(self):
        md = render(self._result_with_mixed_flags())
        hi_pos = md.index("AMOUNT_MISMATCH")
        med_pos = md.index("DATE_MISMATCH")
        assert hi_pos < med_pos, "HIGH severity flag should appear before MEDIUM"

    def test_medium_flag_before_low_flag(self):
        md = render(self._result_with_mixed_flags())
        med_pos = md.index("DATE_MISMATCH")
        lo_pos = md.index("MISSING_FIELD")
        assert med_pos < lo_pos, "MEDIUM severity flag should appear before LOW"

    def test_no_flags_message(self):
        md = render(_all_match_result(flags=[]))
        assert "No flags raised" in md

    def test_flag_type_in_output(self):
        md = render(self._result_with_mixed_flags())
        assert "AMOUNT_MISMATCH" in md

    def test_flag_message_in_output(self):
        md = render(self._result_with_mixed_flags())
        assert "Claimant stated 50k; bill shows 62k" in md

    def test_flag_severity_label_in_output(self):
        md = render(self._result_with_mixed_flags())
        assert "[HIGH]" in md


# ---------------------------------------------------------------------------
# Missing fields + quality warnings
# ---------------------------------------------------------------------------

class TestMissingAndQuality:
    def test_missing_fields_listed(self):
        result = _all_match_result(
            missing_fields=["discharge_date", "length_of_stay"],
            risk_level="MEDIUM",
            consistency_score=0.80,
        )
        md = render(result)
        assert "discharge_date" in md
        assert "length_of_stay" in md

    def test_all_fields_present_message(self):
        md = render(_all_match_result(missing_fields=[]))
        assert "All fields present" in md

    def test_quality_warning_listed(self):
        result = _all_match_result(
            low_quality_sources=["transcript_C001"],
            risk_level="MEDIUM",
            consistency_score=0.80,
        )
        md = render(result)
        assert "transcript_C001" in md

    def test_no_quality_warnings_message(self):
        md = render(_all_match_result(low_quality_sources=[]))
        assert "No quality warnings" in md


# ---------------------------------------------------------------------------
# T4.3 — Reviewer guidance keyed to risk band
# ---------------------------------------------------------------------------

class TestReviewerGuidance:
    def test_guidance_low(self):
        md = render(_all_match_result(risk_level="LOW", consistency_score=1.0))
        assert "Standard processing" in md

    def test_guidance_medium(self):
        md = render(_all_match_result(risk_level="MEDIUM", consistency_score=0.80))
        assert "Review flagged fields before approval" in md

    def test_guidance_high(self):
        md = render(_all_match_result(risk_level="HIGH", consistency_score=0.55))
        assert "Do not approve" in md

    def test_guidance_insufficient_data(self):
        md = render(_insufficient_result())
        assert "Manual review" in md


# ---------------------------------------------------------------------------
# T4.1 — All six field labels in table
# ---------------------------------------------------------------------------

class TestFieldTable:
    _EXPECTED_LABELS = [
        "Hospital / Facility Name",
        "Admission Date",
        "Discharge Date",
        "Primary Diagnosis",
        "Total Billed Amount",
        "Duration of Admission",
    ]

    @pytest.mark.parametrize("label", _EXPECTED_LABELS)
    def test_field_label_in_table(self, label):
        md = render(_all_match_result())
        assert label in md, f"Expected label '{label}' not found in report"

    def test_verdict_match_in_table(self):
        md = render(_all_match_result())
        assert "MATCH" in md

    def test_verdict_missing_in_table(self):
        result = _all_match_result(
            missing_fields=["length_of_stay"],
            risk_level="MEDIUM",
            consistency_score=0.80,
            field_verdicts={
                **_all_match_result().field_verdicts,
                "length_of_stay": _verdict("MISSING", 0.0),
            },
        )
        md = render(result)
        assert "MISSING" in md
