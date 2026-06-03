"""T8.6 — test_metrics.py: verify evaluation metric math."""
import pytest

from claim_verifier.eval.metrics import (
    _LEAKAGE_CAVEAT,
    _TARGETS,
    compute_eval_summary,
    compute_field_metrics,
    compute_fp_rate_on_clean,
    format_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(error_type: str, verdicts: dict[str, str]) -> dict:
    return {"error_type": error_type, "expected_verdicts": verdicts}


def _all_match() -> dict[str, str]:
    return {
        "hospital_name": "MATCH",
        "admission_date": "MATCH",
        "discharge_date": "MATCH",
        "diagnosis": "MATCH",
        "billed_amount": "MATCH",
        "length_of_stay": "MATCH",
    }


def _amount_mismatch() -> dict[str, str]:
    d = _all_match()
    d["billed_amount"] = "MISMATCH"
    return d


# ---------------------------------------------------------------------------
# TestFieldMetrics
# ---------------------------------------------------------------------------

class TestFieldMetrics:
    def test_perfect_predictions(self):
        preds  = ["MATCH", "MISMATCH", "MATCH", "MISMATCH"]
        truths = ["MATCH", "MISMATCH", "MATCH", "MISMATCH"]
        fm = compute_field_metrics("billed_amount", preds, truths)
        assert fm.tp == 2
        assert fm.fp == 0
        assert fm.fn == 0
        assert fm.tn == 2
        assert fm.precision == 1.0
        assert fm.recall == 1.0
        assert fm.f1 == 1.0

    def test_all_false_negatives(self):
        # Always predict MATCH; truth has MISMATCH cases → all missed
        preds  = ["MATCH", "MATCH", "MATCH"]
        truths = ["MISMATCH", "MISMATCH", "MATCH"]
        fm = compute_field_metrics("billed_amount", preds, truths)
        assert fm.tp == 0
        assert fm.fn == 2
        assert fm.tn == 1
        assert fm.recall == 0.0
        assert fm.precision is None  # no positive predictions

    def test_all_false_positives(self):
        # Always predict MISMATCH; truth is all MATCH → all FP
        preds  = ["MISMATCH", "MISMATCH", "MISMATCH"]
        truths = ["MATCH", "MATCH", "MATCH"]
        fm = compute_field_metrics("billed_amount", preds, truths)
        assert fm.tp == 0
        assert fm.fp == 3
        assert fm.fn == 0
        assert fm.tn == 0
        assert fm.precision == 0.0
        assert fm.recall is None  # no positive ground truth

    def test_precision_recall_values(self):
        # tp=2, fp=1, fn=1 → precision=2/3, recall=2/3
        preds  = ["MISMATCH", "MISMATCH", "MISMATCH", "MATCH"]
        truths = ["MISMATCH", "MISMATCH", "MATCH",    "MISMATCH"]
        fm = compute_field_metrics("billed_amount", preds, truths)
        assert fm.tp == 2
        assert fm.fp == 1
        assert fm.fn == 1
        assert abs(fm.precision - 2 / 3) < 1e-9
        assert abs(fm.recall - 2 / 3) < 1e-9

    def test_missing_counts_as_fn_when_truth_mismatch(self):
        preds  = ["MISSING", "MISMATCH"]
        truths = ["MISMATCH", "MISMATCH"]
        fm = compute_field_metrics("admission_date", preds, truths)
        assert fm.fn == 1
        assert fm.tp == 1
        assert fm.n_missing == 1

    def test_missing_counts_as_tn_when_truth_match(self):
        preds  = ["MISSING", "MATCH"]
        truths = ["MATCH",   "MATCH"]
        fm = compute_field_metrics("hospital_name", preds, truths)
        assert fm.tn == 2
        assert fm.fp == 0
        assert fm.fn == 0

    def test_precision_none_when_no_positive_predictions(self):
        preds  = ["MATCH", "MATCH"]
        truths = ["MATCH", "MISMATCH"]
        fm = compute_field_metrics("billed_amount", preds, truths)
        assert fm.precision is None

    def test_recall_none_when_no_positive_truth(self):
        preds  = ["MISMATCH", "MATCH"]
        truths = ["MATCH",    "MATCH"]
        fm = compute_field_metrics("billed_amount", preds, truths)
        assert fm.recall is None

    def test_n_missing_tracked_correctly(self):
        preds  = ["MISSING", "MISSING", "MISMATCH"]
        truths = ["MISMATCH", "MATCH",  "MISMATCH"]
        fm = compute_field_metrics("admission_date", preds, truths)
        assert fm.n_missing == 1  # only MISSING vs MISMATCH counts


# ---------------------------------------------------------------------------
# TestFPRate
# ---------------------------------------------------------------------------

class TestFPRate:
    def test_zero_fp_rate_all_correct_clean(self):
        records = [
            _record("CLEAN", _all_match()),
            _record("CLEAN", _all_match()),
        ]
        preds = [_all_match(), _all_match()]
        assert compute_fp_rate_on_clean(preds, records) == 0.0

    def test_full_fp_rate_all_clean_have_mismatch(self):
        records = [
            _record("CLEAN", _all_match()),
            _record("CLEAN", _all_match()),
        ]
        preds = [_amount_mismatch(), _amount_mismatch()]
        assert compute_fp_rate_on_clean(preds, records) == 1.0

    def test_partial_fp_rate(self):
        records = [
            _record("CLEAN", _all_match()),
            _record("CLEAN", _all_match()),
            _record("CLEAN", _all_match()),
            _record("CLEAN", _all_match()),
        ]
        preds = [_amount_mismatch(), _all_match(), _amount_mismatch(), _all_match()]
        rate = compute_fp_rate_on_clean(preds, records)
        assert abs(rate - 0.5) < 1e-9

    def test_zero_fp_rate_no_clean_cases(self):
        records = [
            _record("AMOUNT", _amount_mismatch()),
            _record("DATE",   _all_match()),
        ]
        preds = [_amount_mismatch(), _all_match()]
        assert compute_fp_rate_on_clean(preds, records) == 0.0

    def test_error_cases_excluded_from_fp_rate(self):
        records = [
            _record("CLEAN",  _all_match()),
            _record("AMOUNT", _amount_mismatch()),
        ]
        # Error case has a MISMATCH prediction but that's expected — not a FP
        preds = [_all_match(), _amount_mismatch()]
        assert compute_fp_rate_on_clean(preds, records) == 0.0


# ---------------------------------------------------------------------------
# TestEvalSummary
# ---------------------------------------------------------------------------

class TestEvalSummary:
    def _make_simple(self, n_clean=2, n_error=2):
        records = [_record("CLEAN", _all_match()) for _ in range(n_clean)]
        records += [_record("AMOUNT", _amount_mismatch()) for _ in range(n_error)]
        # Pipeline perfect on error, no FP on clean
        preds_clean = [_all_match() for _ in range(n_clean)]
        preds_error = [_amount_mismatch() for _ in range(n_error)]
        return records, preds_clean + preds_error

    def test_n_cases(self):
        records, preds = self._make_simple(2, 2)
        s = compute_eval_summary(records, preds)
        assert s.n_cases == 4

    def test_n_clean_n_error(self):
        records, preds = self._make_simple(3, 5)
        s = compute_eval_summary(records, preds)
        assert s.n_clean == 3
        assert s.n_error == 5

    def test_all_fields_in_metrics(self):
        records, preds = self._make_simple()
        s = compute_eval_summary(records, preds)
        expected = {
            "hospital_name", "admission_date", "discharge_date",
            "diagnosis", "billed_amount", "length_of_stay",
        }
        assert set(s.field_metrics.keys()) == expected

    def test_fp_rate_zero_when_no_fp(self):
        records, preds = self._make_simple()
        s = compute_eval_summary(records, preds)
        assert s.fp_rate_clean == 0.0


# ---------------------------------------------------------------------------
# TestFormatReport
# ---------------------------------------------------------------------------

class TestFormatReport:
    def _simple_summary(self):
        records = [
            _record("CLEAN", _all_match()),
            _record("AMOUNT", _amount_mismatch()),
        ]
        preds = [_all_match(), _amount_mismatch()]
        return compute_eval_summary(records, preds)

    def test_report_contains_leakage_caveat(self):
        s = self._simple_summary()
        report = format_report(s)
        assert "LEAKAGE CAVEAT" in report

    def test_report_contains_target_check(self):
        s = self._simple_summary()
        report = format_report(s)
        assert "precision" in report.lower()
        assert "FP rate" in report or "fp_rate" in report.lower()

    def test_report_contains_case_counts(self):
        s = self._simple_summary()
        report = format_report(s)
        assert "2" in report  # n_cases = 2

    def test_report_holdout_section_when_provided(self):
        s = self._simple_summary()
        report = format_report(s, holdout_summary=s)
        assert "Holdout" in report

    def test_report_no_holdout_section_when_omitted(self):
        s = self._simple_summary()
        report = format_report(s, holdout_summary=None)
        # "Holdout (" is the section header pattern; leakage caveat also contains "Holdout"
        assert "Holdout (" not in report
