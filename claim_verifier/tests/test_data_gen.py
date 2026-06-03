"""T7.6 — Synthetic data generator tests."""
import json
import random
from datetime import date
from pathlib import Path

import pdfplumber
import pytest

from claim_verifier.data_gen.generate_bill_pdf import generate_bill_pdf
from claim_verifier.data_gen.generate_claims import (
    _HOSPITALS,
    _generate_one_case,
    _inject_amount_error,
    _inject_date_error,
    _inject_hospital_error,
    generate_dataset,
    round_trip_check,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_FACTS = {
    "claim_id": "TEST001",
    "hospital_name": "Apollo Hospitals",
    "hospital_address": "Jubilee Hills, Hyderabad - 500033",
    "admission_date": "12/03/2025",
    "discharge_date": "16/03/2025",
    "diagnosis": "Acute Appendicitis",
    "billed_amount": 62_000,
    "length_of_stay": 4,
}

_ROUND_TRIP_TEXT = (
    "i was admitted to Apollo Hospitals in Hyderabad on 12/03/2025 "
    "for acute appendicitis i stayed for 4 days and discharged on 16/03/2025 "
    "the total bill was rupees 62,000"
)


def _pdf_text(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "".join(page.extract_text() or "" for page in pdf.pages)


def _faker():
    from faker import Faker
    f = Faker("en_IN")
    f.seed_instance(99)
    return f


# ---------------------------------------------------------------------------
# T7.1 — generate_bill_pdf
# ---------------------------------------------------------------------------

class TestGenerateBillPDF:
    def test_creates_file(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert out.exists()

    def test_text_layer_not_empty(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert len(_pdf_text(out)) >= 100

    def test_text_contains_hospital_name(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "APOLLO HOSPITALS" in _pdf_text(out)

    def test_text_contains_claim_id(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "TEST001" in _pdf_text(out)

    def test_text_contains_admission_date(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "12/03/2025" in _pdf_text(out)

    def test_text_contains_discharge_date(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "16/03/2025" in _pdf_text(out)

    def test_text_contains_diagnosis(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "Acute Appendicitis" in _pdf_text(out)

    def test_text_contains_amount(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "62,000" in _pdf_text(out)

    def test_text_contains_los(self, tmp_path):
        out = tmp_path / "bill.pdf"
        generate_bill_pdf(_SAMPLE_FACTS, out)
        assert "4 days" in _pdf_text(out)


# ---------------------------------------------------------------------------
# T7.4 — round_trip_check
# ---------------------------------------------------------------------------

class TestRoundTripCheck:
    def test_passes_all_values_present(self):
        assert round_trip_check(
            _ROUND_TRIP_TEXT,
            "Apollo Hospitals", "12/03/2025", "16/03/2025",
            "Acute Appendicitis", "62,000", 4,
        )

    def test_fails_hospital_missing(self):
        assert not round_trip_check(
            _ROUND_TRIP_TEXT,
            "Fortis Healthcare", "12/03/2025", "16/03/2025",
            "Acute Appendicitis", "62,000", 4,
        )

    def test_fails_admission_date_missing(self):
        assert not round_trip_check(
            _ROUND_TRIP_TEXT,
            "Apollo Hospitals", "15/03/2025", "16/03/2025",
            "Acute Appendicitis", "62,000", 4,
        )

    def test_fails_discharge_date_missing(self):
        assert not round_trip_check(
            _ROUND_TRIP_TEXT,
            "Apollo Hospitals", "12/03/2025", "20/03/2025",
            "Acute Appendicitis", "62,000", 4,
        )

    def test_fails_diagnosis_missing(self):
        assert not round_trip_check(
            _ROUND_TRIP_TEXT,
            "Apollo Hospitals", "12/03/2025", "16/03/2025",
            "Dengue Fever", "62,000", 4,
        )

    def test_fails_amount_missing(self):
        assert not round_trip_check(
            _ROUND_TRIP_TEXT,
            "Apollo Hospitals", "12/03/2025", "16/03/2025",
            "Acute Appendicitis", "50,000", 4,
        )


# ---------------------------------------------------------------------------
# T7.3 — error injectors
# ---------------------------------------------------------------------------

class TestErrorInjection:
    def test_amount_error_changes_amount(self):
        rng = random.Random(1)
        new_amount, _ = _inject_amount_error(62_000, rng)
        assert new_amount != 62_000

    def test_amount_error_exceeds_tolerance(self):
        for seed in range(20):
            rng = random.Random(seed)
            true_amount = 62_000
            new_amount, _ = _inject_amount_error(true_amount, rng)
            assert abs(new_amount - true_amount) / true_amount > 0.05

    def test_amount_error_delta_keys(self):
        rng = random.Random(1)
        _, delta = _inject_amount_error(62_000, rng)
        assert {"field", "true_value", "transcript_value", "pct_deviation"} <= delta.keys()

    def test_date_error_changes_admission(self):
        rng = random.Random(2)
        true_date = date(2025, 3, 12)
        new_date, _ = _inject_date_error(true_date, rng)
        assert new_date != true_date

    def test_date_error_offset_in_range(self):
        for seed in range(20):
            rng = random.Random(seed)
            _, delta = _inject_date_error(date(2025, 3, 12), rng)
            assert 3 <= abs(delta["offset_days"]) <= 15

    def test_date_error_delta_keys(self):
        rng = random.Random(2)
        _, delta = _inject_date_error(date(2025, 3, 12), rng)
        assert {"field", "true_value", "transcript_value", "offset_days"} <= delta.keys()

    def test_hospital_error_picks_different_hospital(self):
        rng = random.Random(3)
        true_hospital = _HOSPITALS[0]
        new_hospital, _ = _inject_hospital_error(true_hospital, rng)
        assert new_hospital["name"] != true_hospital["name"]

    def test_hospital_error_result_is_from_pool(self):
        rng = random.Random(3)
        true_hospital = _HOSPITALS[0]
        new_hospital, _ = _inject_hospital_error(true_hospital, rng)
        assert new_hospital in _HOSPITALS

    def test_clean_case_empty_delta_and_all_match(self):
        rng = random.Random(42)
        case = _generate_one_case("X001", "CLEAN", rng, _faker())
        assert case["injected_delta"] == {}
        assert all(v == "MATCH" for v in case["expected_verdicts"].values())


# ---------------------------------------------------------------------------
# T7.5–T7.6 — dataset generation
# ---------------------------------------------------------------------------

class TestDataset:
    def test_total_count(self, tmp_path):
        records = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        assert len(records) == 5

    def test_ground_truth_file_created(self, tmp_path):
        generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        assert (tmp_path / "ground_truth.jsonl").exists()

    def test_ground_truth_parseable(self, tmp_path):
        generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        lines = (tmp_path / "ground_truth.jsonl").read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "case_id" in parsed

    def test_all_bill_pdfs_created(self, tmp_path):
        records = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        for r in records:
            assert (tmp_path / r["bill_pdf"]).exists()

    def test_all_transcripts_created(self, tmp_path):
        records = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        for r in records:
            assert (tmp_path / r["transcript_txt"]).exists()

    def test_case_ids_unique(self, tmp_path):
        records = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        ids = [r["case_id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_records_have_required_keys(self, tmp_path):
        records = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path, seed=42)
        required = {
            "case_id", "error_type", "true_facts",
            "transcript_facts", "expected_verdicts", "injected_delta",
        }
        for r in records:
            assert required <= r.keys()

    def test_clean_count(self, tmp_path):
        records = generate_dataset(n_total=6, n_clean=2, output_dir=tmp_path, seed=42)
        assert sum(1 for r in records if r["error_type"] == "CLEAN") == 2

    def test_error_distribution_correct(self, tmp_path):
        # n_total=6, n_clean=2 → n_error=4 → AMOUNT=2, DATE=1, HOSPITAL=1
        records = generate_dataset(n_total=6, n_clean=2, output_dir=tmp_path, seed=42)
        counts: dict[str, int] = {}
        for r in records:
            counts[r["error_type"]] = counts.get(r["error_type"], 0) + 1
        assert counts.get("CLEAN", 0) == 2
        assert counts.get("AMOUNT", 0) == 2
        assert counts.get("DATE", 0) + counts.get("HOSPITAL", 0) == 2

    def test_reproducible_same_seed(self, tmp_path):
        r1 = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path / "a", seed=7)
        r2 = generate_dataset(n_total=5, n_clean=2, output_dir=tmp_path / "b", seed=7)
        assert [r["error_type"] for r in r1] == [r["error_type"] for r in r2]
        assert (
            [r["true_facts"]["billed_amount"] for r in r1]
            == [r["true_facts"]["billed_amount"] for r in r2]
        )
