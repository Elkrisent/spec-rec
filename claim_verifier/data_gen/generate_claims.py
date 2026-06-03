"""Synthetic claim dataset generator (W7 — T7.2–T7.5).

No LLM — templates + Faker only.  Eliminates generator/extractor leakage.

Run directly to emit the full 50-case dataset:
    python -m claim_verifier.data_gen.generate_claims
"""
from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from faker import Faker

from claim_verifier.data_gen.generate_bill_pdf import generate_bill_pdf

ErrorType = Literal["CLEAN", "AMOUNT", "DATE", "HOSPITAL"]

_DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data_gen" / "output"

_HOSPITALS: list[dict] = [
    {"name": "Apollo Hospitals",             "city": "Hyderabad",   "address": "Jubilee Hills, Hyderabad - 500033"},
    {"name": "Fortis Healthcare",            "city": "Bengaluru",   "address": "Bannerghatta Road, Bengaluru - 560076"},
    {"name": "Max Super Speciality Hospital","city": "New Delhi",   "address": "Saket, New Delhi - 110017"},
    {"name": "KIMS Hospital",               "city": "Secunderabad","address": "Minister Road, Secunderabad - 500003"},
    {"name": "Manipal Hospital",            "city": "Bengaluru",   "address": "Old Airport Road, Bengaluru - 560017"},
    {"name": "Aster CMI Hospital",          "city": "Bengaluru",   "address": "Hebbal, Bengaluru - 560092"},
    {"name": "Medanta Hospital",            "city": "Gurugram",    "address": "Sector 38, Gurugram - 122001"},
    {"name": "Lilavati Hospital",           "city": "Mumbai",      "address": "Bandra West, Mumbai - 400050"},
    {"name": "Narayana Health",             "city": "Bengaluru",   "address": "Bommasandra, Bengaluru - 560099"},
    {"name": "Kokilaben Hospital",          "city": "Mumbai",      "address": "Andheri West, Mumbai - 400053"},
]

_DIAGNOSES: list[str] = [
    "Acute Appendicitis",
    "Dengue Fever",
    "Typhoid Fever",
    "Viral Pneumonia",
    "Acute Gastroenteritis",
    "Urinary Tract Infection",
    "Hypertensive Emergency",
    "Diabetic Ketoacidosis",
    "Acute Myocardial Infarction",
    "Community-Acquired Pneumonia",
    "Acute Pancreatitis",
    "Acute Cholecystitis",
    "Peptic Ulcer Disease",
    "Deep Vein Thrombosis",
    "Cellulitis",
]

# Placeholders: caller_name, hospital_name, hospital_city,
# admission_date, discharge_date, diagnosis, billed_amount_str,
# length_of_stay, claim_id.
_TEMPLATES: list[str] = [
    (
        "yes hello i am {caller_name} calling to register a claim for a recent hospitalisation "
        "i was admitted to {hospital_name} in {hospital_city} on {admission_date} "
        "for {diagnosis} i was hospitalised for {length_of_stay} days "
        "and discharged on {discharge_date} "
        "the total bill came to rupees {billed_amount_str} "
        "my claim reference is {claim_id}"
    ),
    (
        "hi yes this is {caller_name} regarding a medical claim "
        "i was admitted to {hospital_name} {hospital_city} on {admission_date} "
        "i was diagnosed with {diagnosis} and stayed for {length_of_stay} days "
        "i was discharged on {discharge_date} "
        "the total amount charged was {billed_amount_str} rupees "
        "claim id is {claim_id}"
    ),
    (
        "hello i am {caller_name} i need to file a health insurance claim "
        "the hospitalisation was at {hospital_name} in {hospital_city} "
        "admission date was {admission_date} "
        "the reason for admission was {diagnosis} "
        "i stayed for {length_of_stay} days "
        "discharge date was {discharge_date} "
        "total hospital bill rupees {billed_amount_str} "
        "claim number {claim_id}"
    ),
    (
        "good afternoon this is {caller_name} i am calling about a recent hospital stay "
        "i checked into {hospital_name} {hospital_city} on {admission_date} "
        "i had {diagnosis} and was admitted for {length_of_stay} days "
        "i was discharged on {discharge_date} "
        "total billing amount rupees {billed_amount_str} "
        "this is for claim {claim_id}"
    ),
    (
        "hello my name is {caller_name} i want to initiate a claim for my hospitalisation "
        "i was at {hospital_name} {hospital_city} "
        "from {admission_date} for {length_of_stay} days "
        "my diagnosis was {diagnosis} "
        "i was discharged on {discharge_date} "
        "total bill amount rupees {billed_amount_str} "
        "claim id {claim_id}"
    ),
    (
        "hi yes this is {caller_name} hospitalisation claim "
        "admitted at {hospital_name} {hospital_city} "
        "date of admission {admission_date} "
        "primary diagnosis {diagnosis} "
        "length of stay {length_of_stay} days "
        "date of discharge {discharge_date} "
        "billed amount rupees {billed_amount_str} "
        "claim reference {claim_id}"
    ),
]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_date_transcript(d: date, style: int) -> str:
    """Return a parseable date string in one of three spoken styles."""
    styles = [
        d.strftime("%d/%m/%Y"),    # 12/03/2025
        d.strftime("%d %B %Y"),    # 12 March 2025
        d.strftime("%B %d, %Y"),   # March 12, 2025
    ]
    return styles[style % 3]


def _format_date_bill(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _amount_str(amount: int) -> str:
    return f"{amount:,}"


# ---------------------------------------------------------------------------
# Error injectors
# ---------------------------------------------------------------------------

def _inject_amount_error(true_amount: int, rng: random.Random) -> tuple[int, dict]:
    """Deviate amount by 10–40%, rounded to nearest 500."""
    pct = rng.uniform(0.10, 0.40)
    direction = rng.choice([-1, 1])
    raw = true_amount + direction * int(true_amount * pct)
    transcript_amount = max(5_000, round(raw / 500) * 500)
    if transcript_amount == true_amount:
        transcript_amount = true_amount + direction * 1_000
    return transcript_amount, {
        "field": "billed_amount",
        "true_value": true_amount,
        "transcript_value": transcript_amount,
        "pct_deviation": round(abs(transcript_amount - true_amount) / true_amount * 100, 1),
    }


def _inject_date_error(true_date: date, rng: random.Random) -> tuple[date, dict]:
    """Offset admission date by 3–15 days."""
    offset_days = rng.randint(3, 15)
    direction = rng.choice([-1, 1])
    transcript_date = true_date + timedelta(days=direction * offset_days)
    return transcript_date, {
        "field": "admission_date",
        "true_value": str(true_date),
        "transcript_value": str(transcript_date),
        "offset_days": direction * offset_days,
    }


def _inject_hospital_error(true_hospital: dict, rng: random.Random) -> tuple[dict, dict]:
    """Pick a different hospital (guaranteed fuzzy MISMATCH)."""
    other = [h for h in _HOSPITALS if h["name"] != true_hospital["name"]]
    transcript_hospital = rng.choice(other)
    return transcript_hospital, {
        "field": "hospital_name",
        "true_value": true_hospital["name"],
        "transcript_value": transcript_hospital["name"],
    }


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def round_trip_check(
    text: str,
    hospital_name: str,
    admission_date_str: str,
    discharge_date_str: str,
    diagnosis: str,
    amount_str: str,
    length_of_stay: int,
) -> bool:
    """Return True iff all injected values appear verbatim in *text*."""
    text_lower = text.lower()
    return all([
        hospital_name.lower() in text_lower,
        diagnosis.lower() in text_lower,
        amount_str in text,
        admission_date_str in text,
        discharge_date_str in text,
        str(length_of_stay) in text,
    ])


# ---------------------------------------------------------------------------
# Case generation
# ---------------------------------------------------------------------------

def _generate_one_case(
    case_id: str,
    error_type: ErrorType,
    rng: random.Random,
    faker: Faker,
) -> dict:
    """Return a ground-truth record dict (includes _transcript_text and _date_style)."""
    hospital = rng.choice(_HOSPITALS)
    diagnosis = rng.choice(_DIAGNOSES)
    year = rng.choice([2024, 2025])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    true_admission = date(year, month, day)
    true_los = rng.randint(2, 10)
    true_discharge = true_admission + timedelta(days=true_los)
    true_amount = rng.randrange(15_000, 200_001, 1_000)

    t_hospital = hospital
    t_admission = true_admission
    t_discharge = true_discharge
    t_amount = true_amount
    injected_delta: dict = {}
    expected_verdicts: dict[str, str] = {
        "hospital_name": "MATCH",
        "admission_date": "MATCH",
        "discharge_date": "MATCH",
        "diagnosis": "MATCH",
        "billed_amount": "MATCH",
        "length_of_stay": "MATCH",
    }

    if error_type == "AMOUNT":
        t_amount, injected_delta = _inject_amount_error(true_amount, rng)
        expected_verdicts["billed_amount"] = "MISMATCH"
    elif error_type == "DATE":
        t_admission, injected_delta = _inject_date_error(true_admission, rng)
        t_discharge = t_admission + timedelta(days=true_los)
        expected_verdicts["admission_date"] = "MISMATCH"
        expected_verdicts["discharge_date"] = "MISMATCH"
    elif error_type == "HOSPITAL":
        t_hospital, injected_delta = _inject_hospital_error(hospital, rng)
        expected_verdicts["hospital_name"] = "MISMATCH"

    template_idx = rng.randint(0, len(_TEMPLATES) - 1)
    date_style = rng.randint(0, 2)
    caller_name = faker.first_name()

    adm_str = _format_date_transcript(t_admission, date_style)
    dis_str = _format_date_transcript(t_discharge, date_style)
    amount_s = _amount_str(t_amount)

    transcript_text = _TEMPLATES[template_idx].format(
        caller_name=caller_name,
        hospital_name=t_hospital["name"],
        hospital_city=t_hospital["city"],
        admission_date=adm_str,
        discharge_date=dis_str,
        diagnosis=diagnosis.lower(),
        billed_amount_str=amount_s,
        length_of_stay=true_los,
        claim_id=case_id,
    )

    if not round_trip_check(transcript_text, t_hospital["name"], adm_str, dis_str, diagnosis, amount_s, true_los):
        raise RuntimeError(f"Round-trip check failed for {case_id}")

    return {
        "case_id": case_id,
        "error_type": error_type,
        "true_facts": {
            "hospital_name": hospital["name"],
            "hospital_address": hospital["address"],
            "admission_date": str(true_admission),
            "discharge_date": str(true_discharge),
            "diagnosis": diagnosis,
            "billed_amount": true_amount,
            "length_of_stay": true_los,
        },
        "transcript_facts": {
            "hospital_name": t_hospital["name"],
            "hospital_city": t_hospital["city"],
            "admission_date": str(t_admission),
            "discharge_date": str(t_discharge),
            "diagnosis": diagnosis,
            "billed_amount": t_amount,
            "length_of_stay": true_los,
        },
        "expected_verdicts": expected_verdicts,
        "injected_delta": injected_delta,
        "_transcript_text": transcript_text,
        "_date_style": date_style,
    }


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_total: int = 50,
    n_clean: int = 15,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic dataset. Writes transcripts, PDFs, and ground_truth.jsonl.

    Returns list of ground-truth records.
    """
    rng = random.Random(seed)
    faker = Faker("en_IN")
    faker.seed_instance(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_error = n_total - n_clean
    error_schedule: list[ErrorType] = ["CLEAN"] * n_clean
    error_types: list[ErrorType] = ["AMOUNT", "DATE", "HOSPITAL"]
    counts = [n_error // 3 + (1 if i < n_error % 3 else 0) for i in range(3)]
    for etype, count in zip(error_types, counts):
        error_schedule.extend([etype] * count)
    rng.shuffle(error_schedule)

    records = []
    for i, error_type in enumerate(error_schedule, start=1):
        case_id = f"SYN{i:03d}"
        case = _generate_one_case(case_id, error_type, rng, faker)
        transcript_text = case.pop("_transcript_text")
        case.pop("_date_style")

        transcript_name = f"{case_id}_transcript.txt"
        (output_dir / transcript_name).write_text(transcript_text, encoding="utf-8")

        bill_name = f"{case_id}_bill.pdf"
        tf = case["true_facts"]
        bill_facts = {
            "claim_id": case_id,
            "hospital_name": tf["hospital_name"],
            "hospital_address": tf["hospital_address"],
            "admission_date": _format_date_bill(date.fromisoformat(tf["admission_date"])),
            "discharge_date": _format_date_bill(date.fromisoformat(tf["discharge_date"])),
            "diagnosis": tf["diagnosis"],
            "billed_amount": tf["billed_amount"],
            "length_of_stay": tf["length_of_stay"],
        }
        generate_bill_pdf(bill_facts, output_dir / bill_name)

        case["bill_pdf"] = bill_name
        case["transcript_txt"] = transcript_name
        records.append(case)

    gt_path = output_dir / "ground_truth.jsonl"
    with gt_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return records


if __name__ == "__main__":
    records = generate_dataset()
    counts_out: dict[str, int] = {}
    for r in records:
        counts_out[r["error_type"]] = counts_out.get(r["error_type"], 0) + 1
    print(f"Generated {len(records)} cases → {_DEFAULT_OUTPUT_DIR}")
    for etype in sorted(counts_out):
        print(f"  {etype}: {counts_out[etype]}")
