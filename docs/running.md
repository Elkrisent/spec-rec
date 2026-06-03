# Running the Medical Claim Verification Assistant

This document is updated at the end of each completed week. It describes the current
capabilities of the system and how to run it. Anything marked **Planned** is not yet
implemented.

---

## Current state — Week 8 (evaluation harness)

The pipeline can accept a plain-text transcript and a hospital bill PDF, extract
structured facts from both using a local LLM, verify consistency field-by-field,
and produce a cited Markdown report with a risk score. Everything runs locally —
no data leaves the machine. A synthetic 50-case labeled dataset is now available
for evaluation (W8).

### What it can do

| Capability | Status |
|---|---|
| Ingest plain-text transcript (.txt) | ✓ Done |
| Ingest hospital bill PDF (text-layer only) | ✓ Done |
| Reject scanned / image-only PDFs | ✓ Done |
| Reject non-English input | ✓ Done |
| PII redaction before LLM call (phone / Aadhaar / PAN / email) | ✓ Done |
| Extract 6 fields via local LLM (JSON-schema-constrained) | ✓ Done |
| One repair retry on malformed LLM response | ✓ Done |
| Normalize dates / amounts / hospital names | ✓ Done |
| Verify each field (fuzzy / date / numeric / LLM diagnosis judge) | ✓ Done |
| Risk score (LOW ≥ 0.90 / MEDIUM 0.70–0.89 / HIGH < 0.70) | ✓ Done |
| Markdown report with verbatim citations and reviewer guidance | ✓ Done |
| Graceful partial report on stage failure | ✓ Done |
| LLM response cache (offline/CI safe) | ✓ Done |
| Synthetic 50-case labeled dataset (data_gen/) | ✓ Done |
| Evaluation harness (precision/recall per field) | ✓ Done |
| 10-case hand-built holdout set | ✓ Done |
| Audio input (--audio) | **Planned — W9** |
| Scanned PDF via OCR | **Planned — deferred** |
| Web UI (FastAPI + HTML frontend) | **Planned — W10** |
| REST API | **Planned — W10** |

### Known limitations (as of W6)

**1. Photo / scanned bill PDFs are rejected.**

A bill photographed on a phone, or any image-scanned PDF, contains no machine-readable
text layer. pdfplumber extracts zero characters; the system raises an `IngestionError`
with a clear message. Only digitally-generated PDFs (produced by hospital billing
software) are accepted. This is a deliberate architectural decision — OCR on dense
Indian hospital bill tables is unreliable, especially for tabular charge breakdowns.
Production-grade support would require Google Vision or AWS Textract, which are
deferred (see `architecture.md` §2).

**2. Vague transcript language inflates the risk score.**

The normalization stage expects structured values (ISO dates, numeric amounts). When a
claimant uses spoken-language forms — "twelfth of march this year", "around fifty
thousand rupees maybe" — normalization fails and those fields are marked MISSING,
excluded from the denominator, and the score rises. In the extreme case, a claimant who
is deliberately vague on all financial and date details can produce a LOW RISK / 100%
score even when the bill shows completely different figures.

This is a known scoring model limitation. The MISSING field count and UNPARSEABLE flags
are visible in the report, so a human reviewer can see how many fields were actually
compared. Mitigations are tracked for a later phase (capped denominator, minimum
verified-fields threshold before LOW RISK is awarded).

**3. LLM extraction quality varies with hardware.**

On CPU with the 7B model, inference can exceed 120 s per call. Use `OLLAMA_MODEL_FAST`
(3B model) if the 7B is too slow. The 3B model has slightly lower precision but
performs adequately on short, focused medical texts. The timeout in
`backends/ollama.py` is set to 300 s.

### Fields extracted and verified

| Field | Match type | Tolerance |
|---|---|---|
| Hospital / Facility | Fuzzy string (rapidfuzz) | ratio ≥ 0.85 |
| Admission Date | Date | ±1 day; month/day if year absent |
| Discharge Date | Date | ±1 day; optional |
| Primary Diagnosis | LLM equivalence judge | disease ≠ procedure |
| Total Billed Amount | Numeric | ±5% |
| Duration of Admission | Numeric | ±1 day |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| [Ollama](https://ollama.com) | Local LLM inference — free, CPU-capable |
| qwen2.5:7b-instruct-q4_K_M | ~4.7 GB RAM; quality model |
| qwen2.5:3b-instruct | ~2.5 GB RAM; fallback if 7B is >60 s/call |

### Pull models (one-time)

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct
```

---

## Installation

```bash
git clone <repo>
cd spec-rec

# create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install uv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

---

## Running the demo

The `demo/` directory contains a canned scenario:
- `transcript.txt` — a phone call where the claimant reports ≈ ₹50,000
- `bill.pdf` — hospital bill showing ₹62,000 (Apollo Hospitals, Jubilee Hills)

```bash
# start Ollama (if not already running)
ollama serve &

# run the pipeline
python -m claim_verifier.cli \
  --claim-id DEMO001 \
  --transcript demo/transcript.txt \
  --document  demo/bill.pdf \
  --out        demo/report.md

cat demo/report.md
```

**Expected output (excerpt):**

```markdown
# Claim Verification Report — DEMO001

**Risk Level:** LOW RISK
**Consistency Score:** 100.0%

| Field               | Verdict | Note                                       |
|---------------------|---------|--------------------------------------------|
| Hospital / Facility | MATCH   | fuzzy match, ratio=0.97                    |
| Admission Date      | MISSING | spoken date unparseable → excluded         |
| Discharge Date      | MISSING | absent from transcript                     |
| Primary Diagnosis   | MATCH   | LLM judge: same condition                  |
| Total Billed Amount | MISSING | spoken amount unparseable → excluded       |
| Duration            | MATCH   | 4 days (extracted from "four or five days")|

Flags: UNPARSEABLE_DATE (×2), UNPARSEABLE_AMOUNT (×1), MISSING_FIELD (×3)
```

> **Reading this result:** 100% / LOW RISK looks reassuring, but only 2 of 6
> fields were actually compared (hospital name and LOS). Three fields — admission
> date, billed amount, and discharge date — fell out of the denominator because the
> claimant used vague spoken language the normalizer could not parse. The
> UNPARSEABLE flags at the top of the Flags section are the signal that matters.
> A reviewer seeing 3 UNPARSEABLE flags should treat this as incomplete, not clean.
> See "Known limitations" above.

**Inference time:** 2–3 LLM calls × 15–60 s each on CPU with the 3B model.
The response cache means subsequent runs on the same inputs are instant.

---

## Running on your own files

```bash
python -m claim_verifier.cli \
  --claim-id  <YOUR_CLAIM_ID> \
  --transcript <path/to/transcript.txt> \
  --document  <path/to/bill.pdf> \
  [--out <path/to/report.md>]   # omit to print to stdout
```

**Transcript requirements:**
- Plain UTF-8 text file (`.txt`)
- English only
- Any length; longer transcripts give the LLM more context

**Document requirements:**
- PDF with a machine-readable text layer (digitally created, not scanned)
- English only
- Must yield ≥ 100 characters of extractable text

---

## Running the test suite

```bash
# all 264 tests — no Ollama required (W5 tests use pre-seeded cache)
python -m pytest

# single module
python -m pytest claim_verifier/tests/test_pipeline.py -v
```

Tests are grouped by week:

| File | Week | Tests | What it covers |
|---|---|---|---|
| `test_contracts.py` | W1 | 31 | Pydantic models, schema weights |
| `test_normalization.py` | W2 | 49 | Date/amount/hospital normalizers |
| `test_verification.py` | W3 | 40 | Matchers, scoring, flag ordering |
| `test_reporting.py` | W4 | 44 | Jinja2 template, citations, guidance |
| `test_extraction.py` | W5 | 47 | LLM extraction, retry, quote verification |
| `test_pipeline.py` | W6 | 53 | Ingestion, redaction, pipeline E2E |
| `test_data_gen.py` | W7 | 34 | Bill PDF, round-trip, error injection, dataset |
| `test_metrics.py`  | W8 | 23 | Precision/recall/FP-rate math, report format |

---

## Understanding the report

### Risk levels

| Band | Score | Meaning |
|---|---|---|
| LOW RISK | ≥ 0.90 | All verified fields match; minor missing fields acceptable |
| MEDIUM RISK | 0.70 – 0.89 | One or more discrepancies; review flagged fields before approval |
| HIGH RISK | < 0.70 | Significant mismatches; escalate to senior reviewer |
| INSUFFICIENT DATA | — | Too many fields absent to compute a score |

### Scoring formula

```
consistency_score = Σ weight(f) × field_score(f)  /  Σ weight(included fields)

field_score = 1.0 (MATCH)  |  0.0 (MISMATCH)
MISSING fields are flagged and excluded from the denominator
```

### Field weights

| Field | Weight | Rationale |
|---|---|---|
| hospital_name | 0.10 | Advisory; abbreviations common |
| admission_date | 0.20 | High-importance objective fact |
| discharge_date | 0.10 | Optional in transcripts |
| diagnosis | 0.25 | Core clinical fact |
| billed_amount | 0.25 | Primary fraud signal |
| length_of_stay | 0.10 | Derivable from dates |

### Flag severities

| Severity | Examples |
|---|---|
| HIGH | AMOUNT_MISMATCH, DATE_MISMATCH, EXTRACTION_FAILURE, INGESTION_FAILURE |
| MEDIUM | HOSPITAL_MISMATCH, DIAGNOSIS_MISMATCH, UNPARSEABLE_AMOUNT |
| LOW | MISSING_FIELD, LOW_CONFIDENCE_EXTRACTION |

### Citations

Each value in the field table is shown with its verbatim source quote:
```
apollo hospitals [source: "APOLLO HOSPITALS Jubilee Hills, Hyderabad"]
```
A missing citation means `quote_verified=False` — the LLM paraphrased rather than
copying verbatim. The value is still used; only the citation is flagged.

---

## Model selection

The config uses `qwen2.5:7b-instruct-q4_K_M` by default. To switch to the faster
3B model (useful on constrained hardware):

```python
# claim_verifier/config.py
OLLAMA_MODEL = OLLAMA_MODEL_FAST   # qwen2.5:3b-instruct
```

Accuracy tradeoff: the 3B model has lower extraction precision (~65–75% vs ~75–85%
for 7B on clean synthetic data). Quote fidelity degrades most.

---

## LLM response cache

All LLM responses are cached in `.llm_cache/` (SHA-256 keyed by model + messages +
schema). Once a transcript/document pair has been processed, re-running is instant
and requires no Ollama server. The cache is safe to commit for offline CI.

```bash
ls .llm_cache/   # each file is one cached response
```

---

## Generating the synthetic dataset

```bash
# Creates data_gen/output/ with 50 cases (35 error, 15 clean) + ground_truth.jsonl
python -m claim_verifier.data_gen.generate_claims
```

Output files per case: `SYNxxx_bill.pdf` (text-layer PDF) + `SYNxxx_transcript.txt`.
Ground truth includes `expected_verdicts` per field — used by the evaluation harness in W8.

Error types in the dataset:
- **AMOUNT** (12 cases) — billed amount in transcript differs 10–40% from bill
- **DATE** (12 cases) — admission date off by 3–15 days; discharge shifts accordingly
- **HOSPITAL** (11 cases) — transcript names a different hospital
- **CLEAN** (15 cases) — all fields match

---

## Running the evaluation

```bash
# Requires Ollama running. First run: ~120 LLM calls (~30 min–2 hrs on CPU).
# Subsequent runs are instant (LLM cache).
ollama serve &
python -m claim_verifier.eval.run_eval

# Save report to file
python -m claim_verifier.eval.run_eval --out eval/report.md
```

The report prints:
- Per-field precision / recall / F1 on the 50-case synthetic dataset
- Case-level FP rate on CLEAN cases
- Holdout results (10 hand-built messy cases)
- Target check: precision ≥ 80%, FP rate ≤ 15%
- Leakage caveat (synthetic is an upper bound)

---

## What's coming next

| Week | Feature |
|---|---|
| W9 | Audio input via faster-whisper (CPU, --audio flag) |
| W10 | Web UI (FastAPI + HTML frontend), REST API, tagged release |
