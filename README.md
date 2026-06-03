# Medical Claim Verification Assistant

Automated, privacy-preserving verification of medical insurance claims against
submitted hospital bills. Runs entirely on your machine — no data leaves the
device.

---

## What it does

Given a claimant's phone-call transcript and a hospital bill PDF, the system:

1. Extracts 6 structured fields from both sources using a local LLM
2. Normalizes dates, amounts, and hospital names
3. Compares each field (fuzzy / date / numeric / LLM diagnosis judge)
4. Produces a cited Markdown report with a risk score and reviewer guidance

**Audio input**: pass a WAV/MP3 file instead of a transcript — the call is
transcribed locally via faster-whisper before extraction.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| [Ollama](https://ollama.com) | Local LLM inference — free, CPU-capable |
| qwen2.5:7b-instruct-q4_K_M | ~4.7 GB RAM; quality extraction model |
| qwen2.5:3b-instruct | ~2.5 GB RAM; faster fallback |

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct
```

---

## Installation

```bash
git clone <repo>
cd spec-rec

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install uv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

---

## Quick demo

```bash
ollama serve &

python -m claim_verifier.cli verify \
  --claim-id DEMO001 \
  --transcript demo/transcript.txt \
  --document   demo/bill.pdf \
  --out        demo/report.md

cat demo/report.md
```

The demo transcript mentions ≈ ₹50,000; the bill shows ₹62,000.
Expected output: LOW RISK / 100% (amount and dates are vague spoken language,
so those fields are MISSING from the denominator). The UNPARSEABLE flags in
the report are the signal — a reviewer seeing 3 such flags should not treat
this as a clean claim.

---

## Audio input

```bash
python -m claim_verifier.cli verify \
  --claim-id C001 \
  --audio      call_recording.wav \
  --document   bill.pdf \
  --out        report.md
```

Supported formats: WAV, MP3, M4A, FLAC, OGG, MP4 (up to 50 MB).
Transcription uses faster-whisper `base` model, CPU-only, no GPU needed.
The transcribed text never touches an external server.

---

## Web UI and REST API

### Run the web server locally

```bash
ollama serve &

# Start the API server (open http://localhost:8000 in a browser)
python -m claim_verifier.api

# Or with uvicorn directly:
uvicorn claim_verifier.api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — you get a clean form to upload a transcript (or audio
recording) + hospital bill PDF and see the verification report rendered inline.

The REST endpoint is also available for programmatic use:

```bash
curl -X POST http://localhost:8000/verify \
  -F "claim_id=CLM001" \
  -F "transcript=@call.txt" \
  -F "document=@bill.pdf"
```

Returns JSON:
```json
{
  "claim_id": "CLM001",
  "report": "# Claim Verification Report — CLM001\n...",
  "risk_level": "MEDIUM",
  "consistency_score": 0.75,
  "errors": []
}
```

Interactive API docs: `http://localhost:8000/docs`

### Securing the API

Set the `API_KEY` environment variable; the server will then require an
`X-API-Key` header on all `/verify` requests:

```bash
API_KEY=mysecret uvicorn claim_verifier.api.app:app --port 8000

curl -X POST http://localhost:8000/verify \
  -H "X-API-Key: mysecret" \
  -F "claim_id=CLM001" \
  -F "transcript=@call.txt" \
  -F "document=@bill.pdf"
```

### Hosting with Anthropic API (no local Ollama needed)

For cloud deployments where Ollama cannot run, set `BACKEND_TYPE=anthropic`
and provide your `ANTHROPIC_API_KEY`. The system uses the same `LLMBackend`
interface — no code changes required:

```bash
BACKEND_TYPE=anthropic \
ANTHROPIC_API_KEY=sk-ant-... \
API_KEY=mysecret \
uvicorn claim_verifier.api.app:app --host 0.0.0.0 --port 8000
```

Recommended model: `claude-haiku-4-5-20251001` (fast, inexpensive, accurate on
structured extraction). Set `ANTHROPIC_MODEL` env var to override.

**Deploy to Render / Railway / Fly.io:**
1. Set the three env vars above in the platform's environment config
2. Set the start command to: `uvicorn claim_verifier.api.app:app --host 0.0.0.0 --port $PORT`
3. Faster-whisper audio transcription is disabled in hosted mode (no GPU); use
   text transcripts via `--transcript` / the text file upload in the UI

---

## Running on your own files

```bash
python -m claim_verifier.cli verify \
  --claim-id  <YOUR_CLAIM_ID> \
  --transcript <path/to/transcript.txt> \
  --document  <path/to/bill.pdf> \
  [--out <path/to/report.md>]
```

**Transcript** — plain UTF-8 `.txt`, English only.  
**Document** — PDF with a machine-readable text layer (digitally created, not
scanned), English only, must yield ≥ 100 characters, max 20 MB.

---

## Fields verified

| Field | Match type | Tolerance |
|---|---|---|
| Hospital / Facility | Fuzzy string (rapidfuzz) | ratio ≥ 0.85 |
| Admission Date | Date | ±1 day |
| Discharge Date | Date | ±1 day; optional |
| Primary Diagnosis | LLM equivalence judge | disease ≠ procedure |
| Total Billed Amount | Numeric | ±5% |
| Duration of Admission | Numeric | ±1 day |

---

## Risk levels

| Band | Score | Action |
|---|---|---|
| LOW RISK | ≥ 0.90 | Routine review |
| MEDIUM RISK | 0.70–0.89 | Verify flagged fields before approval |
| HIGH RISK | < 0.70 | Escalate to senior reviewer |
| INSUFFICIENT DATA | — | Too many fields absent to score |

---

## Evaluation results

Run the eval harness to generate these numbers on your machine:

```bash
ollama serve &
python -m claim_verifier.eval.run_eval --out eval/report.md
```

First run: ~120 LLM calls (30 min–2 hrs on CPU). Subsequent runs are instant
(LLM response cache).

| Dataset | Cases | Precision | Recall | FP rate |
|---|---|---|---|---|
| Synthetic (upper bound) | 50 | *run eval* | *run eval* | *run eval* |
| Holdout (realistic) | 10 | *run eval* | *run eval* | *run eval* |

Targets: precision ≥ 80%, FP rate on CLEAN cases ≤ 15%.

---

## Running tests

```bash
# All 348 tests — no Ollama needed (W5+ tests use pre-seeded LLM cache)
python -m pytest

# Single module
python -m pytest claim_verifier/tests/test_pipeline.py -v
```

| File | Week | Tests |
|---|---|---|
| `test_contracts.py` | W1 | 31 |
| `test_normalization.py` | W2 | 49 |
| `test_verification.py` | W3 | 40 |
| `test_reporting.py` | W4 | 44 |
| `test_extraction.py` | W5 | 47 |
| `test_pipeline.py` | W6 | 53 |
| `test_data_gen.py` | W7 | 34 |
| `test_metrics.py` | W8 | 23 |
| `test_audio.py` | W9 | 27 |
| `test_api.py` | W10 | 24 |

---

## Known limitations

**1. Scanned / photo PDFs are rejected.**  
Only digitally-created PDFs (text layer) are accepted. OCR on dense hospital
bill tables is unreliable; production-grade support would require Google Vision
or AWS Textract (see Production Roadmap below).

**2. Vague transcript language inflates the risk score.**  
Spoken forms ("around fifty thousand", "twelfth of March") fail normalization
and are excluded from the denominator. A claimant who is vague on all details
can produce a LOW RISK / 100% score even if the bill shows different figures.
The UNPARSEABLE flags in the report are the signal — count them.

**3. LLM extraction quality varies with hardware.**  
On CPU with the 7B model, one call can exceed 120 s. Switch to the 3B fallback
if this is too slow (`OLLAMA_MODEL = OLLAMA_MODEL_FAST` in `config.py`).
The 3B model has slightly lower extraction precision (~65–75% vs ~75–85%).

---

## Production Roadmap

Items deferred from the current scope. Ordered by impact:

| Item | Why deferred | What it needs |
|---|---|---|
| OCR for scanned bills | Google Vision / AWS Textract costs; unreliable on tables | Cloud API + post-OCR cleaning |
| GPU / batched inference | CPU latency acceptable for demo; GPU adds cost/complexity | CUDA environment |
| Hosted API | Ollama is local-only; Anthropic API backend already wired in `LLMBackend` | API key + cloud hosting |
| DIAGNOSIS error injection in synthetic data | Currently only AMOUNT/DATE/HOSPITAL errors; adds coverage | Synonym corpus for diagnosis paraphrasing |
| Confidence calibration reliability plot | Useful for setting thresholds; not needed for MVP | Platt scaling or isotonic regression |
| ICD-10 code normalization | Would sharpen diagnosis matching; overkill for MVP | ICD API or local lookup table |
| Capped denominator / minimum verified-fields threshold | Prevents 100% score when most fields are MISSING | Scoring model change + threshold tuning |
| Multi-language support | Only English currently; Indian regional languages common | Translation layer or multilingual model |
| Audit log / tamper-evident report export | Required for production insurance workflows | Signed PDF + event log |
| Web UI (FastAPI + HTML frontend) | Done — W10 | FastAPI, HTML/JS form, Ollama or Anthropic API backend |

---

## Project structure

```
claim_verifier/
  config.py           — thresholds, model names, cache dir, retention settings
  models.py           — Pydantic v2 data contracts
  llm_cache.py        — SHA-256-keyed LLM response cache
  redaction.py        — PII masking (phone/Aadhaar/PAN/email)
  retention.py        — secure_delete / apply_retention (W9)
  judge.py            — DiagnosisJudge Protocol + LLMJudge + StubJudge
  pipeline.py         — orchestrator: ingestion → extraction → normalize → verify → report
  cli.py              — Typer CLI (verify --transcript/--audio --document)
  backends/           — LLMBackend Protocol, OllamaBackend, StubBackend
  stages/
    ingestion.py      — ingest_transcript / ingest_document / ingest_audio
    normalization.py  — date / amount / hospital / LOS normalization
    extraction.py     — LLM-based FactSet extraction with repair retry
    verification.py   — field-by-field matching and scoring
    reporting.py      — Jinja2 Markdown report renderer
  eval/
    metrics.py        — precision/recall/F1/FP-rate computation
    run_eval.py       — runs pipeline over 50 synthetic + 10 holdout cases
    holdout/          — 10 hand-built messy test cases
  data_gen/
    generate_claims.py  — 50-case synthetic dataset (Faker + error injection)
    generate_bill_pdf.py — text-layer PDF generator (reportlab)
  tests/              — 348 tests, no Ollama required
```
