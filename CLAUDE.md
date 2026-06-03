# Project Context

## Goal
Build a Medical Claim Verification Assistant.

## Constraints
- Solo developer
- 1 month timeline max
- Limited budget
- Portfolio project

## Priorities
- Practicality
- Industry relevance
- Good engineering
- extensive testing

## Review Style
Be critical.
Identify flaws before implementation.
update requirements.txt and .gitignore every time a new dependency is introduced.

Authoritative documents:
1. architecture.md
2. task_breakdown.md

Non-authoritative:
- project_spec.md (historical only)
- roadmap.md (planning only)

If conflicts exist:
task_breakdown.md > architecture.md > roadmap.md

## Manual prerequisites per week

| Wk | Needed before starting | Notes |
|---|---|---|
| 1 | **✓ Done** — Ollama running, models pulled, T1.9 benchmark passed | `ollama serve` + `python scripts/check_ollama.py` |
| 2 | None — fully deterministic | |
| 3 | None — stub judge, no LLM | |
| 4 | None — Jinja2 reporting only | |
| 5 | **Ollama server running** with both models loaded | Run `.venv/bin/python -m claim_verifier.scripts.check_ollama` to verify before starting |
| 6 | None | |
| 7 | None | |
| 8 | None | |
| 9 | Uncomment `faster-whisper>=1.0` in `requirements.txt`; run `uv pip install -r requirements.txt`; have a short English `.wav` ready for smoke testing | CPU only, no CUDA needed |
| 10 | None | |

---

## Current progress

**Status as of last update:** Week 10 complete. Project done.

| Wk | Phase | Status | Notes |
|---|---|---|---|
| 1 | Foundations & contracts | **✓ Done** | 31/31 tests passing |
| 2 | Normalization | **✓ Done** | 80/80 tests passing |
| 3 | Verification + scoring | **✓ Done** | 120/120 tests passing |
| 4 | Reporting | **✓ Done** | 164/164 tests passing |
| 5 | LLM extraction | **✓ Done** | 211/211 tests passing |
| 6 | Ingestion + pipeline + CLI | **✓ Done** | 264/264 tests passing |
| 7 | Synthetic data generator | **✓ Done** | 298/298 tests passing |
| 8 | Evaluation harness | **✓ Done** | 321/321 tests passing |
| 9 | Audio + hardening | **✓ Done** | 348/348 tests passing |
| 10 | Web UI + polish | **✓ Done** | 372/372 tests passing |

**W1 deliverables shipped:**
- `claim_verifier/config.py` — schema, thresholds, Ollama model names
- `claim_verifier/models.py` — Pydantic v2: `FactValue`, `FactSet`, `FieldVerdict`, `Flag`, `VerificationResult`
- `claim_verifier/llm_cache.py` — SHA-256-keyed response cache
- `claim_verifier/schema/verification_schema.json` — corrected weights (sum = 1.0)
- `claim_verifier/tests/fixtures/` — 3 hand-written fixtures (C001 claim, amount-mismatch scenario)
- `claim_verifier/scripts/check_ollama.py` — verify Ollama + pull models (T1.6/T1.7)
- `claim_verifier/scripts/benchmark_llm.py` — go/no-go benchmark: quality vs fast model (T1.9)
- `requirements.txt` + `requirements-dev.txt` at repo root

**W2 deliverables shipped:**
- `claim_verifier/stages/normalization.py` — `normalize(raw) → (normalized, flags)`: date/amount/hospital/LOS/diagnosis
- `claim_verifier/tests/test_normalization.py` — 49 table-driven cases (T2.1–T2.6)

**W3 deliverables shipped:**
- `claim_verifier/judge.py` — `DiagnosisJudge` Protocol + `StubJudge` (configurable verdict for testing)
- `claim_verifier/stages/verification.py` — `verify(claim_id, transcript, document, judge) → VerificationResult`: fuzzy/date/numeric/medical_semantic matchers, binary scoring, MISSING exclusion, INSUFFICIENT_DATA guard, flag severity ordering
- `claim_verifier/tests/test_verification.py` — 40 tests (T3.1–T3.8) incl. 8 golden score scenarios hand-verified; year-absent date bug caught + fixed

**W4 deliverables shipped:**
- `claim_verifier/templates/report.md.j2` — Jinja2 template: header, field table with citations, flags, missing fields, quality warnings, reviewer guidance
- `claim_verifier/stages/reporting.py` — `render(result: VerificationResult) → str`; citation format `value [source: "quote"]`; reviewer guidance keyed to all four risk bands
- `claim_verifier/tests/test_reporting.py` — 44 tests (T4.1–T4.4): citation helper, badge, score %, flag ordering, missing/quality messaging, guidance text, field labels

**W5 deliverables shipped:**
- `claim_verifier/backends/__init__.py` — `LLMBackend` Protocol (runtime_checkable) + `StubBackend` with call tracking
- `claim_verifier/backends/ollama.py` — `OllamaBackend`: cache-first, `/api/chat`, JSON-schema `format` param, full response cached
- `claim_verifier/stages/extraction.py` — `EXTRACTION_SCHEMA` JSON schema + `extract(source_type, source_id, text, backend) → FactSet`; one repair retry; `_verify_quotes()` marks `quote_verified=False` without zeroing confidence
- `claim_verifier/judge.py` — `LLMJudge` added (T5.6); `StubJudge` retained; `JUDGE_SCHEMA` defined
- `claim_verifier/tests/test_extraction.py` — 47 tests (T5.1–T5.7): Protocol checks, cache hit path, schema structure, extract() logic, repair retry, quote verification, LLMJudge, 5-texts offline via pre-seeded LLMCache

**W6 deliverables shipped:**
- `claim_verifier/stages/ingestion.py` — `ingest_transcript(path) → str`; `ingest_document(path) → str`; pdfplumber text-layer; `<100 chars → reject (scanned)`; non-English → reject (langdetect, seed=0)
- `claim_verifier/redaction.py` — `redact(text) → str`; regex masks email/PAN/Aadhaar/phone before any LLM call; original text kept intact for quote verification
- `claim_verifier/pipeline.py` — `run()` (file paths, stages 1–5) + `run_from_text()` (text strings, stages 2–5); `PipelineResult` dataclass; `{status, data, error}` contract; failed stage → partial INSUFFICIENT_DATA report + HIGH flag, no crash; normalization flags merged and re-sorted
- `claim_verifier/cli.py` — Typer `verify` command; `--claim-id`, `--transcript/--audio`, `--document`, `--out`; exit code 2 on pipeline errors
- `claim_verifier/tests/test_pipeline.py` — 53 tests (T6.1–T6.5): ingestion happy-path + error cases (scanned PDF, non-English, empty, missing), redaction all 4 PII families, pipeline E2E with SeqStub + StubJudge, extraction failure → partial report, ingestion failure via run()

**W7 deliverables shipped:**
- `claim_verifier/data_gen/__init__.py`
- `claim_verifier/data_gen/generate_bill_pdf.py` — `generate_bill_pdf(facts, path) → Path`; reportlab A4 text-layer PDF; extractable by pdfplumber (T7.1)
- `claim_verifier/data_gen/generate_claims.py` — `generate_dataset(n_total, n_clean, output_dir, seed) → list[dict]`; 6 Faker-based transcript templates; `_inject_amount_error` (10–40%); `_inject_date_error` (3–15 d); `_inject_hospital_error` (different hospital → guaranteed fuzzy MISMATCH); `round_trip_check` verifies all injected values are verbatim in transcript text; writes 50 cases (35 error/15 clean) + `ground_truth.jsonl` to `data_gen/output/` (T7.2–T7.5)
- `claim_verifier/tests/test_data_gen.py` — 34 tests (T7.6): PDF text-layer (9), round-trip (6), error injectors (9), dataset generation (10)

**W8 deliverables shipped:**
- `claim_verifier/eval/__init__.py`
- `claim_verifier/eval/metrics.py` — `compute_field_metrics` (TP/FP/FN/TN + n_missing + precision/recall/f1); `compute_fp_rate_on_clean` (case-level FP rate on CLEAN cases); `compute_eval_summary` (all fields); `format_report` (with leakage caveat + target check) (T8.1–T8.2, T8.5)
- `claim_verifier/eval/run_eval.py` — `eval_dataset`, `eval_holdout`, `main` CLI; runs pipeline on 50 synthetic + 10 holdout cases; prints eval report; **requires Ollama running on first run** (T8.3)
- `claim_verifier/eval/holdout/HOLD001–HOLD010/` — 10 hand-built holdout cases: clean (6), single error (3: AMOUNT/DATE/HOSPITAL), double error (1: AMOUNT+DATE); each has transcript.txt + bill.pdf + ground_truth.json; covers tolerance boundaries (±5% amount, ±1 day date, hospital abbreviation, Indian comma notation, vague transcript) (T8.4)
- `claim_verifier/tests/test_metrics.py` — 23 tests (T8.6): field metrics math (9), FP rate (5), eval summary (4), report formatting (5)

**W9 deliverables shipped:**
- `claim_verifier/stages/ingestion.py` — `ingest_audio(path) → str`; faster-whisper `base` model, CPU, int8; non-English reject; size limit 50 MB; extension allowlist; lazy import (works when faster-whisper not installed with clear error) (T9.1); added size (20 MB) + extension validation to `ingest_document` and extension validation to `ingest_transcript` (T9.2)
- `claim_verifier/retention.py` — `secure_delete(path, *, secure=False)` (overwrite with zeros then unlink); `apply_retention(paths, *, secure=False)` (T9.3)
- `claim_verifier/config.py` — `RETENTION_DELETE_SOURCES`, `RETENTION_SECURE_DELETE` knobs
- `claim_verifier/cli.py` — `--audio` flag fully wired: `ingest_audio → run_from_text` path; `--transcript` and `--audio` are mutually exclusive
- `README.md` — project overview, prerequisites, installation, demo, audio input, fields table, risk levels, eval results table, known limitations, production roadmap (deferred items) (T9.4)
- `claim_verifier/tests/test_audio.py` — 27 tests: audio ingestion (14), document/transcript validation (3), secure_delete (5), apply_retention (5)

**W10 deliverables shipped:**
- `claim_verifier/backends/anthropic_backend.py` — `AnthropicBackend`: Claude API via tool use for JSON-schema-constrained output; same LLMCache used by OllamaBackend; splits system/user messages for Anthropic format (T10.3)
- `claim_verifier/api/__init__.py` — package init
- `claim_verifier/api/app.py` — FastAPI app factory (`create_app(backend, judge, api_key)`); `GET /` (HTML frontend), `GET /health`, `POST /verify` (multipart); backend selection via `BACKEND_TYPE=ollama|anthropic|stub` env var; optional `API_KEY` header auth; temp-file handling + cleanup; audio path via `ingest_audio`; transcript path via `pipeline.run()`; graceful errors (always 200 with INSUFFICIENT_DATA on pipeline failure) (T10.3)
- `claim_verifier/api/__main__.py` — `python -m claim_verifier.api` entry point
- HTML frontend (embedded in app.py) — single-page form; transcript/audio toggle; PDF upload; marked.js markdown rendering; loading spinner; risk-level badge
- `README.md` — web API/UI section with local and hosted (Render/Railway) deployment instructions; Anthropic backend env vars; REST API curl examples
- `claim_verifier/tests/test_api.py` — 24 tests: health (3), root HTML (3), verify happy-path (7), input validation (4), audio path (2), auth (5)

---