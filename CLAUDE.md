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

**Status as of last update:** Week 6 complete.

| Wk | Phase | Status | Notes |
|---|---|---|---|
| 1 | Foundations & contracts | **✓ Done** | 31/31 tests passing |
| 2 | Normalization | **✓ Done** | 80/80 tests passing |
| 3 | Verification + scoring | **✓ Done** | 120/120 tests passing |
| 4 | Reporting | **✓ Done** | 164/164 tests passing |
| 5 | LLM extraction | **✓ Done** | 211/211 tests passing |
| 6 | Ingestion + pipeline + CLI | **✓ Done** | 264/264 tests passing |
| 7 | Synthetic data generator | Not started | Next up |
| 8 | Evaluation harness | Not started | |
| 9 | Audio + hardening | Not started | Stretch |
| 10 | Buffer / polish | Not started | Stretch |

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
- `claim_verifier/cli.py` — Typer `verify` command; `--claim-id`, `--transcript`, `--document`, `--out`; `--audio` deferred with clear error message; exit code 2 on pipeline errors
- `claim_verifier/tests/test_pipeline.py` — 53 tests (T6.1–T6.5): ingestion happy-path + error cases (scanned PDF, non-English, empty, missing), redaction all 4 PII families, pipeline E2E with SeqStub + StubJudge, extraction failure → partial report, ingestion failure via run()

---