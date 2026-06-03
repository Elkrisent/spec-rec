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

**Status as of last update:** Week 2 complete.

| Wk | Phase | Status | Notes |
|---|---|---|---|
| 1 | Foundations & contracts | **✓ Done** | 31/31 tests passing |
| 2 | Normalization | **✓ Done** | 80/80 tests passing |
| 3 | Verification + scoring | **✓ Done** | 120/120 tests passing |
| 4 | Reporting | **✓ Done** | 164/164 tests passing |
| 5 | LLM extraction | Not started | Next up |
| 6 | Ingestion + pipeline + CLI | Not started | |
| 7 | Synthetic data generator | Not started | |
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

---