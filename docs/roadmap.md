# Implementation Roadmap (V2)

Constraints: **1 undergraduate dev · 2–3 months (~45–55 dev-days) · limited budget · 16GB CPU laptop.**

**Integration-risk strategy:** build *inside-out*. The deterministic core (W1–W4) is built and
tested against hand-written fixtures with **no LLM and no file I/O**. The LLM extraction (W5) is
added behind a **hash-keyed response cache** so CI is offline and deterministic. The pipeline,
data-gen, eval, and audio are integrated last. Every seam is unit-tested before integration.

Rules honored every phase: **independently testable · ships a working artifact · integration
deferred until inputs are already tested.**

---

## Phase / Week overview
| Wk | Phase | Working artifact |
|---|---|---|
| 1 | Foundations & contracts | self-validating config + models |
| 2 | Normalization | `normalize(raw)→normalized` |
| 3 | Verification + scoring | `verify(a,b,judge)→Result` (judge stubbed) |
| 4 | Reporting | rendered Markdown report |
| 5 | LLM extraction (cached) | `extract(text)→FactSet` |
| 6 | Ingestion + pipeline + CLI | working CLI MVP (text inputs) |
| 7 | Synthetic data generator | 50-case labeled dataset |
| 8 | Evaluation harness | the eval report |
| 9 | Whisper audio + hardening | full MVP + audio demo |
| 10 | Buffer / stretch / polish | tagged portfolio release |

---

## Week 1 — Foundations, contracts & local-model spike
- **Goal:** Lock data contracts AND verify local LLM is feasible before building on it.
- **Deliverables:** repo scaffold; `config.py` (schema/weights/thresholds/Ollama model names);
  Pydantic models; `requirements.txt`; verification schema JSON; `llm_cache.py`;
  Ollama setup + benchmark scripts.
- **Files:** `config.py`, `models.py`, `llm_cache.py`, `schema/verification_schema.json`,
  `requirements.txt`, `pytest.ini`, `scripts/check_ollama.py`, `scripts/benchmark_llm.py`,
  `tests/conftest.py`, `tests/test_contracts.py`, `tests/fixtures/` (3 JSON files).
- **Tests:** schema loads; **weights sum to 1.0**; models validate good fixtures and reject
  malformed; `LLMCache` stores and retrieves deterministically.
- **Local-model gate (T1.9):** run `python -m claim_verifier.scripts.benchmark_llm` manually;
  if 7B valid-JSON ≥80% AND avg ≤60s → use quality model; else fall back to fast model.
- **Exit criteria:** `pytest` green on contract tests; Ollama running with chosen model pulled.

## Week 2 — Normalization (deterministic, no API)
- **Goal:** Deterministic date/amount/hospital transforms.
- **Deliverables:** `normalize(raw_factset) → normalized_factset`.
- **Files:** `stages/normalization.py`, `tests/test_normalization.py`.
- **Tests:** table-driven — `"1,20,000"→120000`; `"March 12"` keeps year absent; `"Hosp."→"Hospital"`;
  unparseable date → confidence 0 + flag; `DATE_ORDER='DMY'` respected.
- **Exit criteria:** ≥20 cases incl. edge cases pass; module coverage high.

## Week 3 — Verification engine + scoring (judge stubbed)
- **Goal:** Typed matchers + binary scoring on hand-written normalized FactSets; diagnosis judge
  injected as a stub (Protocol).
- **Deliverables:** `verify(factset_a, factset_b, judge) → VerificationResult`.
- **Files:** `stages/verification.py`, `judge.py` (Protocol + stub), `tests/test_verification.py`.
- **Tests:** golden scenarios — clean match; 24% amount mismatch; date off-by-N; missing field;
  all-missing → `INSUFFICIENT_DATA`; **score arithmetic checked by hand** on ≥6 cases.
- **Exit criteria:** deterministic golden tests pass; the entire core works with no LLM, no I/O.

## Week 4 — Reporting
- **Goal:** VerificationResult → Markdown.
- **Deliverables:** report renderer + template.
- **Files:** `stages/reporting.py`, `templates/report.md.j2`, `tests/test_reporting.py`.
- **Tests:** snapshot-render golden results; assert citations present, risk badge correct, flags
  severity-ordered, reviewer guidance matches risk level.
- **Exit criteria:** snapshot tests pass + manual eyeball of one report. First **demo-able** artifact.

## Week 5 — Local LLM extraction (the hard one)
- **Goal:** Ollama JSON-schema-constrained FactSet extraction + diagnosis judge, with all calls cached.
- **Deliverables:** `extract(text, source) → FactSet`; real diagnosis judge; `LLMBackend` Protocol
  + Ollama impl (in `backends/ollama.py`).
- **Files:** `stages/extraction.py`, `backends/__init__.py`, `backends/ollama.py`, `judge.py`
  (real impl), `tests/test_extraction.py`.
- **Tests:** known text → schema-valid FactSet (via cache); quote is a substring of source;
  cache hit deterministic; CI runs entirely from cache (no Ollama server required).
- **Accuracy note:** expect 75–85% extraction precision on clean synthetic; lower on holdout.
  Verbatim quote fidelity is the weakest point — substring-validation is the critical guard.
- **Exit criteria:** valid FactSet on ≥5 sample texts; offline cached tests green.

## Week 6 — Ingestion (text-layer PDF) + pipeline + CLI + error contract
- **Goal:** End-to-end on `--transcript` + text-layer PDF.
- **Deliverables:** ingestion (pdfplumber text-layer, <100-char reject, non-English reject); pipeline
  orchestration with the Result/error contract; Typer CLI; regex redaction.
- **Files:** `stages/ingestion.py`, `pipeline.py`, `cli.py`, `redaction.py`, `tests/test_pipeline.py`.
- **Tests:** e2e fixture transcript + bill → full report; empty-PDF → partial report + flag (no
  crash); redaction masks phone/Aadhaar/PAN/email.
- **Exit criteria:** `verify --claim-id X --transcript t.txt --document bill.pdf` produces a report;
  error-contract test passes. **Working CLI MVP (text inputs).**

## Week 7 — Synthetic data generator (long pole)
- **Goal:** Labeled evaluation cases.
- **Deliverables:** transcript generator (**template + Faker-based**, no LLM — eliminates
  generator/extractor leakage; ground truth + 4 error types: AMOUNT/DATE/HOSPITAL/CLEAN) +
  text-layer bill PDF generator.
- **Files:** `data_gen/generate_claims.py`, `data_gen/generate_bill_pdf.py`, `tests/test_data_gen.py`.
- **Tests:** bill PDF has an extractable text layer; **round-trip** — injected fact appears verbatim
  in its transcript; ground_truth JSON well-formed; error-type labels correct.
- **Exit criteria:** ~50 cases (35 error / 15 clean) generated; round-trip checks pass.

## Week 8 — Evaluation harness (centerpiece)
- **Goal:** Honest measurement against ground truth.
- **Deliverables:** `metrics.py` (explicit tolerance-aware TP/FP/FN per field and per flag);
  `run_eval.py` (**pre-compute and cache all model outputs first**, then aggregate vs. targets);
  a hand-built hard holdout (~10–15 cases).
- **Files:** `eval/metrics.py`, `eval/run_eval.py`, `eval/holdout/`, `tests/test_metrics.py`.
- **Tests:** metrics on synthetic confusion fixtures (known TP/FP/FN → known precision/recall);
  run_eval emits a report.
- **Relaxed targets (local-model hypotheses):** extraction precision >80% (not 90%); contradiction
  detection precision >80% (not 88%); FP rate on clean <15% (not 12%). Frame as hypotheses.
- **Exit criteria:** eval report for 50 synthetic + holdout; reproducible via cache; **leakage
  caveat printed** (template-based transcripts reduce but don't eliminate cross-model bias);
  synthetic labeled as upper-bound. The **eval report** ships.

## Week 9 — faster-whisper audio (stretch #1) + hardening
- **Goal:** Optional `--audio` (English, faster-whisper base, CPU) + robustness pass.
- **Deliverables:** audio ingestion path (non-English reject); input size/type validation; retention
  policy on derived artifacts; README + Production Roadmap.
- **Files:** extend `stages/ingestion.py`, `README.md`.
- **Tests:** short English wav → transcript → existing pipeline → report; size/type validation
  rejects bad input; all prior tests still green.
- **Exit criteria:** `--audio sample.wav` yields a report; full suite green. **Full MVP + audio demo.**

## Week 10 — Buffer / stretch / polish
- **Goal:** Absorb slippage; add stretch goals if ahead.
- **Stretch menu:** confidence-calibration plot; remaining error types (DIAGNOSIS, MISSING_FIELD) +
  scale synthetic to 200; FastAPI `/verify` + API key; LOS/discharge derivation; HTML report.
- **Files:** as selected.
- **Tests:** full suite + a full eval run.
- **Exit criteria:** tagged release; README results table; reproducible eval. **Shippable repo.**

---

## Local compute notes (replaces API budget)
**Cost: $0.** No API keys, no cloud services, no recurring spend.

- **Latency:** ~15–60s/call on CPU with 7B model; ~5–20s with 3B. A 50-case eval (≈2–3 calls
  each) runs ~1–2 hours. Use `llm_cache.py` to avoid re-running; develop on ~10 cases.
- **RAM:** 7B Q4_K_M ≈ 4.7GB; 3B ≈ 2.5GB. Leave ≥2GB free for OS + Python. Load models
  sequentially — don't hold Whisper + LLM resident at the same time (use Ollama load/unload).
- **Accuracy:** expect 75–85% on clean synthetic (upper bound); lower on holdout. Document the
  delta vs. hosted models as the honest engineering judgment section of the portfolio.
- **W1 gate:** run the benchmark script; if 7B is too slow, fall back to 3B for dev iterations
  and reserve 7B only for final eval runs.
