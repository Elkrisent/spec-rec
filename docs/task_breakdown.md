# Task Breakdown (V2)

Granular, checkable tasks per phase. IDs are stable references (e.g. `T3.2`). Effort in dev-days
(undergrad, incl. learning + debugging). Status: `[ ]` todo · `[~]` in progress · `[x]` done.

Legend for each task: **what** · *test that proves it* · dependency.

---

## Phase 1 — Foundations, contracts & local-model spike  (W1, ~2–3 d)
- [ ] **T1.1** Repo scaffold + `requirements.txt` + `pytest.ini` + `pytest` runner. *`pytest` collects 0 failures.* ← nothing
- [ ] **T1.2** `schema/verification_schema.json` with corrected weights. *loads; **weights sum to 1.0**.* ← T1.1
- [ ] **T1.3** `config.py`: load schema, thresholds (0.70/0.90), fuzzy 0.85, Ollama model names + URL. *config imports; thresholds present.* ← T1.2
- [ ] **T1.4** `models.py`: Pydantic v2 `FactValue`, `FactSet`, `FieldVerdict`, `Flag`, `VerificationResult`. *valid fixture parses; malformed rejected; INSUFFICIENT_DATA validator fires.* ← T1.1
- [ ] **T1.5** `tests/fixtures/`: 2 hand-written FactSets (transcript + document) + 1 VerificationResult (C001, amount mismatch scenario). *fixtures validate against models.* ← T1.4
- [ ] **T1.6** `scripts/check_ollama.py`: verify Ollama server running; list available models. *exits 0 if Ollama responds; prints model list.* ← T1.1
- [ ] **T1.7** Ollama model pull (via `check_ollama.py`): pull quality + fast models. *`ollama list` shows both models.* ← T1.6
- [ ] **T1.8** `llm_cache.py`: `LLMCache` — SHA-256-keyed, store/retrieve JSON responses. *cache hit returns identical dict; miss returns None.* ← T1.3
- [ ] **T1.9** `scripts/benchmark_llm.py`: 5-trial JSON-mode benchmark per model; prints valid-JSON %, avg latency, go/no-go verdict. *runs to completion; prints RECOMMENDATION line.* ← T1.7, T1.8
- **Exit:** `pytest` green on contract tests; weights-sum test passes; Ollama running with chosen model pulled (T1.9 gate cleared manually).

## Phase 2 — Normalization  (W2, ~3–4 d)
- [ ] **T2.1** Date normalization: `dateparser(DATE_ORDER='DMY')`; preserve year-absent; no year inference. *`"March 12"`→`--03-12` marker; `"12/03/2025"`→`2025-03-12`.* ← T1.4
- [ ] **T2.2** Amount normalization: strip ₹/Rs/INR/$; Indian grouping → int. *`"1,20,000"`→`120000`; `"₹50,000"`→`50000`.*
- [ ] **T2.3** Hospital canonicalization: lowercase/strip/expand abbrevs. *`"Apollo Hosp."`→`"apollo hospital"`.*
- [ ] **T2.4** Diagnosis passthrough (no ICD) + entity_type carried through. *value unchanged; entity_type preserved.*
- [ ] **T2.5** Unparseable-value handling → confidence 0 + flag. *bad date → flagged, not crash.*
- [ ] **T2.6** `tests/test_normalization.py` — ≥20 table-driven cases incl. edges. *all pass.* ← T2.1–T2.5
- **Exit:** ≥20 cases incl. edges pass.

## Phase 3 — Verification engine + scoring  (W3, ~5–7 d)
- [ ] **T3.1** `judge.py` Protocol + **stub** judge (returns fixed verdict). *stub injectable.* ← T1.4
- [ ] **T3.2** `fuzzy_string` matcher (rapidfuzz, threshold from schema). *ratio≥0.85→MATCH.*
- [ ] **T3.3** `date` matcher; month/day compare when year absent; ±1 day. *off-by-1→MATCH; off-by-5→MISMATCH; yearless compares M/D.*
- [ ] **T3.4** `numeric` matcher (±5% / ±1 abs). *50k vs 62k→MISMATCH; 50k vs 51k→MATCH.*
- [ ] **T3.5** `medical_semantic` via injected judge (disease≠procedure guardrail). *appendicitis vs appendectomy→MISMATCH (stubbed).* ← T3.1
- [ ] **T3.6** Scoring: binary field_score; MISSING excluded+flagged; empty→`INSUFFICIENT_DATA`. *hand-checked on ≥6 scenarios.* ← T3.2–T3.5
- [ ] **T3.7** Flag generation (AMOUNT_MISMATCH, DATE_MISMATCH, LOW_CONFIDENCE, etc.) ordered by severity.
- [ ] **T3.8** `tests/test_verification.py` golden scenarios. *all pass.* ← T3.6, T3.7
- **Exit:** golden tests pass; score arithmetic verified by hand; core runs with no LLM/no I/O.

## Phase 4 — Reporting  (W4, ~2–3 d)
- [ ] **T4.1** `templates/report.md.j2`: header, field table, flags, missing, quality, guidance.
- [ ] **T4.2** `stages/reporting.py`: render VerificationResult → Markdown; citation format `value [source: "quote"]`. ← T1.4
- [ ] **T4.3** Reviewer-guidance text keyed to risk band (LOW/MEDIUM/HIGH).
- [ ] **T4.4** `tests/test_reporting.py` snapshot tests. *citations present; badge correct; flags ordered.* ← T4.2, T4.3
- **Exit:** snapshot tests pass + manual eyeball.

## Phase 5 — Local LLM extraction  (W5, ~5–7 d)
- [ ] **T5.1** `LLMBackend` Protocol in `backends/__init__.py`: `complete(messages, schema) → dict`. *stub impl passes type check.*
- [ ] **T5.2** `backends/ollama.py`: Ollama REST impl of `LLMBackend`; JSON-schema-constrained `format` param; cache-first. *known text → schema-valid dict; cache hit skips inference.* ← T1.8
- [ ] **T5.3** Extraction JSON schema (FactSet shape) defined in `stages/extraction.py`. ← T1.4
- [ ] **T5.4** `stages/extraction.py`: call `LLMBackend.complete` → validate against Pydantic FactSet; **one repair retry** on parse error. *malformed→retry→valid or clean error.* ← T5.1–T5.3
- [ ] **T5.5** Quote substring-validation; on fail mark `quote_verified=False` (don't zero confidence). *paraphrased quote→`quote_verified=False` flag.* ← T5.4
- [ ] **T5.6** Real diagnosis judge impl in `judge.py` (replaces stub, same Protocol); calls `LLMBackend`. ← T3.1, T5.2
- [ ] **T5.7** `tests/test_extraction.py` against **cached** responses (no Ollama server needed). *≥5 texts→valid FactSets; offline green.* ← T5.4–T5.6
- **Exit:** valid FactSet on ≥5 texts; offline cached tests green.
- **Accuracy note:** expect 75–85% precision; quote fidelity weakest point.

## Phase 6 — Ingestion (PDF) + pipeline + CLI  (W6, ~3–4 d)
- [ ] **T6.1** `stages/ingestion.py`: pdfplumber text-layer; <100 chars → reject (scanned); non-English → reject.
- [ ] **T6.2** `redaction.py`: regex for phone/Aadhaar/PAN/email; applied to API-bound copy only. *masks all four.*
- [ ] **T6.3** `pipeline.py`: orchestrate stages 1–5 with `{status,data,error}` contract; partial report on stage failure. *empty-PDF→partial report+flag, no crash.* ← T2–T5
- [ ] **T6.4** `cli.py` (Typer): `verify --claim-id --transcript/--audio --document --out`. ← T6.3
- [ ] **T6.5** `tests/test_pipeline.py` end-to-end on fixtures + failure injection. *e2e report; error-contract holds.* ← T6.3, T6.4
- **Exit:** CLI produces a report on text inputs; error-contract test passes. **MVP (text).**

## Phase 7 — Synthetic data generator  (W7, ~6–9 d)  ⚠ long pole
- [ ] **T7.1** `generate_bill_pdf.py`: text-layer bill PDF (reportlab/fpdf) from fact dict. *extracted text layer matches injected values.*
- [ ] **T7.2** `generate_claims.py`: **template + Faker-based** transcript generation from ground truth (no LLM — eliminates generator/extractor leakage).
- [ ] **T7.3** Error injection: AMOUNT (10–40%), DATE (3–15 d), HOSPITAL, CLEAN; labeled. *labels match injected deltas.*
- [ ] **T7.4** **Round-trip check**: injected fact appears verbatim in transcript; reject/regenerate otherwise. *0 cases fail round-trip.* ← T7.2, T7.3
- [ ] **T7.5** Emit dataset: 50 cases (35 error/15 clean) + ground_truth JSON. ← T7.1–T7.4
- [ ] **T7.6** `tests/test_data_gen.py`. *PDF text-layer + round-trip + label tests pass.*
- **Exit:** ~50 cases generated; round-trip passes.

## Phase 8 — Evaluation harness  (W8, ~4–6 d)
- [ ] **T8.1** Write explicit tolerance-aware **TP/FP/FN definitions** (per field, per flag) in a doc comment. *peer-readable.*
- [ ] **T8.2** `metrics.py`: precision/recall per field + per flag; FP-rate on clean cases. *known confusion fixture→known P/R.* ← T8.1
- [ ] **T8.3** `run_eval.py`: run pipeline over dataset via cache; aggregate; report vs. targets. ← T6.3, T7.5, T8.2
- [ ] **T8.4** Hand-build `eval/holdout/` (~10–15 messy cases) the system never trained on.
- [ ] **T8.5** Report: synthetic = upper-bound smoke test; headline holdout; print **leakage caveat**;
       **print local-model accuracy delta** vs. the relaxed targets (precision >80%, FP rate <15%).
- [ ] **T8.6** `tests/test_metrics.py`. *metric math verified.*
- **Exit:** eval report for 50 synthetic + holdout; reproducible; caveat printed. **Eval report ships.**

## Phase 9 — Whisper audio + hardening  (W9, ~stretch)
- [ ] **T9.1** Audio ingestion: **faster-whisper** base (CPU); non-English reject. *English wav→transcript→report.* ← T6.1
- [ ] **T9.2** Input size/type validation (50MB audio / 20MB doc; extension allowlist). *bad input rejected.*
- [ ] **T9.3** Retention policy on derived artifacts (configurable) + secure-delete of sources.
- [ ] **T9.4** `README.md` incl. **Production Roadmap** (deferred list) + results table.
- **Exit:** `--audio` yields a report; full suite green.

## Phase 10 — Buffer / stretch / polish  (W10)
- [ ] **T10.1** (stretch) Confidence-calibration reliability plot.
- [ ] **T10.2** (stretch) Remaining error types (DIAGNOSIS, MISSING_FIELD) + scale synthetic to 200.
- [ ] **T10.3** (stretch) FastAPI `/verify` + API key.
- [ ] **T10.4** Final full eval run; tag release; demo in README.
- **Exit:** tagged release; reproducible eval; results table.

---

## Critical path & risk notes
- **Critical path:** T1 → T2/T3 (core) → T5 (extraction) → T6 (integration) → T7 → T8 (eval).
  Reporting (T4) and audio (T9) are off the critical path.
- **Long pole:** Phase 7 (data_gen). Start scoping its PDF generator early if W1–W4 finish ahead.
- **Highest-skill:** T5 (Ollama constrained extraction + LLMBackend Protocol) and T8 (metric
  definitions + honest eval framing) — budget extra time.
- **Top risk (local):** CPU inference latency + JSON validity rate. T1.9 benchmark gates this;
  if 7B fails go/no-go, fall back to 3B immediately.
- **De-risked by design:** T1–T4 need no Ollama and no audio; T5+ runs entirely from cache in CI.
