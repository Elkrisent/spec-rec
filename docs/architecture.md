# Medical Claim Verification Assistant — Architecture (V2)

> Authoritative architecture. Supersedes `project_spec.md` (V1), which is retained for history.
> Derived from a flaw-driven critique of V1. Design rule: **cut fragile peripheral subsystems;
> double down on the parts that are simultaneously low-risk and high-depth.**

---

## 0. Mission
Given a phone-call transcript (or audio) and a hospital bill, produce a structured, cited,
risk-scored consistency report telling a human reviewer exactly where the claimant's statements
**match, conflict with, or are unsupported by** the submitted bill.

---

## 1. Design thesis — what carries the portfolio
Depth lives in four inspectable, trustworthy things:
1. **Structured LLM extraction** — tool-use / JSON-schema-enforced FactSet with verbatim quotes,
   validation + repair. Real LLM engineering, not prompt-and-pray.
2. **Deterministic verification engine** — typed matchers, defensible scoring. Auditable IP.
3. **Correct scoring methodology** — confidence flags rather than multiplies; mismatches penalized;
   empty-denominator handled.
4. **Honest evaluation** — explicit TP/FP/FN definitions, synthetic labeled as a smoke test, a small
   hard holdout, leakage acknowledged.

Everything removed below is a risk/complexity sink with low marginal portfolio value.

---

## 2. Decisions vs. V1

### Removed
| Removed | Why | Replaced by |
|---|---|---|
| scispacy + `en_core_sci_sm` | install hell, marginal value | entity-type tagging inside the extraction LLM call |
| `simple_icd_10` mapping | wrong tool (code-tree nav, not text→code); category match arbitrary | LLM diagnosis-equivalence judge with disease≠procedure guardrail |
| Tesseract / pdf2image / poppler | garbles dense bill tables; never exercised by synthetic PDFs | text-layer PDFs only; scanned input rejected with a clear error |
| Helsinki-NLP / translation | whole subsystem for an out-of-scope path | English-only; non-English detected and **rejected** |
| Confidence multiplication in score | uncalibrated; deflates perfect matches | confidence drives **flags only** |
| Dual HTML + Markdown rendering | two renderers for one MVP | Markdown only |
| FastAPI as primary interface | unauthenticated PHI endpoint, extra surface | CLI primary; FastAPI optional stretch |
| Presidio (heavy PII) | US/EU recognizers miss Aadhaar/PAN; breaks verbatim citations | best-effort regex redaction on API-bound copy; full Presidio deferred |

### Simplified
- **Score:** `Σ w·field_score / Σ w(included)`; **MATCH=1.0 / MISMATCH=0.0**; MISSING ⇒ flagged +
  excluded; empty denominator ⇒ `INSUFFICIENT_DATA`. Confidence is out of the formula.
- **Diagnosis:** one LLM equivalence call (verdict + rationale + entity types); no code DB.
- **Hospital name:** deterministic normalize → `rapidfuzz`; **weight lowered to 0.10**, advisory.
- **Dates:** `dateparser(DATE_ORDER='DMY')`; transcript year absent ⇒ compare month/day only; ±1 day.
- **Modules:** 5 logical **functions in one pipeline**; one FactSet schema; V1 ceremony dropped.
- **Extraction:** **tool-use / structured output** + one validation-retry.
- **Inputs:** accept `--audio` **or** `--transcript` (decouples STT; fully testable without audio).
- **Quotes:** substring-validate; on fail keep value, mark quote `unverified` (don't zero).

### Deferred (Production Roadmap — documented, not built)
Scanned/handwritten OCR · multilingual + translation · diagnosis→ICD/UMLS mapping · multi-document
reconciliation · full Presidio PII (Aadhaar/PAN recognizers) · production auth/encryption/retention ·
cloud/DB storage · any real patient data.

---

## 3. Verification schema (corrected)
Weights **sum to 1.0**. Hospital lowered (advisory); the authoritative/objective fields carry more.

| id | label | weight | match_type | tolerance |
|---|---|---|---|---|
| `hospital_name` | Hospital / Facility | **0.10** | fuzzy_string | ratio ≥ 0.85 |
| `admission_date` | Admission Date | 0.20 | date | ±1 day (month/day if year absent) |
| `discharge_date` | Discharge Date | 0.10 | date | ±1 day; optional |
| `diagnosis` | Primary Diagnosis | 0.25 | medical_semantic | LLM equivalence |
| `billed_amount` | Total Billed Amount | 0.25 | numeric | ±5% |
| `length_of_stay` | Duration (days) | 0.10 | numeric | ±1 day |

**Score (binary per field):**
```
field_score(f) = 1.0 if verdict == MATCH else 0.0      # present in both sources
               = excluded from denominator             # missing from one (flagged) or both
consistency_score = Σ weight(f)·field_score(f) / Σ weight(f over included fields)
if no included fields: risk_level = INSUFFICIENT_DATA (no numeric score)
```
Risk bands: ≥0.90 LOW · 0.70–0.89 MEDIUM · <0.70 HIGH. Continuous ratios/deltas are kept in each
verdict's `note` for the reviewer (not in the score).

---

## 4. Pipeline
CLI-first. Each stage returns `{status, data, error}` — **no error silently crosses a boundary**; a
failed stage yields a *partial report + flag*, never a bare crash.

```
verify --claim-id C123  (--audio call.mp3 | --transcript call.txt)  --document bill.pdf  [--out report.md]

[1] INGESTION   audio→faster-whisper(base) OR --transcript bypass ; doc→pdfplumber TEXT LAYER ONLY
                (<100 chars ⇒ reject scanned ; non-English ⇒ reject)
        ▼  raw transcript text + raw document text
[2] EXTRACTION  Ollama LLM via LLMBackend, 1 call/source → FactSet{value,confidence,quote,entity_type}
                JSON-schema-constrained (valid JSON guaranteed) ; PII redaction before call
                response cache hit ⇒ skip inference (offline/CI safe)
        ▼  two FactSets
[3] NORMALIZE   dates(DMY,year-absent) · amounts(₹→int) · hospital(canonical) · diagnosis(passthrough)
        ▼  two normalized FactSets
[4] VERIFY      fuzzy(hospital) · date(month/day if no year) · numeric(±5%/±1) · diagnosis(LLM judge)
                SCORE binary ; MISMATCH→0 ; empty→INSUFFICIENT_DATA
        ▼  VerificationResult JSON
[5] REPORT      Jinja2 → Markdown: summary · field table · flags(severity) · missing · quality · cited quotes
```

**Cross-cutting:** best-effort regex PII redaction (phone/Aadhaar/PAN/email) applied before any
LLM call (defense-in-depth; all processing is local so no data egress risk); retention policy
covers **derived artifacts** (reports, FactSets), not just source files; input size/type
validation. All audio, documents, and extracted data remain on the local machine — **no data
leaves the machine**.

---

## 5. API vs. local — decision & tradeoffs

**Decision: fully local, $0 recurring cost.**

For a privacy-centric medical portfolio project, local inference is the stronger story, not a
compromise:
- Resolves V1's "all data stays local" contradiction — now literally true.
- Demonstrates harder, more differentiated skills: quantized CPU inference, constrained JSON
  decoding, model-selection tradeoffs, offline reproducibility.
- Documenting the accuracy tradeoff (local 7B ~75–85% vs Sonnet ~90%+) is itself a portfolio
  asset — it shows engineering judgment.

**Tradeoffs to document honestly:**
- Extraction precision: ~75–85% on clean synthetic / lower on holdout (vs ~90%+ Sonnet).
- Verbatim quote fidelity degrades most (small models paraphrase); substring-validation essential.
- Inference speed: ~15–60s/call on CPU vs ~1–2s API; caching mandatory; develop on ~10 cases.

The LLM is behind a thin `LLMBackend` interface (one Ollama impl). A hosted model remains a
config swap — no API SDK in `requirements.txt`.

---

## 5. Tech stack
```
Language          : Python 3.12+
Interface         : CLI (Typer)              # FastAPI deferred (optional stretch)
STT               : faster-whisper (base default, CPU) — bypassable via --transcript  # W9
PDF text          : pdfplumber               # text-layer only
Fuzzy match       : rapidfuzz
Date parsing      : dateparser (DATE_ORDER='DMY')
LLM backend       : Ollama (local, $0, OpenAI-compatible REST, JSON-schema constrained output)
  Quality model   : qwen2.5:7b-instruct-q4_K_M  (~4.7GB RAM)
  Fast/dev model  : qwen2.5:3b-instruct          (~2.5GB RAM; use if 7B >60s/call)
LLM interface     : LLMBackend Protocol (one Ollama impl; hosted model = config swap)
Response cache    : llm_cache.py (SHA-256 keyed; mandatory for CPU; makes CI offline)
Diagnosis judge   : local LLM via LLMBackend Protocol
                    optional: sentence-transformers MiniLM as secondary similarity signal
PII redaction     : stdlib regex (best-effort)   # Presidio deferred
Report rendering  : Jinja2 → Markdown
Storage           : local filesystem; retention covers derived artifacts
RAM strategy      : load models sequentially (Ollama load/unload); never co-resident
                    Whisper + 7B LLM + embeddings

REMOVED vs V1: scispacy, en_core_sci_sm, simple_icd_10, pytesseract/Tesseract,
               pdf2image/poppler, Helsinki-NLP/transformers, presidio-analyzer,
               Anthropic SDK, FastAPI(→optional stretch)
```

---

## 6. Directory
```
claim_verifier/
├── cli.py                      # primary entrypoint (Typer)
├── config.py                   # schema, weights, thresholds, model id
├── models.py                   # Pydantic: FactSet, VerificationResult
├── pipeline.py                 # orchestrates stages 1–5, error contract
├── redaction.py                # best-effort regex PII
├── llm_cache.py                # hash-keyed response cache (offline CI)
├── stages/
│   ├── ingestion.py            # Whisper / --transcript + pdfplumber text-layer
│   ├── extraction.py           # Claude structured tool-use → FactSet (+validate/retry)
│   ├── normalization.py        # dates / amounts / hospital (no ICD)
│   ├── verification.py         # matchers + diagnosis judge + scoring
│   └── reporting.py            # Jinja2 → Markdown
├── judge.py                    # diagnosis-equivalence LLM judge (Protocol + impl)
├── schema/verification_schema.json
├── templates/report.md.j2
├── data_gen/ (generate_claims.py, generate_bill_pdf.py)
├── eval/ (run_eval.py, metrics.py, holdout/)
├── tests/ (test_normalization.py, test_verification.py, test_extraction.py,
│           test_pipeline.py, test_metrics.py, fixtures/)
├── requirements.txt
└── README.md                   # incl. Production Roadmap (deferred list)
```
