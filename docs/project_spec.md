# Medical Claim Verification Assistant — Claude Code Planning Spec

> This document is the authoritative system specification for implementation.
> It supersedes the original proposal and closes all identified design gaps.
> Claude Code should use this as the full planning input.

---

## 0. One-Line Mission

Given a phone call recording and one or more medical documents, produce a structured, cited,
confidence-scored consistency report that tells a human reviewer exactly where the claimant's
statements match, conflict with, or are unsupported by the submitted evidence.

---

## 1. What Is and Is Not Being Built

### In Scope (MVP)
- Audio transcription of claimant phone calls
- OCR + text extraction from one document type: **hospital bills (PDF or image)**
- Structured fact extraction from both sources with per-fact confidence scores
- Cross-verification of extracted facts against a fixed schema
- A weighted consistency score computed from a defined formula
- A human-readable report with source citations for every finding
- An evaluation harness using synthetic claims to benchmark the pipeline

### Explicitly Out of Scope (do not build, do not scaffold)
- Agentic document requests (requesting missing files automatically)
- Fraud risk scoring or fraud classification
- X-ray, lab report, or prescription analysis
- Discharge summary parsing (Phase 2)
- Real-time streaming or live call analysis
- Any external database lookups (hospital registries, fraud blacklists)

---

## 2. System Architecture

The system has five strictly separated modules. Each module has its own input contract,
output contract, and accuracy metric. Errors must not silently cross module boundaries.

```
┌─────────────────────────────────────────────────────────┐
│  MODULE 1: Ingestion & Pre-processing                   │
│  Input : audio file + document file(s)                  │
│  Output: raw transcript text + raw document text        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  MODULE 2: Fact Extraction (runs independently          │
│            on transcript AND documents)                  │
│  Input : raw text from one source                       │
│  Output: FactSet JSON with confidence scores            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  MODULE 3: Fact Normalization                           │
│  Input : raw FactSet JSON                               │
│  Output: normalized FactSet (dates → ISO, amounts →     │
│          integers, hospital names → canonical form)      │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  MODULE 4: Cross-Verification Engine                    │
│  Input : normalized FactSet from transcript +           │
│          normalized FactSet from documents              │
│  Output: VerificationResult JSON (field-by-field        │
│          verdict + weighted consistency score)          │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  MODULE 5: Report Generation                            │
│  Input : VerificationResult JSON                        │
│  Output: human-readable HTML or Markdown report         │
│          with citations, verdicts, and risk flags       │
└─────────────────────────────────────────────────────────┘
```

Modules 2 and 3 run **twice**: once for the transcript, once for each document.
The outputs are two independent FactSets that are only combined in Module 4.

---

## 3. Verification Schema (Fixed Field Ontology)

This schema defines exactly what fields the system extracts and verifies.
All weights must sum to 1.0.

```json
{
  "schema_version": "1.0",
  "fields": [
    {
      "id": "hospital_name",
      "label": "Hospital / Facility Name",
      "weight": 0.20,
      "match_type": "fuzzy_string",
      "match_tolerance": 0.85,
      "notes": "Use token-level fuzzy match. 'Apollo Hospitals' == 'Apollo Hospital'. Abbreviations must expand before comparison."
    },
    {
      "id": "admission_date",
      "label": "Admission Date",
      "weight": 0.20,
      "match_type": "date",
      "match_tolerance_days": 1,
      "notes": "ISO 8601 after normalization. ±1 day tolerance to account for midnight admissions."
    },
    {
      "id": "discharge_date",
      "label": "Discharge Date",
      "weight": 0.10,
      "match_type": "date",
      "match_tolerance_days": 1,
      "notes": "Optional in MVP — only verified if present in both sources."
    },
    {
      "id": "diagnosis",
      "label": "Primary Diagnosis",
      "weight": 0.20,
      "match_type": "medical_semantic",
      "notes": "Do not use raw string match. Map to ICD-10 category before comparing. 'Appendicitis' and 'Appendectomy' are different categories and must NOT match."
    },
    {
      "id": "billed_amount",
      "label": "Total Billed Amount",
      "weight": 0.20,
      "match_type": "numeric",
      "match_tolerance_pct": 5,
      "notes": "Compare in same currency. ±5% tolerance. Claimant often rounds; bill is authoritative."
    },
    {
      "id": "length_of_stay",
      "label": "Duration of Admission (days)",
      "weight": 0.10,
      "match_type": "numeric",
      "match_tolerance_abs": 1,
      "notes": "Derived from admission/discharge dates if not stated explicitly."
    }
  ]
}
```

### Consistency Score Formula

```
For each field f:
  if field present in both sources:
    field_score(f) = match_result(f) × confidence_transcript(f) × confidence_document(f)
  if field missing from one source:
    field_score(f) = 0   (counts as unverified, flagged separately)
  if field missing from both sources:
    field is excluded from score denominator

consistency_score = Σ(weight(f) × field_score(f)) / Σ(weight(f) for included fields)
```

This produces a score in [0.0, 1.0]. Convert to percentage for display.
A score below 0.70 should be flagged as HIGH RISK for human review.
A score of 0.70–0.89 is MEDIUM. Above 0.90 is LOW.

---

## 4. Module Specifications

### Module 1: Ingestion & Pre-processing

**Audio transcription:**
- Use **OpenAI Whisper** (open source, local) as the default STT engine.
- Model size: `medium` for development, `large-v3` for production evaluation.
- Output: raw transcript text + per-segment timestamps + detected language code.
- If detected language is not English: flag document as `requires_translation = true`
  and pass through a translation step (use Helsinki-NLP/opus-mt models or call
  an external translation API). Do not silently proceed with non-English text.

**Document OCR:**
- Accept input formats: PDF (text-layer or scanned), JPEG, PNG.
- Step 1: Attempt to extract embedded text layer from PDF using `pdfplumber`.
  If extracted text length > 100 characters, skip OCR.
- Step 2: If no text layer, rasterize PDF pages to 300 DPI images using `pdf2image`
  and run **Tesseract OCR** (`pytesseract`) with `--oem 3 --psm 6`.
- Handwriting detection: If Tesseract confidence score (mean word confidence) < 60,
  set `document_quality = "low"` and append a flag to the report:
  `"Warning: Possible handwritten content detected. Manual review recommended."`
- Output: raw extracted text + `document_quality` flag (`high` / `medium` / `low`).

---

### Module 2: Fact Extraction

This module runs once per source (transcript, each document) and outputs a FactSet.

**Model:** Use the Anthropic Claude API (`claude-sonnet-4-20250514`).

**System prompt (extraction):**
```
You are a medical claim fact extractor. Extract only the fields defined in the schema
from the provided text. For each field:
- Return the extracted value as found in the text (do not infer or guess).
- Return a confidence score from 0.0 to 1.0 reflecting how clearly the value
  was stated (1.0 = explicitly stated, 0.7 = implied, 0.4 = ambiguous).
- Return a source_quote: the exact phrase from the text that supports the extraction.
  This quote must be verbatim. If you cannot find a supporting quote, set
  confidence to 0.0 and leave source_quote null.
- If a field is not present in the text, return null for value and 0.0 for confidence.
- Do not hallucinate values. Absence of evidence is not evidence of absence.
- Return ONLY valid JSON matching the FactSet schema. No preamble or explanation.
```

**Output schema (FactSet):**
```json
{
  "source_type": "transcript | document",
  "source_id": "string (filename or 'call_recording')",
  "extraction_timestamp": "ISO 8601",
  "facts": {
    "hospital_name": {
      "value": "Apollo Hospital",
      "confidence": 0.95,
      "source_quote": "I was admitted to Apollo Hospital"
    },
    "admission_date": {
      "value": "2025-03-12",
      "confidence": 1.0,
      "source_quote": "on March 12"
    },
    "diagnosis": {
      "value": "Appendicitis",
      "confidence": 0.90,
      "source_quote": "for appendicitis"
    },
    "billed_amount": {
      "value": 50000,
      "confidence": 0.80,
      "source_quote": "around fifty thousand rupees"
    },
    "discharge_date": {
      "value": null,
      "confidence": 0.0,
      "source_quote": null
    },
    "length_of_stay": {
      "value": null,
      "confidence": 0.0,
      "source_quote": null
    }
  }
}
```

**Medical NER layer:**
Before passing text to the LLM, run a lightweight medical NER pass using
`scispacy` with the `en_core_sci_sm` model to identify and tag medical entities
(diseases, procedures, drugs, body parts). Inject the tagged entity list into
the extraction prompt as structured context:

```
Identified medical entities in this text:
- "appendicitis" → DISEASE
- "appendectomy" → PROCEDURE
```

This prevents the LLM from conflating diagnosis with procedure during extraction.

---

### Module 3: Fact Normalization

Applies deterministic transformations to raw FactSet values. No LLM involved here.

**Hospital name:**
- Lowercase, strip punctuation, tokenize.
- Expand known abbreviations: "Hosp." → "Hospital", "Pvt." → "Private", "Ltd." → "Limited".
- Store canonical form for comparison.

**Dates:**
- Use `dateparser` library to parse any date format to ISO 8601 (YYYY-MM-DD).
- If parsing fails, set confidence to 0.0 and flag for human review.
- Do not attempt to infer year from context.

**Amounts:**
- Strip currency symbols (₹, Rs., INR, $).
- Parse Indian number formats (e.g., "1,20,000" → 120000).
- Convert to integer (paise/cents are not material at this scale).

**Diagnosis:**
- Map extracted diagnosis string to an ICD-10 chapter code using a local lookup table
  (use `simple_icd_10` Python package).
- If mapping fails (no match found), keep raw string and set `icd_mapped = false`.
- Two diagnoses are compared at the ICD-10 **category** level (3-character code), not
  the full 7-character code, to allow for specificity differences between sources.

---

### Module 4: Cross-Verification Engine

Takes two normalized FactSets and produces a VerificationResult.

**Field-level match logic:**

```python
def match_field(field_id, fact_a, fact_b, schema):
    if fact_a.value is None or fact_b.value is None:
        return FieldVerdict(status="MISSING", score=0.0, note="Field absent in one source")

    match_type = schema[field_id].match_type

    if match_type == "fuzzy_string":
        ratio = fuzz.token_sort_ratio(fact_a.value, fact_b.value) / 100
        matched = ratio >= schema[field_id].match_tolerance
        return FieldVerdict(status="MATCH" if matched else "MISMATCH", score=ratio)

    if match_type == "date":
        delta = abs((parse(fact_a.value) - parse(fact_b.value)).days)
        matched = delta <= schema[field_id].match_tolerance_days
        return FieldVerdict(status="MATCH" if matched else "MISMATCH",
                            score=1.0 if matched else 0.0,
                            note=f"Delta: {delta} day(s)")

    if match_type == "numeric":
        pct_diff = abs(fact_a.value - fact_b.value) / max(fact_a.value, fact_b.value)
        tolerance = schema[field_id].get("match_tolerance_pct", 0) / 100
        matched = pct_diff <= tolerance
        return FieldVerdict(status="MATCH" if matched else "MISMATCH",
                            score=1.0 - pct_diff,
                            note=f"Difference: {pct_diff*100:.1f}%")

    if match_type == "medical_semantic":
        # Compare at ICD-10 category level (first 3 chars)
        code_a = fact_a.icd_code[:3] if fact_a.icd_mapped else None
        code_b = fact_b.icd_code[:3] if fact_b.icd_mapped else None
        if code_a is None or code_b is None:
            # Fall back to fuzzy string if ICD mapping failed
            ratio = fuzz.token_sort_ratio(fact_a.value, fact_b.value) / 100
            return FieldVerdict(status="MATCH" if ratio >= 0.80 else "MISMATCH",
                                score=ratio, note="ICD mapping failed; used string fallback")
        matched = code_a == code_b
        return FieldVerdict(status="MATCH" if matched else "MISMATCH",
                            score=1.0 if matched else 0.0,
                            note=f"ICD codes: {code_a} vs {code_b}")
```

**Output schema (VerificationResult):**
```json
{
  "claim_id": "string",
  "verified_at": "ISO 8601",
  "consistency_score": 0.87,
  "risk_level": "MEDIUM",
  "field_verdicts": {
    "hospital_name": {
      "status": "MATCH",
      "score": 1.0,
      "transcript_value": "Apollo Hospital",
      "document_value": "Apollo Hospital",
      "transcript_quote": "I was admitted to Apollo Hospital",
      "document_quote": "Apollo Hospital, Jubilee Hills"
    },
    "billed_amount": {
      "status": "MISMATCH",
      "score": 0.19,
      "transcript_value": 50000,
      "document_value": 62000,
      "note": "Difference: 24.0%",
      "transcript_quote": "around fifty thousand rupees",
      "document_quote": "Total Bill Amount: ₹62,000"
    }
  },
  "flags": [
    {
      "type": "AMOUNT_MISMATCH",
      "severity": "HIGH",
      "message": "Claimant stated ₹50,000; bill shows ₹62,000 (24% difference)"
    },
    {
      "type": "LOW_CONFIDENCE_EXTRACTION",
      "severity": "LOW",
      "message": "Billed amount extracted from transcript with confidence 0.80 (stated as approximate)"
    }
  ],
  "low_quality_sources": [],
  "missing_fields": ["discharge_date", "length_of_stay"]
}
```

---

### Module 5: Report Generation

Converts VerificationResult JSON into a human-readable report.

**Format:** Markdown (primary). HTML rendering optional.

**Report structure:**
1. **Summary header** — claim ID, date, overall score, risk level badge
2. **Field-by-field table** — status icon, transcript value, document value, confidence, note
3. **Flags section** — ordered by severity (HIGH first), each with explanation
4. **Missing fields** — list of unverifiable fields and which source they're absent from
5. **Source quality warnings** — OCR confidence issues, translation flags
6. **Reviewer guidance** — templated language based on risk level:
   - LOW: "No significant inconsistencies found. Standard approval workflow applies."
   - MEDIUM: "Minor discrepancies detected. Review flagged fields before decision."
   - HIGH: "Significant inconsistencies detected. Manual review of all flagged fields required."

**Citation format:**
Every value in the report must be followed by its source quote in brackets, e.g.:
`Apollo Hospital [transcript: "I was admitted to Apollo Hospital"]`

---

## 5. Tech Stack

```
Language         : Python 3.11+
STT              : openai-whisper (local)
Translation      : Helsinki-NLP/opus-mt (via HuggingFace transformers, local)
PDF text extract : pdfplumber
PDF rasterization: pdf2image + poppler
OCR              : pytesseract (Tesseract 5.x backend)
Medical NER      : scispacy + en_core_sci_sm
ICD-10 mapping   : simple_icd_10
Fuzzy matching   : rapidfuzz
Date parsing     : dateparser
Fact extraction  : Anthropic Python SDK → claude-sonnet-4-20250514
Report rendering : Jinja2 templates
API layer        : FastAPI (single /verify endpoint)
Storage          : Local filesystem only (no cloud, no DB, for MVP)
```

No external cloud services except the Anthropic API.
All audio and document data stays local to comply with data sensitivity requirements.

---

## 6. Data & Evaluation Strategy

No real patient data is used during development.

### Synthetic Data Generation

Build a `data_gen/` module that generates synthetic evaluation cases:

```
SyntheticClaim = {
    transcript: LLM-generated call transcript with known facts,
    document: programmatically generated fake hospital bill PDF,
    ground_truth: exact fact values injected during generation,
    injected_errors: list of deliberate mismatches with type labels
}
```

Error types to inject:
- `AMOUNT_MISMATCH` — transcript amount differs from bill by 10–40%
- `DATE_MISMATCH` — admission date off by 3–15 days
- `HOSPITAL_MISMATCH` — different hospital name
- `DIAGNOSIS_MISMATCH` — different ICD-10 category
- `MISSING_FIELD` — a field present in transcript, absent from document
- `CLEAN` — no errors (true negative cases, must be ~30% of eval set)

Target: generate 200 synthetic cases (140 with errors, 60 clean).

### Evaluation Metrics

All metrics measured against synthetic ground truth:

| Metric | Target (hypothesis) | Measured on |
|--------|-------------------|-------------|
| Fact extraction precision | >90% | Per-field on 200 cases |
| Fact extraction recall | >85% | Per-field on 200 cases |
| Contradiction detection precision | >88% | Per-flag on 140 error cases |
| Contradiction detection recall | >83% | Per-flag on 140 error cases |
| False positive rate (clean cases) | <12% | 60 clean cases |

These are **hypotheses to validate**, not guarantees. The eval harness must produce
a report showing actual measured values against these targets.

---

## 7. Data Compliance Notes

This system handles personally identifiable health information. The following must be
implemented before any real data touches the system:

- All audio files and documents must be deleted from local storage immediately after
  processing (configurable retention window, default: 0 seconds after report generation).
- No audio, text, or extracted facts should be sent to any external service except
  the Anthropic API for the extraction step. Anthropic's data processing agreement
  must be reviewed before production use.
- PII redaction: Before sending transcript text to the Anthropic API, run a PII
  redaction pass using `presidio-analyzer` to mask names, phone numbers, and
  Aadhaar/PAN numbers. Replace with `[REDACTED_NAME]`, `[REDACTED_PHONE]` etc.
- Log only claim IDs and verification scores. Never log extracted fact values or
  source quotes in application logs.
- Implement at minimum: input file size limits, file type validation,
  and process isolation per claim.

---

## 8. Project Directory Structure

```
claim_verifier/
├── main.py                    # FastAPI app, /verify endpoint
├── config.py                  # Schema, weights, thresholds, model config
├── modules/
│   ├── ingestion.py           # Module 1: STT + OCR
│   ├── extraction.py          # Module 2: Fact extraction via Claude API
│   ├── normalization.py       # Module 3: Deterministic normalization
│   ├── verification.py        # Module 4: Cross-verification engine
│   └── reporting.py           # Module 5: Report generation
├── schema/
│   └── verification_schema.json   # Field ontology (Section 3)
├── templates/
│   └── report.md.j2           # Jinja2 report template
├── data_gen/
│   ├── generate_claims.py     # Synthetic claim generator
│   └── generate_bill_pdf.py   # Fake hospital bill PDF generator
├── eval/
│   ├── run_eval.py            # Evaluation harness
│   └── metrics.py             # Precision/recall calculation
├── tests/
│   ├── test_extraction.py
│   ├── test_normalization.py
│   ├── test_verification.py
│   └── fixtures/              # Static test inputs
├── requirements.txt
└── README.md
```

---

## 9. API Contract

Single endpoint for MVP:

```
POST /verify
Content-Type: multipart/form-data

Fields:
  claim_id     : string (required)
  audio        : file (required, .wav or .mp3, max 50MB)
  document     : file (required, .pdf or .jpg or .png, max 20MB)

Response 200:
  Content-Type: application/json
  Body: VerificationResult JSON (Section 4, Module 4 output schema)

Response 422:
  Validation error (missing fields, unsupported file type)

Response 500:
  Internal error with module identifier (e.g., "extraction_failed")
```

---

## 10. Build Order for Claude Code

Build in this sequence. Each step must be independently testable before the next begins.

1. `config.py` — schema, weights, thresholds
2. `modules/ingestion.py` — STT (Whisper) + OCR (pdfplumber + Tesseract), with quality flags
3. `modules/extraction.py` — Claude API call, FactSet output, confidence scores, source quotes
4. `modules/normalization.py` — date, amount, hospital name, ICD-10 normalization
5. `modules/verification.py` — field-by-field matching, consistency score, VerificationResult
6. `modules/reporting.py` — Jinja2 report from VerificationResult
7. `data_gen/` — synthetic claim + bill PDF generator
8. `eval/` — evaluation harness against synthetic ground truth
9. `main.py` — FastAPI wrapper
10. `tests/` — unit tests for modules 3 and 4 (deterministic, no API calls needed)