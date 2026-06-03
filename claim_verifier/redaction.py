"""
Best-effort regex PII redaction (T6.2).

Applied to a copy of the source text before any LLM call — keeps PII
off the local model as a defence-in-depth measure.  The original text
is kept intact for quote-substring verification after extraction.

Patterns masked (in application order):
  email     →  <EMAIL>
  PAN       →  <PAN>      (ABCDE1234F format)
  Aadhaar   →  <AADHAAR>  (formatted XXXX XXXX XXXX / XXXX-XXXX-XXXX)
  phone     →  <PHONE>    (Indian 10-digit, with optional +91/0 prefix)
  Aadhaar   →  <AADHAAR>  (plain 12 consecutive digits, catch-all)

All processing is local — no data leaves the machine.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Email — applied first; @ symbol makes it unambiguous
    (
        re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'),
        "<EMAIL>",
    ),
    # PAN: exactly 5 uppercase letters, 4 digits, 1 uppercase letter
    (
        re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b'),
        "<PAN>",
    ),
    # Aadhaar (formatted): XXXX XXXX XXXX or XXXX-XXXX-XXXX
    (
        re.compile(r'\b\d{4}[\s\-]\d{4}[\s\-]\d{4}\b'),
        "<AADHAAR>",
    ),
    # Indian mobile with +91 prefix (no space between prefix and number)
    (
        re.compile(r'\+91[\s\-]?[6-9]\d{9}\b'),
        "<PHONE>",
    ),
    # Indian mobile with 0 STD prefix
    (
        re.compile(r'\b0[6-9]\d{9}\b'),
        "<PHONE>",
    ),
    # Indian mobile — bare 10 digits starting with 6-9
    (
        re.compile(r'\b[6-9]\d{9}\b'),
        "<PHONE>",
    ),
    # Aadhaar (plain 12 digits) — after phone to avoid masking phone+country-code
    (
        re.compile(r'\b\d{12}\b'),
        "<AADHAAR>",
    ),
]


def redact(text: str) -> str:
    """Return a copy of *text* with PII replaced by pattern labels."""
    for pattern, label in _PATTERNS:
        text = pattern.sub(label, text)
    return text
