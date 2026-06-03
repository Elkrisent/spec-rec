"""
Stage 3 — Normalization.

Converts raw FactSet values into canonical forms:
  - Dates   : dateparser DMY-first; year-absent ⇒ "--MM-DD" marker; no year inference
  - Amounts : strip ₹/Rs/INR/$; Indian comma grouping ⇒ int
  - Hospital: lowercase + strip + abbreviation expansion
  - Diagnosis: passthrough (entity_type preserved)
  - LOS     : string → int if needed

Returns (normalized_FactSet, list[Flag]).  Flags are emitted for unparseable values;
those fields get value=None and confidence=0.0 (not dropped).
"""

from __future__ import annotations

import re
from datetime import date as _date
from typing import Optional, Union

import dateparser

from claim_verifier.models import FactSet, FactValue, Flag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefix used to mark dates where no year was present in the source string.
# "--MM-DD" is deliberately not a valid ISO date so it can never be confused
# with a year-present date.
YEAR_ABSENT_PREFIX = "--"

# Detect an explicit 4-digit calendar year in a date string.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Fast-path: YYYY-MM-DD or YYYY-M-D (already ISO; dateparser mis-applies DMY ordering to these).
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")

# Strip leading currency symbols/codes (case-insensitive; handles ₹, Rs, Rs., INR, $).
_CURRENCY_STRIP_RE = re.compile(r"^(?:₹|Rs\.?|INR|\$)\s*", re.IGNORECASE)

# Validate the remaining numeric portion: digits with optional Indian commas,
# then optional 1–2 decimal places.  E.g. "1,20,000" or "62,000.50".
_NUMBER_RE = re.compile(r"^([0-9][0-9,]*)(?:\.\d{1,2})?$")

# Hospital abbreviation map (token → expansion).  Both "hosp" and "hosp." are
# included so the lookup works after lowercasing a period-ending token.
HOSPITAL_ABBREVS: dict[str, str] = {
    "hosp": "hospital",
    "hosp.": "hospital",
    "med": "medical",
    "med.": "medical",
    "ctr": "center",
    "ctr.": "center",
    "inst": "institute",
    "inst.": "institute",
    "gen": "general",
    "gen.": "general",
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "dr": "doctor",
    "dr.": "doctor",
    "st": "saint",
    "st.": "saint",
    "dept": "department",
    "dept.": "department",
    "natl": "national",
    "natl.": "national",
    "intl": "international",
    "intl.": "international",
    "univ": "university",
    "univ.": "university",
}

# dateparser settings shared by all date parsing calls.
_DATEPARSER_SETTINGS: dict = {
    "DATE_ORDER": "DMY",
    "PREFER_DAY_OF_MONTH": "first",
    "RETURN_AS_TIMEZONE_AWARE": False,
    "PREFER_LOCALE_DATE_ORDER": False,
}

# ---------------------------------------------------------------------------
# Internal normalizers
# ---------------------------------------------------------------------------

def _normalize_date(
    raw_value: str,
    field_id: str,
    source_id: str,
) -> tuple[Optional[str], Optional[Flag]]:
    """Parse a date string.  Returns (canonical_string, flag_or_None)."""
    # Fast path: already ISO (YYYY-MM-DD). dateparser with DATE_ORDER='DMY' would
    # misinterpret the trailing two parts as D/M, so handle this format directly.
    if m := _ISO_DATE_RE.match(raw_value.strip()):
        try:
            d = _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return d.strftime("%Y-%m-%d"), None
        except ValueError:
            pass  # fall through to dateparser for its error handling

    has_year = bool(_YEAR_RE.search(raw_value))

    parsed = dateparser.parse(raw_value, settings=_DATEPARSER_SETTINGS)
    if parsed is None:
        return None, Flag(
            type="UNPARSEABLE_DATE",
            severity="MEDIUM",
            message=(
                f"source '{source_id}' field '{field_id}': "
                f"cannot parse date value '{raw_value}'"
            ),
        )

    if has_year:
        return parsed.strftime("%Y-%m-%d"), None
    # Year absent: use "--MM-DD" marker — do NOT infer the current year.
    return YEAR_ABSENT_PREFIX + parsed.strftime("%m-%d"), None


def _normalize_amount(
    raw_value: Union[str, int, float],
    field_id: str,
    source_id: str,
) -> tuple[Optional[int], Optional[Flag]]:
    """Parse a currency/amount value to int (paise truncated)."""
    if isinstance(raw_value, (int, float)):
        return int(raw_value), None

    text = str(raw_value).strip()
    text = _CURRENCY_STRIP_RE.sub("", text)  # strip ₹ / Rs. / INR / $

    m = _NUMBER_RE.match(text)
    if not m:
        return None, Flag(
            type="UNPARSEABLE_AMOUNT",
            severity="MEDIUM",
            message=(
                f"source '{source_id}' field '{field_id}': "
                f"cannot parse amount value '{raw_value}'"
            ),
        )

    # Remove commas — valid for both Indian (1,20,000) and Western (120,000) grouping.
    digits = m.group(1).replace(",", "")
    return int(digits), None


def _normalize_los(
    raw_value: Union[str, int, float],
    field_id: str,
    source_id: str,
) -> tuple[Optional[int], Optional[Flag]]:
    """Parse length-of-stay to int days.  Handles 'N', 'N days', 'N nights'."""
    if isinstance(raw_value, (int, float)):
        return int(raw_value), None

    text = str(raw_value).strip()
    m = re.match(r"^(\d+)(?:\s+(?:day|night)s?)?$", text, re.IGNORECASE)
    if not m:
        return None, Flag(
            type="UNPARSEABLE_LOS",
            severity="LOW",
            message=(
                f"source '{source_id}' field '{field_id}': "
                f"cannot parse LOS value '{raw_value}'"
            ),
        )
    return int(m.group(1)), None


def _normalize_hospital(raw_value: str) -> str:
    """Canonical hospital name: lowercase → strip → expand abbreviations."""
    tokens = raw_value.lower().strip().split()
    return " ".join(HOSPITAL_ABBREVS.get(tok, tok) for tok in tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(raw: FactSet) -> tuple[FactSet, list[Flag]]:
    """
    Normalize all fields in *raw* and return (normalized_factset, flags).

    Fields with unparseable values get value=None and confidence=0.0 and emit
    a Flag — they are kept in the FactSet (not dropped) so downstream stages
    can treat them as MISSING.
    """
    flags: list[Flag] = []
    normalized_facts: dict[str, FactValue] = {}

    for field_id, fv in raw.facts.items():
        # Null values are already "missing" — pass through without touching confidence.
        if fv.value is None:
            normalized_facts[field_id] = fv
            continue

        new_value: Optional[Union[str, int, float]] = fv.value
        new_confidence = fv.confidence
        flag: Optional[Flag] = None

        if field_id in ("admission_date", "discharge_date"):
            new_value, flag = _normalize_date(str(fv.value), field_id, raw.source_id)

        elif field_id == "billed_amount":
            new_value, flag = _normalize_amount(fv.value, field_id, raw.source_id)

        elif field_id == "length_of_stay":
            new_value, flag = _normalize_los(fv.value, field_id, raw.source_id)

        elif field_id == "hospital_name":
            new_value = _normalize_hospital(str(fv.value))

        # "diagnosis" and any unknown fields: passthrough — value + entity_type unchanged.

        if flag is not None:
            flags.append(flag)
            new_value = None
            new_confidence = 0.0

        normalized_facts[field_id] = FactValue(
            value=new_value,
            confidence=new_confidence,
            source_quote=fv.source_quote,
            entity_type=fv.entity_type,
            quote_verified=fv.quote_verified,
        )

    return (
        FactSet(
            source_type=raw.source_type,
            source_id=raw.source_id,
            extraction_timestamp=raw.extraction_timestamp,
            facts=normalized_facts,
        ),
        flags,
    )
