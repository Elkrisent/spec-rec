from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rapidfuzz import fuzz

from claim_verifier.models import FactValue

# ---------------------------------------------------------------------------
# Medical abbreviation expansion (applied before LLM comparison)
# ---------------------------------------------------------------------------

# Each entry: (compiled pattern, replacement).  Applied in order; expansions
# are wrapped in parentheses so the original term is preserved alongside the
# plain-English gloss (e.g. "MC (metacarpal)" instead of just "metacarpal").
_ABBREV_TABLE: list[tuple[re.Pattern, str]] = [
    # "both bone leg left" = both bones (tibia+fibula) of the left leg
    (re.compile(r"\bboth\s+bone\b", re.IGNORECASE), "both bones of the"),
    # MC = metacarpal (hand bone)
    (re.compile(r"\bMC\b"), "metacarpal (hand bone)"),
    # Common fracture / orthopaedic shorthand
    (re.compile(r"\bFx\b", re.IGNORECASE),  "fracture"),
    (re.compile(r"\bDx\b", re.IGNORECASE),  "diagnosis"),
    (re.compile(r"\bHx\b", re.IGNORECASE),  "history"),
    (re.compile(r"\bSOB\b", re.IGNORECASE), "shortness of breath"),
    (re.compile(r"\bMI\b"),                 "myocardial infarction (heart attack)"),
    (re.compile(r"\bCVA\b"),                "stroke (cerebrovascular accident)"),
    (re.compile(r"\bDVT\b"),                "deep vein thrombosis (blood clot)"),
]


def _expand_abbrevs(text: str) -> str:
    """Expand common medical abbreviations so the LLM sees plain language."""
    for pattern, replacement in _ABBREV_TABLE:
        text = pattern.sub(replacement, text)
    return text

if TYPE_CHECKING:
    from claim_verifier.backends import LLMBackend


@runtime_checkable
class DiagnosisJudge(Protocol):
    """
    Compares two diagnosis FactValues and returns a verdict.

    W3: fulfilled by StubJudge.
    W5: fulfilled by the real LLM-backed judge (same Protocol).
    """

    def compare(
        self,
        transcript_fv: FactValue,
        document_fv: FactValue,
    ) -> tuple[str, str]:
        """Return (verdict, note). verdict is 'MATCH' or 'MISMATCH'."""
        ...


class StubJudge:
    """
    Fixed-verdict stub for W3 tests.

    Configurable so test scenarios can inject either MATCH or MISMATCH
    without needing an Ollama server.
    """

    def __init__(
        self,
        verdict: str = "MISMATCH",
        note: str = "stub: fixed verdict",
    ) -> None:
        self._verdict = verdict
        self._note = note

    def compare(
        self,
        transcript_fv: FactValue,
        document_fv: FactValue,
    ) -> tuple[str, str]:
        return self._verdict, self._note


# ---------------------------------------------------------------------------
# T5.6 — Real LLM-backed diagnosis judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a medical diagnosis equivalence judge. One description comes from a patient's verbal \
transcript (lay language); the other from a formal medical document (clinical language).

Rules:
- MATCH: both describe the same underlying injury or condition, even if phrased very differently.
  - Lay terms are equivalent to their clinical counterparts:
      "broke my leg" = "fracture of leg/tibia/fibula"
      "broken arm / cracked bone" = "fracture"
      "heart attack" = "myocardial infarction"
      "blood clot" = "thrombosis / embolism"
      "burst appendix" = "perforated appendicitis"
  - A patient describing the same injury in simpler or partial terms still MATCHES the full
    clinical description (e.g. "broke my left leg and hurt my hand" matches
    "fracture both bone leg left, fracture 2nd-4th metacarpal right").
  - Synonyms, abbreviations, and severity qualifiers (acute/chronic) are acceptable aliases.
- MISMATCH: they describe clearly different conditions, or one is a disease and the other is a
  procedure for a different condition (e.g. "appendicitis" vs "appendectomy" is MISMATCH).

Return only valid JSON with keys "verdict" ("MATCH" or "MISMATCH") and "rationale" (one sentence).
No explanation outside the JSON.\
"""

JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict":   {"type": "string", "enum": ["MATCH", "MISMATCH"]},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
}


class LLMJudge:
    """
    LLM-backed diagnosis equivalence judge (T5.6).

    Implements DiagnosisJudge Protocol. Replaces StubJudge in production.
    """

    def __init__(self, backend: "LLMBackend") -> None:
        self._backend = backend

    # token_sort_ratio >= this → skip the LLM call and return MATCH immediately
    _FASTPATH_THRESHOLD = 0.95

    def compare(
        self,
        transcript_fv: FactValue,
        document_fv: FactValue,
    ) -> tuple[str, str]:
        a = str(transcript_fv.value or "").strip()
        b = str(document_fv.value or "").strip()
        if fuzz.token_sort_ratio(a, b) / 100.0 >= self._FASTPATH_THRESHOLD:
            return "MATCH", "fast match: near-identical diagnosis strings"

        # Expand abbreviations so the model sees plain language
        a_exp = _expand_abbrevs(a)
        b_exp = _expand_abbrevs(b)

        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Diagnosis A (transcript): {a_exp!r}\n"
                    f"Entity type A: {transcript_fv.entity_type}\n\n"
                    f"Diagnosis B (document): {b_exp!r}\n"
                    f"Entity type B: {document_fv.entity_type}\n\n"
                    "Are these the same medical condition?"
                ),
            },
        ]
        raw = self._backend.complete(messages, schema=JUDGE_SCHEMA)
        verdict = str(raw.get("verdict", "MISMATCH")).upper()
        if verdict not in ("MATCH", "MISMATCH"):
            verdict = "MISMATCH"
        rationale = str(raw.get("rationale", "no rationale provided"))
        return verdict, f"LLM judge: {rationale}"
