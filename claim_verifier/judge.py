from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from claim_verifier.models import FactValue

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
You are a medical diagnosis equivalence judge. Compare two diagnosis descriptions.

Rules:
- MATCH: both refer to the same disease, or both to the same procedure, even if phrased differently.
  Synonyms, abbreviations, and severity qualifiers (acute/chronic) are acceptable aliases.
- MISMATCH: they describe different conditions, or one is a disease and the other is a procedure
  (e.g. "appendicitis" vs "appendectomy" is a MISMATCH — disease vs procedure).

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

    def compare(
        self,
        transcript_fv: FactValue,
        document_fv: FactValue,
    ) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Diagnosis A (transcript): {transcript_fv.value!r}\n"
                    f"Entity type A: {transcript_fv.entity_type}\n\n"
                    f"Diagnosis B (document): {document_fv.value!r}\n"
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
