from __future__ import annotations

from typing import Protocol, runtime_checkable

from claim_verifier.models import FactValue


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
