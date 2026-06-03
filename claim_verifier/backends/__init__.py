"""
LLM backend abstraction (T5.1).

LLMBackend is a runtime-checkable Protocol. The real implementation
(OllamaBackend) lives in backends/ollama.py. StubBackend is used in tests.

complete() contract:
  - Input : OpenAI-style messages + optional JSON schema for the format param.
  - Output: parsed JSON dict (never a raw string, never the full Ollama response).
  - The backend owns caching; callers do not cache.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    def complete(
        self,
        messages: list[dict],
        schema: dict | None = None,
    ) -> dict:
        """
        Send messages to the LLM and return the parsed JSON response dict.

        schema: JSON schema passed to Ollama's format param to constrain output.
                None falls back to generic JSON mode ("format": "json").
        """
        ...


class StubBackend:
    """
    Configurable fixed-response stub for tests.

    Pass response= to control what complete() returns.
    call_count and calls let tests verify interaction without mocks.
    """

    def __init__(self, response: dict) -> None:
        self._response = response
        self.call_count: int = 0
        self.calls: list[tuple[list[dict], dict | None]] = []

    def complete(
        self,
        messages: list[dict],
        schema: dict | None = None,
    ) -> dict:
        self.call_count += 1
        self.calls.append((messages, schema))
        return self._response
