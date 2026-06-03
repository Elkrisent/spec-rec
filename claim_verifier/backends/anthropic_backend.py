"""
AnthropicBackend — calls the Claude API for LLM extraction (T10.3).

Used when BACKEND_TYPE=anthropic env var is set.
Requires ANTHROPIC_API_KEY in the environment.

Uses tool use for JSON-schema-constrained output (Claude's equivalent of
Ollama's format param). Responses are cached with the same SHA-256 scheme
used by OllamaBackend.
"""

from __future__ import annotations

import os

from claim_verifier.config import LLM_CACHE_DIR
from claim_verifier.llm_cache import LLMCache


class AnthropicBackend:
    """Anthropic Claude backend. Satisfies LLMBackend Protocol."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "anthropic package is not installed. Run: uv pip install anthropic"
            ) from exc

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._cache = cache if cache is not None else LLMCache(LLM_CACHE_DIR)

    def complete(self, messages: list[dict], schema: dict | None = None) -> dict:
        """Send messages to Claude; return a schema-valid dict. Cache-first."""
        cache_kwargs = {}
        if schema is not None:
            cache_kwargs["schema"] = schema

        hit = self._cache.get(self._model, messages, **cache_kwargs)
        if hit is not None:
            return hit  # stored as the result dict directly

        result = self._call_claude(messages, schema)
        self._cache.put(self._model, messages, result, **cache_kwargs)
        return result

    def _call_claude(self, messages: list[dict], schema: dict | None) -> dict:
        system = ""
        user_messages: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)

        if not user_messages:
            raise ValueError("No user messages provided to AnthropicBackend")

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 2048,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system

        if schema:
            kwargs["tools"] = [
                {
                    "name": "structured_output",
                    "description": "Return the structured extraction result.",
                    "input_schema": schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}
            response = self._client.messages.create(**kwargs)
            for block in response.content:
                if block.type == "tool_use":
                    return dict(block.input)
            raise RuntimeError("Claude did not return a tool_use block")
        else:
            response = self._client.messages.create(**kwargs)
            text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {"text": text}
