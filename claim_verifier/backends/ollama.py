"""
Ollama REST backend with SHA-256-keyed response cache (T5.2).

Cache-first: if a response exists for (model, messages, format_schema),
it is returned without calling the Ollama server. This makes CI and dev
iteration fully offline once the cache is populated.

Wire format: POST /api/chat with stream=false, format=schema|"json".
The full response dict is stored in the cache; content is parsed on read.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from claim_verifier.config import (
    LLM_CACHE_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
)
from claim_verifier.llm_cache import LLMCache


class OllamaBackend:
    """Ollama chat backend. Satisfies LLMBackend Protocol."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        cache: LLMCache | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._cache = cache if cache is not None else LLMCache(LLM_CACHE_DIR)

    def complete(
        self,
        messages: list[dict],
        schema: dict | None = None,
    ) -> dict:
        """Return parsed JSON dict. Cache-first; calls Ollama on cache miss."""
        cache_kwargs: dict[str, Any] = {}
        if schema is not None:
            cache_kwargs["format"] = schema

        hit = self._cache.get(self._model, messages, **cache_kwargs)
        if hit is not None:
            return _parse_content(hit)

        response = self._call_ollama(messages, schema)
        self._cache.put(self._model, messages, response, **cache_kwargs)
        return _parse_content(response)

    def _call_ollama(self, messages: list[dict], schema: dict | None) -> dict:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": OLLAMA_TEMPERATURE},
            "format": schema if schema is not None else "json",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Ollama not reachable at {self._base_url}: {exc}"
            ) from exc


def _parse_content(response: dict) -> dict:
    """Extract and parse the JSON string from the assistant message content."""
    content: str = response["message"]["content"]
    return json.loads(content)
