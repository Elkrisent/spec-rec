"""
SHA-256-keyed response cache for local LLM calls.

Why this is mandatory: CPU inference runs 15–60s per call. The cache makes:
  - CI fully offline (no Ollama server required once responses are cached)
  - Eval reproducible across runs
  - Dev iteration fast (~10-case working set)

Cache entries are keyed by (model, messages, extra kwargs) so different prompts
never collide and model changes automatically bypass stale entries.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


class LLMCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, messages: list[dict], **kwargs: Any) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, **kwargs},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, model: str, messages: list[dict], **kwargs: Any) -> dict | None:
        path = self._path(self._key(model, messages, **kwargs))
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, model: str, messages: list[dict], response: dict, **kwargs: Any) -> None:
        path = self._path(self._key(model, messages, **kwargs))
        path.write_text(json.dumps(response, indent=2, ensure_ascii=True), encoding="utf-8")

    def has(self, model: str, messages: list[dict], **kwargs: Any) -> bool:
        return self._path(self._key(model, messages, **kwargs)).exists()
