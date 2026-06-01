"""
T1.6 / T1.7 — Verify Ollama is running and pull required models.

Usage (from repo root):
    python -m claim_verifier.scripts.check_ollama

Exit 0: all good.
Exit 1: Ollama not running or a model failed to pull.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request

from claim_verifier.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_FAST,
    OLLAMA_MODEL_QUALITY,
)

REQUIRED_MODELS = [OLLAMA_MODEL_QUALITY, OLLAMA_MODEL_FAST]


def _get(path: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def check_running() -> bool:
    try:
        _get("/api/tags")
        return True
    except Exception:
        return False


def list_models() -> list[str]:
    data = _get("/api/tags")
    return [m["name"] for m in data.get("models", [])]


def pull_model(name: str) -> None:
    print(f"  Pulling {name} ... (this may take several minutes on first run)")
    result = subprocess.run(["ollama", "pull", name])
    if result.returncode != 0:
        print(f"  ERROR: failed to pull {name}")
        sys.exit(1)


def main() -> None:
    print("=== T1.6 — Ollama server check ===")
    if not check_running():
        print(
            "FAIL: Ollama is not running.\n"
            "Start it with:  ollama serve\n"
            "Install from:   https://ollama.com"
        )
        sys.exit(1)
    print(f"  OK: Ollama is running at {OLLAMA_BASE_URL}")

    print("\n=== T1.7 — Model availability check ===")
    available = list_models()
    for model in REQUIRED_MODELS:
        if any(model in m for m in available):
            print(f"  OK: {model}")
        else:
            print(f"  MISSING: {model}")
            pull_model(model)

    print("\nAll models ready. Run the benchmark next:")
    print("  python -m claim_verifier.scripts.benchmark_llm")


if __name__ == "__main__":
    main()
