"""
T1.9 — Feasibility benchmark: valid-JSON rate, avg latency, RAM note.

Outputs a go/no-go: use quality (7B) or fall back to fast (3B).
Decision rule: quality model if valid-JSON >= 80% AND avg latency <= 60s; else fast.

Usage (from repo root):
    python -m claim_verifier.scripts.benchmark_llm

Note: RAM usage must be checked manually with `htop` or `free -h` during inference.
Target: >= 2 GB free after model loads.
"""

import json
import sys
import time
import urllib.error
import urllib.request

from claim_verifier.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_FAST,
    OLLAMA_MODEL_QUALITY,
)

N_TRIALS = 5

# A realistic extraction-like prompt — same structure as the real extraction call.
_PROMPT = (
    "Extract facts from the text below and return ONLY valid JSON with this exact schema:\n"
    '{"hospital": "<string>", "amount": <number>, "date": "<string>"}\n\n'
    "Text: \"I was admitted to City Hospital on March 12 and the total bill was 45000 rupees.\""
)
_EXPECTED_KEYS = {"hospital", "amount", "date"}


def _call_ollama(model: str, prompt: str) -> tuple[str, float]:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "format": "json",
         "options": {"temperature": 0}}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data.get("response", ""), time.time() - t0


def _benchmark(model: str, label: str) -> dict:
    print(f"\n  [{label}] {model}")
    valid = 0
    times: list[float] = []
    for i in range(N_TRIALS):
        try:
            response, elapsed = _call_ollama(model, _PROMPT)
            times.append(elapsed)
            parsed = json.loads(response)
            if _EXPECTED_KEYS.issubset(parsed.keys()):
                valid += 1
                print(f"    trial {i + 1}: {elapsed:.1f}s  PASS")
            else:
                print(f"    trial {i + 1}: {elapsed:.1f}s  FAIL (missing keys: {_EXPECTED_KEYS - parsed.keys()})")
        except json.JSONDecodeError as e:
            print(f"    trial {i + 1}: FAIL (invalid JSON — {e})")
        except urllib.error.URLError as e:
            print(f"    trial {i + 1}: FAIL (network — {e})")

    avg = sum(times) / len(times) if times else float("inf")
    rate = valid / N_TRIALS
    print(f"  → valid-JSON: {rate * 100:.0f}%  avg: {avg:.1f}s")
    return {"model": model, "label": label, "valid_json_rate": rate, "avg_seconds": avg}


def main() -> None:
    print("=== T1.9 — LLM feasibility benchmark ===")
    print(f"Trials per model: {N_TRIALS}\n")

    results = {
        "quality": _benchmark(OLLAMA_MODEL_QUALITY, "quality"),
        "fast": _benchmark(OLLAMA_MODEL_FAST, "fast"),
    }

    print("\n=== Verdict ===")
    q = results["quality"]
    q_pass = q["valid_json_rate"] >= 0.80 and q["avg_seconds"] <= 60.0

    if q_pass:
        recommendation = "quality"
        model_name = OLLAMA_MODEL_QUALITY
        reason = f"valid-JSON={q['valid_json_rate']*100:.0f}% >= 80%  avg={q['avg_seconds']:.1f}s <= 60s"
    else:
        recommendation = "fast"
        model_name = OLLAMA_MODEL_FAST
        reason = (
            f"quality model: valid-JSON={q['valid_json_rate']*100:.0f}%  avg={q['avg_seconds']:.1f}s"
            f" — failed go/no-go (need >=80% AND <=60s)"
        )

    print(f"  RECOMMENDATION: use {recommendation.upper()} model")
    print(f"  Reason: {reason}")
    print(f"\n  → Update OLLAMA_MODEL in claim_verifier/config.py:")
    print(f"      OLLAMA_MODEL = \"{model_name}\"")

    print(
        "\nRAM note: check manually with 'htop' or 'free -h' during inference.\n"
        "Target: >= 2 GB free after model load.\n"
        "If RAM is tight, prefer the fast (3B) model regardless of latency.\n"
    )


if __name__ == "__main__":
    main()
