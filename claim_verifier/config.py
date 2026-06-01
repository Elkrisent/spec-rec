import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema" / "verification_schema.json"


def load_schema() -> dict:
    with _SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


# Risk band thresholds (consistency_score)
RISK_LOW_THRESHOLD = 0.90    # score >= 0.90 → LOW
RISK_HIGH_THRESHOLD = 0.70   # score <  0.70 → HIGH; between → MEDIUM

# Fuzzy match
FUZZY_HOSPITAL_THRESHOLD = 0.85  # rapidfuzz token_sort_ratio / 100

# Numeric tolerances
AMOUNT_TOLERANCE_PCT = 0.05   # ±5%
LOS_TOLERANCE_ABS = 1         # ±1 day

# Date tolerance
DATE_TOLERANCE_DAYS = 1       # ±1 day

# Ollama local LLM
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL_QUALITY = "qwen2.5:7b-instruct-q4_K_M"   # ~4.7 GB RAM; use after T1.9 benchmark
OLLAMA_MODEL_FAST = "qwen2.5:3b-instruct"              # ~2.5 GB RAM; fallback if 7B >60s/call
OLLAMA_MODEL = OLLAMA_MODEL_QUALITY                    # active model (update after T1.9)
OLLAMA_TEMPERATURE = 0.0

# LLM response cache (SHA-256 keyed; keeps CI offline; mandatory on CPU)
LLM_CACHE_DIR = Path(__file__).parent.parent / ".llm_cache"
