import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def factset_transcript() -> dict:
    return json.loads((FIXTURES_DIR / "factset_transcript.json").read_text(encoding="utf-8"))


@pytest.fixture
def factset_document() -> dict:
    return json.loads((FIXTURES_DIR / "factset_document.json").read_text(encoding="utf-8"))


@pytest.fixture
def verification_result() -> dict:
    return json.loads((FIXTURES_DIR / "verification_result.json").read_text(encoding="utf-8"))
