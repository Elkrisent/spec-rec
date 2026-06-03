"""
Stage 1 — Ingestion (T6.1).

Reads raw text from a transcript file (plain text) or a PDF document
(pdfplumber text-layer only). Rejects scanned PDFs (<100 chars) and
non-English input.

Public API:
    ingest_transcript(path) -> str
    ingest_document(path) -> str
    IngestionError
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # reproducible language detection

_MIN_TEXT_CHARS = 100  # fewer chars after text extraction → likely scanned image PDF


class IngestionError(Exception):
    """Raised when ingestion fails: scanned PDF, non-English, unreadable file."""


def _check_english(text: str, source: str) -> None:
    try:
        lang = detect(text)
    except LangDetectException as exc:
        raise IngestionError(
            f"{source}: language detection failed — {exc}"
        ) from exc
    if lang != "en":
        raise IngestionError(
            f"{source}: non-English input detected (language={lang!r}); "
            "only English is supported"
        )


def ingest_transcript(path: str | Path) -> str:
    """Read a plain-text transcript file. Reject empty or non-English input."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestionError(f"Cannot read transcript {p}: {exc}") from exc
    text = text.strip()
    if not text:
        raise IngestionError(f"Transcript {p} is empty")
    _check_english(text, str(p))
    return text


def ingest_document(path: str | Path) -> str:
    """
    Extract text layer from a PDF document.

    Rejects files where text extraction yields fewer than _MIN_TEXT_CHARS
    characters (likely a scanned image PDF) and files whose content is
    not English.
    """
    p = Path(path)
    try:
        with pdfplumber.open(str(p)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
    except Exception as exc:
        raise IngestionError(f"Cannot read document {p}: {exc}") from exc
    text = "\n".join(pages).strip()
    if len(text) < _MIN_TEXT_CHARS:
        raise IngestionError(
            f"Document {p} yielded only {len(text)} characters after text extraction — "
            "likely a scanned image PDF. Only digitally-created (text-layer) PDFs "
            "are supported."
        )
    _check_english(text, str(p))
    return text
