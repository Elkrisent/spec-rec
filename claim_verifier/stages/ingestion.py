"""
Stage 1 — Ingestion (T6.1, T9.1, T9.2).

Reads raw text from a transcript file (plain text), a PDF document
(pdfplumber text-layer only), or an audio file (faster-whisper, CPU).

Rejects:
  - Scanned PDFs (< 100 chars of text)
  - Non-English input (all sources)
  - Unsupported file extensions / oversized files (T9.2)

Public API:
    ingest_transcript(path) -> str
    ingest_document(path) -> str
    ingest_audio(path) -> str       # T9.1
    IngestionError
"""

from __future__ import annotations

from pathlib import Path

import pdfplumber
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # reproducible language detection

_MIN_TEXT_CHARS = 100  # fewer chars after extraction → likely scanned image PDF

# File size limits (T9.2)
_MAX_AUDIO_BYTES = 50 * 1024 * 1024   # 50 MB
_MAX_DOC_BYTES   = 20 * 1024 * 1024   # 20 MB

# Extension allowlists (T9.2)
_AUDIO_EXTENSIONS      = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}
_DOCUMENT_EXTENSIONS   = {".pdf"}
_TRANSCRIPT_EXTENSIONS = {".txt"}


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
    """Read a plain-text (.txt) transcript. Reject empty or non-English input."""
    p = Path(path)
    if p.suffix.lower() not in _TRANSCRIPT_EXTENSIONS:
        raise IngestionError(
            f"Unsupported transcript format {p.suffix!r}. Only .txt files are supported."
        )
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

    Rejects: non-.pdf extensions; files > 20 MB; files where text extraction
    yields fewer than _MIN_TEXT_CHARS characters (likely a scanned image PDF);
    files whose content is not English.
    """
    p = Path(path)
    if p.suffix.lower() not in _DOCUMENT_EXTENSIONS:
        raise IngestionError(
            f"Unsupported document format {p.suffix!r}. Only PDF (.pdf) is supported."
        )
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise IngestionError(f"Cannot read document {p}: {exc}") from exc
    if size > _MAX_DOC_BYTES:
        raise IngestionError(
            f"Document {p} is {size / 1024 / 1024:.1f} MB; "
            "maximum supported size is 20 MB"
        )
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


def ingest_audio(path: str | Path) -> str:
    """
    Transcribe an audio file using faster-whisper (CPU, base model).

    Rejects: unsupported extensions; files > 50 MB; non-English audio;
    audio that produces an empty transcription after joining segments.

    Requires faster-whisper to be installed:
        uv pip install faster-whisper>=1.0
    """
    p = Path(path)
    if p.suffix.lower() not in _AUDIO_EXTENSIONS:
        raise IngestionError(
            f"Unsupported audio format {p.suffix!r}. "
            f"Supported: {', '.join(sorted(_AUDIO_EXTENSIONS))}"
        )
    try:
        size = p.stat().st_size
    except OSError as exc:
        raise IngestionError(f"Cannot access audio file {p}: {exc}") from exc
    if size > _MAX_AUDIO_BYTES:
        raise IngestionError(
            f"Audio file {p} is {size / 1024 / 1024:.1f} MB; "
            "maximum supported size is 50 MB"
        )

    try:
        from faster_whisper import WhisperModel  # lazy import — optional dep
    except ImportError as exc:
        raise IngestionError(
            "faster-whisper is not installed. "
            "Uncomment 'faster-whisper>=1.0' in requirements.txt and run "
            "'uv pip install -r requirements.txt'."
        ) from exc

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(p), beam_size=5)

    if info.language != "en":
        raise IngestionError(
            f"{p}: non-English audio detected (language={info.language!r}); "
            "only English is supported"
        )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        raise IngestionError(f"Audio {p} produced an empty transcription")

    return text
