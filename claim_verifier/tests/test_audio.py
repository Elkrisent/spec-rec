"""
W9 tests — audio ingestion, input validation, retention policy (T9.1–T9.3).

TestIngestAudio           : ingest_audio() happy-path and all rejection cases
TestDocumentValidation    : size / extension validation added to ingest_document (T9.2)
TestTranscriptValidation  : extension validation added to ingest_transcript (T9.2)
TestSecureDelete          : secure_delete() overwrite + unlink
TestApplyRetention        : apply_retention() multi-path + edge cases
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claim_verifier.retention import apply_retention, secure_delete
from claim_verifier.stages.ingestion import (
    IngestionError,
    _MAX_AUDIO_BYTES,
    _MAX_DOC_BYTES,
    ingest_audio,
    ingest_document,
    ingest_transcript,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_ENGLISH_TEXT = (
    "Patient was admitted to Apollo Hospitals on March 12 for acute appendicitis. "
    "The total bill amount is Rs 62,000. Length of stay was 4 days."
)


@pytest.fixture
def mock_whisper():
    """Inject a fake faster_whisper module so tests never touch the real model."""
    mock_fw = MagicMock()
    mock_model = MagicMock()
    mock_fw.WhisperModel.return_value = mock_model
    with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
        yield mock_fw, mock_model


def _make_audio_file(tmp_path: Path, suffix: str = ".wav") -> Path:
    p = tmp_path / f"audio{suffix}"
    p.write_bytes(b"dummy-audio-content")
    return p


# ---------------------------------------------------------------------------
# TestIngestAudio
# ---------------------------------------------------------------------------


class TestIngestAudio:
    def test_wrong_extension_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.xyz"
        p.write_bytes(b"data")
        with pytest.raises(IngestionError, match="Unsupported audio format"):
            ingest_audio(p)

    def test_wrong_extension_error_lists_supported(self, tmp_path: Path) -> None:
        p = tmp_path / "audio.xyz"
        p.write_bytes(b"data")
        with pytest.raises(IngestionError) as exc_info:
            ingest_audio(p)
        assert ".wav" in str(exc_info.value)

    def test_file_too_large_rejected(self, tmp_path: Path, monkeypatch) -> None:
        p = _make_audio_file(tmp_path)
        import os
        original = p.stat
        fake_stat = MagicMock()
        fake_stat.st_size = _MAX_AUDIO_BYTES + 1
        monkeypatch.setattr(type(p), "stat", lambda self: fake_stat)
        with pytest.raises(IngestionError, match="50 MB"):
            ingest_audio(p)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.wav"
        with pytest.raises(IngestionError, match="Cannot access audio file"):
            ingest_audio(p)

    def test_faster_whisper_not_installed(self, tmp_path: Path) -> None:
        p = _make_audio_file(tmp_path)
        with patch.dict(sys.modules, {"faster_whisper": None}):
            with pytest.raises(IngestionError, match="not installed"):
                ingest_audio(p)

    def test_english_audio_transcribed(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_seg = MagicMock()
        mock_seg.text = _ENGLISH_TEXT
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path)
        result = ingest_audio(p)
        assert "Apollo Hospitals" in result

    def test_non_english_rejected(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "de"
        mock_seg = MagicMock()
        mock_seg.text = "Krankenhaus Aufnahme"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path)
        with pytest.raises(IngestionError, match="non-English"):
            ingest_audio(p)

    def test_non_english_error_mentions_language(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "de"
        mock_seg = MagicMock()
        mock_seg.text = "Krankenhaus"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path)
        with pytest.raises(IngestionError) as exc_info:
            ingest_audio(p)
        assert "de" in str(exc_info.value)

    def test_empty_transcription_rejected(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_model.transcribe.return_value = ([], mock_info)

        p = _make_audio_file(tmp_path)
        with pytest.raises(IngestionError, match="empty transcription"):
            ingest_audio(p)

    def test_whitespace_only_transcription_rejected(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_seg = MagicMock()
        mock_seg.text = "   "
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path)
        with pytest.raises(IngestionError, match="empty transcription"):
            ingest_audio(p)

    def test_whisper_model_uses_base_cpu_int8(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_seg = MagicMock()
        mock_seg.text = "text"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path)
        ingest_audio(p)
        mock_fw.WhisperModel.assert_called_once_with("base", device="cpu", compute_type="int8")

    def test_transcribe_uses_beam_size_5(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_seg = MagicMock()
        mock_seg.text = "text"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path)
        ingest_audio(p)
        mock_model.transcribe.assert_called_once_with(str(p), beam_size=5)

    def test_multiple_segments_joined(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock()
        mock_info.language = "en"
        seg1 = MagicMock(); seg1.text = "Hello"
        seg2 = MagicMock(); seg2.text = "world"
        mock_model.transcribe.return_value = ([seg1, seg2], mock_info)

        p = _make_audio_file(tmp_path)
        result = ingest_audio(p)
        assert result == "Hello world"

    def test_mp3_extension_accepted(self, tmp_path: Path, mock_whisper) -> None:
        mock_fw, mock_model = mock_whisper
        mock_info = MagicMock(); mock_info.language = "en"
        mock_seg = MagicMock(); mock_seg.text = "text"
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        p = _make_audio_file(tmp_path, suffix=".mp3")
        result = ingest_audio(p)
        assert result == "text"


# ---------------------------------------------------------------------------
# TestDocumentValidation (T9.2 — size + extension on ingest_document)
# ---------------------------------------------------------------------------


class TestDocumentValidation:
    def test_wrong_extension_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bill.docx"
        p.write_bytes(b"data")
        with pytest.raises(IngestionError, match="Unsupported document format"):
            ingest_document(p)

    def test_file_too_large_rejected(self, tmp_path: Path, monkeypatch) -> None:
        p = tmp_path / "big.pdf"
        p.write_bytes(b"x")
        fake_stat = MagicMock()
        fake_stat.st_size = _MAX_DOC_BYTES + 1
        monkeypatch.setattr(type(p), "stat", lambda self: fake_stat)
        with pytest.raises(IngestionError, match="20 MB"):
            ingest_document(p)


# ---------------------------------------------------------------------------
# TestTranscriptValidation (T9.2 — extension check on ingest_transcript)
# ---------------------------------------------------------------------------


class TestTranscriptValidation:
    def test_wrong_extension_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "transcript.pdf"
        p.write_bytes(b"data")
        with pytest.raises(IngestionError, match="Unsupported transcript format"):
            ingest_transcript(p)


# ---------------------------------------------------------------------------
# TestSecureDelete (T9.3)
# ---------------------------------------------------------------------------


class TestSecureDelete:
    def test_deletes_existing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "source.wav"
        p.write_bytes(b"audio data")
        secure_delete(p)
        assert not p.exists()

    def test_silent_on_missing_file(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.wav"
        secure_delete(p)  # must not raise

    def test_overwrite_before_delete_when_secure(self, tmp_path: Path) -> None:
        p = tmp_path / "sensitive.wav"
        p.write_bytes(b"ABCDE")
        secure_delete(p, secure=True)
        assert not p.exists()

    def test_non_secure_delete_does_not_overwrite(self, tmp_path: Path) -> None:
        p = tmp_path / "normal.wav"
        content = b"original content"
        p.write_bytes(content)
        secure_delete(p, secure=False)
        assert not p.exists()

    def test_secure_overwrite_writes_zeros(self, tmp_path: Path) -> None:
        p = tmp_path / "to_zero.bin"
        original_content = b"HELLO"
        p.write_bytes(original_content)
        # Prevent actual deletion so we can read the overwritten content
        with patch.object(Path, "unlink"):
            secure_delete(p, secure=True)
        assert p.read_bytes() == b"\x00" * len(original_content)


# ---------------------------------------------------------------------------
# TestApplyRetention (T9.3)
# ---------------------------------------------------------------------------


class TestApplyRetention:
    def test_deletes_all_files(self, tmp_path: Path) -> None:
        files = [tmp_path / f"f{i}.wav" for i in range(3)]
        for f in files:
            f.write_bytes(b"data")
        apply_retention(files)
        assert not any(f.exists() for f in files)

    def test_skips_nonexistent_paths(self, tmp_path: Path) -> None:
        apply_retention([tmp_path / "ghost.wav"])  # must not raise

    def test_empty_list_ok(self) -> None:
        apply_retention([])  # must not raise

    def test_secure_flag_passed_through(self, tmp_path: Path) -> None:
        p = tmp_path / "sec.wav"
        p.write_bytes(b"secret")
        apply_retention([p], secure=True)
        assert not p.exists()

    def test_partial_list_all_existing_deleted(self, tmp_path: Path) -> None:
        existing = tmp_path / "exists.wav"
        existing.write_bytes(b"x")
        missing = tmp_path / "missing.wav"
        apply_retention([existing, missing])
        assert not existing.exists()
