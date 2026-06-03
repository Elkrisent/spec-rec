"""
CLI entrypoint (T6.4, T9.1).

Usage:
    python -m claim_verifier.cli verify \\
        --claim-id C001 \\
        --transcript call.txt \\
        --document bill.pdf \\
        [--out report.md]

    # Audio input (W9):
    python -m claim_verifier.cli verify \\
        --claim-id C001 \\
        --audio call.wav \\
        --document bill.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from claim_verifier.backends.ollama import OllamaBackend
from claim_verifier.judge import LLMJudge
from claim_verifier.pipeline import run, run_from_text

app = typer.Typer(
    name="claim-verifier",
    help="Medical Claim Verification Assistant — local, offline, privacy-preserving.",
    add_completion=False,
)


@app.command()
def verify(
    claim_id: str = typer.Option(..., "--claim-id", help="Unique claim identifier."),
    transcript: Optional[Path] = typer.Option(
        None, "--transcript", help="Path to a plain-text transcript file (.txt)."
    ),
    audio: Optional[Path] = typer.Option(
        None,
        "--audio",
        help="Path to an audio file (WAV/MP3/M4A/FLAC). Transcribed via faster-whisper.",
    ),
    document: Path = typer.Option(..., "--document", help="Path to the hospital bill PDF."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write report to this file (default: print to stdout)."
    ),
) -> None:
    """Verify a medical claim against the submitted hospital bill."""
    if transcript is not None and audio is not None:
        typer.echo(
            "Error: --transcript and --audio are mutually exclusive; provide one.",
            err=True,
        )
        raise typer.Exit(code=1)

    if transcript is None and audio is None:
        typer.echo(
            "Error: one of --transcript or --audio is required.",
            err=True,
        )
        raise typer.Exit(code=1)

    if transcript is not None and not transcript.exists():
        typer.echo(f"Error: transcript file not found: {transcript}", err=True)
        raise typer.Exit(code=1)

    if audio is not None and not audio.exists():
        typer.echo(f"Error: audio file not found: {audio}", err=True)
        raise typer.Exit(code=1)

    if not document.exists():
        typer.echo(f"Error: document file not found: {document}", err=True)
        raise typer.Exit(code=1)

    backend = OllamaBackend()
    judge = LLMJudge(backend)

    if audio is not None:
        from claim_verifier.stages.ingestion import IngestionError, ingest_audio, ingest_document

        try:
            transcript_text = ingest_audio(audio)
            document_text = ingest_document(document)
        except IngestionError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=2)
        result = run_from_text(claim_id, transcript_text, document_text, backend, judge)
    else:
        result = run(
            claim_id=claim_id,
            transcript_path=transcript,
            document_path=document,
            backend=backend,
            judge=judge,
        )

    if result.errors:
        for err in result.errors:
            typer.echo(f"[pipeline error] {err}", err=True)

    if out is not None:
        out.write_text(result.report, encoding="utf-8")
        typer.echo(f"Report written to {out}")
    else:
        typer.echo(result.report)

    if result.errors:
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
