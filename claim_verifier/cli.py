"""
CLI entrypoint (T6.4).

Usage:
    python -m claim_verifier.cli verify \\
        --claim-id C001 \\
        --transcript call.txt \\
        --document bill.pdf \\
        [--out report.md]

Audio support (--audio) is planned for W9 (faster-whisper).
Use --transcript to provide a pre-transcribed text file in the meantime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from claim_verifier.backends.ollama import OllamaBackend
from claim_verifier.judge import LLMJudge
from claim_verifier.pipeline import run

app = typer.Typer(
    name="claim-verifier",
    help="Medical Claim Verification Assistant — local, offline, privacy-preserving.",
    add_completion=False,
)


@app.command()
def verify(
    claim_id: str = typer.Option(..., "--claim-id", help="Unique claim identifier."),
    transcript: Optional[Path] = typer.Option(
        None, "--transcript", help="Path to a plain-text transcript file."
    ),
    audio: Optional[Path] = typer.Option(
        None, "--audio", help="Path to an audio file (not yet available; use --transcript)."
    ),
    document: Path = typer.Option(..., "--document", help="Path to the hospital bill PDF."),
    out: Optional[Path] = typer.Option(
        None, "--out", help="Write report to this file (default: print to stdout)."
    ),
) -> None:
    """Verify a medical claim against the submitted hospital bill."""
    # Mutual-exclusion checks
    if audio is not None:
        typer.echo(
            "Error: --audio is not yet available (planned for W9). "
            "Use --transcript with a pre-transcribed text file.",
            err=True,
        )
        raise typer.Exit(code=1)

    if transcript is None:
        typer.echo(
            "Error: one of --transcript or --audio is required.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not transcript.exists():
        typer.echo(f"Error: transcript file not found: {transcript}", err=True)
        raise typer.Exit(code=1)

    if not document.exists():
        typer.echo(f"Error: document file not found: {document}", err=True)
        raise typer.Exit(code=1)

    backend = OllamaBackend()
    judge = LLMJudge(backend)

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
