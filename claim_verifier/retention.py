"""
Retention policy — secure deletion of source files (T9.3).

Configurable via config.py:
    RETENTION_DELETE_SOURCES  — if True, delete source files after pipeline run
    RETENTION_SECURE_DELETE   — if True, overwrite with zeros before unlinking

Public API:
    secure_delete(path, *, secure=False) -> None
    apply_retention(paths, *, secure=False) -> None
"""

from __future__ import annotations

from pathlib import Path


def secure_delete(path: str | Path, *, secure: bool = False) -> None:
    """Delete a file. If secure=True, overwrite with zero bytes first."""
    p = Path(path)
    if not p.exists():
        return
    if secure:
        size = p.stat().st_size
        with p.open("wb") as fh:
            fh.write(b"\x00" * size)
    p.unlink(missing_ok=True)


def apply_retention(paths: list[str | Path], *, secure: bool = False) -> None:
    """Delete each path. Non-existent paths are silently skipped."""
    for p in paths:
        secure_delete(p, secure=secure)
