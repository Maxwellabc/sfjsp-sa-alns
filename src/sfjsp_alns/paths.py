"""Repository paths. Historical docs say ``/workspace/fjsp-combo-b``; the package lives at the repo root."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Directory that contains ``src/``, ``scripts/``, ``docs/``, and ``results/``."""
    return Path(__file__).resolve().parents[2]
