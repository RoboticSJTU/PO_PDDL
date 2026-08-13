"""Resolve optional native planner assets without repository-specific paths."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def despot_root() -> Path:
    configured = os.environ.get("PO_PDDL_DESPOT_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root() / "third_party" / "despot",
        project_root() / "POMDPDDL_ccplus",
        project_root().parent / "POMDPDDL" / "POMDPDDL_ccplus",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "despot").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "DESPOT sources were not found. Set PO_PDDL_DESPOT_ROOT to a directory "
        "containing the `despot/` source tree."
    )


def hyperparams_path() -> Path | None:
    configured = os.environ.get("PO_PDDL_HYPERPARAMS")
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root() / "config" / "hyperparams.yaml",
        Path(__file__).resolve().parents[2] / "config" / "hyperparams.yaml",
    ]
    return next((path.resolve() for path in candidates if path is not None and path.is_file()), None)
