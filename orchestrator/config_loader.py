"""Load and validate config/orchestrator.json against config/orchestrator.schema.json.

Mirrors agents/config_loader.py, fetchers/config_loader.py and
pipeline/config_loader.py so the batch runner's retry policy is versioned,
schema-checked config rather than hardcoded orchestration logic (AGENTS.md
tech conventions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ORCHESTRATOR_PATH = REPO_ROOT / "config" / "orchestrator.json"
DEFAULT_ORCHESTRATOR_SCHEMA_PATH = REPO_ROOT / "config" / "orchestrator.schema.json"


def _load_and_validate(path: Path, schema_path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        instance = json.load(fh)
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    try:
        jsonschema.validate(instance, schema)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"invalid config {path}: {exc.message}") from exc
    return instance


def load_orchestrator_config(
    path: Optional[Path] = None, schema_path: Optional[Path] = None
) -> dict:
    return _load_and_validate(
        path or DEFAULT_ORCHESTRATOR_PATH, schema_path or DEFAULT_ORCHESTRATOR_SCHEMA_PATH
    )
