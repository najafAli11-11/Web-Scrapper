"""Load and validate config/agents.json against config/agents.schema.json.

Mirrors fetchers/config_loader.py so agent settings (provider/model/api-key
env var) are config, not code (AGENTS.md tech conventions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_AGENTS_PATH = REPO_ROOT / "config" / "agents.json"
DEFAULT_AGENTS_SCHEMA_PATH = REPO_ROOT / "config" / "agents.schema.json"


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


def load_agent_config(
    path: Optional[Path] = None, schema_path: Optional[Path] = None
) -> dict:
    return _load_and_validate(path or DEFAULT_AGENTS_PATH, schema_path or DEFAULT_AGENTS_SCHEMA_PATH)
