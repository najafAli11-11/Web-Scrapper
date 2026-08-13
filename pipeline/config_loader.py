"""Load and validate config/embeddings.json against config/embeddings.schema.json.

Mirrors agents/config_loader.py and fetchers/config_loader.py so embedding
model selection, chunking limits, and store paths are config, not code
(AGENTS.md tech conventions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EMBEDDINGS_PATH = REPO_ROOT / "config" / "embeddings.json"
DEFAULT_EMBEDDINGS_SCHEMA_PATH = REPO_ROOT / "config" / "embeddings.schema.json"


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


def load_pipeline_config(
    path: Optional[Path] = None, schema_path: Optional[Path] = None
) -> dict:
    return _load_and_validate(path or DEFAULT_EMBEDDINGS_PATH, schema_path or DEFAULT_EMBEDDINGS_SCHEMA_PATH)
