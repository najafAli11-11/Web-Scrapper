"""Load and validate the versioned config files.

`config/obstacles.json` is validated against `config/obstacles.schema.json`
(formal JSON Schema, the source of truth for allowed obstacle keys,
detection methods, and resolution policies — per AGENTS.md). `config/fetch.json`
is validated against `config/fetch.schema.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OBSTACLES_PATH = REPO_ROOT / "config" / "obstacles.json"
DEFAULT_OBSTACLES_SCHEMA_PATH = REPO_ROOT / "config" / "obstacles.schema.json"
DEFAULT_FETCH_PATH = REPO_ROOT / "config" / "fetch.json"
DEFAULT_FETCH_SCHEMA_PATH = REPO_ROOT / "config" / "fetch.schema.json"


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


def load_obstacle_config(
    path: Optional[Path] = None, schema_path: Optional[Path] = None
) -> dict:
    return _load_and_validate(path or DEFAULT_OBSTACLES_PATH, schema_path or DEFAULT_OBSTACLES_SCHEMA_PATH)


def load_fetch_config(path: Optional[Path] = None, schema_path: Optional[Path] = None) -> dict:
    return _load_and_validate(path or DEFAULT_FETCH_PATH, schema_path or DEFAULT_FETCH_SCHEMA_PATH)


def is_enabled(obstacle_cfg: dict, name: str) -> bool:
    return bool(obstacle_cfg.get(name, {}).get("enabled"))


def policy_for(obstacle_cfg: dict, name: str) -> str:
    return str(obstacle_cfg.get(name, {}).get("policy", ""))


def max_retries_for(obstacle_cfg: dict, name: str) -> int:
    try:
        return int(obstacle_cfg.get(name, {}).get("max_retries", 3))
    except (TypeError, ValueError):
        return 3
