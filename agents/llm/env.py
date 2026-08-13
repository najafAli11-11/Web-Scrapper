"""Tiny .env loader for API keys (no dependency).

Keys live in the environment, never in versioned config. If an env var is
already set it wins; otherwise `config/.env`-style file at the repo root is
loaded. Handles `KEY=VALUE`, quotes, and `#` comments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"


def load_dotenv(path: Optional[Path] = None) -> None:
    path = path or DEFAULT_ENV_PATH
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_env(name: str) -> str | None:
    """Return an env var, falling back to the repo-root .env file."""
    if name in os.environ:
        return os.environ[name]
    load_dotenv()
    return os.environ.get(name)
