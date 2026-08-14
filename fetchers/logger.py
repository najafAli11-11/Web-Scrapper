"""Structured event logging for the pipeline.

Single SQLite `events` table in WAL mode. The shape is deliberately general
so later milestones (validation flags, chunking events) reuse the same table —
this is exactly what the Milestone 9 Logs view reads from (Spec req. 16).

Minimal columns for Spec req. 4 are url / timestamp / outcome / reason; the
extra columns (event_type, details_json) exist to make the Logs view useful
(obstacle detections as their own rows, structured extras).

Logging is best-effort: a broken DB must never crash a fetch.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from fetchers.types import FetchResult

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "logs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event_type TEXT NOT NULL,
    url TEXT,
    outcome TEXT,
    reason TEXT,
    details_json TEXT
)
"""


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class FetchLogger:
    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        *,
        check_same_thread: bool = True,
    ):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False is required when the logger is cached and
        # reused across threads (the M9 UI caches it via st.cache_resource,
        # and Streamlit runs each rerun on a different thread). SQLite is
        # serialized by file locking + WAL, and Streamlit runs are serialized,
        # so a single cross-thread connection is safe there.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.execute(_SCHEMA)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_url ON events(url)")

    def log_event(
        self,
        event_type: Optional[str] = None,
        *,
        url: Optional[str] = None,
        outcome: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO events (ts, event_type, url, outcome, reason, details_json)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        now_utc_iso(),
                        event_type,
                        url,
                        outcome,
                        reason,
                        json.dumps(details, default=str) if details else None,
                    ),
                )
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[logger] failed to write event: {exc}", file=sys.stderr)

    def log_fetch_attempt(self, result: FetchResult) -> None:
        details = {
            "status_code": result.status_code,
            "content_type": result.content_type,
            "final_url": result.final_url,
            "fetcher": result.fetcher,
        }
        self.log_event(
            event_type="fetch_attempt",
            url=result.url,
            outcome=result.outcome.value,
            reason=result.reason,
            details=details,
        )

    def rows_for_url(self, url: str, limit: int = 200) -> list[dict]:
        cur = self.conn.execute(
            "SELECT ts, event_type, url, outcome, reason, details_json"
            " FROM events WHERE url = ? ORDER BY ts, id",
            (url,),
        )
        rows = [{k: r[k] for k in r.keys()} for r in cur.fetchall()]
        return rows[-limit:]

    def recent_events(self, limit: int = 200) -> list[dict]:
        """Latest events across all URLs — the M9 Logs view feed."""
        cur = self.conn.execute(
            "SELECT ts, event_type, url, outcome, reason, details_json"
            " FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "FetchLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
