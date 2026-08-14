"""Resumable URL frontier for the batch orchestrator (Rule 6, Spec req. 6).

Single SQLite state table (WAL) holding one row per URL. State machine:

    pending -> processing -> done | blocked | flagged | retrying | failed
    retrying -> (deadline passed) pending-due, reprocessed next invocation

- pending:      queued, not started
- processing:   in flight; a row left here by a crash is reverted to pending
                by recover_interrupted() WITHOUT touching its attempt count.
- done/blocked/flagged/failed: terminal. done/blocked/flagged are never
                reprocessed; failed means the retry budget is exhausted.
- retrying:     a transient failure with budget remaining. run_batch is
                single-pass and never blocks (no in-process backoff sleep):
                the row is reprocessed on the next invocation once
                next_attempt_ts <= now. If a run ends before the deadline,
                the URL stays in retrying until a manual re-run.

attempts is monotonic: incremented only in mark_processing(), never reset.
recover_interrupted() does NOT touch it, so a URL that keeps crashing still
reaches the terminal failed state at the max_attempts bound — the recovery
path cannot create an unbounded retry loop (replayability guard).

add_urls() dedupes within a batch: each unique URL is inserted once across
the whole table (Spec edge case: duplicate URLs in an ingestion batch are
deduped before fetching).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_DB_PATH = REPO_ROOT / "data" / "queue.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
    url TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_ts TEXT,
    reason TEXT,
    added_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_ROW_COLS = "url, state, attempts, next_attempt_ts, reason, added_at, updated_at"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _as_iso(ts: Union[str, datetime]) -> str:
    if isinstance(ts, str):
        return ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat(timespec="milliseconds")


class UrlQueue:
    """SQLite-backed URL frontier. Opens with WAL and recovers interrupted rows."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_QUEUE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        with self.conn:
            self.conn.execute(_SCHEMA)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_urls_state ON urls(state)")
        self.recover_interrupted()

    # -- enqueue ---------------------------------------------------------

    def add_urls(self, urls: Iterable[str]) -> int:
        """Insert pending rows for URLs not already present (batch dedup). Returns new rows."""
        added = 0
        now = now_utc_iso()
        with self.conn:
            for url in urls:
                if not url:
                    continue
                if self.conn.execute("SELECT 1 FROM urls WHERE url = ?", (url,)).fetchone() is None:
                    self.conn.execute(
                        "INSERT INTO urls (url, state, attempts, added_at, updated_at)"
                        " VALUES (?, 'pending', 0, ?, ?)",
                        (url, now, now),
                    )
                    added += 1
        return added

    # -- scheduling ------------------------------------------------------

    def pending_due(self, now: Optional[Union[str, datetime]] = None) -> list[str]:
        """Pending URLs plus retrying URLs whose backoff deadline has passed."""
        cutoff = _as_iso(now) if now is not None else now_utc_iso()
        cur = self.conn.execute(
            "SELECT url FROM urls"
            " WHERE state = 'pending' OR (state = 'retrying' AND next_attempt_ts <= ?)"
            " ORDER BY added_at, url",
            (cutoff,),
        )
        return [r["url"] for r in cur.fetchall()]

    def mark_processing(self, url: str) -> int:
        """Mark a URL in flight; increments the (monotonic) attempt count. Returns attempts."""
        if self.get(url) is None:
            raise ValueError(f"mark_processing: unknown url {url!r}")
        with self.conn:
            self.conn.execute(
                "UPDATE urls SET state = 'processing', attempts = attempts + 1, updated_at = ? WHERE url = ?",
                (now_utc_iso(), url),
            )
        return int(self.get(url)["attempts"])

    # -- outcomes --------------------------------------------------------

    def complete(self, url: str, state: str, reason: Optional[str] = None) -> None:
        """Terminal success/block/flag: done, blocked, or flagged."""
        if state not in ("done", "blocked", "flagged"):
            raise ValueError(f"complete() accepts only done/blocked/flagged, got {state!r}")
        if self.get(url) is None:
            raise ValueError(f"complete: unknown url {url!r}")
        with self.conn:
            self.conn.execute(
                "UPDATE urls SET state = ?, reason = ?, updated_at = ? WHERE url = ?",
                (state, reason, now_utc_iso(), url),
            )

    def record_failure(
        self,
        url: str,
        *,
        max_attempts: int,
        backoff_seconds: float,
        now: Optional[Union[str, datetime]] = None,
    ) -> str:
        """Apply the retry policy to a transient failure. Returns the new state.

        attempts >= max_attempts -> terminal 'failed' (budget exhausted).
        Otherwise -> 'retrying', scheduled next_attempt_ts = now + backoff.
        """
        row = self.get(url)
        if row is None:
            raise ValueError(f"record_failure: unknown url {url!r}")
        attempts = int(row["attempts"])
        now_str = _as_iso(now) if now is not None else now_utc_iso()
        if attempts >= max_attempts:
            state = "failed"
            next_ts = None
        else:
            state = "retrying"
            next_ts = (
                datetime.fromisoformat(now_str) + timedelta(seconds=backoff_seconds)
            ).isoformat(timespec="milliseconds")
        with self.conn:
            self.conn.execute(
                "UPDATE urls SET state = ?, next_attempt_ts = ?, reason = ?, updated_at = ? WHERE url = ?",
                (state, next_ts, f"attempts={attempts}/{max_attempts}", now_str, url),
            )
        return state

    # -- recovery / lifecycle -------------------------------------------

    def recover_interrupted(self) -> int:
        """Revert in-flight rows to pending WITHOUT resetting their attempt counts."""
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM urls WHERE state = 'processing'")
        n = int(cur.fetchone()["n"])
        if n:
            with self.conn:
                self.conn.execute(
                    "UPDATE urls SET state = 'pending', updated_at = ? WHERE state = 'processing'",
                    (now_utc_iso(),),
                )
        return n

    def reset(self) -> int:
        """Clear the whole frontier for a fresh run. Returns rows cleared."""
        cur = self.conn.execute("SELECT COUNT(*) AS n FROM urls")
        n = int(cur.fetchone()["n"])
        with self.conn:
            self.conn.execute("DELETE FROM urls")
        return n

    # -- inspection ------------------------------------------------------

    def get(self, url: str) -> Optional[dict]:
        cur = self.conn.execute(f"SELECT {_ROW_COLS} FROM urls WHERE url = ?", (url,))
        row = cur.fetchone()
        return {k: row[k] for k in row.keys()} if row else None

    def states(self) -> list[dict]:
        cur = self.conn.execute(f"SELECT {_ROW_COLS} FROM urls ORDER BY added_at, url")
        return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]

    def counts(self) -> dict[str, int]:
        cur = self.conn.execute("SELECT state, COUNT(*) AS n FROM urls GROUP BY state")
        return {r["state"]: int(r["n"]) for r in cur.fetchall()}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "UrlQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
