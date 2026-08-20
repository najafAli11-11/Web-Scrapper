"""Read-only data access for the M9 Streamlit UI (queue + events + batch spawn).

The UI must never write pipeline state. Two hard consequences:

- Never open `UrlQueue` from the UI: its ``__init__`` calls
  ``recover_interrupted()``, which WRITES (reverts processing -> pending) on
  every open. Reads go through a read-only SQLite connection (uri ``mode=ro``),
  so any accidental write raises ``OperationalError`` structurally.
- Never open `FetchLogger` from the UI either: its connection is a writer.
  Read the events table directly, also read-only.

Concurrency: queue.db and logs.db are WAL. Read-only readers coexist with the
detached orchestrator writer child and still see consistent snapshots.

`BatchRunner` spawns the orchestrator CLI as a detached child using
``sys.executable`` (never a bare "python" — on a machine with multiple
installs or a venv, a wrong interpreter must not be launched silently), and
refuses a second submit while a batch is running against the same queue.db
(see `BatchAlreadyRunning`). Liveness is checked in-process (child poll) and
via a per-queue_db pid marker, so a crashed UI cannot permanently block new
submits and two different corpora cannot deadlock each other.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_DB_PATH = REPO_ROOT / "data" / "queue.db"
DEFAULT_LOGS_DB_PATH = REPO_ROOT / "data" / "logs.db"
RUN_OUTPUT_DIR = REPO_ROOT / "data" / "ui-runs"

_QUEUE_COLS = "url, state, attempts, next_attempt_ts, reason, added_at, updated_at"


def read_only_conn(db_path: Union[str, Path]) -> sqlite3.Connection:
    """Open a SQLite DB read-only; any write raises ``OperationalError``."""
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


class QueueView:
    """Read-only view over the orchestrator's queue.db (never writes)."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_QUEUE_DB_PATH

    def states(self) -> list[dict]:
        return self._rows(f"SELECT {_QUEUE_COLS} FROM urls ORDER BY added_at, url")

    def counts(self) -> dict[str, int]:
        rows = self._rows("SELECT state, COUNT(*) AS n FROM urls GROUP BY state")
        return {r["state"]: int(r["n"]) for r in rows}

    def status_for(self, url: str) -> Optional[dict]:
        rows = self._rows(f"SELECT {_QUEUE_COLS} FROM urls WHERE url = ?", (url,))
        return rows[0] if rows else None

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            conn = read_only_conn(self.db_path)
        except sqlite3.OperationalError:
            return []
        try:
            cur = conn.execute(sql, params)
            return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()


class EventLogView:
    """Read-only view over the shared events DB (the M9 Logs view feed)."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_LOGS_DB_PATH

    def event_types(self) -> list[str]:
        rows = self._rows("SELECT DISTINCT event_type FROM events ORDER BY event_type")
        return [r["event_type"] for r in rows]

    def recent(
        self,
        limit: int = 200,
        event_types: Optional[Iterable[str]] = None,
        url_filter: Optional[str] = None,
    ) -> list[dict]:
        sql = "SELECT ts, event_type, url, outcome, reason, details_json FROM events"
        clauses: list[str] = []
        params: list = []
        if event_types:
            clauses.append("event_type IN (%s)" % ",".join("?" * len(event_types)))
            params.extend(event_types)
        if url_filter:
            clauses.append("url LIKE ?")
            params.append(f"%{url_filter}%")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        return self._rows(sql, tuple(params))

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        try:
            conn = read_only_conn(self.db_path)
        except sqlite3.OperationalError:
            return []
        try:
            cur = conn.execute(sql, params)
            return [{k: r[k] for k in r.keys()} for r in cur.fetchall()]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()


class BatchAlreadyRunning(RuntimeError):
    """Raised when a submit is attempted while a batch is running against the same queue.db."""


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        # PROCESS_QUERY_LIMITED_INFORMATION; a null handle means no such process.
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_urls_file(urls: Iterable[str], run_dir: Union[str, Path] = RUN_OUTPUT_DIR) -> Path:
    """Normalize submitted URLs into a temp urls file (M7 CLI ``_read_urls`` format)."""
    cleaned = [
        str(u).strip()
        for u in urls
        if str(u).strip() and not str(u).strip().startswith("#")
    ]
    if not cleaned:
        raise ValueError("no valid URLs to ingest")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="urls-", suffix=".txt", dir=str(run_dir))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(cleaned) + "\n")
    return Path(tmp)


class BatchRunner:
    """Spawns the M7 orchestrator batch CLI as a detached child (non-blocking).

    The orchestrator CLI is the single source of orchestration — the UI never
    reimplements run_batch. Spawning is non-blocking (Popen, no wait); the UI
    watches liveness via is_running() (resolution #4's bounded refresh loop).

    Concurrency guard (one batch per queue.db): spawn() raises
    BatchAlreadyRunning if is_running() for the same queue_db. is_running()
    considers the in-process child AND the per-queue_db pid marker, so a
    crashed UI / orphaned child is detected (stale marker cleared, submit
    unblocked) rather than silently double-submitting.
    """

    def __init__(
        self,
        *,
        queue_db: Optional[Union[str, Path]] = None,
        logs_db: Optional[Union[str, Path]] = None,
        urls_file: Optional[Union[str, Path]] = None,
        run_dir: Optional[Union[str, Path]] = None,
    ):
        self.queue_db = Path(queue_db) if queue_db else DEFAULT_QUEUE_DB_PATH
        self.logs_db = Path(logs_db) if logs_db else DEFAULT_LOGS_DB_PATH
        self.urls_file = Path(urls_file) if urls_file else None
        self.run_dir = Path(run_dir) if run_dir else RUN_OUTPUT_DIR
        self.run_output: Optional[Path] = None
        self._proc: Optional[subprocess.Popen] = None

    def _marker_path(self) -> Path:
        digest = hashlib.sha256(str(self.queue_db.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.run_dir / f".batch.{digest}.pid"

    def is_running(self) -> bool:
        if self._proc is not None and self._proc.poll() is None:
            return True
        marker = self._marker_path()
        if marker.exists():
            pid = int(marker.read_text(encoding="utf-8").strip())
            if _pid_alive(pid):
                return True
            marker.unlink(missing_ok=True)
        return False

    def _build_command(self) -> list[str]:
        if self.urls_file is None:
            raise ValueError("BatchRunner needs a urls file before spawn")
        return [
            sys.executable,
            "-m",
            "orchestrator",
            "run",
            str(self.urls_file),
            "--db",
            str(self.queue_db),
            "--log",
            str(self.logs_db),
        ]

    def spawn(self) -> subprocess.Popen:
        if self.is_running():
            raise BatchAlreadyRunning(f"a batch is already running against {self.queue_db}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_output = self.run_dir / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        with open(self.run_output, "w", encoding="utf-8") as out:
            self._proc = subprocess.Popen(
                self._build_command(),
                cwd=REPO_ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
        self._marker_path().write_text(str(self._proc.pid), encoding="utf-8")
        return self._proc

    def stop(self) -> None:
        """Terminate the child (bounded wait) and clear the liveness marker."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._marker_path().unlink(missing_ok=True)


def spawn_ingestion(
    urls: Iterable[str],
    *,
    queue_db: Optional[Union[str, Path]] = None,
    logs_db: Optional[Union[str, Path]] = None,
    run_dir: Optional[Union[str, Path]] = None,
) -> BatchRunner:
    """Write submitted URLs to a temp file and spawn a detached batch run."""
    urls_file = _write_urls_file(urls, run_dir=run_dir or RUN_OUTPUT_DIR)
    runner = BatchRunner(queue_db=queue_db, logs_db=logs_db, urls_file=urls_file, run_dir=run_dir)
    try:
        runner.spawn()
    except Exception:
        urls_file.unlink(missing_ok=True)
        raise
    return runner


# ---------------------------------------------------------------------------
# Delete / clear helpers (called from the UI for URL removal and log clearing).
# These open a short-lived write connection, execute, commit, and close.
# ---------------------------------------------------------------------------


def delete_url_from_queue(url: str, *, db_path: Union[str, Path]) -> int:
    """Delete a URL row from the queue.db ``urls`` table.

    Returns the number of rows deleted (0 or 1).  Safe to call even if the
    URL doesn't exist or the DB is empty — no error is raised.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path.resolve()))
    try:
        cur = conn.execute("DELETE FROM urls WHERE url = ?", (url,))
        conn.commit()
        return cur.rowcount
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def delete_url_logs(url: str, *, db_path: Union[str, Path]) -> int:
    """Delete all events for a URL from the logs.db ``events`` table.

    Returns the number of rows deleted.  Safe to call even if no events
    exist for the URL or the DB is empty.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path.resolve()))
    try:
        cur = conn.execute("DELETE FROM events WHERE url = ?", (url,))
        conn.commit()
        return cur.rowcount
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()


def delete_url_everywhere(
    url: str,
    *,
    queue_db: Union[str, Path],
    logs_db: Union[str, Path],
    collection_name: str,
    store: "VectorStore",
) -> dict:
    """Delete a URL from queue.db, logs.db, and the ChromaDB vector store.

    Returns a summary dict: ``{"queue_deleted": int, "logs_deleted": int}``.
    ChromaDB deletion is fire-and-forget (no count returned by the driver).
    """
    queue_deleted = delete_url_from_queue(url, db_path=queue_db)
    logs_deleted = delete_url_logs(url, db_path=logs_db)
    store.delete_url(url, collection_name=collection_name)
    return {"queue_deleted": queue_deleted, "logs_deleted": logs_deleted}


def clear_all_logs(*, db_path: Union[str, Path]) -> int:
    """Delete ALL events from the logs.db ``events`` table.

    Returns the number of rows deleted.  Safe to call on an empty or
    missing table.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    conn = sqlite3.connect(str(db_path.resolve()))
    try:
        cur = conn.execute("DELETE FROM events")
        conn.commit()
        return cur.rowcount
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()
