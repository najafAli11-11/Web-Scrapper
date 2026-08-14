"""Hermetic tests for the UI's read-only data access layer (M9)."""

import sqlite3

import pytest

from fetchers.logger import FetchLogger
from orchestrator.queue import UrlQueue
from ui.db_view import (
    BatchAlreadyRunning,
    BatchRunner,
    EventLogView,
    QueueView,
    read_only_conn,
    spawn_ingestion,
    _write_urls_file,
)


class SleepRunner(BatchRunner):
    """Real subprocess (never mocked): survives 2s, fast to terminate."""

    def _build_command(self):
        return [__import__("sys").executable, "-c", "import time; time.sleep(2)"]


def test_queue_view_readonly_does_not_recover_processing_rows(tmp_path):
    qdb = tmp_path / "queue.db"
    with UrlQueue(qdb) as q:
        q.add_urls(["https://a.example", "https://b.example"])
        q.mark_processing("https://a.example")

    view = QueueView(qdb)
    states = view.states()
    a = next(s for s in states if s["url"] == "https://a.example")
    # UrlQueue.__init__ would have reverted processing -> pending on open.
    assert a["state"] == "processing"
    assert view.counts() == {"pending": 1, "processing": 1}
    assert view.status_for("https://a.example")["state"] == "processing"
    assert view.status_for("https://missing.example") is None


def test_read_only_conn_rejects_writes(tmp_path):
    qdb = tmp_path / "queue.db"
    with UrlQueue(qdb) as q:
        q.add_urls(["https://a.example"])
    conn = read_only_conn(qdb)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO urls (url, state, attempts, added_at, updated_at)"
                " VALUES ('https://x.example', 'pending', 0, 't', 't')"
            )
            conn.commit()
    finally:
        conn.close()


def test_queue_view_missing_db_is_empty(tmp_path):
    view = QueueView(tmp_path / "nope.db")
    assert view.states() == []
    assert view.counts() == {}
    assert view.status_for("https://a.example") is None


def test_event_log_view_missing_db_is_empty(tmp_path):
    view = EventLogView(tmp_path / "nope.db")
    assert view.recent() == []
    assert view.event_types() == []


def test_event_log_view_recent_ordering_and_filters(tmp_path):
    db = tmp_path / "events.db"
    with FetchLogger(db) as logger:
        logger.log_event("fetch_attempt", url="https://a.example", outcome="ok")
        logger.log_event("extraction_flagged", url="https://a.example", outcome="flagged")
        logger.log_event("fetch_attempt", url="https://b.example", outcome="blocked")

    view = EventLogView(db)
    recent = view.recent()
    assert len(recent) == 3
    # Newest first.
    assert recent[0]["event_type"] == "fetch_attempt"
    assert recent[0]["url"] == "https://b.example"

    only_flags = view.recent(event_types=["extraction_flagged"])
    assert len(only_flags) == 1
    assert only_flags[0]["url"] == "https://a.example"

    only_b = view.recent(url_filter="b.example")
    assert len(only_b) == 1
    assert only_b[0]["event_type"] == "fetch_attempt"

    filtered = view.recent(event_types=["fetch_attempt"], url_filter="a.example")
    assert len(filtered) == 1
    assert filtered[0]["url"] == "https://a.example"

    assert view.event_types() == ["extraction_flagged", "fetch_attempt"]


def test_batch_runner_build_command_uses_sys_executable(tmp_path):
    import sys

    runner = BatchRunner(
        queue_db=tmp_path / "queue.db",
        logs_db=tmp_path / "logs.db",
        urls_file=tmp_path / "urls.txt",
    )
    cmd = runner._build_command()
    assert cmd[0] == sys.executable  # never a bare "python"
    assert cmd[1:3] == ["-m", "orchestrator"]
    assert "run" in cmd
    assert cmd[cmd.index("--db") + 1] == str((tmp_path / "queue.db").resolve())
    assert cmd[cmd.index("--log") + 1] == str((tmp_path / "logs.db").resolve())


def test_spawn_returns_immediately_and_tracks_running(tmp_path):
    runner = SleepRunner(queue_db=tmp_path / "queue.db", run_dir=tmp_path / "runs")
    proc = runner.spawn()
    assert proc.poll() is None  # still running right after spawn
    assert runner.is_running()
    assert runner._marker_path().exists()
    assert runner.run_output.exists()
    runner.stop()
    assert not runner.is_running()
    assert not runner._marker_path().exists()


def test_second_spawn_blocked_while_running_against_same_queue(tmp_path):
    qdb = tmp_path / "queue.db"
    runner = SleepRunner(queue_db=qdb, run_dir=tmp_path / "runs")
    runner.spawn()
    try:
        assert runner.is_running()
        other = SleepRunner(queue_db=qdb, run_dir=tmp_path / "runs")
        with pytest.raises(BatchAlreadyRunning):
            other.spawn()
        # A different corpus (different queue.db) is not blocked.
        other_db = SleepRunner(queue_db=tmp_path / "other.db", run_dir=tmp_path / "runs")
        other_db.spawn()
        try:
            assert other_db.is_running()
        finally:
            other_db.stop()
    finally:
        runner.stop()


def test_stale_marker_is_cleared_and_does_not_block(tmp_path):
    runner = SleepRunner(queue_db=tmp_path / "queue.db", run_dir=tmp_path / "runs")
    marker = runner._marker_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("999999999", encoding="utf-8")
    assert not runner.is_running()  # dead pid -> stale marker
    assert not marker.exists()  # cleared
    runner.spawn()  # must not be blocked
    assert runner.is_running()
    runner.stop()


def test_write_urls_file_filters_comments_and_blanks(tmp_path):
    urls_file = _write_urls_file(
        ["  https://a.example  ", "# a comment", "", "https://b.example"], run_dir=tmp_path
    )
    try:
        lines = urls_file.read_text(encoding="utf-8").splitlines()
        assert lines == ["https://a.example", "https://b.example"]
    finally:
        urls_file.unlink(missing_ok=True)


def test_write_urls_file_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError):
        _write_urls_file(["", "# only a comment"], run_dir=tmp_path)


def test_spawn_ingestion_rejects_no_valid_urls(tmp_path):
    with pytest.raises(ValueError):
        spawn_ingestion([], queue_db=tmp_path / "queue.db", run_dir=tmp_path)
