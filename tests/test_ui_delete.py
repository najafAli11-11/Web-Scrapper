"""Hermetic tests for URL deletion and log clearing in the UI data layer."""

import sqlite3

import pytest

from fetchers.logger import FetchLogger
from orchestrator.queue import UrlQueue
from pipeline.store import VectorStore
from ui.db_view import (
    clear_all_logs,
    delete_url_everywhere,
    delete_url_from_queue,
    delete_url_logs,
)


def _setup_queue(qdb, urls):
    """Helper: populate a queue.db with the given URLs in 'done' state."""
    with UrlQueue(qdb) as q:
        q.add_urls(urls)
        for url in urls:
            q.complete(url, "done")
    return q


def _setup_logs(ldb, urls):
    """Helper: populate logs.db with events for each URL."""
    with FetchLogger(ldb) as logger:
        for url in urls:
            logger.log_event("fetch_attempt", url=url, outcome="ok")
            logger.log_event("extraction_attempt", url=url, outcome="extracted")
            logger.log_event("ingest_lifecycle", url=url, outcome="completed")
    return ldb


def _event_count(ldb):
    """Count total events in logs.db."""
    conn = sqlite3.connect(str(ldb))
    try:
        return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# delete_url_from_queue
# ---------------------------------------------------------------------------


def test_delete_url_from_queue_removes_row(tmp_path):
    qdb = tmp_path / "queue.db"
    _setup_queue(qdb, ["https://a.example", "https://b.example"])

    deleted = delete_url_from_queue("https://a.example", db_path=qdb)
    assert deleted == 1

    view_conn = sqlite3.connect(str(qdb))
    rows = view_conn.execute("SELECT url FROM urls ORDER BY url").fetchall()
    view_conn.close()
    assert [r[0] for r in rows] == ["https://b.example"]


def test_delete_url_from_queue_nonexistent_is_zero(tmp_path):
    qdb = tmp_path / "queue.db"
    _setup_queue(qdb, ["https://a.example"])

    deleted = delete_url_from_queue("https://missing.example", db_path=qdb)
    assert deleted == 0


def test_delete_url_from_queue_empty_db(tmp_path):
    qdb = tmp_path / "queue.db"
    deleted = delete_url_from_queue("https://a.example", db_path=qdb)
    assert deleted == 0


# ---------------------------------------------------------------------------
# delete_url_logs
# ---------------------------------------------------------------------------


def test_delete_url_logs_removes_only_matching_events(tmp_path):
    ldb = tmp_path / "logs.db"
    _setup_logs(ldb, ["https://a.example", "https://b.example"])

    assert _event_count(ldb) == 6

    deleted = delete_url_logs("https://a.example", db_path=ldb)
    assert deleted == 3

    remaining = _event_count(ldb)
    assert remaining == 3

    view_conn = sqlite3.connect(str(ldb))
    urls_left = [r[0] for r in view_conn.execute("SELECT DISTINCT url FROM events").fetchall()]
    view_conn.close()
    assert urls_left == ["https://b.example"]


def test_delete_url_logs_nonexistent_is_zero(tmp_path):
    ldb = tmp_path / "logs.db"
    _setup_logs(ldb, ["https://a.example"])

    deleted = delete_url_logs("https://missing.example", db_path=ldb)
    assert deleted == 0


def test_delete_url_logs_empty_db(tmp_path):
    ldb = tmp_path / "logs.db"
    deleted = delete_url_logs("https://a.example", db_path=ldb)
    assert deleted == 0


# ---------------------------------------------------------------------------
# clear_all_logs
# ---------------------------------------------------------------------------


def test_clear_all_logs_empties_table(tmp_path):
    ldb = tmp_path / "logs.db"
    _setup_logs(ldb, ["https://a.example", "https://b.example"])
    assert _event_count(ldb) == 6

    deleted = clear_all_logs(db_path=ldb)
    assert deleted == 6
    assert _event_count(ldb) == 0


def test_clear_all_logs_empty_db(tmp_path):
    ldb = tmp_path / "logs.db"
    deleted = clear_all_logs(db_path=ldb)
    assert deleted == 0


# ---------------------------------------------------------------------------
# delete_url_everywhere (integration)
# ---------------------------------------------------------------------------


def _make_store(tmp_path):
    """Create a minimal VectorStore for testing."""
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    return VectorStore(str(chroma_dir), collection_prefix="test")


def test_delete_url_everywhere_removes_from_all_stores(tmp_path):
    qdb = tmp_path / "queue.db"
    ldb = tmp_path / "logs.db"

    _setup_queue(qdb, ["https://a.example", "https://b.example"])
    _setup_logs(ldb, ["https://a.example", "https://b.example"])

    store = _make_store(tmp_path)
    collection = "test_BAAI_bge_m3_1024"

    result = delete_url_everywhere(
        "https://a.example",
        queue_db=qdb,
        logs_db=ldb,
        collection_name=collection,
        store=store,
    )

    assert result["queue_deleted"] == 1
    assert result["logs_deleted"] == 3

    # Verify queue
    view_conn = sqlite3.connect(str(qdb))
    queue_urls = [r[0] for r in view_conn.execute("SELECT url FROM urls ORDER BY url").fetchall()]
    view_conn.close()
    assert queue_urls == ["https://b.example"]

    # Verify logs
    remaining_events = _event_count(ldb)
    assert remaining_events == 3

    log_conn = sqlite3.connect(str(ldb))
    log_urls = [r[0] for r in log_conn.execute("SELECT DISTINCT url FROM events").fetchall()]
    log_conn.close()
    assert log_urls == ["https://b.example"]


def test_delete_url_everywhere_nonexistent_url(tmp_path):
    qdb = tmp_path / "queue.db"
    ldb = tmp_path / "logs.db"

    _setup_queue(qdb, ["https://a.example"])
    _setup_logs(ldb, ["https://a.example"])

    store = _make_store(tmp_path)
    collection = "test_BAAI_bge_m3_1024"

    result = delete_url_everywhere(
        "https://missing.example",
        queue_db=qdb,
        logs_db=ldb,
        collection_name=collection,
        store=store,
    )

    assert result["queue_deleted"] == 0
    assert result["logs_deleted"] == 0
    assert _event_count(ldb) == 3
