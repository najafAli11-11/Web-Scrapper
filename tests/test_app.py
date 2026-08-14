"""Hermetic AppTest smoke tests for the M9 Streamlit UI.

The app is rendered with env-overridden scratch (empty) DB paths, so missing
DBs must render gracefully (empty queue/logs, no exception) without loading
the embedder or opening a real corpus. Auto-refresh must NOT spin when no
batch is running (resolution #4: refresh is bounded by the child's exit).
"""

import os

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = "ui/app.py"


@pytest.fixture
def scratch_env(tmp_path, monkeypatch):
    monkeypatch.setenv("UI_QUEUE_DB", str(tmp_path / "queue.db"))
    monkeypatch.setenv("UI_LOGS_DB", str(tmp_path / "logs.db"))
    monkeypatch.setenv("UI_CHROMA_PATH", str(tmp_path / "chroma"))
    return tmp_path


def test_app_renders_three_tabs_without_exception(scratch_env):
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert len(at.exception) == 0
    assert [t.label for t in at.tabs] == ["Chat", "Ingestion", "Logs"]
    # Ingestion shows the empty-queue hint, not a crash.
    assert any("Queue is empty" in str(info.value) for info in at.info)
    # Chat inputs present (corpus question, URL live query).
    keys = {ti.key for ti in at.text_input}
    assert {"chat_question", "chat_url", "chat_url_query"} <= keys


def test_app_no_auto_refresh_when_no_batch_running(scratch_env):
    """Resolution #4: with no batch child alive, auto-refresh must not arm
    the sleep+rerun loop (the run completes; the script would otherwise hang
    on an unconditional refresh)."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert len(at.exception) == 0
    # No rerun was scheduled: the run completed once and session state never
    # armed auto-refresh.
    assert "auto_refresh" not in at.session_state or at.session_state["auto_refresh"] is False


def test_app_renders_events_and_queue_when_present(scratch_env, tmp_path):
    from fetchers.logger import FetchLogger
    from orchestrator.queue import UrlQueue
    from ui.db_view import BatchRunner

    # Seed scratch DBs so the views have real data to render.
    with UrlQueue(tmp_path / "queue.db") as q:
        q.add_urls(["https://a.example"])
        q.complete("https://a.example", "done")
    with FetchLogger(tmp_path / "logs.db") as logger:
        logger.log_event("fetch_attempt", url="https://a.example", outcome="ok")
        logger.log_event("extraction_flagged", url="https://a.example", outcome="flagged")

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert len(at.exception) == 0
    assert any("done" in str(metric.value) for metric in at.metric) or any(
        "done" in str(i.value) for i in at.markdown
    )
