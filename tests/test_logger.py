"""Hermetic tests for FetchLogger thread handling (M9 UI cross-thread reuse)."""

import threading

from fetchers.logger import FetchLogger


def _run_in_thread(fn):
    def worker():
        try:
            fn()
        except Exception:  # noqa: BLE001 - failures surface in the caller's assertions
            pass

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=10)


def test_logger_with_check_same_thread_false_usable_across_threads(tmp_path):
    logger = FetchLogger(tmp_path / "events.db", check_same_thread=False)
    try:
        _run_in_thread(lambda: logger.log_event("fetch_attempt", url="u", outcome="ok"))
        rows = logger.recent_events()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "fetch_attempt"
    finally:
        logger.close()


def test_logger_default_drops_cross_thread_events(tmp_path):
    """Default (thread-bound) mode: a cross-thread write is silently dropped by
    best-effort logging — the exact failure the M9 UI hit before
    check_same_thread=False."""
    logger = FetchLogger(tmp_path / "events.db")
    try:
        _run_in_thread(lambda: logger.log_event("fetch_attempt", url="u", outcome="ok"))
        assert len(logger.recent_events()) == 0, "thread-bound logger must drop cross-thread events"
    finally:
        logger.close()
