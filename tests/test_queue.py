"""Hermetic tests for the resumable URL queue (M7 commit 2)."""

from datetime import datetime, timedelta, timezone

import pytest

from orchestrator.queue import UrlQueue

U = "https://example.com/a"
V = "https://example.com/b"
T0 = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)


def make_queue(tmp_path):
    return UrlQueue(tmp_path / "queue.db")


def test_add_and_pending_due(tmp_path):
    q = make_queue(tmp_path)
    assert q.add_urls([U, V]) == 2
    assert q.counts() == {"pending": 2}
    assert q.pending_due() == [U, V]
    q.close()


def test_add_urls_dedupes_within_batch(tmp_path):
    q = make_queue(tmp_path)
    assert q.add_urls([U, U, V, ""]) == 2
    assert q.counts() == {"pending": 2}
    assert q.add_urls([U, "https://example.com/c"]) == 1
    assert q.counts() == {"pending": 3}
    q.close()


def test_mark_processing_increments_attempts_monotonically(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U])
    assert q.mark_processing(U) == 1
    assert q.get(U)["state"] == "processing"
    assert q.mark_processing(U) == 2
    q.close()


def test_mark_processing_unknown_url_raises(tmp_path):
    q = make_queue(tmp_path)
    with pytest.raises(ValueError):
        q.mark_processing(U)
    q.close()


def test_complete_sets_terminal_states(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U, V, "https://example.com/c"])
    urls = q.states()
    a, b, c = urls[0]["url"], urls[1]["url"], urls[2]["url"]
    q.complete(a, "done")
    q.complete(b, "blocked", reason="captcha")
    q.complete(c, "flagged", reason="validation failed")
    assert q.get(a)["state"] == "done"
    assert q.get(b)["state"] == "blocked"
    assert q.get(b)["reason"] == "captcha"
    assert q.get(c)["state"] == "flagged"
    assert q.get(c)["reason"] == "validation failed"
    with pytest.raises(ValueError):
        q.complete(a, "retrying")
    q.close()


def test_record_failure_retries_then_fails_at_bound(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U])
    q.mark_processing(U)
    assert q.record_failure(U, max_attempts=2, backoff_seconds=5, now=T0) == "retrying"
    row = q.get(U)
    assert row["attempts"] == 1
    assert row["next_attempt_ts"] == (T0 + timedelta(seconds=5)).isoformat(timespec="milliseconds")
    assert U not in q.pending_due(now=T0 + timedelta(seconds=4))
    assert q.pending_due(now=T0 + timedelta(seconds=5)) == [U]
    q.mark_processing(U)
    assert q.record_failure(U, max_attempts=2, backoff_seconds=5, now=T0 + timedelta(seconds=6)) == "failed"
    assert q.get(U)["state"] == "failed"
    assert q.counts() == {"failed": 1}
    q.close()


def test_record_failure_terminal_when_max_attempts_is_one(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U])
    q.mark_processing(U)
    assert q.record_failure(U, max_attempts=1, backoff_seconds=5, now=T0) == "failed"
    assert q.get(U)["state"] == "failed"
    assert q.get(U)["next_attempt_ts"] is None
    q.close()


def test_recover_interrupted_reverts_processing_to_pending(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U, V])
    q.mark_processing(U)
    q.mark_processing(V)
    assert q.recover_interrupted() == 2
    assert q.counts() == {"pending": 2}
    assert q.recover_interrupted() == 0
    q.close()


def test_recover_preserves_attempts_and_still_hits_failed_bound(tmp_path):
    """Guard test: a repeatedly-crashing URL must NOT retry forever.

    recover_interrupted() reverts processing -> pending but must not touch
    the monotonic attempt count, so a URL that crashes every attempt still
    reaches the terminal failed state at max_attempts.
    """
    q = make_queue(tmp_path)
    q.add_urls([U])
    q.mark_processing(U)          # attempt 1 in flight -> process dies
    q.recover_interrupted()
    assert q.get(U)["state"] == "pending"
    assert q.get(U)["attempts"] == 1  # preserved, not reset

    q.mark_processing(U)          # attempt 2 in flight -> process dies
    q.recover_interrupted()
    assert q.get(U)["attempts"] == 2  # preserved again

    q.mark_processing(U)          # attempt 3 starts
    assert q.record_failure(U, max_attempts=2, backoff_seconds=5, now=T0) == "failed"
    assert q.get(U)["state"] == "failed"
    assert q.counts() == {"failed": 1}
    assert q.pending_due(now=T0 + timedelta(hours=24)) == []  # never back to retrying
    q.close()


def test_reset_clears_frontier(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U, V])
    q.complete(U, "done")
    assert q.reset() == 2
    assert q.counts() == {}
    assert q.states() == []
    q.close()


def test_states_shape(tmp_path):
    q = make_queue(tmp_path)
    q.add_urls([U])
    q.complete(U, "done", reason="ok")
    row = q.states()[0]
    assert set(row.keys()) == {"url", "state", "attempts", "next_attempt_ts", "reason", "added_at", "updated_at"}
    q.close()
