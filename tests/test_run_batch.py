"""Hermetic integration tests for run_batch (M7 commit 4).

The ingest callable is injected so no network/LLM/chroma is touched. These
tests lock in: the exact status->state mapping from the M7 plan, orchestrator
visibility in the shared events table (batch_url_state + batch_run), the
single-pass/no-blocking backoff behavior, resume semantics, and batch dedup.
"""

import json

from fetchers.logger import FetchLogger
from orchestrator.queue import UrlQueue
from orchestrator.run_batch import run_batch
from pipeline.ingest import IngestOutcome

RETRY = {"retry": {"max_attempts": 2, "backoff_seconds": 5}}

STORED = "https://a.example/stored"
BLOCKED = "https://a.example/blocked"
FLAGGED = "https://a.example/flagged"
NO_CONTENT = "https://a.example/no-content"
TRANSIENT = "https://a.example/transient"
EXHAUSTED = "https://a.example/exhausted"


def make_context(tmp_path):
    queue = UrlQueue(tmp_path / "queue.db")
    logger = FetchLogger(tmp_path / "events.db")
    return queue, logger


def outcome_for(url):
    return {
        STORED: IngestOutcome(status="stored", stored_chunks=3),
        BLOCKED: IngestOutcome(status="blocked", reason="captcha gate detected"),
        FLAGGED: IngestOutcome(status="flagged", reason="validation failed: 2 errors"),
        NO_CONTENT: IngestOutcome(status="no_content", reason="no usable content"),
        TRANSIENT: IngestOutcome(status="fetch_failed", reason="timeout"),
        EXHAUSTED: IngestOutcome(status="fetch_failed", reason="timeout"),
    }[url]


def run(urls, queue, logger, ingest=None, retry_cfg=RETRY):
    return run_batch(
        urls,
        queue=queue,
        logger=logger,
        agent_cfg={},
        client=None,
        embedder=None,
        store=None,
        pipeline_cfg={},
        retry_cfg=retry_cfg,
        ingest=ingest or (lambda url, **kw: outcome_for(url)),
    )


def test_exact_status_to_state_mapping(tmp_path):
    queue, logger = make_context(tmp_path)
    urls = [STORED, BLOCKED, FLAGGED, NO_CONTENT, TRANSIENT, EXHAUSTED]
    summary = run(urls, queue, logger)
    assert queue.get(STORED)["state"] == "done"
    assert queue.get(BLOCKED)["state"] == "blocked"
    assert queue.get(BLOCKED)["reason"] == "captcha gate detected"
    assert queue.get(FLAGGED)["state"] == "flagged"
    assert queue.get(NO_CONTENT)["state"] == "failed"       # no_content -> failed immediately
    assert queue.get(NO_CONTENT)["reason"].startswith("no_content")
    assert queue.get(TRANSIENT)["state"] == "retrying"      # fetch_failed, attempt 1 < max 2
    assert queue.get(EXHAUSTED)["state"] == "retrying"      # same on first pass
    assert summary["by_state"] == {"done": 1, "blocked": 1, "flagged": 1, "failed": 1, "retrying": 2}
    assert summary["total"] == 6

    # drive the retry bound: second due pass pushes EXHAUSTED to terminal failed
    queue.conn.execute(
        "UPDATE urls SET next_attempt_ts = '2000-01-01T00:00:00.000+00:00' WHERE url = ?", (EXHAUSTED,)
    )
    queue.conn.commit()
    second = run([EXHAUSTED], queue, logger)
    assert queue.get(EXHAUSTED)["state"] == "failed"
    assert queue.get(EXHAUSTED)["attempts"] == 2
    assert second["by_state"] == {"failed": 1}


def test_no_content_fails_without_retry_budget(tmp_path):
    queue, logger = make_context(tmp_path)
    run([NO_CONTENT], queue, logger)
    assert queue.get(NO_CONTENT)["state"] == "failed"
    assert queue.get(NO_CONTENT)["attempts"] == 1
    # deadline-agnostic: even a far-future re-run must not reprocess it
    later = run([NO_CONTENT], queue, logger)
    assert later["total"] == 0


def test_fetch_failed_respects_retry_bound(tmp_path):
    queue, logger = make_context(tmp_path)
    calls = {"n": 0}

    def transient_then_fail(url, **kw):
        calls["n"] += 1
        return IngestOutcome(status="fetch_failed", reason="timeout")

    run([TRANSIENT], queue, logger, ingest=transient_then_fail)
    assert queue.get(TRANSIENT)["state"] == "retrying"
    queue.conn.execute(
        "UPDATE urls SET next_attempt_ts = '2000-01-01T00:00:00.000+00:00' WHERE url = ?", (TRANSIENT,)
    )
    queue.conn.commit()
    run([TRANSIENT], queue, logger, ingest=transient_then_fail)
    assert queue.get(TRANSIENT)["state"] == "failed"
    run([TRANSIENT], queue, logger, ingest=transient_then_fail)
    assert calls["n"] == 2  # terminal failed is never reprocessed
    assert queue.get(TRANSIENT)["attempts"] == 2


def test_retrying_not_due_is_skipped_not_waited(tmp_path):
    """Single-pass guarantee: a retrying row whose deadline hasn't passed is
    skipped entirely — run_batch must not block/sleep on the backoff."""
    queue, logger = make_context(tmp_path)
    calls = {"n": 0}

    def failing(url, **kw):
        calls["n"] += 1
        return IngestOutcome(status="fetch_failed", reason="timeout")

    first = run([TRANSIENT], queue, logger, ingest=failing)
    assert first["total"] == 1
    assert queue.get(TRANSIENT)["state"] == "retrying"

    second = run([TRANSIENT], queue, logger, ingest=failing)
    assert second["total"] == 0          # deadline (now + 5s) not passed: skipped
    assert calls["n"] == 1               # ingest never called again: no waiting/retry loop
    assert queue.get(TRANSIENT)["state"] == "retrying"


def test_retrying_due_is_reprocessed(tmp_path):
    queue, logger = make_context(tmp_path)
    calls = {"n": 0}

    def once_then_ok(url, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return IngestOutcome(status="fetch_failed", reason="timeout")
        return IngestOutcome(status="stored", stored_chunks=2)

    run([TRANSIENT], queue, logger, ingest=once_then_ok)
    assert queue.get(TRANSIENT)["state"] == "retrying"
    # advance the deadline via queue.record_failure's stored next_attempt_ts is
    # in the past of a second run once 5s elapse; simulate by rewriting ts
    queue.conn.execute("UPDATE urls SET next_attempt_ts = '2000-01-01T00:00:00.000+00:00' WHERE url = ?", (TRANSIENT,))
    queue.conn.commit()
    run([TRANSIENT], queue, logger, ingest=once_then_ok)
    assert queue.get(TRANSIENT)["state"] == "done"
    assert calls["n"] == 2


def test_batch_run_and_state_events_in_shared_log(tmp_path):
    queue, logger = make_context(tmp_path)
    run([STORED, BLOCKED], queue, logger)
    events = logger.recent_events(limit=50)
    assert [e["event_type"] for e in events].count("batch_run") == 2
    started = [e for e in events if e["event_type"] == "batch_run" and e["outcome"] == "started"]
    finished = [e for e in events if e["event_type"] == "batch_run" and e["outcome"] == "finished"]
    assert len(started) == 1
    assert len(finished) == 1
    assert json.loads(started[0]["details_json"])["urls_submitted"] == 2

    states = [e for e in events if e["event_type"] == "batch_url_state"]
    assert len(states) == 2
    by_url = {e["url"]: e for e in states}
    assert by_url[STORED]["outcome"] == "done"
    assert by_url[BLOCKED]["outcome"] == "blocked"
    assert by_url[BLOCKED]["reason"] == "captcha gate detected"
    assert json.loads(by_url[BLOCKED]["details_json"])["ingest_status"] == "blocked"
    assert json.loads(finished[0]["details_json"])["total"] == 2


def test_batch_run_events_cover_retry_and_fail_transitions(tmp_path):
    queue, logger = make_context(tmp_path)
    run([TRANSIENT, EXHAUSTED], queue, logger)
    states = {}
    for e in logger.recent_events(limit=50):
        if e["event_type"] == "batch_url_state":
            states.setdefault(e["url"], e)  # newest event per URL (list is newest-first)
    assert states[TRANSIENT]["outcome"] == "retrying"
    assert states[EXHAUSTED]["outcome"] == "retrying"
    queue.conn.execute(
        "UPDATE urls SET next_attempt_ts = '2000-01-01T00:00:00.000+00:00' WHERE url = ?", (EXHAUSTED,)
    )
    queue.conn.commit()
    run([EXHAUSTED], queue, logger)
    states = {}
    for e in logger.recent_events(limit=50):
        if e["event_type"] == "batch_url_state":
            states.setdefault(e["url"], e)
    assert states[EXHAUSTED]["outcome"] == "failed"


def test_resume_does_not_reprocess_completed(tmp_path):
    queue, logger = make_context(tmp_path)
    run([STORED, TRANSIENT], queue, logger)   # stored -> done, transient -> retrying
    assert queue.get(STORED)["state"] == "done"

    # second invocation: stored is done (untouched), transient is not yet due
    second = run([STORED, TRANSIENT], queue, logger)
    assert second["total"] == 0
    assert queue.get(STORED)["state"] == "done"
    assert queue.get(TRANSIENT)["state"] == "retrying"


def test_batch_dedupes_duplicate_urls(tmp_path):
    queue, logger = make_context(tmp_path)
    calls = {"n": 0}

    def counting(url, **kw):
        calls["n"] += 1
        return IngestOutcome(status="stored", stored_chunks=1)

    run([STORED, STORED, STORED], queue, logger, ingest=counting)
    assert calls["n"] == 1  # each unique URL processed once per batch
    assert queue.get(STORED)["state"] == "done"
