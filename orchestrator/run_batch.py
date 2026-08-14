"""Deterministic batch runner wiring the URL queue to the shared ingest pipeline (Spec req. 6, Rule 1).

Pure control flow: run_batch contains no LLM calls. Each URL goes through the
single shared ingest_url() (pipeline/ingest.py, Spec req. 7) and the outcome
is mapped onto a queue state per the M7 plan:

    stored       -> done          (terminal)
    blocked      -> blocked       (terminal; Rule 7, never retried)
    flagged      -> flagged       (terminal; repair budget already spent)
    no_content   -> failed        (immediately terminal, NOT retried)
    fetch_failed -> retrying when attempts < max_attempts, else failed

Single-pass by design: run_batch never sleeps or blocks on a backoff
deadline. pending_due() returns pending rows plus retrying rows whose
deadline has passed; anything else stays in the queue and is picked up on a
later invocation (resumable, Rule 6).

Every transition is mirrored into the shared SQLite events table (the same
--log DB used by M6's fetch/validation events) so the M9 logs view sees the
orchestrator: one batch_url_state event per URL transition (outcome = new
state, reason = the block/flag/fail reason) plus one batch_run
started|finished summary per invocation.
"""

from __future__ import annotations

from typing import Callable, Optional

from fetchers.logger import FetchLogger
from orchestrator.queue import UrlQueue
from pipeline.ingest import IngestOutcome, ingest_url


def _transition(
    outcome: IngestOutcome,
    attempts: int,
    retry_cfg: dict,
    queue: UrlQueue,
    url: str,
) -> str:
    """Apply the plan's status->state mapping. Returns the new queue state."""
    retry = retry_cfg["retry"]
    if outcome.status == "stored":
        queue.complete(url, "done")
        return "done"
    if outcome.status == "blocked":
        queue.complete(url, "blocked", reason=outcome.reason)
        return "blocked"
    if outcome.status == "flagged":
        queue.complete(url, "flagged", reason=outcome.reason)
        return "flagged"
    if outcome.status == "no_content":
        reason = f"no_content: {outcome.reason}" if outcome.reason else "no_content"
        queue.complete(url, "failed", reason=reason)
        return "failed"
    # fetch_failed: retry while budget remains, else terminal failed
    return queue.record_failure(
        url,
        max_attempts=int(retry["max_attempts"]),
        backoff_seconds=float(retry["backoff_seconds"]),
    )


def run_batch(
    urls: list[str],
    *,
    queue: UrlQueue,
    logger: FetchLogger,
    agent_cfg: dict,
    client,
    embedder,
    store,
    pipeline_cfg: dict,
    retry_cfg: dict,
    mode: str = "batch",
    ingest: Callable = ingest_url,
    obstacle_cfg: Optional[dict] = None,
    fetch_cfg: Optional[dict] = None,
) -> dict:
    """Run one single-pass batch over pending/due URLs. Returns a summary dict."""
    added = queue.add_urls(urls)
    logger.log_event(
        "batch_run",
        outcome="started",
        details={"urls_submitted": len(urls), "newly_queued": added},
    )

    summary: dict = {"total": 0, "by_state": {}}
    for url in queue.pending_due():
        attempts = queue.mark_processing(url)
        summary["total"] += 1
        outcome = ingest(
            url,
            mode=mode,
            client=client,
            agent_cfg=agent_cfg,
            embedder=embedder,
            store=store,
            pipeline_cfg=pipeline_cfg,
            logger=logger,
            obstacle_cfg=obstacle_cfg,
            fetch_cfg=fetch_cfg,
        )
        state = _transition(outcome, attempts, retry_cfg, queue, url)
        row = queue.get(url)
        logger.log_event(
            "batch_url_state",
            url=url,
            outcome=state,
            reason=row["reason"] if row else None,
            details={"attempt": attempts, "ingest_status": outcome.status},
        )
        summary["by_state"][state] = summary["by_state"].get(state, 0) + 1

    logger.log_event("batch_run", outcome="finished", details=summary)
    return summary
