"""Shared single-URL ingestion pipeline (fetch -> strip -> extract -> validate -> chunk -> embed -> store).

The chain that previously lived inline in pipeline/__main__.py::_run_ingest is
lifted here so the batch orchestrator (M7) and the CLI share exactly one
implementation — Spec req. 7: batch and single-shot share the same schema and
logic, differing only in latency/throughput expectations.

IngestOutcome.status is the queue-transition contract for the orchestrator:

    stored       -> queue 'done'      (terminal: fully ingested)
    blocked      -> queue 'blocked'   (terminal: Rule 7, never retried)
    flagged      -> queue 'flagged'   (terminal: repair budget already spent)
    fetch_failed -> queue retry policy (transient: FAILED/EMPTY fetch)
    no_content   -> queue 'failed'    (immediately terminal, NOT retried:
                  fetch succeeded but yielded no usable content — a
                  deterministic per-page condition, retrying wastes budget)

    stored + written=False means "valid extraction, NOT persisted" — only
    produced by the single-shot live-query path (M8) via write_to_corpus=False.
    The batch runner must never pass write_to_corpus=False; the queue contract
    above assumes storage happened.

fetch_failed vs no_content are distinct statuses produced by real, reachable
code paths: fetch_failed comes from any non-SUCCESS, non-BLOCKED fetch outcome
(FAILED/EMPTY); no_content comes from a SUCCESS fetch whose result carries
neither HTML nor raw bytes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

from agents.config_loader import load_agent_config
from agents.extractor import extract_content, mime_to_content_type
from agents.llm.client import LiteLLMClient
from agents.validator import validate_result
from fetchers.config_loader import load_fetch_config, load_obstacle_config
from fetchers.fetch import fetch_page
from fetchers.logger import FetchLogger
from fetchers.types import FetchOutcome
from pipeline.chunk import chunk_result
from pipeline.config_loader import load_pipeline_config
from pipeline.embed import load_embedder
from pipeline.store import VectorStore, collection_name_for
from pipeline.strip import strip_html
from schemas.extraction import ContentType, ExtractionResult, ValidationResult


@dataclass
class IngestOutcome:
    status: str
    stored_chunks: int = 0
    reason: Optional[str] = None
    fetch_outcome: str = ""
    result: Optional[ExtractionResult] = None
    validation: Optional[ValidationResult] = None
    written: bool = False


def store_result(
    result: ExtractionResult,
    cfg: dict,
    embedder,
    store: VectorStore,
    logger: FetchLogger,
) -> tuple[int, Optional[str]]:
    """Chunk, embed, and store one validated result. Returns (stored, error)."""
    try:
        chunks = chunk_result(result, max_chunk_chars=cfg["chunk"]["max_chunk_chars"], logger=logger)
    except ValueError as exc:
        if logger is not None:
            logger.log_event(
                event_type="chunk_failed",
                url=result.source_url,
                outcome="failed",
                reason=str(exc),
                details={"stage": "chunk"},
            )
        return 0, str(exc)
    try:
        embeddings = embedder.embed([c.content for c in chunks])
    except Exception as exc:
        if logger is not None:
            logger.log_event(
                event_type="chunk_failed",
                url=result.source_url,
                outcome="failed",
                reason=f"embedder failed: {exc}",
                details={"stage": "embed"},
            )
        return 0, f"embedder failed: {exc}"
    try:
        collection = collection_name_for(
            embedder.model_name, embedder.dimension, cfg["store"]["collection_prefix"]
        )
        stored = store.store_chunks(chunks, embeddings, collection_name=collection, logger=logger)
    except Exception as exc:
        if logger is not None:
            logger.log_event(
                event_type="chunk_failed",
                url=result.source_url,
                outcome="failed",
                reason=f"store failed: {exc}",
                details={"stage": "store"},
            )
        return 0, f"store failed: {exc}"
    return stored, None


def ingest_url(
    url: str,
    *,
    mode: str,
    client: LiteLLMClient,
    agent_cfg: dict,
    embedder,
    store: VectorStore,
    pipeline_cfg: dict,
    logger: FetchLogger,
    obstacle_cfg: Optional[dict] = None,
    fetch_cfg: Optional[dict] = None,
    write_to_corpus: bool = True,
    browser=None,
) -> IngestOutcome:
    """Run the full single-URL pipeline and report a queue-mappable outcome.

    write_to_corpus=False runs the same fetch -> strip -> extract -> validate
    chain but skips chunk/embed/store entirely (no compute, no write-back) —
    this is the single-shot live-query path (Spec req. 15). The result is a
    validated ExtractionResult on outcome.result with written=False. Only the
    live-query path sets this; the batch runner always uses the default True
    (queue transitions assume storage happened).
    """
    obstacle_cfg = obstacle_cfg if obstacle_cfg is not None else load_obstacle_config()
    fetch_cfg = fetch_cfg if fetch_cfg is not None else load_fetch_config()
    start = time.monotonic()
    fetched = fetch_page(url, obstacle_cfg=obstacle_cfg, fetch_cfg=fetch_cfg, logger=logger, browser=browser)

    if fetched.outcome != FetchOutcome.SUCCESS:
        if fetched.outcome == FetchOutcome.BLOCKED:
            _log_ingest_lifecycle(logger, url, "fetch_done", start, status="blocked",
                                 fetch_outcome="blocked")
            return IngestOutcome(status="blocked", reason=fetched.reason, fetch_outcome="blocked")
        _log_ingest_lifecycle(logger, url, "fetch_done", start, status="fetch_failed",
                             fetch_outcome=fetched.outcome.value)
        return IngestOutcome(
            status="fetch_failed", reason=fetched.reason, fetch_outcome=fetched.outcome.value
        )

    ctype = mime_to_content_type(fetched.content_type)
    if ctype == ContentType.HTML and fetched.html:
        stripped = strip_html(fetched.html, url=url, logger=logger, browser_visible_text=fetched.visible_text)
        content: object = stripped.text
        page_title = stripped.title
    elif fetched.raw is not None:
        content = fetched.raw
        page_title = None
    else:
        _log_ingest_lifecycle(logger, url, "fetch_done", start, status="no_content")
        return IngestOutcome(
            status="no_content", reason="fetch succeeded but no usable content (no html, no raw bytes)"
        )

    result = extract_content(
        content,
        content_type=ctype,
        source_url=url,
        page_title=page_title,
        mode=mode,
        client=client,
        agent_cfg=agent_cfg,
        logger=logger,
    )
    validation, final_result = validate_result(
        result,
        content=content,
        mode=mode,
        client=client,
        agent_cfg=agent_cfg,
        logger=logger,
        obstacle_cfg=obstacle_cfg,
        skip_repair=not write_to_corpus,
    )
    if not validation.is_valid:
        _log_ingest_lifecycle(logger, url, "validate_done", start, status="flagged",
                             num_sections=len(result.sections))
        return IngestOutcome(
            status="flagged",
            reason=f"validation failed: {len(validation.errors)} errors, repair budget exhausted",
            validation=validation,
        )

    if not write_to_corpus:
        _log_ingest_lifecycle(logger, url, "completed", start, status="stored",
                             num_sections=len(final_result.sections), written=False)
        return IngestOutcome(
            status="stored",
            stored_chunks=0,
            result=final_result,
            written=False,
        )

    _log_ingest_lifecycle(logger, url, "store_start", start,
                         num_sections=len(final_result.sections))
    stored, err = store_result(final_result, pipeline_cfg, embedder, store, logger)
    if err:
        _log_ingest_lifecycle(logger, url, "completed", start, status="flagged",
                             reason=err, num_sections=len(final_result.sections))
        return IngestOutcome(status="flagged", reason=err, result=final_result)
    _log_ingest_lifecycle(logger, url, "completed", start, status="stored",
                         stored_chunks=stored, num_sections=len(final_result.sections))
    return IngestOutcome(status="stored", stored_chunks=stored, result=final_result, written=True)


def _log_ingest_lifecycle(
    logger, url: str, stage: str, start: float, **extra,
) -> None:
    if logger is None:
        return
    details = {"stage": stage, "duration_seconds": round(time.monotonic() - start, 2)}
    details.update(extra)
    logger.log_event(
        "ingest_lifecycle",
        url=url,
        outcome=stage,
        details=details,
    )


def format_ingest_report(url: str, outcome: IngestOutcome) -> tuple[str, int]:
    """(report, exit_code) — byte-identical to the pre-refactor CLI output."""
    if outcome.status in ("blocked", "fetch_failed"):
        return f"fetch outcome={outcome.fetch_outcome} reason={outcome.reason} -> no ingestion", 0
    if outcome.status == "no_content":
        return "fetch succeeded but no content to ingest", 0
    if outcome.status == "flagged" and outcome.result is None:
        report = (
            "validation failed -> not ingested (flagged, see validation_flagged event)\n"
            + json.dumps(outcome.validation.model_dump(mode="json"), indent=2, ensure_ascii=False)
        )
        return report, 0
    result = outcome.result
    report = (
        f"url={url} is_valid=True stored_chunks={outcome.stored_chunks} "
        f"sections={len(result.sections)} confidence={result.confidence}"
    )
    if outcome.reason:
        report += f"\nchunk_failed reason={outcome.reason}"
    return report, 1 if outcome.reason else 0


def format_chunk_report(stored: int, collection: str, err: Optional[str]) -> tuple[str, int]:
    """(report, exit_code) — byte-identical to the pre-refactor chunk CLI output."""
    report = f"stored_chunks={stored} collection={collection}"
    if err:
        report += f"\nchunk_failed reason={err}"
    return report, 1 if err else 0
