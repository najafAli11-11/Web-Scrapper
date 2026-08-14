"""Hybrid live-query path (Spec req. 15, acceptance criterion 6).

Corpus first, live scrape as fallback:

  1. Corpus hit (the URL's chunks are already stored): answer instantly from
     storage. With a query text we do a semantic top-k over that URL's chunks;
     without one we return ALL of the URL's chunks via store.get() — no
     evidence cap (M6 rule: retrieval never silently truncates).
  2. Corpus miss: one single-shot scrape (fetch -> strip -> extract ->
     validate, mode='single') answering from the freshly extracted sections.
     write_to_corpus=False means the shared ingest chain runs with NO chunking,
     NO embedding, and NO write-back — a live answer is never persisted; the
     corpus only changes via explicit batch ingestion (static freshness).

Two naming/selection rules locked here:
  - A successful live scrape is reported as status "single_shot_ok", NEVER the
    raw ingest status "stored" — nobody should mistake a live answer for a
    persisted one. The other live outcomes surface with their ingest statuses
    (blocked/flagged/no_content/fetch_failed) verbatim.
  - The Chroma collection is always derived from the INJECTED embedder's
    model_name + dimension (collection_name_for), never a hardcoded default —
    a future embedding-model config change cannot silently query the wrong or
    a nonexistent collection.

Pure control flow (Rule 1): live_query contains no LLM calls itself; it
orchestrates the same shared fetch/extract/validate agents as the batch path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from fetchers.logger import FetchLogger
from pipeline.ingest import IngestOutcome, ingest_url
from pipeline.store import corpus_collection, row_provenance
from schemas.extraction import ExtractionResult, Section

LIVE_OUTCOME_STATUSES = ("blocked", "flagged", "no_content", "fetch_failed")


@dataclass
class Evidence:
    text: str
    provenance: dict


@dataclass
class LiveQueryResult:
    url: str
    found_in_corpus: bool
    source_used: str            # "corpus" | "live_scrape"
    status: str                 # ok | single_shot_ok | blocked | flagged | no_content | fetch_failed
    provenance: dict
    evidence: list[Evidence] = field(default_factory=list)
    query: Optional[str] = None
    reason: Optional[str] = None


def _row_provenance(meta: dict) -> dict:
    """Provenance for a corpus chunk row (shared helper, see pipeline.store.row_provenance)."""
    return row_provenance(meta)


def _section_provenance(result: ExtractionResult, section: Section) -> dict:
    """Provenance for a freshly extracted section (live scrape)."""
    prov: dict = {
        "source_url": result.source_url,
        "scrape_timestamp": result.scrape_timestamp.isoformat(),
    }
    if result.page_title is not None:
        prov["page_title"] = result.page_title
    if section.heading is not None:
        prov["section_heading"] = section.heading
    return prov


def _corpus_provenance(url: str, meta: dict) -> dict:
    prov: dict = {"source_url": url, "scrape_timestamp": meta.get("scrape_timestamp")}
    if meta.get("page_title") is not None:
        prov["page_title"] = meta["page_title"]
    return prov


def _collection(embedder, pipeline_cfg: dict) -> str:
    """Collection is ALWAYS derived from the injected embedder's model + dimension."""
    return corpus_collection(embedder, pipeline_cfg)


def live_query(
    url: str,
    *,
    query: Optional[str] = None,
    logger: FetchLogger,
    agent_cfg: dict,
    client,
    embedder,
    store,
    pipeline_cfg: dict,
    mode: str = "single",
    ingest: Callable = ingest_url,
    k: int = 5,
    obstacle_cfg: Optional[dict] = None,
    fetch_cfg: Optional[dict] = None,
) -> LiveQueryResult:
    """Answer for one URL: corpus first, single-shot scrape on miss. No write-back."""
    collection = _collection(embedder, pipeline_cfg)
    present = store.count(collection_name=collection, where={"source_url": url})

    if present > 0:
        if query:
            query_embedding = embedder.embed([query])[0]
            rows = store.query(
                query_embedding, k=k, collection_name=collection, where={"source_url": url}
            )
        else:
            rows = store.get(collection_name=collection, where={"source_url": url})
        evidence = [Evidence(text=r["document"], provenance=_row_provenance(r["metadata"])) for r in rows]
        provenance = _corpus_provenance(url, rows[0]["metadata"]) if rows else {"source_url": url}
        logger.log_event(
            "live_query",
            url=url,
            outcome="corpus_hit",
            details={
                "found_in_corpus": True,
                "source_used": "corpus",
                "evidence_count": len(evidence),
                "query_given": query is not None,
            },
        )
        return LiveQueryResult(
            url=url,
            query=query,
            found_in_corpus=True,
            source_used="corpus",
            status="ok",
            provenance=provenance,
            evidence=evidence,
        )

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
        write_to_corpus=False,
    )

    if outcome.status == "stored":
        # successful live scrape: report single_shot_ok, never the raw ingest status
        if outcome.result is None:
            status, reason = "flagged", "stored outcome without a result"
            event_outcome = "corpus_miss_single_shot_flagged"
            evidence: list[Evidence] = []
            provenance = {"source_url": url}
        else:
            status, reason = "single_shot_ok", None
            event_outcome = "corpus_miss_single_shot_ok"
            result = outcome.result
            evidence = [
                Evidence(text=s.content, provenance=_section_provenance(result, s))
                for s in result.sections
            ]
            provenance = {
                "source_url": result.source_url,
                "scrape_timestamp": result.scrape_timestamp.isoformat(),
            }
            if result.page_title is not None:
                provenance["page_title"] = result.page_title
    else:
        # blocked / flagged / no_content / fetch_failed surface verbatim
        status, reason = outcome.status, outcome.reason
        event_outcome = f"corpus_miss_single_shot_{outcome.status}"
        evidence = []
        provenance = {"source_url": url}

    logger.log_event(
        "live_query",
        url=url,
        outcome=event_outcome,
        reason=reason,
        details={
            "found_in_corpus": False,
            "source_used": "live_scrape",
            "evidence_count": len(evidence),
            "query_given": query is not None,
            "ingest_status": outcome.status,
        },
    )
    return LiveQueryResult(
        url=url,
        query=query,
        found_in_corpus=False,
        source_used="live_scrape",
        status=status,
        provenance=provenance,
        evidence=evidence,
        reason=reason,
    )
