"""Hermetic tests for the hybrid live-query path (M8 commit 3, Spec req. 15).

Store/embedder/ingest are injected; only the events DB is real. Locks in:
corpus-first behavior, single_shot_ok status naming (never the raw ingest
"stored"), collection selection driven by the injected embedder's model+dim,
no-cap whole-URL retrieval on a no-query hit, no write-back on a miss, and
the closed enum of miss-branch event names.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fetchers.logger import FetchLogger
from orchestrator.live_query import LIVE_OUTCOME_STATUSES, live_query
from pipeline.ingest import IngestOutcome
from schemas.extraction import ContentType, ExtractionResult, Section

URL = "https://example.com/page"
TS = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)
CFG = {"store": {"collection_prefix": "corpus"}}
COLLECTION = "corpus_fake_model_4"  # embedder.model_name=fake-model, dimension=4


def make_result(sections=3):
    return ExtractionResult(
        source_url=URL,
        scrape_timestamp=TS,
        page_title="Page",
        content_type=ContentType.HTML,
        sections=[Section(heading=f"H{i}", content=f"section {i} content") for i in range(sections)],
        confidence=0.9,
    )


def make_chunk_rows(n, url=URL):
    return [
        {
            "id": f"{url}#{i}",
            "document": f"chunk {i} of {url}",
            "metadata": {
                "source_url": url,
                "scrape_timestamp": TS.isoformat(),
                "page_title": "Page",
                "section_heading": f"H{i}",
                "chunk_index": 1,
                "chunk_total": 1,
            },
            "distance": i / 10.0,
        }
        for i in range(n)
    ]


class FakeEmbedder:
    model_name = "fake-model"
    dimension = 4

    def __init__(self):
        self.embed_calls = []

    def embed(self, texts):
        self.embed_calls.append(texts)
        return [[0.25] * 4 for _ in texts]


class FakeStore:
    def __init__(self, count=0, rows=None):
        self.count_result = count
        self.rows = rows or []
        self.calls = {"count": [], "query": [], "get": []}

    def count(self, *, collection_name, where=None):
        self.calls["count"].append((collection_name, where))
        return self.count_result

    def query(self, embedding, k, *, collection_name, where=None):
        self.calls["query"].append((collection_name, where, k))
        return self.rows

    def get(self, *, collection_name, where=None):
        self.calls["get"].append((collection_name, where))
        return self.rows


def run(url=URL, *, query=None, store=None, ingest=None, embedder=None, logger=None, tmp_path=None):
    store = store or FakeStore()
    embedder = embedder or FakeEmbedder()
    logger = logger or FetchLogger(tmp_path / "events.db")
    result = live_query(
        url,
        query=query,
        logger=logger,
        agent_cfg={},
        client=None,
        embedder=embedder,
        store=store,
        pipeline_cfg=CFG,
        ingest=ingest or (lambda *a, **k: IngestOutcome(status="stored", result=make_result(), written=False)),
    )
    return result, store, embedder, logger


# -- corpus hit --------------------------------------------------------------

def test_corpus_hit_with_query_returns_topk(tmp_path):
    store = FakeStore(count=3, rows=make_chunk_rows(3))
    result, store, embedder, _ = run(query="what is scraping?", store=store, tmp_path=tmp_path)
    assert result.found_in_corpus is True
    assert result.source_used == "corpus"
    assert result.status == "ok"
    assert len(result.evidence) == 3
    assert result.evidence[0].text == "chunk 0 of https://example.com/page"
    assert result.evidence[0].provenance["source_url"] == URL
    assert result.evidence[0].provenance["scrape_timestamp"] == TS.isoformat()
    assert result.evidence[0].provenance["page_title"] == "Page"
    assert result.evidence[0].provenance["section_heading"] == "H0"
    assert embedder.embed_calls == [["what is scraping?"]]
    assert store.calls["query"][0][0] == COLLECTION  # collection from injected embedder
    assert store.calls["query"][0][1] == {"source_url": URL}
    assert store.calls["get"] == []


def test_corpus_hit_without_query_returns_all_chunks_no_cap(tmp_path):
    store = FakeStore(count=6, rows=make_chunk_rows(6))
    result, store, _, _ = run(store=store, tmp_path=tmp_path)
    assert result.found_in_corpus is True
    assert result.status == "ok"
    assert len(result.evidence) == 6          # more than any default k: no cap
    assert store.calls["get"][0][0] == COLLECTION
    assert store.calls["query"] == []
    assert store.calls["get"][0][1] == {"source_url": URL}


def test_corpus_hit_logs_event(tmp_path):
    logger = FetchLogger(tmp_path / "events.db")
    run(store=FakeStore(count=1, rows=make_chunk_rows(1)), logger=logger, tmp_path=tmp_path)
    events = [e for e in logger.recent_events() if e["event_type"] == "live_query"]
    assert len(events) == 1
    assert events[0]["outcome"] == "corpus_hit"
    assert events[0]["url"] == URL
    assert json.loads(events[0]["details_json"])["found_in_corpus"] is True
    assert json.loads(events[0]["details_json"])["evidence_count"] == 1


# -- corpus miss -------------------------------------------------------------

def test_corpus_miss_single_shot_ok_without_write_back(tmp_path):
    calls = {"n": 0, "kwargs": None}

    def ing(url, **kw):
        calls["n"] += 1
        calls["kwargs"] = kw
        return IngestOutcome(status="stored", result=make_result(), written=False)

    store = FakeStore(count=0)
    result, store, _, _ = run(ingest=ing, store=store, tmp_path=tmp_path)
    assert result.found_in_corpus is False
    assert result.source_used == "live_scrape"
    assert result.status == "single_shot_ok"   # never the raw ingest status "stored"
    assert calls["n"] == 1
    assert calls["kwargs"]["write_to_corpus"] is False   # no write-back
    assert len(result.evidence) == 3
    assert result.evidence[0].provenance["source_url"] == URL
    assert result.evidence[0].provenance["scrape_timestamp"] == TS.isoformat()
    assert result.evidence[0].provenance["page_title"] == "Page"
    assert result.evidence[0].provenance["section_heading"] == "H0"
    assert result.provenance["source_url"] == URL


def test_corpus_miss_statuses_surface_with_sensible_event_names(tmp_path):
    """The miss branch maps ingest statuses to event names via interpolation.

    ingest_url's status enum is closed and tested (test_ingest.py): stored is
    handled explicitly as single_shot_ok; the other four produce explicit
    corpus_miss_single_shot_<status> names. This test is the sanity check that
    every one of those names is exactly what a reader expects.
    """
    expected = {
        "blocked": "corpus_miss_single_shot_blocked",
        "flagged": "corpus_miss_single_shot_flagged",
        "no_content": "corpus_miss_single_shot_no_content",
        "fetch_failed": "corpus_miss_single_shot_fetch_failed",
    }
    assert set(expected) == set(LIVE_OUTCOME_STATUSES)

    for status, event_name in expected.items():
        logger = FetchLogger(tmp_path / f"events_{status}.db")

        def ing(url, _status=status, **kw):
            return IngestOutcome(status=_status, reason=f"reason-{_status}")

        result, _, _, _ = run(ingest=ing, store=FakeStore(count=0), logger=logger, tmp_path=tmp_path)
        assert result.status == status
        assert result.reason == f"reason-{status}"
        assert result.evidence == []
        assert result.provenance == {"source_url": URL}
        events = [e for e in logger.recent_events() if e["event_type"] == "live_query"]
        assert events[0]["outcome"] == event_name, f"{status} -> {event_name}"


def test_corpus_miss_logs_single_shot_ok_event(tmp_path):
    logger = FetchLogger(tmp_path / "events.db")
    run(logger=logger, tmp_path=tmp_path)
    events = [e for e in logger.recent_events() if e["event_type"] == "live_query"]
    assert len(events) == 1
    assert events[0]["outcome"] == "corpus_miss_single_shot_ok"
    assert json.loads(events[0]["details_json"])["found_in_corpus"] is False
    assert json.loads(events[0]["details_json"])["source_used"] == "live_scrape"
