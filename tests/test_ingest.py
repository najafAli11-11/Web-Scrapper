"""Hermetic tests for the shared single-URL pipeline (M7 commit 3).

ingest_url() is the single implementation shared by the CLI and the M7 batch
orchestrator (Spec req. 7). These tests lock in the IngestOutcome status
contract the queue-transition mapping depends on: stored / blocked / flagged /
fetch_failed / no_content — each produced by a real code path, with
fetch_failed and no_content kept distinct.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

from fetchers.logger import FetchLogger
from fetchers.types import FetchOutcome, FetchResult
from pipeline import ingest as ing
from pipeline.ingest import (
    IngestOutcome,
    format_chunk_report,
    format_ingest_report,
    ingest_url,
    store_result,
)
from schemas.extraction import ContentType, ExtractionResult, Section, ValidationResult

U = "https://example.com/a"
TS = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)
CFG = {"chunk": {"max_chunk_chars": 2000}, "store": {"collection_prefix": "corpus"}}


def make_result(sections=2, confidence=0.9):
    return ExtractionResult(
        source_url=U,
        scrape_timestamp=TS,
        page_title="Page",
        content_type=ContentType.HTML,
        sections=[Section(heading=f"H{i}", content=f"section {i} content " * 3) for i in range(sections)],
        confidence=confidence,
    )


def make_validation(valid=True):
    return ValidationResult(
        source_url=U,
        scrape_timestamp=TS,
        is_valid=valid,
        errors=[] if valid else ["content looks empty"],
        should_retry=not valid,
        retry_count=0,
    )


class FakeEmbedder:
    model_name = "fake-model"
    dimension = 4

    def embed(self, texts):
        return [[0.25] * 4 for _ in texts]


class FakeStore:
    def __init__(self, stored=2):
        self.stored = stored
        self.store_calls = 0

    def store_chunks(self, chunks, embeddings, *, collection_name, logger=None):
        self.store_calls += 1
        return self.stored


def fetch_result(outcome, **kw):
    return FetchResult(url=U, outcome=outcome, **kw)


def stub_valid_pipeline(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.SUCCESS, html="<html><body>x</body></html>", content_type="text/html"
    ))
    monkeypatch.setattr(ing, "strip_html", lambda html, url=None, logger=None, browser_visible_text=None: SimpleNamespace(text="stripped", title="Page"))
    monkeypatch.setattr(ing, "extract_content", lambda *a, **k: make_result())
    monkeypatch.setattr(ing, "validate_result", lambda *a, **k: (make_validation(True), make_result()))


def run(url=U, **kw):
    return ingest_url(
        url,
        mode=kw.get("mode", "batch"),
        client=None,
        agent_cfg={},
        embedder=kw.get("embedder", FakeEmbedder()),
        store=kw.get("store", FakeStore()),
        pipeline_cfg=CFG,
        logger=kw.get("logger"),
        write_to_corpus=kw.get("write_to_corpus", True),
    )


# -- statuses from real fetch outcomes -------------------------------------

def test_blocked_fetch_is_terminal(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.BLOCKED, reason="captcha gate detected"
    ))
    out = run()
    assert out.status == "blocked"
    assert out.fetch_outcome == "blocked"
    assert out.reason == "captcha gate detected"


def test_failed_fetch_is_fetch_failed(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.FAILED, reason="timeout after retries"
    ))
    out = run()
    assert out.status == "fetch_failed"
    assert out.fetch_outcome == "failed"


def test_empty_fetch_is_fetch_failed(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.EMPTY, reason="content looks like a JS shell"
    ))
    out = run()
    assert out.status == "fetch_failed"
    assert out.fetch_outcome == "empty"


def test_no_content_when_success_has_no_payload(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.SUCCESS, html=None, raw=None, content_type="text/plain"
    ))
    out = run()
    assert out.status == "no_content"
    assert out.stored_chunks == 0


def test_pdf_content_routes_to_raw_path(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.SUCCESS, html=None, raw=b"%PDF-1.4 fake", content_type="application/pdf"
    ))
    monkeypatch.setattr(ing, "extract_content", lambda *a, **k: make_result())
    monkeypatch.setattr(ing, "validate_result", lambda *a, **k: (make_validation(True), make_result()))
    out = run()
    assert out.status == "stored"
    assert out.stored_chunks == 2


# -- post-fetch statuses -----------------------------------------------------

def test_flagged_when_validation_fails(monkeypatch):
    stub_valid_pipeline(monkeypatch)
    monkeypatch.setattr(ing, "validate_result", lambda *a, **k: (make_validation(False), make_result()))
    out = run()
    assert out.status == "flagged"
    assert out.result is None
    assert out.validation is not None
    assert not out.validation.is_valid


def test_stored_on_success(monkeypatch):
    stub_valid_pipeline(monkeypatch)
    out = run()
    assert out.status == "stored"
    assert out.stored_chunks == 2
    assert out.result is not None


# -- single-shot / write_to_corpus=False (M8 live-query path) -----------------

def test_write_to_corpus_false_skips_storage(monkeypatch):
    stub_valid_pipeline(monkeypatch)
    store = FakeStore()
    out = run(store=store, write_to_corpus=False)
    assert out.status == "stored"
    assert out.written is False          # valid extraction, NOT persisted
    assert out.stored_chunks == 0
    assert out.result is not None        # answer material available to live-query
    assert store.store_calls == 0        # chunk/embed/store never touched


def test_write_to_corpus_true_writes_by_default(monkeypatch):
    stub_valid_pipeline(monkeypatch)
    store = FakeStore()
    out = run(store=store)
    assert out.status == "stored"
    assert out.written is True
    assert out.stored_chunks == 2
    assert store.store_calls == 1


def test_write_to_corpus_false_still_flags_on_validation_failure(monkeypatch):
    stub_valid_pipeline(monkeypatch)
    monkeypatch.setattr(ing, "validate_result", lambda *a, **k: (make_validation(False), make_result()))
    store = FakeStore()
    out = run(store=store, write_to_corpus=False)
    assert out.status == "flagged"
    assert out.written is False
    assert store.store_calls == 0


def test_write_to_corpus_false_unchanged_for_blocked_and_no_content(monkeypatch):
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.BLOCKED, reason="captcha gate detected"
    ))
    out = run(write_to_corpus=False)
    assert out.status == "blocked"

    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.SUCCESS, html=None, raw=None, content_type="text/plain"
    ))
    out = run(write_to_corpus=False)
    assert out.status == "no_content"


def test_flagged_when_chunking_fails_but_keeps_result(monkeypatch):
    stub_valid_pipeline(monkeypatch)

    def boom(result, max_chunk_chars=2000, **kwargs):
        raise ValueError("no sections to chunk: extraction result is empty")

    monkeypatch.setattr(ing, "chunk_result", boom)
    out = run()
    assert out.status == "flagged"
    assert out.reason == "no sections to chunk: extraction result is empty"
    assert out.result is not None  # validation passed; the record is flagged at the store step


# -- store_result ------------------------------------------------------------

def test_store_result_success(tmp_path):
    logger = FetchLogger(tmp_path / "events.db")
    stored, err = store_result(make_result(), CFG, FakeEmbedder(), FakeStore(stored=2), logger)
    assert stored == 2
    assert err is None


def test_store_result_logs_chunk_failed_event(tmp_path, monkeypatch):
    logger = FetchLogger(tmp_path / "events.db")

    def boom(result, max_chunk_chars=2000, **kwargs):
        raise ValueError("no sections to chunk: extraction result is empty")

    monkeypatch.setattr(ing, "chunk_result", boom)
    stored, err = store_result(make_result(), CFG, FakeEmbedder(), FakeStore(), logger)
    assert stored == 0
    assert "no sections to chunk" in err
    assert any(r["event_type"] == "chunk_failed" for r in logger.recent_events())


# -- report formatters (byte-identical CLI output) ----------------------------

def test_format_ingest_report_stored():
    out = IngestOutcome(status="stored", stored_chunks=2, result=make_result())
    report, code = format_ingest_report(U, out)
    assert report == f"url={U} is_valid=True stored_chunks=2 sections=2 confidence=0.9"
    assert code == 0


def test_format_ingest_report_blocked_preserves_none_reason():
    out = IngestOutcome(status="blocked", reason=None, fetch_outcome="blocked")
    report, code = format_ingest_report(U, out)
    assert report == f"fetch outcome=blocked reason=None -> no ingestion"
    assert code == 0


def test_format_ingest_report_fetch_failed():
    out = IngestOutcome(status="fetch_failed", reason="timeout", fetch_outcome="failed")
    report, code = format_ingest_report(U, out)
    assert report == "fetch outcome=failed reason=timeout -> no ingestion"
    assert code == 0


def test_format_ingest_report_no_content():
    out = IngestOutcome(status="no_content", reason="no usable content")
    report, code = format_ingest_report(U, out)
    assert report == "fetch succeeded but no content to ingest"
    assert code == 0


def test_format_ingest_report_flagged_validation():
    out = IngestOutcome(status="flagged", reason="validation failed", validation=make_validation(False))
    report, code = format_ingest_report(U, out)
    assert report.startswith("validation failed -> not ingested (flagged, see validation_flagged event)")
    assert '"is_valid": false' in report
    assert code == 0


def test_format_ingest_report_flagged_chunk_failure():
    out = IngestOutcome(status="flagged", reason="no sections to chunk", result=make_result())
    report, code = format_ingest_report(U, out)
    assert report == (
        f"url={U} is_valid=True stored_chunks=0 sections=2 confidence=0.9"
        "\nchunk_failed reason=no sections to chunk"
    )
    assert code == 1


def test_format_chunk_report():
    report, code = format_chunk_report(5, "corpus_fake_1024", None)
    assert report == "stored_chunks=5 collection=corpus_fake_1024"
    assert code == 0
    report, code = format_chunk_report(0, "corpus_fake_1024", "no sections to chunk")
    assert report == "stored_chunks=0 collection=corpus_fake_1024\nchunk_failed reason=no sections to chunk"
    assert code == 1


def test_ingest_lifecycle_events_logged(monkeypatch, tmp_path):
    stub_valid_pipeline(monkeypatch)
    with FetchLogger(tmp_path / "logs.db") as logger:
        out = run(logger=logger)
        assert out.status == "stored"
        rows = logger.rows_for_url(U)
    lifecycle = [r for r in rows if r["event_type"] == "ingest_lifecycle"]
    stages = [r["outcome"] for r in lifecycle]
    assert "store_start" in stages
    assert "completed" in stages
    import json
    completed = [r for r in lifecycle if r["outcome"] == "completed"][0]
    details = json.loads(completed["details_json"])
    assert "duration_seconds" in details


def test_ingest_lifecycle_for_blocked(monkeypatch, tmp_path):
    import json as _json
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.BLOCKED, reason="captcha gate detected"
    ))
    with FetchLogger(tmp_path / "logs.db") as logger:
        out = run(logger=logger)
        assert out.status == "blocked"
        rows = logger.rows_for_url(U)
    lifecycle = [r for r in rows if r["event_type"] == "ingest_lifecycle"]
    stages = [r["outcome"] for r in lifecycle]
    assert "fetch_done" in stages
    blocked_lc = [r for r in lifecycle if r["outcome"] == "fetch_done"][0]
    details = _json.loads(blocked_lc["details_json"])
    assert details["status"] == "blocked"


def test_ingest_lifecycle_for_fetch_failed(monkeypatch, tmp_path):
    import json as _json
    monkeypatch.setattr(ing, "fetch_page", lambda url, **kw: fetch_result(
        FetchOutcome.FAILED, reason="navigation error"
    ))
    with FetchLogger(tmp_path / "logs.db") as logger:
        out = run(logger=logger)
        assert out.status == "fetch_failed"
        rows = logger.rows_for_url(U)
    lifecycle = [r for r in rows if r["event_type"] == "ingest_lifecycle"]
    stages = [r["outcome"] for r in lifecycle]
    assert "fetch_done" in stages
    ff_lc = [r for r in lifecycle if r["outcome"] == "fetch_done"][0]
    details = _json.loads(ff_lc["details_json"])
    assert details["status"] == "fetch_failed"


def test_store_result_embed_failure_logs_chunk_failed(tmp_path):
    class BrokenEmbedder:
        model_name = "broken"
        dimension = 4
        def embed(self, texts):
            raise RuntimeError("CUDA out of memory")

    logger = FetchLogger(tmp_path / "events.db")
    stored, err = store_result(make_result(), CFG, BrokenEmbedder(), FakeStore(), logger)
    assert stored == 0
    assert "CUDA out of memory" in err
    events = [r for r in logger.recent_events() if r["event_type"] == "chunk_failed"]
    assert len(events) == 1
    assert "embedder failed" in events[0]["reason"]


def test_store_result_store_failure_logs_chunk_failed(tmp_path):
    class BrokenStore:
        def store_chunks(self, chunks, embeddings, *, collection_name, logger=None):
            raise RuntimeError("ChromaDB connection refused")

    logger = FetchLogger(tmp_path / "events.db")
    stored, err = store_result(make_result(), CFG, FakeEmbedder(), BrokenStore(), logger)
    assert stored == 0
    assert "ChromaDB connection refused" in err
    events = [r for r in logger.recent_events() if r["event_type"] == "chunk_failed"]
    assert len(events) == 1
    assert "store failed" in events[0]["reason"]
