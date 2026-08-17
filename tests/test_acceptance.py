"""M10 acceptance-criterion tests (SPEC.md AC1-AC8), one criterion-shaped test each.

Hermetic where the criterion's meaning allows injection (AC2/4/5/6/7/8 use the
repo's established pattern: injected fetch/ingest, deterministic ScriptedClient
in place of the LLM, fake embedder, real VectorStore on scratch dirs). AC1 is
the durable E2E proof: it runs the REAL chain for three content types (real
local HTTP server on a dynamic port, real Playwright/Chromium for the SPA's
JS-rendering path, real pypdf route for the PDF, real chroma on a scratch dir),
skipping with a clear reason if Chromium is unavailable (SPEC AC1's real-browser
requirement must never crash or hang a browserless dev machine).

AC3 (chat answer carries a traceable source URL) is already covered exactly by
tests/test_answer.py (citation provenance incl. source_url + verbatim-quote
rule) and tests/test_app.py (answer rendering); the M10 live pass re-proves it
against the real corpus + real LLM, so no redundant test is added here.
"""

from __future__ import annotations

import functools
import http.server
import json
import threading
from datetime import datetime, timezone

import pytest

from fetchers.logger import FetchLogger
from fetchers.types import FetchOutcome, FetchResult
from orchestrator.live_query import live_query
from orchestrator.queue import UrlQueue
from orchestrator.run_batch import run_batch
from pipeline import ingest as ing
from pipeline.ingest import IngestOutcome
from pipeline.store import VectorStore
from schemas.extraction import ContentType, ExtractionResult, Section
from ui.db_view import EventLogView, QueueView

U = "https://ac.example.com/good"
TS = datetime(2026, 8, 14, 12, 0, 0, tzinfo=timezone.utc)
AGENT_CFG = {
    "llm": {
        "provider": "fake",
        "model": "scripted",
        "temperature": 0.1,
        "max_tokens": 1024,
        "parse_retry": 1,
    }
}
PIPELINE_CFG = {"chunk": {"max_chunk_chars": 2000}, "store": {"collection_prefix": "corpus"}}
RETRY_CFG = {"retry": {"max_attempts": 3, "backoff_seconds": 0.0}}
COLLECTION = "corpus_fake_model_4"  # embedder model_name=fake-model, dimension=4


# -- deterministic LLM / embedder stand-ins -----------------------------------

class ScriptedClient:
    """Deterministic complete_structured that routes on the page content.

    Returns schema-valid ExtractionResult dicts; near-empty content ("hello
    world") produces a low-confidence empty result so the validator's
    repair-then-flag path is exercised deterministically. Call counts per URL
    let AC8 assert "exactly one repair attempt".
    """

    def __init__(self):
        self.calls: dict[str, int] = {}

    def complete_structured(self, *, messages, tool_name, tool_schema, temperature, max_tokens):
        full_text = "\n".join(m["content"] for m in messages)
        url = ""
        for line in full_text.splitlines():
            if line.strip().startswith("source_url:"):
                url = line.split(":", 1)[1].strip()
        self.calls[url] = self.calls.get(url, 0) + 1
        if "hello world" in full_text.lower():
            return {
                "source_url": url,
                "scrape_timestamp": TS.isoformat(),
                "page_title": "Fake Page Title",
                "content_type": "html",
                "sections": [],
                "confidence": 0.3,
                "truncated": False,
                "extraction_notes": "no meaningful content",
            }
        return {
            "source_url": url,
            "scrape_timestamp": TS.isoformat(),
            "page_title": "Fake Page Title",
            "content_type": "html",
            "sections": [
                Section(heading="Section A", content="Meaningful extracted content for the acceptance suite. " * 4).model_dump(),
                Section(heading="Section B", content="More meaningful extracted content for the acceptance suite. " * 4).model_dump(),
            ],
            "confidence": 0.9,
            "truncated": False,
            "extraction_notes": None,
        }


class FakeEmbedder:
    model_name = "fake-model"
    dimension = 4

    def embed(self, texts):
        return [[0.25] * 4 for _ in texts]


# -- local HTTP server on a dynamic port (never a fixed one) ------------------

_FIXTURES = {
    "static.html": (
        "<!doctype html><html><head><title>Static Test Page</title></head><body>"
        "<h1>Static Test Page</h1>"
        "<p>The first paragraph of a static HTML acceptance fixture. Web scraping "
        "pipelines fetch, strip boilerplate, and extract structured content from "
        "arbitrary pages without hardcoded selectors.</p>"
        "<h2>Section One</h2>"
        "<p>Multi-agent extraction relies on language models constrained by explicit "
        "schemas so the same logic generalizes across unknown layouts. Provenance "
        "metadata must accompany every extracted record to the vector store.</p>"
        "<h2>Section Two</h2>"
        "<p>Source URLs, scrape timestamps, and page titles are mandatory provenance "
        "fields on every RAG chunk, making answers verifiable end to end.</p>"
        "</body></html>"
    ),
    "spa.html": (
        "<!doctype html><html><head><title>SPA Test Page</title></head><body>"
        "<div id=\"root\"></div>"
        "<script>document.getElementById('root').innerHTML = "
        "'<h1>SPA Test Page</h1>' + "
        "'<p>This content is rendered entirely by JavaScript, so a static fetch sees "
        "an empty shell and the pipeline must fall back to a headless browser. The "
        "browser executes the script and renders the headings and paragraphs that "
        "follow.</p>' + "
        "'<h2>Rendered Section</h2>' + "
        "'<p>Headless-browser rendering is the baseline requirement for JS-rendered "
        "sites, not an edge-case fallback, and the final DOM feeds the stripping and "
        "extraction stages.</p>';</script>"
        "</body></html>"
    ),
    "captcha.html": (
        "<!doctype html><html><head><title>Challenge</title></head><body>"
        "<div id=\"root\"></div>"
        "<script>document.getElementById('root').innerText = "
        "'Please verify you are human to continue accessing this page';</script>"
        "</body></html>"
    ),
    "empty.html": (
        "<!doctype html><html><head><title>Nearly Empty Page</title></head><body>"
        "<p>Hello world</p></body></html>"
    ),
    "opaque.bin": b"not html, not pdf, not text - an unrouteable content type\n",
}


def _make_pdf() -> bytes:
    lines = [
        "PDF Acceptance Test Document",
        "This is a test PDF generated for the acceptance suite.",
        "It contains several sentences of meaningful text about data pipelines.",
        "Extraction agents produce structured content from PDF documents.",
        "Source URLs and scrape timestamps must accompany every stored chunk.",
    ]
    content = b"\n".join(b"BT /F1 14 Tf 72 %d Td (%s) Tj ET" % (700 - i * 24, ln.encode()) for i, ln in enumerate(lines))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i
        out += obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    out += b"startxref\n%d\n%%%%EOF\n" % xref_pos
    return bytes(out)


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("acceptance_fixtures")
    for name, body in _FIXTURES.items():
        if isinstance(body, bytes):
            (root / name).write_bytes(body)
        else:
            (root / name).write_text(body, encoding="utf-8")
    (root / "sample.pdf").write_bytes(_make_pdf())
    return root


@pytest.fixture(scope="module")
def server(fixture_root):
    """Local static file server on 127.0.0.1:<dynamic port> (resolution 3)."""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(fixture_root)
    )
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _chromium_launchable() -> bool:
    """Probe Chromium before AC1: missing browser => skip, never crash/hang."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
    except Exception:
        return False
    return True


# -- shared batch wiring ------------------------------------------------------

def _run_batch(scratch, urls, *, fetch_route=None, monkeypatch=None, client=None, store=None, queue_db=None):
    if fetch_route is not None:
        monkeypatch.setattr(ing, "fetch_page", fetch_route)
    logger = FetchLogger(scratch / "events.db")
    queue = UrlQueue(queue_db or scratch / "queue.db")
    store = store or VectorStore(scratch / "chroma")
    client = client or ScriptedClient()
    summary = run_batch(
        urls,
        queue=queue,
        logger=logger,
        agent_cfg=AGENT_CFG,
        client=client,
        embedder=FakeEmbedder(),
        store=store,
        pipeline_cfg=PIPELINE_CFG,
        retry_cfg=RETRY_CFG,
    )
    return summary, queue, logger, store, client


def _mixed_run(tmp_path, monkeypatch):
    captcha = "https://ac.example.com/captcha"
    good = "https://ac.example.com/good"
    empty = "https://ac.example.com/empty"

    def route(url, **kw):
        if url == captcha:
            return FetchResult(url=url, outcome=FetchOutcome.BLOCKED, reason="captcha gate detected")
        if url == empty:
            return FetchResult(
                url=url,
                outcome=FetchOutcome.SUCCESS,
                html="<html><body><p>Hello world</p></body></html>",
                content_type="text/html",
            )
        return FetchResult(
            url=url,
            outcome=FetchOutcome.SUCCESS,
            html=(
                "<html><body><h1>Good Page</h1><p>"
                + "meaningful content for the acceptance good page. " * 12
                + "</p></body></html>"
            ),
            content_type="text/html",
        )

    summary, queue, logger, store, client = _run_batch(
        tmp_path, [good, captcha, empty], fetch_route=route, monkeypatch=monkeypatch
    )
    return captcha, good, empty, summary, queue, logger, store, client


def _make_result(sections=2):
    return ExtractionResult(
        source_url=U,
        scrape_timestamp=TS,
        page_title="Page",
        content_type=ContentType.HTML,
        sections=[Section(heading=f"H{i}", content=f"section {i} content") for i in range(sections)],
        confidence=0.9,
    )


# -- AC1: three content types ingested with full provenance -------------------

def test_acceptance_criterion1_three_content_types_with_provenance(tmp_path, server):
    if not _chromium_launchable():
        pytest.skip(
            "playwright chromium not installed/launchable: AC1's SPA proof needs a "
            "real headless browser (SPEC AC1's JS-rendering baseline); skipping "
            "instead of crashing or hanging the suite"
        )
    urls = [f"{server}/static.html", f"{server}/spa.html", f"{server}/sample.pdf"]
    summary, queue, logger, store, _ = _run_batch(tmp_path, urls)

    assert summary["by_state"].get("done", 0) == 3, summary["by_state"]
    for url in urls:
        assert store.count(collection_name=COLLECTION, where={"source_url": url}) > 0

    rows = store.get(collection_name=COLLECTION, where={"source_url": urls[0]})
    assert rows
    for row in rows:
        meta = row["metadata"]
        assert meta["source_url"] == urls[0]
        assert meta["scrape_timestamp"]
        assert meta["page_title"]
        assert meta["section_heading"] is not None

    spa_events = [
        e
        for e in logger.recent_events()
        if e["event_type"] == "fetch_attempt" and e["url"] == urls[1]
    ]
    assert any(json.loads(e["details_json"]).get("fetcher") == "browser" for e in spa_events), (
        "SPA must be fetched by the real headless browser, not the static fetcher"
    )


# -- AC2: CAPTCHA terminal-blocked while the batch continues ------------------

def test_acceptance_criterion2_captcha_blocked_batch_continues(tmp_path, monkeypatch):
    captcha, good, _, summary, queue, logger, _, _ = _mixed_run(tmp_path, monkeypatch)

    assert summary["by_state"] == {"done": 1, "blocked": 1, "flagged": 1}, summary["by_state"]
    captcha_row = queue.get(captcha)
    assert captcha_row["state"] == "blocked"
    assert captcha_row["reason"] == "captcha gate detected"
    assert queue.get(good)["state"] == "done"
    runs = [e for e in logger.recent_events() if e["event_type"] == "batch_run"]
    assert runs[0]["outcome"] == "finished"


# -- AC4: ingestion view shows the per-URL status ------------------------------

def test_acceptance_criterion4_ingestion_statuses(tmp_path, monkeypatch):
    captcha, good, empty, _, queue, _, _, _ = _mixed_run(tmp_path, monkeypatch)
    view = QueueView(queue.db_path)

    assert view.status_for(good)["state"] == "done"
    assert view.status_for(captcha)["state"] == "blocked"
    assert view.status_for(empty)["state"] == "flagged"
    assert view.counts()["done"] == 1
    assert view.counts()["blocked"] == 1
    assert view.counts()["flagged"] == 1


# -- AC5: logs view shows blocked + flagged with timestamp and reason ---------

def test_acceptance_criterion5_logs_show_blocked_and_flagged(tmp_path, monkeypatch):
    captcha, _, empty, _, _, logger, _, _ = _mixed_run(tmp_path, monkeypatch)
    view = EventLogView(logger.db_path)

    blocked = [e for e in view.recent(event_types=["batch_url_state"]) if e["outcome"] == "blocked"]
    assert any(e["url"] == captcha and e["reason"] for e in blocked)

    flagged = [e for e in view.recent(event_types=["validation_flagged"]) if e["url"] == empty]
    assert flagged and all(e["reason"] for e in flagged)

    assert all(e["ts"] for e in view.recent())


# -- AC6: live-query falls back to a single-shot scrape, no batch -------------

def test_acceptance_criterion6_live_query_single_shot(tmp_path):
    calls = {"n": 0, "write_to_corpus": None}

    def ing(url, **kw):
        calls["n"] += 1
        calls["write_to_corpus"] = kw.get("write_to_corpus")
        return IngestOutcome(status="stored", result=_make_result(), written=False)

    class EmptyStore:
        def count(self, *, collection_name, where=None):
            return 0

    logger = FetchLogger(tmp_path / "events.db")
    result = live_query(
        U,
        query=None,
        logger=logger,
        agent_cfg={},
        client=None,
        embedder=FakeEmbedder(),
        store=EmptyStore(),
        pipeline_cfg=PIPELINE_CFG,
        ingest=ing,
    )
    assert result.found_in_corpus is False
    assert result.source_used == "live_scrape"
    assert result.status == "single_shot_ok"
    assert calls["n"] == 1
    assert calls["write_to_corpus"] is False
    assert result.evidence and result.evidence[0].provenance["source_url"] == U


# -- AC7: same URL ingested twice produces one set of chunks ------------------

def test_acceptance_criterion7_reingest_replaces_no_duplicates(tmp_path, monkeypatch):
    url = "https://ac.example.com/dupe"

    def route(u, **kw):
        return FetchResult(
            url=u,
            outcome=FetchOutcome.SUCCESS,
            html=(
                "<html><body><h1>Dupe Page</h1><p>"
                + "content to ingest twice, identically. " * 8
                + "</p></body></html>"
            ),
            content_type="text/html",
        )

    store = VectorStore(tmp_path / "chroma")
    summary1, _, _, _, client = _run_batch(
        tmp_path, [url], fetch_route=route, monkeypatch=monkeypatch, store=store
    )
    n1 = store.count(collection_name=COLLECTION, where={"source_url": url})
    assert summary1["by_state"].get("done", 0) == 1
    assert n1 > 0

    # A FRESH queue against the SAME store: re-ingesting the same URL through
    # the batch path must replace the prior set, not duplicate it.
    summary2, _, _, _, _ = _run_batch(
        tmp_path,
        [url],
        fetch_route=route,
        monkeypatch=monkeypatch,
        store=store,
        queue_db=tmp_path / "queue2.db",
    )
    n2 = store.count(collection_name=COLLECTION, where={"source_url": url})
    assert summary2["by_state"].get("done", 0) == 1
    assert n1 == n2  # delete-then-insert: same set, no duplicates


# -- AC8: validation failure retried exactly once, then flagged ----------------

def test_acceptance_criterion8_retry_once_then_flag(tmp_path, monkeypatch):
    empty = "https://ac.example.com/empty"

    def route(u, **kw):
        return FetchResult(
            url=u,
            outcome=FetchOutcome.SUCCESS,
            html="<html><body><p>Hello world</p></body></html>",
            content_type="text/html",
        )

    store = VectorStore(tmp_path / "chroma")
    summary, queue, logger, store, client = _run_batch(
        tmp_path, [empty], fetch_route=route, monkeypatch=monkeypatch, store=store
    )

    assert client.calls[empty] == 2, "initial extraction + exactly one repair attempt"
    assert queue.get(empty)["state"] == "flagged"
    assert store.count(collection_name=COLLECTION, where={"source_url": empty}) == 0

    attempts = [
        e
        for e in logger.recent_events()
        if e["event_type"] == "validation_attempt" and e["url"] == empty
    ]
    # recent_events() is newest-first; assert the chronological order.
    assert [e["outcome"] for e in reversed(attempts)] == ["repairing", "failed"]

    flagged = [
        e
        for e in logger.recent_events()
        if e["event_type"] == "validation_flagged" and e["url"] == empty
    ]
    assert flagged
    assert json.loads(flagged[0]["details_json"])["flag_reason"] == "repair_budget_exhausted"
    assert flagged[0]["reason"]
