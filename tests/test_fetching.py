"""Hermetic tests for the Milestone 2 fetcher layer.

No real network: an in-thread http.server serves fixture pages. Playwright
browser tests run against the same local server (real chromium, no internet).
"""

from __future__ import annotations

import http.server
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import pytest

import fetchers.fetch as fetch_mod
from fetchers.config_loader import load_fetch_config, load_obstacle_config
from fetchers.content_heuristics import is_html_like, looks_empty, visible_text_length
from fetchers.logger import FetchLogger
from fetchers.rate_limit import RateLimiter
from fetchers.types import FetchOutcome, FetchResult


class _Handler(http.server.BaseHTTPRequestHandler):
    routes: dict = {}

    def do_GET(self):
        route = _Handler.routes.get(self.path)
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        if callable(route):
            route = route(self)
        self.send_response(route.get("status", 200))
        self.send_header("Content-Type", route.get("content_type", "text/html; charset=utf-8"))
        for key, value in (route.get("headers") or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(route.get("body", b""))

    def log_message(self, *args):  # silence request logging
        pass


@pytest.fixture
def server():
    _Handler.routes = {}
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


def url_for(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


def make_logger(tmp_path: Path) -> FetchLogger:
    return FetchLogger(tmp_path / "logs.db")


@pytest.fixture
def pipeline(tmp_path):
    """Shared fetcher deps: no rate limiting, instant backoff, temp log DB."""

    def _make():
        fetch_cfg = load_fetch_config()
        fetch_cfg["backoff_base_seconds"] = 0.0
        obstacle_cfg = load_obstacle_config()
        return {
            "obstacle_cfg": obstacle_cfg,
            "fetch_cfg": fetch_cfg,
            "logger": make_logger(tmp_path),
            "rate_limiter": RateLimiter(0),
        }

    yield _make


STATIC_HTML = (
    "<html><head><title>T</title></head><body>"
    "<h1>Welcome</h1>"
    "<p>This is a statically served page with a good amount of visible text content "
    "to exceed the emptiness threshold and be considered meaningful.</p>"
    "<p>Second paragraph adding even more words so the visible text length clearly "
    "passes the threshold of two hundred characters without any doubt.</p>"
    "</body></html>"
)

JS_SHELL_HTML = (
    "<html><head><script src='/app.js'></script></head>"
    "<body><div id='root'></div></body></html>"
)

RENDERED_HTML = (
    "<html><body><h1>Rendered App</h1>"
    "<p>Client-side rendered content produced by the virtual DOM after hydration. "
    "This text only exists once the bundle executes, with plenty of words here so "
    "that the visible character count climbs comfortably above the threshold of "
    "two hundred characters that marks a page as meaningful rather than a shell.</p>"
    "</body></html>"
)


def test_static_success(server, pipeline):
    port = server
    _Handler.routes["/"] = {"body": STATIC_HTML.encode()}
    deps = pipeline()
    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.SUCCESS
    assert res.fetcher == "static"
    assert res.html and "Welcome" in res.html

    rows = FetchLogger(deps["logger"].db_path).rows_for_url(url_for(port, "/"))
    attempts = [r for r in rows if r["event_type"] == "fetch_attempt"]
    assert len(attempts) == 1
    assert attempts[0]["outcome"] == "success"
    assert attempts[0]["url"] == url_for(port, "/")
    assert not [r for r in rows if r["event_type"] == "fetch_decision"]  # no fallback


def test_static_403_blocked_no_browser(server, pipeline, monkeypatch):
    port = server
    _Handler.routes["/"] = {"status": 403, "body": b"forbidden"}
    deps = pipeline()

    called = []
    monkeypatch.setattr(fetch_mod, "fetch_browser", lambda *a, **kw: called.append(True))

    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.BLOCKED
    assert res.fetcher == "static"
    assert "403" in (res.reason or "")
    assert called == [], "browser must not be invoked for a hard 403 (Rule 7)"


def test_js_shell_falls_back_to_browser(server, pipeline, monkeypatch):
    port = server
    _Handler.routes["/"] = {"body": JS_SHELL_HTML.encode()}
    deps = pipeline()

    calls = []
    fake_browser_result = FetchResult(
        url=url_for(port, "/"), outcome=FetchOutcome.SUCCESS, html=RENDERED_HTML, fetcher="browser"
    )

    def fake_browser(url, **kw):
        calls.append(url)
        return fake_browser_result

    monkeypatch.setattr(fetch_mod, "fetch_browser", fake_browser)

    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.SUCCESS
    assert res.fetcher == "browser"
    assert "Rendered App" in (res.html or "")
    assert calls == [url_for(port, "/")]

    rows = FetchLogger(deps["logger"].db_path).rows_for_url(url_for(port, "/"))
    decisions = [r for r in rows if r["event_type"] == "fetch_decision"]
    assert len(decisions) == 1
    assert decisions[0]["outcome"] == "fallback"
    attempts = [r for r in rows if r["event_type"] == "fetch_attempt"]
    assert len(attempts) == 1  # the mocked browser path logs its own attempt in the real fetcher


def test_non_html_no_fallback(server, pipeline, monkeypatch):
    port = server
    _Handler.routes["/"] = {"content_type": "application/pdf", "body": b"%PDF-1.4 fake"}
    deps = pipeline()

    called = []
    monkeypatch.setattr(fetch_mod, "fetch_browser", lambda *a, **kw: called.append(True))

    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.SUCCESS
    assert res.fetcher == "static"
    assert res.content_type == "application/pdf"
    assert res.html is None
    assert called == []


def test_static_429_retry_then_success(server, pipeline):
    port = server
    hits = {"n": 0}

    def route(h):
        hits["n"] += 1
        if hits["n"] == 1:
            return {"status": 429, "headers": {"Retry-After": "0"}, "body": b"slow down"}
        return {"body": STATIC_HTML.encode()}

    _Handler.routes["/"] = route
    deps = pipeline()
    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.SUCCESS
    assert hits["n"] == 2

    rows = FetchLogger(deps["logger"].db_path).rows_for_url(url_for(port, "/"))
    obstacles = [r for r in rows if r["event_type"] == "obstacle_detected"]
    assert any("rate_limit" in (r.get("details_json") or "") for r in obstacles)


def test_static_5xx_exhausted_then_browser(server, pipeline, monkeypatch):
    port = server
    _Handler.routes["/"] = {"status": 503, "body": b"unavailable"}
    deps = pipeline()

    calls = []
    monkeypatch.setattr(
        fetch_mod,
        "fetch_browser",
        lambda url, **kw: calls.append(url) or FetchResult(url=url, outcome=FetchOutcome.FAILED, reason="browser also failed", fetcher="browser"),
    )

    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.FAILED
    assert res.fetcher == "browser"
    assert len(calls) == 1  # browser fallback exactly once


def test_heuristics():
    assert looks_empty("", 200)
    assert looks_empty("<html><body><div id='root'></div></body></html>", 200)
    assert not looks_empty(STATIC_HTML, 200)
    assert visible_text_length("<p>hi</p>") == 2
    assert is_html_like("text/html; charset=utf-8")
    assert is_html_like("application/xhtml+xml")
    assert not is_html_like("application/pdf")
    assert not is_html_like("image/png")


def test_obstacle_config_validates():
    cfg = load_obstacle_config()
    assert cfg["captcha_gate"]["policy"] == "terminal_blocked"
    assert cfg["cookie_banner"]["policy"] == "dismiss"


def test_obstacle_config_rejects_bad_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"captcha_gate": {"policy": "solve"}}))
    with pytest.raises(ValueError):
        load_obstacle_config(bad)


def test_fetch_config_validates():
    cfg = load_fetch_config()
    assert cfg["empty_content_threshold_chars"] >= 0


def test_logging_shape_and_schema(tmp_path):
    logger = make_logger(tmp_path)
    try:
        logger.log_event(
            event_type="fetch_attempt",
            url="https://example.com/x",
            outcome="blocked",
            reason="captcha",
            details={"status_code": 200},
        )
    finally:
        logger.close()

    conn = sqlite3.connect(tmp_path / "logs.db")
    conn.row_factory = sqlite3.Row
    cols = [r[1] for r in conn.execute("PRAGMA table_info(events)")]
    assert cols == ["id", "ts", "event_type", "url", "outcome", "reason", "details_json"]
    row = conn.execute("SELECT * FROM events WHERE url=?", ("https://example.com/x",)).fetchone()
    row = {k: row[k] for k in row.keys()}
    datetime.fromisoformat(row["ts"])  # must be parseable ISO
    assert row["outcome"] == "blocked"
    assert json.loads(row["details_json"])["status_code"] == 200
    conn.close()


try:
    _HAS_PLAYWRIGHT = bool(__import__("playwright"))
except Exception:
    _HAS_PLAYWRIGHT = False


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")
def test_browser_dismiss_cookie_banner(server, pipeline):
    port = server
    # JS-shell served statically (so the decision logic falls back to browser);
    # the app then renders content and a dismissable cookie banner client-side.
    body = (
        "<html><head><script>"
        "window.addEventListener('DOMContentLoaded', function(){"
        "  document.getElementById('root').innerHTML = "
        "    '<h1>Page Content</h1><p>Substantial real content lives here, well over the "
        "two hundred character threshold so the page is treated as meaningful after the "
        "banner is dismissed. This paragraph is intentionally long enough to make the "
        "emptiness check pass without any ambiguity.</p>';"
        "  var b = document.createElement('div');"
        "  b.id = 'banner'; b.setAttribute('role', 'dialog');"
        "  b.innerHTML = 'We use cookies to improve your experience. "
        "<button id=\"accept\">Accept all</button>';"
        "  document.body.appendChild(b);"
        "  document.getElementById('accept').addEventListener('click', function(){ b.remove(); });"
        "});"
        "</script></head><body><div id='root'></div></body></html>"
    )
    _Handler.routes["/"] = {"body": body.encode()}
    deps = pipeline()
    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.SUCCESS
    assert res.fetcher == "browser"
    assert "Page Content" in (res.html or "")

    rows = FetchLogger(deps["logger"].db_path).rows_for_url(url_for(port, "/"))
    obstacles = [r for r in rows if r["event_type"] == "obstacle_detected"]
    assert any("cookie_banner" in (r.get("details_json") or "") for r in obstacles)


@pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")
def test_browser_blocked_clicks_noop_exhausted_fails(server, pipeline):
    port = server
    # JS-shell served statically; the app renders content plus a dead "Load More"
    # button with no handler, so the click is a no-op and must resolve to FAILED.
    body = (
        "<html><head><script>"
        "window.addEventListener('DOMContentLoaded', function(){"
        "  document.getElementById('root').innerHTML = "
        "    '<h1>Static Page</h1><p>This load-more button is dead: it has no handler and "
        "produces no change. After the configured retry budget the fetch must resolve to "
        "FAILED, never an infinite loop or silent success. Plenty of words here to clear "
        "the visible text threshold so the result is decided by the no-op logic.</p>"
        "<button id=\"more\">Load More</button>';"
        "});"
        "</script></head><body><div id='root'></div></body></html>"
    )
    _Handler.routes["/"] = {"body": body.encode()}
    deps = pipeline()
    try:
        res = fetch_mod.fetch_page(url_for(port, "/"), **deps)
    finally:
        deps["logger"].close()

    assert res.outcome == FetchOutcome.FAILED
    assert res.fetcher == "browser"
    assert "no-op" in (res.reason or "").lower()

    rows = FetchLogger(deps["logger"].db_path).rows_for_url(url_for(port, "/"))
    obstacles = [r for r in rows if r["event_type"] == "obstacle_detected"]
    assert any("blocked_clicks" in (r.get("details_json") or "") for r in obstacles)
