"""Hermetic tests for Milestone 3 boilerplate stripping (pipeline/strip.py).

No network access. Trafilatura is fully offline; the HTML inputs below are
shaped like the realistic pages it reliably structures (verified against
trafilatura 2.2.0), and every input type it receives from the fetch layer
(HTML, plain text, malformed HTML, JS shells) is covered.
"""

from __future__ import annotations

from fetchers.logger import FetchLogger
from pipeline.strip import StripOutcome, strip_html

ARTICLE_HTML = """<html><head><title>Realistic Test Page</title></head><body>
<header><nav id='main-nav'><ul><li><a href='/'>Home</a></li><li><a href='/about'>About</a></li><li><a href='/products'>Products</a></li></ul></nav></header>
<main><article>
<h1>Understanding Web Scraping</h1>
<p>Web scraping is the process of automatically extracting data from websites. It is widely used for research, price comparison, and data journalism.</p>
<p>The legal landscape varies by jurisdiction, but public data is generally fair game under most interpretations.</p>
<h2>Technical Approaches</h2>
<p>There are two main approaches: static HTML parsing and headless browser automation.</p>
<h2>Ethical Considerations</h2>
<p>Respect rate limits and terms of service when scraping at scale.</p>
</article></main>
<footer>Copyright 2026 Example Corp. All rights reserved.</footer>
<script>var x = 1;</script>
</body></html>"""

TABLE_HTML = """<html><head><title>Stats Page</title></head><body><main><article>
<h1>Annual Report</h1>
<p>This report summarizes several years of data collected across multiple regions and teams. It is long enough that trafilatura treats this document as having real, substantial content rather than a sparse stub.</p>
<p>The numbers below represent yearly totals and are the authoritative figures referenced throughout the rest of this page and its appendices.</p>
<h2>Yearly Totals</h2>
<table><thead><tr><th>Year</th><th>Total</th><th>Note</th></tr></thead><tbody>
<tr><td>2021</td><td>100</td><td>value1</td></tr>
<tr><td>2022</td><td>200</td><td>value2</td></tr>
</tbody></table>
<h2>Methodology</h2>
<p>Figures were collected via automated pipelines and manually audited. Any discrepancy should be reported to the data team for correction before publication.</p>
</article></main></body></html>"""

MALFORMED_HTML = """<html><head><title>broken</title></head><body><main><article>
<h1>Broken Page
<p>This paragraph is never closed and <b>has stray < tags and an <img src=foo> inside</b>.
<h2>Still here</h2><p>More text with unclosed table <table><tr><td>cell1</td></body></html>"""


def test_strips_boilerplate():
    result = strip_html(ARTICLE_HTML, url="https://example.com/article")
    assert result.outcome == StripOutcome.STRIPPED
    assert "Understanding Web Scraping" in result.text
    assert "Web scraping is the process" in result.text
    assert "Products" not in result.text
    assert "Copyright 2026" not in result.text
    assert "var x" not in result.text
    assert result.chars_before > result.chars_after
    assert result.title == "Realistic Test Page"


def test_preserves_heading_structure():
    result = strip_html(ARTICLE_HTML, url="https://example.com/article")
    headings = {b.heading: b for b in result.blocks if b.heading}
    assert headings["Understanding Web Scraping"].level == 1
    assert "Web scraping is the process" in headings["Understanding Web Scraping"].text
    assert headings["Technical Approaches"].level == 2
    assert "static HTML parsing" in headings["Technical Approaches"].text
    assert headings["Ethical Considerations"].text.strip()


def test_table_rendered_and_counted():
    result = strip_html(TABLE_HTML, url="https://example.com/stats")
    assert result.num_tables >= 1
    assert "Year | Total | Note" in result.text
    assert "2021 | 100 | value1" in result.text


def test_malformed_html_does_not_crash():
    result = strip_html(MALFORMED_HTML, url="https://example.com/broken")
    assert result.outcome in (StripOutcome.STRIPPED, StripOutcome.FALLBACK, StripOutcome.EMPTY)


def test_empty_inputs():
    for blank in ("", "   ", "\n\t", None):
        result = strip_html(blank, url="https://example.com/blank")
        assert result.outcome == StripOutcome.EMPTY
        assert result.blocks == []
        assert result.text == ""


def test_plain_text_fallback():
    text = "This is a plain text document, not HTML at all."
    result = strip_html(text, url="https://example.com/plain")
    assert result.outcome == StripOutcome.FALLBACK
    assert len(result.blocks) == 1
    assert "plain text document" in result.text


def test_script_only_html_falls_back():
    html = "<html><head><title>x</title></head><body><script>var x=1;</script></body></html>"
    result = strip_html(html, url="https://example.com/script")
    assert result.outcome == StripOutcome.FALLBACK
    assert "var x" in result.text


def test_js_shell_html_never_stripped_as_rich_content():
    html = "<html><head><title>x</title></head><body><div id='root'></div><script src='app.js'></script></body></html>"
    result = strip_html(html, url="https://example.com/shell")
    assert result.outcome != StripOutcome.STRIPPED
    assert result.text.strip() == "x"


def test_logs_strip_attempt(tmp_path):
    db = tmp_path / "logs.db"
    with FetchLogger(db) as logger:
        result = strip_html(ARTICLE_HTML, url="https://example.com/article", logger=logger)
        assert result.outcome == StripOutcome.STRIPPED
        rows = logger.rows_for_url("https://example.com/article")
    strip_rows = [r for r in rows if r["event_type"] == "strip_attempt"]
    assert len(strip_rows) == 1
    assert strip_rows[0]["outcome"] == "stripped"
    assert "method" in strip_rows[0]["details_json"]
    assert strip_rows[0]["url"] == "https://example.com/article"


def test_strip_parse_warning_logged_on_trafilatura_failure(tmp_path, monkeypatch):
    import pipeline.strip as strip_mod

    def boom(*args, **kwargs):
        raise RuntimeError("trafilatura exploded")

    monkeypatch.setattr(strip_mod, "trafilatura", type("M", (), {"extract": staticmethod(boom)})())

    with FetchLogger(tmp_path / "logs.db") as logger:
        result = strip_html("<html><body><p>test content here</p></body></html>",
                            url="https://example.com/bad", logger=logger)
        rows = logger.rows_for_url("https://example.com/bad")

    warnings = [r for r in rows if r["event_type"] == "strip_parse_warning"]
    assert len(warnings) >= 1
    assert warnings[0]["outcome"] == "fallback"
    assert "trafilatura" in (warnings[0]["reason"] or "").lower()
    assert result.text  # content still returned via fallback


def test_strip_title_extraction_failure_logged(tmp_path):
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = strip_html("<html><body><p>no title here</p></body></html>",
                            url="https://example.com/notitle", logger=logger)
        rows = logger.rows_for_url("https://example.com/notitle")

    # No lxml failure on valid HTML, just no title found — no warning expected
    warnings = [r for r in rows if r["event_type"] == "strip_parse_warning"]
    title_warnings = [r for r in warnings if "title_extraction" in (r.get("details_json") or "")]
    # For valid HTML, title extraction simply returns None — no warning
    assert len(title_warnings) == 0
    assert result.title is None
