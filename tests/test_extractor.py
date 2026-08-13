"""Hermetic tests for Milestone 4 extractor (agents/extractor.py).

No LLM calls and no network: a stub LLM client is injected. Covers routing,
parse-retry (bounded), schema validation, flag-on-failure (never silent
drop), provenance preservation, and the extraction_attempt / extraction_flagged
log events.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os

import pytest

from agents.extractor import extract_content, mime_to_content_type
from fetchers.logger import FetchLogger
from schemas.extraction import ContentType, ExtractionResult

ARTICLE_CONTENT = """Understanding Web Scraping

Web scraping is the process of automatically extracting data from websites. It is widely used for research and data journalism.

Technical Approaches

There are two main approaches: static HTML parsing and headless browser automation.

Ethical Considerations

Respect rate limits and terms of service when scraping at scale.
"""


class StubClient:
    """Stub LLM client: returns a canned dict, None, raises, or counts calls."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_structured(self, *, messages, tool_name, tool_schema, temperature, max_tokens):
        self.calls.append({"messages": messages, "tool_name": tool_name})
        if not self.responses:
            return None
        return self.responses.pop(0)


VALID_RESPONSE = {
    "source_url": "https://example.com/article",
    "scrape_timestamp": "2026-08-13T12:00:00+00:00",
    "page_title": "Understanding Web Scraping",
    "content_type": "html",
    "sections": [
        {"heading": "Technical Approaches", "level": 2, "content": "Static HTML parsing and headless browser automation."}
    ],
    "confidence": 0.95,
    "truncated": False,
}


def _cfg():
    return {"llm": {"provider": "stub", "model": "stub", "api_key_env": "", "temperature": 0.1, "max_tokens": 1000, "tool_choice": "required", "parse_retry": 1}}


def test_returns_valid_extraction_result():
    client = StubClient([VALID_RESPONSE])
    result = extract_content(
        ARTICLE_CONTENT,
        content_type=ContentType.HTML,
        source_url="https://example.com/article",
        page_title="Understanding Web Scraping",
        client=client,
        agent_cfg=_cfg(),
    )
    assert isinstance(result, ExtractionResult)
    assert result.confidence == 0.95
    assert result.sections[0].heading == "Technical Approaches"
    assert result.source_url == "https://example.com/article"
    assert result.content_type == ContentType.HTML


def test_provenance_defaults_to_now_utc():
    client = StubClient([VALID_RESPONSE])
    result = extract_content(
        ARTICLE_CONTENT,
        content_type=ContentType.HTML,
        source_url="https://example.com/article",
        client=client,
        agent_cfg=_cfg(),
    )
    assert result.scrape_timestamp.tzinfo is not None
    assert result.scrape_timestamp.astimezone(timezone.utc).year == datetime.now(timezone.utc).year


def test_batch_and_single_mode_use_same_schema_and_call():
    for mode in ("batch", "single"):
        client = StubClient([VALID_RESPONSE])
        extract_content(
            ARTICLE_CONTENT,
            content_type=ContentType.HTML,
            source_url="https://example.com/article",
            mode=mode,
            client=client,
            agent_cfg=_cfg(),
        )
        msg = client.calls[0]["messages"][0]["content"]
        assert f"Extraction mode: {mode}" in msg
        assert client.calls[0]["tool_name"] == "extract_meaningful_content"


def test_parse_retry_recovers_then_succeeds(tmp_path):
    client = StubClient([None, VALID_RESPONSE])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            ARTICLE_CONTENT,
            content_type=ContentType.HTML,
            source_url="https://example.com/article",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.confidence == 0.95
        assert len(client.calls) == 2
        assert "could not be used" in client.calls[1]["messages"][-1]["content"]
        rows = logger.rows_for_url("https://example.com/article")
        attempts = [r for r in rows if r["event_type"] == "extraction_attempt"]
        assert len(attempts) == 1 and attempts[0]["outcome"] == "extracted"


def test_parse_retry_exhausted_flags_record(tmp_path):
    client = StubClient([None, None])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            ARTICLE_CONTENT,
            content_type=ContentType.HTML,
            source_url="https://example.com/article",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.confidence == 0.0
        assert result.sections == []
        assert "no parseable structured output" in result.extraction_notes
        assert result.source_url == "https://example.com/article"
        rows = logger.rows_for_url("https://example.com/article")
        flagged = [r for r in rows if r["event_type"] == "extraction_flagged"]
        attempts = [r for r in rows if r["event_type"] == "extraction_attempt"]
        assert len(flagged) == 1 and flagged[0]["outcome"] == "flagged"
        assert len(attempts) == 1 and attempts[0]["outcome"] == "flagged"


def test_schema_violation_triggers_retry_then_flags(tmp_path):
    bad = dict(VALID_RESPONSE)
    bad["confidence"] = 2.5  # out of 0..1 range
    client = StubClient([bad, VALID_RESPONSE])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            ARTICLE_CONTENT,
            content_type=ContentType.HTML,
            source_url="https://example.com/article",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.confidence == 0.95
        assert len(client.calls) == 2
        assert "schema validation failed" in client.calls[1]["messages"][-1]["content"]


def test_unknown_content_type_flagged_without_llm_call(tmp_path):
    client = StubClient([])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            "some unknown binary-ish content",
            content_type=ContentType.UNKNOWN,
            source_url="https://example.com/unknown",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.confidence == 0.0
        assert result.sections == []
        assert "unsupported content type" in result.extraction_notes
        assert client.calls == []
        rows = logger.rows_for_url("https://example.com/unknown")
        assert any(r["event_type"] == "extraction_flagged" for r in rows)


def test_empty_content_flagged(tmp_path):
    client = StubClient([])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            "   ",
            content_type=ContentType.HTML,
            source_url="https://example.com/empty",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.confidence == 0.0
        assert client.calls == []
        assert "empty content" in result.extraction_notes


def test_pdf_routes_through_text_extraction(tmp_path, monkeypatch):
    from agents import extractor

    extracted_text = "PDF page one text."
    monkeypatch.setattr(extractor, "_pdf_to_text", lambda content: extracted_text)

    client = StubClient([VALID_RESPONSE])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            b"%PDF-1.4 fake pdf bytes",
            content_type=ContentType.PDF,
            source_url="https://example.com/file.pdf",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.content_type == ContentType.PDF
        assert client.calls[0]["messages"][0]["content"].count(extracted_text) >= 1
        assert result.confidence == 0.95


def test_corrupt_pdf_flagged(tmp_path):
    client = StubClient([])
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = extract_content(
            b"definitely not a pdf",
            content_type=ContentType.PDF,
            source_url="https://example.com/broken.pdf",
            client=client,
            agent_cfg=_cfg(),
            logger=logger,
        )
        assert result.confidence == 0.0
        assert "pdf text extraction failed" in result.extraction_notes
        assert client.calls == []


def test_text_content_type_extracts_directly():
    client = StubClient([VALID_RESPONSE])
    result = extract_content(
        "Just some plain text content.",
        content_type=ContentType.TEXT,
        source_url="https://example.com/plain.txt",
        client=client,
        agent_cfg=_cfg(),
    )
    assert result.content_type == ContentType.TEXT


@pytest.mark.parametrize(
    "mime,expected",
    [
        ("text/html; charset=utf-8", ContentType.HTML),
        ("application/xhtml+xml", ContentType.HTML),
        ("application/pdf", ContentType.PDF),
        ("text/plain; charset=us-ascii", ContentType.TEXT),
        ("application/json", ContentType.TEXT),
        ("application/octet-stream", ContentType.UNKNOWN),
        (None, ContentType.UNKNOWN),
        ("", ContentType.UNKNOWN),
    ],
)
def test_mime_to_content_type(mime, expected):
    assert mime_to_content_type(mime) == expected


def test_dotenv_loader_parses_and_ignores_existing(tmp_path, monkeypatch):
    from agents.llm.env import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\nMY_TEST_KEY=abc123\nQUOTED=\"value with spaces\"\n\n"
        "=no_key_line\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_TEST_KEY", "already-set")
    monkeypatch.delenv("QUOTED", raising=False)
    load_dotenv(env_file)
    assert os.environ["MY_TEST_KEY"] == "already-set"
    assert os.environ["QUOTED"] == "value with spaces"


def test_api_key_reads_from_dotenv(tmp_path, monkeypatch):
    from agents.llm import env as envmod
    from agents.llm.client import LiteLLMClient

    env_file = tmp_path / ".env"
    env_file.write_text("MY_PROVIDER_KEY=sekrit-value\n", encoding="utf-8")
    monkeypatch.delenv("MY_PROVIDER_KEY", raising=False)
    monkeypatch.setattr(envmod, "DEFAULT_ENV_PATH", env_file)

    cfg = {"llm": {"provider": "x", "model": "y", "api_key_env": "MY_PROVIDER_KEY",
                   "temperature": 0.1, "max_tokens": 10, "tool_choice": "required",
                   "parse_retry": 1}}
    client = LiteLLMClient(cfg)
    assert client._api_key() == "sekrit-value"
