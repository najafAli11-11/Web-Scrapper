"""Hermetic tests for schema-constrained chat answer generation (M9)."""

import json

from agents.answer import TOOL_NAME, generate_answer
from fetchers.logger import FetchLogger

CFG = {"llm": {"temperature": 0.1, "max_tokens": 1024}}

TS = "2026-08-14T12:00:00+00:00"


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_structured(self, *, messages, tool_name, tool_schema, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "tool_name": tool_name,
                "tool_schema": tool_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return self.response(self.calls[-1]) if callable(self.response) else self.response


def evidence(text, *, url="https://example.com/a", heading="How it works", title="Example page"):
    return {
        "text": text,
        "provenance": {
            "source_url": url,
            "scrape_timestamp": TS,
            "page_title": title,
            "section_heading": heading,
        },
    }


def failure_events(logger):
    return [e for e in logger.recent_events() if e["event_type"] == "answer_generation_failed"]


def test_generate_answer_returns_answer_with_verbatim_quote(tmp_path):
    src = (
        "The extraction agent runs after boilerplate stripping. "
        "Stripping removes nav and ads before any LLM call."
    )
    client = FakeClient(
        {
            "answer": "The pipeline strips boilerplate before extraction.",
            "citations": [
                {
                    "source_url": "https://example.com/a",
                    "scrape_timestamp": TS,
                    "page_title": "Example page",
                    "section_heading": "How it works",
                    "quote": "Stripping removes nav and ads before any LLM call.",
                }
            ],
        }
    )
    logger = FetchLogger(tmp_path / "events.db")
    try:
        answer = generate_answer(
            "What runs before extraction?", [evidence(src)], client=client, agent_cfg=CFG, logger=logger
        )

        assert answer is not None
        assert answer.answer == "The pipeline strips boilerplate before extraction."
        assert len(answer.citations) == 1
        c = answer.citations[0]
        assert c.source_url == "https://example.com/a"
        assert c.scrape_timestamp == TS
        assert c.page_title == "Example page"
        assert c.section_heading == "How it works"
        # The quote must be a VERBATIM substring of the evidence text (answer.txt
        # prompt instruction). Metadata lining up is not enough — a fabricated
        # quote with correct source_url would still pass a provenance-only check.
        assert c.quote in src

        # Tool wiring: correct tool name and an Answer schema that requires citations.
        call = client.calls[0]
        assert call["tool_name"] == TOOL_NAME == "answer_question"
        props = call["tool_schema"]["properties"]
        assert "answer" in props
        assert "citations" in props
        assert "required" in call["tool_schema"] and "citations" in call["tool_schema"]["required"]
        assert "source_url" in call["tool_schema"]["$defs"]["Citation"]["properties"]
        # Evidence text with provenance reaches the prompt (now in user message).
        user_msg = call["messages"][1]["content"]
        assert src in user_msg
        assert "source_url=https://example.com/a" in user_msg

        assert failure_events(logger) == []
    finally:
        logger.close()


def test_verbatim_check_catches_fabricated_quotes(tmp_path):
    """A fabricated quote passes schema validation (metadata aligns) but is not
    in the evidence text. The schema alone cannot catch this — it is the
    prompt instruction plus the happy-path verbatim assertion that pin it
    down, which is exactly the drift a real LLM could introduce later."""
    ev = evidence("Real evidence text about boilerplate stripping.")
    client = FakeClient(
        {
            "answer": "x",
            "citations": [
                {
                    "source_url": "https://example.com/a",
                    "scrape_timestamp": TS,
                    "section_heading": "How it works",
                    "quote": "Completely fabricated claim never in the evidence.",
                }
            ],
        }
    )
    answer = generate_answer("Q", [ev], client=client, agent_cfg=CFG)
    assert answer is not None
    assert answer.citations[0].quote not in ev["text"]


def test_generate_answer_falls_back_and_logs_event_when_client_returns_none(tmp_path):
    logger = FetchLogger(tmp_path / "events.db")
    try:
        answer = generate_answer(
            "Q", [evidence("some text")], client=FakeClient(None), agent_cfg=CFG, logger=logger
        )
        events = failure_events(logger)

        assert answer is None
        assert len(events) == 1
        assert events[0]["reason"] == "model returned no parseable structured output"
        assert json.loads(events[0]["details_json"])["evidence_count"] == 1
    finally:
        logger.close()


def test_generate_answer_logs_event_on_schema_invalid_output(tmp_path):
    # Empty citations violates Answer's min_length=1 (traceability requirement).
    client = FakeClient({"answer": "x", "citations": []})
    logger = FetchLogger(tmp_path / "events.db")
    try:
        answer = generate_answer(
            "Q", [evidence("some text")], client=client, agent_cfg=CFG, logger=logger
        )
        events = failure_events(logger)

        assert answer is None
        assert len(events) == 1
        assert "schema validation failed" in events[0]["reason"]
    finally:
        logger.close()


def test_generate_answer_logs_event_when_llm_call_raises(tmp_path):
    def boom(_call):
        raise RuntimeError("provider down")

    logger = FetchLogger(tmp_path / "events.db")
    try:
        answer = generate_answer(
            "Q", [evidence("some text")], client=FakeClient(boom), agent_cfg=CFG, logger=logger
        )
        events = failure_events(logger)

        assert answer is None
        assert len(events) == 1
        assert "llm call failed" in events[0]["reason"]
    finally:
        logger.close()
