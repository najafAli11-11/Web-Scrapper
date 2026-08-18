"""Hermetic tests for Milestone 5 validator (agents/validator.py).

No LLM calls and no network: a stub LLM client is injected. Covers the
deterministic checks, the retry-once-then-flag policy (Spec req. 9) for
every error type, the unknown-content-type instant-flag exception, config
overrides, provenance preservation, mode threading, and the
validation_attempt / validation_flagged log events.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.config_loader import load_agent_config
from agents.extractor import extract_content
from agents.validator import deterministic_errors, validate_result
from fetchers.logger import FetchLogger
from schemas.extraction import ContentType, ExtractionResult


TS = "2026-08-13T12:00:00+00:00"
URL = "https://example.com/article"


class StubClient:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_structured(self, *, messages, tool_name, tool_schema, temperature, max_tokens):
        self.calls.append({"messages": messages})
        if not self.responses:
            return None
        return self.responses.pop(0)


def _cfg(**validator_overrides) -> dict:
    validator = {"min_confidence": 0.5, "repair_budget": 1}
    validator.update(validator_overrides)
    return {"llm": {"provider": "stub", "model": "stub",
                    "temperature": 0.1, "max_tokens": 1000},
            "validator": validator}


def _result(confidence: float = 0.9, sections: list | None = None,
            content_type: ContentType = ContentType.HTML) -> ExtractionResult:
    return ExtractionResult(
        source_url=URL,
        scrape_timestamp=datetime.fromisoformat(TS),
        page_title="Article",
        content_type=content_type,
        sections=sections if sections is not None else [{"heading": "h", "content": "real body"}],
        confidence=confidence,
    )


def _fixed_response(confidence: float = 0.9) -> dict:
    return {
        "source_url": URL,
        "scrape_timestamp": TS,
        "page_title": "Article",
        "content_type": "html",
        "sections": [{"heading": "h", "content": "real body"}],
        "confidence": confidence,
        "truncated": False,
    }


def _broken_response() -> dict:
    return {
        "source_url": URL,
        "scrape_timestamp": TS,
        "page_title": "Article",
        "content_type": "html",
        "sections": [],
        "confidence": 0.0,
    }


def _broken_section_response() -> dict:
    return {
        "source_url": URL,
        "scrape_timestamp": TS,
        "page_title": "Article",
        "content_type": "html",
        "sections": [{"heading": "h", "content": "   \n  "}],
        "confidence": 0.9,
    }


# --- valid path ---------------------------------------------------------

def test_valid_result_no_llm_call(tmp_path):
    client = StubClient([])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, res = validate_result(_result(), content="body", client=client,
                                  agent_cfg=_cfg(), logger=logger)
        rows = logger.rows_for_url(URL)
    assert vr.is_valid and vr.retry_count == 0 and vr.errors == []
    assert res is not None
    assert client.calls == []
    attempts = [r for r in rows if r["event_type"] == "validation_attempt"]
    assert len(attempts) == 1 and attempts[0]["outcome"] == "valid"
    assert not any(r["event_type"] == "validation_flagged" for r in rows)


# --- repair then valid ---------------------------------------------------

def test_broken_repairs_once_then_valid(tmp_path):
    broken = _result(confidence=0.0, sections=[])
    client = StubClient([_fixed_response()])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, res = validate_result(broken, content="page text", client=client,
                                  agent_cfg=_cfg(), logger=logger)
    assert vr.is_valid and vr.retry_count == 1
    assert len(client.calls) == 1
    repair_msg = client.calls[0]["messages"][-1]["content"]
    assert "Your previous extraction failed validation" in repair_msg
    assert "1. no content extracted: sections list is empty" in repair_msg
    assert "2. confidence below threshold" in repair_msg
    assert "[" not in repair_msg.split("Validation errors:")[1]
    with FetchLogger(tmp_path / "logs.db") as logger:
        outcomes = [r["outcome"] for r in logger.rows_for_url(URL)
                    if r["event_type"] == "validation_attempt"]
    assert outcomes == ["repairing", "valid"]


# --- stays broken -> flag ------------------------------------------------

def test_stays_broken_flags_after_exactly_one_repair(tmp_path):
    broken = _result(confidence=0.0, sections=[])
    client = StubClient([_broken_response()])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, res = validate_result(broken, content="page text", client=client,
                                  agent_cfg=_cfg(), logger=logger)
        import json
        rows = logger.rows_for_url(URL)
        assert [r["outcome"] for r in rows if r["event_type"] == "validation_attempt"] == ["repairing", "failed"]
        flags = [r for r in rows if r["event_type"] == "validation_flagged"]
        assert len(flags) == 1
        assert "confidence below threshold" in flags[0]["reason"]
        assert json.loads(flags[0]["details_json"])["flag_reason"] == "repair_budget_exhausted"
    assert vr.is_valid is False
    assert vr.should_retry is False
    assert vr.retry_count == 1
    assert len(client.calls) == 1
    assert vr.errors == ["confidence below threshold"]


# --- every deterministic error goes through repair-then-flag -------------

def test_low_confidence_repairs_once_then_flags(tmp_path):
    client = StubClient([_fixed_response(confidence=0.3)])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(_result(confidence=0.3), content="page text",
                                client=client, agent_cfg=_cfg(), logger=logger)
        flagged = any(r["event_type"] == "validation_flagged" for r in logger.rows_for_url(URL))
    assert vr.is_valid is False
    assert len(client.calls) == 1
    assert "confidence below threshold" in vr.errors
    assert flagged


def test_empty_section_content_repairs_once_then_flags(tmp_path):
    empty_section = ExtractionResult(
        source_url=URL,
        scrape_timestamp=datetime.fromisoformat(TS),
        page_title="Article",
        content_type=ContentType.HTML,
        sections=[{"heading": "h", "content": "   \n  "}],
        confidence=0.9,
    )
    client = StubClient([_broken_section_response()])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, res = validate_result(empty_section, content="page text",
                                  client=client, agent_cfg=_cfg(), logger=logger)
        flagged = any(r["event_type"] == "validation_flagged" for r in logger.rows_for_url(URL))
    assert vr.is_valid is True
    assert len(res.sections) == 1
    assert res.sections[0].content == "page text"


def test_unknown_content_type_flags_immediately_zero_repairs(tmp_path):
    unknown = _result(content_type=ContentType.UNKNOWN)
    client = StubClient([])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(unknown, content=None, client=client,
                                agent_cfg=_cfg(), logger=logger)
        import json
        rows = logger.rows_for_url(URL)
        flags = [r for r in rows if r["event_type"] == "validation_flagged"]
        assert json.loads(flags[0]["details_json"])["flag_reason"] == "unsupported_content_type"
        assert [r["outcome"] for r in rows if r["event_type"] == "validation_attempt"] == ["failed"]
    assert vr.is_valid is False
    assert vr.retry_count == 0
    assert client.calls == []
    assert any("no repair possible" in e for e in vr.errors)


# --- provenance, modes, config, helpers ----------------------------------

def test_validation_result_preserves_provenance(tmp_path):
    broken = _result(confidence=0.0, sections=[])
    client = StubClient([_fixed_response()])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(broken, content="page text", client=client,
                                agent_cfg=_cfg(), logger=logger)
    assert vr.source_url == URL
    assert vr.scrape_timestamp == datetime.fromisoformat(TS)


def test_mode_threads_into_repair_prompt(tmp_path):
    for mode in ("batch", "single"):
        client = StubClient([_fixed_response()])
        with FetchLogger(tmp_path / f"{mode}.db") as logger:
            validate_result(_result(confidence=0.0, sections=[]), content="page text",
                            mode=mode, client=client, agent_cfg=_cfg(), logger=logger)
        assert len(client.calls) == 1
        repair_prompt = client.calls[0]["messages"][0]["content"]
        assert f"Extraction mode: {mode}" in repair_prompt


def test_config_override_min_confidence(tmp_path):
    result_08 = _result(confidence=0.8)
    client_strict = StubClient([_fixed_response(confidence=0.8)])
    with FetchLogger(tmp_path / "strict.db") as logger:
        vr, _ = validate_result(result_08, content="page text", client=client_strict,
                                agent_cfg=_cfg(min_confidence=0.9), logger=logger)
    assert vr.is_valid is False
    assert len(client_strict.calls) == 1

    client_lenient = StubClient([])
    with FetchLogger(tmp_path / "lenient.db") as logger:
        vr, _ = validate_result(result_08, content="page text", client=client_lenient,
                                agent_cfg=_cfg(), logger=logger)
    assert vr.is_valid is True
    assert client_lenient.calls == []


def test_config_override_repair_budget_zero(tmp_path):
    client = StubClient([])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(_result(confidence=0.0, sections=[]), content="page text",
                                client=client, agent_cfg=_cfg(repair_budget=0), logger=logger)
        rows = logger.rows_for_url(URL)
        assert [r["outcome"] for r in rows if r["event_type"] == "validation_attempt"] == ["failed"]
        flagged = any(r["event_type"] == "validation_flagged" for r in rows)
    assert vr.is_valid is False
    assert vr.retry_count == 0
    assert client.calls == []
    assert flagged


def test_config_override_repair_budget_two(tmp_path):
    client = StubClient([_broken_response(), _broken_response()])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(_result(confidence=0.0, sections=[]), content="page text",
                                client=client, agent_cfg=_cfg(repair_budget=2), logger=logger)
    assert vr.is_valid is False
    assert vr.retry_count == 2
    assert len(client.calls) == 2


def test_agent_cfg_none_loads_real_config():
    vr, _ = validate_result(_result(), content="body")
    assert vr.is_valid is True
    assert vr.source_url == URL


def test_repair_errors_rendered_into_extractor_prompt():
    client = StubClient([_fixed_response()])
    extract_content(
        "page text",
        content_type=ContentType.HTML,
        source_url=URL,
        scrape_timestamp=datetime.fromisoformat(TS),
        client=client,
        agent_cfg=_cfg(),
        repair_errors="1. section 0 has empty content",
    )
    assert len(client.calls) == 1
    assert "Your previous extraction failed validation" in client.calls[0]["messages"][-1]["content"]
    assert "1. section 0 has empty content" in client.calls[0]["messages"][-1]["content"]


# --- deterministic_errors unit checks ------------------------------------

def test_deterministic_errors_checks():
    assert deterministic_errors(_result()) == []
    errors = deterministic_errors(_result(confidence=0.0, sections=[]))
    assert errors == ["no content extracted: sections list is empty", "confidence below threshold"]
    ws = _result(sections=[{"heading": "h", "content": " \t "}])
    assert deterministic_errors(ws) == ["section 0 has empty content"]
    unknown = _result(content_type=ContentType.UNKNOWN)
    assert deterministic_errors(unknown) == ["unsupported content type: no extraction path exists (no repair possible)"]


def test_real_agent_config_validates_with_validator_key():
    cfg = load_agent_config()
    assert cfg["validator"]["min_confidence"] == 0.5
    assert cfg["validator"]["repair_budget"] == 1


# --- data_integrity obstacle event logging --------------------------------

def test_data_integrity_obstacle_logged_on_flag(tmp_path):
    """When data_integrity is enabled in obstacle_cfg, a validation failure
    logs an obstacle_detected event (Rule 7: obstacle config is source of
    truth for obstacle policy)."""
    client = StubClient([_broken_response(), _broken_response()])
    obstacle_cfg = {
        "data_integrity": {"enabled": True, "detection_method": "schema_validation_failed", "policy": "flag_and_skip"},
    }
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(
            _result(sections=[], confidence=0.0),
            content="body",
            client=client,
            agent_cfg=_cfg(repair_budget=1),
            logger=logger,
            obstacle_cfg=obstacle_cfg,
        )
        rows = logger.rows_for_url(URL)

    assert not vr.is_valid
    obstacle_events = [r for r in rows if r["event_type"] == "obstacle_detected"]
    assert len(obstacle_events) == 1
    details = __import__("json").loads(obstacle_events[0]["details_json"])
    assert details["obstacle"] == "data_integrity"
    assert details["detection_method"] == "schema_validation_failed"
    assert details["policy"] == "flag_and_skip"


def test_data_integrity_obstacle_not_logged_when_disabled(tmp_path):
    """When data_integrity is disabled in obstacle_cfg, no obstacle_detected
    event is logged on validation failure."""
    client = StubClient([_broken_response(), _broken_response()])
    obstacle_cfg = {
        "data_integrity": {"enabled": False, "detection_method": "schema_validation_failed", "policy": "flag_and_skip"},
    }
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(
            _result(sections=[], confidence=0.0),
            content="body",
            client=client,
            agent_cfg=_cfg(repair_budget=1),
            logger=logger,
            obstacle_cfg=obstacle_cfg,
        )
        rows = logger.rows_for_url(URL)

    assert not vr.is_valid
    obstacle_events = [r for r in rows if r["event_type"] == "obstacle_detected"]
    assert len(obstacle_events) == 0


def test_data_integrity_backward_compatible_no_obstacle_cfg(tmp_path):
    """validate_result works without obstacle_cfg (backward compatibility)."""
    client = StubClient([_broken_response(), _broken_response()])
    with FetchLogger(tmp_path / "logs.db") as logger:
        vr, _ = validate_result(
            _result(sections=[], confidence=0.0),
            content="body",
            client=client,
            agent_cfg=_cfg(repair_budget=1),
            logger=logger,
        )
        rows = logger.rows_for_url(URL)

    assert not vr.is_valid
    obstacle_events = [r for r in rows if r["event_type"] == "obstacle_detected"]
    assert len(obstacle_events) == 0
