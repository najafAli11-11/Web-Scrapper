"""Schema-constrained extraction agent (Spec req. 6-8).

Takes stripped content in (Milestone 3 output), calls the LLM through the
provider-agnostic client with the `ExtractionResult` tool schema, and returns
a pydantic-validated `ExtractionResult`. Works in both batch and single-shot
modes with the same schema/logic (Spec req. 7).

Non-HTML routing (Spec req. 8):
  - html / text: extract directly from the text.
  - pdf: text-extract via pypdf first, then the same LLM path.
  - unknown: flagged (low-confidence result with provenance + notes), never
    silently dropped.

Retry policy: the internal parse-retry (bounded, default 1, from config) is
format recovery only — retrying to obtain a structurally valid response. It
is distinct from the validator's single content-repair budget (M5); total
semantic repair attempts per record stay at exactly one. When the validator
re-runs extraction for content repair, it passes `repair_errors` (a
pre-formatted multi-line string of validation errors) which is appended as a
user message rendered from prompts/repair.txt.

Events written to the shared SQLite `events` table:
  - extraction_attempt (outcome: extracted | flagged)
  - extraction_flagged (reason + details)
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from agents.config_loader import load_agent_config
from agents.llm.client import LLMClient, LiteLLMClient
from fetchers.logger import FetchLogger
from schemas.extraction import ContentType, ExtractionResult

PROMPT_PATH = Path(__file__).resolve().parents[1] / "agents" / "prompts" / "extract.txt"
REPAIR_PROMPT_PATH = Path(__file__).resolve().parents[1] / "agents" / "prompts" / "repair.txt"
TOOL_NAME = "extract_meaningful_content"
TOOL_DESCRIPTION = "Structured representation of the page's meaningful content"

RETRY_REASON_TEMPLATE = (
    "\n\nYour previous response could not be used: {error}. "
    "Return a corrected structured response now."
)


def _pdf_to_text(content: Union[bytes, str]) -> str:
    from pypdf import PdfReader

    stream = io.BytesIO(content) if isinstance(content, bytes) else io.StringIO(content)
    reader = PdfReader(stream)
    pages = [page.extract_text() for page in reader.pages]
    return "\n\n".join(text for text in pages if text and text.strip())


def mime_to_content_type(mime: Optional[str]) -> ContentType:
    """Map an HTTP Content-Type MIME string to the routable ContentType."""
    if not mime:
        return ContentType.UNKNOWN
    m = mime.lower()
    if "html" in m:
        return ContentType.HTML
    if "pdf" in m:
        return ContentType.PDF
    if "text" in m or "json" in m or "xml" in m:
        return ContentType.TEXT
    return ContentType.UNKNOWN


def _flagged_result(
    *,
    content_type: ContentType,
    source_url: str,
    scrape_timestamp: datetime,
    page_title: Optional[str],
    reason: str,
) -> ExtractionResult:
    return ExtractionResult(
        source_url=source_url,
        scrape_timestamp=scrape_timestamp,
        page_title=page_title,
        content_type=content_type,
        sections=[],
        confidence=0.0,
        extraction_notes=reason,
    )


def _log_attempt(
    logger: Optional[FetchLogger],
    url: str,
    *,
    outcome: str,
    mode: str,
    content_type: str,
    confidence: Optional[float],
    num_sections: int,
    truncated: bool,
    page_title: Optional[str],
    model: Optional[str],
    **extra,
) -> None:
    if logger is None:
        return
    details = {
        "mode": mode,
        "content_type": content_type,
        "confidence": confidence,
        "num_sections": num_sections,
        "truncated": truncated,
        "title": page_title,
        "model": model,
    }
    details.update(extra)
    logger.log_event(
        "extraction_attempt",
        url=url,
        outcome=outcome,
        details=details,
    )


def _log_flagged(
    logger: Optional[FetchLogger],
    url: str,
    *,
    reason: str,
    mode: str,
    content_type: str,
    parse_retry_used: int,
) -> None:
    if logger is None:
        return
    logger.log_event(
        "extraction_flagged",
        url=url,
        outcome="flagged",
        reason=reason,
        details={
            "mode": mode,
            "content_type": content_type,
            "parse_retry_used": parse_retry_used,
        },
    )


def _build_messages(
    *,
    mode: str,
    content_type: ContentType,
    source_url: str,
    scrape_timestamp: datetime,
    page_title: Optional[str],
    content: str,
    repair_errors: Optional[str] = None,
) -> list[dict]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        mode="single" if mode == "single" else "batch",
        content_type=content_type.value,
        source_url=source_url,
        scrape_timestamp=scrape_timestamp.isoformat(),
        page_title=page_title or "",
        content=content,
    )
    if "---" in prompt:
        sys_part, user_part = prompt.split("---", 1)
        messages = [
            {"role": "system", "content": sys_part.strip()},
            {"role": "user", "content": user_part.strip()},
        ]
    else:
        messages = [{"role": "system", "content": prompt}]
    if repair_errors:
        repair_block = REPAIR_PROMPT_PATH.read_text(encoding="utf-8").format(errors=repair_errors)
        messages.append({"role": "user", "content": repair_block})
    return messages


def extract_content(
    content: Union[str, bytes, None],
    *,
    content_type: ContentType,
    source_url: str,
    scrape_timestamp: Optional[datetime] = None,
    page_title: Optional[str] = None,
    mode: str = "batch",
    client: Optional[LLMClient] = None,
    agent_cfg: Optional[dict] = None,
    logger: Optional[FetchLogger] = None,
    repair_errors: Optional[str] = None,
) -> ExtractionResult:
    """Extract a structured ExtractionResult from cleaned content.

    Never raises on content problems — every failure path resolves to a
    flagged, low-confidence result that carries provenance (Rule 5).

    `repair_errors`, when set, is a pre-formatted multi-line string of
    validation failures (from the validator's content-repair pass) appended
    to the prompt via prompts/repair.txt.
    """
    agent_cfg = agent_cfg if agent_cfg is not None else load_agent_config()
    client = client if client is not None else LiteLLMClient(agent_cfg, logger=logger)
    scrape_timestamp = scrape_timestamp or datetime.now(timezone.utc)
    llm_cfg = agent_cfg["llm"]
    parse_retry_budget = int(llm_cfg.get("parse_retry", 1))
    model_name = f"{llm_cfg['provider']}/{llm_cfg['model']}"

    if content_type == ContentType.UNKNOWN:
        reason = "unsupported content type: no extraction path exists"
        result = _flagged_result(
            content_type=content_type,
            source_url=source_url,
            scrape_timestamp=scrape_timestamp,
            page_title=page_title,
            reason=reason,
        )
        _log_attempt(
            logger, source_url, outcome="flagged", mode=mode,
            content_type=content_type.value, confidence=0.0, num_sections=0,
            truncated=False, page_title=page_title, model=model_name,
        )
        _log_flagged(logger, source_url, reason=reason, mode=mode,
                     content_type=content_type.value, parse_retry_used=0)
        return result

    if content_type == ContentType.PDF:
        try:
            if content:
                content = _pdf_to_text(content)
            else:
                content = ""
        except Exception as exc:
            reason = f"pdf text extraction failed: {exc}"
            result = _flagged_result(
                content_type=content_type,
                source_url=source_url,
                scrape_timestamp=scrape_timestamp,
                page_title=page_title,
                reason=reason,
            )
            _log_attempt(
                logger, source_url, outcome="flagged", mode=mode,
                content_type=content_type.value, confidence=0.0, num_sections=0,
                truncated=False, page_title=page_title, model=model_name,
            )
            _log_flagged(logger, source_url, reason=reason, mode=mode,
                         content_type=content_type.value, parse_retry_used=0)
            return result

    if not content or not str(content).strip():
        reason = "empty content: nothing to extract"
        result = _flagged_result(
            content_type=content_type,
            source_url=source_url,
            scrape_timestamp=scrape_timestamp,
            page_title=page_title,
            reason=reason,
        )
        _log_attempt(
            logger, source_url, outcome="flagged", mode=mode,
            content_type=content_type.value, confidence=0.0, num_sections=0,
            truncated=False, page_title=page_title, model=model_name,
        )
        _log_flagged(logger, source_url, reason=reason, mode=mode,
                     content_type=content_type.value, parse_retry_used=0)
        return result

    messages = _build_messages(
        mode=mode,
        content_type=content_type,
        source_url=source_url,
        scrape_timestamp=scrape_timestamp,
        page_title=page_title,
        content=str(content),
        repair_errors=repair_errors,
    )

    raw: Optional[dict] = None
    error: Optional[str] = None
    parse_retry_used = 0
    for attempt in range(parse_retry_budget + 1):
        if error:
            messages = messages + [
                {"role": "user", "content": RETRY_REASON_TEMPLATE.format(error=error)}
            ]
        try:
            raw = client.complete_structured(
                messages=messages,
                tool_name=TOOL_NAME,
                tool_schema=ExtractionResult.model_json_schema(),
                temperature=float(llm_cfg["temperature"]),
                max_tokens=int(llm_cfg["max_tokens"]),
            )
        except Exception as exc:
            raw = None
            error = f"llm call failed: {exc}"
        if raw is None:
            error = error or "model returned no parseable structured output"
            parse_retry_used += 1
            _log_attempt(
                logger, source_url, outcome="retrying", mode=mode,
                content_type=content_type.value, confidence=0.0, num_sections=0,
                truncated=False, page_title=page_title, model=model_name,
                attempt=attempt + 1, error=error,
            )
            continue
        try:
            result = ExtractionResult.model_validate(raw)
        except ValidationError as exc:
            error = f"schema validation failed: {exc}"
            raw = None
            parse_retry_used += 1
            _log_attempt(
                logger, source_url, outcome="retrying", mode=mode,
                content_type=content_type.value, confidence=0.0, num_sections=0,
                truncated=False, page_title=page_title, model=model_name,
                attempt=attempt + 1, error=error,
            )
            continue
        # Provenance (Rule 4) belongs to the pipeline, not the model: the fetch
        # layer already determined these, and content_type reflects how this
        # record was routed (Spec req. 8). Never trust the model to relabel them.
        result.source_url = source_url
        result.scrape_timestamp = scrape_timestamp
        result.content_type = content_type
        _log_attempt(
            logger, source_url, outcome="extracted", mode=mode,
            content_type=content_type.value, confidence=result.confidence,
            num_sections=len(result.sections), truncated=result.truncated,
            page_title=result.page_title, model=model_name,
        )
        return result

    reason = error or "extraction failed after retries"
    result = _flagged_result(
        content_type=content_type,
        source_url=source_url,
        scrape_timestamp=scrape_timestamp,
        page_title=page_title,
        reason=reason,
    )
    _log_attempt(
        logger, source_url, outcome="flagged", mode=mode,
        content_type=content_type.value, confidence=0.0, num_sections=0,
        truncated=False, page_title=page_title, model=model_name,
    )
    _log_flagged(logger, source_url, reason=reason, mode=mode,
                 content_type=content_type.value, parse_retry_used=parse_retry_used)
    return result
