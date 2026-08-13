"""Validator agent (Spec req. 9, Rule 5).

Deterministic checks first (Rule 1: no LLM in control flow) — if an
extraction fails validation and the repair budget is not exhausted, the
extractor is re-run on the ORIGINAL content with the validation errors fed
back as `repair_errors` (Spec req. 9: one repair attempt, then flag). The
retry-then-flag policy applies uniformly to every deterministic error; the
single exception is `content_type == unknown`, which flags immediately
without repair — the extractor already had no extraction path for the type
(Spec req. 8), so re-prompting cannot help.

Output uses the existing `ValidationResult` schema (its model validator
encodes the budget-then-flag rule; retry_count tracks attempts). Every
returned ValidationResult carries the record's own provenance (Rule 4).

Flag-reason bookkeeping for the event logger (added in the following
commit): both flag modes share this return shape, but are distinguishable
by `errors` (the unknown-type error string is unique to this stage) and by
`retry_count` on the ValidationResult — both in scope here.
"""

from __future__ import annotations

from typing import Optional, Union

from agents.config_loader import load_agent_config
from agents.extractor import extract_content
from agents.llm.client import LLMClient
from fetchers.logger import FetchLogger
from schemas.extraction import ContentType, ExtractionResult, ValidationResult

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_REPAIR_BUDGET = 1

# Distinct from the extractor's "unsupported content type: no extraction
# path exists" flag: the "no repair possible" clause names the validator's
# instant-flag decision, which only this stage makes.
UNKNOWN_TYPE_ERROR = "unsupported content type: no extraction path exists (no repair possible)"


def _format_errors(errors: list[str]) -> str:
    """Pre-formatted multi-line error block for prompts/repair.txt.

    Never a raw list repr — the repair prompt expects a readable numbered
    list, and that contract is fixed at the repair.txt boundary.
    """
    return "\n".join(f"{i}. {e}" for i, e in enumerate(errors, start=1))


def deterministic_errors(
    result: ExtractionResult,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[str]:
    """Deterministic sanity checks; empty list means the result is valid.

    `min_confidence` is a parameter so the checks are testable without
    config; validate_result passes the configured value through.
    """
    errors: list[str] = []
    if not result.sections:
        errors.append("no content extracted: sections list is empty")
    for i, section in enumerate(result.sections):
        if not section.content or not section.content.strip():
            errors.append(f"section {i} has empty content")
    if result.confidence < min_confidence:
        errors.append("confidence below threshold")
    if result.content_type == ContentType.UNKNOWN:
        errors.append(UNKNOWN_TYPE_ERROR)
    return errors


def validate_result(
    result: ExtractionResult,
    *,
    content: Union[str, bytes, None],
    mode: str = "batch",
    client: Optional[LLMClient] = None,
    agent_cfg: Optional[dict] = None,
    logger: Optional[FetchLogger] = None,
    retry_count: int = 0,
) -> tuple[ValidationResult, ExtractionResult]:
    """Validate an ExtractionResult, repairing once then flagging (Spec req. 9).

    `content` is the ORIGINAL cleaned page content the extractor consumed —
    repairs re-run the extractor on it with the validation errors appended
    (prompts/repair.txt). Provenance is derived from `result` itself, the
    single source of truth; the extractor re-asserts the same values on any
    repaired result.

    `agent_cfg=None` loads the versioned config (the codebase convention
    used by extract_content); hand-built cfgs may omit the "validator" key
    and fall back to DEFAULT_MIN_CONFIDENCE / DEFAULT_REPAIR_BUDGET.
    """
    agent_cfg = agent_cfg if agent_cfg is not None else load_agent_config()
    validator_cfg = agent_cfg.get("validator", {}) or {}
    min_confidence = float(validator_cfg.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
    repair_budget = int(validator_cfg.get("repair_budget", DEFAULT_REPAIR_BUDGET))

    errors = deterministic_errors(result, min_confidence=min_confidence)

    if not errors:
        return (
            ValidationResult(
                source_url=result.source_url,
                scrape_timestamp=result.scrape_timestamp,
                is_valid=True,
                errors=[],
                should_retry=False,
                retry_count=retry_count,
            ),
            result,
        )

    if result.content_type == ContentType.UNKNOWN or retry_count >= repair_budget:
        return (
            ValidationResult(
                source_url=result.source_url,
                scrape_timestamp=result.scrape_timestamp,
                is_valid=False,
                errors=errors,
                should_retry=False,
                retry_count=retry_count,
            ),
            result,
        )

    repaired = extract_content(
        content,
        content_type=result.content_type,
        source_url=result.source_url,
        scrape_timestamp=result.scrape_timestamp,
        page_title=result.page_title,
        mode=mode,
        client=client,
        agent_cfg=agent_cfg,
        logger=logger,
        repair_errors=_format_errors(errors),
    )
    return validate_result(
        repaired,
        content=content,
        mode=mode,
        client=client,
        agent_cfg=agent_cfg,
        logger=logger,
        retry_count=retry_count + 1,
    )
