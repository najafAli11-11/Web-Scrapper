"""Schema-constrained chat answer generation (M9).

Consumer-side chat agent, NOT the scraper pipeline: invoked by the UI's Chat
view, never by the orchestrator — Rule 1 control flow stays LLM-free in the
scraper. Rules 2 and 5 apply: output must validate against `Answer`, and a
failure is logged as an `answer_generation_failed` event (visible in the M9
Logs view) rather than silently dropped; the caller then falls back to
surfacing the raw evidence.

Single attempt by design — no parse-retry budget like the extractor: a failed
chat answer has no persistence risk, so the raw-evidence fallback is the
safety net. A retry loop would only add latency to a chat path whose ethos is
low latency (single-shot). This is a deliberate deviation from the extractor's
pattern, not an oversight.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from agents.llm.client import LLMClient, LiteLLMClient
from fetchers.logger import FetchLogger
from schemas.answer import Answer

PROMPT_PATH = Path(__file__).resolve().parents[1] / "agents" / "prompts" / "answer.txt"
TOOL_NAME = "answer_question"


def _unwrap_answer(answer: Answer, *, logger=None, url: str = "") -> Answer:
    """Unwrap JSON-inside-JSON: when the model returns {\"answer\": ..., \"citations\": ...}
    as the answer string instead of as the top-level structure."""
    text = answer.answer.strip()
    if not text.startswith("{"):
        return answer
    try:
        import json
        inner = json.loads(text)
        if isinstance(inner, dict) and "answer" in inner:
            citations = answer.citations
            if not citations and "citations" in inner:
                citations = inner["citations"]
            unwrapped = Answer(answer=inner["answer"], citations=citations)
            if logger is not None:
                logger.log_event(
                    "answer_unwrap_rescue",
                    url=url,
                    outcome="unwrapped",
                    reason="JSON-inside-JSON detected and unwrapped",
                    details={"original_length": len(text), "answer_length": len(unwrapped.answer)},
                )
            return unwrapped
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return answer


def _render_evidence(evidence: list[dict]) -> str:
    blocks: list[str] = []
    for i, ev in enumerate(evidence, start=1):
        prov = ev.get("provenance") or {}
        src = prov.get("source_url", "?")
        heading = prov.get("section_heading")
        label = f"[{i}] source_url={src}"
        if heading:
            label += f" section_heading={heading}"
        blocks.append(f"{label}\n{ev.get('text', '')}")
    return "\n\n".join(blocks)


def _fallback_from_markdown(text: str, evidence: list[dict]) -> Optional[Answer]:
    """Extract an Answer from markdown prose when the model ignores the schema.

    Handles the common pattern where the model returns:
        **Answer**
        <answer text>

        **Citations**
        - source_url: ...
          quote: ...
    """
    answer_match = re.search(
        r"\*\*Answer\*\*\s*\n(.*?)(?=\n\*\*|\Z)", text, re.DOTALL
    )
    answer_text = answer_match.group(1).strip() if answer_match else text.strip()
    if not answer_text:
        return None

    citations = []
    for ev in evidence:
        prov = ev.get("provenance") or {}
        src = prov.get("source_url", "")
        quote = ev.get("text", "")
        if not src or not quote:
            continue
        citations.append({
            "source_url": src,
            "scrape_timestamp": prov.get("scrape_timestamp", ""),
            "page_title": prov.get("page_title"),
            "section_heading": prov.get("section_heading"),
            "quote": quote[:300],
        })
        if len(citations) >= 3:
            break

    if not citations:
        return None
    try:
        return Answer(answer=answer_text, citations=citations)
    except ValidationError:
        return None


def generate_answer(
    question: str,
    evidence: list[dict],
    *,
    client: LLMClient,
    agent_cfg: dict,
    logger: Optional[FetchLogger] = None,
    url: Optional[str] = None,
) -> Optional[Answer]:
    """Synthesize a grounded Answer from RAG evidence.

    Returns the validated Answer, or None on failure (unparseable output or
    schema validation error). On failure an `answer_generation_failed` event
    is logged (Rule 5: flag, don't drop) so the event is queryable in the
    Logs view, and the caller falls back to raw evidence.
    """
    llm_cfg = agent_cfg["llm"]
    prompt = PROMPT_PATH.read_text(encoding="utf-8").format(
        question=question,
        evidence=_render_evidence(evidence),
    )
    if "---" in prompt:
        sys_part, user_part = prompt.split("---", 1)
        messages = [
            {"role": "system", "content": sys_part.strip()},
            {"role": "user", "content": user_part.strip()},
        ]
    else:
        messages = [{"role": "system", "content": prompt}]

    raw: Optional[dict] = None
    error: Optional[str] = None
    try:
        raw = client.complete_structured(
            messages=messages,
            tool_name=TOOL_NAME,
            tool_schema=Answer.model_json_schema(),
            temperature=float(llm_cfg["temperature"]),
            max_tokens=int(llm_cfg["max_tokens"]),
        )
    except Exception as exc:
        error = f"llm call failed: {exc}"

    if raw is not None:
        try:
            return _unwrap_answer(Answer.model_validate(raw), logger=logger, url=url)
        except ValidationError as exc:
            error = f"schema validation failed: {exc}"
    else:
        error = error or "model returned no parseable structured output"

    if isinstance(client, LiteLLMClient) and client.last_raw_text:
        fb = _fallback_from_markdown(client.last_raw_text, evidence)
        if fb is not None:
            if logger is not None:
                logger.log_event(
                    "answer_fallback_rescued",
                    url=url,
                    outcome="fallback_rescue",
                    reason=f"structural path failed ({error}), rescued from markdown",
                    details={"evidence_count": len(evidence), "question": question[:200]},
                )
            return fb

    if logger is not None:
        logger.log_event(
            "answer_generation_failed",
            url=url,
            outcome="failed",
            reason=error,
            details={
                "evidence_count": len(evidence),
                "question": question[:200],
            },
        )
    return None
