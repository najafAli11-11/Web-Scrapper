"""Boilerplate stripping — the "strip before you prompt" stage (AGENTS.md
Rule 3, SPEC.md req. 5).

Takes already-decoded HTML (or plain text) in, returns cleaned text plus a
block-level representation that preserves heading hierarchy where Trafilatura
surfaced it — the input the extractor agent (M4) consumes and the structure
semantic chunking (Spec req. 10) needs.

Scope boundary (Milestone 4, Spec req. 8): content-type *classification* and
PDF/non-HTML *routing* live entirely in the extractor layer, never here. This
module only ever receives HTML or plain-text strings and makes no decision
about what to do with a PDF.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import lxml.etree as etree
import lxml.html
import trafilatura

from fetchers.logger import FetchLogger

_HEADING_RE = re.compile(r"[hH]([1-9])")
_INLINE_BLOCK_TAGS = {"code", "quote"}


def _local(tag: str) -> str:
    """Local tag name, tolerant of XML namespaces."""
    return tag.rsplit("}", 1)[-1]


class StripOutcome(str, Enum):
    """Terminal outcome of one stripping pass.

    - STRIPPED: Trafilatura produced cleaned content.
    - FALLBACK: Trafilatura found nothing usable; tag-stripped text was kept
      so the pipeline never silently drops content (Rule 5). No routing
      decision is implied.
    - EMPTY: nothing usable at all (blank input, no text anywhere).
    """

    STRIPPED = "stripped"
    FALLBACK = "fallback"
    EMPTY = "empty"


@dataclass
class StripBlock:
    """One structural unit of cleaned content.

    Content before any heading lands in a block with `heading=None`; each
    heading starts a new block that subsequent paragraphs accumulate into.
    """

    heading: Optional[str] = None
    level: Optional[int] = None
    text: str = ""


@dataclass
class StripResult:
    url: Optional[str] = None
    title: Optional[str] = None
    outcome: StripOutcome = StripOutcome.EMPTY
    text: str = ""
    blocks: list[StripBlock] = field(default_factory=list)
    method: str = "trafilatura"
    chars_before: int = 0
    chars_after: int = 0
    num_blocks: int = 0
    num_tables: int = 0


def _node_text(el) -> str:
    return " ".join(" ".join(el.itertext()).split())


def _heading_level(el) -> Optional[int]:
    m = _HEADING_RE.search(el.get("rend", ""))
    return int(m.group(1)) if m else None


def _normalize_block(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    joined = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def _append_text(current: StripBlock, text: str, sep: str = "\n\n") -> None:
    text = _node_text(text) if not isinstance(text, str) else " ".join(text.split())
    if text:
        current.text = current.text + (sep if current.text else "") + text


def _render_table(table_el) -> str:
    rows = []
    for row in table_el.iter():
        if _local(row.tag) != "row":
            continue
        cells = [_node_text(c) for c in row if _local(c.tag) in ("cell", "head")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _build_blocks(root) -> tuple[list[StripBlock], int]:
    """Walk Trafilatura's cleaned XML into heading-scoped blocks."""
    blocks: list[StripBlock] = []
    tables = 0
    current = StripBlock()

    def flush() -> None:
        nonlocal current
        text = _normalize_block(current.text)
        if current.heading or text:
            current.text = text
            blocks.append(current)
        current = StripBlock()

    for el in root.iter():
        if _local(el.tag) != "main":
            continue
        for child in el:
            tag = _local(child.tag)
            if tag == "head":
                flush()
                current.heading = _node_text(child)
                current.level = _heading_level(child)
            elif tag == "p":
                _append_text(current, _node_text(child))
            elif tag == "list":
                items = [_node_text(i) for i in child.iter() if _local(i.tag) == "item"]
                _append_text(current, "\n".join(x for x in items if x))
            elif tag == "table":
                _append_text(current, _render_table(child))
                tables += 1
            elif tag in _INLINE_BLOCK_TAGS:
                _append_text(current, _node_text(child))
    flush()
    return blocks, tables


def _title_from(html: str, *, logger=None, url: str = "") -> Optional[str]:
    try:
        tree = lxml.html.fromstring(html)
    except Exception as exc:
        if logger is not None:
            logger.log_event(
                "strip_parse_warning",
                url=url,
                outcome="title_extraction_failed",
                reason=f"lxml HTML parse failed for title extraction: {exc}",
                details={"stage": "title_extraction", "error": str(exc)},
            )
        return None
    for xpath in (".//title", ".//h1"):
        try:
            node = tree.find(xpath)
        except Exception:
            continue
        if node is not None:
            title = " ".join(node.itertext()).strip()
            if title:
                return title
    return None


def _looks_html(s: str) -> bool:
    return "<" in s and ">" in s


def _log_strip(logger: Optional[FetchLogger], result: StripResult) -> None:
    if logger is None:
        return
    try:
        logger.log_event(
            "strip_attempt",
            url=result.url,
            outcome=result.outcome.value,
            details={
                "method": result.method,
                "title": result.title,
                "chars_before": result.chars_before,
                "chars_after": result.chars_after,
                "num_blocks": result.num_blocks,
                "num_tables": result.num_tables,
            },
        )
    except Exception as exc:  # pragma: no cover - best-effort logging
        print(f"[strip] failed to log event: {exc}", file=sys.stderr)


def strip_html(
    html: Optional[str],
    *,
    url: Optional[str] = None,
    logger: Optional[FetchLogger] = None,
) -> StripResult:
    """Strip boilerplate from an HTML/plain-text string.

    Returns cleaned text + structural blocks. Never raises on malformed,
    empty, or non-HTML input — the caller routes by content type upstream
    (fetch layer) and downstream (M4 extractor).
    """
    chars_before = len(html) if html else 0
    result = StripResult(url=url, chars_before=chars_before)

    if not html or not html.strip():
        result.outcome = StripOutcome.EMPTY
        _log_strip(logger, result)
        return result

    result.title = _title_from(html, logger=logger, url=url)

    try:
        xml = trafilatura.extract(
            html,
            url=url,
            output_format="xml",
            favor_recall=True,
            include_tables=True,
            include_comments=False,
        )
    except Exception as exc:
        xml = None
        if logger is not None:
            logger.log_event(
                "strip_parse_warning",
                url=url,
                outcome="fallback",
                reason=f"trafilatura raised: {exc}",
                details={"stage": "trafilatura"},
            )

    if xml:
        try:
            root = etree.fromstring(xml.encode("utf-8"))
            blocks, tables = _build_blocks(root)
        except Exception as exc:
            blocks, tables = [], 0
            if logger is not None:
                logger.log_event(
                    "strip_parse_warning",
                    url=url,
                    outcome="fallback",
                    reason=f"xml parse failed: {exc}",
                    details={"stage": "etree_parse"},
                )
        if blocks:
            result.blocks = blocks
            result.num_blocks = len(blocks)
            result.num_tables = tables
            result.outcome = StripOutcome.STRIPPED
            result.text = "\n\n".join(
                (b.heading + "\n" if b.heading else "") + b.text
                for b in blocks
                if b.text
            )
            result.chars_after = len(result.text)
            _log_strip(logger, result)
            return result

    # Fallback: Trafilatura found nothing usable. Keep the text so nothing is
    # silently dropped (Rule 5); this is NOT a content-type routing decision.
    result.method = "html-text-content" if _looks_html(html) else "raw"
    if _looks_html(html):
        try:
            text = lxml.html.fromstring(html).text_content()
        except Exception as exc:
            text = html
            if logger is not None:
                logger.log_event(
                    "strip_parse_warning",
                    url=url,
                    outcome="fallback",
                    reason=f"lxml fallback failed: {exc}",
                    details={"stage": "lxml_fallback"},
                )
    else:
        text = html
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        result.outcome = StripOutcome.EMPTY
    else:
        block = StripBlock(text=text)
        result.blocks = [block]
        result.num_blocks = 1
        result.text = text
        result.chars_after = len(text)
        result.outcome = StripOutcome.FALLBACK
    _log_strip(logger, result)
    return result
