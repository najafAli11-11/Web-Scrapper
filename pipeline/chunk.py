"""Semantic chunking (Spec req. 10) — one chunk per extracted section.

Chunks follow the page's actual structure (sections surfaced by extraction),
never fixed token windows and never an LLM pass (AGENTS.md chunking rule).

Oversized single sections (Spec's "very large pages" edge case) are split
further with NO hard cap that silently truncates:
  * lines are greedily packed into chunks of <= max_chunk_chars
  * a single over-long line is cut at a sentence boundary, else whitespace,
    else (only when the token has no whitespace at all) mid-word
  * invariant: the concatenation of chunk.content across a section exactly
    equals the section's content — zero characters lost, chunk count unbounded
  * the page-level ExtractionResult.truncated flag is carried onto every
    chunk (a truncated page stays marked); the chunker never truncates itself

Empty sections are an invariant violation (the validator flags empty
sections before they reach here) and raise ValueError — the caller decides
whether to flag-and-continue (per-URL, Rule 6) or abort.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Optional

from schemas.chunk import DocumentChunk
from schemas.extraction import ExtractionResult

_SENTENCE_RE = re.compile(r"[.!?](?: |$)")


def make_chunk_id(source_url: str, section_index: int, chunk_index: int) -> str:
    """Deterministic chunk id — stable across re-ingestions of the same URL."""
    return hashlib.sha1(f"{source_url}\x00{section_index}\x00{chunk_index}".encode("utf-8")).hexdigest()


def _cut_long_line(line: str, max_chunk_chars: int) -> list[str]:
    """Split a single line longer than max_chunk_chars into <= max pieces.

    Prefers a sentence boundary, then whitespace, then (for a token with no
    whitespace) a bare length cut — the only case that breaks mid-word, since
    a >max single token cannot be split otherwise. Concatenation preserved.
    """
    pieces: list[str] = []
    rest = line
    while len(rest) > max_chunk_chars:
        window = rest[:max_chunk_chars]
        sentence_cuts = [m.end() for m in _SENTENCE_RE.finditer(window)]
        if sentence_cuts:
            cut = sentence_cuts[-1]
        else:
            ws = window.rfind(" ")
            cut = ws if ws > 0 else max_chunk_chars
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        pieces.append(rest)
    return pieces


def _split_section(content: str, max_chunk_chars: int) -> list[str]:
    """Split one section into <= max_chunk_chars pieces, preserving text exactly."""
    if len(content) <= max_chunk_chars:
        return [content]
    pieces: list[str] = []
    current = ""
    for line in content.splitlines(keepends=True):
        if current and len(current) + len(line) <= max_chunk_chars:
            current += line
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(line) <= max_chunk_chars:
            current = line
        else:
            for piece in _cut_long_line(line, max_chunk_chars):
                pieces.append(piece)
    if current:
        pieces.append(current)
    return pieces


def chunk_result(result: ExtractionResult, *, max_chunk_chars: int = 2000, logger=None) -> list[DocumentChunk]:
    """Convert a validated ExtractionResult into semantic chunks.

    One chunk per section (split further only when a section exceeds
    max_chunk_chars). Returns an empty list for a result with no sections is
    NOT silent: an empty result raises ValueError so the caller can flag the
    URL and continue (Rule 5 / Rule 6) instead of storing nothing.
    """
    if not result.sections:
        raise ValueError("no sections to chunk: extraction result is empty")
    ingest_timestamp = datetime.now(timezone.utc)
    chunks: list[DocumentChunk] = []
    sections_split: list[int] = []
    for section_index, section in enumerate(result.sections, start=1):
        if not section.content.strip():
            raise ValueError(f"section {section_index} has empty content")
        pieces = _split_section(section.content, max_chunk_chars)
        if len(pieces) > 1:
            sections_split.append(section_index)
        for chunk_index, piece in enumerate(pieces, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=make_chunk_id(result.source_url, section_index, chunk_index),
                    source_url=result.source_url,
                    scrape_timestamp=result.scrape_timestamp,
                    page_title=result.page_title,
                    content_type=result.content_type,
                    confidence=result.confidence,
                    section_heading=section.heading,
                    section_level=section.level,
                    chunk_index=chunk_index,
                    chunk_total=len(pieces),
                    content=piece,
                    truncated=result.truncated,
                    extraction_notes=result.extraction_notes,
                    ingest_timestamp=ingest_timestamp,
                )
            )
    if logger is not None:
        logger.log_event(
            "chunk_complete",
            url=result.source_url,
            outcome="ok",
            reason=None,
            details={
                "num_sections": len(result.sections),
                "num_chunks": len(chunks),
                "sections_split": sections_split,
                "max_chunk_chars": max_chunk_chars,
            },
        )
    return chunks
