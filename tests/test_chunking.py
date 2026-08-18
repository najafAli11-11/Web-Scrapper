"""Hermetic tests for Milestone 6 chunking (pipeline/chunk.py, schemas/chunk.py).

No network, no model. Covers: one-section-one-chunk with full provenance
metadata, oversized-section splitting with the exact-concatenation invariant,
sentence/whitespace/last-resort boundary cutting, chunk_index/chunk_total,
deterministic chunk ids, truncated-flag propagation, and the empty-result
ValueError contract.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pipeline.chunk import chunk_result, make_chunk_id
from schemas.extraction import ContentType, ExtractionResult, Section

TS = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def _result(sections: list[Section], *, truncated: bool = False, title: str = "Test Page") -> ExtractionResult:
    return ExtractionResult(
        source_url="https://example.com/article",
        scrape_timestamp=TS,
        page_title=title,
        content_type=ContentType.HTML,
        sections=sections,
        confidence=0.8,
        extraction_notes="note" if truncated else None,
        truncated=truncated,
    )


def _sec(content: str, heading: str | None = None, level: int | None = None) -> Section:
    return Section(heading=heading, content=content, level=level)


def test_single_section_becomes_single_chunk_with_provenance():
    chunks = chunk_result(_result([_sec("hello world", heading="Intro", level=1)]))
    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_index == 1 and c.chunk_total == 1
    assert c.content == "hello world"
    assert c.section_heading == "Intro"
    assert c.section_level == 1
    assert c.source_url == "https://example.com/article"
    assert c.scrape_timestamp == TS
    assert c.page_title == "Test Page"
    assert c.content_type == ContentType.HTML
    assert c.confidence == 0.8
    assert c.truncated is False
    assert c.ingest_timestamp.tzinfo is not None


def test_multiple_sections_each_become_one_chunk():
    chunks = chunk_result(
        _result([_sec("one", heading="A"), _sec("two", heading="B"), _sec("three")])
    )
    assert [c.section_heading for c in chunks] == ["A", "B", None]
    assert all(c.chunk_index == 1 and c.chunk_total == 1 for c in chunks)


def test_oversized_section_splits_into_ordered_chunks_with_exact_concat_invariant():
    paras = "\n\n".join(f"Paragraph number {i}. " + "word " * 40 for i in range(10))
    sections = [_sec(paras, heading="Long", level=2)]
    max_chars = 500
    chunks = chunk_result(_result(sections), max_chunk_chars=max_chars)
    assert len(chunks) > 1
    assert all(len(c.content) <= max_chars for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(1, len(chunks) + 1))
    assert all(c.chunk_total == len(chunks) for c in chunks)
    assert all(c.section_heading == "Long" and c.section_level == 2 for c in chunks)
    assert "".join(c.content for c in chunks) == sections[0].content


def test_long_line_without_paragraph_breaks_splits_at_sentence_boundaries():
    sentences = " ".join(f"Sentence number {i} of the long running text." for i in range(30))
    section = _sec(sentences)
    max_chars = 300
    chunks = chunk_result(_result([section]), max_chunk_chars=max_chars)
    assert len(chunks) > 1
    assert all(len(c.content) <= max_chars for c in chunks)
    assert "".join(c.content for c in chunks) == sentences
    assert all(c.content.rstrip().endswith((".", "!", "?")) for c in chunks[:-1])


def test_single_token_longer_than_max_splits_midword_as_last_resort_without_loss():
    token = "x" * 5000
    chunks = chunk_result(_result([_sec(token)]), max_chunk_chars=1000)
    assert len(chunks) > 1
    assert all(len(c.content) <= 1000 for c in chunks)
    assert "".join(c.content for c in chunks) == token


def test_concat_invariant_holds_for_mixed_structure():
    content = ("Short lead. " * 5) + ("\n\n") + (" ".join(f"Word{i}" for i in range(4000)))
    chunks = chunk_result(_result([_sec(content)]), max_chunk_chars=900)
    assert "".join(c.content for c in chunks) == content


def test_chunk_ids_are_deterministic_and_section_scoped():
    a = chunk_result(_result([_sec("content " * 300, heading="A")]), max_chunk_chars=200)
    b = chunk_result(_result([_sec("content " * 300, heading="A")]), max_chunk_chars=200)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert make_chunk_id("https://example.com/article", 1, 1) != make_chunk_id(
        "https://example.com/article", 2, 1
    )


def test_truncated_flag_and_notes_carried_onto_every_chunk():
    chunks = chunk_result(
        _result([_sec("content " * 300, heading="A")], truncated=True), max_chunk_chars=200
    )
    assert len(chunks) > 1
    assert all(c.truncated for c in chunks)
    assert all(c.extraction_notes == "note" for c in chunks)


def test_empty_result_raises_valueerror():
    result = ExtractionResult(
        source_url="https://example.com/article",
        scrape_timestamp=TS,
        content_type=ContentType.HTML,
        sections=[],
        confidence=0.4,
    )
    with pytest.raises(ValueError, match="no sections to chunk"):
        chunk_result(result)


def test_section_with_whitespace_only_content_raises_valueerror():
    with pytest.raises(ValueError, match="empty content"):
        chunk_result(_result([_sec("   \n  ")]))


def test_exactly_max_section_does_not_split():
    content = "a" * 2000
    chunks = chunk_result(_result([_sec(content)]), max_chunk_chars=2000)
    assert len(chunks) == 1
    assert chunks[0].content == content


def test_small_sections_beside_oversized_section_coexist():
    chunks = chunk_result(
        _result([_sec("small", heading="S"), _sec("big " * 300, heading="B")]),
        max_chunk_chars=200,
    )
    assert chunks[0].section_heading == "S"
    assert chunks[0].content == "small"
    assert chunks[1].section_heading == "B"
    assert all(c.chunk_index <= c.chunk_total for c in chunks)


def test_chunk_complete_event_logged(tmp_path):
    from fetchers.logger import FetchLogger
    with FetchLogger(tmp_path / "logs.db") as logger:
        result = _result([_sec("hello world", heading="Intro"), _sec("another section", heading="Body")])
        chunks = chunk_result(result, max_chunk_chars=2000, logger=logger)
        rows = logger.rows_for_url(result.source_url)
    events = [r for r in rows if r["event_type"] == "chunk_complete"]
    assert len(events) == 1
    import json
    details = json.loads(events[0]["details_json"])
    assert details["num_sections"] == 2
    assert details["num_chunks"] == len(chunks)
    assert details["sections_split"] == []


def test_chunk_complete_event_logs_split_sections(tmp_path):
    from fetchers.logger import FetchLogger
    with FetchLogger(tmp_path / "logs.db") as logger:
        long_text = "word " * 500
        result = _result([_sec(long_text, heading="Long")])
        chunk_result(result, max_chunk_chars=200, logger=logger)
        rows = logger.rows_for_url(result.source_url)
    events = [r for r in rows if r["event_type"] == "chunk_complete"]
    assert len(events) == 1
    import json
    details = json.loads(events[0]["details_json"])
    assert details["sections_split"] == [1]
