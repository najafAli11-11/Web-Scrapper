"""Hermetic tests for Milestone 6 storage (pipeline/store.py).

Real chromadb in a tmp directory, fake fixed-dim embeddings, no network, no
model. Covers delete-then-insert dedup (Spec req. 17 / acceptance criterion 5),
provenance metadata persistence into Chroma, collection naming (model+dim),
begin/complete crash markers, and guard rails.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fetchers.logger import FetchLogger
from pipeline.chunk import chunk_result
from pipeline.store import VectorStore, chunk_metadata, collection_name_for
from schemas.extraction import ContentType, ExtractionResult, Section

TS = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
DIM = 4


def _chunks(url: str, sections: list[str], *, title: str = "Page", truncated: bool = False):
    result = ExtractionResult(
        source_url=url,
        scrape_timestamp=TS,
        page_title=title,
        content_type=ContentType.HTML,
        sections=[Section(heading=f"H{i}", content=s) for i, s in enumerate(sections)],
        confidence=0.8,
        truncated=truncated,
    )
    return chunk_result(result, max_chunk_chars=2000)


def _vec(seed: float) -> list[float]:
    return [seed + i for i in range(DIM)]


def _store(vs, chunks, *, logger=None) -> str:
    name = collection_name_for("BAAI/bge-m3", DIM)
    vs.store_chunks(chunks, [_vec(i) for i in range(len(chunks))], collection_name=name, logger=logger)
    return name


@pytest.fixture()
def vs(tmp_path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


def test_store_then_query_returns_chunk_with_full_provenance_metadata(vs):
    name = _store(vs, _chunks(URL_A, ["hello world", "second chunk"]))
    assert vs.count(collection_name=name) == 2
    rows = vs.query(_vec(0), k=5, collection_name=name, where={"source_url": URL_A})
    assert len(rows) == 2
    meta = rows[0]["metadata"]
    for key in (
        "source_url",
        "scrape_timestamp",
        "ingest_timestamp",
        "content_type",
        "confidence",
        "chunk_index",
        "chunk_total",
        "truncated",
        "page_title",
        "section_heading",
    ):
        assert key in meta, f"missing metadata key {key}"
    assert meta["source_url"] == URL_A
    assert meta["scrape_timestamp"] == TS.isoformat()
    assert meta["section_heading"] == "H0"


def test_metadata_drops_none_fields():
    chunk = _chunks(URL_A, ["x"])[0]
    meta = chunk_metadata(chunk)
    for key in ("page_title", "section_heading", "section_level", "extraction_notes"):
        if getattr(chunk, key) is None:
            assert key not in meta


def test_reingest_same_url_replaces_not_duplicates(vs):
    name = _store(vs, _chunks(URL_A, ["one", "two", "three"]))
    assert vs.count(collection_name=name, where={"source_url": URL_A}) == 3
    _store(vs, _chunks(URL_A, ["only one now"]))
    assert vs.count(collection_name=name) == 1
    assert vs.count(collection_name=name, where={"source_url": URL_A}) == 1
    rows = vs.query(_vec(0), k=5, collection_name=name, where={"source_url": URL_A})
    assert rows[0]["document"] == "only one now"


def test_different_urls_coexist_and_delete_is_scoped(vs):
    name = _store(vs, _chunks(URL_A, ["a1", "a2"]))
    _store(vs, _chunks(URL_B, ["b1"]))
    assert vs.count(collection_name=name) == 3
    vs.delete_url(URL_A, collection_name=name)
    assert vs.count(collection_name=name) == 1
    assert vs.count(collection_name=name, where={"source_url": URL_B}) == 1


def test_chunk_ids_reused_across_reingestions(vs):
    name = _store(vs, _chunks(URL_A, ["stable content"]))
    first_ids = {r["id"] for r in vs.query(_vec(0), k=5, collection_name=name)}
    _store(vs, _chunks(URL_A, ["stable content"]))
    second_ids = {r["id"] for r in vs.query(_vec(0), k=5, collection_name=name)}
    assert first_ids == second_ids


def test_begin_complete_markers_written(tmp_path):
    vs = VectorStore(tmp_path / "chroma")
    with FetchLogger(tmp_path / "logs.db") as logger:
        _store(vs, _chunks(URL_A, ["x", "y", "z"]), logger=logger)
        events = logger.recent_events()
    types = [e["event_type"] for e in events]
    assert types.count("chunk_ingest_begin") == 1
    assert types.count("chunk_ingest_complete") == 1
    begin = next(e for e in events if e["event_type"] == "chunk_ingest_begin")
    assert begin["url"] == URL_A


def test_collection_name_encodes_model_and_dim():
    assert collection_name_for("BAAI/bge-m3", 1024) == "corpus_BAAI_bge_m3_1024"
    assert collection_name_for("all-MiniLM-L6-v2", 384) == "corpus_all_MiniLM_L6_v2_384"
    assert collection_name_for("BAAI/bge-m3", 384) != collection_name_for("all-MiniLM-L6-v2", 384)
    with pytest.raises(ValueError):
        collection_name_for("a" * 200, 384)


def test_mismatched_chunk_embedding_counts_raise(vs):
    chunks = _chunks(URL_A, ["one", "two"])
    with pytest.raises(ValueError, match="count mismatch"):
        vs.store_chunks(chunks, [_vec(0)], collection_name="corpus_BAAI_bge_m3_4")


def test_empty_chunk_list_raises(vs):
    with pytest.raises(ValueError, match="no chunks to store"):
        vs.store_chunks([], [], collection_name="corpus_BAAI_bge_m3_4")


def test_query_on_missing_collection_returns_empty(vs):
    assert vs.query(_vec(0), k=5, collection_name="corpus_none_4") == []
    assert vs.count(collection_name="corpus_none_4") == 0


def test_get_returns_all_chunks_for_url_without_cap(vs):
    """Whole-URL retrieval: every chunk for the URL, no evidence cap.

    Six chunks exceed any query() default k — the no-silent-truncation rule
    must hold for get() regardless of fixture size.
    """
    sections = [f"section {i} content" for i in range(6)]
    name = _store(vs, _chunks(URL_A, sections))
    rows = vs.get(collection_name=name, where={"source_url": URL_A})
    assert len(rows) == 6
    assert all(r["metadata"]["chunk_index"] == 1 for r in rows)  # 1-based, per-section
    assert all(r["metadata"]["chunk_total"] == 1 for r in rows)
    assert all(r["metadata"]["source_url"] == URL_A for r in rows)
    assert {r["document"] for r in rows} == set(sections)
    assert rows[0]["metadata"]["scrape_timestamp"] == TS.isoformat()


def test_get_no_match_returns_empty(vs):
    name = _store(vs, _chunks(URL_A, ["one"]))
    assert vs.get(collection_name=name, where={"source_url": "https://other/x"}) == []
    assert vs.get(collection_name="corpus_none_4", where={"source_url": URL_A}) == []


def test_get_requires_where(vs):
    name = _store(vs, _chunks(URL_A, ["one"]))
    with pytest.raises(ValueError, match="where is required"):
        vs.get(collection_name=name)
