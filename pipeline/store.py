"""Chroma vector store (Spec req. 13, 17) — dedup-by-source-url on write.

Dedup is delete-then-insert, not upsert-by-id: if a re-extraction yields
fewer chunks than the previous one, upsert would leave stale chunks behind.
Every ingestion deletes ALL prior chunks for the URL (where=source_url)
before inserting the new set in one add().

Crash protocol (deliberate v1 scope boundary — human-in-the-loop recovery):
chunk_ingest_begin / chunk_ingest_complete markers are written to the SQLite
events table around each URL's write. An interrupted write leaves old/empty/
partial state that is DETECTABLE (begin without complete, visible in the M9
logs view) rather than silent, and it self-heals on the next ingestion
because the delete-before-insert is total — no mixed old/new state survives
a completed re-ingestion. Automated recovery is out of scope for v1.

Collections are named corpus_<model-slug>_<dim> so both a dimension change
AND a same-dimension model swap create a distinct collection (no silent
cross-model sharing).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from fetchers.logger import FetchLogger
from schemas.chunk import DocumentChunk

_COLLECTION_NAME_OK = re.compile(r"^[A-Za-z0-9_]{3,128}$")


class _ExplicitEmbeddingsOnly:
    """Embedding function that must never be called — vectors are always supplied explicitly.

    Guards against a silent default-EF (ONNX) path if a caller forgets to pass embeddings.
    """

    is_legacy = True

    def __call__(self, input):  # pragma: no cover - guard, never expected to run
        raise RuntimeError("store: embeddings must be supplied explicitly, the embedding function is never used")

    def name(self) -> str:  # pragma: no cover - chroma inspects EF names
        return "explicit-embeddings-only"


def collection_name_for(model_name: str, dimension: int, prefix: str = "corpus") -> str:
    """corpus_<model-slug>_<dim> — encodes model AND dimension (plan decision)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_")
    name = f"{prefix}_{slug}_{dimension}"
    if not _COLLECTION_NAME_OK.match(name):
        raise ValueError(f"invalid chroma collection name: {name!r}")
    return name


def row_provenance(meta: dict) -> dict:
    """Provenance for a corpus chunk row (metadata keys are chroma-safe strings).

    Shared by the M8 live-query path and the M9 chat UI — one mapping, never
    re-implemented per caller.
    """
    prov: dict = {
        "source_url": meta.get("source_url"),
        "scrape_timestamp": meta.get("scrape_timestamp"),
    }
    for key in ("page_title", "section_heading"):
        if meta.get(key) is not None:
            prov[key] = meta[key]
    return prov


def corpus_collection(embedder, pipeline_cfg: dict) -> str:
    """Collection name for the corpus, derived from the injected embedder.

    Single source of truth shared by the M8 live-query path and the M9 chat
    UI. The name is ALWAYS derived from the embedder's model_name + dimension
    (never a hardcoded default) so an embedding-model config change cannot
    make one path query a different or nonexistent collection.
    """
    return collection_name_for(
        embedder.model_name,
        embedder.dimension,
        pipeline_cfg["store"]["collection_prefix"],
    )


def chunk_metadata(chunk: DocumentChunk) -> dict:
    """Chroma-safety: str/int/float/bool only; None fields are omitted."""
    meta: dict = {
        "source_url": chunk.source_url,
        "scrape_timestamp": chunk.scrape_timestamp.isoformat(),
        "ingest_timestamp": chunk.ingest_timestamp.isoformat(),
        "content_type": chunk.content_type.value,
        "confidence": chunk.confidence,
        "chunk_index": chunk.chunk_index,
        "chunk_total": chunk.chunk_total,
        "truncated": chunk.truncated,
    }
    if chunk.page_title is not None:
        meta["page_title"] = chunk.page_title
    if chunk.section_heading is not None:
        meta["section_heading"] = chunk.section_heading
    if chunk.section_level is not None:
        meta["section_level"] = chunk.section_level
    if chunk.extraction_notes is not None:
        meta["extraction_notes"] = chunk.extraction_notes
    return meta


class VectorStore:
    """Thin wrapper over a persistent Chroma client."""

    def __init__(self, chroma_path, *, collection_prefix: str = "corpus"):
        import chromadb

        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection_prefix = collection_prefix

    def get_or_create_collection(self, name: str):
        return self._client.get_or_create_collection(
            name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_ExplicitEmbeddingsOnly(),
        )

    def store_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        *,
        collection_name: str,
        logger: Optional[FetchLogger] = None,
    ) -> int:
        """Store chunks, replacing any prior chunks for the same source URL.

        Returns the number of chunks stored.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        if not chunks:
            raise ValueError("no chunks to store")

        by_url: dict[str, list[tuple[DocumentChunk, list[float]]]] = defaultdict(list)
        for chunk, emb in zip(chunks, embeddings):
            by_url[chunk.source_url].append((chunk, emb))

        collection = self.get_or_create_collection(collection_name)
        stored = 0
        for url, items in by_url.items():
            if logger is not None:
                logger.log_event(
                    event_type="chunk_ingest_begin",
                    url=url,
                    outcome="begin",
                    details={"planned_chunks": len(items), "collection": collection_name},
                )
            collection.delete(where={"source_url": url})
            collection.add(
                ids=[c.chunk_id for c, _ in items],
                embeddings=[e for _, e in items],
                documents=[c.content for c, _ in items],
                metadatas=[chunk_metadata(c) for c, _ in items],
            )
            stored += len(items)
            if logger is not None:
                logger.log_event(
                    event_type="chunk_ingest_complete",
                    url=url,
                    outcome="complete",
                    details={"chunk_count": len(items), "collection": collection_name},
                )
        return stored

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        *,
        collection_name: str,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Top-k nearest chunks. Returns [] for a missing/empty collection."""
        collection = self.get_or_create_collection(collection_name)
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict] = []
        for doc_id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            rows.append({"id": doc_id, "document": doc, "metadata": meta, "distance": dist})
        return rows

    def get(self, collection_name: str, *, where: Optional[dict] = None) -> list[dict]:
        """All chunks matching `where` — NO cap (M6 rule: never silently truncate).

        The reverse of query(): returns {id, document, metadata} for every
        matching chunk, regardless of how many there are. The live-query path
        (M8) uses this for whole-URL retrieval when no query text is given.

        `where` is required and guarded with ValueError — retrieving an entire
        collection is never a legitimate call for this codebase, so it must be
        a bug, not a silent giant dump. Returns [] for a missing/empty collection.
        """
        if where is None:
            raise ValueError("store.get: where is required (never retrieve a whole collection)")
        collection = self.get_or_create_collection(collection_name)
        if collection.count() == 0:
            return []
        result = collection.get(where=where, include=["documents", "metadatas"])
        rows: list[dict] = []
        for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            rows.append({"id": doc_id, "document": doc, "metadata": meta})
        return rows

    def count(self, *, collection_name: str, where: Optional[dict] = None) -> int:
        collection = self.get_or_create_collection(collection_name)
        if where is None:
            return collection.count()
        return len(collection.get(where=where)["ids"])

    def delete_url(self, url: str, *, collection_name: str) -> None:
        collection = self.get_or_create_collection(collection_name)
        collection.delete(where={"source_url": url})
