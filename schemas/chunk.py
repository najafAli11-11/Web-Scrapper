"""Chunk record schema (Spec req. 10-11) — the storage unit of the RAG corpus.

Every chunk carries full provenance (Rule 4): source URL, scrape timestamp,
page title, and section heading survive all the way to the vector store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.extraction import ContentType


class DocumentChunk(BaseModel):
    """One semantic chunk, produced from a single extraction section.

    chunk_id is deterministic (sha1 over source_url + section index + chunk
    index) so re-ingesting a URL produces identical ids for identical content,
    which is what makes delete-then-insert idempotent.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="Deterministic sha1(source_url|section_index|chunk_index)")
    source_url: str = Field(..., description="Provenance: source page URL (http/https only)")
    scrape_timestamp: datetime = Field(..., description="Provenance: UTC scrape timestamp of the source page")
    page_title: Optional[str] = Field(None, description="Provenance: page title")
    content_type: ContentType = Field(..., description="Routable content type of the source page")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence of the source record")
    section_heading: Optional[str] = Field(None, description="Provenance: heading of the source section")
    section_level: Optional[int] = Field(None, ge=1, description="Heading level of the source section")
    chunk_index: int = Field(..., ge=1, description="1-based position within this section's split")
    chunk_total: int = Field(..., ge=1, description="Total chunk count for this section's split")
    content: str = Field(..., min_length=1, description="Chunk text (non-empty)")
    truncated: bool = Field(False, description="True if the source extraction was incomplete (carried from ExtractionResult)")
    extraction_notes: Optional[str] = Field(None, description="Caveats from the extractor agent")
    ingest_timestamp: datetime = Field(..., description="UTC timestamp when this chunk was created/stored")

    @field_validator("source_url")
    @classmethod
    def _source_url_must_be_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be an absolute http(s) URL")
        return value

    @field_validator("scrape_timestamp", "ingest_timestamp")
    @classmethod
    def _timestamps_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware (UTC)")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _chunk_index_within_total(self) -> "DocumentChunk":
        if self.chunk_index > self.chunk_total:
            raise ValueError(f"chunk_index {self.chunk_index} exceeds chunk_total {self.chunk_total}")
        return self
