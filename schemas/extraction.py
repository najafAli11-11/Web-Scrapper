"""
Generic extraction schema — per AGENTS.md Rule 2 (schema-constrained
extraction only) and the confirmed "generic, no per-domain schemas"
principle.

This schema is intentionally content-type-agnostic: it describes
"meaningful content of a page," not a product/article/forum-specific shape.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_CONFIDENCE_FOR_EMPTY_RESULT = 0.5
"""Empty-section results may not claim confidence above this cap (SPEC.md
empty/near-empty extraction edge case: flagged as low-confidence, never
silently stored as a near-empty chunk)."""


class ContentType(str, Enum):
    """Routable content types (Spec req. 8). `unknown` is valid but must be
    flagged, never silently dropped."""

    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    UNKNOWN = "unknown"


class Section(BaseModel):
    """A structural unit of the page, used for semantic chunking (Rule: chunking)."""

    model_config = ConfigDict(extra="forbid")

    heading: Optional[str] = Field(None, description="Section heading, if present")
    content: str = Field(..., description="Text content of this section")
    level: Optional[int] = Field(None, ge=1, description="Heading level (1=h1, 2=h2, etc.) if known")


class ExtractionResult(BaseModel):
    """Output contract for both batch and single-shot extraction modes (Spec req. 6-7)."""

    model_config = ConfigDict(extra="forbid")

    source_url: str = Field(..., description="Source URL of the page (http/https only)")
    scrape_timestamp: datetime = Field(..., description="UTC-aware scrape timestamp")
    page_title: Optional[str] = None
    content_type: ContentType = Field(..., description="Routable content type (Spec req. 8)")
    sections: list[Section] = Field(
        default_factory=list,
        description="Structural units surfaced during extraction, used for semantic chunking (Spec req. 10)",
    )

    @field_validator("sections", mode="before")
    @classmethod
    def _coerce_sections(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return v
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence, drives validator retry/flag logic"
    )
    extraction_notes: Optional[str] = Field(None, description="Any caveats from the extractor agent")
    truncated: bool = Field(False, description="True if extraction was incomplete (e.g. page too large)")

    @field_validator("source_url")
    @classmethod
    def _source_url_must_be_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be an absolute http(s) URL")
        return value

    @field_validator("scrape_timestamp")
    @classmethod
    def _scrape_timestamp_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("scrape_timestamp must be timezone-aware (UTC)")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _empty_sections_cannot_be_high_confidence(self) -> "ExtractionResult":
        if not self.sections and self.confidence > MAX_CONFIDENCE_FOR_EMPTY_RESULT:
            raise ValueError(
                f"empty sections cannot carry confidence > {MAX_CONFIDENCE_FOR_EMPTY_RESULT}"
            )
        return self


class ValidationResult(BaseModel):
    """Output of the validator agent (Spec req. 9).

    Provenance lives on the record itself so flagged records remain
    traceable (Rule 4: no exceptions).
    """

    model_config = ConfigDict(extra="forbid")

    source_url: str = Field(..., description="Source URL of the record being validated")
    scrape_timestamp: datetime = Field(..., description="UTC scrape timestamp of the record being validated")
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    should_retry: bool = Field(False, description="True if a repair attempt should be made")
    retry_count: int = Field(0, ge=0, description="Repair attempts already made (Spec req. 9: budget is 1, then flag)")

    @field_validator("source_url")
    @classmethod
    def _source_url_must_be_http(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("source_url must be an absolute http(s) URL")
        return value

    @field_validator("scrape_timestamp")
    @classmethod
    def _scrape_timestamp_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("scrape_timestamp must be timezone-aware (UTC)")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _retry_budget_respected(self) -> "ValidationResult":
        if self.should_retry and self.retry_count >= 1:
            raise ValueError("retry budget exhausted: retry_count >= 1, must flag instead of retry")
        return self
