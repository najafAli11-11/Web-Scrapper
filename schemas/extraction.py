"""
Generic extraction schema — per AGENTS.md Rule 2 (schema-constrained
extraction only) and the confirmed "generic, no per-domain schemas"
principle.

This schema is intentionally content-type-agnostic: it describes
"meaningful content of a page," not a product/article/forum-specific shape.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Section(BaseModel):
    """A structural unit of the page, used for semantic chunking (Rule: chunking)."""
    heading: Optional[str] = Field(None, description="Section heading, if present")
    content: str = Field(..., description="Text content of this section")
    level: Optional[int] = Field(None, description="Heading level (1=h1, 2=h2, etc.) if known")


class ExtractionResult(BaseModel):
    """Output contract for both batch and single-shot extraction modes (Spec req. 6-7)."""
    source_url: str
    scrape_timestamp: datetime
    page_title: Optional[str] = None
    content_type: str = Field(..., description="e.g. 'html', 'pdf', 'unknown'")
    sections: list[Section] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence, drives validator retry/flag logic")
    extraction_notes: Optional[str] = Field(None, description="Any caveats from the extractor agent")


class ValidationResult(BaseModel):
    """Output of the validator agent (Spec req. 9)."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    should_retry: bool = False
