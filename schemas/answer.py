"""Chat answer schema (M9) — traceable citations enforced by the schema.

Spec req. 16 requires the chatbot's answer to include traceable source URL(s).
min_length=1 on citations makes that a schema-level requirement, not a
prompt-level nicety: an Answer without at least one citation is invalid and
triggers the answer_generation_failed fallback path (agents/answer.py).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source_url: str = Field(min_length=1)
    scrape_timestamp: str = Field(min_length=1)
    page_title: Optional[str] = None
    section_heading: Optional[str] = None
    quote: str = Field(min_length=1)


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)
