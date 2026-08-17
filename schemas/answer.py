"""Chat answer schema (M9) — traceable citations enforced by the schema.

Spec req. 16 requires the chatbot's answer to include traceable source URL(s).
min_length=1 on citations makes that a schema-level requirement, not a
prompt-level nicety: an Answer without at least one citation is invalid and
triggers the answer_generation_failed fallback path (agents/answer.py).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    source_url: str = Field(min_length=1)
    scrape_timestamp: str = ""
    page_title: Optional[str] = None
    section_heading: Optional[str] = None
    quote: str = Field(min_length=1)


class Answer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v: Any) -> list:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return v
