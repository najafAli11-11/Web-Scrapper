"""Pipeline stages: stripping, chunking, embedding, storage."""

from pipeline.chunk import chunk_result, make_chunk_id
from pipeline.strip import StripBlock, StripOutcome, StripResult, strip_html

__all__ = [
    "strip_html",
    "StripResult",
    "StripBlock",
    "StripOutcome",
    "chunk_result",
    "make_chunk_id",
]
