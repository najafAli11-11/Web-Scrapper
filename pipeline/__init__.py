"""Pipeline stages: stripping, chunking, embedding, storage."""

from pipeline.strip import StripBlock, StripOutcome, StripResult, strip_html

__all__ = ["strip_html", "StripResult", "StripBlock", "StripOutcome"]
