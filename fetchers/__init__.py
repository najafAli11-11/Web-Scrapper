"""Fetcher layer: static HTTP + headless browser fetch with obstacle handling."""

from fetchers.fetch import fetch_page
from fetchers.types import FetchOutcome, FetchResult

__all__ = ["fetch_page", "FetchOutcome", "FetchResult"]
