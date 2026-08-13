"""Shared contracts for the fetcher layer.

Every fetch attempt produces a FetchResult carrying outcome + provenance
(url, final_url, status, content_type) so callers and the event log never
lose context (Rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

FetcherName = Literal["static", "browser"]


class FetchOutcome(str, Enum):
    """Terminal outcome of a single fetch attempt.

    - SUCCESS: usable content retrieved (html, or a non-HTML content-type that
      later stages route/flag).
    - EMPTY: fetched but content looks like a JS shell / insufficient text.
    - BLOCKED: terminal access-control refusal (CAPTCHA, hard 403, auth).
      Never retried, never worked around (Rule 7).
    - FAILED: transport/HTTP error after bounded retries, or an interaction
      (blocked_clicks) that exhausted its retry budget.
    """

    SUCCESS = "success"
    EMPTY = "empty"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class FetchResult:
    url: str
    outcome: FetchOutcome
    html: Optional[str] = None
    raw: Optional[bytes] = None
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    final_url: Optional[str] = None
    fetcher: Optional[FetcherName] = None
    reason: Optional[str] = None
