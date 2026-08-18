"""Per-domain rate limiter (Spec req. 3).

Deterministic layer — plain time arithmetic, no LLM. Enforces a minimum
interval between requests to the same domain so the scraper does not
overwhelm target servers. Configurable via `config/fetch.json`.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from fetchers.logger import FetchLogger


class RateLimiter:
    def __init__(self, min_interval_seconds: float = 1.0):
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _domain(url: str) -> str:
        return urlparse(url).netloc or url

    def wait(self, url: str, *, logger: Optional[FetchLogger] = None) -> None:
        """Block until the minimum interval since the last request to this
        domain has elapsed."""
        if self.min_interval_seconds <= 0:
            return
        domain = self._domain(url)
        with self._lock:
            last = self._last_request.get(domain, 0.0)
            elapsed = time.monotonic() - last
            delay = self.min_interval_seconds - elapsed
            if delay > 0:
                if logger is not None:
                    logger.log_event(
                        "rate_limit_wait",
                        url=url,
                        outcome="wait",
                        reason=f"rate limiter sleeping {delay:.2f}s for domain {domain}",
                        details={"delay_seconds": round(delay, 2), "domain": domain},
                    )
                time.sleep(delay)
            self._last_request[domain] = time.monotonic()
