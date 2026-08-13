"""Plain HTTP fetch (urllib, stdlib) — the first hop of the fetcher layer.

Maps HTTP reality to FetchOutcome:
  - 200 + text content   -> SUCCESS (raw HTML in `html`, or None for non-text)
  - 403                  -> BLOCKED (hard access refusal: no retry, no fallback — Rule 7)
  - 429 / 5xx            -> retry-with-backoff (bounded by config), then FAILED
  - other 4xx / transport-> FAILED
Redirects are followed and each hop logged as an obstacle event
(unexpected_redirect -> follow_and_log).
"""

from __future__ import annotations

import random
import socket
import time
import urllib.error
import urllib.request
from typing import Optional

from fetchers.config_loader import max_retries_for
from fetchers.content_heuristics import is_html_like
from fetchers.logger import FetchLogger
from fetchers.rate_limit import RateLimiter
from fetchers.types import FetchOutcome, FetchResult


class _RedirectCapture(urllib.request.HTTPRedirectHandler):
    """Records every redirect hop so `follow_and_log` has a chain to log."""

    def __init__(self, logger: FetchLogger, url: str, max_redirections: int):
        super().__init__()
        self.logger = logger
        self.url = url
        self.hops: list[tuple[str, str, int]] = []
        self.max_redirections = max_redirections

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.hops.append((req.full_url, newurl, code))
        self.logger.log_event(
            "obstacle_detected",
            url=self.url,
            outcome="follow_and_log",
            reason=f"redirect {code}: {req.full_url} -> {newurl}",
            details={
                "obstacle": "unexpected_redirect",
                "policy": "follow_and_log",
                "detection_method": "redirect_chain_detected",
            },
        )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _backoff_seconds(attempt: int, base: float, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    return min(30.0, base * (2 ** attempt)) + random.uniform(0, 0.5)


def _decode_body(body: bytes, charset: Optional[str]) -> Optional[str]:
    if charset:
        try:
            return body.decode(charset)
        except (LookupError, UnicodeDecodeError):
            pass
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        return body.decode("latin-1", errors="replace")


def fetch_static(
    url: str,
    *,
    obstacle_cfg: dict,
    fetch_cfg: dict,
    logger: FetchLogger,
    rate_limiter: RateLimiter,
) -> FetchResult:
    timeout = float(fetch_cfg["timeout_seconds"])
    max_redirects = int(fetch_cfg["max_redirects"])
    base_backoff = float(fetch_cfg["backoff_base_seconds"])
    user_agent = str(fetch_cfg["user_agent"])
    max_retries_429 = max_retries_for(obstacle_cfg, "rate_limit")
    max_retries_5xx = max_retries_for(obstacle_cfg, "server_errors")

    redirect_capture = _RedirectCapture(logger, url, max_redirects)
    opener = urllib.request.build_opener(redirect_capture)

    attempt = 0
    while True:
        rate_limiter.wait(url)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with opener.open(req, timeout=timeout) as resp:
                status = resp.getcode()
                final_url = resp.geturl()
                content_type = resp.headers.get_content_type() or ""
                body = resp.read()
                break
        except urllib.error.HTTPError as exc:
            status = exc.code
            final_url = exc.geturl() or url
            headers = exc.headers
            content_type = headers.get_content_type() if headers else ""
            retry_after = headers.get("Retry-After") if headers else None
            if status == 403:
                reason = "hard 403 access refusal (no retry, no fallback)"
                result = FetchResult(
                    url=url,
                    outcome=FetchOutcome.BLOCKED,
                    status_code=status,
                    content_type=content_type,
                    final_url=final_url,
                    fetcher="static",
                    reason=reason,
                )
                logger.log_fetch_attempt(result)
                return result
            if status == 429 and attempt < max_retries_429:
                delay = _backoff_seconds(attempt, base_backoff, retry_after)
                logger.log_event(
                    "obstacle_detected",
                    url=url,
                    outcome="retry_with_backoff",
                    reason=f"rate_limit: 429 (attempt {attempt + 1}/{max_retries_429 + 1}), "
                    f"backing off {delay:.1f}s",
                    details={
                        "obstacle": "rate_limit",
                        "policy": "retry_with_backoff",
                        "detection_method": "http_status_429",
                        "attempt": attempt + 1,
                    },
                )
                time.sleep(delay)
                attempt += 1
                continue
            if 500 <= status < 600 and attempt < max_retries_5xx:
                delay = _backoff_seconds(attempt, base_backoff, None)
                logger.log_event(
                    "obstacle_detected",
                    url=url,
                    outcome="retry_with_backoff",
                    reason=f"server_errors: {status} (attempt {attempt + 1}/{max_retries_5xx + 1}), "
                    f"backing off {delay:.1f}s",
                    details={
                        "obstacle": "server_errors",
                        "policy": "retry_with_backoff",
                        "detection_method": "http_status_5xx",
                        "attempt": attempt + 1,
                    },
                )
                time.sleep(delay)
                attempt += 1
                continue
            reason = "too many redirects" if "redirect" in str(exc.reason).lower() else f"http {status}"
            result = FetchResult(
                url=url,
                outcome=FetchOutcome.FAILED,
                status_code=status,
                content_type=content_type,
                final_url=final_url,
                fetcher="static",
                reason=reason,
            )
            logger.log_fetch_attempt(result)
            return result
        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            reason = f"network error: {exc}"
            result = FetchResult(
                url=url,
                outcome=FetchOutcome.FAILED,
                fetcher="static",
                reason=reason,
            )
            logger.log_fetch_attempt(result)
            return result

    if is_html_like(content_type):
        charset = resp.headers.get_content_charset()
        html = _decode_body(body, charset)
    else:
        html = None

    result = FetchResult(
        url=url,
        outcome=FetchOutcome.SUCCESS,
        html=html,
        status_code=status,
        content_type=content_type,
        final_url=final_url,
        fetcher="static",
    )
    logger.log_fetch_attempt(result)
    return result
