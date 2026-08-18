"""Fetch decision logic (Spec req. 1): static first, browser fallback.

- static BLOCKED (hard 403 / challenge) -> return BLOCKED, no fallback (Rule 7)
- static SUCCESS with meaningful HTML      -> return static
- static SUCCESS with non-HTML content     -> return static (routed later)
- static SUCCESS but empty / JS shell      -> browser fallback
- static FAILED                            -> browser fallback once

Every static and browser attempt is its own logged `fetch_attempt`; the
fallback decision itself is logged as a `fetch_decision` event.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from fetchers.browser_fetch import fetch_browser
from fetchers.config_loader import load_fetch_config, load_obstacle_config
from fetchers.content_heuristics import is_html_like, looks_empty, visible_text_length
from fetchers.logger import FetchLogger
from fetchers.rate_limit import RateLimiter
from fetchers.static_fetch import fetch_static
from fetchers.types import FetchOutcome, FetchResult


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def fetch_page(
    url: str,
    *,
    obstacle_cfg: Optional[dict] = None,
    fetch_cfg: Optional[dict] = None,
    logger: Optional[FetchLogger] = None,
    rate_limiter: Optional[RateLimiter] = None,
    browser=None,
) -> FetchResult:
    obstacle_cfg = obstacle_cfg if obstacle_cfg is not None else load_obstacle_config()
    fetch_cfg = fetch_cfg if fetch_cfg is not None else load_fetch_config()
    logger = logger if logger is not None else FetchLogger()
    rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter(fetch_cfg["rate_limit_min_interval_seconds"])

    static = fetch_static(url, obstacle_cfg=obstacle_cfg, fetch_cfg=fetch_cfg, logger=logger, rate_limiter=rate_limiter)

    if static.outcome == FetchOutcome.BLOCKED:
        logger.log_event(
            "fetch_decision",
            url=url,
            outcome="no_fallback",
            reason="blocked by site (Rule 7), no browser fallback",
            details={"static_status": static.status_code},
        )
        return static

    if static.outcome == FetchOutcome.SUCCESS:
        if is_html_like(static.content_type or ""):
            if static.html is None or looks_empty(static.html, float(fetch_cfg["empty_content_threshold_chars"])):
                logger.log_event(
                    "fetch_decision",
                    url=url,
                    outcome="fallback",
                    reason="static content insufficient, falling back to browser fetch",
                    details={"static_status": static.status_code, "content_type": static.content_type},
                )
                return fetch_browser(url, obstacle_cfg=obstacle_cfg, fetch_cfg=fetch_cfg, logger=logger, rate_limiter=rate_limiter, browser=browser)
            logger.log_event(
                "fetch_decision",
                url=url,
                outcome="kept_static",
                reason="static fetch sufficient",
                details={"static_status": static.status_code, "content_type": static.content_type},
            )
            return static
        logger.log_event(
            "fetch_decision",
            url=url,
            outcome="kept_static",
            reason="non-HTML content type, static only",
            details={"static_status": static.status_code, "content_type": static.content_type},
        )
        return static

    logger.log_event(
        "fetch_decision",
        url=url,
        outcome="fallback",
        reason=f"static fetch failed ({static.reason}), falling back to browser fetch",
        details={"static_status": static.status_code},
    )
    return fetch_browser(url, obstacle_cfg=obstacle_cfg, fetch_cfg=fetch_cfg, logger=logger, rate_limiter=rate_limiter, browser=browser)


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Fetch a URL (static first, browser fallback) and record the attempt in the event log."
    )
    parser.add_argument("url")
    parser.add_argument("--db", default=None, help="SQLite log DB path (default: data/logs.db)")
    args = parser.parse_args(argv)

    with FetchLogger(args.db) as logger:
        result = fetch_page(args.url, logger=logger)

    print(
        f"url={result.url}\n"
        f"outcome={result.outcome.value}\n"
        f"fetcher={result.fetcher}\n"
        f"status={result.status_code}\n"
        f"final_url={result.final_url}\n"
        f"reason={result.reason}\n"
        f"content_type={result.content_type}\n"
        f"html_chars={visible_text_length(result.html) if result.html else 0}\n"
        f"raw_bytes={len(result.raw) if result.raw else 0}"
    )
    print("--- log rows (chronological) ---")
    with FetchLogger(args.db) as log:
        for row in log.rows_for_url(result.url):
            print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
