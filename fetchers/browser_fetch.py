"""Playwright-based headless browser fetch with config-driven obstacle handling.

Every enabled obstacle in `config/obstacles.json` is handled according to its
detection_method + policy. Detection is heuristic/role/text-based and generic
across sites (never per-site CSS selectors, per AGENTS.md).

Terminal obstacles (captcha_gate / session_expiry / random_logout) are a hard
stop: detect, log with reason, return BLOCKED — never solved or worked around
(Rule 7). `blocked_clicks` that exhaust their retry budget without a detected
challenge resolve to FAILED (no infinite retry, no silent success).
"""

from __future__ import annotations

import random
import re
import time
from typing import Optional

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from fetchers.config_loader import (
    is_enabled,
    load_fetch_config,
    load_obstacle_config,
    max_retries_for,
)
from fetchers.content_heuristics import looks_empty
from fetchers.logger import FetchLogger
from fetchers.rate_limit import RateLimiter
from fetchers.types import FetchOutcome, FetchResult

_CHALLENGE_FRAME_KEYWORDS = ("recaptcha", "hcaptcha", "captcha", "turnstile", "challenge")
_CHALLENGE_TEXT_KEYWORDS = (
    "verify you are human",
    "verify you're human",
    "unusual traffic",
    "just a moment",
    "checking your browser",
    "enter the captcha",
    "i'm not a robot",
    "im not a robot",
)
_AUTH_URL_RE = re.compile(r"/(login|signin|sign-in|logon)([/?#]|$)", re.IGNORECASE)
_AUTH_TEXT_KEYWORDS = ("sign in", "log in", "enter your password")

_CONSENT_KEYWORDS = ("cookie", "consent", "gdpr", "privacy policy", "we use cookies")
_CONSENT_REJECT = ("reject", "decline", "deny", "necessary only")
_CONSENT_ACCEPT = ("accept all", "accept", "agree", "allow all", "allow")
_CLOSE_TEXT = ("close", "dismiss", "ok", "×", "x", "✕", "✖")
_LOAD_MORE_RE = re.compile(r"load more|show more|view more|see more|more results", re.IGNORECASE)


class _FetchFailed(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _TerminalBlock(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _ClickNoOp(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _result(
    url: str,
    outcome: FetchOutcome,
    reason: Optional[str] = None,
    *,
    fetcher: str,
    html: Optional[str] = None,
    status_code: Optional[int] = None,
    content_type: Optional[str] = None,
    final_url: Optional[str] = None,
) -> FetchResult:
    return FetchResult(
        url=url,
        outcome=outcome,
        html=html,
        status_code=status_code,
        content_type=content_type,
        final_url=final_url,
        fetcher=fetcher,  # type: ignore[arg-type]
        reason=reason,
    )


def _log_obstacle(logger: FetchLogger, obstacle_cfg: dict, url: str, obstacle: str, reason: str, details: Optional[dict] = None) -> None:
    entry = obstacle_cfg.get(obstacle, {})
    d = dict(details or {})
    d["obstacle"] = obstacle
    d["policy"] = entry.get("policy", "")
    d["detection_method"] = entry.get("detection_method", "")
    logger.log_event("obstacle_detected", url=url, outcome=d["policy"], reason=reason, details=d)


def _backoff_seconds(attempt: int, base: float, retry_after: Optional[str]) -> float:
    if retry_after:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass
    return min(30.0, base * (2 ** attempt)) + random.uniform(0, 0.5)


def _log_retry(logger: FetchLogger, obstacle_cfg: dict, url: str, obstacle: str, attempt: int, max_retries: int, base_backoff: float, retry_after: Optional[str] = None) -> float:
    delay = _backoff_seconds(attempt, base_backoff, retry_after)
    _log_obstacle(
        logger,
        obstacle_cfg,
        url,
        obstacle,
        f"{obstacle}: attempt {attempt + 1}/{max_retries + 1}, backing off {delay:.1f}s",
        {"attempt": attempt + 1, "max_retries": max_retries},
    )
    return delay


def _visible_len(page) -> int:
    try:
        return len(re.sub(r"\s", "", page.locator("body").inner_text(timeout=3000)))
    except PlaywrightError:
        return 0


def _terminal_block_present(page, obstacle_cfg: dict) -> Optional[tuple[str, str]]:
    """Detect terminal access-control obstacles. Returns (obstacle, reason) or None."""
    if is_enabled(obstacle_cfg, "captcha_gate"):
        for frame in page.frames:
            if frame is page.main_frame:
                continue
            frame_url = (frame.url or "").lower()
            if any(k in frame_url for k in _CHALLENGE_FRAME_KEYWORDS):
                return ("captcha_gate", f"challenge widget frame detected: {frame.url}")
        try:
            text = page.locator("body").inner_text(timeout=4000).lower()
        except PlaywrightError:
            text = ""
        if any(k in text for k in _CHALLENGE_TEXT_KEYWORDS):
            return ("captcha_gate", "challenge text detected on page")
    if is_enabled(obstacle_cfg, "session_expiry") or is_enabled(obstacle_cfg, "random_logout"):
        auth_block = _detect_auth_block(page, obstacle_cfg)
        if auth_block:
            return auth_block
    return None


def _detect_auth_block(page, obstacle_cfg: dict) -> Optional[tuple[str, str]]:
    obstacle = "random_logout" if is_enabled(obstacle_cfg, "random_logout") else "session_expiry"
    current = page.url or ""
    if _AUTH_URL_RE.search(current):
        return (obstacle, f"redirected to auth page: {current}")
    try:
        if page.locator("input[type=password]").count() > 0:
            return (obstacle, "password field present (auth-gated content)")
    except PlaywrightError:
        pass
    try:
        text = page.locator("body").inner_text(timeout=3000).lower()
    except PlaywrightError:
        text = ""
    if any(k in text for k in _AUTH_TEXT_KEYWORDS):
        return (obstacle, "auth challenge text detected")
    return None


def _find_consent_banner(page):
    for sel in ("[role=dialog]", "dialog", "[aria-modal=true]"):
        try:
            loc = page.locator(sel)
            n = min(loc.count(), 5)
        except PlaywrightError:
            continue
        for i in range(n):
            el = loc.nth(i)
            try:
                if not el.is_visible():
                    continue
                text = el.inner_text(timeout=2000).lower()
            except PlaywrightError:
                continue
            if any(k in text for k in _CONSENT_KEYWORDS):
                return el
    return None


def _find_popup(page):
    for sel in ("[aria-modal=true]", "[role=dialog]", "dialog"):
        try:
            loc = page.locator(sel)
            n = min(loc.count(), 5)
        except PlaywrightError:
            continue
        for i in range(n):
            el = loc.nth(i)
            try:
                if el.is_visible():
                    return el
            except PlaywrightError:
                continue
    return None


def _choose_dismiss_button(banner, kind: str):
    try:
        buttons = banner.locator("button")
        n = min(buttons.count(), 12)
    except PlaywrightError:
        return None
    for i in range(n):
        el = buttons.nth(i)
        try:
            text = (el.inner_text(timeout=1500) or "").strip().lower()
            aria = (el.get_attribute("aria-label") or "").lower()
        except PlaywrightError:
            continue
        label = f"{text} {aria}".strip()
        if not label:
            continue
        if kind == "cookie_banner":
            if any(p in label for p in _CONSENT_REJECT):
                return el
            if any(p in label for p in _CONSENT_ACCEPT):
                return el
        if any(c in label for c in _CLOSE_TEXT) or label.strip() in ("×", "x", "✕", "✖"):
            return el
    return None


def _safe_click(page, locator, obstacle_cfg: dict, fetch_cfg: dict, logger: FetchLogger, url: str, kind: str, change_threshold: int = 10, gone_locator=None) -> None:
    """Click with blocked_clicks no-op detection, dom_drift/stale requery,
    and terminal-block escalation. No-op exhaustion -> _ClickNoOp -> FAILED.
    A challenge appearing after the click escalates to _TerminalBlock -> BLOCKED.
    """
    max_retries = max_retries_for(obstacle_cfg, "blocked_clicks")
    base_backoff = float(fetch_cfg["backoff_base_seconds"])
    base_url = page.url
    base_len = _visible_len(page)

    for attempt in range(max_retries + 1):
        try:
            locator.click(timeout=5000)
        except PlaywrightTimeoutError:
            _log_obstacle(logger, obstacle_cfg, url, "dom_drift", "element not actionable, re-querying")
            if attempt >= max_retries:
                raise _ClickNoOp(f"{kind}: element not actionable after {max_retries + 1} attempts (dom_drift/requery)")
            time.sleep(_backoff_seconds(attempt, base_backoff, None))
            continue
        except PlaywrightError as exc:
            msg = (exc.message or "").lower()
            _log_obstacle(
                logger, obstacle_cfg, url, "stale_element",
                f"stale element on {kind}: {exc.message}",
                {"error": exc.message, "attempt": attempt + 1},
            )
            if attempt >= max_retries:
                raise _ClickNoOp(f"{kind}: element not actionable after {max_retries + 1} attempts (stale_element)")
            time.sleep(_backoff_seconds(attempt, base_backoff, None))
            continue

        page.wait_for_timeout(700)

        terminal = _terminal_block_present(page, obstacle_cfg)
        if terminal:
            name, reason = terminal
            _log_obstacle(logger, obstacle_cfg, url, name, reason)
            raise _TerminalBlock(reason)

        changed = page.url != base_url or (_visible_len(page) - base_len) > change_threshold
        if gone_locator is not None:
            try:
                gone = not gone_locator.is_visible() or gone_locator.count() == 0
            except PlaywrightError:
                gone = True
            changed = changed or gone
        if changed:
            return

        _log_obstacle(
            logger,
            obstacle_cfg,
            url,
            "blocked_clicks",
            f"click no-op (attempt {attempt + 1}/{max_retries + 1}): nothing changed",
            {"attempt": attempt + 1, "max_retries": max_retries},
        )
        if attempt >= max_retries:
            raise _ClickNoOp(f"{kind}: click no-op after {max_retries + 1} attempts")
        time.sleep(_backoff_seconds(attempt, base_backoff, None))

    raise _ClickNoOp(f"{kind}: click no-op after {max_retries + 1} attempts")


def _dismiss_banner(page, banner, obstacle_cfg: dict, fetch_cfg: dict, logger: FetchLogger, url: str, kind: str) -> None:
    target = _choose_dismiss_button(banner, kind)
    if target is None:
        raise _ClickNoOp(f"{kind}: no dismiss/accept button found on banner")
    _safe_click(page, target, obstacle_cfg, fetch_cfg, logger, url, kind, change_threshold=50, gone_locator=banner)


def _dismiss_cookie_banner(page, obstacle_cfg: dict, fetch_cfg: dict, logger: FetchLogger, url: str) -> None:
    banner = _find_consent_banner(page)
    if banner is None:
        return
    _log_obstacle(logger, obstacle_cfg, url, "cookie_banner", "cookie banner detected, dismissing")
    _dismiss_banner(page, banner, obstacle_cfg, fetch_cfg, logger, url, "cookie_banner")


def _dismiss_popup(page, obstacle_cfg: dict, fetch_cfg: dict, logger: FetchLogger, url: str) -> None:
    popup = _find_popup(page)
    if popup is None:
        return
    _log_obstacle(logger, obstacle_cfg, url, "popup_modal", "popup/modal overlay detected, dismissing")
    _dismiss_banner(page, popup, obstacle_cfg, fetch_cfg, logger, url, "popup_modal")


def _find_load_more(page):
    try:
        buttons = page.locator("button")
        n = min(buttons.count(), 20)
    except PlaywrightError:
        return None
    for i in range(n):
        el = buttons.nth(i)
        try:
            label = el.inner_text(timeout=1500).strip()
            if label and _LOAD_MORE_RE.search(label) and el.is_visible():
                return el
        except PlaywrightError:
            continue
    return None


def _paginate(page, obstacle_cfg: dict, fetch_cfg: dict, logger: FetchLogger, url: str) -> None:
    max_clicks = int(fetch_cfg.get("pagination_max_clicks", 5))
    for i in range(max_clicks):
        btn = _find_load_more(page)
        if btn is None:
            return
        before = _visible_len(page)
        _safe_click(page, btn, obstacle_cfg, fetch_cfg, logger, url, "load_more", change_threshold=10)
        after = _visible_len(page)
        if after <= before + 10:
            return
        _log_obstacle(
            logger,
            obstacle_cfg,
            url,
            "load_more",
            f"loaded more content (click {i + 1}), +{after - before} chars",
            {"click": i + 1},
        )


def _goto_with_retries(page, url: str, obstacle_cfg: dict, fetch_cfg: dict, logger: FetchLogger, timeout_ms: int):
    max_429 = max_retries_for(obstacle_cfg, "rate_limit")
    max_5xx = max_retries_for(obstacle_cfg, "server_errors")
    base_backoff = float(fetch_cfg["backoff_base_seconds"])
    attempt = 0
    extended = False
    while True:
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except PlaywrightTimeoutError:
            if not extended:
                _log_obstacle(logger, obstacle_cfg, url, "slow_responses", "navigation timed out, extending timeout")
                timeout_ms = int(float(fetch_cfg["extended_timeout_seconds"]) * 1000)
                extended = True
                continue
            raise _FetchFailed("slow_responses: navigation timed out after extended timeout")
        except PlaywrightError as exc:
            raise _FetchFailed(f"navigation error: {exc}")
        if response is None:
            raise _FetchFailed("navigation returned no response")
        status = response.status
        if status == 429:
            retry_after = (response.headers or {}).get("retry-after")
            if attempt < max_429:
                delay = _log_retry(logger, obstacle_cfg, url, "rate_limit", attempt, max_429, base_backoff, retry_after)
                time.sleep(delay)
                attempt += 1
                continue
            raise _FetchFailed(f"rate_limit: exhausted {max_429 + 1} retries")
        if status and 500 <= status < 600:
            if attempt < max_5xx:
                delay = _log_retry(logger, obstacle_cfg, url, "server_errors", attempt, max_5xx, base_backoff, None)
                time.sleep(delay)
                attempt += 1
                continue
            raise _FetchFailed(f"server_errors: exhausted {max_5xx + 1} retries (last status {status})")
        return response


def _log_redirect_hops(request, logger: FetchLogger, url: str) -> None:
    if request.redirected_from is not None:
        logger.log_event(
            "obstacle_detected",
            url=url,
            outcome="follow_and_log",
            reason=f"redirect: {request.redirected_from.url} -> {request.url}",
            details={
                "obstacle": "unexpected_redirect",
                "policy": "follow_and_log",
                "detection_method": "redirect_chain_detected",
            },
        )


def _settle(page, fetch_cfg: dict) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except PlaywrightError:
        pass
    page.wait_for_timeout(int(float(fetch_cfg["browser_settle_seconds"]) * 1000))


def fetch_browser(
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

    owns_browser = browser is None
    pw = sync_playwright().start()
    browser_instance = browser
    try:
        if browser_instance is None:
            browser_instance = pw.chromium.launch(headless=True)
        context = browser_instance.new_context(user_agent=str(fetch_cfg["user_agent"]))
        page = context.new_page()
        try:
            page.on("request", lambda r: _log_redirect_hops(r, logger, url))
            rate_limiter.wait(url)
            timeout_ms = int(float(fetch_cfg["timeout_seconds"]) * 1000)
            try:
                response = _goto_with_retries(page, url, obstacle_cfg, fetch_cfg, logger, timeout_ms)
            except _FetchFailed as exc:
                result = _result(url, FetchOutcome.FAILED, exc.reason, fetcher="browser")
                logger.log_fetch_attempt(result)
                return result

            _settle(page, fetch_cfg)

            try:
                terminal = _terminal_block_present(page, obstacle_cfg)
                if terminal:
                    name, reason = terminal
                    _log_obstacle(logger, obstacle_cfg, url, name, reason)
                    result = _result(url, FetchOutcome.BLOCKED, reason, fetcher="browser", status_code=response.status)
                    logger.log_fetch_attempt(result)
                    return result

                if is_enabled(obstacle_cfg, "cookie_banner"):
                    _dismiss_cookie_banner(page, obstacle_cfg, fetch_cfg, logger, url)
                if is_enabled(obstacle_cfg, "popup_modal"):
                    _dismiss_popup(page, obstacle_cfg, fetch_cfg, logger, url)

                terminal = _terminal_block_present(page, obstacle_cfg)
                if terminal:
                    name, reason = terminal
                    _log_obstacle(logger, obstacle_cfg, url, name, reason)
                    result = _result(url, FetchOutcome.BLOCKED, reason, fetcher="browser", status_code=response.status)
                    logger.log_fetch_attempt(result)
                    return result

                if is_enabled(obstacle_cfg, "load_more"):
                    _paginate(page, obstacle_cfg, fetch_cfg, logger, url)
            except _TerminalBlock as exc:
                result = _result(url, FetchOutcome.BLOCKED, exc.reason, fetcher="browser", status_code=response.status)
                logger.log_fetch_attempt(result)
                return result
            except _ClickNoOp as exc:
                result = _result(url, FetchOutcome.FAILED, exc.reason, fetcher="browser", status_code=response.status)
                logger.log_fetch_attempt(result)
                return result

            html = page.content()
            status = response.status if response is not None else None
            content_type = (response.headers or {}).get("content-type") or "text/html"
            final_url = page.url

            if looks_empty(html, float(fetch_cfg["empty_content_threshold_chars"])):
                result = _result(url, FetchOutcome.EMPTY, "browser rendered page is empty/JS shell", fetcher="browser", html=html, status_code=status, content_type=content_type, final_url=final_url)
                logger.log_fetch_attempt(result)
                return result

            result = _result(url, FetchOutcome.SUCCESS, None, fetcher="browser", html=html, status_code=status, content_type=content_type, final_url=final_url)
            logger.log_fetch_attempt(result)
            return result
        finally:
            context.close()
    finally:
        if owns_browser:
            try:
                if browser_instance is not None:
                    browser_instance.close()
            finally:
                pw.stop()
