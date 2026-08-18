"""Deterministic content heuristics shared by the fetcher decision logic.

No LLM and no per-site assumptions here (Rule 1): these are plain string
rules used to decide whether fetched content is a JS shell / insufficient.
"""

from __future__ import annotations

import html as _html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style.*?</style>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s")


def visible_text_length(raw_html: str) -> int:
    """Count non-whitespace characters of visible text in raw HTML.

    Scripts and styles are stripped first so a JS-shell page (lots of script,
    no content) scores near zero.
    """
    if not raw_html:
        return 0
    stripped = _SCRIPT_RE.sub(" ", raw_html)
    stripped = _STYLE_RE.sub(" ", stripped)
    stripped = _TAG_RE.sub(" ", stripped)
    text = _html.unescape(stripped)
    return len(_WS_RE.sub("", text))


def _is_known_error_page(raw_html: str) -> bool:
    """True if the HTML is a known error/verification boilerplate page
    that is never real content — JavaScript disabled, robot check, etc."""
    lower = raw_html.lower()
    error_phrases = [
        "javascript is disabled",
        "robot check",
        "verify you're not a robot",
        "access denied",
        "cloudflare",
    ]
    return any(phrase in lower for phrase in error_phrases)


def looks_empty(raw_html: str, threshold_chars: int = 200) -> bool:
    """True when fetched HTML has too little visible text to be meaningful
    (empty body, JS shell that renders nothing server-side).

    Checks three conditions:
      1. Absolute: visible text < threshold_chars
      2. Ratio: visible text < 1% of total HTML size (JS shell detection)
      3. Known error boilerplate: page is a verification prompt
          (JavaScript disabled, robot check, etc.) — never real content
    A large HTML page with very little visible text is almost certainly a
    client-side rendered SPA that needs browser rendering.
    """
    if not raw_html:
        return True
    if _is_known_error_page(raw_html):
        return True
    visible = visible_text_length(raw_html)
    if visible < threshold_chars:
        return True
    total = len(raw_html)
    if total > 5000 and visible / total < 0.01:
        return True
    # Detect error/verification boilerplate that passes absolute/ratio checks
    # but is not real content — e.g. "JavaScript is disabled", "robot check"
    lower = raw_html.lower()
    error_phrases = [
        "javascript is disabled",
        "robot check",
        "verify you're not a robot",
        "access denied",
        "cloudflare",
    ]
    found = sum(1 for phrase in error_phrases if phrase in lower)
    if found >= 2 and visible < total * 0.05:
        return True
    return False


def is_html_like(content_type: Optional[str]) -> bool:
    """True if a Content-Type suggests HTML/XML that the static fetcher can
    decode into text. Non-HTML types (PDF, images, octet-stream) return False
    so they are not sent down the browser-fallback path."""
    if not content_type:
        return True
    ct = content_type.lower()
    return ct.startswith("text/") or "html" in ct or "xml" in ct
