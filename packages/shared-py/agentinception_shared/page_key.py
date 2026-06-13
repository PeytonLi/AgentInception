"""Bank identity: page_key, not DOM hash. See CONTRACTS.md §3.

Banks are looked up by *page type*, not exact DOM hash, because HN comment
counts vary per article and exact structural hashes would never match.

Mapping (first match wins):
    news.ycombinator.com/         or /news?p=N   -> "hn:front"
    news.ycombinator.com/item?id=*               -> "hn:item"
    localhost:*/popup*  or  file://*popup*        -> "popup:demo"
    anything else                                 -> "unknown"
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

HN_HOST = "news.ycombinator.com"

# Front-page paths on Hacker News (with or without trailing slash).
_HN_FRONT_PATHS = {"", "/", "/news", "/newest", "/front"}


def page_key(url: str) -> str:
    """Map a URL to its page_key. Unknown pages return "unknown"."""
    if not url:
        return "unknown"

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""

    # --- Hacker News -----------------------------------------------------
    if host == HN_HOST or host.endswith("." + HN_HOST):
        norm_path = path.rstrip("/") if path != "/" else "/"
        if path == "/item" or norm_path == "/item":
            qs = parse_qs(parsed.query)
            if "id" in qs:
                return "hn:item"
            return "unknown"
        if norm_path in _HN_FRONT_PATHS:
            return "hn:front"
        return "unknown"

    # --- Popup fixture page ----------------------------------------------
    # localhost (any port) with "popup" in the path, OR a file:// URL whose
    # path references the popup fixture.
    # localhost (any port) that serves the popup fixture. The path is
    # irrelevant — the fixture lives at the server root.
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        return "popup:demo"

    # A file:// URL whose path references the popup fixture.
    if scheme == "file" and "popup" in path.lower():
        return "popup:demo"

    return "unknown"
