"""page_key() — CONTRACTS.md §3. THE one implementation; all apps import this.

Banks are looked up by page type, never by exact DOM hash:
    news.ycombinator.com/ or /news?p=N   -> "hn:front"
    news.ycombinator.com/item?id=*       -> "hn:item"
    localhost:*/popup* or file://*popup* -> "popup:demo"
    anything else                        -> "unknown"  (=> no bank; plain-prompt fallback)
"""

from urllib.parse import parse_qs, urlsplit


def page_key(url: str) -> str:
    raw = url.strip()
    parts = urlsplit(raw if "://" in raw else f"https://{raw}")
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    path = parts.path or "/"

    if scheme == "file":
        return "popup:demo" if "popup" in raw.lower() else "unknown"

    if host == "localhost":
        return "popup:demo" if "popup" in path.lower() else "unknown"

    if host == "news.ycombinator.com":
        if path in ("", "/", "/news"):
            return "hn:front"
        if path == "/item" and parse_qs(parts.query).get("id"):
            return "hn:item"

    return "unknown"
