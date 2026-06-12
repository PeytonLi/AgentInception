"""DOM extraction: load a URL via Playwright (or read a local HTML file),
strip <script>/<style>/comments, return cleaned HTML + visible text + structural hash.

Playwright is imported lazily so unit tests run without it installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

from agentinception_shared.dom_hash import dom_structural_hash


@dataclass(frozen=True)
class DomExtract:
    url: str
    html: str        # cleaned HTML (scripts/styles/comments removed)
    text: str        # innerText-style visible text
    dom_structural_hash: str


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def strip_dom(html: str) -> tuple[str, str]:
    """Return (cleaned_html, visible_text). Removes scripts, styles, comments."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    cleaned_html = str(soup)
    text = soup.get_text(separator="\n", strip=True)
    return cleaned_html, text


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def extract_from_html(path: str) -> DomExtract:
    """Read a local .html file from disk and return a DomExtract."""
    abs_path = os.path.abspath(path)
    raw = Path(abs_path).read_text(encoding="utf-8")
    cleaned_html, text = strip_dom(raw)
    return DomExtract(
        url="file://" + abs_path,
        html=cleaned_html,
        text=text,
        dom_structural_hash=dom_structural_hash(cleaned_html),
    )


def extract_from_url(url: str, *, timeout_ms: int = 30000) -> DomExtract:
    """Load a URL via headless Chromium, strip scripts/styles, return DomExtract."""
    # Lazy import — Playwright is a heavy dep; tests don't need it.
    from playwright.sync_api import sync_playwright  # type: ignore

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36 AgentInceptionCompiler"
                )
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            raw_html = page.content()
        finally:
            browser.close()

    cleaned_html, text = strip_dom(raw_html)
    return DomExtract(
        url=url,
        html=cleaned_html,
        text=text,
        dom_structural_hash=dom_structural_hash(cleaned_html),
    )


def load_dom(*, url: str | None = None, html: str | None = None) -> DomExtract:
    """Unified entrypoint: one of url= or html= (a local file path)."""
    if (url is None) == (html is None):
        raise ValueError("Pass exactly one of url= or html= (local file path)")
    if html is not None:
        return extract_from_html(html)
    assert url is not None
    return extract_from_url(url)


def domain_of(url: str) -> str:
    parsed = urlparse(url)
    return (parsed.hostname or "").lower()
