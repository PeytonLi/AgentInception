#!/usr/bin/env python
"""Capture fresh DOM snapshots from live Hacker News for bank compilation.

Produces clean HTML snapshots + metadata for:
  - HN front page → demo-assets/snapshots/hn_front.html
  - HN item page  → demo-assets/snapshots/hn_item.html

Each snapshot comes with a `.meta.json` sidecar containing the
dom_structural_hash, extracted text length, capture timestamp, and source URL.

The popup fixture is always read from the local file (no live capture needed).

Prerequisites:
    pip install playwright && playwright install chromium

Usage:
    python scripts/capture_dom.py                          # default output
    python scripts/capture_dom.py --out demo-assets/snapshots  # explicit
    python scripts/capture_dom.py --item-url https://news.ycombinator.com/item?id=12345
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))

from agentinception_shared.dom_hash import dom_structural_hash  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "demo-assets" / "snapshots"
HN_FRONT_URL = "https://news.ycombinator.com/"


def _strip_scripts_and_styles(html: str) -> tuple[str, str]:
    """Strip <script>/<style>/<noscript>/comments, return (cleaned_html, visible_text)."""
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    cleaned = str(soup)
    text = soup.get_text(separator="\n", strip=True)
    return cleaned, text


def _capture_page(url: str, *, timeout_ms: int = 30000) -> str:
    """Load a URL via headless Chromium and return the raw HTML."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36 AgentInceptionCapture"
                )
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except Exception:
                pass
            return page.content()
        finally:
            browser.close()


def _find_item_url(front_html: str) -> str | None:
    """Extract the first item?id=... URL from the HN front page HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(front_html, "html.parser")
    # Look for comment links (they contain "comments" or link to item?id=)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "item?id=" in href:
            if not href.startswith("http"):
                href = "https://news.ycombinator.com/" + href.lstrip("/")
            return href
    return None


def _save_snapshot(
    out_dir: Path, name: str, raw_html: str, url: str
) -> dict:
    """Save HTML + metadata, return the meta dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned, text = _strip_scripts_and_styles(raw_html)
    struct_hash = dom_structural_hash(cleaned)

    html_path = out_dir / f"{name}.html"
    html_path.write_text(raw_html, encoding="utf-8")

    meta = {
        "page_key": name.replace("_", ":").replace("hn:", "hn:"),
        "source_url": url,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dom_structural_hash": struct_hash,
        "text_length_chars": len(text),
        "html_length_bytes": len(raw_html.encode("utf-8")),
    }
    meta_path = out_dir / f"{name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"  {name:<10}  {len(raw_html):>8} bytes  hash={struct_hash[:12]}...  → {html_path}")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--item-url",
        help="Specific HN item URL to capture (default: auto-detect from front page)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)

    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except ImportError:
        print(
            "ERROR: playwright not installed. Run:\n"
            "  pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 1

    try:
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError:
        print(
            "ERROR: beautifulsoup4 not installed. Run:\n"
            "  pip install beautifulsoup4",
            file=sys.stderr,
        )
        return 1

    print("[capture] Fetching HN front page...")
    front_html = _capture_page(HN_FRONT_URL)
    _save_snapshot(out_dir, "hn_front", front_html, HN_FRONT_URL)

    # Find or use provided item URL.
    item_url = args.item_url
    if not item_url:
        item_url = _find_item_url(front_html)
    if item_url:
        print(f"[capture] Fetching HN item page: {item_url}")
        item_html = _capture_page(item_url)
        _save_snapshot(out_dir, "hn_item", item_html, item_url)
    else:
        print("[capture] WARNING: could not find an item URL on the front page.",
              file=sys.stderr)

    print(f"\nSnapshots saved to {out_dir}")
    print("Hand these to R1 for bank compilation:")
    print(f"  python scripts/compile_real_banks.py --out banks/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
