"""page_key() contract tests — CONTRACTS.md §3. Lookup is by page type, never DOM hash."""

import pytest

from ghost_shared.page_key import page_key


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # hn:front — front page and paginated front page
        ("https://news.ycombinator.com/", "hn:front"),
        ("https://news.ycombinator.com", "hn:front"),
        ("http://news.ycombinator.com/", "hn:front"),
        ("https://news.ycombinator.com/news", "hn:front"),
        ("https://news.ycombinator.com/news?p=2", "hn:front"),
        ("https://news.ycombinator.com/news?p=14", "hn:front"),
        # hn:item — comment pages
        ("https://news.ycombinator.com/item?id=44001234", "hn:item"),
        ("http://news.ycombinator.com/item?id=1", "hn:item"),
        # popup:demo — local fixture page on any port, or file:// URL
        ("http://localhost:8080/popup.html", "popup:demo"),
        ("http://localhost:3999/demo/popup", "popup:demo"),
        ("http://localhost:5173/popup-page/", "popup:demo"),
        ("file:///C:/Users/x/demo-assets/popup-page/index.html", "popup:demo"),
        ("file:///home/rahul/popup.html", "popup:demo"),
        # unknown — everything else falls back to plain prompt
        ("https://example.com/article", "unknown"),
        ("https://news.ycombinator.com/newest", "unknown"),
        ("https://news.ycombinator.com/from?site=example.com", "unknown"),
        ("https://news.ycombinator.com/user?id=pg", "unknown"),
        ("http://localhost:8080/other.html", "unknown"),
        ("file:///tmp/somepage.html", "unknown"),
        ("https://arxiv.org/abs/2605.06225", "unknown"),
    ],
)
def test_page_key(url: str, expected: str) -> None:
    assert page_key(url) == expected


def test_item_without_id_is_unknown() -> None:
    # An /item URL with no id param is not a comment page we banked
    assert page_key("https://news.ycombinator.com/item") == "unknown"
