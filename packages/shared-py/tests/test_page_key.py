"""CONTRACTS.md §3 — every URL pattern maps correctly."""

import pytest

from ghost_shared.page_key import page_key

HN_FRONT_CASES = [
    "https://news.ycombinator.com",
    "https://news.ycombinator.com/",
    "https://news.ycombinator.com/news",
    "https://news.ycombinator.com/news?p=2",
    "https://news.ycombinator.com/news?p=10",
    "http://news.ycombinator.com/newest",
]

HN_ITEM_CASES = [
    "https://news.ycombinator.com/item?id=123",
    "https://news.ycombinator.com/item?id=40000000",
    "https://news.ycombinator.com/item?id=1&p=2",
]

POPUP_CASES = [
    "http://localhost:3000/popup",
    "http://localhost:8080/popup-page/index.html",
    "http://127.0.0.1:5500/demo-assets/popup-page/index.html",
    "file:///C:/code/demo-assets/popup-page/index.html",
    "file:///home/user/popup/index.html",
]

UNKNOWN_CASES = [
    "https://www.amazon.com/",
    "https://example.com/popup",  # popup path but not localhost/file
    "https://news.ycombinator.com/item",  # item without id
    "https://news.ycombinator.com/user?id=pg",
    "https://google.com/search?q=hn",
    "",
]


@pytest.mark.parametrize("url", HN_FRONT_CASES)
def test_hn_front(url):
    assert page_key(url) == "hn:front"


@pytest.mark.parametrize("url", HN_ITEM_CASES)
def test_hn_item(url):
    assert page_key(url) == "hn:item"


@pytest.mark.parametrize("url", POPUP_CASES)
def test_popup(url):
    assert page_key(url) == "popup:demo"


@pytest.mark.parametrize("url", UNKNOWN_CASES)
def test_unknown(url):
    assert page_key(url) == "unknown"
