"""A3 test: 50k-token page -> extracted text <= 4000 tokens, scripts absent."""

from __future__ import annotations

from agent_runner.dom import extract_dom_text, html_to_text
from agent_runner.tokenizer import HeuristicTokenCounter


def _big_html(paragraphs: int = 6000) -> str:
    body = "\n".join(
        f"<p>Story number {i} about distributed systems and latency budgets.</p>"
        for i in range(paragraphs)
    )
    return (
        "<!doctype html><html><head>"
        "<style>.x{color:red} body{font:12px}</style>"
        "<script>var SECRETSCRIPTTOKEN = function(){ return 1; };</script>"
        f"</head><body>{body}"
        "<script>window.SECRETSCRIPTTOKEN2 && track();</script>"
        "</body></html>"
    )


def test_scripts_and_styles_are_stripped():
    text = html_to_text(_big_html(50))
    assert "SECRETSCRIPTTOKEN" not in text
    assert "SECRETSCRIPTTOKEN2" not in text
    assert "color:red" not in text
    assert "Story number 0" in text


def test_extraction_truncates_to_token_cap():
    counter = HeuristicTokenCounter()
    html = _big_html(6000)
    text = html_to_text(html)

    full_count = counter.count(text)
    assert full_count > 50_000, f"fixture should exceed 50k tokens, got {full_count}"

    truncated, reported_full = extract_dom_text(text, counter, max_tokens=4000)
    assert reported_full == full_count
    assert counter.count(truncated) <= 4000
    assert "SECRETSCRIPTTOKEN" not in truncated
