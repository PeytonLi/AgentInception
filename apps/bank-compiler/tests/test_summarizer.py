"""Haiku summarizer tests — Anthropic client mocked, no network calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bank_compiler.summarizer import (
    POPUP_BEHAVIOR_LINE,
    HaikuSummarizer,
    SummaryError,
    word_count,
)


def _fake_response(text: str) -> MagicMock:
    """Match anthropic.Messages.create()'s response shape: response.content[0].text."""
    block = MagicMock()
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    return resp


def test_summarizer_calls_haiku_with_stripped_dom():
    client = MagicMock()
    summary = " ".join(["fact"] * 250)  # 250 words, within 200-400
    client.messages.create.return_value = _fake_response(summary)

    sx = HaikuSummarizer(client=client, model="claude-haiku-4-5-20251001")
    result = sx.summarize(
        dom_text="Header content here\nMain article body\nFooter",
        url="https://news.ycombinator.com/",
        page_key="hn:front",
    )

    assert result.strip() == summary.strip()

    args, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    sent_messages = kwargs["messages"]
    assert sent_messages[0]["role"] == "user"
    sent_text = sent_messages[0]["content"]
    if isinstance(sent_text, list):
        sent_text = sent_text[0].get("text", "")
    assert "Main article body" in sent_text


def test_summarizer_word_count_in_range():
    """Returned summary should fall within (or be coerced toward) 200-400 words."""
    client = MagicMock()
    base = "word " * 350
    client.messages.create.return_value = _fake_response(base)

    sx = HaikuSummarizer(client=client)
    result = sx.summarize(dom_text="x", url="https://news.ycombinator.com/", page_key="hn:front")
    wc = word_count(result)
    assert 200 <= wc <= 400, f"summary length {wc} out of 200-400 range"


def test_summarizer_appends_popup_behavior_for_popup_key():
    client = MagicMock()
    base = " ".join(["info"] * 230)
    client.messages.create.return_value = _fake_response(base)

    sx = HaikuSummarizer(client=client)
    result = sx.summarize(
        dom_text="<html>...</html>",
        url="http://localhost:8080/popup",
        page_key="popup:demo",
    )
    assert POPUP_BEHAVIOR_LINE in result
    assert result.endswith(POPUP_BEHAVIOR_LINE)


def test_summarizer_does_not_append_popup_line_for_other_keys():
    client = MagicMock()
    base = " ".join(["info"] * 230)
    client.messages.create.return_value = _fake_response(base)

    sx = HaikuSummarizer(client=client)
    result = sx.summarize(dom_text="x", url="https://news.ycombinator.com/", page_key="hn:front")
    assert POPUP_BEHAVIOR_LINE not in result


def test_summarizer_raises_on_empty_response():
    client = MagicMock()
    client.messages.create.return_value = _fake_response("   ")

    sx = HaikuSummarizer(client=client)
    with pytest.raises(SummaryError):
        sx.summarize(dom_text="x", url="https://news.ycombinator.com/", page_key="hn:front")
