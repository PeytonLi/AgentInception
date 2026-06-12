"""Haiku-backed DOM summarizer.

Calls claude-haiku-4-5-20251001 to turn a stripped DOM into a 200-400 word
plain-prose structural description that a web agent can act on. The summary
is what later gets encoded into the KV bank.

For `popup:demo`, an explicit behavioral line is appended so the chaos-test
demo (a cookie-modal blocks the page) routes correctly.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ghost_shared.constants import HAIKU_MODEL, SUMMARY_WORDS

# Behavioral nudge appended to the popup:demo bank summary so the agent learns
# to dismiss the cookie modal before resuming the task. Per spec brief.
POPUP_BEHAVIOR_LINE = (
    "If a cookie-consent or marketing modal is blocking the page, dismiss it "
    "via its accept/close button, then resume the original task."
)

# Limit DOM text sent to Haiku — Haiku has a large window but we keep the
# input lean to make the call cheap and deterministic.
_DOM_CHAR_LIMIT = 12_000


class SummaryError(RuntimeError):
    """Haiku returned an empty/unusable response."""


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _build_prompt(dom_text: str, url: str, page_key: str) -> str:
    dom = (dom_text or "").strip()
    if len(dom) > _DOM_CHAR_LIMIT:
        dom = dom[:_DOM_CHAR_LIMIT] + "\n...[truncated]"
    low, high = SUMMARY_WORDS
    return (
        "You are documenting a web page so that another LLM agent can navigate it "
        "without ever seeing the raw HTML.\n\n"
        f"URL: {url}\n"
        f"Page type: {page_key}\n\n"
        "Page contents (text extracted from cleaned DOM):\n"
        "<<<\n"
        f"{dom}\n"
        ">>>\n\n"
        "Write a single paragraph of plain prose, no bullet points, no headings, "
        f"between {low} and {high} words. Describe: (a) the main visual regions of "
        "the page (header / nav / main column / sidebars / footer); (b) the "
        "interactive elements that matter for an agent (links, buttons, forms, "
        "modals) and where they live; (c) where the key data the agent might want "
        "to read is located; (d) one or two sentences on how to act on this page "
        "(e.g. \"click a story title to open its comment page\"). "
        "Do not invent elements you cannot see in the contents above."
    )


class HaikuSummarizer:
    """Thin wrapper around anthropic.Messages.create that produces page summaries."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = HAIKU_MODEL,
        max_tokens: int = 1200,
    ) -> None:
        self._client = client
        self.model = model
        self.max_tokens = max_tokens

    def _get_client(self):
        if self._client is not None:
            return self._client
        # Lazy import — keeps unit tests independent of the anthropic SDK.
        import anthropic  # type: ignore

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        return self._client

    def summarize(self, *, dom_text: str, url: str, page_key: str) -> str:
        prompt = _build_prompt(dom_text, url, page_key)
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(resp).strip()
        if not text:
            raise SummaryError("Haiku returned an empty response")

        low, high = SUMMARY_WORDS
        wc = word_count(text)
        if wc > high:
            text = _truncate_words(text, high)
            wc = word_count(text)
        # If the model came back too short, we accept (Haiku is fine here) but
        # tests guard the high end; the low end is enforced in CLI validation
        # where a human can re-roll.
        if page_key == "popup:demo":
            if POPUP_BEHAVIOR_LINE not in text:
                text = text.rstrip() + " " + POPUP_BEHAVIOR_LINE
            # Make sure it sits at the end so it's the model's last instruction.
            text = text.replace(POPUP_BEHAVIOR_LINE, "").rstrip()
            text = text + " " + POPUP_BEHAVIOR_LINE
        return text


def _extract_text(resp: Any) -> str:
    """Pull text out of an Anthropic Messages response (handles mocks too)."""
    content = getattr(resp, "content", None)
    if content is None:
        return ""
    out: list[str] = []
    for block in content:
        t = getattr(block, "text", None)
        if isinstance(t, str):
            out.append(t)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            out.append(block["text"])
    return "".join(out)


def summarize_dom(*, dom_text: str, url: str, page_key: str, client: Any | None = None) -> str:
    """Functional convenience wrapper (mockable from compiler tests)."""
    return HaikuSummarizer(client=client).summarize(
        dom_text=dom_text, url=url, page_key=page_key
    )
