"""DOM summarizer: Anthropic Haiku or DeepSeek (OpenAI-compatible).

Calls a cloud LLM to turn a stripped DOM into a 200-400 word plain-prose
structural description that a web agent can act on. The summary is what later
gets encoded into the KV bank.

Provider auto-detection: if ANTHROPIC_API_KEY is set, Haiku is used. Otherwise
DEEPSEEK_API_KEY is tried via the OpenAI-compatible endpoint. Both env vars
absent -> error.

For `popup:demo`, an explicit behavioral line is appended so the chaos-test
demo (a cookie-modal blocks the page) routes correctly.
"""

from __future__ import annotations

import os
import re
from typing import Any

from agentinception_shared.constants import DEEPSEEK_MODEL, HAIKU_MODEL, SUMMARY_WORDS

# Behavioral nudge appended to the popup:demo bank summary so the agent learns
# to dismiss the cookie modal before resuming the task. Per spec brief.
POPUP_BEHAVIOR_LINE = (
    "If a cookie-consent or marketing modal is blocking the page, dismiss it "
    "via its accept/close button, then resume the original task."
)

# Limit DOM text sent to the summarizer — keep the input lean for cheap,
# deterministic calls.
_DOM_CHAR_LIMIT = 12_000

# DeepSeek OpenAI-compatible endpoint.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class SummaryError(RuntimeError):
    """Summarizer returned an empty/unusable response."""


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
        '(e.g. "click a story title to open its comment page"). '
        "Do not invent elements you cannot see in the contents above."
    )


# --------------------------------------------------------------------------
# Provider auto-detection
# --------------------------------------------------------------------------


def _detect_provider() -> str:
    """Return 'anthropic' or 'deepseek' based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    # Neither key set — if a client was injected (tests), default to anthropic;
    # otherwise the caller will get a clear error at summarization time.
    return "anthropic"


class Summarizer:
    """Thin wrapper that calls either Anthropic or DeepSeek to summarize a page.

    Provider is auto-detected from env vars; pass ``provider=`` to override.
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int = 1200,
    ) -> None:
        self._client = client
        self._provider = provider or _detect_provider()
        self._model = model
        self.max_tokens = max_tokens

    @property
    def model(self) -> str:
        if self._model is not None:
            return self._model
        return HAIKU_MODEL if self._provider == "anthropic" else DEEPSEEK_MODEL

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._provider == "anthropic":
            self._client = self._make_anthropic_client()
        else:
            self._client = self._make_deepseek_client()
        return self._client

    def _make_anthropic_client(self):
        # Lazy import — keeps unit tests independent of the anthropic SDK.
        import anthropic  # type: ignore

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        return (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def _make_deepseek_client(self):
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise SummaryError("DEEPSEEK_API_KEY is not set")
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def summarize(self, *, dom_text: str, url: str, page_key: str) -> str:
        prompt = _build_prompt(dom_text, url, page_key)
        if self._provider == "anthropic":
            text = self._call_anthropic(prompt)
        else:
            text = self._call_deepseek(prompt)
        return self._postprocess(text, page_key)

    def _call_anthropic(self, prompt: str) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_anthropic_text(resp)

    def _call_deepseek(self, prompt: str) -> str:
        client = self._get_client()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _extract_anthropic_text(resp: Any) -> str:
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

    def _postprocess(self, text: str, page_key: str) -> str:
        text = text.strip()
        if not text:
            raise SummaryError(f"{self._provider} returned an empty response")

        low, high = SUMMARY_WORDS
        wc = word_count(text)
        if wc > high:
            text = _truncate_words(text, high)

        if page_key == "popup:demo":
            if POPUP_BEHAVIOR_LINE not in text:
                text = text.rstrip() + " " + POPUP_BEHAVIOR_LINE
            # Make sure it sits at the end so it's the model's last instruction.
            text = text.replace(POPUP_BEHAVIOR_LINE, "").rstrip()
            text = text + " " + POPUP_BEHAVIOR_LINE
        return text


# --------------------------------------------------------------------------
# Top-level convenience (mockable from compiler tests via client=)
# --------------------------------------------------------------------------


def summarize_dom(
    *,
    dom_text: str,
    url: str,
    page_key: str,
    client: Any | None = None,
    provider: str | None = None,
) -> str:
    """Functional wrapper — auto-detects provider, returns summary string."""
    return Summarizer(client=client, provider=provider).summarize(
        dom_text=dom_text, url=url, page_key=page_key
    )
