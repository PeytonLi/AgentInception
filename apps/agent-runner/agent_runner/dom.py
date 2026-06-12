"""DOM text extraction + truncation.

The live browser yields ``document.body.innerText`` (which already excludes
script/style content). For tests and any HTML-in-hand path, :func:`html_to_text`
reproduces that behaviour from a raw HTML string. :func:`extract_dom_text`
truncates to a token budget and reports the FULL token count baseline would
have sent (CONTRACTS.md s6 ``dom_token_count``).
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from .tokenizer import TokenCounter

# Content of these elements is never user-visible text.
_SKIP_TAGS = {"script", "style", "noscript", "template", "head", "svg"}
# Block-level tags get a newline so words don't run together.
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "section", "article",
    "header", "footer", "nav", "h1", "h2", "h3", "h4", "h5", "h6", "pre",
    "blockquote", "td", "th",
}

_WS_RUN = re.compile(r"[ \t\f\v]+")
_NL_RUN = re.compile(r"\n\s*\n\s*")


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        if tag in _BLOCK_TAGS and tag not in _SKIP_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def normalize_whitespace(text: str) -> str:
    text = _WS_RUN.sub(" ", text)
    text = _NL_RUN.sub("\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    """Strip scripts/styles and return normalized visible text."""
    parser = _TextExtractor()
    parser.feed(html or "")
    parser.close()
    return normalize_whitespace(parser.text())


def extract_dom_text(
    text: str, counter: TokenCounter, max_tokens: int
) -> tuple[str, int]:
    """Return ``(truncated_text, full_token_count)``.

    ``full_token_count`` is the token count of the UNtruncated text - the
    number of DOM tokens a baseline prompt would have carried this step.
    """
    full_count = counter.count(text)
    truncated = counter.truncate(text, max_tokens)
    return truncated, full_count
