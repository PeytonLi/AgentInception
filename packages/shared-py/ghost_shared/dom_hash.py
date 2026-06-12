"""Informational structural hash of a DOM. CONTRACTS.md §3.

sha256 over the tag-skeleton: strip scripts/styles/comments/text, keep tag
names + sorted class lists in document order. Stored as metadata ONLY — it is
never used for bank lookup (that's page_key's job).
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser

_SKIP_TAGS = {"script", "style", "noscript"}


class _SkeletonParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tokens: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        classes: list[str] = []
        for name, value in attrs:
            if name == "class" and value:
                classes.extend(value.split())
        if classes:
            token = tag + "." + ".".join(sorted(classes))
        else:
            token = tag
        self.tokens.append(token)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # Self-closing tags (e.g. <br/>, <img/>) still count structurally.
        if tag in _SKIP_TAGS or self._skip_depth:
            return
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    # Comments and text are intentionally ignored.
    def handle_comment(self, data: str) -> None:  # noqa: D401
        return

    def handle_data(self, data: str) -> None:  # noqa: D401
        return


def dom_structural_hash(html: str) -> str:
    """Return the sha256 hex digest of the DOM tag-skeleton."""
    parser = _SkeletonParser()
    parser.feed(html or "")
    parser.close()
    skeleton = ">".join(parser.tokens)
    return hashlib.sha256(skeleton.encode("utf-8")).hexdigest()
