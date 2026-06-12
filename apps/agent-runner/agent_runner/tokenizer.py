"""Token counting for the comparison chart.

CONTRACTS.md s6 specifies Llama-tokenizer counts. We use the real
``transformers.AutoTokenizer`` for ``MODEL_ID`` when it is importable and the
model files are reachable; otherwise we fall back to a deterministic heuristic
counter so the runner and its tests work with no GPU, no HF token, and no
network. The active backend is reported via :meth:`TokenCounter.name`.
"""

from __future__ import annotations

import re
import threading
from typing import Protocol

try:
    from ghost_shared.constants import MODEL_ID
except Exception:  # pragma: no cover - shared-py always present in the monorepo
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Splits text into word and punctuation tokens. A crude but stable proxy for
# BPE that is good enough for the demo's relative comparison.
_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    def count(self, text: str | None) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    @property
    def name(self) -> str: ...


class HeuristicTokenCounter:
    """Backend-free approximation: one token per word or punctuation mark."""

    @property
    def name(self) -> str:
        return "heuristic"

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        return len(_WORD_RE.findall(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if not text:
            return ""
        end: int | None = None
        for i, match in enumerate(_WORD_RE.finditer(text)):
            if i == max_tokens - 1:
                end = match.end()
                break
        if end is None:
            return text  # fewer tokens than the cap
        return text[:end]


class HFTokenCounter:
    """Real Llama tokenizer backed counter."""

    def __init__(self, tokenizer: object) -> None:
        self._tok = tokenizer

    @property
    def name(self) -> str:
        return "transformers"

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        ids = self._tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self._tok.decode(ids[:max_tokens])


_lock = threading.Lock()
_cached: TokenCounter | None = None


def get_token_counter(
    model_id: str = MODEL_ID, prefer_hf: bool = True
) -> TokenCounter:
    """Return a process-wide cached token counter.

    Tries the real Llama tokenizer first; silently degrades to the heuristic
    if ``transformers`` is unavailable or the model cannot be loaded.
    """
    global _cached
    if _cached is not None:
        return _cached
    with _lock:
        if _cached is not None:
            return _cached
        counter: TokenCounter | None = None
        if prefer_hf:
            try:
                from transformers import AutoTokenizer  # type: ignore

                tok = AutoTokenizer.from_pretrained(model_id)
                counter = HFTokenCounter(tok)
            except Exception:
                counter = None
        _cached = counter or HeuristicTokenCounter()
        return _cached


def reset_token_counter() -> None:
    """Clear the cache (tests only)."""
    global _cached
    _cached = None
