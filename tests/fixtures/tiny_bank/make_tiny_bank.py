"""Fixture generator — CONTRACTS §10.

Writes a random-valued but shape-correct bank ([8, 16, 128] float32 per
selected layer) for page_key "hn:front" into ./generated/, manifest included.
A1 develops against this until B1's real banks arrive at the H+4 shape sync.

Run:  python tests/fixtures/tiny_bank/make_tiny_bank.py
"""

from pathlib import Path

import numpy as np

from ghost_shared import bank_io
from ghost_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS

NUM_SLOTS = 16
PAGE_KEY = "hn:front"
OUT_DIR = Path(__file__).parent / "generated"


def main() -> None:
    rng = np.random.default_rng(1234)
    layers = {
        layer: (
            rng.standard_normal((NUM_KV_HEADS, NUM_SLOTS, HEAD_DIM), dtype=np.float32),
            rng.standard_normal((NUM_KV_HEADS, NUM_SLOTS, HEAD_DIM), dtype=np.float32),
        )
        for layer in SELECTED_LAYERS
    }
    bank_io.save_bank(
        str(OUT_DIR),
        PAGE_KEY,
        layers,
        meta={
            "domain": "news.ycombinator.com",
            "dom_structural_hash": "0" * 64,
            "summary_text_path": "",
            "compiled_at": "2026-06-12T00:00:00Z",
        },
    )
    print(
        f"fixture bank written to {OUT_DIR} (page_key={PAGE_KEY}, "
        f"{len(layers)} layers x [{NUM_KV_HEADS}, {NUM_SLOTS}, {HEAD_DIM}])"
    )


if __name__ == "__main__":
    main()
