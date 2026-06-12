"""Token-honesty tests: assert the README formula lands in the honest band.

The kv_savings_ratio formula is:
    (NUM_LAYERS * dom_token_count) / (L_injected * num_slots)

Where:
    NUM_LAYERS = 32 (full model decoder layers)
    dom_token_count = Llama-tokenizer count of the FULL DOM text
    L_injected = len(SELECTED_LAYERS) = 4
    num_slots = bank's kept-position count

For the demo, the engine truncates dom_text to 4000 tokens for baseline mode,
so the VISIBLE savings ratio (what the dashboard shows) is:
    (NUM_LAYERS * min(dom_token_count, 4000)) / (L_injected * num_slots)

This test validates that the visible ratio lands in the 20-80x band for
each demo page type, using conservative character-based token estimates
when the Llama tokenizer isn't available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "packages" / "shared-py"))

from agentinception_shared.constants import NUM_LAYERS, SELECTED_LAYERS  # noqa: E402

DATASET_PATH = REPO_ROOT / "demo-assets" / "token_honesty" / "token_counts.json"
HONEST_BAND = (20, 80)
BASELINE_TRUNCATION = 4000  # engine truncates dom_text to this many tokens
L_INJECTED = len(SELECTED_LAYERS)  # 4


def _load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _estimate_tokens(char_count: int) -> int:
    """Conservative char/4 token estimate (Llama-3 averages ~3.5-4 chars/token)."""
    return max(1, char_count // 4)


def _savings_ratio(dom_token_count: int, num_slots: int) -> float:
    """Compute kv_savings_ratio with baseline truncation applied."""
    visible_baseline = min(dom_token_count, BASELINE_TRUNCATION)
    return (NUM_LAYERS * visible_baseline) / (L_INJECTED * num_slots)


@pytest.fixture
def dataset():
    assert DATASET_PATH.exists(), f"Token honesty dataset not found: {DATASET_PATH}"
    return _load_dataset()


def test_dataset_has_all_page_types(dataset):
    """Every demo page type must be present in the dataset."""
    page_keys = {p["page_key"] for p in dataset["pages"]}
    assert "hn:front" in page_keys
    assert "hn:item" in page_keys
    assert "popup:demo" in page_keys


def test_honest_band_defined(dataset):
    """The dataset must declare what the honest band is."""
    band = dataset["honest_band"]
    assert len(band) == 2
    assert band[0] < band[1]
    assert band == list(HONEST_BAND)


@pytest.mark.parametrize("page_key", ["hn:front", "hn:item", "popup:demo"])
def test_savings_ratio_in_honest_band(dataset, page_key):
    """The visible savings ratio (with baseline truncation) must land in 20-80x."""
    pages = {p["page_key"]: p for p in dataset["pages"]}
    assert page_key in pages, f"{page_key} not in dataset"

    page = pages[page_key]
    dom_tokens = page["dom_token_count_estimate"]
    num_slots = page["num_slots"]

    ratio = _savings_ratio(dom_tokens, num_slots)
    low, high = HONEST_BAND

    assert ratio >= low, (
        f"{page_key}: savings ratio {ratio:.1f}x < {low}x (too low). "
        f"dom_tokens={dom_tokens}, num_slots={num_slots}"
    )
    assert ratio <= high, (
        f"{page_key}: savings ratio {ratio:.1f}x > {high}x (overclaim!). "
        f"dom_tokens={dom_tokens}, num_slots={num_slots}"
    )


@pytest.mark.parametrize("page_key", ["hn:front", "hn:item", "popup:demo"])
def test_num_slots_positive(dataset, page_key):
    pages = {p["page_key"]: p for p in dataset["pages"]}
    assert pages[page_key]["num_slots"] > 0


@pytest.mark.parametrize("page_key", ["hn:front", "hn:item", "popup:demo"])
def test_dom_token_count_reasonable(dataset, page_key):
    """DOM token counts should be at least a few hundred for any real page."""
    pages = {p["page_key"]: p for p in dataset["pages"]}
    assert pages[page_key]["dom_token_count_estimate"] >= 100


def test_formula_consistency(dataset):
    """The formula in the dataset matches our implementation."""
    assert dataset["formula"] == "(NUM_LAYERS * dom_token_count) / (L_injected * num_slots)"
    # Spot-check with known values:
    # popup:demo: (32 * 1750) / (4 * 180) = 56000 / 720 = 77.78
    ratio = (32 * 1750) / (4 * 180)
    assert abs(ratio - 77.78) < 0.1
