"""Test 1 — Bank binary contract (B1 ↔ shared-py).

Asserts that every bank file referenced in manifest.json loads via
shared-py.bank_io with correct shape [8, S, 128], float32, equal S across layers.
"""

from __future__ import annotations

import os

import numpy as np
from conftest import REQUIRED_PAGE_KEYS

from ghost_shared import bank_io, constants


def test_all_bank_files_exist(banks_dir):
    """Every .bin file named in the manifest is present on disk."""
    manifest = bank_io.read_manifest(str(banks_dir))
    for entry in manifest["banks"]:
        pk = entry["page_key"]
        for layer_str, names in entry["files"].items():
            for kind in ("k", "v"):
                path = banks_dir / names[kind]
                assert path.exists(), f"{pk} L{layer_str} {kind}: {path} missing"


def test_bank_byte_lengths_match_num_slots(banks_dir):
    """Per-file byte size == 8 * num_slots * 128 * 4."""
    manifest = bank_io.read_manifest(str(banks_dir))
    for entry in manifest["banks"]:
        pk = entry["page_key"]
        num_slots = int(entry["num_slots"])
        expected = constants.NUM_KV_HEADS * num_slots * constants.HEAD_DIM * 4
        for layer_str, names in entry["files"].items():
            for kind in ("k", "v"):
                path = banks_dir / names[kind]
                size = os.path.getsize(path)
                assert size == expected, (
                    f"{pk} L{layer_str} {kind}: size {size} != expected {expected}"
                )


def test_bank_io_load_yields_correct_shape_and_dtype(banks_dir):
    """load_bank() returns [8, num_slots, 128] float32 for each layer."""
    manifest = bank_io.read_manifest(str(banks_dir))
    for entry in manifest["banks"]:
        pk = entry["page_key"]
        num_slots = int(entry["num_slots"])
        loaded = bank_io.load_bank(entry, str(banks_dir))
        for layer, (k, v) in loaded.items():
            exp = (constants.NUM_KV_HEADS, num_slots, constants.HEAD_DIM)
            assert k.shape == exp, f"{pk} L{layer} K shape {k.shape} != {exp}"
            assert v.shape == exp, f"{pk} L{layer} V shape {v.shape} != {exp}"
            assert str(k.dtype) == "float32", f"{pk} L{layer} K dtype {k.dtype}"
            assert str(v.dtype) == "float32", f"{pk} L{layer} V dtype {v.dtype}"


def test_num_slots_equal_across_layers(banks_dir):
    """Within one bank, num_slots is identical across all 4 layers."""
    manifest = bank_io.read_manifest(str(banks_dir))
    for entry in manifest["banks"]:
        pk = entry["page_key"]
        loaded = bank_io.load_bank(entry, str(banks_dir))
        slot_counts = {k.shape[1] for k, _ in loaded.values()}
        assert len(slot_counts) == 1, (
            f"{pk}: num_slots differs across layers: {slot_counts}"
        )


def test_manifest_covers_all_required_page_keys(banks_dir):
    """All 3 demo page_keys are present in the manifest."""
    manifest = bank_io.read_manifest(str(banks_dir))
    page_keys = {e["page_key"] for e in manifest["banks"]}
    missing = REQUIRED_PAGE_KEYS - page_keys
    assert not missing, f"manifest missing: {sorted(missing)}"


def test_to_from_bytes_roundtrip():
    """Low-level to_bytes → from_bytes is byte-identical."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(
        (constants.NUM_KV_HEADS, 100, constants.HEAD_DIM), dtype=np.float32
    )
    buf = bank_io.to_bytes(arr)
    arr2 = bank_io.from_bytes(buf, 100)
    assert np.array_equal(arr, arr2)
