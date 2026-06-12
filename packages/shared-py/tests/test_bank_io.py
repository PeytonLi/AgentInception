"""CONTRACTS.md §4 — bank serialization roundtrip + validation + manifest."""

import json

import numpy as np
import pytest

from ghost_shared import bank_io
from ghost_shared.bank_io import BankFormatError
from ghost_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS


def _rand_bank(num_slots=16, seed=0):
    rng = np.random.default_rng(seed)
    banks = {}
    for layer in SELECTED_LAYERS:
        k = rng.standard_normal(
            (NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32
        )
        v = rng.standard_normal(
            (NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32
        )
        banks[layer] = (k, v)
    return banks


def test_to_from_bytes_exact():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((NUM_KV_HEADS, 16, HEAD_DIM), dtype=np.float32)
    back = bank_io.from_bytes(bank_io.to_bytes(arr), 16)
    assert np.array_equal(arr, back)


def test_to_bytes_rejects_wrong_dtype():
    arr = np.zeros((NUM_KV_HEADS, 16, HEAD_DIM), dtype=np.float64)
    with pytest.raises(BankFormatError):
        bank_io.to_bytes(arr)


def test_to_bytes_rejects_wrong_shape():
    arr = np.zeros((4, 16, HEAD_DIM), dtype=np.float32)  # wrong head count
    with pytest.raises(BankFormatError):
        bank_io.to_bytes(arr)
    arr2 = np.zeros((NUM_KV_HEADS, 16, 64), dtype=np.float32)  # wrong head_dim
    with pytest.raises(BankFormatError):
        bank_io.to_bytes(arr2)


def test_from_bytes_rejects_truncated_buffer():
    arr = np.zeros((NUM_KV_HEADS, 16, HEAD_DIM), dtype=np.float32)
    buf = bank_io.to_bytes(arr)
    with pytest.raises(BankFormatError):
        bank_io.from_bytes(buf[:-4], 16)


def test_bank_filename():
    assert bank_io.bank_filename("hn:front", 8, "k") == "hn_front__L8__k.bin"
    assert bank_io.bank_filename("popup:demo", 20, "v") == "popup_demo__L20__v.bin"


def test_save_load_roundtrip(tmp_path):
    banks = _rand_bank(num_slots=24, seed=7)
    entry = bank_io.save_bank(
        str(tmp_path),
        "hn:front",
        banks,
        meta={
            "domain": "news.ycombinator.com",
            "dom_structural_hash": "deadbeef",
            "summary_text_path": "banks/hn_front.summary.txt",
        },
    )
    assert entry["num_slots"] == 24
    assert set(entry["files"].keys()) == {str(l) for l in SELECTED_LAYERS}

    loaded = bank_io.load_bank(entry, str(tmp_path))
    for layer in SELECTED_LAYERS:
        assert np.array_equal(banks[layer][0], loaded[layer][0])
        assert np.array_equal(banks[layer][1], loaded[layer][1])


def test_save_bank_requires_all_selected_layers(tmp_path):
    banks = _rand_bank()
    del banks[SELECTED_LAYERS[0]]
    with pytest.raises(BankFormatError):
        bank_io.save_bank(str(tmp_path), "hn:front", banks)


def test_manifest_schema(tmp_path):
    """Written manifest validates against the CONTRACTS §4 shape."""
    pydantic = pytest.importorskip("pydantic")
    from typing import Dict

    class FilePair(pydantic.BaseModel):
        k: str
        v: str

    class BankEntry(pydantic.BaseModel):
        page_key: str
        domain: str
        num_slots: int
        dom_structural_hash: str
        summary_text_path: str
        files: Dict[str, FilePair]
        compiled_at: str

    class Manifest(pydantic.BaseModel):
        model_id: str
        selected_layers: list[int]
        banks: list[BankEntry]

    bank_io.save_bank(str(tmp_path), "hn:item", _rand_bank(8, seed=3))
    bank_io.save_bank(str(tmp_path), "popup:demo", _rand_bank(10, seed=4))

    with open(tmp_path / "manifest.json", encoding="utf-8") as fh:
        data = json.load(fh)
    manifest = Manifest(**data)
    assert manifest.selected_layers == SELECTED_LAYERS
    assert {b.page_key for b in manifest.banks} == {"hn:item", "popup:demo"}


def test_load_all_banks_from_dir(tmp_path):
    bank_io.save_bank(str(tmp_path), "hn:front", _rand_bank(12, seed=1))
    bank_io.save_bank(str(tmp_path), "hn:item", _rand_bank(15, seed=2))
    all_banks = bank_io.load_all_banks_from_dir(str(tmp_path))
    assert set(all_banks.keys()) == {"hn:front", "hn:item"}
    assert set(all_banks["hn:front"].keys()) == set(SELECTED_LAYERS)
