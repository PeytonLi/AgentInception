"""bank_io contract tests — CONTRACTS.md §4 binary format.

Per layer: two arrays [NUM_KV_HEADS=8, num_slots, HEAD_DIM=128] float32 C-order,
serialized with arr.tobytes(), read back with np.frombuffer().reshape(...).
"""

import json

import numpy as np
import pytest

from ghost_shared import bank_io


def _rand_bank(num_slots: int = 16, kv_heads: int = 8, head_dim: int = 128) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((kv_heads, num_slots, head_dim), dtype=np.float32)


def test_roundtrip_exact() -> None:
    arr = _rand_bank(num_slots=312)
    buf = bank_io.serialize_bank_array(arr)
    assert isinstance(buf, bytes)
    assert len(buf) == 8 * 312 * 128 * 4  # float32
    back = bank_io.deserialize_bank_array(buf)
    assert back.shape == (8, 312, 128)
    assert back.dtype == np.float32
    np.testing.assert_array_equal(back, arr)


def test_serialize_casts_to_float32_c_order() -> None:
    arr64 = np.asfortranarray(_rand_bank().astype(np.float64))
    buf = bank_io.serialize_bank_array(arr64)
    back = bank_io.deserialize_bank_array(buf)
    np.testing.assert_array_equal(back, arr64.astype(np.float32))


def test_serialize_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        bank_io.serialize_bank_array(np.zeros((16, 128), dtype=np.float32))


def test_deserialize_infers_num_slots() -> None:
    arr = _rand_bank(num_slots=7)
    back = bank_io.deserialize_bank_array(bank_io.serialize_bank_array(arr))
    assert back.shape == (8, 7, 128)


def test_deserialize_rejects_truncated_buffer() -> None:
    buf = bank_io.serialize_bank_array(_rand_bank())
    with pytest.raises(ValueError):
        bank_io.deserialize_bank_array(buf[:-3])


def test_bank_file_name() -> None:
    assert bank_io.bank_file_name("hn:front", 8, "k") == "hn_front__L8__k.bin"
    assert bank_io.bank_file_name("popup:demo", 20, "v") == "popup_demo__L20__v.bin"
    with pytest.raises(ValueError):
        bank_io.bank_file_name("hn:front", 8, "q")


def test_write_and_load_banks(tmp_path) -> None:
    layers = {8: (_rand_bank(), _rand_bank() * 2), 12: (_rand_bank() + 1, _rand_bank() - 1)}
    files = bank_io.write_bank_files(tmp_path, "hn:front", layers)
    assert files["8"]["k"] == "hn_front__L8__k.bin"
    assert (tmp_path / "hn_front__L12__v.bin").exists()

    entry = {
        "page_key": "hn:front",
        "domain": "news.ycombinator.com",
        "num_slots": 16,
        "dom_structural_hash": "ab" * 32,
        "summary_text_path": "banks/hn_front.summary.txt",
        "files": files,
        "compiled_at": "2026-06-12T18:00:00Z",
    }
    bank_io.write_manifest(tmp_path, [entry])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert manifest["selected_layers"] == [8, 12, 16, 20]

    loaded = bank_io.load_banks(tmp_path)
    assert set(loaded.keys()) == {"hn:front"}
    assert set(loaded["hn:front"].keys()) == {8, 12}
    k8, v8 = loaded["hn:front"][8]
    np.testing.assert_array_equal(k8, layers[8][0])
    np.testing.assert_array_equal(v8, layers[8][1])


def test_write_bank_files_rejects_slot_mismatch(tmp_path) -> None:
    # num_slots must be identical across layers within one bank (CONTRACTS §4)
    layers = {8: (_rand_bank(16), _rand_bank(16)), 12: (_rand_bank(17), _rand_bank(17))}
    with pytest.raises(ValueError):
        bank_io.write_bank_files(tmp_path, "hn:front", layers)


def test_write_bank_files_rejects_k_v_slot_mismatch(tmp_path) -> None:
    layers = {8: (_rand_bank(16), _rand_bank(15))}
    with pytest.raises(ValueError):
        bank_io.write_bank_files(tmp_path, "hn:front", layers)
