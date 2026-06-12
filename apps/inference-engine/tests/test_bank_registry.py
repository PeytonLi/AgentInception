"""Bank registry — CONTRACTS §5: all banks read once at startup into an
in-memory dict keyed by page_key; ClickHouse first, manifest/.bin fallback,
loud logging about which path was used."""

import logging

import numpy as np
import pytest
import torch
from ghost_shared import bank_io
from ghost_shared.constants import HEAD_DIM, NUM_KV_HEADS, SELECTED_LAYERS

from inference_engine.bank_registry import BankRegistry


def _write_fixture_banks(banks_dir, page_keys=("hn:front",), num_slots=16):
    rng = np.random.default_rng(1)
    for pk in page_keys:
        layers = {
            layer: (
                rng.standard_normal(
                    (NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32
                ),
                rng.standard_normal(
                    (NUM_KV_HEADS, num_slots, HEAD_DIM), dtype=np.float32
                ),
            )
            for layer in SELECTED_LAYERS
        }
        bank_io.save_bank(
            banks_dir,
            pk,
            layers,
            meta={
                "domain": "news.ycombinator.com",
                "dom_structural_hash": "00" * 32,
                "summary_text_path": f"banks/{pk.replace(':', '_')}.summary.txt",
                "compiled_at": "2026-06-12T18:00:00Z",
            },
        )


def test_from_manifest_dir(tmp_path):
    _write_fixture_banks(tmp_path, page_keys=("hn:front", "hn:item"))
    reg = BankRegistry.from_manifest_dir(tmp_path)

    assert reg.source == "manifest"
    assert sorted(reg.page_keys) == ["hn:front", "hn:item"]
    per_layer = reg.get("hn:front")
    assert sorted(per_layer.keys()) == [8, 12, 16, 20]
    k, v = per_layer[8]
    assert isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
    assert k.shape == (8, 16, 128)
    assert k.dtype == torch.float32
    assert reg.num_slots("hn:front") == 16


def test_get_unknown_returns_none(tmp_path):
    _write_fixture_banks(tmp_path)
    reg = BankRegistry.from_manifest_dir(tmp_path)
    assert reg.get("unknown") is None
    assert reg.get("amazon:cart") is None
    assert reg.num_slots("amazon:cart") == 0


class _FakeQueryResult:
    def __init__(self, rows):
        self.result_rows = rows


class _FakeClient:
    def __init__(self, rows):
        self._rows = rows
        self.queries = []

    def query(self, sql, **kwargs):
        self.queries.append(sql)
        return _FakeQueryResult(self._rows)


def test_from_clickhouse(monkeypatch):
    rng = np.random.default_rng(2)
    k = rng.standard_normal((8, 9, 128), dtype=np.float32)
    v = rng.standard_normal((8, 9, 128), dtype=np.float32)
    rows = [
        (
            "hn:front",
            8,
            9,
            bank_io.to_bytes(k),
            bank_io.to_bytes(v),
        ),
        (
            "hn:front",
            12,
            9,
            bank_io.to_bytes(k * 2),
            bank_io.to_bytes(v * 2),
        ),
    ]
    fake = _FakeClient(rows)
    monkeypatch.setattr(
        "inference_engine.bank_registry._clickhouse_client", lambda url: fake
    )

    reg = BankRegistry.from_clickhouse("http://localhost:8123")

    assert reg.source == "clickhouse"
    assert reg.page_keys == ["hn:front"]
    assert sorted(reg.get("hn:front").keys()) == [8, 12]
    torch.testing.assert_close(reg.get("hn:front")[8][0], torch.from_numpy(k))
    assert "latent_memory_banks" in fake.queries[0]


def test_from_clickhouse_rejects_slot_mismatch(monkeypatch):
    k = np.zeros((8, 9, 128), dtype=np.float32)
    rows = [
        (
            "hn:front",
            8,
            12,
            bank_io.to_bytes(k),
            bank_io.to_bytes(k),
        )
    ]
    monkeypatch.setattr(
        "inference_engine.bank_registry._clickhouse_client",
        lambda url: _FakeClient(rows),
    )
    with pytest.raises(ValueError):
        BankRegistry.from_clickhouse("http://localhost:8123")


def test_load_falls_back_to_manifest(tmp_path, monkeypatch, caplog):
    _write_fixture_banks(tmp_path)

    def boom(url):
        raise ConnectionError("clickhouse is down")

    monkeypatch.setattr("inference_engine.bank_registry._clickhouse_client", boom)

    with caplog.at_level(logging.WARNING):
        reg = BankRegistry.load("http://localhost:8123", tmp_path)

    assert reg.source == "manifest"
    assert reg.page_keys == ["hn:front"]
    assert any("falling back" in r.message.lower() for r in caplog.records)


def test_load_empty_when_everything_fails(tmp_path, monkeypatch, caplog):
    def boom(url):
        raise ConnectionError("clickhouse is down")

    monkeypatch.setattr("inference_engine.bank_registry._clickhouse_client", boom)

    with caplog.at_level(logging.ERROR):
        reg = BankRegistry.load("http://localhost:8123", tmp_path / "nope")

    assert reg.source == "empty"
    assert reg.page_keys == []
