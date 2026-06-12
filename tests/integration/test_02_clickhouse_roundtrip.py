"""Test 2 — ClickHouse roundtrip with real banks (B2 ↔ A2).

Upload all banks from the filesystem into ClickHouse, then load_all_banks()
and assert byte-identical arrays for every (page_key, layer).

Marked @pytest.mark.slow because it needs a live ClickHouse.
"""
from __future__ import annotations
import numpy as np
import pytest
from agentinception_shared import bank_io, storage

pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def _ch_client():
    try:
        client = storage.get_client()
        client.command("SELECT 1")
    except Exception:
        pytest.skip("ClickHouse not reachable — start it with scripts/ch_init.sh")
    return client

def test_upload_then_load_byte_identical(banks_dir, _ch_client):
    client = _ch_client
    manifest = bank_io.read_manifest(str(banks_dir))
    for entry in manifest["banks"]:
        page_key = entry["page_key"]
        original = bank_io.load_bank(entry, str(banks_dir))
        client.command(
            f"ALTER TABLE {storage.BANKS_TABLE} DELETE WHERE page_key = %(pk)s",
            parameters={"pk": page_key},
        )
        storage.insert_bank(
            client,
            page_key=page_key,
            domain=entry.get("domain", ""),
            banks=original,
            dom_structural_hash=entry.get("dom_structural_hash", ""),
        )
    loaded = storage.load_all_banks(client)
    assert set(loaded.keys()) == {e["page_key"] for e in manifest["banks"]}
    for entry in manifest["banks"]:
        pk = entry["page_key"]
        orig = bank_io.load_bank(entry, str(banks_dir))
        rt = loaded[pk]
        for layer in orig:
            assert np.array_equal(orig[layer][0], rt[layer][0]), f"{pk} L{layer} K mismatch"
            assert np.array_equal(orig[layer][1], rt[layer][1]), f"{pk} L{layer} V mismatch"

def test_load_all_banks_includes_all_three(banks_dir, _ch_client):
    loaded = storage.load_all_banks(_ch_client)
    assert {"hn:front", "hn:item", "popup:demo"}.issubset(loaded.keys()), (
        f"missing keys: {sorted(loaded.keys())}"
    )
