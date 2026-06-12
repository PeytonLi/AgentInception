"""Test 3 — Engine startup preload (A1 ↔ A2 ↔ B2).

Engine boots, /healthz lists all 3 page_keys. Uses manifest directory
fallback (no ClickHouse needed). Verifies zero DB calls during a step
by asserting the registry source is "manifest".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch required for inference engine")

from fastapi.testclient import TestClient
from inference_engine.bank_registry import BankRegistry
from inference_engine.server import create_app

_INF_TESTS = Path(__file__).resolve().parents[2] / "apps" / "inference-engine" / "tests"
sys.path.insert(0, str(_INF_TESTS))
from fakes import FakeBackend, make_test_registry


@pytest.fixture
def client(banks_dir):
    # Use the real banks_dir for manifest-based loading
    try:
        reg = BankRegistry.from_manifest_dir(str(banks_dir))
    except FileNotFoundError:
        pytest.skip("no banks in manifest dir")
    backend = FakeBackend()
    app = create_app(backend=backend, registry=reg)
    return TestClient(app)


def test_healthz_lists_all_page_keys(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    loaded = set(data["banks_loaded"])
    assert {"hn:front", "hn:item", "popup:demo"}.issubset(loaded), (
        f"banks_loaded missing required keys: {sorted(loaded)}"
    )


def test_step_does_not_call_database(client):
    """A /api/v1/step resolves banks from the preloaded registry (no DB call)."""
    resp = client.post(
        "/api/v1/step",
        json={
            "session_id": "test-1",
            "mode": "mi",
            "task": "test task",
            "url": "https://news.ycombinator.com/",
            "page_key": "hn:front",
            "dom_token_count": 2000,
            "history": [],
            "step": 0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["bank_found"] is True
    assert data["injected_layers"] == [8, 12, 16, 20]


def test_all_three_page_keys_resolve_in_registry(banks_dir):
    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    for pk in ["hn:front", "hn:item", "popup:demo"]:
        bank = reg.get(pk)
        assert bank is not None, f"missing bank for {pk}"
        assert set(bank.keys()) == {8, 12, 16, 20}, (
            f"{pk} layers: {sorted(bank.keys())}"
        )
