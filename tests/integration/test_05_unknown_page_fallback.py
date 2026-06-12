"""Test 5 — Unknown page graceful fallback (A1).

Step with page_key="unknown" in mi mode MUST:
- Return bank_found=false
- NOT raise an exception
- Use dom_text when available (graceful-fallback demo moment per CONTRACTS §6)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch required for inference engine")

from conftest import import_app_fakes
from fastapi.testclient import TestClient
from inference_engine.bank_registry import BankRegistry
from inference_engine.server import create_app

FakeBackend = import_app_fakes("inference-engine").FakeBackend


@pytest.fixture
def client(banks_dir):
    from inference_engine.bank_registry import BankRegistry

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    backend = FakeBackend()
    app = create_app(backend=backend, registry=reg)
    return TestClient(app)


def test_unknown_page_key_returns_bank_found_false(client):
    resp = client.post(
        "/api/v1/step",
        json={
            "session_id": "test-unknown",
            "mode": "mi",
            "task": "navigate a random page",
            "url": "https://example.com/random",
            "page_key": "unknown",
            "dom_text": "Some page text.",
            "dom_token_count": 500,
            "history": [],
            "step": 0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["bank_found"] is False
    assert data["injected_layers"] == []


def test_unknown_page_with_dom_text_does_not_error(client):
    """Graceful fallback: still generates an action even with no bank."""
    resp = client.post(
        "/api/v1/step",
        json={
            "session_id": "test-fallback",
            "mode": "mi",
            "task": "extract data",
            "url": "https://other.example.com",
            "page_key": "unknown",
            "dom_text": "Page content here.",
            "dom_token_count": 300,
            "history": [],
            "step": 0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data
    # visible_tokens > 0 means the model saw the dom_text
    assert data["visible_tokens"] > 0


def test_baseline_mode_with_unknown_page_uses_dom(client):
    resp = client.post(
        "/api/v1/step",
        json={
            "session_id": "test-base",
            "mode": "baseline",
            "task": "go",
            "url": "https://example.com",
            "page_key": "unknown",
            "dom_text": "Baseline DOM.",
            "dom_token_count": 200,
            "history": [],
            "step": 0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["bank_found"] is False
