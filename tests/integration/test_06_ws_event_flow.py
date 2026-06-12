"""Test 6 — WebSocket event order and schema (A1 ↔ A4 contract).

One /api/v1/step in mi mode with hn:front bank → WS client receives:
  layer_injection → action → token_metrics
in that order, all schema-valid per CONTRACTS §7.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch required for inference engine")
pytest.importorskip("httpx_ws", reason="httpx_ws required for WS tests")
from conftest import import_app_fakes  # noqa: E402
from httpx import AsyncClient  # noqa: E402
from httpx_ws import aconnect_ws  # noqa: E402
from httpx_ws.transport import ASGIWebSocketTransport  # noqa: E402
from inference_engine.bank_registry import BankRegistry  # noqa: E402
from inference_engine.server import create_app  # noqa: E402

FakeBackend = import_app_fakes("inference-engine").FakeBackend


@pytest.mark.asyncio
async def test_ws_events_arrive_in_correct_order(banks_dir):
    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    backend = FakeBackend()
    app = create_app(backend=backend, registry=reg)

    # ASGIWebSocketTransport serves both the HTTP /api/v1/step call and the
    # /ws/events WebSocket upgrade; plain httpx.ASGITransport 404s on WS.
    transport = ASGIWebSocketTransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with aconnect_ws("http://test/ws/events", client) as ws:
            # Send one mi step with a known bank
            resp = await client.post(
                "/api/v1/step",
                json={
                    "session_id": "ws-test",
                    "mode": "mi",
                    "task": "find AI stories",
                    "url": "https://news.ycombinator.com/",
                    "page_key": "hn:front",
                    "dom_token_count": 2000,
                    "history": [],
                    "step": 0,
                },
            )
            assert resp.status_code == 200

            # A bank-found mi step emits 4 events (layer_injection, log,
            # action, token_metrics). Collect what arrives and stop on the
            # idle timeout rather than blocking on a fixed count.
            events = []
            try:
                while len(events) < 8:
                    msg = await ws.receive_text(timeout=2)
                    events.append(json.loads(msg))
            except TimeoutError:
                pass

    types = [e["type"] for e in events]
    # layer_injection must appear before action and token_metrics
    assert "layer_injection" in types
    assert "action" in types
    assert "token_metrics" in types
    li_idx = types.index("layer_injection")
    a_idx = types.index("action")
    tm_idx = types.index("token_metrics")
    assert li_idx < a_idx, "layer_injection must precede action"
    assert a_idx < tm_idx, "action must precede token_metrics"

    # Schema validation
    for ev in events:
        assert "ts" in ev, f"event missing ts: {ev['type']}"
        if ev["type"] == "layer_injection":
            assert isinstance(ev.get("layers"), list)
            assert isinstance(ev.get("active"), bool)
            assert isinstance(ev.get("page_key"), str)
        elif ev["type"] == "action":
            assert isinstance(ev.get("action"), dict)
        elif ev["type"] == "token_metrics":
            assert isinstance(ev.get("session_id"), str)
            assert isinstance(ev.get("cum_visible"), int)
            assert isinstance(ev.get("cum_baseline"), int)
