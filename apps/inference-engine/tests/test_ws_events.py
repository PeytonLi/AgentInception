"""WS event hub — CONTRACTS §7. A step emits layer_injection -> action -> token_metrics."""

from fastapi.testclient import TestClient

from inference_engine.server import create_app

from fakes import FakeBackend, make_test_registry
from test_step_endpoint import mi_request


def make_client():
    backend = FakeBackend()
    app = create_app(backend=backend, registry=make_test_registry())
    return TestClient(app), backend


def drain(ws, count):
    return [ws.receive_json() for _ in range(count)]


def test_ws_event_sequence():
    client, _ = make_client()
    with client.websocket_connect("/ws/events") as ws:
        resp = client.post("/api/v1/step", json=mi_request())
        assert resp.status_code == 200

        events, types = [], []
        while len([t for t in types if t in ("layer_injection", "action", "token_metrics")]) < 3:
            e = ws.receive_json()
            events.append(e)
            types.append(e["type"])

        core = [t for t in types if t in ("layer_injection", "action", "token_metrics")]
        assert core == ["layer_injection", "action", "token_metrics"]
        assert all("ts" in e for e in events)

        injection = next(e for e in events if e["type"] == "layer_injection")
        assert injection["layers"] == [8, 12, 16, 20]
        assert injection["active"] is True
        assert injection["page_key"] == "hn:front"
        assert injection["num_slots"] == 16

        action = next(e for e in events if e["type"] == "action")
        assert action["step"] == 1
        assert action["action"]["action"] == "click"

        metrics = next(e for e in events if e["type"] == "token_metrics")
        assert metrics["session_id"] == "s1"
        assert metrics["mode"] == "mi"
        assert metrics["visible_tokens"] > 0
        assert metrics["baseline_tokens"] == 5000 + metrics["visible_tokens"]
        assert metrics["cum_visible"] == metrics["visible_tokens"]
        assert metrics["cum_baseline"] == metrics["baseline_tokens"]
        assert metrics["kv_savings_ratio"] > 1


def test_token_metrics_accumulate_per_session():
    client, _ = make_client()
    with client.websocket_connect("/ws/events") as ws:
        client.post("/api/v1/step", json=mi_request(step=1))
        client.post("/api/v1/step", json=mi_request(step=2))

        metrics = []
        while len(metrics) < 2:
            e = ws.receive_json()
            if e["type"] == "token_metrics":
                metrics.append(e)

        assert metrics[1]["cum_visible"] == metrics[0]["cum_visible"] + metrics[1]["visible_tokens"]
        assert metrics[1]["cum_baseline"] == metrics[0]["cum_baseline"] + metrics[1]["baseline_tokens"]


def test_inactive_injection_event_for_unknown_page():
    client, _ = make_client()
    with client.websocket_connect("/ws/events") as ws:
        client.post(
            "/api/v1/step",
            json=mi_request(page_key="unknown", url="https://example.com", dom_text="X", dom_token_count=10),
        )
        e = ws.receive_json()
        while e["type"] != "layer_injection":
            e = ws.receive_json()
        assert e["layers"] == []
        assert e["active"] is False
        assert e["num_slots"] == 0


def test_internal_frame_rebroadcast():
    client, _ = make_client()
    with client.websocket_connect("/ws/events") as ws:
        resp = client.post("/internal/frame", json={"jpeg_base64": "aGVsbG8="})
        assert resp.status_code == 200
        e = ws.receive_json()
        assert e["type"] == "viewport_frame"
        assert e["jpeg_base64"] == "aGVsbG8="
