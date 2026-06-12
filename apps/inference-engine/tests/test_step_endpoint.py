"""/api/v1/step contract — CONTRACTS §6. Model mocked (FakeBackend)."""

import pytest
from fastapi.testclient import TestClient

from inference_engine.server import create_app

from fakes import FakeBackend, make_test_registry


def make_client(backend=None, registry=None):
    backend = backend or FakeBackend()
    registry = registry or make_test_registry()
    app = create_app(backend=backend, registry=registry)
    return TestClient(app), backend


def baseline_request(**overrides):
    req = {
        "session_id": "s1",
        "mode": "baseline",
        "task": "Find the top AI story",
        "url": "https://news.ycombinator.com/",
        "page_key": "hn:front",
        "dom_text": "STORY ONE two three four five six seven",
        "dom_token_count": 5000,
        "history": ["goto https://news.ycombinator.com/"],
        "step": 1,
    }
    req.update(overrides)
    return req


def mi_request(**overrides):
    overrides.setdefault("dom_text", None)
    return baseline_request(mode="mi", **overrides)


def test_step_endpoint_baseline():
    client, backend = make_client()
    resp = client.post("/api/v1/step", json=baseline_request())
    assert resp.status_code == 200
    body = resp.json()

    assert body["action"] == {"action": "click", "selector": "a.morelink"}
    assert body["bank_found"] is False
    assert body["injected_layers"] == []

    # the prompt actually sent contains task + dom + history
    text = backend.last_prompt_text()
    assert "Find the top AI story" in text
    assert "STORY ONE two three four five six seven" in text
    assert "goto https://news.ycombinator.com/" in text

    # token accounting: baseline mode sends everything -> baseline == visible
    assert body["visible_tokens"] == backend.count_prompt_tokens(backend.prompts[-1])
    assert body["baseline_tokens"] == body["visible_tokens"]

    # baseline never injects banks
    assert backend.applied_banks == [None]


def test_step_endpoint_baseline_requires_dom_text():
    client, _ = make_client()
    resp = client.post("/api/v1/step", json=baseline_request(dom_text=None))
    assert resp.status_code == 422


def test_step_endpoint_mi_with_bank():
    client, backend = make_client()
    resp = client.post("/api/v1/step", json=mi_request())
    body = resp.json()

    assert body["bank_found"] is True
    assert body["injected_layers"] == [8, 12, 16, 20]

    # visible prompt stays small: task + url + history, NO dom text
    text = backend.last_prompt_text()
    assert "Find the top AI story" in text
    assert "https://news.ycombinator.com/" in text
    assert "STORY ONE" not in text

    # banks were applied from the registry
    assert backend.applied_banks[-1] is not None
    assert sorted(backend.applied_banks[-1].keys()) == [8, 12, 16, 20]

    # baseline_tokens = dom_token_count + prompt overhead (the dom-less prompt)
    assert body["visible_tokens"] == backend.count_prompt_tokens(backend.prompts[-1])
    assert body["baseline_tokens"] == 5000 + body["visible_tokens"]


def test_step_endpoint_mi_unknown_page_falls_back_to_dom():
    """bank_found=false is the graceful-fallback demo moment, not an error (§6)."""
    client, backend = make_client()
    resp = client.post(
        "/api/v1/step",
        json=mi_request(page_key="unknown", url="https://example.com/article",
                        dom_text="ARTICLE BODY words here", dom_token_count=900),
    )
    body = resp.json()

    assert body["bank_found"] is False
    assert body["injected_layers"] == []
    assert backend.applied_banks[-1] is None

    # engine silently included the dom text for this step
    assert "ARTICLE BODY words here" in backend.last_prompt_text()

    # overhead excludes the dom: baseline_tokens = dom_token_count + dom-less prompt size
    assert body["baseline_tokens"] < 900 + body["visible_tokens"]
    assert body["baseline_tokens"] > 900


def test_step_endpoint_mi_unknown_page_without_dom_still_works():
    client, backend = make_client()
    resp = client.post(
        "/api/v1/step",
        json=mi_request(page_key="unknown", url="https://example.com/x"),
    )
    assert resp.status_code == 200
    assert resp.json()["bank_found"] is False


def test_malformed_action_retries_once_with_strict_suffix():
    backend = FakeBackend(responses=["sure! here you go", '{"action": "done", "result": {"score": 1}}'])
    client, backend = make_client(backend=backend)
    resp = client.post("/api/v1/step", json=baseline_request())
    assert resp.status_code == 200
    assert resp.json()["action"]["action"] == "done"
    assert len(backend.prompts) == 2
    assert "Respond with only the JSON object." in backend.prompts[-1][-1]["content"]


def test_malformed_action_twice_is_502():
    backend = FakeBackend(responses=["nope", "still nope"])
    client, backend = make_client(backend=backend)
    resp = client.post("/api/v1/step", json=baseline_request())
    assert resp.status_code == 502


def test_action_json_extracted_from_prose():
    backend = FakeBackend(responses=['Here it is: {"action": "goto", "url": "https://x.y"} hope that helps'])
    client, backend = make_client(backend=backend)
    resp = client.post("/api/v1/step", json=baseline_request())
    assert resp.json()["action"] == {"action": "goto", "url": "https://x.y"}
    assert len(backend.prompts) == 1


def test_healthz():
    client, _ = make_client(registry=make_test_registry(page_keys=("hn:front", "popup:demo")))
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert sorted(body["banks_loaded"]) == ["hn:front", "popup:demo"]


def test_step_rejects_unknown_mode():
    client, _ = make_client()
    resp = client.post("/api/v1/step", json=baseline_request(mode="turbo"))
    assert resp.status_code == 422
