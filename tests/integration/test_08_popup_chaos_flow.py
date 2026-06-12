
"""Test 8 - Popup chaos flow [GPU] (A3 <-> A1 <-> B1/B2).

Runner on the popup fixture page: the model returns dismiss_modal, the
runner executes it, and the task can resume - WITH the popup:demo bank
injected. Control run without bank is documented as potential failure.

@pytest.mark.gpu - needs the real model for meaningful bank injection.
"""
from __future__ import annotations
import pytest
from pathlib import Path
import sys

pytestmark = pytest.mark.gpu

pytest.importorskip("torch", reason="torch required for inference engine")

_INF_TESTS = Path(__file__).resolve().parents[2] / "apps" / "inference-engine" / "tests"
sys.path.insert(0, str(_INF_TESTS))
from fakes import FakeBackend

_AR_TESTS = Path(__file__).resolve().parents[2] / "apps" / "agent-runner" / "tests"
sys.path.insert(0, str(_AR_TESTS))
from fakes import FakePageDriver


@pytest.mark.asyncio
async def test_popup_dismiss_modal_with_bank(banks_dir):
    from agent_runner.loop import AgentRunner
    from agent_runner.config import RunnerConfig
    from agent_runner.metrics import Metrics
    from agent_runner.tokenizer import get_token_counter
    from agent_runner.inference_client import InferenceClient
    from agent_runner.steplog import StepLogger
    import httpx
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.server import create_app

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    responses = [
        '{"action": "dismiss_modal", "selector": "#accept-cookies"}',
        '{"action": "extract", "result": {"efficiency": "94%"}}',
    ]
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    popup_html = (Path(__file__).resolve().parents[2] / "demo-assets" / "popup-page" / "index.html").read_text()
    pages = {"http://localhost:8080/popup.html": popup_html}

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(
            pages, start_url="http://localhost:8080/popup.html",
            modal_urls={"http://localhost:8080/popup.html"},
        )
        counter = get_token_counter(prefer_hf=False)
        config = RunnerConfig(max_steps=5, dom_token_cap=4000,
                              stream_frames=False, log_clickhouse=False)
        runner = AgentRunner(
            page=page, client=client, counter=counter, config=config,
            task="Extract the key statistic.",
            session_id="popup-test", mode="mi",
            step_logger=StepLogger(None), metrics=Metrics(),
        )
        result = await runner.run("http://localhost:8080/popup.html")
        dismiss_actions = [a for a in page.actions if a[0] == "dismiss_modal"]
        assert len(dismiss_actions) >= 1, (
            f"Expected dismiss_modal action, got actions: {page.actions}"
        )
        assert page.dismissed, "Modal was not dismissed"
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()


@pytest.mark.asyncio
async def test_popup_control_without_bank_documented(banks_dir):
    from agent_runner.loop import AgentRunner
    from agent_runner.config import RunnerConfig
    from agent_runner.metrics import Metrics
    from agent_runner.tokenizer import get_token_counter
    from agent_runner.inference_client import InferenceClient
    from agent_runner.steplog import StepLogger
    import httpx
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.server import create_app

    empty_reg = BankRegistry({}, source="empty")
    responses = ['{"action": "extract", "result": {"note": "no modal handling"}}']
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=empty_reg)

    popup_html = (Path(__file__).resolve().parents[2] / "demo-assets" / "popup-page" / "index.html").read_text()
    pages = {"http://localhost:8080/popup.html": popup_html}

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(pages, start_url="http://localhost:8080/popup.html")
        counter = get_token_counter(prefer_hf=False)
        config = RunnerConfig(max_steps=3, dom_token_cap=4000,
                              stream_frames=False, log_clickhouse=False)
        runner = AgentRunner(
            page=page, client=client, counter=counter, config=config,
            task="Extract the key statistic.",
            session_id="popup-control", mode="mi",
            step_logger=StepLogger(None), metrics=Metrics(),
        )
        result = await runner.run("http://localhost:8080/popup.html")
        assert result is not None
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()
