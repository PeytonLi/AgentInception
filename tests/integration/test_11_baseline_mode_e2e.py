"""Test 11 - Baseline mode end-to-end [GPU] (A3 <-> A1).

@pytest.mark.gpu
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

pytest.importorskip("torch", reason="torch required for inference engine")

from conftest import import_app_fakes

FakeBackend = import_app_fakes("inference-engine").FakeBackend
FakePageDriver = import_app_fakes("agent-runner").FakePageDriver


@pytest.mark.asyncio
async def test_baseline_mode_cum_visible_equals_cum_baseline(banks_dir):
    import httpx
    from agent_runner.config import RunnerConfig
    from agent_runner.inference_client import InferenceClient
    from agent_runner.loop import AgentRunner
    from agent_runner.metrics import Metrics
    from agent_runner.steplog import StepLogger
    from agent_runner.tokenizer import get_token_counter
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.server import create_app

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    responses = [
        """{"action": "click", "selector": "a.link"}""",
        """{"action": "extract", "result": {"score": 42, "top_commenters": ["a"]}}""",
    ]
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    fat_html = "<html><body>" + "Article text. " * 2000 + "</body></html>"
    pages = {
        "https://news.ycombinator.com/": fat_html,
        "https://news.ycombinator.com/news?p=2": fat_html,
    }

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(
            pages,
            start_url="https://news.ycombinator.com/",
            link_map={"a.link": "https://news.ycombinator.com/news?p=2"},
            valid_selectors={"a.link"},
        )
        counter = get_token_counter(prefer_hf=False)
        metrics = Metrics()
        config = RunnerConfig(
            max_steps=5, dom_token_cap=4000, stream_frames=False, log_clickhouse=False
        )
        runner = AgentRunner(
            page=page,
            client=client,
            counter=counter,
            config=config,
            task="Extract story score",
            session_id="baseline-test",
            mode="baseline",
            step_logger=StepLogger(None),
            metrics=metrics,
        )
        result = await runner.run("https://news.ycombinator.com/")
        assert result is not None
        assert metrics.cum_visible > 0
        assert metrics.cum_baseline > 0
        assert metrics.cum_visible == metrics.cum_baseline, (
            f"baseline mode: cum_visible={metrics.cum_visible} should equal "
            f"cum_baseline={metrics.cum_baseline}"
        )
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()


@pytest.mark.asyncio
async def test_baseline_mode_never_injects_banks(banks_dir):
    import httpx
    from agent_runner.config import RunnerConfig
    from agent_runner.inference_client import InferenceClient
    from agent_runner.loop import AgentRunner
    from agent_runner.metrics import Metrics
    from agent_runner.steplog import StepLogger
    from agent_runner.tokenizer import get_token_counter
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.server import create_app

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    responses = ["""{"action": "extract", "result": {"ok": true}}"""]
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    html = "<html><body>HN Front Page</body></html>"
    pages = {"https://news.ycombinator.com/": html}

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(pages, start_url="https://news.ycombinator.com/")
        counter = get_token_counter(prefer_hf=False)
        config = RunnerConfig(
            max_steps=2, dom_token_cap=4000, stream_frames=False, log_clickhouse=False
        )
        runner = AgentRunner(
            page=page,
            client=client,
            counter=counter,
            config=config,
            task="Go",
            session_id="baseline-no-bank",
            mode="baseline",
            step_logger=StepLogger(None),
            metrics=Metrics(),
        )
        await runner.run("https://news.ycombinator.com/")
        for row in runner.step_logger.rows:
            assert row["bank_found"] is False, (
                f"baseline step should not find banks: {row}"
            )
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()
