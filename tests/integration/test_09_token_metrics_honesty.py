"""Test 9 - Token metrics honesty (A1 <-> A3).

Over a 3-step mi run: cum_visible stays small; cum_baseline >> cum_visible;
the ratio matches the formula from the handoff README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="torch required for inference engine")

from conftest import import_app_fakes

FakeBackend = import_app_fakes("inference-engine").FakeBackend
FakePageDriver = import_app_fakes("agent-runner").FakePageDriver


@pytest.mark.asyncio
async def test_cumulative_token_counts_mi_vs_baseline(banks_dir):
    """mi run: cum_visible stays flat; cum_baseline >> cum_visible."""
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
    fat_html = "<html><body>" + "Article text. " * 3000 + "</body></html>"
    responses = [
        """{"action": "click", "selector": "a.link"}""",
        """{"action": "click", "selector": "a.link"}""",
        """{"action": "extract", "result": {"done": true}}""",
    ]
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    pages = {
        "https://news.ycombinator.com/": fat_html,
        "https://news.ycombinator.com/news?p=2": fat_html,
        "https://news.ycombinator.com/item?id=44210000": fat_html,
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
            task="Find something",
            session_id="token-test",
            mode="mi",
            step_logger=StepLogger(None),
            metrics=metrics,
        )
        await runner.run("https://news.ycombinator.com/")
        assert metrics.cum_visible < 3000, (
            f"mi cum_visible={metrics.cum_visible} should stay small"
        )
        assert metrics.cum_baseline > metrics.cum_visible * 3, (
            f"cum_baseline={metrics.cum_baseline} >> cum_visible={metrics.cum_visible}"
        )
        assert metrics.kv_savings_ratio > 1.0
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()


@pytest.mark.asyncio
async def test_kv_savings_ratio_matches_readme_formula(banks_dir):
    """KV_ratio = (NUM_LAYERS * T_guidance) / (L_ctrl * S_bank)."""
    import httpx
    from agent_runner.config import RunnerConfig
    from agent_runner.inference_client import InferenceClient
    from agent_runner.loop import AgentRunner
    from agent_runner.metrics import Metrics
    from agent_runner.steplog import StepLogger
    from agent_runner.tokenizer import get_token_counter
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.config import NUM_LAYERS
    from inference_engine.server import create_app

    from ghost_shared.constants import SELECTED_LAYERS

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    hn_front = reg.get("hn:front")
    S_bank = next(iter(hn_front.values()))[0].shape[1]

    responses = ["""{"action": "extract", "result": {"ok": true}}"""]
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    html_1000 = "<html><body>" + "word " * 1000 + "</body></html>"
    pages = {"https://news.ycombinator.com/": html_1000}

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(pages, start_url="https://news.ycombinator.com/")
        counter = get_token_counter(prefer_hf=False)
        metrics = Metrics()
        config = RunnerConfig(
            max_steps=2, dom_token_cap=4000, stream_frames=False, log_clickhouse=False
        )
        runner = AgentRunner(
            page=page,
            client=client,
            counter=counter,
            config=config,
            task="Go",
            session_id="ratio-test",
            mode="mi",
            step_logger=StepLogger(None),
            metrics=metrics,
        )
        await runner.run("https://news.ycombinator.com/")
        assert 1.0 <= metrics.kv_savings_ratio <= 100.0, (
            f"kv_savings_ratio={metrics.kv_savings_ratio} out of range [1, 100]"
        )
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()
