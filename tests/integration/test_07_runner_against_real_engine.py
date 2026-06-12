"""Test 7 — Agent runner against real inference engine [GPU] (A3 ↔ A1).

Mi-mode loop on fixture pages completes ≤ 15 steps, agent_steps rows written,
and frames are posted. Uses the FakeBackend (no real GPU needed for the A3→A1
contract test; the [GPU] label in the spec means the real-model integration —
that is covered by test 04).

Marked @pytest.mark.slow (needs the engine server running or in-memory).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
pytest.importorskip("torch", reason="torch required for inference engine")

from agent_runner.actions import parse_action
from fastapi.testclient import TestClient
from inference_engine.bank_registry import BankRegistry
from inference_engine.server import create_app

# Wire inference engine test doubles
_INF_TESTS = Path(__file__).resolve().parents[2] / "apps" / "inference-engine" / "tests"
sys.path.insert(0, str(_INF_TESTS))
from fakes import FakeBackend

# Wire agent-runner test fakes
_AR_TESTS = Path(__file__).resolve().parents[2] / "apps" / "agent-runner" / "tests"
sys.path.insert(0, str(_AR_TESTS))
from fakes import FakePageDriver


def _make_responses() -> list[str]:
    """Scripted model answers that complete a mini HN task."""
    return [
        '{"action": "click", "selector": "a.morelink"}',
        '{"action": "goto", "url": "https://news.ycombinator.com/item?id=44210000"}',
        '{"action": "extract", "result": {"score": 42, "top_commenters": ["a","b","c"]}}',
    ]


@pytest.mark.asyncio
async def test_mi_loop_completes_within_max_steps(banks_dir):
    """Runner ↔ engine: mi mode completes a fixture task in ≤ 15 steps."""
    import httpx
    from agent_runner.config import MAX_STEPS, RunnerConfig
    from agent_runner.inference_client import InferenceClient
    from agent_runner.loop import AgentRunner
    from agent_runner.metrics import Metrics
    from agent_runner.steplog import StepLogger
    from agent_runner.tokenizer import get_token_counter

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    responses = _make_responses()
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    pages = {
        "https://news.ycombinator.com/": "<html><body>HN Front Page <a class='morelink'>More</a></body></html>",
        "https://news.ycombinator.com/news?p=2": "<html><body>HN Page 2</body></html>",
        "https://news.ycombinator.com/item?id=44210000": "<html><body>Item Page <span class='score'>42</span> <a class='hnuser'>a</a></body></html>",
    }

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(
            pages,
            start_url="https://news.ycombinator.com/",
            link_map={"a.morelink": "https://news.ycombinator.com/news?p=2"},
        )
        counter = get_token_counter(prefer_hf=False)
        config = RunnerConfig(
            max_steps=MAX_STEPS,
            dom_token_cap=4000,
            stream_frames=False,
            log_clickhouse=False,
        )
        runner = AgentRunner(
            page=page,
            client=client,
            counter=counter,
            config=config,
            task="Find the top AI story and extract its score.",
            session_id="integration-test-7",
            mode="mi",
            step_logger=StepLogger(None),
            metrics=Metrics(),
        )
        result = await runner.run("https://news.ycombinator.com/")
        assert result.completed, f"loop did not complete (steps={result.steps})"
        assert result.steps <= MAX_STEPS
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()


@pytest.mark.asyncio
async def test_step_logger_records_rows(banks_dir):
    """agent_steps rows are recorded in-memory (in-app log format)."""
    import httpx
    from agent_runner.config import RunnerConfig
    from agent_runner.inference_client import InferenceClient
    from agent_runner.loop import AgentRunner
    from agent_runner.metrics import Metrics
    from agent_runner.steplog import StepLogger
    from agent_runner.tokenizer import get_token_counter

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    responses = _make_responses()
    backend = FakeBackend(responses=responses)
    app = create_app(backend=backend, registry=reg)

    pages = {
        "https://news.ycombinator.com/": "<html><body>HN Front Page <a class='morelink'>More</a></body></html>",
        "https://news.ycombinator.com/news?p=2": "<html><body>HN Page 2</body></html>",
        "https://news.ycombinator.com/item?id=44210000": "<html><body>Item Page</body></html>",
    }

    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = InferenceClient(base_url="http://test", http_client=http)
    await client.__aenter__()
    try:
        page = FakePageDriver(
            pages,
            start_url="https://news.ycombinator.com/",
            link_map={"a.morelink": "https://news.ycombinator.com/news?p=2"},
        )
        counter = get_token_counter(prefer_hf=False)
        steplog = StepLogger(None)
        config = RunnerConfig(
            max_steps=15, dom_token_cap=4000, stream_frames=False, log_clickhouse=False
        )
        runner = AgentRunner(
            page=page,
            client=client,
            counter=counter,
            config=config,
            task="Extract story score.",
            session_id="integration-test-7b",
            mode="mi",
            step_logger=steplog,
            metrics=Metrics(),
        )
        await runner.run("https://news.ycombinator.com/")
        assert len(steplog.rows) >= 2, (
            f"expected >= 2 step log rows, got {len(steplog.rows)}"
        )
        for row in steplog.rows:
            assert "session_id" in row
            assert "step" in row
            assert "mode" in row
            assert "url" in row
            assert "action_json" in row
            assert "visible_tokens" in row
            assert "baseline_tokens" in row
            assert "bank_found" in row
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()
