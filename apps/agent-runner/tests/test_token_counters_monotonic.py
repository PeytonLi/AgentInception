"""A3 test: cumulative counters strictly increase; mi grows < 500/step."""

from __future__ import annotations

import pytest

from agent_runner.config import RunnerConfig
from agent_runner.loop import AgentRunner
from agent_runner.metrics import Metrics
from agent_runner.tokenizer import HeuristicTokenCounter
from fakes import FakePageDriver


def test_metrics_record_is_monotonic():
    m = Metrics()
    prev_v = prev_b = 0
    for _ in range(5):
        m.record(200, 14000)
        assert m.cum_visible > prev_v
        assert m.cum_baseline > prev_b
        prev_v, prev_b = m.cum_visible, m.cum_baseline
    assert m.kv_savings_ratio == pytest.approx(70.0, rel=0.01)


@pytest.mark.asyncio
async def test_mi_cum_visible_grows_under_budget(pages, inference_client):
    client, _app = inference_client
    runner = AgentRunner(
        page=FakePageDriver(pages, "https://news.ycombinator.com/"),
        client=client,
        counter=HeuristicTokenCounter(),
        config=RunnerConfig(stream_frames=False, log_clickhouse=False),
        task="Find the top AI story and extract its score and top 3 commenters.",
        session_id="mono-mi-001",
        mode="mi",
        metrics=Metrics(),
    )
    deltas: list[int] = []
    prev = 0

    # Drive the loop step-by-step to inspect per-step growth.
    await runner.page.goto("https://news.ycombinator.com/")
    for step in range(runner.config.max_steps):
        action, _resp, _pk = await runner._run_step(step)
        deltas.append(runner.metrics.cum_visible - prev)
        prev = runner.metrics.cum_visible
        if action is None or action.is_terminal:
            break
        await runner._execute(action)

    assert runner.metrics.steps >= 2
    assert all(d > 0 for d in deltas), "cum_visible must strictly increase"
    assert all(d < 500 for d in deltas), f"mi step grew >= 500 tokens: {deltas}"
    assert runner.metrics.cum_baseline > runner.metrics.cum_visible
