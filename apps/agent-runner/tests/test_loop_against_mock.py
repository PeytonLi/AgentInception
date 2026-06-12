"""A3 test: full loop vs mock_inference on fixture pages.

Both modes complete the scripted HN task in 2 steps and write 2 step-log rows.
"""

from __future__ import annotations

import json

import pytest

from agent_runner.config import RunnerConfig
from agent_runner.loop import AgentRunner
from agent_runner.metrics import Metrics
from agent_runner.steplog import StepLogger
from agent_runner.tokenizer import HeuristicTokenCounter
from fakes import FakePageDriver


async def _run_mode(mode, pages, client):
    step_logger = StepLogger(None)  # in-memory fake step log
    runner = AgentRunner(
        page=FakePageDriver(pages, "https://news.ycombinator.com/"),
        client=client,
        counter=HeuristicTokenCounter(),
        config=RunnerConfig(stream_frames=False, log_clickhouse=False),
        task="Find the top AI story and extract score + top 3 commenters.",
        session_id=f"loop-{mode}-001",
        mode=mode,
        step_logger=step_logger,
        metrics=Metrics(),
    )
    outcome = await runner.run("https://news.ycombinator.com/")
    return outcome, step_logger


@pytest.mark.asyncio
async def test_mi_loop_completes_task(pages, inference_client):
    client, _app = inference_client
    outcome, step_logger = await _run_mode("mi", pages, client)

    assert outcome.completed is True
    assert outcome.steps == 2
    assert outcome.result == {"score": 312, "top_commenters": ["pg", "dang", "patio11"]}
    assert len(step_logger.rows) == 2

    front, item = step_logger.rows
    assert front["page_key"] == "hn:front"
    assert front["bank_found"] is True
    assert json.loads(front["action_json"])["action"] == "goto"
    assert item["page_key"] == "hn:item"
    assert json.loads(item["action_json"])["action"] == "extract"
    # mi savings: baseline carried far more tokens than visible.
    assert outcome.metrics.cum_baseline > outcome.metrics.cum_visible


@pytest.mark.asyncio
async def test_baseline_loop_completes_task(pages, inference_client):
    client, _app = inference_client
    outcome, step_logger = await _run_mode("baseline", pages, client)

    assert outcome.completed is True
    assert outcome.steps == 2
    assert len(step_logger.rows) == 2
    # baseline: visible == baseline by construction (CONTRACTS s6).
    for row in step_logger.rows:
        assert row["visible_tokens"] == row["baseline_tokens"]
    assert outcome.metrics.cum_visible == outcome.metrics.cum_baseline


@pytest.mark.asyncio
async def test_healthz(inference_client):
    client, _app = inference_client
    health = await client.healthz()
    assert health["status"] == "ok"
    assert "hn:front" in health["banks_loaded"]
