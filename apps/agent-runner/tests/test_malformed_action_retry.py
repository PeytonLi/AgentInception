"""A3 test: malformed action -> one re-prompt; twice -> abort the step."""

from __future__ import annotations

import pytest

from agent_runner.config import RunnerConfig
from agent_runner.loop import AgentRunner
from agent_runner.tokenizer import HeuristicTokenCounter
from fakes import FakePageDriver


def _runner(client, session_id, pages):
    return AgentRunner(
        page=FakePageDriver(pages, "https://news.ycombinator.com/item?id=44210000"),
        client=client,
        counter=HeuristicTokenCounter(),
        config=RunnerConfig(stream_frames=False, log_clickhouse=False),
        task="extract the score",
        session_id=session_id,
        mode="mi",
    )


def _request(runner, step=0):
    return {
        "session_id": runner.session_id,
        "mode": "mi",
        "task": runner.task,
        "url": "https://news.ycombinator.com/item?id=44210000",
        "page_key": "hn:item",
        "dom_text": None,
        "dom_token_count": 1500,
        "history": [],
        "step": step,
        "cum_visible": 0,
        "cum_baseline": 0,
    }


@pytest.mark.asyncio
async def test_malformed_once_recovers_on_reprompt(pages, inference_client):
    client, app = inference_client
    runner = _runner(client, "malformed-once-abc", pages)
    action, resp = await runner._step_with_retry(_request(runner))
    assert action is not None
    assert action.type == "extract"
    assert app.state.calls["malformed-once-abc"] == 2  # original + one reprompt


@pytest.mark.asyncio
async def test_malformed_twice_aborts(pages, inference_client):
    client, app = inference_client
    runner = _runner(client, "malformed-twice-xyz", pages)
    action, resp = await runner._step_with_retry(_request(runner))
    assert action is None
    assert app.state.calls["malformed-twice-xyz"] == 2  # gave up after two
