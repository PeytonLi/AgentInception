"""A3 test: click / goto / dismiss_modal mutate page state as expected."""

from __future__ import annotations

import pytest

from agent_runner.actions import parse_action
from agent_runner.config import RunnerConfig
from agent_runner.loop import AgentRunner
from agent_runner.tokenizer import HeuristicTokenCounter
from fakes import FakePageDriver


def _runner(page) -> AgentRunner:
    return AgentRunner(
        page=page,
        client=None,  # _execute never touches the client
        counter=HeuristicTokenCounter(),
        config=RunnerConfig(stream_frames=False, log_clickhouse=False),
        task="t",
        session_id="exec-001",
        mode="mi",
    )


@pytest.mark.asyncio
async def test_goto_navigates():
    page = FakePageDriver({}, "https://news.ycombinator.com/")
    runner = _runner(page)
    await runner._execute(parse_action({"action": "goto", "url": "https://news.ycombinator.com/item?id=1"}))
    assert await page.url() == "https://news.ycombinator.com/item?id=1"


@pytest.mark.asyncio
async def test_click_follows_link():
    page = FakePageDriver(
        {},
        "https://news.ycombinator.com/",
        link_map={"a.morelink": "https://news.ycombinator.com/news?p=2"},
    )
    runner = _runner(page)
    await runner._execute(parse_action({"action": "click", "selector": "a.morelink"}))
    assert await page.url() == "https://news.ycombinator.com/news?p=2"
    assert ("click", "a.morelink") in page.actions


@pytest.mark.asyncio
async def test_dismiss_modal_removes_modal():
    url = "http://localhost:8080/popup.html"
    page = FakePageDriver({}, url, modal_urls={url})
    runner = _runner(page)
    await runner._execute(parse_action({"action": "dismiss_modal", "selector": "#accept-cookies"}))
    assert url in page.dismissed
    # A second dismiss now fails because the modal is gone -> proves state changed.
    with pytest.raises(Exception):
        await page.dismiss_modal("#accept-cookies")


@pytest.mark.asyncio
async def test_click_retries_once_then_aborts():
    page = FakePageDriver({}, "https://news.ycombinator.com/")  # no valid selectors
    runner = _runner(page)
    from agent_runner.errors import ActionExecutionError

    with pytest.raises(ActionExecutionError):
        await runner._execute(parse_action({"action": "click", "selector": "a.missing"}))
    # one initial attempt + one retry = two recorded clicks
    assert page.actions.count(("click", "a.missing")) == 2
