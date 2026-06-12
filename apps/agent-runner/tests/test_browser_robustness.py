"""P3 task 7: PlaywrightPageDriver retries/waits for live-site flakiness.

Drives the real driver against a Playwright-shaped fake page so we can assert
that a transient navigation/click failure is retried and that the driver waits
for an element (and settles the load) before acting - without a real Chromium.
"""

from __future__ import annotations

import pytest

from agent_runner.browser import PlaywrightPageDriver


class _FakePWPage:
    def __init__(self, *, goto_fail_times=0, click_fail_times=0):
        self.url = "about:blank"
        self._goto_fail = goto_fail_times
        self._click_fail = click_fail_times
        self.goto_calls = 0
        self.click_calls = 0
        self.waited_selectors: list[str] = []
        self.load_states: list[str] = []
        self.scrolled = 0

    async def goto(self, url, wait_until=None, timeout=None):
        self.goto_calls += 1
        if self.goto_calls <= self._goto_fail:
            raise TimeoutError("navigation timeout")
        self.url = url

    async def wait_for_selector(self, selector, timeout=None, state=None):
        self.waited_selectors.append(selector)

    async def query_selector(self, selector):
        class _H:
            async def scroll_into_view_if_needed(self_inner, timeout=None):
                pass

        self.scrolled += 1
        return _H()

    async def click(self, selector, timeout=None):
        self.click_calls += 1
        if self.click_calls <= self._click_fail:
            raise RuntimeError("element detached")

    async def wait_for_load_state(self, state, timeout=None):
        self.load_states.append(state)


@pytest.mark.asyncio
async def test_goto_retries_then_succeeds():
    page = _FakePWPage(goto_fail_times=1)
    driver = PlaywrightPageDriver(page)
    await driver.goto("https://news.ycombinator.com/")
    assert page.goto_calls == 2  # one failure + one success
    assert page.url == "https://news.ycombinator.com/"
    assert "networkidle" in page.load_states  # settled after navigation


@pytest.mark.asyncio
async def test_goto_raises_after_exhausting_retries():
    page = _FakePWPage(goto_fail_times=99)
    driver = PlaywrightPageDriver(page)
    with pytest.raises(Exception):
        await driver.goto("https://news.ycombinator.com/")
    assert page.goto_calls >= 2  # retried before giving up


@pytest.mark.asyncio
async def test_click_waits_for_selector_and_scrolls_into_view():
    page = _FakePWPage()
    driver = PlaywrightPageDriver(page)
    await driver.click("a.titlelink")
    assert page.waited_selectors == ["a.titlelink"]
    assert page.scrolled == 1
    assert page.click_calls == 1


@pytest.mark.asyncio
async def test_click_retries_transient_failure():
    page = _FakePWPage(click_fail_times=1)
    driver = PlaywrightPageDriver(page)
    await driver.click("a.commentlink")
    assert page.click_calls == 2  # retried once


@pytest.mark.asyncio
async def test_dismiss_modal_routes_through_robust_click():
    page = _FakePWPage()
    driver = PlaywrightPageDriver(page)
    await driver.dismiss_modal("#accept-cookies")
    assert page.waited_selectors == ["#accept-cookies"]
    assert page.click_calls == 1
