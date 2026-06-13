"""Browser abstraction.

The loop talks to a :class:`PageDriver` so it can run against a real Chromium
(via Playwright) in the demo, or against an in-memory fake in tests. Playwright
is imported lazily so this module loads with no browser installed.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol, runtime_checkable

from .dom import html_to_text

logger = logging.getLogger("agent_runner.browser")

# Live HN is occasionally slow/flaky; navigations and clicks get a couple of
# bounded retries before we surface the error to the loop.
NAV_TIMEOUT_MS = 15000
NAV_RETRIES = 2
CLICK_RETRIES = 2


@runtime_checkable
class PageDriver(Protocol):
    async def url(self) -> str: ...

    async def inner_text(self) -> str: ...

    async def goto(self, url: str) -> None: ...

    async def click(self, selector: str, timeout_ms: int = 5000) -> None: ...

    async def dismiss_modal(self, selector: str, timeout_ms: int = 5000) -> None: ...

    async def settle(self) -> None: ...

    async def screenshot_jpeg(self, quality: int = 50) -> bytes: ...


class PlaywrightPageDriver:
    """Wraps a Playwright async ``Page``."""

    def __init__(self, page: object) -> None:
        self._page = page

    async def url(self) -> str:
        return self._page.url

    async def inner_text(self) -> str:
        try:
            return await self._page.inner_text("body")
        except Exception:
            html = await self._page.content()
            return html_to_text(html)

    async def goto(self, url: str) -> None:
        """Navigate with bounded retries for live-site navigation timeouts."""
        last_exc: Exception | None = None
        for attempt in range(NAV_RETRIES + 1):
            try:
                await self._page.goto(
                    url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS
                )
                await self.settle()
                return
            except Exception as exc:  # navigation timeout / transient net error
                last_exc = exc
                logger.warning(
                    "goto %s failed (attempt %s/%s): %s",
                    url,
                    attempt + 1,
                    NAV_RETRIES + 1,
                    exc,
                )
                await asyncio.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def click(self, selector: str, timeout_ms: int = 5000) -> None:
        """Click after waiting for the element; scroll it into view first.

        Live HN renders title and comment links separately; the model supplies
        the discriminating selector and we make sure the element is actually
        present and in view before clicking so a lazy layout does not miss it.
        """
        last_exc: Exception | None = None
        for attempt in range(CLICK_RETRIES + 1):
            try:
                await self._page.wait_for_selector(
                    selector, timeout=timeout_ms, state="visible"
                )
                try:
                    handle = await self._page.query_selector(selector)
                    if handle is not None:
                        await handle.scroll_into_view_if_needed(timeout=timeout_ms)
                except Exception:  # scroll is best-effort
                    pass
                await self._page.click(selector, timeout=timeout_ms)
                await self.settle()
                return
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "click %s failed (attempt %s/%s): %s",
                    selector,
                    attempt + 1,
                    CLICK_RETRIES + 1,
                    exc,
                )
                await asyncio.sleep(0.3 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def dismiss_modal(self, selector: str, timeout_ms: int = 5000) -> None:
        # Known popup selectors tried first (model often guesses wrong).
        known = [
            "#accept-cookies",
            "#reject-cookies",
            "[data-testid=cookie-accept]",
            "[data-testid=cookie-reject]",
            ".cookie-buttons button",
        ]
        for sel in known:
            try:
                await self.click(sel, timeout_ms=min(timeout_ms, 2000))
                return
            except Exception:
                pass
        await self.click(selector, timeout_ms=timeout_ms)

    async def settle(self) -> None:
        """Best-effort wait for the network to go idle after a navigation."""
        try:
            await self._page.wait_for_load_state("networkidle", timeout=3000)
        except Exception:
            pass

    async def screenshot_jpeg(self, quality: int = 50) -> bytes:
        return await self._page.screenshot(type="jpeg", quality=quality)


@asynccontextmanager
async def playwright_session(
    *, headless: bool = True, viewport: tuple[int, int] = (1280, 720)
) -> AsyncIterator[PlaywrightPageDriver]:
    """Launch Chromium and yield a :class:`PlaywrightPageDriver`."""
    from playwright.async_api import async_playwright  # lazy import

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(
            viewport={"width": viewport[0], "height": viewport[1]}
        )
        page = await context.new_page()
        try:
            yield PlaywrightPageDriver(page)
        finally:
            await context.close()
            await browser.close()
