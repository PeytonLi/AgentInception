"""Browser abstraction.

The loop talks to a :class:`PageDriver` so it can run against a real Chromium
(via Playwright) in the demo, or against an in-memory fake in tests. Playwright
is imported lazily so this module loads with no browser installed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Protocol, runtime_checkable

from .dom import html_to_text


@runtime_checkable
class PageDriver(Protocol):
    async def url(self) -> str: ...

    async def inner_text(self) -> str: ...

    async def goto(self, url: str) -> None: ...

    async def click(self, selector: str, timeout_ms: int = 5000) -> None: ...

    async def dismiss_modal(
        self, selector: str, timeout_ms: int = 5000
    ) -> None: ...

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
        await self._page.goto(url, wait_until="domcontentloaded")

    async def click(self, selector: str, timeout_ms: int = 5000) -> None:
        await self._page.click(selector, timeout=timeout_ms)

    async def dismiss_modal(self, selector: str, timeout_ms: int = 5000) -> None:
        await self._page.click(selector, timeout=timeout_ms)

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
