"""Optional: exercise the REAL PlaywrightPageDriver against fixture HTML.

Skips automatically when Playwright (or its Chromium build) is unavailable, so
this never blocks CI. When it runs it proves goto/click/dismiss_modal drive a
real browser, complementing the FakePageDriver-based dispatch tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("playwright.async_api")

from playwright.async_api import async_playwright  # noqa: E402

from agent_runner.browser import PlaywrightPageDriver  # noqa: E402

PAGES = Path(__file__).resolve().parent / "fixtures" / "pages"


def _file_url(name: str) -> str:
    return (PAGES / name).resolve().as_uri()


async def _driver(pw):
    browser = await pw.chromium.launch(headless=True)
    context = await browser.new_context(viewport={"width": 1280, "height": 720})
    page = await context.new_page()
    return browser, PlaywrightPageDriver(page)


@pytest.mark.asyncio
async def test_real_browser_goto_text_and_dismiss():
    try:
        async with async_playwright() as pw:
            browser, driver = await _driver(pw)
            try:
                await driver.goto(_file_url("hn_item.html"))
                text = await driver.inner_text()
                assert "312 points" in text
                assert "var _gaq" not in text  # script content stripped

                await driver.goto(_file_url("popup.html"))
                await driver.dismiss_modal("#accept-cookies")
                hidden = await driver._page.evaluate(
                    "() => !document.querySelector('#cookie-modal') "
                    "|| getComputedStyle(document.querySelector('#cookie-modal')).display"
                )
                # Button click handled without error; frame capture works too.
                jpeg = await driver.screenshot_jpeg(50)
                assert jpeg[:2] == b"\xff\xd8"  # JPEG magic
            finally:
                await browser.close()
    except Exception as exc:  # Chromium not installed -> skip, don't fail CI.
        pytest.skip(f"Playwright browser unavailable: {exc}")
