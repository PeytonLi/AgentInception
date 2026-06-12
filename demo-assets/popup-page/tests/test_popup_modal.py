"""Playwright test for the popup fixture page (B2 spec, test #1).

Asserts:
  - On load, the cookie-consent modal (data-testid="cookie-modal-overlay")
    becomes visible within ~1s of the 300ms scripted delay.
  - Clicking #accept-cookies removes/hides the overlay.
  - The extractable fact (#key-statistic with "94%") is then visible.

Requires:
    pip install pytest-playwright
    playwright install chromium

The page is served by `python -m http.server 8080` from
demo-assets/popup-page/. This fixture starts/stops that server itself so
the test is hermetic.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402

PAGE_DIR = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((HOST, port))
                return
            except OSError:
                time.sleep(0.05)
    raise RuntimeError(f"http.server on port {port} never came up")


@pytest.fixture(scope="module")
def http_server():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", HOST],
        cwd=str(PAGE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port)
        yield f"http://{HOST}:{port}/index.html"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_popup_modal_behavior(http_server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(http_server)

            overlay = page.locator('[data-testid="cookie-modal-overlay"]')
            expect(overlay).to_be_visible(timeout=2_000)

            accept = page.locator("#accept-cookies")
            reject = page.locator("#reject-cookies")
            expect(accept).to_be_visible()
            expect(reject).to_be_visible()

            accept.click()
            expect(overlay).to_be_hidden(timeout=1_000)

            stat = page.locator("#key-statistic")
            expect(stat).to_be_visible()
            assert "94%" in stat.inner_text()
        finally:
            browser.close()


def test_reject_button_also_dismisses(http_server):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(http_server)
            overlay = page.locator('[data-testid="cookie-modal-overlay"]')
            expect(overlay).to_be_visible(timeout=2_000)
            page.locator("#reject-cookies").click()
            expect(overlay).to_be_hidden(timeout=1_000)
        finally:
            browser.close()
