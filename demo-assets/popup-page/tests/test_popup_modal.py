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


def test_popup_lifecycle_pending_open_dismissed(http_server):
    """The data-popup-state attribute transitions pending -> open -> dismissed,
    and window.__popupModal.state() mirrors it. This is the deterministic
    signal P3's chaos test keys off, independent of any CSS classes."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            # Intercept the page right at load (before the 300ms timer fires)
            # so we can observe the 'pending' state.
            page.goto(http_server, wait_until="domcontentloaded")

            # Allow the script tag to execute and register window.__popupModal.
            page.wait_for_function("typeof window.__popupModal === 'object'")

            initial_state = page.evaluate("window.__popupModal.state()")
            assert initial_state == "pending", f"expected pending, got {initial_state}"
            assert page.evaluate("window.__popupModal.isOpen()") is False

            # Wait for the scripted 300ms delay to elevate state to 'open'.
            page.wait_for_function(
                "window.__popupModal.state() === 'open'", timeout=2_000
            )
            assert page.evaluate("window.__popupModal.isOpen()") is True
            overlay = page.locator('[data-testid="cookie-modal-overlay"]')
            assert overlay.get_attribute("data-popup-state") == "open"

            # Accept -> dismissed.
            page.locator('[data-testid="cookie-accept"]').click()
            page.wait_for_function(
                "window.__popupModal.state() === 'dismissed'", timeout=1_000
            )
            assert page.evaluate("window.__popupModal.isOpen()") is False
            assert overlay.get_attribute("data-popup-state") == "dismissed"
        finally:
            browser.close()


def test_popup_dismiss_event_fires_with_via(http_server):
    """The 'popup:dismiss' CustomEvent fires with detail.via set to 'accept'
    or 'reject' depending on which button was clicked. P3 uses this to
    confirm the popup bank's action_hint was actually executed."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(http_server)

            # Listen for the custom event on window (it bubbles) and stash the
            # detail on a global so we can poll it from Python.
            page.evaluate(
                """
                () => {
                    window.__lastDismiss = null;
                    window.addEventListener('popup:dismiss', (e) => {
                        window.__lastDismiss = e.detail;
                    });
                }
                """
            )

            overlay = page.locator('[data-testid="cookie-modal-overlay"]')
            expect(overlay).to_be_visible(timeout=2_000)

            page.locator('[data-testid="cookie-reject"]').click()
            page.wait_for_function(
                "window.__lastDismiss && window.__lastDismiss.via === 'reject'",
                timeout=1_000,
            )
            detail = page.evaluate("window.__lastDismiss")
            assert detail == {"via": "reject"}, f"unexpected detail: {detail!r}"
        finally:
            browser.close()


def test_popup_dismiss_is_idempotent(http_server):
    """Calling dismiss() twice does not fire a second event or change state."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(http_server)

            overlay = page.locator('[data-testid="cookie-modal-overlay"]')
            expect(overlay).to_be_visible(timeout=2_000)

            # Track how many dismiss events fire.
            page.evaluate(
                """
                () => {
                    window.__dismissCount = 0;
                    window.addEventListener('popup:dismiss', () => {
                        window.__dismissCount++;
                    });
                }
                """
            )

            # First dismiss.
            page.locator('[data-testid="cookie-accept"]').click()
            page.wait_for_function(
                "window.__popupModal.state() === 'dismissed'", timeout=1_000
            )
            assert page.evaluate("window.__dismissCount") == 1

            # Second dismiss (programmatic) — should be a no-op.
            page.evaluate("window.__popupModal.dismiss('accept')")
            assert page.evaluate("window.__dismissCount") == 1
            assert page.evaluate("window.__popupModal.state()") == "dismissed"
        finally:
            browser.close()
