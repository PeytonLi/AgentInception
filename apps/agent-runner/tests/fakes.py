"""In-memory browser double implementing the PageDriver protocol.

Models just enough state to verify the loop and action dispatch without a real
Chromium: navigation, link-driven clicks, and dismissable modals.
"""

from __future__ import annotations

from agent_runner.dom import html_to_text


class FakePageDriver:
    def __init__(
        self,
        pages: dict[str, str],
        start_url: str,
        *,
        link_map: dict[str, str] | None = None,
        valid_selectors: set[str] | None = None,
        modal_urls: set[str] | None = None,
    ) -> None:
        self._pages = pages
        self._url = start_url
        self._link_map = link_map or {}
        self._valid = set(valid_selectors or set()) | set(self._link_map)
        self._modals = set(modal_urls or set())
        self.actions: list[tuple[str, str]] = []
        self.dismissed: set[str] = set()
        self.screenshots = 0

    async def url(self) -> str:
        return self._url

    async def inner_text(self) -> str:
        return html_to_text(self._pages.get(self._url, ""))

    async def goto(self, url: str) -> None:
        self.actions.append(("goto", url))
        self._url = url

    async def click(self, selector: str, timeout_ms: int = 5000) -> None:
        self.actions.append(("click", selector))
        if selector in self._link_map:
            self._url = self._link_map[selector]
        elif selector not in self._valid:
            raise RuntimeError(f"selector not found: {selector!r}")

    async def dismiss_modal(self, selector: str, timeout_ms: int = 5000) -> None:
        self.actions.append(("dismiss_modal", selector))
        if self._url not in self._modals:
            raise RuntimeError(f"no modal to dismiss on {self._url}")
        self._modals.discard(self._url)
        self.dismissed.add(self._url)

    async def settle(self) -> None:
        pass

    async def screenshot_jpeg(self, quality: int = 50) -> bytes:
        self.screenshots += 1
        return b"\xff\xd8\xff\xe0fakejpeg"
