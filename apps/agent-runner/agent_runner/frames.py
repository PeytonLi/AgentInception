"""Viewport frame streaming (A3 brief task 4).

A background task screenshots the page every ``interval_ms`` and POSTs the
base64 JPEG to the engine's /internal/frame, which rebroadcasts it on the WS.
Strictly fire-and-forget: any error is swallowed so the loop never blocks on a
frame.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

logger = logging.getLogger("agent_runner.frames")


class FrameStreamer:
    def __init__(
        self,
        page: Any,
        client: Any,
        *,
        interval_ms: int = 300,
        quality: int = 50,
    ) -> None:
        self._page = page
        self._client = client
        self._interval = max(interval_ms, 50) / 1000.0
        self._quality = quality
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                jpeg = await self._page.screenshot_jpeg(self._quality)
                b64 = base64.b64encode(jpeg).decode("ascii")
                await self._client.post_frame(b64)
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("frame capture skipped: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
