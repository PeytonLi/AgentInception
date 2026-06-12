"""A3 test: viewport streaming captures frames and posts them fire-and-forget."""

from __future__ import annotations

import asyncio

import pytest

from agent_runner.frames import FrameStreamer
from fakes import FakePageDriver


class _RecordingClient:
    def __init__(self) -> None:
        self.frames: list[str] = []

    async def post_frame(self, b64: str) -> None:
        self.frames.append(b64)


@pytest.mark.asyncio
async def test_streamer_captures_and_posts_frames():
    page = FakePageDriver({}, "https://news.ycombinator.com/")
    client = _RecordingClient()
    streamer = FrameStreamer(page, client, interval_ms=50, quality=50)
    streamer.start()
    await asyncio.sleep(0.18)
    await streamer.stop()

    assert page.screenshots >= 2
    assert len(client.frames) >= 2
    # base64 of the fake JPEG bytes
    assert all(isinstance(f, str) and f for f in client.frames)


@pytest.mark.asyncio
async def test_streamer_never_raises_on_capture_error():
    class _BrokenPage(FakePageDriver):
        async def screenshot_jpeg(self, quality: int = 50) -> bytes:
            raise RuntimeError("capture failed")

    page = _BrokenPage({}, "https://news.ycombinator.com/")
    client = _RecordingClient()
    streamer = FrameStreamer(page, client, interval_ms=50)
    streamer.start()
    await asyncio.sleep(0.12)
    await streamer.stop()  # must not raise despite capture errors
    assert client.frames == []
