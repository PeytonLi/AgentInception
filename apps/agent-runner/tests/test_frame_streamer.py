"""A3 test: viewport streaming captures frames and posts them fire-and-forget."""

from __future__ import annotations

import asyncio

import pytest
from fakes import FakePageDriver

from agent_runner.frames import FrameStreamer


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
async def test_streamer_uses_configured_quality_and_cadence():
    # P3 task 4: frames captured at JPEG quality 50, ~300 ms cadence.
    qualities: list[int] = []

    class _QualityPage(FakePageDriver):
        async def screenshot_jpeg(self, quality: int = 50) -> bytes:
            qualities.append(quality)
            return b"\xff\xd8\xff\xe0fakejpeg"

    page = _QualityPage({}, "https://news.ycombinator.com/")
    client = _RecordingClient()
    streamer = FrameStreamer(page, client, interval_ms=300, quality=50)
    streamer.start()
    await asyncio.sleep(0.35)
    await streamer.stop()

    assert qualities and all(q == 50 for q in qualities)
    # ~300 ms cadence over ~350 ms => roughly 1-2 frames, never a tight spin.
    assert 1 <= len(client.frames) <= 3


def test_default_frame_config_matches_contract():
    # CONTRACTS s7 + A3 brief: 300 ms cadence, quality 50, viewport 1280x720.
    from agent_runner.config import RunnerConfig

    cfg = RunnerConfig()
    assert cfg.frame_interval_ms == 300
    assert cfg.frame_quality == 50
    assert cfg.viewport == (1280, 720)


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
