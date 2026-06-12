"""Test bootstrap: wire up sys.path for the monorepo layout and shared fixtures.

No installs required - we add the agent-runner package, shared-py, and the
mock-inference module to sys.path so tests run from a bare checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pytest_asyncio

HERE = Path(__file__).resolve().parent  # apps/agent-runner/tests
APP_DIR = HERE.parent  # apps/agent-runner
REPO_ROOT = APP_DIR.parent.parent  # worktree root
SHARED = REPO_ROOT / "packages" / "shared-py"
MOCKS = REPO_ROOT / "tests" / "mocks"
FIXTURE_PAGES = HERE / "fixtures" / "pages"

for path in (APP_DIR, SHARED, MOCKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_page(name: str) -> str:
    return (FIXTURE_PAGES / name).read_text(encoding="utf-8")


@pytest.fixture
def pages() -> dict[str, str]:
    """fixture-page HTML keyed by the URL the runner will see."""
    return {
        "https://news.ycombinator.com/": load_page("hn_front.html"),
        "https://news.ycombinator.com/news?p=2": load_page("hn_front.html"),
        "https://news.ycombinator.com/item?id=44210000": load_page("hn_item.html"),
        "http://localhost:8080/popup.html": load_page("popup.html"),
    }


@pytest_asyncio.fixture
async def inference_client():
    """An entered InferenceClient backed by a fresh in-process mock app.

    Yields ``(client, app)`` so tests can inspect ``app.state.calls``.
    """
    import httpx
    import mock_inference
    from agent_runner.inference_client import InferenceClient

    app = mock_inference.create_app()
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://mock")
    client = InferenceClient(base_url="http://mock", http_client=http)
    await client.__aenter__()
    try:
        yield client, app
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()
