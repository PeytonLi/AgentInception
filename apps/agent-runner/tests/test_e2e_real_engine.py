"""P3 tasks 1-2: end-to-end against the REAL engine on live HN.

These are the demo's acceptance tests. They are gated behind ``INFERENCE_URL``
and run a real Chromium via Playwright, so they SKIP on CI (no GPU, no engine,
no browser) and only execute on the P1 box. The mock-based loop tests remain the
CI guarantee; these prove the same loop drives the real engine end to end.

Run on the GPU box:
    INFERENCE_URL=http://<p1-box>:8000 \
        python -m pytest tests/test_e2e_real_engine.py -m gpu -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

TASK = (
    "Find the top story about AI on the Hacker News front page (scan up to 2 "
    "pages), open its comment page, and extract the story score and the top 3 "
    "commenter usernames."
)
START_URL = "https://news.ycombinator.com/"
TRANSCRIPTS = (
    Path(__file__).resolve().parents[3]
    / "docs" / "handoff" / "phase-2" / "notes" / "p3-transcripts"
)


def _inference_url() -> str:
    url = os.environ.get("INFERENCE_URL")
    if not url:
        pytest.skip("INFERENCE_URL not set; real-engine e2e skipped")
    return url


pytest.importorskip("playwright.async_api")


async def _run_real(mode: str, url: str):
    from agent_runner.bank_slots import load_num_slots_by_page
    from agent_runner.browser import playwright_session
    from agent_runner.config import RunnerConfig
    from agent_runner.inference_client import InferenceClient
    from agent_runner.loop import AgentRunner
    from agent_runner.metrics import Metrics
    from agent_runner.tokenizer import get_token_counter

    config = RunnerConfig.from_env(inference_url=url, stream_frames=False)
    async with (
        playwright_session(headless=True, viewport=config.viewport) as page,
        InferenceClient(config.inference_url) as client,
    ):
        health = await client.healthz()
        assert health["status"] == "ok"
        runner = AgentRunner(
            page=page,
            client=client,
            counter=get_token_counter(),
            config=config,
            task=TASK,
            session_id=f"p3-real-{mode}",
            mode=mode,
            metrics=Metrics(),
            num_slots_by_page=load_num_slots_by_page(),
        )
        return await runner.run(START_URL)


def _save(mode: str, outcome) -> None:
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    (TRANSCRIPTS / f"{mode}-live.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "completed": outcome.completed,
                "steps": outcome.steps,
                "result": outcome.result,
                "cum_visible": outcome.metrics.cum_visible,
                "cum_baseline": outcome.metrics.cum_baseline,
                "kv_savings_ratio": outcome.metrics.kv_savings_ratio,
                "structural_kv_ratio": outcome.metrics.structural_kv_ratio,
                "transcript": outcome.transcript,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_mi_completes_live_hn():
    url = _inference_url()
    outcome = await _run_real("mi", url)
    _save("mi", outcome)
    assert outcome.completed is True
    assert outcome.steps <= 15  # terminal action within budget
    # Token honesty (brief task 3): mi visible is tiny vs the DOM baseline.
    assert outcome.metrics.cum_visible < 1500
    assert outcome.metrics.cum_baseline > 10 * outcome.metrics.cum_visible


@pytest.mark.asyncio
async def test_baseline_completes_live_hn_with_no_savings():
    url = _inference_url()
    outcome = await _run_real("baseline", url)
    _save("baseline", outcome)
    assert outcome.completed is True
    # Honesty control: baseline has no savings, visible ~= baseline.
    assert outcome.metrics.cum_visible == pytest.approx(
        outcome.metrics.cum_baseline, rel=0.02
    )
