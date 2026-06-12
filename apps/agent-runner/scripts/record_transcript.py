"""Record agent-runner transcripts for the P3 demo artifacts.

Drives the full loop in both ``mi`` and ``baseline`` modes and writes one
transcript JSON per mode under ``docs/handoff/phase-2/notes/p3-transcripts/``.

Two sources, both reproducible with no GPU and no HF token:

* ``--source mock`` (default) - in-process mock engine + bundled fixture pages.
  Fully offline; the savings are small because the fixtures are tiny.
* ``--source live-dom`` - mock engine, but the page bodies are the REAL live HN
  front + a real comment page. The DOM token counts (and therefore
  ``cum_baseline``) are real, so this is the honest token-savings money shot.

The transcript schema matches the real-engine run in
``tests/test_e2e_real_engine.py`` (which writes ``*-live.json`` on the GPU box),
so P5 can swap the artifacts in place.

    python scripts/record_transcript.py --source mock
    python scripts/record_transcript.py --source live-dom
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # apps/agent-runner/scripts
APP_DIR = HERE.parent  # apps/agent-runner
REPO_ROOT = APP_DIR.parent.parent  # worktree root
SHARED = REPO_ROOT / "packages" / "shared-py"
MOCKS = REPO_ROOT / "tests" / "mocks"
TESTS = APP_DIR / "tests"
FIXTURES = TESTS / "fixtures" / "pages"
OUT_DIR = REPO_ROOT / "docs" / "handoff" / "phase-2" / "notes" / "p3-transcripts"

for path in (APP_DIR, SHARED, MOCKS, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent_runner.bank_slots import load_num_slots_by_page  # noqa: E402
from agent_runner.config import RunnerConfig  # noqa: E402
from agent_runner.loop import AgentRunner  # noqa: E402
from agent_runner.metrics import Metrics, kv_cache_ratio  # noqa: E402
from agent_runner.steplog import StepLogger  # noqa: E402
from agent_runner.tokenizer import get_token_counter  # noqa: E402

TASK = (
    "Find the top story about AI on the Hacker News front page (scan up to 2 "
    "pages), open its comment page, and extract the story score and the top 3 "
    "commenter usernames."
)


def _fixture_pages() -> dict[str, str]:
    def load(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return {
        "https://news.ycombinator.com/": load("hn_front.html"),
        "https://news.ycombinator.com/news?p=2": load("hn_front.html"),
        "https://news.ycombinator.com/item?id=44210000": load("hn_item.html"),
        "http://localhost:8080/popup.html": load("popup.html"),
    }


def _live_dom_pages() -> dict[str, str]:
    """Fetch the real HN front + a real item page for honest token accounting.

    The engine stays mocked (scripted actions), but the DOM token counts the
    runner computes are from the live pages - so cum_baseline reflects the real
    ~10k+ token cost a baseline agent would pay, and the savings are real.
    """
    import re

    import httpx

    front = httpx.get("https://news.ycombinator.com/", timeout=15).text
    m = re.search(r"item\?id=(\d+)", front)
    item_id = m.group(1) if m else "1"
    item = httpx.get(f"https://news.ycombinator.com/item?id={item_id}", timeout=15).text
    return {
        "https://news.ycombinator.com/": front,
        "https://news.ycombinator.com/news?p=2": front,
        # The mock always scripts a goto to this fixed id; map it to a real page.
        "https://news.ycombinator.com/item?id=44210000": item,
    }


async def _run_mock(
    mode: str, num_slots: dict[str, int], pages: dict[str, str] | None = None
) -> dict:
    import httpx
    import mock_inference
    from fakes import FakePageDriver

    from agent_runner.inference_client import InferenceClient

    app = mock_inference.create_app()
    transport = httpx.ASGITransport(app=app)
    http = httpx.AsyncClient(transport=transport, base_url="http://mock")
    client = InferenceClient(base_url="http://mock", http_client=http)
    await client.__aenter__()
    counter = get_token_counter()
    try:
        runner = AgentRunner(
            page=FakePageDriver(
                pages or _fixture_pages(), "https://news.ycombinator.com/"
            ),
            client=client,
            counter=counter,
            config=RunnerConfig(stream_frames=False, log_clickhouse=False),
            task=TASK,
            session_id=f"p3-{mode}-mock",
            mode=mode,
            step_logger=StepLogger(None),
            metrics=Metrics(),
            num_slots_by_page=num_slots,
        )
        outcome = await runner.run("https://news.ycombinator.com/")
    finally:
        await client.__aexit__(None, None, None)
        await http.aclose()
    return _summary(mode, outcome, counter.name)


def _summary(mode: str, outcome, token_backend: str) -> dict:
    m = outcome.metrics
    return {
        "mode": mode,
        "completed": outcome.completed,
        "steps": outcome.steps,
        "result": outcome.result,
        "token_backend": token_backend,
        "cum_visible": m.cum_visible,
        "cum_baseline": m.cum_baseline,
        "kv_savings_ratio": m.kv_savings_ratio,
        "structural_kv_ratio": m.structural_kv_ratio,
        "structural_inputs": {
            "num_layers": 32,
            "t_guidance_peak_dom_tokens": m.peak_dom_tokens,
            "injected_layers": m.injected_layers,
            "s_bank_num_slots": m.bank_slots,
            "formula": "(NUM_LAYERS * T_guidance) / (L_injected * S_bank)",
        },
        "transcript": outcome.transcript,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["mock", "live-dom"], default="mock")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    num_slots = load_num_slots_by_page()
    pages = _live_dom_pages() if args.source == "live-dom" else None
    written = []
    for mode in ("mi", "baseline"):
        summary = asyncio.run(_run_mock(mode, num_slots, pages))
        out = OUT_DIR / f"{mode}-{args.source}.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        written.append((out, summary))
        print(
            f"{mode:8s} steps={summary['steps']} completed={summary['completed']} "
            f"cum_visible={summary['cum_visible']} "
            f"cum_baseline={summary['cum_baseline']} "
            f"observed_ratio={summary['kv_savings_ratio']} "
            f"structural_ratio={summary['structural_kv_ratio']}"
        )

    # Project the structural ratio for the real ~14k-token HN DOM, for the PR.
    projected = kv_cache_ratio(14000, num_slots.get("hn:front", 312), 4)
    print(f"projected structural ratio for 14k-token HN front DOM: {projected}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
