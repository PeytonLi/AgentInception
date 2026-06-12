"""CLI entrypoint. A3 brief task 1.

    python -m agent_runner --mode=mi \
        --task="Find the top AI story..." \
        --start-url=https://news.ycombinator.com \
        --session-id=demo-001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from .bank_slots import load_num_slots_by_page
from .browser import playwright_session
from .config import RunnerConfig
from .frames import FrameStreamer
from .inference_client import InferenceClient
from .loop import AgentRunner
from .metrics import Metrics
from .steplog import StepLogger
from .tokenizer import get_token_counter

DEFAULT_TASK = (
    "Find the top story about AI on the Hacker News front page (scan up to 2 "
    "pages), open its comment page, and extract the story score and the top 3 "
    "commenter usernames."
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_runner",
        description="GhostBrowser OS agent runner (baseline | mi).",
    )
    p.add_argument("--mode", choices=["baseline", "mi"], required=True)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--start-url", default="https://news.ycombinator.com")
    p.add_argument("--session-id", default=None)
    p.add_argument("--inference-url", default=None)
    p.add_argument("--clickhouse-url", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--headed",
        dest="headless",
        action="store_false",
        default=None,
        help="run a visible browser (default headless)",
    )
    p.add_argument("--no-frames", dest="stream_frames", action="store_false")
    p.add_argument("--no-clickhouse", dest="log_clickhouse", action="store_false")
    p.add_argument(
        "--record-transcript",
        default=None,
        help="write the per-step run transcript (JSON) to this path",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(stream_frames=True, log_clickhouse=True)
    return p


async def _run(args: argparse.Namespace) -> int:
    session_id = args.session_id or f"{args.mode}-{uuid.uuid4().hex[:8]}"
    config = RunnerConfig.from_env(
        inference_url=args.inference_url,
        clickhouse_url=args.clickhouse_url,
        headless=True if args.headless is None else args.headless,
        max_steps=args.max_steps,
        stream_frames=args.stream_frames,
        log_clickhouse=args.log_clickhouse,
    )
    counter = get_token_counter()
    logging.getLogger("agent_runner").info("token backend: %s", counter.name)
    num_slots_by_page = load_num_slots_by_page()
    step_logger = StepLogger.connect(
        config.clickhouse_url, enabled=config.log_clickhouse
    )

    async with (
        playwright_session(headless=config.headless, viewport=config.viewport) as page,
        InferenceClient(config.inference_url) as client,
    ):
        streamer = None
        if config.stream_frames:
            streamer = FrameStreamer(
                page,
                client,
                interval_ms=config.frame_interval_ms,
                quality=config.frame_quality,
            )
            streamer.start()
        runner = AgentRunner(
            page=page,
            client=client,
            counter=counter,
            config=config,
            task=args.task,
            session_id=session_id,
            mode=args.mode,
            step_logger=step_logger,
            metrics=Metrics(),
            num_slots_by_page=num_slots_by_page,
        )
        try:
            outcome = await runner.run(args.start_url)
        finally:
            if streamer is not None:
                await streamer.stop()

    summary = {
        "session_id": session_id,
        "mode": args.mode,
        "completed": outcome.completed,
        "steps": outcome.steps,
        "result": outcome.result,
        "cum_visible": outcome.metrics.cum_visible,
        "cum_baseline": outcome.metrics.cum_baseline,
        "kv_savings_ratio": outcome.metrics.kv_savings_ratio,
        "structural_kv_ratio": outcome.metrics.structural_kv_ratio,
    }
    if args.record_transcript:
        path = Path(args.record_transcript)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **summary,
                    "token_backend": counter.name,
                    "transcript": outcome.transcript,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2))
    return 0 if outcome.completed else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
