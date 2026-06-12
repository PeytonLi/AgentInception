"""Runtime configuration for the agent runner.

Defaults follow CONTRACTS.md s9 (ports/env) and the A3 brief (viewport
1280x720, max 15 steps, frames every 300 ms at JPEG quality 50, DOM truncated
to ~4000 Llama tokens).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_INFERENCE_URL = "http://localhost:8000"
DEFAULT_VIEWPORT: tuple[int, int] = (1280, 720)

MAX_STEPS = 15
DOM_TOKEN_CAP = 4000
FRAME_INTERVAL_MS = 300
FRAME_JPEG_QUALITY = 50

# mi-mode visible-token budget. The A3 brief requires mi cum_visible to grow by
# < 500 tokens/step; we warn if a single step exceeds this.
MI_VISIBLE_BUDGET = 500


@dataclass
class RunnerConfig:
    """All knobs for one run. Build via :meth:`from_env` then override."""

    inference_url: str = DEFAULT_INFERENCE_URL
    clickhouse_url: str | None = None
    headless: bool = True
    max_steps: int = MAX_STEPS
    dom_token_cap: int = DOM_TOKEN_CAP
    frame_interval_ms: int = FRAME_INTERVAL_MS
    frame_quality: int = FRAME_JPEG_QUALITY
    viewport: tuple[int, int] = DEFAULT_VIEWPORT
    stream_frames: bool = True
    log_clickhouse: bool = True

    @classmethod
    def from_env(cls, **overrides: object) -> "RunnerConfig":
        base: dict[str, object] = {
            "inference_url": os.environ.get(
                "INFERENCE_URL", DEFAULT_INFERENCE_URL
            ),
            "clickhouse_url": os.environ.get("CLICKHOUSE_URL"),
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)  # type: ignore[arg-type]
