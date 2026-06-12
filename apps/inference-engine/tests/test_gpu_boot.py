"""Real-model GPU smoke test (P1).

The first test that actually loads Llama-3.1-8B-Instruct and runs the real
forward + generate path. It is marked ``gpu`` and skips cleanly when no CUDA
device is present, so CI off-GPU stays green; on the A10G box it proves the
bring-up end to end.

Run on the box:   pytest tests/test_gpu_boot.py -m gpu -s
Skips elsewhere:  no CUDA -> collected, skipped, suite green.

Requires HF_TOKEN in the environment and the model warm in the HF cache
(see docs/handoff/phase-2/notes/p1-box.md).
"""

import pytest
import torch

from inference_engine.config import Settings
from inference_engine.engine import (
    build_messages,
    cuda_memory_summary,
    parse_action_json,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires a CUDA GPU (real Llama load)"
    ),
]

# Keep generation short: this is a liveness smoke, not a quality benchmark.
SMOKE_MAX_NEW_TOKENS = 64


@pytest.fixture(scope="module")
def backend():
    from inference_engine.engine import LlamaBackend

    return LlamaBackend.load(Settings.from_env())


def test_model_loads_on_gpu(backend):
    assert backend.model_loaded is True
    assert backend.device.type == "cuda"
    # Real weights occupy VRAM; ~16 GiB in bf16, but assert only that it is on-GPU.
    assert torch.cuda.memory_allocated() > 1 * 1024**3
    print("\n" + cuda_memory_summary())


def test_generate_returns_action_json(backend):
    """One real generate must yield a parseable Action JSON object (CONTRACTS section 8)."""
    messages = build_messages(
        task="Open the comments page for the top Hacker News story.",
        url="https://news.ycombinator.com/",
        history=["goto https://news.ycombinator.com/"],
        dom_text=(
            "1. A new compiler for latent memory (news.ycombinator.com) "
            "| 312 points | 88 comments item?id=1"
        ),
        latent_context=False,
    )
    raw = backend.generate(messages, max_new_tokens=SMOKE_MAX_NEW_TOKENS)
    assert raw.strip(), "model returned empty output"
    print("\nraw model output:", repr(raw[:200]))

    action = parse_action_json(raw)
    assert action["action"] in {
        "goto",
        "click",
        "dismiss_modal",
        "extract",
        "done",
    }
