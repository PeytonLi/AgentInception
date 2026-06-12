"""Constants from CONTRACTS.md §1 plus env-driven settings (§9)."""

import os
from dataclasses import dataclass

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
SELECTED_LAYERS = [8, 12, 16, 20]  # 0-indexed decoder layers (model has 32)
NUM_LAYERS = 32
NUM_Q_HEADS = 32
NUM_KV_HEADS = 8  # GQA: each KV head serves 4 query heads
HEAD_DIM = 128
HIDDEN_SIZE = 4096

GENERATION_MAX_NEW_TOKENS = 256  # brief task 5: temp 0, max ~256 tokens
DOM_TRUNCATE_TOKENS = 4000  # baseline dom_text is truncated upstream to <= 4000 tokens


# Attention kernel for the wrapped (non-MI) layers. "sdpa" is the default and
# what the MI causal-mask rebuild in mi_attention.py is written against; "eager"
# is the escape hatch if a transformers point-release changes sdpa mask plumbing.
ATTN_IMPLEMENTATION = "sdpa"


@dataclass(frozen=True)
class Settings:
    model_id: str
    clickhouse_url: str
    banks_dir: str
    port: int
    hf_token: str | None
    attn_implementation: str = ATTN_IMPLEMENTATION

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_id=os.environ.get("MODEL_ID", MODEL_ID),
            clickhouse_url=os.environ.get("CLICKHOUSE_URL", "http://localhost:8123"),
            banks_dir=os.environ.get("BANKS_DIR", "../../banks"),
            port=int(os.environ.get("INFERENCE_PORT", "8000")),
            hf_token=os.environ.get("HF_TOKEN"),
            attn_implementation=os.environ.get(
                "ATTN_IMPLEMENTATION", ATTN_IMPLEMENTATION
            ),
        )
