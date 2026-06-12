"""Shared fixtures: a tiny random-weight Llama with the same GQA architecture
(8 query heads : 2 KV heads = 4x repeat, same as 32:8 on the real model).

The MI math is architecture-shape-driven, so it is exercised here on CPU;
the real Llama-3.1-8B-Instruct path is identical code with config-derived sizes.
"""

import os

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

TINY_SEED = 42
# 4 decoder layers; we patch [1, 3] (stand-ins for [8, 12, 16, 20] on the 32-layer model)
TINY_PATCH_LAYERS = [1, 3]


def make_tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=503,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA factor 4, same as the real model
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )


def make_tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(TINY_SEED)
    model = LlamaForCausalLM(make_tiny_config())
    model.eval()
    return model


@pytest.fixture()
def tiny_model_factory():
    """Factory: every call returns a model with identical weights (same seed)."""
    return make_tiny_model


@pytest.fixture()
def tiny_bank_factory():
    """Random bank tensors shaped for the tiny model: [kv_heads=2, num_slots, head_dim=8]."""

    def make(num_slots: int = 16, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
        gen = torch.Generator().manual_seed(seed)
        k = torch.randn(2, num_slots, 8, generator=gen)
        v = torch.randn(2, num_slots, 8, generator=gen)
        return k, v

    return make


@pytest.fixture()
def fixed_input_ids() -> torch.Tensor:
    torch.manual_seed(123)
    return torch.randint(low=3, high=500, size=(1, 12))


# --------------------------------------------------------------------------- #
# Real-model (GPU) validation harness - P2.                                    #
#                                                                              #
# Tests marked `@pytest.mark.gpu` exercise the *real* Llama-3.1-8B-Instruct    #
# backend with a *real* compiled bank (the H+4 shape sync). They are the       #
# headline evidence that MI injection works on the production model, but they  #
# need a CUDA box, HF access, and R1's banks - so they auto-skip cleanly       #
# everywhere else. The CPU tiny-model tests above stay the CI workhorse.       #
# --------------------------------------------------------------------------- #

REAL_PROMPT = "The top story on Hacker News right now is"
REAL_PAGE_KEY = os.environ.get("P2_PAGE_KEY", "hn:front")


def pytest_collection_modifyitems(config, items):
    """Skip every `gpu`-marked test when no CUDA device is visible."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device; skipping real-model gpu test")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def real_backend():
    """Load Llama-3.1-8B-Instruct + MI attention once for the whole gpu session."""
    from inference_engine.config import Settings
    from inference_engine.engine import LlamaBackend

    return LlamaBackend.load(Settings.from_env())


@pytest.fixture(scope="session")
def real_bank():
    """Real compiled bank for REAL_PAGE_KEY (ClickHouse, manifest fallback).

    Skips (does not fail) when R1's banks are not yet on the box, so the gpu
    suite degrades to a clean skip rather than a red failure pre-H+4.
    """
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.config import Settings

    settings = Settings.from_env()
    registry = BankRegistry.load(settings.clickhouse_url, settings.banks_dir)
    bank = registry.get(REAL_PAGE_KEY)
    if bank is None:
        pytest.skip(
            f"no real bank for {REAL_PAGE_KEY!r} (have {registry.page_keys}); "
            f"run R1 / upload banks first"
        )
    return bank
