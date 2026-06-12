"""Shared fixtures: a tiny random-weight Llama with the same GQA architecture
(8 query heads : 2 KV heads = 4x repeat, same as 32:8 on the real model).

The MI math is architecture-shape-driven, so it is exercised here on CPU;
the real Llama-3.1-8B-Instruct path is identical code with config-derived sizes.
"""

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
