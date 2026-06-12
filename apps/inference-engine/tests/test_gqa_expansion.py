"""GQA: a bank stored per KV head [8, S, 128] expands to all 32 query heads
with the exact `repeat_kv` semantics the base model uses (factor 4)."""

import pytest
import torch
from transformers.models.llama.modeling_llama import repeat_kv

from inference_engine.config import SELECTED_LAYERS
from inference_engine.mi_attention import expand_bank, mi_layers


def test_gqa_expansion_shapes():
    bank = torch.randn(8, 16, 128)
    out = expand_bank(bank, n_rep=4)
    assert out.shape == (32, 16, 128)

    ref = repeat_kv(bank.unsqueeze(0), 4).squeeze(0)
    assert torch.equal(out, ref)

    # query head q reads KV head q // n_rep — same grouping as the base model
    for q_head in range(32):
        assert torch.equal(out[q_head], bank[q_head // 4])


def test_expand_bank_identity_when_no_grouping():
    bank = torch.randn(4, 5, 16)
    assert torch.equal(expand_bank(bank, n_rep=1), bank)


@pytest.mark.gpu
def test_real_bank_gqa_grouping(real_backend, real_bank):
    """Real model + real bank: a stored [8, S, 128] bank expands via the same
    `repeat_kv` factor the live attention uses, and query head q reads bank KV
    head q // n_rep — identical grouping to base attention on Llama-3.1-8B."""
    model = real_backend.model
    layer = SELECTED_LAYERS[0]
    attn = mi_layers(model)[layer].inner
    n_rep = attn.num_key_value_groups
    assert (attn.num_heads, attn.num_key_value_heads, n_rep) == (32, 8, 4)

    k, _ = real_bank[layer]
    assert k.shape[0] == attn.num_key_value_heads and k.shape[2] == attn.head_dim

    expanded = expand_bank(k, n_rep)
    assert expanded.shape == (attn.num_heads, k.shape[1], attn.head_dim)
    ref = repeat_kv(k.unsqueeze(0), n_rep).squeeze(0)
    assert torch.equal(expanded, ref)
    for q_head in range(attn.num_heads):
        assert torch.equal(expanded[q_head], k[q_head // n_rep])
