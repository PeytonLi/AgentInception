"""GQA: a bank stored per KV head [8, S, 128] expands to all 32 query heads
with the exact `repeat_kv` semantics the base model uses (factor 4)."""

import torch
from transformers.models.llama.modeling_llama import repeat_kv

from inference_engine.mi_attention import expand_bank


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
