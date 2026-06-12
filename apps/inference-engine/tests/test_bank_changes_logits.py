"""A set bank must measurably steer next-token logits (brief: KL > 1e-3),
and clearing it must restore bit-exact pass-through."""

import torch

from inference_engine.mi_attention import swap_mi_attention

from conftest import TINY_PATCH_LAYERS


@torch.no_grad()
def test_bank_changes_logits(tiny_model_factory, tiny_bank_factory, fixed_input_ids):
    base = tiny_model_factory()
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    k, v = tiny_bank_factory(num_slots=16)
    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.set_bank(k, v)

    base_last = base(fixed_input_ids).logits[0, -1].float()
    mi_last = patched(fixed_input_ids).logits[0, -1].float()

    p_base = base_last.softmax(-1)
    kl = (p_base * (base_last.log_softmax(-1) - mi_last.log_softmax(-1))).sum().item()
    assert kl > 1e-3, f"bank injection barely moved the distribution (KL={kl:.2e})"

    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.clear_bank()
    cleared = patched(fixed_input_ids).logits
    assert torch.equal(cleared, base(fixed_input_ids).logits)


@torch.no_grad()
def test_generation_with_bank_runs(tiny_model_factory, tiny_bank_factory, fixed_input_ids):
    """Banked generation must work through the HF cache (prefill + decode steps)."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    k, v = tiny_bank_factory(num_slots=16)
    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.set_bank(k, v)

    out = patched.generate(fixed_input_ids, max_new_tokens=8, do_sample=False)
    assert out.shape == (1, fixed_input_ids.shape[1] + 8)


def test_set_bank_rejects_bad_shapes(tiny_model_factory):
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    attn = patched.model.layers[TINY_PATCH_LAYERS[0]].self_attn

    import pytest

    with pytest.raises(ValueError):
        attn.set_bank(torch.randn(3, 16, 8), torch.randn(3, 16, 8))  # wrong kv_heads
    with pytest.raises(ValueError):
        attn.set_bank(torch.randn(2, 16, 4), torch.randn(2, 16, 4))  # wrong head_dim
    with pytest.raises(ValueError):
        attn.set_bank(torch.randn(2, 16, 8), torch.randn(2, 15, 8))  # K/V slot mismatch
