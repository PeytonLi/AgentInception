"""With no bank set, the patched model must be a bit-exact pass-through
(brief: 'test_pass_through_exact', float tolerance 0)."""

import torch

from inference_engine.mi_attention import MIAttention, swap_mi_attention

from conftest import TINY_PATCH_LAYERS


@torch.no_grad()
def test_pass_through_exact(tiny_model_factory, fixed_input_ids):
    base = tiny_model_factory()
    patched = tiny_model_factory()
    swapped = swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    assert sorted(swapped) == TINY_PATCH_LAYERS
    for idx in TINY_PATCH_LAYERS:
        assert isinstance(patched.model.layers[idx].self_attn, MIAttention)

    base_logits = base(fixed_input_ids).logits
    mi_logits = patched(fixed_input_ids).logits

    assert torch.equal(base_logits, mi_logits)  # tolerance 0, not allclose


@torch.no_grad()
def test_pass_through_generation_uses_cache(tiny_model_factory, fixed_input_ids):
    """Greedy generation (which exercises the HF Cache path) must also match exactly."""
    base = tiny_model_factory()
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)

    out_base = base.generate(fixed_input_ids, max_new_tokens=8, do_sample=False)
    out_mi = patched.generate(fixed_input_ids, max_new_tokens=8, do_sample=False)

    assert torch.equal(out_base, out_mi)
