"""A set bank must measurably steer next-token logits (brief: KL > 1e-3),
and clearing it must restore bit-exact pass-through."""

import pytest
import torch
from conftest import REAL_PROMPT, TINY_PATCH_LAYERS

from inference_engine.mi_attention import swap_mi_attention


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(softmax(p) || softmax(q)) in nats, both 1-D logit rows."""
    p = p_logits.softmax(-1)
    return (p * (p_logits.log_softmax(-1) - q_logits.log_softmax(-1))).sum().item()


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

    kl = _kl(base_last, mi_last)
    assert kl > 1e-3, f"bank injection barely moved the distribution (KL={kl:.2e})"

    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.clear_bank()
    cleared = patched(fixed_input_ids).logits
    assert torch.equal(cleared, base(fixed_input_ids).logits)


@torch.no_grad()
def test_generation_with_bank_runs(
    tiny_model_factory, tiny_bank_factory, fixed_input_ids
):
    """Banked generation must work through the HF cache (prefill + decode steps)."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    k, v = tiny_bank_factory(num_slots=16)
    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.set_bank(k, v)

    out = patched.generate(fixed_input_ids, max_new_tokens=8, do_sample=False)
    assert out.shape == (1, fixed_input_ids.shape[1] + 8)


@pytest.mark.gpu
@torch.no_grad()
def test_real_bank_changes_logits_and_clear_restores(real_backend, real_bank):
    """H+4 shape sync: a real bank shifts the real model's logits (KL > 1e-3),
    and `clear_bank()` restores a bit-exact baseline (brief: definition of done)."""
    model, tokenizer = real_backend.model, real_backend.tokenizer
    input_ids = tokenizer(REAL_PROMPT, return_tensors="pt").input_ids.to(model.device)

    real_backend.apply_banks(None)
    base_logits = model(input_ids).logits[0, -1].clone()

    injected = real_backend.apply_banks(real_bank)
    assert injected, "no MI layers matched the bank's layer ids"
    mi_logits = model(input_ids).logits[0, -1].clone()

    kl = _kl(base_logits.float(), mi_logits.float())
    assert kl > 1e-3, f"real bank did not steer the logits (KL={kl:.2e})"
    top_base = base_logits.argmax().item()
    top_mi = mi_logits.argmax().item()
    print(
        f"\n[H+4] KL(base||mi)={kl:.4f}  top: {top_base}->{top_mi}  "
        f"injected layers {injected}"
    )

    # Removing the bank must restore the exact pre-injection distribution.
    real_backend.apply_banks(None)
    restored_logits = model(input_ids).logits[0, -1]
    assert torch.equal(restored_logits, base_logits), (
        "clear_bank() did not restore a bit-exact baseline"
    )


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
