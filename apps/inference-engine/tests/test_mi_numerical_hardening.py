"""Regression tests for the P2 numerical hardening of `mi_attention.py`.

Each test pins one guard added during the H+4 triage so it can never silently
regress: (1) a corrupt bank is rejected at load, (2) a degenerate all-zero bank
stays a valid distribution, and (3) a non-finite bank *logit* (e.g. a bf16
overflow on the real model) degrades to zero weight instead of NaN-poisoning the
whole step. They run on CPU with the tiny GQA model, so they stay in CI.
"""

import math

import pytest
import torch

from inference_engine.mi_attention import swap_mi_attention

from conftest import TINY_PATCH_LAYERS


def _last_logits(model, input_ids):
    return model(input_ids).logits[0, -1]


def test_set_bank_rejects_non_finite(tiny_model_factory):
    """A bank carrying NaN/inf must be refused at load, not at inference time."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    attn = patched.model.layers[TINY_PATCH_LAYERS[0]].self_attn

    good = torch.randn(2, 8, 8)

    nan_bank = good.clone()
    nan_bank[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        attn.set_bank(nan_bank, good.clone())

    inf_bank = good.clone()
    inf_bank[1, 3, 2] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        attn.set_bank(good.clone(), inf_bank)


def test_degenerate_all_zero_bank_stays_finite(tiny_model_factory, fixed_input_ids):
    """An all-zero bank is valid (if useless): logits stay finite and the
    next-token distribution remains a proper probability vector."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    zeros = torch.zeros(2, 8, 8)
    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.set_bank(zeros.clone(), zeros.clone())

    logits = _last_logits(patched, fixed_input_ids)
    assert torch.isfinite(logits).all()
    probs = logits.float().softmax(-1)
    assert torch.isfinite(probs).all()
    assert math.isclose(probs.sum().item(), 1.0, abs_tol=1e-4)


def test_forward_guard_neutralizes_non_finite_bank_logits(
    tiny_model_factory, tiny_bank_factory, fixed_input_ids
):
    """Simulate a runtime overflow: inject a non-finite value straight into the
    stored bank key (bypassing set_bank's load-time validation). The forward
    guard must keep the output finite instead of NaN-poisoning the whole step."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    k, v = tiny_bank_factory(num_slots=8)
    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.set_bank(k, v)

    # Corrupt one slot post-load, as a bf16 overflow would at matmul time.
    poisoned = patched.model.layers[TINY_PATCH_LAYERS[0]].self_attn
    poisoned.bank_k[0, 0, 0] = float("inf")
    poisoned.bank_k[1, 1, 1] = float("nan")

    logits = _last_logits(patched, fixed_input_ids)
    assert torch.isfinite(logits).all(), "non-finite bank logit poisoned the step"


def test_repeated_finite_runs_are_bit_exact(
    tiny_model_factory, tiny_bank_factory, fixed_input_ids
):
    """The nan_to_num guard must not perturb a healthy (finite) bank: repeated
    deterministic runs stay bit-exact."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    k, v = tiny_bank_factory(num_slots=16)
    for idx in TINY_PATCH_LAYERS:
        patched.model.layers[idx].self_attn.set_bank(k, v)

    first = _last_logits(patched, fixed_input_ids).clone()
    second = _last_logits(patched, fixed_input_ids)
    assert torch.equal(first, second)


def test_large_magnitude_bank_warns(tiny_model_factory, caplog):
    """A pre-RoPE bank with overflow-risk magnitude logs a load-time warning so
    the H+4 triage can spot a mis-compiled bank early."""
    patched = tiny_model_factory()
    swap_mi_attention(patched, layers=TINY_PATCH_LAYERS)
    attn = patched.model.layers[TINY_PATCH_LAYERS[0]].self_attn

    big = torch.full((2, 8, 8), 5e4)
    with caplog.at_level("WARNING", logger="inference_engine.mi_attention"):
        attn.set_bank(big.clone(), big.clone())
    assert any("peak magnitude" in r.message for r in caplog.records)
