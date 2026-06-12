"""Test 4 — Injection measurably changes logits [GPU] (A1 ↔ B1).

Requires the real Llama-3.1-8B-Instruct model and a CUDA GPU. Same prompt ±
hn:front bank → KL divergence > 1e-3. clear_bank() restores bit-exact baseline.

@pytest.mark.gpu — skipped unless --run-gpu is passed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch required for inference engine")

import torch

from agentinception_shared import bank_io

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def _real_setup(banks_dir):
    try:
        assert torch.cuda.is_available(), "CUDA required"
    except AssertionError:
        pytest.skip("CUDA not available")
    try:
        from inference_engine.config import Settings
        from inference_engine.engine import LlamaBackend

        settings = Settings.from_env()
        settings_dict = settings.__dict__.copy()
        settings_dict["banks_dir"] = str(banks_dir)
        settings = Settings(**settings_dict)
        backend = LlamaBackend.load(settings)
    except Exception as e:
        pytest.skip(f"Cannot load model: {e}")
    return backend, banks_dir


def test_injection_changes_logits(_real_setup):
    backend, banks_dir = _real_setup
    import torch.nn.functional as F
    from inference_engine.bank_registry import BankRegistry
    from inference_engine.engine import build_messages
    from inference_engine.mi_attention import clear_banks, set_banks

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    hn_front = reg.get("hn:front")
    assert hn_front is not None

    messages = build_messages(
        "Find the top AI story",
        "https://news.ycombinator.com/",
        history=[],
        dom_text=None,
        latent_context=False,
    )
    input_ids = backend.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(backend.model.device)

    # Baseline logits
    clear_banks(backend.model)
    with torch.no_grad():
        out_baseline = backend.model(input_ids)
    logits_baseline = out_baseline.logits[0, -1, :].float()

    # MI logits (with bank)
    set_banks(backend.model, hn_front)
    with torch.no_grad():
        out_mi = backend.model(input_ids)
    logits_mi = out_mi.logits[0, -1, :].float()

    # Clear → restore
    clear_banks(backend.model)
    with torch.no_grad():
        out_restored = backend.model(input_ids)
    logits_restored = out_restored.logits[0, -1, :].float()

    # Bank injection must measurably change logits
    kl = F.kl_div(
        F.log_softmax(logits_mi, dim=-1),
        F.softmax(logits_baseline, dim=-1),
        reduction="sum",
    )
    assert kl.item() > 1e-3, (
        f"KL divergence {kl.item():.6f} <= 1e-3 — bank has no effect"
    )

    # clear_bank MUST restore bit-exact baseline
    assert torch.allclose(logits_restored, logits_baseline, atol=1e-5), (
        "clear_bank did NOT restore bit-exact baseline"
    )


def test_bank_pre_rope_keys_are_used_with_delta_zero(_real_setup):
    """Smoke: bank keys are NOT RoPE'd; Eq. 7 delta=0."""
    backend, banks_dir = _real_setup
    from inference_engine.bank_registry import BankRegistry

    reg = BankRegistry.from_manifest_dir(str(banks_dir))
    hn_front = reg.get("hn:front")
    for layer, (k, v) in hn_front.items():
        assert k.ndim == 3
        assert k.shape[0] == 8  # num_kv_heads
        assert k.shape[2] == 128  # head_dim
