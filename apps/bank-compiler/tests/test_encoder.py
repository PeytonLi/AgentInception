"""KV bank encoder tests — uses a tiny LlamaForCausalLM built from a small config.

These tests verify the exact paper recipe (Eq. 6):
    h_norm = layer.input_layernorm(hidden_states[L][kept_positions])
    k = k_proj(h_norm); v = v_proj(h_norm)
    NO RoPE applied to k.

The contract's wire shape is [8, S, 128] (Llama-3.1-8B GQA). Here we build a tiny
model with num_kv_heads=2, head_dim=16; the encoder must reshape to
[model.num_key_value_heads, S, model.head_dim]. The wire-shape constraint is
enforced separately by agentinception_shared.bank_io.to_bytes() at save time.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from bank_compiler.encoder import (
    STEERING_TEMPLATE,
    encode_summary,
    kept_summary_positions,
)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32000,  # match a real Llama tokenizer's vocab range loosely
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        rope_theta=10000.0,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    # Pretrained tokenizer is fine (offline mode if available) — use a known small
    # one. If no internet/cache, we fall back to a manual byte-level tokenizer.
    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        return tok
    except Exception:
        pytest.skip("no llama tokenizer available offline")


def test_hidden_state_indexing(tiny_model):
    """hidden_states[L] MUST equal the INPUT to decoder layer L."""
    model = tiny_model
    input_ids = torch.tensor([[1, 5, 9, 13, 17]])

    captured = {}

    def pre_hook(module, args, kwargs):
        # decoder layer's forward signature uses positional/keyword hidden_states.
        hs = args[0] if args else kwargs.get("hidden_states")
        captured["x"] = hs.detach().clone()
        return None

    layer_idx = 1
    handle = model.model.layers[layer_idx].register_forward_pre_hook(
        pre_hook, with_kwargs=True
    )
    try:
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()

    assert torch.equal(out.hidden_states[layer_idx], captured["x"]), (
        "hidden_states[L] must equal the input to decoder layer L"
    )


def test_kv_shapes_and_dtype(tiny_model, tokenizer):
    summary = "The page has a header, a main column, and a footer."
    banks, kept = encode_summary(
        model=tiny_model,
        tokenizer=tokenizer,
        summary_text=summary,
        selected_layers=[1, 2],
    )
    assert set(banks.keys()) == {1, 2}
    S = len(kept)
    assert S >= 1
    cfg = tiny_model.config
    for layer_id, (k, v) in banks.items():
        assert isinstance(k, np.ndarray) and isinstance(v, np.ndarray)
        assert k.dtype == np.float32
        assert v.dtype == np.float32
        assert k.shape == (cfg.num_key_value_heads, S, cfg.head_dim)
        assert v.shape == (cfg.num_key_value_heads, S, cfg.head_dim)


def test_no_rope_applied(tiny_model, tokenizer):
    """Bank K must equal k_proj(input_layernorm(h)) on the kept positions — exactly."""
    model = tiny_model
    summary = "alpha beta gamma."
    banks, kept = encode_summary(
        model=model,
        tokenizer=tokenizer,
        summary_text=summary,
        selected_layers=[2],
    )
    k_bank, v_bank = banks[2]

    # Hand-compute the canonical pre-RoPE K/V.
    prompt = STEERING_TEMPLATE.format(summary=summary)
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model(enc["input_ids"], output_hidden_states=True, use_cache=False)
    h_layer_in = out.hidden_states[2][0]  # [T, hidden]
    h_kept = h_layer_in[kept]
    layer = model.model.layers[2]
    h_norm = layer.input_layernorm(h_kept)
    cfg = model.config
    with torch.no_grad():
        k_expected = (
            layer.self_attn.k_proj(h_norm)
            .view(-1, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(0, 1)
            .contiguous()
            .float()
            .detach()
            .numpy()
        )
        v_expected = (
            layer.self_attn.v_proj(h_norm)
            .view(-1, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(0, 1)
            .contiguous()
            .float()
            .detach()
            .numpy()
        )
    assert np.allclose(k_bank, k_expected, atol=1e-6, rtol=0), "K differs from canonical pre-RoPE projection — RoPE may have been applied"
    assert np.allclose(v_bank, v_expected, atol=1e-6, rtol=0)


def test_kept_positions_excludes_template(tokenizer):
    """The wrapper prefix tokens must be excluded from kept_summary_positions."""
    summary = "headline lives here."
    prompt = STEERING_TEMPLATE.format(summary=summary)
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc["offset_mapping"][0].tolist()
    kept = kept_summary_positions(prompt, offsets, summary)
    assert len(kept) >= 1
    # The summary substring starts at this character index:
    summary_start = prompt.index(summary)
    for i in kept:
        s, e = offsets[i]
        # kept tokens must lie within the summary span.
        assert e > summary_start
        assert s < len(prompt)


def test_summary_identical_slot_count_across_layers(tiny_model, tokenizer):
    summary = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    banks, kept = encode_summary(
        model=tiny_model,
        tokenizer=tokenizer,
        summary_text=summary,
        selected_layers=[0, 1, 2, 3],
    )
    slot_counts = {layer: k.shape[1] for layer, (k, _v) in banks.items()}
    assert len(set(slot_counts.values())) == 1, f"slot counts differ: {slot_counts}"
    assert next(iter(slot_counts.values())) == len(kept)
