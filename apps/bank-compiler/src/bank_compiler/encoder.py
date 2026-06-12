"""Llama forward pass -> canonical pre-RoPE KV banks.

Paper §3.2, Eq. 6:

    h_norm = layer.input_layernorm( hidden_states[L][kept_positions] )
    k      = k_proj(h_norm)   # NO ROTARY APPLIED  (delta = 0)
    v      = v_proj(h_norm)
    reshape to [num_kv_heads, S, head_dim]; cast to float32.

`hidden_states[L]` in HuggingFace is the INPUT to decoder layer L
(hidden_states[0] = embeddings). The test `test_hidden_state_indexing`
guards this invariant.

Only the token positions belonging to the {summary} placeholder of the
steering template are kept; wrapper tokens are dropped.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ghost_shared.constants import MODEL_ID, SELECTED_LAYERS

# Steering wrapper around the summary (CONTRACTS.md isn't picky about exact
# wording; the brief mandates this phrasing).
STEERING_TEMPLATE = "Internal guidance for navigating this page: {summary}"


# --------------------------------------------------------------------------
# Position bookkeeping
# --------------------------------------------------------------------------
def kept_summary_positions(
    prompt: str,
    offset_mapping: Sequence[tuple[int, int]],
    summary: str,
) -> list[int]:
    """Token indices in `prompt` whose character span lies inside `summary`.

    Tokens with offsets (0, 0) (special tokens like BOS) are excluded. A token
    is kept when its end-offset is past the summary start and its start-offset
    is before the summary end.
    """
    if not summary:
        return []
    start = prompt.find(summary)
    if start < 0:
        raise ValueError("summary not found in prompt — template/tokenization mismatch")
    end = start + len(summary)
    kept: list[int] = []
    for i, span in enumerate(offset_mapping):
        s, e = int(span[0]), int(span[1])
        if s == 0 and e == 0:
            continue  # special token (BOS, etc.)
        if e <= start:
            continue
        if s >= end:
            continue
        kept.append(i)
    return kept


# --------------------------------------------------------------------------
# Model load (real Llama-3.1-8B, for production use)
# --------------------------------------------------------------------------
def load_model_and_tokenizer(model_id: str = MODEL_ID, *, dtype: str = "bfloat16"):
    """Load the Llama model + tokenizer in eval mode. Lazy imports.

    Requires HF_TOKEN env var if the model is gated (Llama-3.1 is).
    """
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    device_map = "auto" if torch.cuda.is_available() else None

    tok = AutoTokenizer.from_pretrained(
        model_id, token=os.environ.get("HF_TOKEN")
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()
    return model, tok


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------
def encode_summary(
    *,
    model,
    tokenizer,
    summary_text: str,
    selected_layers: Sequence[int] | None = None,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], list[int]]:
    """Run one forward pass on the steering-wrapped summary and project hidden
    states at each selected layer into canonical pre-RoPE K/V banks.

    Returns (banks, kept_positions):
      banks[L] = (K, V) each shape ``[num_kv_heads, S, head_dim]`` float32.
      kept_positions is the list of token indices belonging to {summary}.
    """
    import torch  # type: ignore

    layers = list(selected_layers) if selected_layers is not None else list(SELECTED_LAYERS)
    if not layers:
        raise ValueError("selected_layers must be non-empty")

    prompt = STEERING_TEMPLATE.format(summary=summary_text)
    enc = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"][0].tolist()
    kept = kept_summary_positions(prompt, offsets, summary_text)
    if not kept:
        raise RuntimeError(
            "no kept summary positions — tokenizer / template mismatch"
        )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = out.hidden_states  # tuple length num_layers + 1
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    kept_idx = torch.as_tensor(kept, dtype=torch.long, device=device)

    banks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for layer_id in layers:
        if layer_id < 0 or layer_id >= len(hidden_states):
            raise IndexError(
                f"layer_id {layer_id} out of range for model with "
                f"{len(hidden_states) - 1} decoder layers"
            )
        # hidden_states[L] = input to decoder layer L  (paper Eq. 6).
        h_in = hidden_states[layer_id][0].index_select(0, kept_idx)  # [S, hidden]

        layer = model.model.layers[layer_id]
        # Run layernorm/projections in the model's compute dtype, then cast to f32.
        h_norm = layer.input_layernorm(h_in)
        k = (
            layer.self_attn.k_proj(h_norm)
            .view(-1, n_kv, head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        v = (
            layer.self_attn.v_proj(h_norm)
            .view(-1, n_kv, head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        banks[layer_id] = (
            k.detach().to(dtype=torch.float32, device="cpu").numpy(),
            v.detach().to(dtype=torch.float32, device="cpu").numpy(),
        )

    return banks, kept
