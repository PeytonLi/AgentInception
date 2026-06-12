"""Memory-Inception attention for Llama (arXiv:2605.06225, Eq. 2 + Eq. 7).

Wraps a stock `LlamaAttention` (any impl: sdpa/eager/flash) at the selected
layers. With no bank set, forward delegates to the wrapped module — bit-exact
pass-through. With a bank set:

  - prompt side keeps exact upstream semantics: RoPE'd q against the RoPE'd
    K cache, causal mask, scaled by 1/sqrt(head_dim);
  - bank side scores the *pre-RoPE* query against canonical pre-RoPE bank
    keys (delta=0) with no mask — bank slots are always visible;
  - one softmax over the concatenation of both logit blocks, output is
    attn @ [V_cache ; V_bank];
  - normal-token K/V still go through the HF Cache so generation works.

Version-sensitive: written against transformers==4.46.* (CONTRACTS §1).
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


def expand_bank(bank: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[num_kv_heads, S, head_dim] -> [num_kv_heads * n_rep, S, head_dim].

    Same grouping as the base model's `repeat_kv`: query head q reads
    bank KV head q // n_rep.
    """
    return repeat_kv(bank.unsqueeze(0), n_rep).squeeze(0)


class MIAttention(nn.Module):
    """Drop-in replacement for `model.model.layers[i].self_attn`."""

    def __init__(self, inner: LlamaAttention):
        super().__init__()
        self.inner = inner
        self.bank_k: Optional[torch.Tensor] = None  # [num_kv_heads, num_slots, head_dim]
        self.bank_v: Optional[torch.Tensor] = None
        # Pre-softmax bank logits for the last query position, [bsz, num_heads, num_slots].
        self.last_bank_scores: Optional[torch.Tensor] = None

    @property
    def layer_idx(self) -> Optional[int]:
        return self.inner.layer_idx

    def set_bank(self, k: torch.Tensor, v: torch.Tensor) -> None:
        a = self.inner
        expected = (a.num_key_value_heads, a.head_dim)
        for name, t in (("k", k), ("v", v)):
            if t.ndim != 3 or (t.shape[0], t.shape[2]) != expected:
                raise ValueError(
                    f"bank {name} must be [num_kv_heads={expected[0]}, num_slots, "
                    f"head_dim={expected[1]}], got {tuple(t.shape)}"
                )
        if k.shape != v.shape:
            raise ValueError(f"bank K shape {tuple(k.shape)} != V shape {tuple(v.shape)}")
        param = a.q_proj.weight  # banks stored float32, cast to model dtype/device at load
        self.bank_k = k.to(device=param.device, dtype=param.dtype)
        self.bank_v = v.to(device=param.device, dtype=param.dtype)

    def clear_bank(self) -> None:
        self.bank_k = None
        self.bank_v = None
        self.last_bank_scores = None

    @property
    def num_slots(self) -> int:
        return 0 if self.bank_k is None else self.bank_k.shape[1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if self.bank_k is None:
            # Bit-exact pass-through: same module object, same code path as unpatched.
            return self.inner(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        a = self.inner
        bsz, q_len, _ = hidden_states.size()

        query_states = a.q_proj(hidden_states).view(bsz, q_len, a.num_heads, a.head_dim).transpose(1, 2)
        key_states = a.k_proj(hidden_states).view(bsz, q_len, a.num_key_value_heads, a.head_dim).transpose(1, 2)
        value_states = a.v_proj(hidden_states).view(bsz, q_len, a.num_key_value_heads, a.head_dim).transpose(1, 2)

        # Canonical pre-RoPE query for the bank block (Eq. 7, delta=0).
        # apply_rotary_pos_emb is out-of-place, so this reference stays un-rotated.
        q_pre_rope = query_states

        if position_embeddings is None:
            cos, sin = a.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, a.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, a.num_key_value_groups)
        value_states = repeat_kv(value_states, a.num_key_value_groups)
        kv_len = key_states.shape[-2]

        # --- prompt-side logits: standard RoPE'd causal attention ---
        logits_prompt = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(a.head_dim)

        if attention_mask is not None:
            logits_prompt = logits_prompt + attention_mask[:, :, :, :kv_len]
        elif q_len > 1:
            # The sdpa code path passes attention_mask=None for unpadded inputs and
            # relies on is_causal=True; rebuild that causal mask here. Query i sits at
            # absolute position cache_position[i] and may attend to cache slots <= it.
            if cache_position is None:
                cache_position = torch.arange(kv_len - q_len, kv_len, device=hidden_states.device)
            kv_idx = torch.arange(kv_len, device=hidden_states.device)
            causal = torch.zeros((q_len, kv_len), dtype=logits_prompt.dtype, device=hidden_states.device)
            causal.masked_fill_(kv_idx[None, :] > cache_position[:, None], torch.finfo(logits_prompt.dtype).min)
            logits_prompt = logits_prompt + causal

        # --- bank-side logits: pre-RoPE q against canonical keys, no mask ---
        bank_k = repeat_kv(self.bank_k.unsqueeze(0), a.num_key_value_groups)  # [1, num_heads, S, head_dim]
        bank_v = repeat_kv(self.bank_v.unsqueeze(0), a.num_key_value_groups)
        logits_bank = torch.matmul(q_pre_rope, bank_k.transpose(2, 3)) / math.sqrt(a.head_dim)
        self.last_bank_scores = logits_bank[:, :, -1, :].float().detach()

        # --- one softmax over [prompt ; bank], fp32 like upstream eager ---
        attn_weights = torch.cat([logits_prompt, logits_bank], dim=-1)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=a.attention_dropout, training=self.training)

        values = torch.cat([value_states, bank_v.expand(bsz, -1, -1, -1)], dim=2)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = a.o_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None), past_key_value


def swap_mi_attention(model, layers: list[int]) -> list[int]:
    """Replace `self_attn` with MIAttention at the given decoder layer indices."""
    swapped: list[int] = []
    decoder_layers = model.model.layers
    for idx in layers:
        if isinstance(decoder_layers[idx].self_attn, MIAttention):
            continue
        decoder_layers[idx].self_attn = MIAttention(decoder_layers[idx].self_attn)
        swapped.append(idx)
    return swapped


def mi_layers(model) -> dict[int, MIAttention]:
    """All MIAttention modules on the model, keyed by decoder layer index."""
    return {
        idx: layer.self_attn
        for idx, layer in enumerate(model.model.layers)
        if isinstance(layer.self_attn, MIAttention)
    }


def set_banks(model, banks: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> list[int]:
    """Set banks on matching MI layers, clear the rest. Returns injected layer indices."""
    injected: list[int] = []
    for idx, attn in mi_layers(model).items():
        if idx in banks:
            attn.set_bank(*banks[idx])
            injected.append(idx)
        else:
            attn.clear_bank()
    return sorted(injected)


def clear_banks(model) -> None:
    for attn in mi_layers(model).values():
        attn.clear_bank()
