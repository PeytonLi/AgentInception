"""The pre-RoPE property (paper Eq. 7, delta=0): bank attention scores depend
only on the query's hidden state, not on its absolute position in the prompt."""

import torch

from inference_engine.mi_attention import MIAttention

from conftest import make_tiny_model


@torch.no_grad()
def test_bank_position_independence(tiny_bank_factory):
    model = make_tiny_model()
    attn = MIAttention(model.model.layers[1].self_attn)
    k, v = tiny_bank_factory(num_slots=4)
    attn.set_bank(k, v)
    rotary = model.model.rotary_emb

    torch.manual_seed(9)
    h_long = torch.randn(1, 11, 64)
    h_short = h_long[:, 6:, :].clone()  # same final hidden state, different absolute position

    def bank_scores_for_last_query(h: torch.Tensor) -> torch.Tensor:
        seq_len = h.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        attn(
            hidden_states=h,
            position_embeddings=rotary(h, position_ids),
            cache_position=torch.arange(seq_len),
        )
        assert attn.last_bank_scores is not None
        return attn.last_bank_scores.clone()  # [1, num_heads, num_slots], pre-softmax

    scores_long = bank_scores_for_last_query(h_long)
    scores_short = bank_scores_for_last_query(h_short)

    # Position 10 vs position 4: identical scores per head — RoPE never touches the bank path.
    torch.testing.assert_close(scores_short, scores_long, rtol=0.0, atol=1e-6)
