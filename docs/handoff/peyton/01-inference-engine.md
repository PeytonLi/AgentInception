# Agent A1 — Inference Engine (FastAPI + MI Attention)

**Owner track:** Peyton
**Builds:** `apps/inference-engine/`
**Reads first:** `docs/handoff/CONTRACTS.md` (§1, §4–§8), paper §3.1–3.2 + Appendix G.2 ([arXiv:2605.06225](https://arxiv.org/abs/2605.06225))
**Depends on:** nothing to start (use `tests/fixtures/tiny_bank/`). Real banks at the H+4 shape sync.
**This is the critical-path component. Start it first.**

## Mission

A FastAPI server that loads Llama-3.1-8B-Instruct in bfloat16, replaces the attention modules at layers `[8, 12, 16, 20]` with an MI-augmented attention that can attend over injected KV banks, and serves the `/api/v1/step` + `/ws/events` contract.

## The math you are implementing (do not improvise)

Reference formulation = paper Eq. 2 (augmented cache) + Eq. 7 (canonical pre-RoPE scoring):

- Prompt-side logits, normal RoPE'd attention: `logits_prompt = (q_rope @ K_cache^T) / sqrt(128)` with causal mask.
- Bank-side logits use the **pre-RoPE query** against **pre-RoPE canonical keys** (δ=0): `logits_bank = (q_pre_rope @ K_bank^T) / sqrt(128)`. No mask (bank slots are always visible).
- One softmax over the concatenation: `attn = softmax(cat(logits_prompt, logits_bank))`, output `attn @ cat(V_cache, V_bank)`.
- GQA: banks are stored per KV head `[8, S, 128]`; expand to 32 query heads with the same `repeat_kv` (factor 4) the base model uses.

## Tasks (TDD — write the test before each piece)

1. **Scaffold** `apps/inference-engine/` — `pyproject.toml`/`requirements.txt` (pin `transformers==4.46.*`, `torch`, `fastapi`, `uvicorn`, `clickhouse-connect`, `numpy`), `src/`, `tests/`, `.env.example`.
2. **`MIAttention` module** wrapping a `LlamaAttention` instance: reuses its `q_proj/k_proj/v_proj/o_proj` weights, reimplements forward with the math above, still writes normal-token K/V into the HF `Cache` object so generation works. Bank tensors settable/clearable per request (`set_bank(k, v)` / `clear_bank()`); with no bank set, behavior must be **bit-exact pass-through** (compose from SDPA the same way upstream does).
3. **Model bootstrap**: load model, swap `model.model.layers[ℓ].self_attn` for ℓ in `SELECTED_LAYERS`, smoke-generate.
4. **Bank registry**: at startup, read all banks from ClickHouse via `shared-py.bank_io` into the in-memory dict (CONTRACTS §5). If ClickHouse is unreachable, fall back to reading `banks/manifest.json` + `.bin` files directly — log loudly which path was used.
5. **`/api/v1/step`**: build the prompt per mode (baseline: task + dom_text + history; mi: task + url + history only), set/clear banks by `page_key`, generate (temp 0, max ~256 tokens), parse Action JSON, return per CONTRACTS §6. Token accounting: `visible_tokens` = tokens the engine actually sent to the model this step; `baseline_tokens` = the request's `dom_token_count` + prompt overhead (the runner computes `dom_token_count` in both modes). Maintain per-session cumulative counters and emit the `token_metrics` WS event after each step.
6. **WS hub** `/ws/events`: broadcast `layer_injection`, `token_metrics`, `action`, `log` events at the right moments; rebroadcast `viewport_frame` posted to `POST /internal/frame`.
7. **`/healthz`** per contract.

## Unit tests (write first, in this order)

- `test_pass_through_exact`: no bank set → logits identical to unpatched model (same seed, same prompt) at float tolerance 0.
- `test_bank_changes_logits`: tiny fixture bank set → next-token logits differ measurably (KL > 1e-3) from pass-through.
- `test_bank_position_independence`: same bank, two prompts of different lengths → bank attention scores per head identical for the same query hidden state (the pre-RoPE property).
- `test_gqa_expansion_shapes`: bank `[8, S, 128]` → effective `[32, S, 128]` matches `repeat_kv` semantics.
- `test_step_endpoint_baseline` / `test_step_endpoint_mi` (model mocked): correct prompt assembly, token counts, bank_found logic incl. `page_key="unknown"` fallback.
- `test_ws_event_sequence`: a step emits `layer_injection` then `action` then `token_metrics`.

## Definition of done

- All unit tests green on the EC2 box.
- `curl /healthz` shows model + 1 fixture bank loaded.
- A demo script `scripts/prove_injection.py` prints the same prompt's top-5 next tokens with and without a bank — visibly different. This is the H+4 sync artifact.

## Suggested skills

`superpowers:test-driven-development`, `everything-claude-code:pytorch-patterns`, `superpowers:verification-before-completion`
