# Agent B1 — Bank Compiler (Offline Pipeline)

**Owner track:** Rahul
**Builds:** `apps/bank-compiler/`
**Reads first:** `docs/handoff/CONTRACTS.md` (§1, §3–§4), paper §3.2 Eq. 6 + "Canonical Pre-RoPE Key Storage" ([arXiv:2605.06225](https://arxiv.org/abs/2605.06225))
**Depends on:** `packages/shared-py` for `bank_io`/`page_key`. If A2 hasn't merged it yet, implement those two modules INSIDE shared-py yourself and tell A2 to adopt them — there must be exactly one implementation.
**No infra dependency — start immediately on any machine with a ≥16GB GPU (or SSH to the EC2 box).**

## Mission

A CLI that turns a webpage into a KV bank: extract DOM → summarize with Haiku → forward-pass the summary through frozen Llama → project hidden states through the model's own K/V weights at the selected layers → write canonical pre-RoPE banks per CONTRACTS §4.

## The pipeline (paper Eq. 6, exactly)

```
1. Playwright: load URL, strip <script>/<style>/comments, capture body HTML + innerText.
2. Haiku (claude-haiku-4-5-20251001): "Describe the structure of this page for a web
   agent: main regions, the interactive elements that matter, where the key data lives,
   and how to act on it. 200-400 words, plain prose." → summary text. Save it
   (banks/<key>.summary.txt) — it goes in the manifest and judges may ask to see it.
3. Wrap summary in the steering template:
   "Internal guidance for navigating this page: {summary}"
4. Tokenize. Forward pass with output_hidden_states=True (bfloat16, no_grad).
   HF indexing: hidden_states[ℓ] is the INPUT to decoder layer ℓ
   (hidden_states[0] = embeddings). Verify this with a hook-based assertion once.
5. Keep only token positions belonging to {summary} (track via offset mapping of the
   template) — drop the wrapper tokens.
6. For each ℓ in SELECTED_LAYERS = [8, 12, 16, 20]:
     h_norm = model.model.layers[ℓ].input_layernorm(hidden_states[ℓ][kept_positions])
     k = k_proj(h_norm)  -> reshape [S, 8, 128] -> transpose to [8, S, 128]
     v = v_proj(h_norm)  -> same
     NO RoPE on keys — canonical pre-RoPE storage (δ=0) is the whole point.
7. Cast to float32, save via shared-py bank_io + append manifest entry
   (page_key, num_slots, dom_structural_hash, summary path).
```

## CLI

```
python -m bank_compiler compile --url https://news.ycombinator.com --page-key hn:front --out banks/
python -m bank_compiler compile --html demo-assets/popup-page/index.html --page-key popup:demo --out banks/
python -m bank_compiler validate banks/          # shapes, dtype, manifest consistency
```

Popup-bank summary note: for `popup:demo`, append one explicit behavioral line to the summary before encoding: *"If a cookie-consent or marketing modal is blocking the page, dismiss it via its accept/close button, then resume the original task."* That sentence is what makes the chaos-test demo work.

## Unit tests (write first)

- `test_hidden_state_indexing`: register a forward pre-hook on layer 8; assert `hidden_states[8]` equals the hook's input. (Guards the #1 likely silent bug.)
- `test_kv_shapes`: tiny prompt → banks shaped `[8, S, 128]` f32, S equals kept-position count, identical S across the 4 layers.
- `test_no_rope_applied`: for a 2-token summary, compare against hand-computed `k_proj(input_layernorm(h))` — exact match (proves no rotary was applied).
- `test_kept_positions`: wrapper tokens excluded; offset mapping correct for a known template.
- `test_summary_call_mocked`: Haiku client mocked; prompt contains stripped DOM, response length-checked.
- `test_roundtrip_via_shared_io`: save → load → `np.array_equal`.

## Definition of done

- `compile` runs end-to-end on the HN front page on your GPU; `validate` passes.
- One bank handed to Peyton's A1 at the **H+4 shape sync** — it must load and measurably change logits via `scripts/prove_injection.py`. Block everything else until this sync passes.

## Suggested skills

`superpowers:test-driven-development`, `everything-claude-code:pytorch-patterns`, `claude-api` (for the Haiku call)
