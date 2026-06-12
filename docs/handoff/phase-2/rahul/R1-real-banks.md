# Agent R1 - Real Bank Compilation (Rahul) [CRITICAL PATH]

**Owner:** Rahul  **Branch:** `phase2/r1-real-banks`  **Worktree:** `.claude/worktrees/phase2-r1`
**Reads first:** `docs/handoff/phase-2/README.md`, `CONTRACTS.md` (s3, s4, s5),
`apps/bank-compiler/src/bank_compiler/{compiler.py,encoder.py,summarizer.py,dom_extract.py,cli.py}`,
`packages/shared-py/agentinception_shared/bank_io.py`, `scripts/{build_demo_banks.py,upload_banks.py,validate_banks_against_engine.py}`,
`banks/manifest.json`.
**Depends on:** P1 (GPU box + model cache). **Unblocks:** P2 (H+4 shape sync), P3, P5.
**This is the single highest-value task in Phase 2.** The demo banks today are random
noise; nothing about the thesis is real until these exist.

## Mission

Run the (already-built, already-tested) bank compiler against the **real** Llama-3.1-8B
for all 3 demo page types, producing genuine pre-RoPE K/V banks, and replace the
synthetic ones everywhere (disk, manifest, ClickHouse). Hand P2 a real `hn:front` bank
at H+4.

## Tasks

1. **Confirm the compiler runs for real.** On the P1 box, install
   `apps/bank-compiler` (`transformers==4.46.*`), set `HF_TOKEN` + `ANTHROPIC_API_KEY`.
   Do one end-to-end `run_compile()` for `popup:demo` first (smallest, deterministic
   fixture) to shake out `load_model_and_tokenizer()`, the Haiku call, and the
   kept-position slicing in `encoder.py`.
2. **Compile `hn:front` FIRST and hand it to P2** for the H+4 shape sync. Do not wait to
   batch all three - the earliest possible real bank de-risks the whole project.
   Source DOM = a captured fresh HN front-page snapshot (coordinate with R3) so the
   summary and `dom_structural_hash` are honest.
3. **Compile `hn:item` and `popup:demo`.** Use real captured DOM for each. Verify each
   bank: 4 layers, `[8, num_slots, 128]` f32, equal `num_slots` across layers, keys are
   pre-RoPE (input_layernorm -> k_proj/v_proj, no rotary, delta=0), per `CONTRACTS.md` s4.
4. **Validate against shared-py + engine.** `bank_io.load_bank` round-trips each; run
   `scripts/validate_banks_against_engine.py` (bytes + `/healthz`) against the P1 engine.
   Byte length must equal `8 * num_slots * 128 * 4` per file.
5. **Replace the synthetic banks.** Update `banks/manifest.json` with the real
   `num_slots`, real `dom_structural_hash`, real `compiled_at`, and the real summary
   text files. **Mark `scripts/build_demo_banks.py` clearly as a shape-only fallback /
   CI fixture generator** (add a docstring warning) so no one mistakes synthetic banks
   for real ones again.
6. **Upload to ClickHouse.** `scripts/upload_banks.py` -> `latent_memory_banks`; confirm
   idempotent re-upload (matches `scripts/tests/test_upload_idempotent.py`). Engine
   startup preload must then list all 3 real page_keys.
7. **Provenance note.** `docs/handoff/phase-2/notes/r1-banks.md`: which DOM snapshot fed
   each bank, the Haiku summary length, `num_slots` per page, and the upload command.

## Definition of done

- 3 real banks on disk + in ClickHouse, manifest updated, summaries committed.
- `hn:front` delivered to P2 and confirmed to shift logits at H+4.
- `validate_banks_against_engine.py` exits 0 against the P1 engine.
- `build_demo_banks.py` clearly labeled synthetic-only.
- Write scope: `apps/bank-compiler/`, `banks/manifest.json`, `banks/*.summary.txt`,
  `scripts/build_demo_banks.py` (docstring only), `docs/handoff/phase-2/notes/`.
  **The `.bin` blobs are gitignored** - commit only the manifest + summaries; ship the
  blobs via ClickHouse upload (and scp/S3 for backup), noted in the PR.

## Commit / push / PR

Commit per page type. Push `phase2/r1-real-banks`; PR "Phase 2 / R1 - real banks". PR
body: per-bank `num_slots`, where the blobs live, and a one-line "P2 H+4 confirmed: KL=..."
once P2 reports back.

## Suggested skills

`tdd`, `diagnose` (model load / slicing), Anthropic API docs for Haiku, `git-commit`.
