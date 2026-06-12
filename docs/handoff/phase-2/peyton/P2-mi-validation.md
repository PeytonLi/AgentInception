# Agent P2 - Real-Bank MI Validation & Attention Hardening (Peyton)

**Owner:** Peyton  **Branch:** `phase2/p2-mi-validation`  **Worktree:** `.claude/worktrees/phase2-p2`
**Reads first:** `docs/handoff/phase-2/README.md`, `CONTRACTS.md` (s4, s5, s6),
`apps/inference-engine/src/inference_engine/{mi_attention.py,bank_registry.py,engine.py}`,
`apps/inference-engine/scripts/prove_injection.py`,
`apps/inference-engine/tests/{test_bank_changes_logits.py,test_position_independence.py,test_gqa_expansion.py}`,
the paper Eq. 6-7 (arXiv:2605.06225, s3 + Appendix B.1).
**Depends on:** P1 (engine boots on GPU) and R1 (first real `hn_front` bank). This is the
**H+4 shape-sync owner** - the single highest-risk integration in the whole project.
**Unblocks:** P3, P5.

## Mission

Prove, on the real model with a **real** bank, that MI injection does what the paper
says: it measurably steers the logits, it is position-independent, and removing it
restores the exact baseline. Then harden `mi_attention.py` against anything that only
worked because the Phase-1 tests used a tiny random model.

## The H+4 shape sync (do this first, with R1 on a call)

1. Take R1's first real `hn_front` bank (4 layers, `[8, S, 128]` f32, pre-RoPE).
2. Load it into the live engine's `BankRegistry` and run one `/api/v1/step` in `mi`
   mode with a fixed prompt and `temperature=0`.
3. Assert vs. the same step with no bank: `KL(logits_mi || logits_base) > 1e-3`, and
   the top-token distribution visibly changes. Then `clear_bank()` and assert the
   logits are **bit-exact** equal to the no-bank baseline.
4. If it fails, triage in priority order: (a) dtype mismatch (bank f32 vs model bf16
   cast point), (b) pre-RoPE vs RoPE convention on the bank keys, (c) `repeat_kv` GQA
   grouping (8 KV heads -> 32 Q heads), (d) slot/seq concat order in the softmax,
   (e) scaling `1/sqrt(d)`. Fix in `mi_attention.py`/`engine.py`, never by weakening
   the test. Coordinate any contract ambiguity with R1 - do not silently change s4.

## Tasks

1. **Promote the logit-shift test to the real model.** Extend
   `test_bank_changes_logits.py` with a `@pytest.mark.gpu` variant that uses the real
   Llama backend + a real bank fixture (small, checked-in path that R1 provides or a
   downloaded-on-demand fixture). Keep the existing tiny-model test for CI.
2. **Position independence on the real model.** Extend `test_position_independence.py`:
   the same bank attended from different absolute query positions yields identical bank
   logits (the paper's canonical pre-RoPE guarantee).
3. **GQA correctness on the real model.** Extend `test_gqa_expansion.py`: bank K/V
   expansion uses the *same* `repeat_kv` path as base attention; assert head grouping.
4. **Numerical hardening.** Audit `mi_attention.py` for bf16/f32 edge cases, NaN/inf
   guards when a bank slot is degenerate, and correct masking (bank is always visible,
   no causal mask). Add a regression test per fix.
5. **Perf sanity.** Measure per-step latency with vs without bank injection on the real
   model; confirm injection overhead is small and constant. Append to
   `docs/handoff/phase-2/notes/p2-mi-validation.md`.
6. **Generalize `prove_injection.py`** so it can run with `--real` against the live
   model + a real bank, printing KL and top-k token deltas - this becomes the demo's
   "here is the proof" script.

## Definition of done

- H+4 shape sync passes: real bank shifts logits (KL > 1e-3); `clear_bank()` restores
  bit-exact baseline. Documented in `p2-mi-validation.md`.
- GPU-marked tests for logit-shift, position-independence, GQA all green on the box;
  all skip cleanly off-GPU.
- `prove_injection.py --real` prints a convincing before/after for `hn:front`.
- Write scope limited to `apps/inference-engine/` + `docs/handoff/phase-2/notes/`.

## Commit / push / PR

Commit each triage fix separately (great demo narrative). Push `phase2/p2-mi-validation`;
PR title "Phase 2 / P2 - real-bank MI validation". In the PR body, paste the KL number
and the clear_bank bit-exactness result - that is the headline evidence the thesis works.

## Suggested skills

`tdd`, `diagnose` (the shape/dtype/RoPE triage), `git-commit`.
