# P2 - Real-Bank MI Validation & Attention Hardening (notes)

Owner: Peyton. Branch `phase2/p2-mi-validation`. Write scope: `apps/inference-engine/`
+ this file. Reads: `docs/handoff/phase-2/peyton/P2-mi-validation.md`, `CONTRACTS.md`
s4-s6, paper Eq. 6-7.

This is the running record for the H+4 shape sync and the `mi_attention.py`
hardening. Numbers that require the GPU box + R1's real bank are marked
**[FILL ON BOX]**; everything else has been validated on CPU in CI.

---

## 1. What P2 added

| Area | Change | Where |
|---|---|---|
| H+4 logit-shift proof (real) | `@pytest.mark.gpu` test: real bank shifts logits (KL > 1e-3); `clear_bank()` restores bit-exact baseline | `tests/test_bank_changes_logits.py::test_real_bank_changes_logits_and_clear_restores` |
| Position independence (real) | `@pytest.mark.gpu` test on Llama dims: same hidden state, two absolute positions -> identical bank scores | `tests/test_position_independence.py::test_real_bank_position_independence` |
| GQA correctness (real) | `@pytest.mark.gpu` test: real bank expands via the live `repeat_kv` factor (8->32, n_rep=4), grouping `q // n_rep` | `tests/test_gqa_expansion.py::test_real_bank_gqa_grouping` |
| Numerical hardening | NaN/inf load-time rejection, large-magnitude warning, forward degenerate-slot guard | `src/inference_engine/mi_attention.py` |
| Hardening regressions | one CPU test per guard | `tests/test_mi_numerical_hardening.py` |
| Proof script | `prove_injection.py --real`: KL both ways, top-k token deltas, bit-exact restore check | `scripts/prove_injection.py` |
| Perf sanity | forward-latency bench, with/without bank | `scripts/bench_mi.py` |

GPU tests auto-skip when `torch.cuda.is_available()` is False (conftest
`pytest_collection_modifyitems`) and skip cleanly (not fail) when R1's bank is
not yet on the box (`real_bank` fixture skips on a missing `page_key`).

---

## 2. H+4 shape sync - procedure

With the engine box up (P1) and R1's first real `hn_front` bank in ClickHouse
(or `banks/`):

```bash
cd apps/inference-engine
# one-shot proof, prints KL + token deltas + clear_bank restore verdict
python scripts/prove_injection.py --real --page-key hn:front \
    --prompt "The top story on Hacker News today is"

# the gated assertion in CI form
pytest -m gpu -q tests/test_bank_changes_logits.py
```

Pass criteria (brief definition of done):
- `KL(base || mi) > 1e-3` and the top token visibly changes.
- `clear_bank()` then re-run -> logits **bit-exact** equal to the no-bank baseline
  (`torch.equal`).

### Result **[FILL ON BOX]**

```
model:           meta-llama/Llama-3.1-8B-Instruct (bf16, sdpa)
bank:            hn:front, <num_slots> slots, layers [8,12,16,20], source <clickhouse|manifest>
prompt:          "The top story on Hacker News today is"
KL(base || mi):  <value>        # must be > 1e-3
KL(mi || base):  <value>
top token:       <id/text>  ->  <id/text>
clear_bank():    <PASS bit-exact | FAIL>
```

---

## 3. Triage checklist (run in this priority order if H+4 fails)

Fix in `mi_attention.py` / `engine.py`, **never by weakening the test**.
Coordinate any contract ambiguity with R1 - do not silently change CONTRACTS s4.

1. **dtype** - bank stored f32, model bf16. Cast happens in `set_bank`
   (`k.to(dtype=q_proj.weight.dtype)`). Confirm the cast point and that the
   bank lands on the model device.
2. **pre-RoPE vs RoPE on bank keys** - bank keys are canonical pre-RoPE (delta=0);
   the bank block scores `q_pre_rope` (the un-rotated query) against `bank_k`.
   `apply_rotary_pos_emb` is out-of-place so `q_pre_rope` stays un-rotated. If the
   compiler applied RoPE to the bank keys, position independence breaks.
3. **repeat_kv GQA grouping** - bank `[8,S,128]` expands to 32 heads via the same
   `repeat_kv(.., num_key_value_groups)` as base attention; query head q reads bank
   KV head `q // 4`. Pinned by `test_real_bank_gqa_grouping`.
4. **slot/seq concat order in the softmax** - one softmax over
   `cat([logits_prompt, logits_bank], dim=-1)`; values `cat([V_cache, V_bank], dim=2)`
   in the same order. Order must match between logits and values.
5. **scaling** - both blocks divide by `sqrt(head_dim)` (= upstream eager scaling
   in transformers 4.46; `LlamaAttention` has no `self.scaling` attribute there).

---

## 4. Hardening fixes (each its own commit + regression test)

All preserve the bank=None pass-through bit-exactly and are no-ops on a healthy
finite bank (the existing logit-shift / position-independence / GQA tests still
pass unchanged).

1. **Load-time NaN/inf rejection.** `set_bank` raises `ValueError` if K or V is
   non-finite, so a corrupt `.bin` is caught at startup instead of silently
   poisoning every step. Regression: `test_set_bank_rejects_non_finite`.
2. **Large-magnitude warning.** `set_bank` warns when peak |value| > 1e4, the
   region where pre-RoPE values risk overflowing bf16 logits - a cheap H+4
   diagnostic for a mis-compiled bank. Regression: `test_large_magnitude_bank_warns`.
3. **Forward degenerate-slot guard.** Bank logits are passed through
   `nan_to_num(.., nan/posinf/neginf = finfo.min)` before the softmax, so a slot
   that overflowed to inf/NaN at matmul time contributes zero weight instead of
   turning the whole softmax row into NaN. No-op on finite logits. Regressions:
   `test_degenerate_all_zero_bank_stays_finite`,
   `test_forward_guard_neutralizes_non_finite_bank_logits`.

Masking was audited and is correct: the bank carries **no** causal/positional
mask (always visible); only the prompt block gets the causal mask (from the HF
`attention_mask` when present, otherwise rebuilt for the sdpa `is_causal` path).

---

## 5. Perf sanity

```bash
cd apps/inference-engine
python scripts/bench_mi.py --real --page-key hn:front --iters 30
```

Forward-pass latency, with vs without the bank at all 4 layers. The bank adds a
fixed `num_slots` to every attended sequence, so the overhead should be small
and roughly constant (independent of prompt length).

### Result **[FILL ON BOX]**

| config | mean | median | p90 |
|---|---|---|---|
| no bank | <ms> | <ms> | <ms> |
| with bank | <ms> | <ms> | <ms> |
| **overhead (median)** | **<+ms / +%>** | | |

CPU tiny-model self-test (sanity that the harness runs, not a real number):
`no bank` ~7.7ms median, `with bank` ~9.0ms median on this dev box.

---

## 6. Validation status (this branch)

- CPU suite: `pytest apps/inference-engine/tests -q` -> 37 passed, 3 gpu skipped.
- `prove_injection.py --tiny`: KL(base||mi) = 0.0153, clear_bank restore PASS.
- GPU tests (`-m gpu`): not run here (no CUDA); they collect and skip cleanly.
- **[FILL ON BOX]** run sections 2 and 5 on the engine box once R1's `hn_front`
  bank is live, paste the KL number + clear_bank verdict into the PR body.
