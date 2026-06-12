# W1 Final Wiring Verification Report

**Date:** 2026-06-12
**Agent:** W1 (Peyton)
**Branch:** `phase2/w1-final-wiring`

**Result:** 16/16 checks passed

## Automated Wiring Check Results

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | s1 SELECTED_LAYERS [8,12,16,20] match across shared-py, engine, manifest | PASS | - |
| 2 | s1 NUM_KV_HEADS=8, HEAD_DIM=128, HIDDEN_SIZE=4096 match | PASS | - |
| 3 | s1 BANK_DTYPE='float32' in shared-py | PASS | - |
| 4 | s1 TRANSFORMERS_PIN='transformers==4.46.*' in shared-py | PASS | - |
| 5 | s1 MODEL_ID='meta-llama/Llama-3.1-8B-Instruct' consistent | PASS | - |
| 6 | s3 page_key() all 12 test cases correct | PASS | - |
| 7 | s4 all 3 banks shape [8,S,128] float32, S consistent | PASS | - |
| 8 | s5 ClickHouse schema matches CONTRACTS.md exactly | PASS | - |
| 9 | s6 HTTP API request/response schemas match CONTRACTS.md | PASS | - |
| 10 | s7 all 5 WS event types broadcast to all clients | PASS | - |
| 11 | s8 Action JSON types + 2-strike retry + 502 on failure | PASS | - |
| 12 | s9 ports and env vars correctly wired | PASS | - |
| 13 | s10 all mock files present | PASS | - |
| 14 | s10 shared-py tests pass without GPU | PASS | 38 passed in 2.35s |
| 15 | B run_demo.sh has all startup phases with health polling | PASS | - |
| 16 | D all fallback paths wired in code | PASS | - |

## Summary

- **Passed:** 16
- **Failed:** 0
- **Total:** 16

### Sign-off

- [x] All CONTRACTS.md s1-s10 audit items pass (code-level).
- [x] `run_demo.sh` startup-order dependency chain verified.
- [x] Graceful fallback paths verified in code.
- [x] Mock files present and shared-py tests pass without GPU.

**Gate status: GREEN at code/contract level.**

> **Note:** Live EC2 checks (sections C, E, F of the brief -
> data-flow trace, token honesty on real HN, crash recovery)
> require a GPU box with the full stack running.
> See Live-Run Gate Checklist below.

## Live-Run Gate Checklist (requires EC2 GPU box)

### C. End-to-end data-flow trace
- [ ] DOM capture -> raw HTML on disk
- [ ] Bank compiler -> Haiku summary -> Llama forward -> .bin
- [ ] `upload_banks.py` -> ClickHouse (byte-identical roundtrip)
- [ ] Engine startup -> BankRegistry preloads all 3 page_keys
- [ ] Agent-runner step -> MI attention -> Action JSON -> Playwright
- [ ] WS events: layer_injection -> action -> token_metrics in order
- [ ] Console renders diverging chart series, highlights L8/12/16/20
- [ ] step_logger writes agent_steps ClickHouse row

### E. Token honesty
- [ ] visible_tokens < 500 per mi step
- [ ] baseline_tokens > 10x visible_tokens
- [ ] cum_visible, cum_baseline monotonic
- [ ] kv_savings_ratio matches (NUM_LAYERS*dom_tokens)/(L_injected*S)
- [ ] agent_steps rows sequential, all steps present
- [ ] Frame cadence 250-350ms, no gaps > 2s

### F. Crash recovery
- [ ] Kill uvicorn mid-step -> restart -> runner recovers
- [ ] Kill ClickHouse mid-run -> engine continues, steplog warns

## Appendix: How to run

```bash
# Code-level contract audit (this script):
python scripts/check_wiring.py

# Full integration test suite (CPU):
pytest tests/integration/ -v

# shared-py unit tests (always green, no GPU):
pytest packages/shared-py/tests/ -v
```
