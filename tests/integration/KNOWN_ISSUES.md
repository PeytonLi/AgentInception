# KNOWN ISSUES - Agent C1 Integration Test Suite

Last updated: 2026-06-12  (P5 integration pass)

## Test Status

| # | Test | Status |
|---|------|--------|
| 1 | Bank binary contract | GREEN (6/6) |
| 2 | ClickHouse roundtrip | GREEN (2/2) with --run-slow |
| 3 | Engine startup preload | GREEN (3/3) |
| 4 | Injection changes logits | SKIP (needs real CUDA + 8B model) |
| 5 | Unknown page fallback | GREEN (3/3) |
| 6 | WS event flow | GREEN (1/1) |
| 7 | Runner against engine | GREEN (2/2) with --run-slow |
| 8 | Popup chaos flow | GREEN (2/2) with --run-gpu |
| 9 | Token metrics honesty | GREEN (2/2) |
| 10 | Console renders live run | GREEN (5/5) with --run-slow |
| 11 | Baseline mode e2e | GREEN (2/2) with --run-gpu |

**Summary:** 28 passed, 3 skipped (test_04 x2 CUDA gated, test_10 vitest requires pnpm on PATH).
All FakeBackend/FakePageDriver tests exercise the full A1-A3-A4 contract
paths on CPU; the only genuinely GPU-gated verification is test_04.

## How to run

```bash
# Fast (torch-free):
pytest tests/integration/ -v

# With --run-slow (ClickHouse + async tests):
pytest tests/integration/ -v --run-slow

# With --run-gpu (attempt CUDA tests; needs real GPU):
pytest tests/integration/ -v --run-gpu --run-slow
```

## P5 integration fixes applied

- **test_10 vitest subprocess fix.** Added `shutil.which("pnpm")` guard;
  the test now cleanly skips when pnpm is not on PATH instead of crashing.
- **StepTimeline wired.** The unused `StepTimeline` component is now rendered
  as a full-width row below the 2×2 dashboard grid (page.tsx).
- **verify_real_banks.py added.** TDD acceptance script that runs all three
  real-bank checks (injection KL, byte contract, efficacy threshold) with
  color-coded PASS/FAIL output.
- **Fakes module collision (root cause).** Both apps' tests/fakes.py modules
  collided in sys.modules. Added conftest.import_app_fakes() with namespaced
  modules; tests 3,5,6,7,8,9,11 updated.
- **Test 6 WS fix.** aconnect_ws arg order reversed, httpx.ASGITransport
  404s on WS (switched to ASGIWebSocketTransport), fixed receive loop.
- **run_demo.sh hardened.** Preflight tool/bank checks, uvicorn --factory,
  banks_loaded health assertion.
- **Environment audit complete.** MODEL_ID added to inference-engine
  .env.example; all 4 apps cover their env vars per CONTRACTS sec 9.
- **Transformers pin identical** (4.46.*) in all 3 Python apps.
- **selected_layers identical** [8,12,16,20] across manifest, engine config,
  compiler, and shared constants.

## Demo rehearsal

- **Not yet rehearsed with real GPU / live HN.** See RUNBOOK.md.
- Pre-warm model, pre-load HN pages, pin the target story.
- Record backup demo (OBS/QuickTime, <=5 min).
- Stop EC2 g5.2xlarge after (~$1.50/hr).
- run_demo.sh provides fail-fast startup with clear error messages.
