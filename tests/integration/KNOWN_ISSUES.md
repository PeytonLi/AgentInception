# KNOWN ISSUES - Agent C1 Integration Test Suite

Last updated: 2026-06-12

## Test Status

| # | Test | Status |
|---|------|--------|
| 1 | Bank binary contract | GREEN (6/6) |
| 2 | ClickHouse roundtrip | GREEN with --run-slow |
| 3-9,11 | Engine/GPU tests | SKIP (needs torch/CUDA) |
| 10 | Console renders | GREEN with --run-slow |

## How to run

Fast: pytest tests/integration/ -v
Slow: pytest tests/integration/ -v --run-slow
GPU:  pytest tests/integration/ -v --run-gpu --run-slow

## Demo rehearsal

For GPU-free demos, use mock inference engine.
Pre-warm model, pre-load pages, record fallback video.

