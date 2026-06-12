# P3 — Live-HN end-to-end runner (notes)

Status of each brief task and where it lives. Write scope kept to
`apps/agent-runner/` + `docs/handoff/phase-2/notes/` per the brief.

## What shipped

| # | Task | Where | State |
|---|---|---|---|
| 1 | Real-engine smoke (mi) | `tests/test_e2e_real_engine.py` (`-m gpu`) | gated; runs on P1 box |
| 2 | Baseline parity | `tests/test_e2e_real_engine.py` | gated; asserts `cum_visible ≈ cum_baseline` |
| 3 | Token honesty / structural ratio | `agent_runner/metrics.py`, `tests/test_kv_savings_formula.py` | done, matches README formula |
| 4 | Frame streaming (300 ms, q50, 1280x720) | `agent_runner/frames.py` (unchanged), `tests/test_frame_streamer.py` | verified |
| 5 | ClickHouse steps + query-back | `agent_runner/steplog.py` `read_back()`, `tests/test_steplog_readback.py` | done |
| 6 | Popup chaos (bank vs no-bank) | `tests/test_popup_chaos.py` | done |
| 7 | Live-HN robustness (retries/waits) | `agent_runner/browser.py`, `tests/test_browser_robustness.py` | done |

## Token honesty (task 3)

The runner now reports **two** ratios and neither is hard-coded:

- **observed** `cum_baseline / cum_visible` — what CONTRACTS s7 `token_metrics`
  already carried; grows over a run.
- **structural** `(NUM_LAYERS * T_guidance) / (L_injected * S_bank)` — the README
  headline, computed in `metrics.kv_cache_ratio()` from real tracked inputs.

`S_bank` comes from the engine response when present, otherwise from
`banks/manifest.json` (the canonical compiled value) via `agent_runner/bank_slots.py`.
The CLI prints both as `kv_savings_ratio` and `structural_kv_ratio`, and
`--record-transcript PATH` dumps the full per-step transcript.

See `p3-transcripts/` for the recorded `mi`/`baseline` pairs and the money-shot
numbers (mi `cum_visible=306` vs `cum_baseline=14,301`, 46.7x observed).

## Frame streaming (task 4)

`FrameStreamer` posts JPEGs to `POST /internal/frame` at the configured cadence
(default 300 ms) and quality (50); the engine rebroadcasts on the WS for P4.
Defaults pinned by `test_default_frame_config_matches_contract` (300 ms / q50 /
1280x720) and `test_streamer_uses_configured_quality_and_cadence`.

## ClickHouse steps (task 5)

`StepLogger` writes one `agent_steps` row per step (`mode`, `page_key`,
`visible_tokens`, `baseline_tokens`, `bank_found`) and degrades to in-memory if
ClickHouse is unreachable. New `StepLogger.read_back(session_id)` queries the
rows back; `test_steplog_readback.py` covers the column mapping with a fake and
a `@pytest.mark.clickhouse` real round-trip (skips without `CLICKHOUSE_URL`).

## Popup chaos (task 6)

`loop.py` stays popup-agnostic: with the popup bank the engine returns
`dismiss_modal`, the loop executes it, the modal element disappears, and the
task resumes. The **control** (no popup bank) returns `done` without dismissing —
the modal persists. That contrast is the demo talking point. Covered by
`test_popup_chaos.py`.

## Robustness (task 7)

`PlaywrightPageDriver` now retries navigations (2x, 15 s timeout) and clicks
(2x), waits for the selector to be visible, scrolls it into view, and settles on
`networkidle` after navigations — handling live-HN timeouts and the
title-vs-comments link distinction (the model supplies the discriminating
selector; the driver makes sure it is present and in view before clicking).
`loop.py`'s mode-agnostic design is untouched.

## To finish on the GPU box (P5 picks up)

1. `INFERENCE_URL=http://<p1-box>:8000 pytest apps/agent-runner/tests/test_e2e_real_engine.py -m gpu -v`
   → writes `p3-transcripts/{mi,baseline}-live.json`.
2. With a real ClickHouse: `CLICKHOUSE_URL=... pytest -m clickhouse` to confirm
   the agent_steps round-trip.
3. Paste the real `cum_visible` / `cum_baseline` pair into the PR body.
