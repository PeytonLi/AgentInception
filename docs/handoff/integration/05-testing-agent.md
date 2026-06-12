# Agent C1 — Integration & Testing Agent (run last)

**Owner track:** shared (run after Tracks A and B converge, ~H+12)
**Builds:** `tests/integration/`, run scripts, bug fixes anywhere in the repo
**Reads first:** `docs/handoff/CONTRACTS.md` (all), every other agent spec's Definition of Done
**Authority:** this agent may edit ANY package to fix integration bugs — but must not change CONTRACTS.md semantics; if two components disagree about the contract, the contract is right and both get fixed toward it.

## Mission

Prove the whole system works as one, using strict TDD: write the integration test first, watch it fail, then wire or fix until green. No test may be weakened to pass — fix the code, not the test. Finish with a full demo rehearsal.

## Method (non-negotiable)

1. Write ALL tests in the checklist below as failing stubs first (`pytest` + Playwright; mark slow/GPU tests). This is the red phase — commit it.
2. Work the list top-to-bottom (it's dependency-ordered). For each: run → diagnose → minimal fix in the owning package → green → commit with the test name in the message.
3. Anything that can't be fixed in <30 min: document in `tests/integration/KNOWN_ISSUES.md` with reproduction + workaround, and move on. One day; ruthless triage.

## Integration test checklist (dependency order)

| # | Test | Asserts | Components |
|---|---|---|---|
| 1 | `test_bank_binary_contract` | B1 output loads via `shared-py.bank_io`: shape `[8,S,128]`, f32, equal S across layers | B1 ↔ shared-py |
| 2 | `test_clickhouse_roundtrip_real_banks` | upload → `load_all_banks()` byte-identical for all 3 banks | B2 ↔ A2 |
| 3 | `test_engine_startup_preload` | engine boots, `/healthz` lists all 3 page_keys, zero DB calls during a step (assert via query log) | A1 ↔ A2 ↔ B2 |
| 4 | `test_injection_changes_logits` [GPU] | same prompt ± hn:front bank → KL(logits) > 1e-3; clear_bank() restores bit-exact baseline | A1 ↔ B1 |
| 5 | `test_unknown_page_fallback` | step with `page_key="unknown"` → `bank_found=false`, no exception, dom_text used | A1 |
| 6 | `test_ws_event_flow` | one `/api/v1/step` → WS client receives `layer_injection` → `action` → `token_metrics` in order, schema-valid | A1 ↔ A4 contract |
| 7 | `test_runner_against_real_engine` [GPU] | mi-mode loop on fixture pages completes ≤ 15 steps, `agent_steps` rows written, frames posted | A3 ↔ A1 |
| 8 | `test_popup_chaos_flow` [GPU] | runner on popup page: model returns `dismiss_modal`, modal gone (data-testid absent), task resumes — WITH the popup bank injected; control run without bank documented (may fail to dismiss — that contrast is a demo talking point) | A3 ↔ A1 ↔ B1/B2 |
| 9 | `test_token_metrics_honesty` | over a 3-step mi run: `cum_visible` < 1,500; `cum_baseline` > 10× that; ratio matches the formula in handoff README | A1 ↔ A3 |
| 10 | `test_console_renders_live_run` (Playwright) | dashboard against a real run: 4 panels populate, layers light up, chart has 2 diverging series | A4 ↔ everything |
| 11 | `test_baseline_mode_e2e` [GPU] | baseline run completes the same fixture task; its `cum_visible` ≈ `cum_baseline` | A3 ↔ A1 |

## Wiring responsibilities (the glue nobody else owns)

- `scripts/run_demo.sh`: starts ClickHouse → engine (tmux on EC2) → console → prints the two runner commands (baseline, mi) ready to paste. Startup-order checks with health polling, fail fast with clear messages.
- `.env.example` completeness audit across all four apps; one `docs/handoff/RUNBOOK.md` section: "cold EC2 box → demo-ready in N commands".
- Version pin audit: `transformers==4.46.*` identical in inference-engine and bank-compiler (a mismatch here silently breaks bank compatibility).

## Demo rehearsal (after all green)

Run the exact stage sequence twice end-to-end: baseline on live HN → mi on live HN → popup chaos test → savings chart close. Time it. Note flaky moments in `KNOWN_ISSUES.md` with mitigations (e.g., pre-warm model, pre-load HN pages). Capture a full screen recording of the best run as the backup demo — **a hackathon demo without a recorded fallback is a gamble; make the recording.**

## Definition of done

- Checklist tests 1–9 green (10–11 green or in KNOWN_ISSUES with cause).
- `run_demo.sh` boots the stack from cold.
- Backup recording exists.
- `KNOWN_ISSUES.md` honest and current.

## Suggested skills

`superpowers:test-driven-development`, `superpowers:systematic-debugging`, `everything-claude-code:e2e-testing`, `superpowers:verification-before-completion`
