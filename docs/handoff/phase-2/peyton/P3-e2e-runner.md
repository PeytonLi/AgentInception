# Agent P3 - Live-HN End-to-End Runner (Peyton)

**Owner:** Peyton  **Branch:** `phase2/p3-e2e-runner`  **Worktree:** `.claude/worktrees/phase2-p3`
**Reads first:** `docs/handoff/phase-2/README.md`, `CONTRACTS.md` (s6, s7, s8),
`apps/agent-runner/agent_runner/{loop.py,browser.py,dom.py,actions.py,steplog.py,frames.py,metrics.py}`,
`apps/agent-runner/tests/test_loop_against_mock.py`.
**Depends on:** P1 (engine on GPU), P2 (banks actually steer), R1 (3 real banks live),
R3 (fresh DOM fixtures + popup page). **Unblocks:** P5.

## Mission

The agent loop has only ever run against `tests/mocks/mock_inference.py`. Drive it
end-to-end against the **real** engine, on **live** Hacker News and the local popup
fixture, in both `baseline` and `mi` modes, and confirm the metrics + ClickHouse rows +
frame stream are all real. This is where the locked demo task actually gets executed.

## The locked task (from Phase-1 handoff, do not change)

"Find the top story about AI on the Hacker News front page (scan up to 2 pages), open
its comment page, extract the story score and top 3 commenter usernames."

## Tasks

1. **Real-engine smoke (mi).** Point the runner at `INFERENCE_URL=http://<p1-box>:8000`.
   Run the task on live HN in `mi` mode. It must reach a terminal `done`/`extract`
   action in <= 15 steps. Capture the transcript.
2. **Baseline parity.** Run the same task in `baseline` mode (full DOM in prompt each
   step). Confirm it also completes, and that its `cum_visible` ~= `cum_baseline`
   (baseline has no savings - this is the honesty control).
3. **Token-honesty check.** Over a 3-step `mi` run: `cum_visible` < ~1,500 and
   `cum_baseline` > 10x that; the printed `kv_savings_ratio` matches the README formula
   `(NUM_LAYERS * dom_token_count) / (L_injected * num_slots)`. If the real numbers are
   off, fix the runner's token accounting (`tokenizer.py`, `metrics.py`) - never fudge
   the display.
4. **Frame streaming.** Confirm `frames.py` posts JPEGs to `POST /internal/frame` at
   ~300 ms cadence and they rebroadcast on the WS (P4 needs this). Verify quality 50,
   1280x720.
5. **ClickHouse steps.** Confirm `agent_steps` rows are written per step with correct
   `mode`, `page_key`, `visible_tokens`, `baseline_tokens`, `bank_found`. Add an
   integration check that queries the rows back.
6. **Popup chaos run.** On the local popup fixture (from R3): in `mi` mode with the
   popup bank, the model returns `dismiss_modal`, the modal disappears (data-testid
   gone), and the task resumes. Also run the **control** without the popup bank and
   record the contrast (it may fail to dismiss - that contrast is a demo talking point).
7. **Robustness.** Handle live-HN flakiness: navigation timeouts, "More" pagination,
   the title-vs-comments link distinction. Add retries/waits in `browser.py` where the
   live site needs them. Keep `loop.py`'s mode-agnostic design intact.

## Definition of done

- One recorded `mi` transcript and one `baseline` transcript completing the task on
  live HN, saved under `docs/handoff/phase-2/notes/p3-transcripts/`.
- Token-honesty numbers captured and matching the formula.
- `agent_steps` rows verified; frames confirmed flowing.
- Popup chaos run (bank vs no-bank) documented.
- New/updated tests run against the real engine behind `@pytest.mark.gpu`/`slow`;
  the mock-based tests still pass for CI.
- Write scope limited to `apps/agent-runner/` + `docs/handoff/phase-2/notes/`.

## Commit / push / PR

Commit per capability (smoke, baseline, honesty, frames, popup). Push
`phase2/p3-e2e-runner`; PR "Phase 2 / P3 - live-HN e2e". Paste the two cumulative-token
numbers in the PR body - that pair is the demo's money shot.

## Suggested skills

`tdd`, `diagnose` (live-site flakiness), `frontend-design` not needed, `git-commit`.
