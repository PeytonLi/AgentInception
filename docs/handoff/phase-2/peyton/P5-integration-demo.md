# Agent P5 - Integration Merge, Demo Orchestration & Recording (Peyton)

**Owner:** Peyton  **Branch:** `phase2/p5-integration-demo`  **Worktree:** `.claude/worktrees/phase2-p5`
**Reads first:** `docs/handoff/phase-2/README.md`, `docs/handoff/integration/05-testing-agent.md`,
`CONTRACTS.md` (all), the unmerged branch `worktree-c1-testing-agent`
(`tests/integration/*`, `scripts/run_demo.sh`, `tests/integration/KNOWN_ISSUES.md`).
**Depends on:** P1, P2, P3, P4 and R1 (everything must be real first). **Runs last.**
**Authority:** may edit any package to fix integration bugs, but must not change
`CONTRACTS.md` semantics - if two components disagree, the contract wins and both get
fixed toward it.

## Mission

The C1 integration suite exists but is stranded on an unmerged branch and almost
entirely SKIPPED (no GPU). Land it on `main`, get the GPU-gated tests genuinely green
against the real stack, then make the demo bulletproof: one-command boot, a written
runbook, two clean rehearsals, and a recorded backup.

## Tasks

1. **Land C1.** Merge `worktree-c1-testing-agent` into your branch (or cherry-pick its
   15 files). Resolve conflicts against current `main`. Get tests 1-2, 6, 10 green
   without GPU first (fast confidence).
2. **Turn on the GPU tests.** Against the live P1 box with R1's real banks, run tests
   3,4,5,7,8,9,11 with `--run-gpu --run-slow`. Each red one: diagnose -> minimal fix in
   the owning package -> green -> commit with the test name. In particular:
   - `test_03_engine_startup_preload`: 3 page_keys, zero DB calls during a step.
   - `test_04_injection_changes_logits`: real-bank KL > 1e-3 + bit-exact restore (mirror
     P2's result; this is the regression lock).
   - `test_07/08/11`: real runner loop, popup chaos, baseline e2e.
   Anything not fixable in <30 min -> document in `tests/integration/KNOWN_ISSUES.md`
   with repro + workaround, and move on. Ruthless triage.
3. **`run_demo.sh` hardening.** From a cold box: ClickHouse -> upload banks -> engine
   (tmux) -> health-poll -> console -> print the two ready-to-paste runner commands
   (baseline, mi). Fail fast with clear messages on any startup-order problem.
4. **Audits.** `.env.example` completeness across all 4 apps; confirm
   `transformers==4.46.*` identical in inference-engine and bank-compiler (a mismatch
   silently breaks bank compatibility); confirm `selected_layers` agree across
   manifest, engine config, and compiler.
5. **`docs/handoff/RUNBOOK.md`.** "Cold EC2 box -> demo-ready in N commands", plus the
   exact stage sequence: baseline on live HN -> mi on live HN -> popup chaos -> savings
   chart close. Include the stop-the-instance reminder.
6. **Rehearse twice + record.** Run the full stage sequence end-to-end twice. Time it.
   Note flaky moments in `KNOWN_ISSUES.md` with mitigations (pre-warm model, pre-load
   HN pages, pin the target story). **Capture a full screen recording of the best run as
   the backup demo** - a hackathon demo without a recorded fallback is a gamble.

## Definition of done

- Integration tests 1-9 green on the real stack; 10-11 green or in `KNOWN_ISSUES.md`
  with a documented cause.
- `scripts/run_demo.sh` boots the whole stack from cold.
- `docs/handoff/RUNBOOK.md` written and accurate.
- Backup screen recording exists (link it in the PR / RUNBOOK).
- `KNOWN_ISSUES.md` honest and current.

## Commit / push / PR

Commit per fixed test (use the test name). Push `phase2/p5-integration-demo`; PR
"Phase 2 / P5 - integration + demo". PR body: the final green checklist, the RUNBOOK
link, and the recording link. This PR is the one that declares the phase done.

## Suggested skills

`diagnose`, `tdd`, `review-architecture` (final pass), `git-commit`.
