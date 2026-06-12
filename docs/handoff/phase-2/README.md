# Phase 2 - "Make It Real and Prove It"

**Project:** AgentInception / GhostBrowser OS
**Predecessor:** `docs/handoff/` (Phase 1). Phase 1 is **done as code**; this directory is the next sprint.
**Status of the repo today:** fully scaffolded, every Phase-1 build agent (A1-A4, B1, B2) merged to `main`. **But it has never run on a GPU, and the demo banks are fake.**

> Phase 1 built the machine. Phase 2 turns it on, feeds it real data, and proves
> the headline claim end-to-end on a live page. Where Phase 1 specs conflict with
> this doc, **this doc wins.** `CONTRACTS.md` is still frozen - flag, don't drift.

---

## 1. Gap analysis: vision vs. what actually exists

The root `README.md` (the vision) and `docs/handoff/` (the Phase-1 plan) describe a
working system that demonstrably replaces ~14k DOM tokens with a ~300-slot latent
bank and measures the savings live. Here is the honest delta between that vision
and `main` as it stands.

| Area | Vision / Phase-1 plan | Actual state on `main` | Severity |
|---|---|---|---|
| Compiled KV banks | Real Llama-3.1-8B forward-pass banks for the 3 page types (Eq. 6, pre-RoPE) | Synthetic random float32 noise from `scripts/build_demo_banks.py`. Shapes right; values meaningless. They will NOT steer the model. | CRITICAL |
| Bank compiler (B1) | Run offline to produce those banks | Code complete + unit-tested, never executed against the real model | CRITICAL |
| GPU bring-up | Llama-3.1-8B on an A10G via FastAPI, real `/healthz` | No instance ever loaded the model; engine only ran with `FakeBackend` doubles | CRITICAL |
| H+4 "shape sync" | First real `.bin` loads and measurably changes logits | Never happened with a real bank | CRITICAL |
| Integration suite (C1) | Tests 1-11 green, `run_demo.sh`, `RUNBOOK.md`, recording | On UNMERGED branch `worktree-c1-testing-agent`; tests 3-9 & 11 SKIP, no RUNBOOK, no recording | HIGH |
| End-to-end agent run | Playwright loop completes the HN task in both modes live | Tested against the mock engine only; never run against a real engine or live HN | HIGH |
| Token-savings proof | Dashboard shows baseline climbing vs MI flat, honest 20-80x | Math in `service.py`; never validated with real token counts | HIGH |
| Web console (A4) | 4 panels populated by a live run | Built/tested against `mock_ws_feed.py`; never pointed at a real WS | MEDIUM |
| Demo readiness | Cold box -> demo-ready in N commands, rehearsed, recorded | `run_demo.sh` drafted (unmerged); no RUNBOOK, no rehearsal, no recording | MEDIUM |
| GuildAI run logger | Optional 30-line run logger | Not started (optional / stretch) | OPTIONAL |

**One-line summary:** the skeleton is excellent and the contracts are sound; what is
missing is *every step that requires a GPU and real data*. Phase 2 is almost
entirely "run the thing that was built, fix what breaks, replace the fake banks."

### Genuinely solid (do not rebuild)

- `packages/shared-py` - `page_key`, `bank_io`, `dom_hash`, ClickHouse storage. Tested.
- `apps/inference-engine` MI attention, registry, service, WS hub. Clean, contract-faithful.
- `apps/agent-runner` loop, retry, frames, steplog, metrics. Tested vs mock.
- `apps/bank-compiler` DOM->Haiku->Llama->bank_io pipeline. Tested vs fakes.
- `apps/web-console` reducer, event feed, 4 panels. Tested vs mock feed.
- `transformers==4.46.*` pinned consistently in all three Python apps.

---

## 2. Phase-2 mission

1. Stand up the real GPU box and load Llama-3.1-8B (P1).
2. Compile **real** banks for the 3 page types and replace the synthetic ones (R1).
3. Prove a real bank measurably steers the model - the H+4 shape sync, for real (P2).
4. Drive the live HN task end-to-end in both modes, writing real metrics (P3).
5. Point the console at the real WS and make it demo-grade (P4).
6. Merge C1, get GPU integration tests green, write `RUNBOOK.md`, rehearse, record (P5).
7. (Rahul) Build a steering-efficacy harness (R2) and refresh demo fixtures + the
   token-honesty dataset (R3).
8. (Peyton) Run the final wiring verification against the deployed system from
   cold — prove every contract, pipe, fallback, and crash path is wired (W1).

**Definition of phase done:** on a cold EC2 box, `scripts/run_demo.sh` brings the
stack up; an `--mode=mi` run on live Hacker News completes the locked task with
`cum_visible` an order of magnitude below `cum_baseline`; the popup chaos test
dismisses the modal *because of* the popup bank; the console shows it live; and a
backup screen recording of the best run exists.

---

## 3. Workload split - Peyton 65% / Rahul 35%

Peyton owns the runtime, infra, integration, and UI (Phase-1 tracks A1-A4 + C1).
Rahul owns the banks and demo data (Phase-1 tracks B1-B2) - smaller surface, but
**R1 is on the critical path and is the single highest-value task in the sprint.**

| Owner | Agents | Rough effort |
|---|---|---|
| **Peyton** | P1, P2, P3, P4, P5, W1 | **~68%** |
| **Rahul** | R1, R2, R3 | **~32%** |

| Agent | Spec | Builds | Worktree branch |
|---|---|---|---|
| P1 | `peyton/P1-gpu-bringup.md` | EC2 provisioning, model load, engine boot on GPU, smoke | `phase2/p1-gpu-bringup` |
| P2 | `peyton/P2-mi-validation.md` | Real-bank logit-shift proof, MI attention hardening, perf | `phase2/p2-mi-validation` |
| P3 | `peyton/P3-e2e-runner.md` | Live-HN end-to-end loop both modes, frames, ClickHouse steps | `phase2/p3-e2e-runner` |
| P4 | `peyton/P4-console-live.md` | Console against real WS, polish, savings chart, screenshot | `phase2/p4-console-live` |
| P5 | `peyton/P5-integration-demo.md` | Merge C1, GPU tests green, run_demo, RUNBOOK, rehearsal, recording | `phase2/p5-integration-demo` |
| W1 | `W1-final-wiring.md` | Cold-box contract audit, data-flow trace, fallback, crash recovery, final gate | `phase2/w1-final-wiring` |
| R1 | `rahul/R1-real-banks.md` | Compile real banks for the 3 page types, replace synthetic, re-upload | `phase2/r1-real-banks` |
| R2 | `rahul/R2-bank-efficacy.md` | Steering-efficacy harness, summary tuning, quality report | `phase2/r2-bank-efficacy` |
| R3 | `rahul/R3-demo-data.md` | Fresh DOM fixtures, popup page polish, token-honesty dataset | `phase2/r3-demo-data` |

---

## 4. Dependency graph and sync points

```
  P1  GPU box + Llama-3.1-8B loaded + engine /healthz green
        |  (GPU box available)
        +--> R1  Real banks compiled (uses P1 box/model)
                   |
            SYNC * H+4  the real shape sync
                   |
        P2  Real bank measurably shifts logits; MI attn hardened
                   |
        P3  Live-HN e2e both modes   <---- R3 fresh fixtures/data
                   |
        P5  Integration + demo       <---- P4 console (parallel)
                   |
        W1  Final wiring gate  (cold-box contract audit + crash recovery)

  R2 (efficacy harness) runs in parallel with R1 and feeds bank-quality
  decisions back into R1. P4 is independent until P5 (mock -> real swap).
  W1 runs last and is the final gate - nothing merges to main after it.
```

**Sync points (sprint clock):**

- **H+4 - The real shape sync (highest risk).** R1 produces the first real
  `hn_front` bank; P2 loads it into the live engine and asserts the logit
  distribution shifts (KL > 1e-3) and that `clear_bank()` restores a bit-exact
  baseline. **Do this before anything downstream.** If shape, dtype, pre-RoPE
  convention, or `repeat_kv` grouping is off, find out at hour 4, not at the demo.
- **H+8 - All 3 real banks live.** R1 finished; banks uploaded to ClickHouse;
  engine startup preload lists all 3 page_keys from real data.
- **H+12 - E2E green.** P3 completes the HN task in both modes against the real
  engine; `agent_steps` rows written; frames flowing.
- **H+16 - Demo locked.** P5 has the integration suite green on GPU, `RUNBOOK.md`
  written, and a recorded backup run.
- **H+18 - Final wiring green.** W1 verifies every contract point, data-flow pipe,
  and crash-recovery path on the cold box. The phase is done.

---

## 5. Worktree workflow (every agent follows this exactly)

Each agent works in its **own git worktree** so sessions never collide on the
working tree, and finishes by **committing, pushing, and opening a PR**:

```
# 1. Isolated worktree + branch off latest main (use your agent id, e.g. p1)
git fetch origin
git worktree add -b phase2/p1-gpu-bringup .claude/worktrees/phase2-p1 origin/main

# 2. Work ONLY inside that directory for the whole session
cd .claude/worktrees/phase2-p1

# 3. Commit early and often, on your branch only
git add -A
git commit -m "P1: load Llama-3.1-8B and boot engine on GPU"

# 4. Push your branch and open a PR (do NOT push to main directly)
git push -u origin phase2/p1-gpu-bringup
gh pr create --fill --base main --title "Phase 2 / P1 - GPU bring-up"

# 5. After merge, retire the worktree
cd ../../.. && git worktree remove .claude/worktrees/phase2-p1
```

**Rules of the road:**

- One agent = one worktree = one branch. Never check out a sibling branch inside
  your worktree; add a new worktree if you must inspect one.
- Disjoint write scope. The split is designed so two agents never edit the same
  file. If you must touch another agent's file, let them make the change or do a
  tiny follow-up PR.
- Never edit `docs/handoff/CONTRACTS.md` semantics. Flag; the contract wins.
- Banks are gitignored except `manifest.json`. R1 commits the manifest + summary
  text; the `.bin` blobs travel via ClickHouse / S3 / scp, not git.
- Push and open a PR even if not fully green - leave a checklist in the PR body.

---

## 6. Prerequisites

1. HF gated access to `meta-llama/Llama-3.1-8B-Instruct` for the account whose
   `HF_TOKEN` lands on the EC2 box (re-verify it is still live).
2. `ANTHROPIC_API_KEY` with access to `claude-haiku-4-5-20251001` (R1/R3 only).
3. AWS `g5.2xlarge` per `docs/handoff/peyton/aws-runbook.md` (1x A10G 24 GB).
4. Docker for ClickHouse on the same box (or local for non-GPU agents).
5. `gh` CLI authenticated for opening PRs.

---

## 7. Suggested skills

- All agents: `tdd` (tests first), then commit via `git-commit`.
- P1, P2, R1, R2 (GPU / model internals): a PyTorch-patterns skill if available;
  lean on `diagnose` for CUDA / shape / dtype bugs.
- P4: `frontend-design`.  P5: `diagnose`, `review-architecture`.
  W1: `diagnose`, `review-architecture`.
- R1/R3 Haiku calls: Anthropic API docs (`claude-haiku-4-5-20251001`).
- Summarizing a long session for the next agent: `handoff`.
