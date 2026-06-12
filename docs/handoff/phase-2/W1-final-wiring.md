# Agent W1 - Final Wiring Verification (Peyton, runs last)

**Owner:** Peyton  **Branch:** `phase2/w1-final-wiring`  **Worktree:** `.claude/worktrees/phase2-w1`
**Reads first:**
- `docs/handoff/phase-2/README.md` (this sprint)
- `docs/handoff/CONTRACTS.md` (all 10 sections, every one is a checklist item)
- `docs/handoff/integration/05-testing-agent.md` (Phase-1 integration spec)
- Every Phase-2 agent's Definition of Done - you verify them, you do not redo them
- The paper arXiv:2605.06225 s3 + Eq. 6-7 for the bank shape/position axioms
**Depends on:** P1, P2, P3, P4, P5, R1, R2, R3 - all merged to `main` first. This
is the **last agent to run in Phase 2.** Nothing follows it.
**Authority:** may read and flag any file in the repo but edits only
`docs/handoff/phase-2/notes/w1-report.md` + the wiring check scripts below plus
`KNOWN_ISSUES.md`. Do NOT change application code - hand bugs back to the owning
agent as a failing test and a clear repro.

## Mission

Every build agent (P1-P5, R1-R3) claims its piece works. Your job is to **prove
the whole is wired** by treating the deployed system as a black box and verifying
every contract point, every data-flow pipe, every startup dependency, and every
graceful fallback from cold. You are the gate. The system is not "done" until this
report is green.

## The wiring verification checklist

Run every check against a **fresh EC2 box** with zero cached state. Start from
scratch, run `scripts/run_demo.sh`, then execute the checks. Stop the instance
when done.

### A. Contracts audit (CONTRACTS.md s1-s10)

- **s1 - Constants.** `SELECTED_LAYERS`, `NUM_KV_HEADS`, `HEAD_DIM`, `HIDDEN_SIZE`,
  `BANK_DTYPE`, `TRANSFORMERS_PIN` agree in: engine `config.py`, compiler
  `encoder.py`, `banks/manifest.json`, `shared-py/constants.py`. One discrepancy
  silently breaks banks. Flag immediately.
- **s3 - `page_key()`.** Hit the engine with URLs for all known types + one unknown.
  The engine's bank lookup / fallback must match the deterministic `shared-py` output.
- **s4 - Bank binary format.** Load any real bank with `bank_io.load_bank()`; assert
  shape `[8, S, 128]` float32, `S` equal across all 4 layers. Keys are pre-RoPE
  (P2's position-independence test proves this; verify it PASSES on the box).
- **s5 - ClickHouse.** `latent_memory_banks` and `agent_steps` schemas match
  CONTRACTS.md exactly. Bank upload idempotent.
- **s6 - HTTP API.** Full request/response round-trip for `GET /healthz`, `POST
  /api/v1/step` (both modes), `POST /internal/frame`. Response bodies match the
  contract exactly - no missing or extra keys.
- **s7 - WS events.** Connect to `/ws/events` with a raw WS client. Fire one step.
  All 5 event types arrive in order and match their schema. A second WS client
  receives the same stream (event hub broadcasts to all).
- **s8 - Action JSON.** Model output in both modes is valid Action JSON. The 2-strike
  malformed-JSON retry works (probe: deliberately broken prompt -> engine returns
  502 with truncated output in the detail field).
- **s9 - Ports and env.** ClickHouse on 8123/9000; engine on 8000; all env vars used
  by the right components. Every key in CONTRACTS.md s9 appears in the right app's
  `.env.example`.
- **s10 - Mocks.** All mock files still present and runnable. `pytest` in each app
  passes without GPU (CI stays green).

### B. Startup-order dependency chain

- **Cold-boot script.** `scripts/run_demo.sh` exits 0 from a truly cold box (no
  Docker images cached, no Python venv, no model on disk). Time the full cold start.
- **Health-poll.** Every `sleep + curl` actually polls and fails fast with a clear
  message. Test by stopping ClickHouse before running - script must print a clear
  error and exit 1, not hang.
- **Engine preload.** After startup, `GET /healthz` returns `banks_loaded` with
  exactly the expected page_keys within 3 seconds of the model finishing its load.

### C. End-to-end data-flow trace

Run the locked demo task in `mi` mode and trace one complete molecule:

1. DOM capture (snapshot or live page) -> raw HTML on disk.
2. Bank compiler -> Haiku summary (200-400 words) -> Llama forward pass -> `.bin`.
3. `scripts/upload_banks.py` -> ClickHouse (query and confirm byte-identical).
4. Engine startup -> `BankRegistry.load()` -> in-memory `page_key -> (K,V)` dict.
5. Agent-runner step -> `POST /api/v1/step` -> bank applied to MI attention -> Action
   JSON -> runner executes via Playwright.
6. Engine pushes `layer_injection -> action -> token_metrics` WS events in order.
7. Web console renders two diverging chart series, highlights layers 8/12/16/20.
8. `step_logger.log()` writes `agent_steps` ClickHouse row (query and confirm).

**Red test:** insert an intentional break at any hop (wrong-shape bank, kill the WS,
wrong port). The system must fail with an actionable error, not a silent wrong
answer. Document each break test and the observed error quality.

### D. Graceful fallback (CONTRACTS s6)

- **Unknown page.** Run a step with `page_key="unknown"` and `dom_text` populated.
  Engine returns `bank_found=false`, `injected_layers=[]`, still produces valid
  Action JSON using `dom_text`. No 500, no crash, no nop.
- **No bank loaded.** Clear the bank registry (or boot without ClickHouse). Engine
  returns `bank_found=false` and falls back to DOM text - not crash at startup or
  hang on the first request.
- **Model OOM or CUDA error mid-generate.** Simulate with a large prompt batch.
  Engine must return 5xx with a clear `detail` string, not a raw traceback. Runner
  must log the error and stop gracefully, not enter an infinite retry loop.
- **WS disconnect / reconnect.** Kill the console mid-run. Engine must not crash.
  New console reconnects and receives events from that point forward.

### E. Telemetry and honesty

- **Token counts.** Over a 3-step `mi` run on real HN front page: (a) each
  `token_metrics` event has `visible_tokens` < 500, (b) `baseline_tokens` > 10x
  `visible_tokens`, (c) `cum_visible` and `cum_baseline` monotonic, (d)
  `kv_savings_ratio` matches `(NUM_LAYERS * dom_token_count) / (L_injected *
  num_slots)` within 1 decimal place.
- **ClickHouse rows.** Query `agent_steps` after a full baseline + mi run. Each
  step has a row; `step` sequential; `mode` matches; no missing steps.
- **Frame cadence.** With runner posting at ~300ms, measure `viewport_frame`
  interval over 30 seconds. Mean 250-350ms; no gaps > 2s.

### F. Crash recovery

- **Engine restart mid-run.** Kill uvicorn mid-step. Restart it. Runner's next
  `POST /api/v1/step` must get a non-crash response. Runner's loop retries or
  logs, does not silently halt.
- **ClickHouse restart.** Kill ClickHouse mid-run. Engine's step endpoint still
  returns valid responses (bank is in-memory). Runner step-logging logs a warning
  and continues, does not crash.

### G. Final sign-off checklist

- [ ] All CONTRACTS.md s1-s10 audit items pass.
- [ ] `run_demo.sh` boots the stack from cold; exits 0; each health check polls.
- [ ] A complete data-flow trace (DOM -> bank -> ClickHouse -> engine preload ->
  runner step -> WS event -> console render) end-to-end green.
- [ ] All 4 fallback scenarios documented and passing.
- [ ] Token-honesty numbers within bounds.
- [ ] Crash-recovery scenarios passing.
- [ ] `docs/handoff/phase-2/notes/w1-report.md` written with every check's
  pass/fail/repro.
- [ ] Any failing check filed as a GitHub issue with the `w1-gate` label.

## Commit / push / PR

Your only committed files are a wiring checklist script (optional:
`scripts/check_wiring.py` or a shell script) and `w1-report.md`. This agent's PR
"Phase 2 / W1 - final wiring verification" is the **last PR before the phase is
declared done.** PR body: the checklist above with checkmarks and links to any
filed issues.

## Suggested skills

`diagnose` (everything is a reproducible probe), `review-architecture` (the
contracts-level pass), `git-commit`.
