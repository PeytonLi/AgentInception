# AgentInception — Handoff & Build Plan

**Project:** AgentInception / AgentInception — a web agent that replaces verbose DOM text in the prompt with pre-computed latent KV banks injected at selected transformer layers, based on Memory Inception (MI).

**Paper:** [arXiv:2605.06225 — Memory Inception: Latent-Space KV Cache Manipulation for Steering LLMs](https://arxiv.org/abs/2605.06225) (Liu et al., 2026). Read §3 and Appendix B.1/C/G.2 before touching inference code.

**Status:** Design fully locked after a deep review session. Zero code written. This directory is the single source of truth — where the original PRD conflicts with this doc, **this doc wins**.

**Timeline:** One-day hackathon sprint.

---

## What changed from the original PRD (and why)

| PRD said | We do instead | Why |
|---|---|---|
| Qwen-2.5-72B / Llama-3.1-70B | **Llama-3.1-8B-Instruct** | Only models tested in the paper are Llama-3.1-8B and Qwen3-30B-A3B; 8B fits a single A10G |
| vLLM attention patching | **HuggingFace `transformers` custom attention at 4 layers** | Paper's vLLM path is itself an approximation; HF matches the reference math (Eq. 2 + Eq. 7) exactly |
| Automated layer selector at runtime | **Hardcoded layers `[8, 12, 16, 20]`** | Selector (paper Algorithm 1) needs offline calibration; out of scope for one day |
| 118× KV savings | **Claim 20–80×** | 118× applies to PHYSICS-length heuristic banks only; our formula gives 20–80× for DOM guidance |
| Airbyte, Composio, Guild.ai orchestration | **Cut.** Playwright direct; GuildAI optional 30-line run logger at the very end | No role in a one-day demo |
| Truefoundry inference | **Plain EC2 + FastAPI** (Truefoundry = post-hackathon note) | Simplicity |
| Demo on enterprise app / Amazon | **Hacker News + one local popup fixture page** | Amazon bot-detection kills live demos; HN is public, stable, judge-recognizable |
| Online bank compilation | **Fully offline pre-compilation** | Every demo page is a cache hit; reproducible |
| Bank lookup by exact DOM hash | **Lookup by `(domain, page_type)`** | HN comment counts change per article — exact structural hashes would miss constantly |

**Honest pitch math** (Llama-3.1-8B, L=32 layers, L_ctrl=4):
`KV_ratio = (32 × T_guidance) / (4 × S_bank)` → 2,000-token DOM guidance vs 200-slot bank = **80×**; 500-token guidance = **20×**. Never claim 118×.

---

## Demo flow (locked)

Task: *"Find the top story about AI on the Hacker News front page (scan up to 2 pages), open its comment page, extract the story score and top 3 commenter usernames."*

1. **Baseline run** (`--mode=baseline`): full DOM text in prompt every step. Token counter climbs visibly.
2. **MI run** (`--mode=mi`): visible prompt stays ~200 tokens; banks injected at layers 8/12/16/20. Counter stays flat.
3. **Chaos test**: navigate to the local popup fixture page; cookie-consent modal fires; the popup bank routes the agent to dismiss it; back to task. Prompt never grew.
4. Close on the cumulative savings chart.

Banks to pre-compile (only 3): `hn:front`, `hn:item`, `popup:demo`.

---

## How to use this handoff

Each file below is a **self-contained brief for one Claude Code agent session**. Agents within the same track are independent of each other — they share only `CONTRACTS.md`, which every agent must read first and must not change unilaterally. If a contract is wrong, flag it; don't drift from it.

Every agent writes its own unit tests (TDD — tests first). The integration agent at the end owns cross-component tests and wiring.

### Track A — Peyton

| Agent | Spec | Builds |
|---|---|---|
| A1 | `peyton/01-inference-engine.md` | FastAPI inference server, custom MI attention at selected layers, WS event emitter |
| A2 | `peyton/02-infra-storage.md` | Monorepo scaffold, ClickHouse (Docker + schema + client + startup preload), AWS runbook |
| A3 | `peyton/03-agent-runner.md` | Playwright agent loop, baseline + MI modes, metrics |
| A4 | `peyton/04-web-console.md` | Next.js 4-panel dashboard with live WS feed |

### Track B — Rahul

| Agent | Spec | Builds |
|---|---|---|
| B1 | `rahul/01-bank-compiler.md` | Offline bank compiler: DOM → Haiku summary → Llama forward pass → K/V bytes |
| B2 | `rahul/02-demo-assets-and-data.md` | Popup fixture page, compile the 3 demo banks, upload + validation scripts |

### Track C — Integration (run last, after A and B converge)

| Agent | Spec | Builds |
|---|---|---|
| C1 | `integration/05-testing-agent.md` | Integration test suite (TDD), wiring, bug fixes, demo rehearsal |

### Sync points (one-day clock)

- **H+1:** A2 scaffold merged → everyone has a directory to work in. B1 starts immediately on a local checkout (no infra dependency).
- **H+4:** **Shape sync** — B1's first `.bin` bank must load into A1's attention wrapper and measurably change logits. This is the single highest-risk integration; do it as early as humanly possible.
- **H+8:** All 3 banks compiled and loaded into ClickHouse; A3 loop runs end-to-end in both modes.
- **H+12:** C1 takes over: integration tests, wiring, demo rehearsal.

---

## Prerequisites (do these before any agent starts)

1. **HF gated model access:** Llama-3.1-8B-Instruct requires accepting Meta's license on Hugging Face. Both machines need `HF_TOKEN` set, and the account must have access approved. Start this NOW — approval is usually fast but not instant.
2. **Env vars** (never commit values): `HF_TOKEN`, `ANTHROPIC_API_KEY` (bank compiler only), `CLICKHOUSE_URL`, `INFERENCE_URL`. Each app ships a `.env.example`.
3. **AWS:** EC2 `g5.2xlarge` (1× A10G 24GB, ~$1.21/hr), AWS Deep Learning AMI (Ubuntu). Runbook in `peyton/02-infra-storage.md`.
4. **Rahul's machine:** needs ~20GB disk for the model and any CUDA GPU with ≥16GB (or run the compiler on the EC2 box over SSH — it's offline batch work, contention with A1 is fine early in the day).
5. **Docker** for ClickHouse locally.

---

## Suggested skills

When running these agent briefs in Claude Code, invoke:

- All agents: `superpowers:test-driven-development` (tests first, every component), `superpowers:verification-before-completion`
- A1, B1 (model internals): `everything-claude-code:pytorch-patterns`
- A2 (ClickHouse): `everything-claude-code:clickhouse-io`
- A4 (dashboard): `frontend-design:frontend-design`, `everything-claude-code:nextjs-turbopack`
- C1 (integration): `superpowers:systematic-debugging`, `everything-claude-code:e2e-testing`
- Bank compiler's Haiku calls: `claude-api` (model id and SDK usage — use `claude-haiku-4-5-20251001`)
