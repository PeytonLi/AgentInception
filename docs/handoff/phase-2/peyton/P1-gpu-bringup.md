# Agent P1 - GPU Bring-up & Engine Boot (Peyton)

**Owner:** Peyton  **Branch:** `phase2/p1-gpu-bringup`  **Worktree:** `.claude/worktrees/phase2-p1`
**Reads first:** `docs/handoff/phase-2/README.md`, `docs/handoff/CONTRACTS.md` (s1, s5, s6, s9),
`docs/handoff/peyton/aws-runbook.md`, `apps/inference-engine/src/inference_engine/{server.py,engine.py,config.py}`.
**Depends on:** nothing (this is the gating task - everything GPU-bound waits on you).
**Unblocks:** R1, P2, P3.

## Mission

Take the inference engine from "only ever ran with `FakeBackend`" to "Llama-3.1-8B
loaded on a real A10G, `/healthz` returns `model_loaded: true` and all 3 page_keys".
This is pure bring-up + reliability, not new features. The code exists; make it run.

## Why this is the gate

Nothing in the vision has touched a GPU. The single A10G box is shared by R1 (compile)
and the engine (serve). You own the box's provisioning and the engine's real startup
path so that R1 can compile and P2 can validate against a live server within hours.

## Tasks

1. **Provision the box** exactly per `aws-runbook.md`: `g5.2xlarge`, Deep Learning AMI,
   200 GB gp3, security group (22 from your IP; 8000 to teammates). Capture the public
   IP. Bring up ClickHouse via `infra/docker-compose.yml` + `scripts/ch_init.sh`.
2. **Env + model access.** Put `HF_TOKEN` in the box's shell/`.env`. `huggingface-cli
   download meta-llama/Llama-3.1-8B-Instruct` to warm the cache before first boot so the
   demo never blocks on a cold download.
3. **Real startup smoke.** Install `apps/inference-engine` (its `requirements.txt`,
   `transformers==4.46.*`). Boot `python -m inference_engine.server`. Confirm the
   `lifespan` path (`server.py`) actually loads `BankRegistry.load(...)` then
   `LlamaBackend.load(settings)` on the GPU. Fix whatever breaks (dtype, device_map,
   `attn_implementation`, OOM, tokenizer pad token). **Capture every fix as a commit.**
4. **`/healthz` contract.** `GET /healthz` must return `model_loaded: true` and
   `banks_loaded` listing the 3 page_keys (synthetic banks are fine for P1 - we only
   need the load path working; R1 swaps in real banks later).
5. **Latency baseline.** Record cold-start time, model VRAM footprint (`nvidia-smi`),
   and single-step `/api/v1/step` latency with a trivial scripted prompt. Write these
   to `docs/handoff/phase-2/notes/p1-bringup-metrics.md` so P2/P5 have a baseline.
6. **GPU smoke test.** Add `apps/inference-engine/tests/test_gpu_boot.py` marked
   `@pytest.mark.gpu` (skips without CUDA): loads the real backend, runs one generate,
   asserts non-empty Action JSON. This is the first test that actually exercises the
   real model - keep it fast (short max_tokens).

## Definition of done

- Engine boots on the A10G; `/healthz` green with `model_loaded: true` + 3 page_keys.
- `test_gpu_boot.py` passes on the box; skips cleanly off-GPU (CI stays green).
- `p1-bringup-metrics.md` committed with VRAM, cold-start, and per-step latency.
- A short `docs/handoff/phase-2/notes/p1-box.md`: public IP placeholder, exact boot
  commands, how to stop/start the instance, and where the HF cache lives.
- No edits outside `apps/inference-engine/`, `infra/`, `scripts/ch_init.sh`,
  `docs/handoff/phase-2/notes/`. (Keep write scope disjoint from P2/P3.)

## Commit / push / PR

Work in the worktree; commit each fix with a clear message; push `phase2/p1-gpu-bringup`;
open a PR titled "Phase 2 / P1 - GPU bring-up" with the metrics file linked and a
"box is live at <ip>" note for teammates. **Do not stop the instance until R1 confirms
it has the model cache it needs.**

## Suggested skills

`tdd`, `diagnose` (CUDA/dtype/OOM), `git-commit`.
