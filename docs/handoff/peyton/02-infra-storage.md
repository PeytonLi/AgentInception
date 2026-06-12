# Agent A2 — Infra, Monorepo Scaffold & Storage

**Owner track:** Peyton
**Builds:** repo root scaffold, `infra/`, `packages/shared-py/`, AWS runbook
**Reads first:** `docs/handoff/CONTRACTS.md` (§1–§5, §9)
**Depends on:** nothing. **Everyone depends on you for the first hour — scaffold first, polish later.**

## Mission

Stand up everything the other agents assume exists: the monorepo skeleton, the shared Python package, ClickHouse with schema, and the EC2 GPU box.

## Tasks (in priority order)

### 1. Monorepo scaffold (first 30 minutes — unblocks everyone)

- `pnpm-workspace.yaml` (`apps/*`, `packages/*`), root `package.json`, `turbo.json` with `build`/`dev`/`test` pipelines.
- Empty-but-importable app skeletons: `apps/inference-engine`, `apps/agent-runner`, `apps/bank-compiler` (each: `src/`, `tests/`, `requirements.txt`, `.env.example`), `apps/web-console` (`pnpm create next-app`, App Router, TS, Tailwind).
- Root `.gitignore`: `banks/*.bin`, `.env*`, model caches, `node_modules`, `__pycache__`.
- Commit and push immediately. Tell the team.

### 2. `packages/shared-py/agentinception_shared/` — the shared contract code

- `page_key.py`: implement exactly CONTRACTS §3 (with the URL pattern table as data, unit-tested against real HN URLs incl. `news.ycombinator.com/news?p=2`).
- `bank_io.py`: `save_bank(dir, page_key, banks: dict[int, tuple[np.ndarray, np.ndarray]], meta) -> manifest entry` and `load_bank(...)`, plus `to_bytes`/`from_bytes` with shape validation that **raises** on wrong dtype/shape. This is the single (de)serialization implementation for the whole project (CONTRACTS §4).
- `dom_hash.py`: informational structural hash per CONTRACTS §3.
- Installable via `pip install -e packages/shared-py`.
- **Coordinate with B1 (Rahul's compiler agent):** if they started first, adopt their implementation into this package rather than writing a second one.

### 3. ClickHouse

- `infra/docker-compose.yml` (clickhouse-server, ports 8123/9000, named volume) + `infra/clickhouse/schema.sql` exactly as CONTRACTS §5.
- `apps/inference-engine/src/storage.py` (or `shared-py`): `clickhouse-connect` client with `insert_bank(...)`, `load_all_banks() -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]`, `log_step(...)` for `agent_steps`.
- `scripts/ch_init.sh`: `docker compose up -d` + apply schema, idempotent.

### 4. AWS EC2 runbook — `docs/handoff/peyton/aws-runbook.md`

Document (and execute) the exact steps:

1. Launch `g5.2xlarge`, AWS Deep Learning AMI GPU PyTorch (Ubuntu 22.04), 200GB gp3 root volume.
2. Security group: 22 (your IPs only), 8000 (inference HTTP/WS), 3000 if console runs there too.
3. `nvidia-smi` sanity, `python -c "import torch; print(torch.cuda.is_available())"`.
4. `export HF_TOKEN=...` + `huggingface-cli login` + start the model download immediately (`huggingface-cli download meta-llama/Llama-3.1-8B-Instruct`) — it's ~16GB, kick it off before anything else.
5. Install Docker + run `scripts/ch_init.sh` on the box (ClickHouse lives next to the engine).
6. **Stop the instance when not in use.** Add a tmux + `uvicorn` start command to the runbook.

## Unit tests (write first)

- `test_page_key_table`: every URL pattern in CONTRACTS §3 maps correctly; unknown domains → `"unknown"`.
- `test_bank_io_roundtrip`: random `[8, 16, 128]` f32 arrays → bytes → back, `np.array_equal` exact; wrong shape raises.
- `test_clickhouse_roundtrip` (needs Docker): insert bank row, `load_all_banks()` returns byte-identical arrays.
- `test_manifest_schema`: written manifest validates against CONTRACTS §4 shape (use a small pydantic model).

## Definition of done

- Fresh clone + `pnpm install` + `pip install -e packages/shared-py` works on a second machine.
- ClickHouse up via one script; roundtrip test green.
- EC2 box reachable, model downloaded, runbook accurate enough that Rahul can SSH in without asking questions.

## Suggested skills

`superpowers:test-driven-development`, `everything-claude-code:clickhouse-io`, `everything-claude-code:backend-patterns`
