# AgentInception

A web agent that replaces verbose DOM text in the prompt with pre-computed
latent **KV banks** injected at selected transformer layers, based on Memory
Inception (MI). See `docs/handoff/` for the full build plan and
`docs/handoff/CONTRACTS.md` for the shared interface contracts.

## Monorepo layout

```
apps/
  inference-engine/   FastAPI + MI attention at layers [8,12,16,20]  (A1)
  agent-runner/       Playwright loop, baseline + MI modes           (A3)
  bank-compiler/      Offline DOM -> Haiku -> Llama -> KV banks       (B1)
  web-console/        Next.js 15 live dashboard                      (A4)
packages/
  shared-py/          page_key, bank_io, dom_hash, ClickHouse client (A2)
infra/                docker-compose + ClickHouse schema             (A2)
scripts/              ch_init.sh, upload_banks.py                    (A2)
banks/                compiled .bin artifacts + manifest.json        (B2)
tests/                integration suite, mocks, fixtures             (C1)
```

## Quick start

```bash
# 1. Node workspace (turbo + web-console)
corepack enable && corepack prepare pnpm@10.30.3 --activate
pnpm install

# 2. Shared Python package (imported by every Python app)
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e packages/shared-py

# 3. ClickHouse (Docker) + schema, idempotent
bash scripts/ch_init.sh

# 4. Run the shared-py test suite
python -m pytest packages/shared-py/tests
```

Each app has its own `requirements.txt` and `.env.example` (copy to `.env`).
Per-app instructions live in `docs/handoff/`.

## Turbo tasks

```bash
pnpm build   # turbo run build across all packages
pnpm test    # turbo run test across all packages
pnpm dev     # turbo run dev (persistent: web-console, engine)
```

## Environment variables (CONTRACTS.md §9)

| Var | Used by | Notes |
|-----|---------|-------|
| `HF_TOKEN` | engine, compiler | gated Llama-3.1-8B access |
| `ANTHROPIC_API_KEY` | bank-compiler | Haiku DOM summary |
| `CLICKHOUSE_URL` | engine, scripts | default `http://localhost:8123` |
| `INFERENCE_URL` | runner, console | default `http://localhost:8000` |

Secrets go in `.env` (gitignored) — never commit values.

## Infra / AWS

ClickHouse runs locally via Docker (`infra/docker-compose.yml`). The GPU
inference box is documented in `docs/handoff/peyton/aws-runbook.md`.
