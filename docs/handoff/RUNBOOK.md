# Demo Runbook — AgentInception / AgentInception

**Owner:** P5 integration  **Last rehearsed:** see `KNOWN_ISSUES.md`
**Hardware target:** AWS g5.2xlarge (1× A10G 24 GB) + local browser machine

---

## 0. Prerequisites

| What | Where |
|---|---|
| EC2 `g5.2xlarge`, CUDA 12.x, Docker, Python 3.11+, 60 GB disk | AWS us-east-1 (or nearest region) |
| `HF_TOKEN` with gated access to `meta-llama/Llama-3.1-8B-Instruct` | SSM / env on the EC2 box |
| Compiled bank `.bin` blobs for `hn:front`, `hn:item`, `popup:demo` | `banks/` on the EC2 box (gitignored; transfer via scp or recompile via R1) |
| `pnpm` (for the web console) | local demo machine |
| Live internet (HN scrape) | both boxes |

**One-time EC2 setup:**
```bash
# Python deps
cd apps/inference-engine && pip install -r requirements.txt
cd apps/bank-compiler && pip install -r requirements.txt
# ClickHouse
docker compose -f infra/docker-compose.yml up -d
bash scripts/ch_init.sh
# Banks: either scp the .bin blobs from a prior R1 run, or recompile them:
#   cd apps/bank-compiler && python -m bank_compiler compile ...
# Upload:
python scripts/upload_banks.py banks/
```

---

## 1. Quick start (cold box -> demo-ready in N commands)

```bash
# 1. Clone and cd
git clone <repo-url> agentinception && cd agentinception

# 2. Set secrets
export HF_TOKEN="hf_..."

# 3. Upload banks (if .bin blobs are already in banks/)
python scripts/upload_banks.py banks/

# 4. Boot the stack (all-in-one)
bash scripts/run_demo.sh
```

The script will:
- validate tooling and bank artifacts
- start ClickHouse (docker)
- upload banks
- start the inference engine (uvicorn)
- health-poll and assert all 3 banks preloaded
- start the web console (pnpm dev)
- print the ready-to-paste runner commands

**If engine is on a remote GPU box:**
```bash
# On the EC2 GPU box:
cd agentinception
tmux new-session -d -s agentinception \
  'cd apps/inference-engine && python -m inference_engine'

# On the local demo machine:
bash scripts/run_demo.sh --ec2 <ec2-public-ip>
```

---

## 2. Demo stage sequence

Open **four terminal windows / tmux panes:**

### Pane 1 — Stack monitor
```bash
bash scripts/run_demo.sh [--ec2 <ip>]
```
Keeps the engine + ClickHouse + console alive. Leave it running.

### Pane 2 — Baseline run
```bash
cd apps/agent-runner
INFERENCE_URL=http://localhost:8000 python -m agent_runner \
  --mode=baseline \
  --task='Find the top story about AI on the Hacker News front page (scan up to 2 pages), open its comment page, and extract the story score and the top 3 commenter usernames.' \
  --start-url=https://news.ycombinator.com \
  --session-id=demo-baseline-001
```
**Expected:** cum_visible ~= cum_baseline. ~10k-15k tokens per step.

### Pane 3 — MI run
```bash
cd apps/agent-runner
INFERENCE_URL=http://localhost:8000 python -m agent_runner \
  --mode=mi \
  --task='Find the top story about AI on the Hacker News front page (scan up to 2 pages), open its comment page, and extract the story score and the top 3 commenter usernames.' \
  --start-url=https://news.ycombinator.com \
  --session-id=demo-mi-001
```
**Expected:** cum_visible stays flat (<=1000 tokens). cum_baseline grows linearly.
kv_savings_ratio >= 20.

### Pane 4 — Web console
Opens at `http://localhost:3000`. Walk the audience through 4 panels:
1. **Live Viewport Mirror** - frames from the runner
2. **Token Cost Comparator** - baseline vs MI bar chart
3. **Layer Injection Graph** - which layers are active per step
4. **Logs & Math** - raw event stream + kv_savings_ratio

### Popup chaos bonus
```bash
# Serve the fixture page locally and run the agent:
python -m http.server 8080 --directory demo-assets/popup-page &
cd apps/agent-runner
INFERENCE_URL=http://localhost:8000 python -m agent_runner \
  --mode=mi \
  --task='Extract the key statistic shown on the page.' \
  --start-url=http://localhost:8080/popup.html \
  --session-id=demo-popup-001
```
The model dismisses the cookie modal **because of** the `popup:demo` bank.

### Demo close
- Point to the savings chart: "40x fewer visible tokens, same task completed."
- **Stop the EC2 instance** — a g5.2xlarge costs ~$1.50/hr.
- `Ctrl+C` the `run_demo.sh` pane to tear down the local stack.

---

## 3. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Engine times out on /healthz | Model download on first boot | Wait 3-5 min; check `HF_TOKEN` |
| `/healthz` `model_loaded=false` | CUDA OOM or missing HF_TOKEN | `nvidia-smi`; verify token |
| `/healthz` `banks_loaded` missing | ClickHouse upload failed or BANKS_DIR wrong | Re-run upload |
| Runner exits instantly | Wrong `INFERENCE_URL` | `curl $INFERENCE_URL/healthz` |
| `run_demo.sh` preflight fails | Missing .bin blobs or tools | Read the error message |
| Console shows empty panels | Engine not broadcasting | Confirm health assertion passed |
| Popup modal not dismissed | popup:demo bank not loaded | Check `/healthz banks_loaded` |
| HN page links broken | Live site drift | Adjust --task selectors |

---

## 4. Backup recording

- Record the demo with OBS or QuickTime (full screen, 5 min max).
- Upload and link in the P5 PR description.
- If live HN flakes, the recording IS the backup demo.

---

## 5. Stop-the-instance reminder

```
aws ec2 stop-instances --instance-ids i-xxxxxxxxx --region us-east-1
```

A g5.2xlarge left running overnight burns ~$36. The demo stack leaves no
persistent data; nothing is lost by stopping.
