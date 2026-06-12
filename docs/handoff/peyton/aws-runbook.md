# AWS EC2 Runbook — Inference Box (A2)

The single GPU box that runs the inference engine + ClickHouse. Llama-3.1-8B
in bf16 fits comfortably on one A10G (24 GB).

> Secrets (`HF_TOKEN`, `ANTHROPIC_API_KEY`) live in your shell / `.env` only —
> never commit them. **Stop the instance when not in use** (~$1.21/hr).

---

## 1. Launch the instance

- **Type:** `g5.2xlarge` (1× NVIDIA A10G 24 GB, 8 vCPU, 32 GB RAM).
- **AMI:** *AWS Deep Learning AMI GPU PyTorch* (Ubuntu 22.04). Ships with
  CUDA + a working PyTorch, so no driver wrangling.
- **Root volume:** 200 GB gp3 (model is ~16 GB; leave room for HF cache,
  Docker images, ClickHouse data).
- **Key pair:** use or create one you control; you'll SSH with it.

### Security group

| Port | Source | Purpose |
|------|--------|---------|
| 22   | your IP(s) only (`x.x.x.x/32`) | SSH |
| 8000 | your IP(s) / teammates | inference HTTP + WS |
| 3000 | your IP(s) (optional) | web-console, if hosted here |

ClickHouse ports (8123/9000) stay **closed to the internet** — the engine
reaches it on `localhost`.

---

## 2. First-boot sanity

```bash
ssh -i <key>.pem ubuntu@<ec2-ip>

nvidia-smi                                   # A10G visible, driver loaded
python -c "import torch; print(torch.cuda.is_available())"   # -> True
```

If `torch.cuda.is_available()` is `False`, you booted the wrong AMI — relaunch
with the Deep Learning AMI.

---

## 3. Start the model download FIRST (it's the long pole)

Llama-3.1-8B-Instruct is **gated** — accept Meta's license on Hugging Face and
confirm your account is approved *before* the hackathon starts.

```bash
export HF_TOKEN=hf_xxx                        # do NOT commit this
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token "$HF_TOKEN"

# Kick this off immediately, in the background, before anything else (~16 GB):
nohup huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  > ~/hf_download.log 2>&1 &
tail -f ~/hf_download.log
```

---

## 4. Clone the repo + install

```bash
git clone <repo-url> ghostbrowser-os && cd ghostbrowser-os

# Node toolchain (for the web-console / turbo, if running here)
corepack enable && corepack prepare pnpm@10.30.3 --activate
pnpm install

# Python: one virtualenv, install the engine's deps (pulls in shared-py editable)
python -m venv .venv && source .venv/bin/activate
pip install -e packages/shared-py
pip install -r apps/inference-engine/requirements.txt
```

---

## 5. ClickHouse (lives next to the engine)

Docker is preinstalled on the DL AMI. If not:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

Bring it up + apply schema (idempotent):

```bash
export CLICKHOUSE_URL=http://localhost:8123
bash scripts/ch_init.sh
# -> SHOW TABLES FROM ghostbrowser lists: agent_steps, latent_memory_banks
```

Load the compiled banks (Rahul scp's `banks/*.bin` + `manifest.json` over):

```bash
python scripts/upload_banks.py banks/
```

---

## 6. Run the inference engine

```bash
tmux new -s engine
source .venv/bin/activate
export HF_TOKEN=hf_xxx CLICKHOUSE_URL=http://localhost:8123
cd apps/inference-engine
uvicorn src.main:app --host 0.0.0.0 --port 8000
# detach: Ctrl-b then d
```

Smoke check from your laptop:

```bash
curl http://<ec2-ip>:8000/healthz
# -> {"status":"ok","model_loaded":true,"banks_loaded":["hn:front", ...]}
```

---

## 7. Shut down (save money)

```bash
docker compose -f infra/docker-compose.yml down     # keeps the named volume
```

Then **Stop** (not Terminate) the instance from the AWS console so the EBS
volume — model cache, ClickHouse data — survives until the next session.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `huggingface-cli download` 401/403 | License not accepted or token lacks gated access. Re-accept on the model page; regenerate token with *read* scope. |
| `torch.cuda.is_available()` False | Wrong AMI. Use the Deep Learning AMI GPU PyTorch. |
| `ch_init.sh` hangs on ping | Docker not running / port 8123 taken. `docker ps`, `sudo systemctl start docker`. |
| OOM loading model | Confirm bf16 (not fp32) and nothing else on the GPU (`nvidia-smi`). |
| Engine can't reach ClickHouse | `CLICKHOUSE_URL` unset or container down. `curl localhost:8123/ping`. |
