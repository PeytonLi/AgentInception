# Lambda Labs Runbook — Inference Box

The single GPU box that runs the inference engine + ClickHouse. Llama-3.1-8B
in bf16 fits comfortably on one A10 (24 GB). Lambda Labs (lambda.ai) is the
lower-cost alternative to EC2 — same GPU tier at roughly half the hourly rate.

> Secrets (`HF_TOKEN`, `ANTHROPIC_API_KEY`) live in your shell / `.env` only —
> never commit them. **Terminate the instance when not in use** (~$0.60/hr).
> Instances are ephemeral — disk is destroyed on terminate unless you use the
> persistent file system add-on (see §2).

---

## 1. Pricing & instance types

| Type | GPU | VRAM | vCPUs | RAM | $/hr | Notes |
|------|-----|------|-------|-----|------|-------|
| `gpu_1x_a10` | 1× A10 | 24 GB | 30 | 200 GB | ~$0.60 | Sweet spot — same GPU as EC2 `g5.2xlarge` at half the price |
| `gpu_1x_a100_sxm4` | 1× A100 SXM4 | 80 GB | 30 | 200 GB | ~$1.10 | Overkill for 8B, needed only for 70B+ or heavy batch throughput |

For Llama-3.1-8B bf16, **`gpu_1x_a10` is all you need.**

---

## 2. Persistent storage strategy

Lambda instances have **no persistent root disk** — the 200 GB system disk is
destroyed on terminate. To persist your HF model cache across launch/terminate
cycles, attach a **Lambda persistent file system**:

- Available sizes: 200 GB – 2 TB. 200 GB is enough (model ~16 GB + cache).
- It **auto-mounts** at `/home/ubuntu/data` on every launch.
- Add it when creating the instance via Dashboard or API (`"file_system_names"`).

**What goes on the persistent file system (`/home/ubuntu/data/`):**

```
/home/ubuntu/data/
├── hf-cache/          # HF_HOME — model weights, tokenizers, hub cache
└── clickhouse-data/   # (optional) bind-mount for Docker named volume
```

Without a persistent file system, you re-download the 16 GB model on every
launch. With it, a warm launch skips the download entirely.

---

## 3. Launch an instance

### 3a. Dashboard (point-and-click)

1. Go to <https://cloud.lambdalabs.com/instances> → **Launch Instance**.
2. **Instance type:** `gpu_1x_a10`.
3. **Region:** pick the one with available capacity (US West / Central).
4. **Image:** *Lambda Stack (PyTorch®)* — ships Ubuntu 22.04, CUDA 12.x,
   PyTorch 2.x, Docker.
5. **SSH key:** add/select your registered public key (same one you use for
   GitHub). Lambda does **not** use `.pem` files — the key is attached to your
   account.
6. **File system:** attach your persistent file system (see §2) or create one.
7. Click **Launch**. Copy the public IP once assigned (~30 seconds).

### 3b. Cloud API (programmatic)

Lambda Cloud API docs: <https://docs.lambdalabs.com/cloud-api>. Auth is a
bearer token in the `Authorization` header.

**List available instance types:**

```bash
curl -sH "Authorization: Bearer $LAMBDA_API_KEY" \
  https://cloud.lambdalabs.com/api/v1/instance-types \
  | python3 -m json.tool
```

**Launch a `gpu_1x_a10` with persistent file system:**

```bash
curl -sH "Authorization: Bearer $LAMBDA_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
  -d '{
    "region_name": "us-west-1",
    "instance_type_name": "gpu_1x_a10",
    "ssh_key_names": ["my-laptop-key"],
    "file_system_names": ["agentinception-hf-cache"],
    "name": "agentinception-inference"
  }' | python3 -m json.tool
```

**List your running instances:**

```bash
curl -sH "Authorization: Bearer $LAMBDA_API_KEY" \
  https://cloud.lambdalabs.com/api/v1/instances \
  | python3 -m json.tool
```

**SSH in (user is `ubuntu`, uses your registered key):**

```bash
ssh ubuntu@<lambda-ip>
```

---

## 4. Firewall (ufw)

Lambda has **no security-group abstraction** — you manage the firewall directly
on the instance with `ufw`. Lock it down immediately after first boot:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing

sudo ufw allow 22/tcp     # SSH
sudo ufw allow 8000/tcp   # inference HTTP + WS
sudo ufw allow 3000/tcp   # web-console (optional, if hosted here)

sudo ufw enable
sudo ufw status verbose
```

ClickHouse ports (8123/9000) are **not** opened — the engine reaches it on
`localhost`.

---

## 5. First-boot sanity check

```bash
ssh ubuntu@<lambda-ip>

nvidia-smi                                   # A10 visible, driver loaded
python3 -c "import torch; print(torch.cuda.is_available())"   # -> True
```

If `torch.cuda.is_available()` is `False`, you launched the wrong image —
re-launch with *Lambda Stack (PyTorch®)*.

Note: Lambda Stack ships `python3` (not `python`). All commands in this
runbook use `python3`.

---

## 6. One-command setup

The repo ships an all-in-one setup script that performs every step below:

```bash
# Clone the repo first, then:
bash agentinception/scripts/lambda_setup.sh

# If using a persistent HF cache mount:
bash agentinception/scripts/lambda_setup.sh --hf-home /home/ubuntu/data/hf-cache

# Skip model download (already cached):
bash agentinception/scripts/lambda_setup.sh --hf-home /home/ubuntu/data/hf-cache --skip-model
```

It is **idempotent** — safe to re-run. See §7 for the manual step-by-step
equivalent.

---

## 7. Manual boot sequence (step by step)

### 7a. HF_HOME — point at the persistent cache

```bash
# If you attached a persistent file system at /home/ubuntu/data:
export HF_HOME=/home/ubuntu/data/hf-cache
mkdir -p "$HF_HOME"
```

Set this in your `.bashrc` so future shells pick it up:

```bash
echo 'export HF_HOME=/home/ubuntu/data/hf-cache' >> ~/.bashrc
```

### 7b. Clone the repo + Python venv

```bash
git clone <repo-url> agentinception && cd agentinception

python3 -m venv .venv && source .venv/bin/activate

# Install shared-py (editable, used by all Python packages)
pip install -e packages/shared-py

# Install inference-engine deps
pip install -r apps/inference-engine/requirements.txt

# Install bank-compiler (for building synthetic / recompiling banks)
pip install -r apps/bank-compiler/requirements.txt 2>/dev/null || true
pip install -e "apps/bank-compiler[runtime]"
```

### 7c. Download the model (the long pole — start this first)

Llama-3.1-8B-Instruct is **gated** — accept Meta's license on Hugging Face
and confirm your account is approved *before* the hackathon starts.

```bash
export HF_TOKEN=hf_xxx                        # do NOT commit this
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token "$HF_TOKEN"

# Kick this off immediately, in the background:
nohup huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  > ~/hf_download.log 2>&1 &
tail -f ~/hf_download.log
```

### 7d. Docker + ClickHouse

Docker is preinstalled on Lambda Stack. If not:

```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

Bring it up + apply schema (idempotent):

```bash
export CLICKHOUSE_URL=http://localhost:8123
bash scripts/ch_init.sh
# -> SHOW TABLES FROM agentinception lists: agent_steps, latent_memory_banks
```

### 7e. Banks

**Option A — synthetic/dev banks (no real compilation needed):**

```bash
python3 scripts/build_demo_banks.py
python3 scripts/upload_banks.py banks/
```

**Option B — real compiled banks (Rahul ships `banks/*.bin` + `manifest.json`):**

```bash
# On Rahul's machine:
scp -r banks/ ubuntu@<lambda-ip>:agentinception/banks/

# On the Lambda box:
python3 scripts/upload_banks.py banks/
```

### 7f. Boot the inference engine

```bash
tmux new -s engine
source .venv/bin/activate
export HF_TOKEN=hf_xxx CLICKHOUSE_URL=http://localhost:8123
cd apps/inference-engine
uvicorn src.main:app --host 0.0.0.0 --port 8000
# detach: Ctrl-b then d
```

### 7g. Verify /healthz

```bash
curl http://localhost:8000/healthz
# -> {"status":"ok","model_loaded":true,"banks_loaded":["hn:front", ...]}
```

### 7h. GPU smoke test

```bash
cd apps/inference-engine
pytest -m gpu
# Expect: tests that exercise real model + bank IO pass
```

---

## 8. Connecting from your local machine

Set the engine URL on your laptop, then run the demo launcher with `--ec2`
(works for any remote IP, Lambda or EC2):

```bash
# On your local machine:
export INFERENCE_URL=http://<lambda-ip>:8000
export NEXT_PUBLIC_INFERENCE_WS=ws://<lambda-ip>:8000/ws/events

# Bring up the local stack (ClickHouse if local, web-console):
bash scripts/run_demo.sh --ec2 <lambda-ip>
```

> The `--ec2` flag is legacy naming — it works with **any remote inference
> host** (Lambda, EC2, RunPod, etc.). It skips local engine + ClickHouse
> startup and health-checks the remote engine at `INFERENCE_URL`.

---

## 9. Shutdown procedure

**Lambda has no "Stop" — only Terminate.** The root disk is destroyed.

Before terminating:

```bash
# Shut down containers gracefully
docker compose -f infra/docker-compose.yml down
```

Then terminate via **one** of:

| Method | Command |
|--------|---------|
| Dashboard | Instances → select → **Terminate** |
| Cloud API | `curl -X POST -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -d '{"instance_ids": ["<instance-id>"]}'` |

**Your persistent file system survives** — model cache, any data you placed
under `/home/ubuntu/data/` will be there on the next launch.

---

## 10. EC2 vs Lambda Labs

| Concern | EC2 | Lambda Labs |
|---------|-----|-------------|
| GPU tier | `g5.2xlarge` — 1× A10G 24 GB | `gpu_1x_a10` — 1× A10 24 GB |
| $/hr | ~$1.21 | ~$0.60 |
| OS image | AWS DL AMI GPU PyTorch (Ubuntu 22.04) | Lambda Stack (PyTorch®) (Ubuntu 22.04) |
| CUDA + PyTorch | Preinstalled | Preinstalled |
| Docker | Preinstalled | Preinstalled |
| Python command | `python` | `python3` |
| SSH user | `ubuntu` (with `.pem` key) | `ubuntu` (with registered public key) |
| Firewall | AWS security groups | `ufw` on the instance |
| Persistent disk | EBS volume (survives Stop) | Lambda persistent file system (mounts at `/home/ubuntu/data`) |
| Stop / Terminate | Stop (EBS persists) or Terminate (EBS destroyed) | Terminate only (root disk destroyed; file system persists) |
| API | AWS CLI / boto3 | Lambda Cloud API (bearer token) |
| Capacity | Generally available | Can be out of stock (check multiple regions) |

---

## 11. Where things live on the box

| Path | What |
|------|------|
| `~/agentinception/` | Repo root |
| `~/agentinception/.venv/` | Python venv (all packages installed here) |
| `~/agentinception/apps/inference-engine/` | Engine source + tests |
| `~/agentinception/apps/bank-compiler/` | Offline bank compiler |
| `~/agentinception/packages/shared-py/` | Shared contract code (editable install) |
| `~/agentinception/banks/` | Compiled `.bin` blobs + `manifest.json` |
| `~/agentinception/scripts/ch_init.sh` | Idempotent ClickHouse init |
| `~/agentinception/scripts/upload_banks.py` | Upload banks → ClickHouse |
| `~/agentinception/scripts/build_demo_banks.py` | Build synthetic dev banks |
| `~/agentinception/scripts/lambda_setup.sh` | All-in-one Lambda setup |
| `~/agentinception/infra/docker-compose.yml` | ClickHouse service definition |
| `/home/ubuntu/data/hf-cache/` | `HF_HOME` — model weights + hub cache (persistent) |
| `/home/ubuntu/data/clickhouse-data/` | (optional) ClickHouse named volume bind-mount |

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `huggingface-cli download` 401/403 | License not accepted or token lacks gated access. Re-accept on the model page; regenerate token with *read* scope. |
| `torch.cuda.is_available()` False | Wrong image. Use *Lambda Stack (PyTorch®)*. |
| `python3` not found / `python` errors | Lambda Stack ships `python3` — do not use bare `python`. |
| `ch_init.sh` hangs on ping | Docker not running / port 8123 taken. `docker ps`, `sudo systemctl start docker`. |
| OOM loading model | Confirm bf16 (not fp32) and nothing else on the GPU (`nvidia-smi`). |
| Engine can't reach ClickHouse | `CLICKHOUSE_URL` unset or container down. `curl localhost:8123/ping`. |
| `ufw` blocks inference traffic | Run `sudo ufw allow 8000/tcp`. Verify with `sudo ufw status`. |
| Persistent file system not mounted | Check Dashboard → File Systems → verify it's attached. Mounts at `/home/ubuntu/data`. |
| Lambda API returns 401 | `LAMBDA_API_KEY` is wrong or expired. Regenerate at <https://cloud.lambdalabs.com/api-keys>. |
| `No instances available` on launch | Region is out of capacity. Try another region (`us-west-1`, `us-central-1`, etc.). |
| `ssh: connect to host ... port 22: Connection refused` | Instance still booting (wait ~60 s). Or SSH key not registered — check Dashboard → SSH Keys. |
| `--ec2` flag with Lambda IP | The flag is legacy-named but works with any remote IP — just pass the Lambda IP. |
