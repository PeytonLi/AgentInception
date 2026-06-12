# P1 - The GPU Box (live reference for teammates)

The single A10G box that runs the inference engine + ClickHouse. Shared by R1
(compile banks) and the engine (serve). **Do not Stop/Terminate it until R1
confirms it has the model cache it needs.**

## Where it is

| Item | Value |
|---|---|
| Public IP | `MEASURE` (fill in after launch; teammates: `INFERENCE_URL=http://<ip>:8000`) |
| Instance type | `g5.2xlarge` (1x A10G 24 GB) |
| AMI | AWS Deep Learning AMI GPU PyTorch (Ubuntu 22.04) |
| Root volume | 200 GB gp3 |
| SSH | `ssh -i <key>.pem ubuntu@<ip>` |
| Security group | 22 from my IP/32; 8000 to teammates; 8123/9000 closed (localhost only) |

WS for the console: `NEXT_PUBLIC_INFERENCE_WS=ws://<ip>:8000/ws/events`.

## Where things live on the box

| Thing | Path |
|---|---|
| Repo | `~/ghostbrowser-os` |
| Python venv | `~/ghostbrowser-os/.venv` |
| HF model cache | `~/.cache/huggingface/hub` (survives Stop; on the 200 GB EBS root) |
| ClickHouse data | Docker named volume from `infra/docker-compose.yml` (survives Stop) |
| HF download log | `~/hf_download.log` |

## Boot sequence (cold box -> /healthz green)

```bash
ssh -i <key>.pem ubuntu@<ip>
cd ~/ghostbrowser-os

# 0. sanity: GPU visible
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"   # -> True

# 1. secrets + venv
export HF_TOKEN=hf_xxx CLICKHOUSE_URL=http://localhost:8123
source .venv/bin/activate     # first time: see aws-runbook.md section 4 to create it

# 2. warm the model cache FIRST (skips if already downloaded; ~16 GB cold)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# 3. ClickHouse up + schema
bash scripts/ch_init.sh        # -> lists agent_steps, latent_memory_banks

# 4. banks. P1 uses synthetic banks (3 page_keys: hn:front, hn:item, popup:demo).
#    R1 replaces these with real banks later; load path is identical.
python scripts/build_demo_banks.py --force-synthetic      # writes banks/
python scripts/upload_banks.py banks/                     # -> ClickHouse

# 5. boot the engine (either form is equivalent)
cd apps/inference-engine
python -m inference_engine.server
#   or, matching aws-runbook.md section 6:
#   uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Run it under tmux so it survives the SSH session:

```bash
tmux new -s engine
# ... boot command ...
# detach: Ctrl-b then d ; reattach: tmux attach -t engine
```

## Verify (from your laptop)

```bash
curl http://<ip>:8000/healthz
# -> {"status":"ok","model_loaded":true,"banks_loaded":["hn:front","hn:item","popup:demo"]}
```

`model_loaded: true` + 3 page_keys = bring-up done. If `banks_loaded` is empty,
ClickHouse was unreachable and the manifest fallback found nothing - re-run step 4.

## Run the GPU smoke test on the box

```bash
cd ~/ghostbrowser-os/apps/inference-engine
pip install -e .[dev]
HF_TOKEN=$HF_TOKEN pytest tests/test_gpu_boot.py -m gpu -s
# off-GPU this same test is collected and skipped, so CI stays green.
```

## Stop / start (save money - ~$1.21/hr)

```bash
# graceful: stop the engine (Ctrl-c in tmux), keep ClickHouse volume:
docker compose -f infra/docker-compose.yml down
```

Then **Stop** (not Terminate) the instance in the AWS console. The EBS root
volume - HF model cache + ClickHouse data - survives. On next start, re-run the
boot sequence; step 2 is a fast no-op because the cache is warm.

> Reminder: keep the instance up until R1 has pulled the model cache it needs.
