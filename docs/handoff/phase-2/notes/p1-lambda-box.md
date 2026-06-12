# P1 - The Lambda GPU Box (live reference for teammates)

The single A10 box that runs the inference engine + ClickHouse. Shared by R1
(compile banks) and the engine (serve). **Do not Terminate it until R1 confirms
it has the model cache it needs.** Lambda has no Stop — only Terminate.

## Where it is

| Item | Value |
|---|---|
| Public IP | `MEASURE` (fill in after launch; teammates: `INFERENCE_URL=http://<ip>:8000`) |
| Instance type | `gpu_1x_a10` (1× A10 24 GB) |
| Image | Lambda Stack (PyTorch) on Ubuntu 22.04 |
| Disk | 200 GB ephemeral (attached file system at `/home/ubuntu/data` for persistence) |
| SSH | `ssh ubuntu@<ip>` (key registered in Lambda dashboard; no .pem file) |
| Firewall | No security groups — manage on-instance: `sudo ufw allow 8000/tcp` |

WS for the console: `NEXT_PUBLIC_INFERENCE_WS=ws://<ip>:8000/ws/events`.

## Where things live on the box

| Thing | Path |
|---|---|
| Repo | `~/agentinception` |
| Python venv | `~/agentinception/.venv` |
| HF model cache | `/home/ubuntu/data/hf-cache` (on Lambda persistent file system) |
| ClickHouse data | Docker named volume from `infra/docker-compose.yml` (ephemeral — lost on terminate) |
| HF download log | `~/hf_download.log` |

## Boot sequence (cold box → /healthz green)

```bash
ssh ubuntu@<ip>
cd ~/agentinception

# 0. sanity: GPU visible
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"   # -> True

# 1. persistent HF cache (Lambda file system)
export HF_HOME=/home/ubuntu/data/hf-cache
export HF_TOKEN=hf_xxx CLICKHOUSE_URL=http://localhost:8123

# 2. venv
source .venv/bin/activate     # first time: see aws-runbook.md section 4 to create it

# 3. warm the model cache FIRST (skips if already downloaded; ~16 GB cold)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
# HF_HOME points to the persistent file system, so this survives terminates

# 4. ClickHouse up + schema
bash scripts/ch_init.sh        # -> lists agent_steps, latent_memory_banks

# 5. banks. P1 uses synthetic banks (3 page_keys: hn:front, hn:item, popup:demo).
#    R1 replaces these with real banks later; load path is identical.
python3 scripts/build_demo_banks.py --force-synthetic      # writes banks/
python3 scripts/upload_banks.py banks/                     # -> ClickHouse

# 6. open the firewall (if not already done)
sudo ufw allow 8000/tcp

# 7. boot the engine (either form is equivalent)
cd apps/inference-engine
python3 -m inference_engine.server
#   or:
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
ClickHouse was unreachable and the manifest fallback found nothing — re-run step 5.

## Run the GPU smoke test on the box

```bash
cd ~/agentinception/apps/inference-engine
pip3 install -e .[dev]
HF_TOKEN=$HF_TOKEN pytest tests/test_gpu_boot.py -m gpu -s
# off-GPU this same test is collected and skipped, so CI stays green.
```

## Stop / start (save money — ~$0.60/hr)

Lambda has **no Stop** — only Terminate. The persistent file system at
`/home/ubuntu/data` survives terminates, so the HF model cache is preserved.
ClickHouse data (Docker volume) is **ephemeral** and must be re-created.

```bash
# graceful: stop the engine (Ctrl-c in tmux), tear down ClickHouse:
docker compose -f infra/docker-compose.yml down
```

Then **Terminate** the instance in the Lambda dashboard. On next launch, re-run
the boot sequence: step 3 is a fast no-op because the HF cache is on the
persistent file system. Steps 4–5 (ClickHouse + banks) must be re-run fresh.

> Reminder: keep the instance up until R1 has pulled the model cache it needs.

## Using run_demo.sh with Lambda

The `--ec2` flag is legacy-named but works with any remote IP, including Lambda:

```bash
scripts/run_demo.sh --ec2 <lambda-ip>
```

It polls ClickHouse + engine health over `http://<lambda-ip>` just like EC2.
