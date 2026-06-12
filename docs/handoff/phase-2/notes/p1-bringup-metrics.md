# P1 - GPU Bring-up Metrics Baseline

Baseline for the inference engine running **Llama-3.1-8B-Instruct** in bf16 on a
single **NVIDIA A10G (24 GB)** (`g5.2xlarge`), captured at first real boot. P2
and P5 use these numbers as the reference point for regression and the demo
budget.

> Numbers below are filled in on the box. Lines marked `MEASURE` are placeholders
> until the A10G run lands; everything else is fixed by the contract/architecture.
> The exact commands to (re)capture each row are in the "How these were captured"
> section so anyone can reproduce them.

## Configuration

| Item | Value |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Precision | bf16 (banks stored f32, cast at load - CONTRACTS section 1) |
| Instance | `g5.2xlarge` (1x A10G 24 GB, 8 vCPU, 32 GB RAM) |
| MI layers | `[8, 12, 16, 20]` |
| Attention impl | `sdpa` (wrapped layers); MI math is manual |
| transformers | `4.46.*` (pinned) |

## Footprint

| Metric | Value | Notes |
|---|---|---|
| Model weights (bf16) | ~16.0 GiB | 8.03B params x 2 bytes (theoretical) |
| VRAM allocated after load | `MEASURE` GiB | `torch.cuda.memory_allocated()` post-load (logged at startup) |
| VRAM reserved after load | `MEASURE` GiB | `torch.cuda.memory_reserved()` |
| `nvidia-smi` used / total | `MEASURE` / 24576 MiB | full process footprint incl. CUDA context |
| Headroom for KV + banks | ~6-7 GiB expected | the 3 synthetic banks are < 50 MiB total |

## Latency

| Metric | Value | Notes |
|---|---|---|
| Cold start (process -> model ready) | `MEASURE` s | from the `model ready in X.Xs` startup log line |
| Load-time smoke generate (8 tokens) | `MEASURE` s | the `Say OK.` smoke in `LlamaBackend.load` |
| Single `/api/v1/step` (baseline, trivial prompt) | `MEASURE` s | scripted prompt, `max_new_tokens=256`, temp 0 |
| Single `/api/v1/step` (mi, bank injected) | `MEASURE` s | same prompt, banks at `[8,12,16,20]` |

## How these were captured

```bash
# On the box, engine running (see p1-box.md), from apps/inference-engine:

# 1. VRAM footprint after load - read straight off the startup log:
#    "model ready in 41.3s; cuda[NVIDIA A10G]: 15.9 GiB allocated, 16.4 GiB reserved"
#    and cross-check with:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 2. Cold start - the engine logs "model ready in X.Xs" once the lifespan load
#    (banks -> model -> smoke generate) finishes.

# 3. Per-step latency - trivial scripted prompt against the live server:
curl -s -w '\n-> %{time_total}s\n' -X POST http://localhost:8000/api/v1/step \
  -H 'content-type: application/json' \
  -d '{"session_id":"p1-bench","mode":"baseline","task":"Open the top story.",
       "url":"https://news.ycombinator.com/","page_key":"hn:front",
       "dom_text":"1. A story (example.com) 312 points 88 comments item?id=1",
       "dom_token_count":40,"history":[],"step":1}'

# Repeat with "mode":"mi","dom_text":null for the bank-injected number.
```

## Notes for P2 / P5

- bf16 leaves comfortable headroom on the A10G; OOM would point at fp32 leaking
  in (check the `loading ... (torch.bfloat16)` startup line) or another process
  on the GPU (`nvidia-smi`).
- These are **synthetic-bank** numbers (P1 scope). Real banks (R1) do not change
  the footprint or per-step latency materially - same shapes, same slot counts.
- The H+4 shape-sync (P2) reuses this exact boot; if cold-start regresses sharply
  after loading real banks, suspect bank dtype/shape, not the model.
