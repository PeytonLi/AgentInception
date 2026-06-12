# CONTRACTS.md — Shared Interface Contracts

Every agent reads this before writing code. **No agent changes this file unilaterally.** If a contract is wrong or incomplete, stop and flag it to the team; a silent local deviation will break someone else's component at integration time.

---

## 1. Constants

```python
MODEL_ID        = "meta-llama/Llama-3.1-8B-Instruct"
SELECTED_LAYERS = [8, 12, 16, 20]   # 0-indexed decoder layers (model has 32)
NUM_LAYERS      = 32
NUM_Q_HEADS     = 32
NUM_KV_HEADS    = 8                  # GQA: each KV head serves 4 query heads
HEAD_DIM        = 128
HIDDEN_SIZE     = 4096
BANK_DTYPE      = "float32"          # serialization dtype (model runs bfloat16; banks stored f32, cast at load)
SUMMARY_WORDS   = (200, 400)         # Haiku DOM summary target length
HAIKU_MODEL     = "claude-haiku-4-5-20251001"
TRANSFORMERS_PIN = "transformers==4.46.*"  # custom attention is version-sensitive; pin everywhere
```

---

## 2. Repo layout

```
ghostbrowser-os/
├── apps/
│   ├── inference-engine/      # Python FastAPI + MI attention        (A1)
│   ├── agent-runner/          # Python Playwright loop               (A3)
│   ├── bank-compiler/         # Python offline compiler CLI          (B1)
│   └── web-console/           # Next.js 15 dashboard                 (A4)
├── packages/
│   └── shared-py/             # page_key(), bank (de)serialization — single impl, imported by all Python apps (A2 creates, others consume)
├── demo-assets/
│   └── popup-page/            # static cookie-modal fixture page     (B2)
├── infra/
│   ├── docker-compose.yml     # ClickHouse                           (A2)
│   └── clickhouse/schema.sql                                         (A2)
├── banks/                     # compiled .bin artifacts + manifest (gitignored except manifest) (B2)
├── tests/integration/         # cross-component tests                (C1)
├── docs/handoff/              # this directory
├── turbo.json / pnpm-workspace.yaml / package.json                   (A2)
```

---

## 3. Bank identity: `page_key`, not DOM hash

Banks are looked up by **page type**, not exact DOM hash (HN comment counts vary per article; exact hashes would never match).

```python
# packages/shared-py/ghost_shared/page_key.py — THE one implementation
def page_key(url: str) -> str:
    # news.ycombinator.com/ or /news?p=N      -> "hn:front"
    # news.ycombinator.com/item?id=*          -> "hn:item"
    # localhost:*/popup* or file://*popup*    -> "popup:demo"
    # anything else                           -> "unknown"   (=> no bank; plain-prompt fallback)
```

`dom_structural_hash` (sha256 over tag-skeleton: strip scripts/styles/comments/text, keep tag names + sorted class lists in document order) is stored as **informational metadata only** — never used for lookup.

---

## 4. Bank binary format

One bank = one page type = K and V tensors for each selected layer.

- Per layer ℓ ∈ SELECTED_LAYERS, two arrays, each shape **`[NUM_KV_HEADS=8, num_slots, HEAD_DIM=128]`**, dtype float32, C-order.
- Serialized via `arr.tobytes()`; deserialized via `np.frombuffer(buf, dtype=np.float32).reshape(8, num_slots, 128)`.
- **Keys are canonical pre-RoPE** (paper Eq. 6–7): hidden states → layer's `input_layernorm` → `k_proj` / `v_proj`, reshaped to KV heads, **no rotary applied**. δ=0.
- `num_slots` is identical across the 4 layers within one bank (same kept token positions).
- File naming in `banks/`: `{page_key.replace(':','_')}__L{layer}__{k|v}.bin` (e.g. `hn_front__L8__k.bin`).

### Manifest — `banks/manifest.json`

```json
{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "selected_layers": [8, 12, 16, 20],
  "banks": [
    {
      "page_key": "hn:front",
      "domain": "news.ycombinator.com",
      "num_slots": 312,
      "dom_structural_hash": "<sha256 hex>",
      "summary_text_path": "banks/hn_front.summary.txt",
      "files": {"8": {"k": "hn_front__L8__k.bin", "v": "hn_front__L8__v.bin"}, "12": {...}, "16": {...}, "20": {...}},
      "compiled_at": "2026-06-12T18:00:00Z"
    }
  ]
}
```

The serializer/deserializer pair lives in `packages/shared-py/ghost_shared/bank_io.py`. B1 writes with it, A1/A2 read with it. Do not hand-roll a second implementation.

---

## 5. ClickHouse

```sql
CREATE DATABASE IF NOT EXISTS ghostbrowser;

CREATE TABLE IF NOT EXISTS ghostbrowser.latent_memory_banks (
    page_key            String,
    domain              String,
    layer_id            UInt32,
    num_slots           UInt32,
    k_bank              String,   -- raw float32 bytes [8, num_slots, 128]
    v_bank              String,   -- raw float32 bytes [8, num_slots, 128]
    dom_structural_hash String,
    compiled_at         DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (page_key, layer_id);

CREATE TABLE IF NOT EXISTS ghostbrowser.agent_steps (
    session_id      String,
    step            UInt32,
    mode            Enum8('baseline' = 1, 'mi' = 2),
    url             String,
    page_key        String,
    action_json     String,
    visible_tokens  UInt64,
    baseline_tokens UInt64,
    bank_found      UInt8,
    ts              DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (session_id, step);
```

Connection: `CLICKHOUSE_URL` env var, default `http://localhost:8123`. Client: `clickhouse-connect`. **At inference-engine startup, all banks are read once into an in-memory `dict[str, dict[int, tuple[Tensor, Tensor]]]` keyed by `page_key` → no DB calls during navigation.**

---

## 6. Inference HTTP API (FastAPI, port **8000**)

```
GET  /healthz                 -> {"status": "ok", "model_loaded": true, "banks_loaded": ["hn:front", ...]}

POST /api/v1/step
  Request:  {
    "session_id": str,
    "mode": "baseline" | "mi",
    "task": str,                      # global user intent
    "url": str,
    "page_key": str,                  # computed by agent-runner via shared page_key()
    "dom_text": str | null,           # REQUIRED in baseline mode (truncated ≤ 4000 tokens); null in mi mode
    "dom_token_count": int,           # Llama-tokenizer count of the FULL dom_text baseline WOULD have sent (runner computes in both modes; feeds the comparison chart)
    "history": [str],                 # prior action strings (mi mode keeps this; it stays tiny)
    "step": int
  }
  Response: {
    "action": <Action JSON, §8>,
    "bank_found": bool,               # mi mode: was a bank injected for this page_key
    "injected_layers": [int],         # [] when no bank
    "visible_tokens": int,            # prompt tokens actually sent to the model this step
    "baseline_tokens": int            # echo of request dom_token_count + prompt overhead (baseline mode: == visible_tokens)
  }

WS   /ws/events               -> server pushes events (§7); web-console is the only consumer
```

`bank_found=false` (e.g. external article page, `page_key="unknown"`) → engine silently falls back to including `dom_text` for that step. This **is** the graceful-fallback demo moment, not an error.

---

## 7. WebSocket event schema (`/ws/events`)

All events: `{"type": str, "ts": iso8601, ...}`. Types:

```json
{"type": "layer_injection", "layers": [8,12,16,20], "active": true,  "page_key": "hn:front", "num_slots": 312}
{"type": "layer_injection", "layers": [],           "active": false, "page_key": "unknown",  "num_slots": 0}
{"type": "token_metrics",   "session_id": "...", "step": 3, "mode": "mi", "visible_tokens": 212, "baseline_tokens": 14200, "cum_visible": 650, "cum_baseline": 41200, "kv_savings_ratio": 63.4}
{"type": "action",          "step": 3, "action": {<Action JSON>}}
{"type": "viewport_frame",  "jpeg_base64": "<base64>"}
{"type": "log",             "level": "info", "message": "Bank hn:item injected at layers [8, 12, 16, 20]"}
```

`viewport_frame` is pushed by **agent-runner** via `POST /internal/frame` on the inference engine (engine just rebroadcasts on the WS). Frame cadence: every 300 ms, JPEG quality 50, viewport 1280×720.

---

## 8. Action JSON (model output ↔ agent-runner)

The model must answer with exactly one JSON object, no prose:

```json
{"action": "goto",    "url": "https://news.ycombinator.com/item?id=123"}
{"action": "click",   "selector": "a.morelink"}
{"action": "dismiss_modal", "selector": "#accept-cookies"}
{"action": "extract", "result": {"score": 312, "top_commenters": ["a", "b", "c"]}}
{"action": "done",    "result": {...final answer...}}
```

Agent-runner parses with one retry on malformed JSON (re-prompt: "Respond with only the JSON object."). Two failures → log + abort step.

---

## 9. Ports & env

| Service | Port | Env var |
|---|---|---|
| inference-engine (HTTP + WS) | 8000 | `INFERENCE_URL=http://<ec2-ip>:8000` |
| web-console | 3000 | `NEXT_PUBLIC_INFERENCE_WS=ws://<ec2-ip>:8000/ws/events` |
| ClickHouse | 8123 (HTTP) / 9000 (native) | `CLICKHOUSE_URL=http://localhost:8123` |

Secrets (`HF_TOKEN`, `ANTHROPIC_API_KEY`) via env only. Every Python app ships `.env.example` with names, never values.

---

## 10. Mocks (what lets agents work in parallel)

- **A3 (agent-runner)** develops against `tests/mocks/mock_inference.py` — a 30-line FastAPI stub returning scripted Action JSON per step. Contract: identical to §6.
- **A4 (web-console)** develops against `tests/mocks/mock_ws_feed.py` — replays a canned sequence of §7 events on a loop at `ws://localhost:8000/ws/events`.
- **A1 (inference-engine)** develops against `tests/fixtures/tiny_bank/` — a random-valued but shape-correct bank (8, 16, 128) per layer, generated by a fixture script. Real banks arrive at the H+4 shape sync.
- **B1 (bank-compiler)** needs nothing from anyone — only the model, this contract, and `shared-py`. If `shared-py` isn't merged yet, B1 implements `bank_io.py` + `page_key.py` THERE FIRST and A2 adopts it (coordinate in chat — exactly one implementation may exist).
