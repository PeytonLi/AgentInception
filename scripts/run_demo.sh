#!/usr/bin/env bash
# GhostBrowser OS / AgentInception — demo launcher (P5, hardened).
# Brings the full stack up in dependency order with preflight checks and
# fail-fast health polling: ClickHouse -> upload banks -> engine -> health
# (asserts all 3 banks preloaded) -> console -> ready-to-paste runner commands.
# Usage: scripts/run_demo.sh [--ec2 <ip>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Parse args ----
EC2_IP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ec2) EC2_IP="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

# ---- Colours ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

_fail() { echo -e "${RED}FAIL:${NC} $*" >&2; exit 1; }
_warn() { echo -e "${YELLOW}warn:${NC} $*" >&2; }
_ok()   { echo -e "${GREEN}  ok${NC}  $*"; }
_step() { echo -e "\n${CYAN}==>${NC} $*"; }
_need() { command -v "$1" >/dev/null 2>&1 || _fail "'$1' not found on PATH — $2"; }

# ---- URLs ----
CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://localhost:8123}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
INFERENCE_URL="http://localhost:${INFERENCE_PORT}"
if [[ -n "$EC2_IP" ]]; then
  INFERENCE_URL="http://${EC2_IP}:${INFERENCE_PORT}"
fi

# ---- 0. Preflight: required tooling + demo artifacts ----
_step "Preflight checks"
_need curl "needed for health polling"
_need python "needed to upload banks and (locally) run the engine"
if [[ -z "$EC2_IP" ]]; then
  _need docker "needed to run ClickHouse locally (or set CLICKHOUSE_URL to a remote instance)"
  _need pnpm  "needed to start the web console"
fi

# Banks must exist AND carry real (non-empty) .bin blobs before we upload.
MANIFEST="$REPO_ROOT/banks/manifest.json"
[[ -f "$MANIFEST" ]] || _fail "missing $MANIFEST — compile banks (R1) or run scripts/build_demo_banks.py first"
python - "$REPO_ROOT/banks" <<'PY' || _fail "bank artifacts incomplete — see message above"
import json, os, sys
banks_dir = sys.argv[1]
manifest = json.load(open(os.path.join(banks_dir, "manifest.json")))
required = {"hn:front", "hn:item", "popup:demo"}
have = {b["page_key"] for b in manifest.get("banks", [])}
missing = required - have
if missing:
    print(f"  manifest is missing page_keys: {sorted(missing)}", file=sys.stderr)
    sys.exit(1)
for b in manifest["banks"]:
    for layer, names in b["files"].items():
        for kind in ("k", "v"):
            p = os.path.join(banks_dir, names[kind])
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                print(f"  missing/empty bank blob: {p}", file=sys.stderr)
                print("  banks are gitignored; copy the .bin files onto this box "
                      "(scp/S3) or recompile them (R1).", file=sys.stderr)
                sys.exit(1)
print(f"  banks present: {sorted(have)}")
PY
_ok "demo artifacts present"

# ---- 1. ClickHouse ----
_step "Starting ClickHouse"
if [[ -n "$EC2_IP" ]]; then
  curl -fsS "$CLICKHOUSE_URL/ping" >/dev/null 2>&1 \
    || _fail "ClickHouse not reachable at $CLICKHOUSE_URL — start it (scripts/ch_init.sh) before an --ec2 demo"
else
  bash "$SCRIPT_DIR/ch_init.sh" || _fail "ClickHouse failed to start — check 'docker compose' and port 8123"
fi
_ok "ClickHouse ready at $CLICKHOUSE_URL"

# ---- 2. Upload banks ----
_step "Uploading banks to ClickHouse"
python "$SCRIPT_DIR/upload_banks.py" "$REPO_ROOT/banks" \
  || _fail "bank upload failed — is the ClickHouse schema applied? (scripts/ch_init.sh)"
_ok "Banks uploaded"

# ---- 3. Inference engine (EC2 path uses tmux; local path is direct) ----
_step "Starting inference engine"
if [[ -n "$EC2_IP" ]]; then
  echo "  EC2 target: $EC2_IP — run the engine on the GPU box, e.g.:"
  echo ""
  echo "    tmux new-session -d -s ghostbrowser \\"
  echo "      'cd ghostbrowser-os/apps/inference-engine && python -m inference_engine'"
  echo ""
  echo -n "  Waiting for engine at $INFERENCE_URL ... "
else
  # Local: start in background. create_app is a factory, so pass --factory.
  cd "$REPO_ROOT/apps/inference-engine"
  python -m uvicorn inference_engine.server:create_app --factory \
    --host 0.0.0.0 --port "$INFERENCE_PORT" &
  ENGINE_PID=$!
  trap 'kill ${ENGINE_PID:-} ${CONSOLE_PID:-} 2>/dev/null' EXIT
  echo -n "  Waiting for engine at $INFERENCE_URL (PID $ENGINE_PID) ... "
fi

# Health-check loop (model + bank load can take a couple of minutes cold).
HEALTH=""
for i in $(seq 1 150); do
  if HEALTH=$(curl -fsS "$INFERENCE_URL/healthz" 2>/dev/null); then
    echo ""
    _ok "engine answered /healthz"
    break
  fi
  if [[ "$i" -eq 150 ]]; then
    echo ""
    _fail "inference engine did not become ready in time at $INFERENCE_URL
       - local: check the uvicorn log above (model download? HF_TOKEN? CUDA OOM?)
       - --ec2: confirm the tmux engine is running and port $INFERENCE_PORT is open"
  fi
  sleep 2
  echo -n "."
done

# Fail fast unless all 3 banks actually preloaded (startup-order / bank bug).
echo "$HEALTH" | python - <<'PY' || _fail "engine is up but banks did not preload — check ClickHouse upload + BANKS_DIR"
import json, sys
data = json.load(sys.stdin)
loaded = set(data.get("banks_loaded", []))
required = {"hn:front", "hn:item", "popup:demo"}
missing = required - loaded
if not data.get("model_loaded"):
    print("  /healthz reports model_loaded=false", file=sys.stderr); sys.exit(1)
if missing:
    print(f"  /healthz banks_loaded missing {sorted(missing)} (have {sorted(loaded)})",
          file=sys.stderr)
    sys.exit(1)
print(f"  engine healthy: model_loaded=true banks_loaded={sorted(loaded)}")
PY
_ok "all 3 banks preloaded"

# ---- 4. Web console ----
_step "Starting web console"
cd "$REPO_ROOT/apps/web-console"
if [[ -n "$EC2_IP" ]]; then
  echo "  Run on a local machine:"
  echo "    NEXT_PUBLIC_INFERENCE_WS=ws://${EC2_IP}:${INFERENCE_PORT}/ws/events pnpm dev"
else
  NEXT_PUBLIC_INFERENCE_WS="ws://localhost:${INFERENCE_PORT}/ws/events" pnpm dev &
  CONSOLE_PID=$!
  _ok "console starting at http://localhost:3000"
fi

# ---- 5. Print runner commands ----
DEFAULT_TASK="Find the top story about AI on the Hacker News front page (scan up to 2 pages), open its comment page, and extract the story score and the top 3 commenter usernames."

echo ""
echo "=========================================================="
echo "  GhostBrowser OS Demo Stack — READY"
echo "=========================================================="
echo ""
echo "  Engine:   $INFERENCE_URL"
echo "  Console:  http://localhost:3000"
echo "  ClickHouse: $CLICKHOUSE_URL"
echo ""
echo "  --- BASELINE RUN ---"
echo "  cd apps/agent-runner"
echo "  python -m agent_runner \\"
echo "    --mode=baseline \\"
echo "    --task='$DEFAULT_TASK' \\"
echo "    --start-url=https://news.ycombinator.com \\"
echo "    --session-id=demo-baseline-001"
echo ""
echo "  --- MI RUN ---"
echo "  cd apps/agent-runner"
echo "  python -m agent_runner \\"
echo "    --mode=mi \\"
echo "    --task='$DEFAULT_TASK' \\"
echo "    --start-url=https://news.ycombinator.com \\"
echo "    --session-id=demo-mi-001"
echo ""
echo "  Press Ctrl+C to stop the stack."

# Keep running until interrupted
wait
