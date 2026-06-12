#!/usr/bin/env bash
# GhostBrowser OS — demo launcher (C1 integration agent).
# Starts the full stack in dependency order with health polling.
# Usage: scripts/run_demo.sh [--ec2 <ip>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/.."

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

_fail() { echo -e "${RED}FAIL:${NC} $*"; exit 1; }
_ok()   { echo -e "${GREEN}  ok${NC}  $*"; }
_step() { echo -e "\n${CYAN}==>${NC} $*"; }

# ---- URLs ----
CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://localhost:8123}"
INFERENCE_PORT="${INFERENCE_PORT:-8000}"
INFERENCE_URL="http://localhost:${INFERENCE_PORT}"
if [[ -n "$EC2_IP" ]]; then
  INFERENCE_URL="http://${EC2_IP}:${INFERENCE_PORT}"
fi

# ---- 1. ClickHouse ----
_step "Starting ClickHouse"
bash "$SCRIPT_DIR/ch_init.sh"
_ok "ClickHouse ready at $CLICKHOUSE_URL"

# ---- 2. Upload banks ----
_step "Uploading banks to ClickHouse"
python "$SCRIPT_DIR/upload_banks.py" "$REPO_ROOT/banks"
_ok "Banks uploaded"

# ---- 3. Inference engine (EC2 path uses tmux; local path is direct) ----
_step "Starting inference engine"
if [[ -n "$EC2_IP" ]]; then
  echo "  EC2 target: $EC2_IP"
  echo "  Run on the EC2 box:"
  echo ""
  echo "    cd ghostbrowser-os && python -m inference_engine"
  echo ""
  echo "  Or in a tmux session:"
  echo "    tmux new-session -d -s ghostbrowser 'cd ghostbrowser-os && python -m inference_engine'"
  echo ""
  echo -n "  Waiting for engine at $INFERENCE_URL ... "
else
  # Local: start in background
  cd "$REPO_ROOT/apps/inference-engine"
  python -m uvicorn inference_engine.server:create_app --host 0.0.0.0 --port "$INFERENCE_PORT" &
  ENGINE_PID=$!
  trap "kill $ENGINE_PID 2>/dev/null" EXIT
  echo -n "  Waiting for engine at $INFERENCE_URL (PID $ENGINE_PID) ... "
fi

# Health-check loop
for i in $(seq 1 120); do
  if curl -fsS "$INFERENCE_URL/healthz" >/dev/null 2>&1; then
    echo ""
    HEALTH=$(curl -sS "$INFERENCE_URL/healthz")
    _ok "engine healthy: $HEALTH"
    break
  fi
  if [[ "$i" -eq 120 ]]; then
    _fail "inference engine did not become ready in time"
  fi
  sleep 2
  echo -n "."
done

# ---- 4. Web console ----
_step "Starting web console"
cd "$REPO_ROOT/apps/web-console"
if [[ -n "$EC2_IP" ]]; then
  echo "  Run on a local machine:"
  echo "    NEXT_PUBLIC_INFERENCE_WS=ws://${EC2_IP}:${INFERENCE_PORT}/ws/events pnpm dev"
else
  NEXT_PUBLIC_INFERENCE_WS="ws://localhost:${INFERENCE_PORT}/ws/events" pnpm dev &
  CONSOLE_PID=$!
  trap "kill $ENGINE_PID $CONSOLE_PID 2>/dev/null" EXIT
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
