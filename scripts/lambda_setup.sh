#!/usr/bin/env bash
# AgentInception / AgentInception — all-in-one Lambda Labs GPU box setup.
# Idempotent: safe to run multiple times. Fails fast on missing prerequisites.
# Usage:
#   scripts/lambda_setup.sh
#   scripts/lambda_setup.sh --hf-home /home/ubuntu/data/hf-cache
#   scripts/lambda_setup.sh --hf-home /home/ubuntu/data/hf-cache --skip-model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Parse args ----
HF_HOME_ARG=""
SKIP_MODEL=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-home)   HF_HOME_ARG="$2"; shift 2 ;;
    --skip-model) SKIP_MODEL=true; shift ;;
    *) echo "Unknown arg: $1"; echo "Usage: $0 [--hf-home <path>] [--skip-model]"; exit 2 ;;
  esac
done

# ---- Colours (match scripts/run_demo.sh) ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

_fail()  { echo -e "${RED}FAIL:${NC} $*" >&2; exit 1; }
_warn()  { echo -e "${YELLOW}warn:${NC} $*" >&2; }
_ok()    { echo -e "${GREEN}  ok${NC}  $*"; }
_step()  { echo -e "\n${CYAN}==>${NC} $*"; }
_info()  { echo -e "      $*"; }
_need()  { command -v "$1" >/dev/null 2>&1 || _fail "'$1' not found on PATH — $2"; }

# ---- 0. Preflight: is this a Lambda GPU box? ----
_step "Preflight checks"

_need nvidia-smi "Lambda Stack should ship with NVIDIA drivers. Wrong image?"
_need python3    "Lambda Stack ships python3 (not python). Wrong image?"
_need docker     "Lambda Stack should ship Docker. Install with: sudo apt-get install -y docker.io"

# GPU visible?
if ! nvidia-smi >/dev/null 2>&1; then
  _fail "nvidia-smi failed — GPU not visible. Is this a gpu_1x_a10 instance?"
fi
_ok "nvidia-smi: GPU visible"

# Torch CUDA?
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  _fail "torch.cuda.is_available() returned False — re-launch with Lambda Stack (PyTorch®) image"
fi
_ok "torch.cuda.is_available(): True"

# Docker running?
if ! docker info >/dev/null 2>&1; then
  _warn "Docker daemon not running, attempting to start..."
  sudo systemctl start docker || _fail "Cannot start Docker. Check: sudo systemctl status docker"
fi
_ok "Docker daemon running"

# ---- 1. HF_HOME — persistent model cache ----
_step "HF_HOME setup"

if [[ -n "$HF_HOME_ARG" ]]; then
  export HF_HOME="$HF_HOME_ARG"
  mkdir -p "$HF_HOME"
  _ok "HF_HOME=$HF_HOME (persistent file system)"

  # Persist across shells
  if ! grep -q "export HF_HOME=" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
    _ok "Added HF_HOME to ~/.bashrc"
  fi
else
  export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  _warn "No --hf-home provided. HF_HOME=$HF_HOME (ephemeral — lost on terminate!)"
  _info "Re-run with: $0 --hf-home /home/ubuntu/data/hf-cache"
fi

# ---- 2. Python venv ----
_step "Python virtual environment"

cd "$REPO_ROOT"

if [[ -d ".venv" ]]; then
  _ok ".venv already exists — skipping creation"
else
  python3 -m venv .venv
  _ok ".venv created"
fi

# Activate for the remainder of this script
source .venv/bin/activate
_info "venv activated ($(which python3))"

# ---- 3. Install Python packages ----
_step "Python package installation"

# shared-py (editable)
_info "Installing shared-py (editable)..."
pip install -e packages/shared-py -q 2>&1 | tail -1
_ok "shared-py installed"

# inference-engine
_info "Installing inference-engine dependencies..."
pip install -r apps/inference-engine/requirements.txt -q 2>&1 | tail -1
_ok "inference-engine deps installed"

# bank-compiler (runtime extras)
_info "Installing bank-compiler..."
pip install -r apps/bank-compiler/requirements.txt -q 2>/dev/null || true
pip install -e "apps/bank-compiler[runtime]" -q 2>&1 | tail -1
_ok "bank-compiler installed"

# huggingface_hub CLI (for model download)
pip install -U "huggingface_hub[cli]" -q 2>&1 | tail -1
_ok "huggingface_hub[cli] up to date"

# ---- 4. Model download ----
_step "Model download (meta-llama/Llama-3.1-8B-Instruct)"

if [[ "$SKIP_MODEL" == true ]]; then
  _warn "--skip-model set — skipping download"
else
  MODEL_DIR="$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct"
  if [[ -d "$MODEL_DIR" ]] && [[ -f "$MODEL_DIR/snapshots/"*/config.json 2>/dev/null ]]; then
    _ok "Model already cached at $MODEL_DIR — skipping download"
  else
    _info "Model NOT cached — starting download (~16 GB). This is the long pole..."

    if [[ -z "${HF_TOKEN:-}" ]]; then
      _fail "HF_TOKEN not set. Export it before running this script.\n       The model is gated — accept Meta's license at huggingface.co first."
    fi

    huggingface-cli login --token "$HF_TOKEN" 2>&1 | tail -1

    _info "Downloading meta-llama/Llama-3.1-8B-Instruct (this may take 5–15 min)..."
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct 2>&1 | tail -3
    _ok "Model downloaded"
  fi
fi

# ---- 5. ClickHouse ----
_step "ClickHouse"

if curl -fsS "${CLICKHOUSE_URL:-http://localhost:8123}/ping" >/dev/null 2>&1; then
  _ok "ClickHouse already running at ${CLICKHOUSE_URL:-http://localhost:8123}"
else
  _info "Starting ClickHouse via ch_init.sh..."
  export CLICKHOUSE_URL="${CLICKHOUSE_URL:-http://localhost:8123}"
  bash "$SCRIPT_DIR/ch_init.sh" || _fail "ch_init.sh failed — check Docker and port 8123"
  _ok "ClickHouse ready at $CLICKHOUSE_URL"
fi

# ---- 6. Banks ----
_step "Banks"

MANIFEST="$REPO_ROOT/banks/manifest.json"
NEED_REAL=false
if [[ -f "$MANIFEST" ]]; then
  # Check if real .bin files exist
  if python3 -c "
import json, os
manifest = json.load(open('$MANIFEST'))
for b in manifest.get('banks', []):
    for layer, names in b.get('files', {}).items():
        for kind in ('k', 'v'):
            p = os.path.join('$REPO_ROOT/banks', names[kind])
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                raise SystemExit(1)
" 2>/dev/null; then
    NEED_REAL=false
    _ok "Real compiled banks found — uploading..."
  else
    NEED_REAL=true
  fi
else
  NEED_REAL=true
fi

if [[ "$NEED_REAL" == true ]]; then
  _warn "No real compiled banks found at banks/ — building synthetic demo banks"
  python3 "$SCRIPT_DIR/build_demo_banks.py" || _fail "build_demo_banks.py failed"
  _ok "Synthetic demo banks built"
fi

# Upload to ClickHouse (idempotent — safe to re-run)
python3 "$SCRIPT_DIR/upload_banks.py" "$REPO_ROOT/banks" \
  || _fail "bank upload failed — is the ClickHouse schema applied?"
_ok "Banks uploaded to ClickHouse"

# ---- 7. Print boot instructions ----
ENGINE_DIR="$REPO_ROOT/apps/inference-engine"

_step "Setup complete — ready to boot"

echo ""
echo "=========================================================="
echo "  AgentInception Lambda Box — SETUP COMPLETE"
echo "=========================================================="
echo ""
echo "  HF_HOME:    $HF_HOME"
echo "  ClickHouse: ${CLICKHOUSE_URL:-http://localhost:8123}"
echo ""
echo "  --- Boot the inference engine ---"
echo ""
echo "  tmux new -s engine"
echo "  source .venv/bin/activate"
echo "  export HF_TOKEN=hf_xxx CLICKHOUSE_URL=http://localhost:8123"
echo "  cd apps/inference-engine"
echo "  uvicorn src.main:app --host 0.0.0.0 --port 8000"
echo "  # detach: Ctrl-b then d"
echo ""
echo "  --- Verify /healthz ---"
echo ""
echo "  curl http://localhost:8000/healthz"
echo ""
echo "  --- GPU smoke test ---"
echo ""
echo "  cd apps/inference-engine && pytest -m gpu"
echo ""
echo "=========================================================="
