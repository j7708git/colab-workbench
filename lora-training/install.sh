#!/usr/bin/env bash
set -euo pipefail

# Fizgig (MOSS-Image LoRA training & Web UI) bootstrap installer for Google Colab (Linux).
# Usage: FIZGIG_DIR=/content/Fizgig FIZGIG_REF=v9.7.7 bash lora-training/install.sh

FIZGIG_DIR="${FIZGIG_DIR:-/content/Fizgig}"
FIZGIG_REF="${FIZGIG_REF:-v9.7.7}"
FIZGIG_REPO_URL="${FIZGIG_REPO_URL:-https://github.com/shootthesound/Fizgig.git}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHM_ROOT="${SHM_ROOT:-/dev/shm}"

log() {
  echo "[fizgig-install] $*"
}

pip_install_host() {
  "$PYTHON_BIN" -m pip install --quiet --no-cache-dir "$@" 2>/dev/null \
    || "$PYTHON_BIN" -m pip install --quiet --no-cache-dir --break-system-packages "$@"
}

log "Preparing workspace directories..."
mkdir -p "$SHM_ROOT/fizgig-tmp" "$FIZGIG_DIR/user_data/models"

if [ ! -d "$FIZGIG_DIR/.git" ]; then
  log "Cloning Fizgig ($FIZGIG_REF) into $FIZGIG_DIR..."
  if [ -e "$FIZGIG_DIR" ]; then
    tmp_models=""
    if [ -d "$FIZGIG_DIR/user_data/models" ] && [ -n "$(ls -A "$FIZGIG_DIR/user_data/models" 2>/dev/null || true)" ]; then
      tmp_models="$(mktemp -d)"
      mv "$FIZGIG_DIR/user_data/models" "$tmp_models/models"
    fi
    rm -rf "$FIZGIG_DIR"
    git clone --branch "$FIZGIG_REF" --depth 1 "$FIZGIG_REPO_URL" "$FIZGIG_DIR"
    mkdir -p "$FIZGIG_DIR/user_data/models"
    if [ -n "$tmp_models" ]; then
      mv "$tmp_models/models/"* "$FIZGIG_DIR/user_data/models/" 2>/dev/null || true
      rm -rf "$tmp_models"
    fi
  else
    git clone --branch "$FIZGIG_REF" --depth 1 "$FIZGIG_REPO_URL" "$FIZGIG_DIR"
  fi
else
  log "Updating existing Fizgig checkout to $FIZGIG_REF..."
  git -C "$FIZGIG_DIR" fetch --depth 1 origin "$FIZGIG_REF"
  git -C "$FIZGIG_DIR" checkout -f FETCH_HEAD
fi

mkdir -p "$FIZGIG_DIR/user_data/models" "$FIZGIG_DIR/user_data/cache" "$FIZGIG_DIR/output_loras"

has_gcc=0
command -v gcc >/dev/null 2>&1 && has_gcc=1
has_pyh="$("$PYTHON_BIN" -c "import sysconfig, os; print(int(os.path.exists(os.path.join(sysconfig.get_path('include'), 'Python.h'))))")"
if [ "$has_gcc" -ne 1 ] || [ "$has_pyh" -ne 1 ]; then
  log "Installing C/Python headers for native extensions (insightface)..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends build-essential python3-dev
fi

log "Installing host transfer helpers (huggingface_hub, hf_xet, pyyaml, uv)..."
pip_install_host --upgrade huggingface_hub hf_xet pyyaml uv

# Colab containers may set UV_SYSTEM_PYTHON=1, which causes `uv pip install` to target
# /usr/local instead of the isolated venv. Always clear UV_SYSTEM_PYTHON first.
unset UV_SYSTEM_PYTHON UV_PYTHON || true

if [ ! -x "$FIZGIG_DIR/venv/bin/python" ]; then
  log "Creating isolated Fizgig venv at $FIZGIG_DIR/venv..."
  "$PYTHON_BIN" -m uv venv --seed "$FIZGIG_DIR/venv"
fi

export VIRTUAL_ENV="$FIZGIG_DIR/venv"
export UV_PYTHON="$FIZGIG_DIR/venv/bin/python"
export PATH="$FIZGIG_DIR/venv/bin:$PATH"
export DISABLE_CUDA=1

log "Installing Fizgig requirements via uv_install_deps.py..."
"$VIRTUAL_ENV/bin/python" -m pip install --quiet --upgrade pip uv
"$VIRTUAL_ENV/bin/python" "$FIZGIG_DIR/uv_install_deps.py" "$FIZGIG_DIR/requirements.txt" "$VIRTUAL_ENV"

check_cuda_venv() {
  "$VIRTUAL_ENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; x=torch.randn(8,8,device='cuda',dtype=torch.bfloat16); print(f'torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} bf16={(x@x).shape}')"
}

if command -v nvidia-smi >/dev/null 2>&1; then
  log "Verifying CUDA and bf16 tensor cores in Fizgig venv..."
  if ! check_cuda_venv; then
    fallback_ok=0
    for wheel_spec in \
      "cu132|torch==2.15.0|torchvision==0.29.2" \
      "cu130|torch==2.14.*|torchvision" \
      "cu128|torch==2.9.*|torchvision"; do
      IFS="|" read -r cuda_tag torch_pkg tv_pkg <<< "$wheel_spec"
      log "cu134 check failed; trying PyTorch $cuda_tag fallback..."
      if "$VIRTUAL_ENV/bin/python" -m uv pip install --reinstall \
        --index-url "https://download.pytorch.org/whl/$cuda_tag" \
        "$torch_pkg" "$tv_pkg" && check_cuda_venv; then
        fallback_ok=1
        break
      fi
    done
    if [ "$fallback_ok" -ne 1 ]; then
      log "ERROR: Failed to initialize CUDA in Fizgig venv across cu134/cu132/cu130/cu128."
      exit 1
    fi
  fi
fi

"$VIRTUAL_ENV/bin/python" -c "import accelerate, diffusers, transformers, safetensors, bitsandbytes, insightface, hqq; print('[fizgig-install] Core training & curation imports verified.')"
log "Fizgig installation complete: $(git -C "$FIZGIG_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
