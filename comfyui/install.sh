#!/usr/bin/env bash
set -euo pipefail

# Zero-disk ComfyUI bootstrap installer for Google Colab (Linux).
# Usage: COMFY_DIR=/content/ComfyUI SHM_ROOT=/dev/shm bash comfyui/install.sh

COMFY_DIR="${COMFY_DIR:-/content/ComfyUI}"
SHM_ROOT="${SHM_ROOT:-/dev/shm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

log() {
  echo "[comfyui-install] $*"
}

pip_install() {
  "$PYTHON_BIN" -m pip install --quiet --no-cache-dir "$@" 2>/dev/null \
    || "$PYTHON_BIN" -m pip install --quiet --no-cache-dir --break-system-packages "$@"
}

log "Preparing RAM disk directories under $SHM_ROOT..."
mkdir -p "$SHM_ROOT/comfy-out" "$SHM_ROOT/comfy-tmp"

if [ ! -d "$COMFY_DIR/.git" ]; then
  log "Cloning ComfyUI into $COMFY_DIR..."
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
  log "Updating existing ComfyUI at $COMFY_DIR (>= v0.37.0 required for Qwen-Image-2.1 / MOSS-Image)..."
  if ! git -C "$COMFY_DIR" pull --ff-only >/dev/null 2>&1; then
    git -C "$COMFY_DIR" fetch origin master --depth 1
    git -C "$COMFY_DIR" reset --hard origin/master
  fi
fi

mkdir -p \
  "$COMFY_DIR/models/diffusion_models" \
  "$COMFY_DIR/models/text_encoders" \
  "$COMFY_DIR/models/vae" \
  "$COMFY_DIR/models/loras"

log "Installing Hugging Face Xet transfer tools and YAML parser..."
pip_install huggingface_hub hf_xet pyyaml

if [ -f "$COMFY_DIR/requirements.txt" ]; then
  log "Installing ComfyUI Python requirements..."
  pip_install -r "$COMFY_DIR/requirements.txt"
fi

export HF_XET_HIGH_PERFORMANCE=1
log "ComfyUI installation complete: $(git -C "$COMFY_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "Zero-disk dirs ready: $SHM_ROOT/comfy-out , $SHM_ROOT/comfy-tmp"
