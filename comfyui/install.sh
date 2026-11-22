#!/usr/bin/env bash
set -euo pipefail

# Zero-disk ComfyUI bootstrap installer for Google Colab (Linux).
# Usage: COMFY_DIR=/content/ComfyUI SHM_ROOT=/dev/shm bash comfyui/install.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMFY_DIR="${COMFY_DIR:-/content/ComfyUI}"
SHM_ROOT="${SHM_ROOT:-/dev/shm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_KJNODES="${INSTALL_KJNODES:-0}"
KJNODES_REPO_URL="${KJNODES_REPO_URL:-https://github.com/kijai/ComfyUI-KJNodes.git}"
UPGRADE_TORCH_CU130="${UPGRADE_TORCH_CU130:-0}"
INSTALL_WORKFLOWS="${INSTALL_WORKFLOWS:-1}"

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
  log "Updating existing ComfyUI at $COMFY_DIR (>= v0.38.0 recommended for MiniMax H3 / Qwen-Image-2.1 / MOSS-Image)..."
  if ! git -C "$COMFY_DIR" pull --ff-only >/dev/null 2>&1; then
    git -C "$COMFY_DIR" fetch origin master --depth 1
    git -C "$COMFY_DIR" reset --hard origin/master
  fi
fi

mkdir -p \
  "$COMFY_DIR/models/diffusion_models" \
  "$COMFY_DIR/models/text_encoders" \
  "$COMFY_DIR/models/vae" \
  "$COMFY_DIR/models/loras" \
  "$COMFY_DIR/models/embeddings" \
  "$COMFY_DIR/models/model_patches"

if ! command -v ffmpeg >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  log "Installing ffmpeg for video/audio muxing..."
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends ffmpeg
fi

if [ "$UPGRADE_TORCH_CU130" = "1" ]; then
  log "Upgrading PyTorch stack to CUDA 13.0 (cu130)..."
  pip_install --upgrade --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
fi

log "Installing Hugging Face Xet transfer tools and YAML parser..."
pip_install --upgrade huggingface_hub hf_xet pyyaml

if [ -f "$COMFY_DIR/requirements.txt" ]; then
  log "Installing ComfyUI Python requirements..."
  pip_install -r "$COMFY_DIR/requirements.txt"
fi

if [ "$INSTALL_KJNODES" = "1" ]; then
  KJNODES_DIR="$COMFY_DIR/custom_nodes/ComfyUI-KJNodes"
  mkdir -p "$COMFY_DIR/custom_nodes"
  if [ ! -d "$KJNODES_DIR/.git" ]; then
    log "Cloning ComfyUI-KJNodes into $KJNODES_DIR..."
    git clone --depth 1 "$KJNODES_REPO_URL" "$KJNODES_DIR"
  else
    log "Updating existing ComfyUI-KJNodes at $KJNODES_DIR..."
    if ! git -C "$KJNODES_DIR" pull --ff-only >/dev/null 2>&1; then
      git -C "$KJNODES_DIR" fetch origin --depth 1
      git -C "$KJNODES_DIR" reset --hard FETCH_HEAD
    fi
  fi
  if [ -f "$KJNODES_DIR/requirements.txt" ]; then
    log "Installing ComfyUI-KJNodes Python requirements..."
    pip_install -r "$KJNODES_DIR/requirements.txt"
  fi
fi

if [ "$INSTALL_WORKFLOWS" = "1" ] && [ -d "$SCRIPT_DIR/workflows" ]; then
  copied_wf=0
  for wf_file in "$SCRIPT_DIR/workflows/"*.json; do
    [ -f "$wf_file" ] || continue
    if [ "$copied_wf" -eq 0 ]; then
      mkdir -p "$COMFY_DIR/user/default/workflows"
    fi
    cp -f "$wf_file" "$COMFY_DIR/user/default/workflows/"
    copied_wf=$((copied_wf + 1))
  done
  if [ "$copied_wf" -gt 0 ]; then
    log "Installed $copied_wf bundled workflow(s) to $COMFY_DIR/user/default/workflows"
  fi
fi

"$PYTHON_BIN" - "$COMFY_DIR" <<'PYEOF' || true
import importlib.metadata as meta
import shutil
import sys
from pathlib import Path

comfy_dir = Path(sys.argv[1])
parts = []
try:
    import torch

    parts.append(f"torch={torch.__version__}")
    parts.append(f"cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        parts.append(f"gpu={props.name} ({vram_gb:.1f} GB)")
    else:
        parts.append("gpu=unavailable")
except Exception as exc:
    parts.append(f"torch=error({exc})")

for pkg in ("comfy-kitchen", "av"):
    try:
        parts.append(f"{pkg}={meta.version(pkg)}")
    except Exception:
        parts.append(f"{pkg}=not-installed")

parts.append(f"ffmpeg={shutil.which('ffmpeg') or 'missing'}")
kj_installed = (comfy_dir / "custom_nodes" / "ComfyUI-KJNodes").is_dir()
parts.append(f"KJNodes={'installed' if kj_installed else 'disabled'}")
print("[comfyui-install] Environment:", " | ".join(parts))
PYEOF

export HF_XET_HIGH_PERFORMANCE=1
log "ComfyUI installation complete: $(git -C "$COMFY_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "Zero-disk dirs ready: $SHM_ROOT/comfy-out , $SHM_ROOT/comfy-tmp"
