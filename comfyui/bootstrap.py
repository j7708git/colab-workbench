"""Zero-disk ComfyUI bootstrapper and model preset downloader for Google Colab."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from shared.hf_download import download_hf_file
from shared.tunnel import TunnelProcess, start_cloudflared_tunnel

DEFAULT_COMFY_DIR = Path("/content/ComfyUI")
DEFAULT_SHM_ROOT = Path("/dev/shm")


def ensure_comfyui_package_ready(
    repo_dir: str | Path = "/content/colab-workbench",
) -> Path:
    """Ensure ``repo_dir`` is first on ``sys.path`` and clear stale namespace caches."""
    repo_path = Path(repo_dir)
    init_file = repo_path / "comfyui" / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(
            f"Missing {init_file}; ensure the repository is checked out on a branch containing the comfyui package."
        )

    repo_str = str(repo_path)
    sys.path[:] = [repo_str, *[p for p in sys.path if p != repo_str]]
    importlib.invalidate_caches()

    resolved_init = init_file.resolve()
    cached = sys.modules.get("comfyui")
    if cached is not None:
        cached_file = getattr(cached, "__file__", None)
        needs_reset = (
            not isinstance(cached_file, (str, os.PathLike))
            or not hasattr(cached, "download_preset_models")
            or Path(cached_file).resolve() != resolved_init
        )
        if needs_reset:
            for mod_name in list(sys.modules):
                if mod_name == "comfyui" or mod_name.startswith("comfyui."):
                    del sys.modules[mod_name]

    return resolved_init


@dataclass(frozen=True)
class ModelEntry:
    """Descriptor for a model artifact hosted on Hugging Face Hub."""

    name: str
    repo_id: str
    filename: str
    target_subdir: str | None = None
    optional: bool = False

    def dest_rel_path(self) -> Path:
        """Return the relative path under ``ComfyUI/models``."""
        src_path = Path(self.filename)
        if not self.target_subdir:
            return src_path
        parts = src_path.parts
        if parts and parts[0] == self.target_subdir:
            return src_path
        return Path(self.target_subdir) / src_path.name


QWEN21_TURBO_INT8_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        name="qwen2.1-turbo-int8-diffusion",
        repo_id="Comfy-Org/Qwen-Image-2.1",
        filename="diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
        target_subdir="diffusion_models",
    ),
    ModelEntry(
        name="qwen3vl-8b-int8-text-encoder",
        repo_id="Comfy-Org/Qwen-Image-2.1",
        filename="text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        target_subdir="text_encoders",
    ),
    ModelEntry(
        name="qwen2.1-vae-bf16",
        repo_id="Comfy-Org/Qwen-Image-2.1",
        filename="vae/qwen_image_2.1_vae_bf16.safetensors",
        target_subdir="vae",
    ),
)

MOSS_SFT_INT8_MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        name="moss-image-sft-int8-diffusion",
        repo_id="Comfy-Org/MOSS-Image",
        filename="diffusion_models/moss_image_sft_int8_convrot.safetensors",
        target_subdir="diffusion_models",
    ),
    ModelEntry(
        name="qwen3.5-9b-int8-text-encoder",
        repo_id="Comfy-Org/MOSS-Image",
        filename="text_encoders/qwen3.5_9b_int8_convrot.safetensors",
        target_subdir="text_encoders",
    ),
    ModelEntry(
        name="flux2-vae",
        repo_id="Comfy-Org/MOSS-Image",
        filename="vae/flux2-vae.safetensors",
        target_subdir="vae",
    ),
    ModelEntry(
        name="moss-image-sft-4steps-lora",
        repo_id="Kijai/MOSS-image_experimental",
        filename="loras/moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors",
        target_subdir="loras",
    ),
)

MODEL_PRESETS: dict[str, tuple[ModelEntry, ...]] = {
    "qwen21_turbo_int8": QWEN21_TURBO_INT8_MODELS,
    "moss_sft_int8": MOSS_SFT_INT8_MODELS,
    "both": QWEN21_TURBO_INT8_MODELS + MOSS_SFT_INT8_MODELS,
}


def resolve_model_preset(preset: str = "both") -> list[ModelEntry]:
    """Return the model entries matching a named preset."""
    key = preset.strip().lower()
    if key not in MODEL_PRESETS:
        valid = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset '{preset}'. Expected one of: {valid}")
    return list(MODEL_PRESETS[key])


def _parse_yaml_scalar(raw: str) -> object:
    text = raw.strip()
    if text in {"", "null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if len(text) >= 2 and (
        (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'")
    ):
        return text[1:-1]
    return text


def _strip_yaml_comment(line: str) -> str:
    in_quote: str | None = None
    for idx, ch in enumerate(line):
        if ch in {'"', "'"}:
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx]
    return line


def _parse_models_yaml_fallback(text: str) -> list[dict[str, object]]:
    """Minimal stdlib parser for ``models:`` lists when PyYAML is absent."""
    items: list[dict[str, object]] = []
    in_models = False
    models_indent = -1
    current: dict[str, object] | None = None

    for raw_line in text.splitlines():
        cleaned = _strip_yaml_comment(raw_line).rstrip()
        if not cleaned.strip():
            continue

        indent = len(cleaned) - len(cleaned.lstrip(" "))
        stripped = cleaned.lstrip(" ")

        if not in_models:
            if stripped == "models:":
                in_models = True
                models_indent = indent
            continue

        if indent <= models_indent and not stripped.startswith("- "):
            break

        if stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {}
            rest = stripped[2:].strip()
            if ":" in rest:
                key, val = rest.split(":", 1)
                current[key.strip()] = _parse_yaml_scalar(val)
            continue

        if ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val_stripped = val.strip()
            if val_stripped == "" and current is None:
                # Section header such as "qwen21_turbo_int8:"
                continue
            if current is not None:
                current[key] = _parse_yaml_scalar(val_stripped)

    if current is not None:
        items.append(current)
    return items


def _entry_from_mapping(raw: dict[str, object]) -> ModelEntry:
    for required_key in ("name", "repo_id", "filename"):
        val = raw.get(required_key)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Model YAML entry missing required field '{required_key}': {raw}")

    target_subdir = raw.get("target_subdir")
    if target_subdir is not None and not isinstance(target_subdir, str):
        raise ValueError(f"Invalid target_subdir in model entry: {raw}")

    return ModelEntry(
        name=raw["name"].strip(),  # type: ignore[union-attr]
        repo_id=raw["repo_id"].strip(),  # type: ignore[union-attr]
        filename=raw["filename"].strip(),  # type: ignore[union-attr]
        target_subdir=target_subdir.strip() if target_subdir else None,
        optional=bool(raw.get("optional", False)),
    )


def load_models_from_yaml(yaml_path: str | Path) -> list[ModelEntry]:
    """Load model entries from a ``models.yaml`` file."""
    text = Path(yaml_path).read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ImportError:
        raw_items = _parse_models_yaml_fallback(text)
        return [_entry_from_mapping(item) for item in raw_items]

    data = yaml.safe_load(text) or {}
    models_section = data.get("models") if isinstance(data, dict) else None
    if models_section is None:
        return []

    raw_items = []
    if isinstance(models_section, list):
        raw_items.extend(models_section)
    elif isinstance(models_section, dict):
        for group_items in models_section.values():
            if isinstance(group_items, list):
                raw_items.extend(group_items)
    else:
        raise ValueError("Top-level 'models' key must map to a dict or list.")

    return [_entry_from_mapping(item) for item in raw_items if isinstance(item, dict)]


def download_preset_models(
    comfy_dir: str | Path = DEFAULT_COMFY_DIR,
    *,
    preset: str = "both",
    extra_entries: Iterable[ModelEntry] | None = None,
    token: str | None = None,
) -> list[Path]:
    """Download preset (and optional extra) models into ``<comfy_dir>/models``."""
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    combined: list[ModelEntry] = []
    seen_rel_paths: set[Path] = set()
    for entry in [*resolve_model_preset(preset), *(extra_entries or ())]:
        rel = entry.dest_rel_path()
        if rel not in seen_rel_paths:
            seen_rel_paths.add(rel)
            combined.append(entry)

    models_root = Path(comfy_dir) / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    downloaded_paths: list[Path] = []

    for entry in combined:
        rel = entry.dest_rel_path()
        dest = models_root / rel
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[comfyui] already present, skip: {rel}", flush=True)
            downloaded_paths.append(dest)
            continue

        print(f"[comfyui] downloading {entry.name} from {entry.repo_id}...", flush=True)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if Path(entry.filename) == rel:
                got = Path(
                    download_hf_file(
                        entry.repo_id,
                        entry.filename,
                        local_dir=models_root,
                        token=token,
                    )
                )
            else:
                with tempfile.TemporaryDirectory(dir=models_root, prefix=".hf-dl-") as tmp_dir:
                    tmp_file = Path(
                        download_hf_file(
                            entry.repo_id,
                            entry.filename,
                            local_dir=tmp_dir,
                            token=token,
                        )
                    )
                    shutil.move(str(tmp_file), str(dest))
                got = dest
            downloaded_paths.append(got)
        except Exception as exc:  # noqa: BLE001
            if entry.optional:
                print(
                    f"[comfyui] warning: optional model failed ({entry.name}): {exc}",
                    flush=True,
                )
                continue
            raise

    return downloaded_paths


def prepare_zero_disk_dirs(shm_root: str | Path = DEFAULT_SHM_ROOT) -> dict[str, Path]:
    """Create RAM-backed output and temp directories under ``shm_root``."""
    base = Path(shm_root)
    out_dir = base / "comfy-out"
    tmp_dir = base / "comfy-tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return {"output": out_dir, "temp": tmp_dir}


def build_comfyui_command(
    comfy_dir: str | Path = DEFAULT_COMFY_DIR,
    *,
    shm_root: str | Path = DEFAULT_SHM_ROOT,
    host: str = "127.0.0.1",
    port: int = 8188,
    extra_args: Sequence[str] | None = None,
    python_bin: str = sys.executable,
) -> list[str]:
    """Build the ComfyUI launch command forcing output and temp into RAM disk."""
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid TCP port: {port}")

    dirs = prepare_zero_disk_dirs(shm_root)
    cmd = [
        python_bin,
        str(Path(comfy_dir) / "main.py"),
        "--listen",
        host,
        "--port",
        str(port),
        "--output-directory",
        str(dirs["output"]),
        "--temp-directory",
        str(dirs["temp"]),
        "--disable-auto-launch",
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def cleanup_expired_files(
    target_dirs: Iterable[str | Path],
    *,
    max_age_sec: float = 300.0,
    now: float | None = None,
) -> list[Path]:
    """Delete regular files older than ``max_age_sec`` inside ``target_dirs``."""
    if max_age_sec < 0:
        raise ValueError("max_age_sec must be non-negative")

    current_time = time.time() if now is None else now
    removed: list[Path] = []
    for raw_dir in target_dirs:
        root = Path(raw_dir)
        if not root.is_dir():
            continue
        for item in root.rglob("*"):
            if item.is_symlink() or not item.is_file():
                continue
            try:
                if current_time - item.stat().st_mtime > max_age_sec:
                    item.unlink()
                    removed.append(item)
            except FileNotFoundError:
                continue
    return removed


def start_ramdisk_reaper(
    target_dirs: Iterable[str | Path],
    *,
    max_age_sec: float = 300.0,
    interval_sec: float = 60.0,
) -> tuple[threading.Event, threading.Thread]:
    """Start a daemon thread periodically pruning expired files from RAM disk."""
    if max_age_sec < 0:
        raise ValueError("max_age_sec must be non-negative")
    if interval_sec <= 0:
        raise ValueError("interval_sec must be positive")

    dirs = [Path(d) for d in target_dirs]
    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.wait(interval_sec):
            try:
                cleanup_expired_files(dirs, max_age_sec=max_age_sec)
            except OSError:
                continue

    thread = threading.Thread(target=_run, name="comfyui-ramdisk-reaper", daemon=True)
    thread.start()
    return stop_event, thread


def wait_for_local_server(
    port: int = 8188,
    *,
    host: str = "127.0.0.1",
    timeout_sec: float = 120.0,
    poll_interval_sec: float = 1.0,
) -> str:
    """Wait until ComfyUI's local ``/system_stats`` endpoint responds HTTP 2xx."""
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid TCP port: {port}")

    local_url = f"http://{host}:{port}"
    probe_url = f"{local_url}/system_stats"
    deadline = time.monotonic() + timeout_sec
    req_timeout = min(5.0, max(0.5, poll_interval_sec))

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(probe_url, timeout=req_timeout) as resp:
                if 200 <= resp.status < 300:
                    return local_url
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"Timed out waiting for ComfyUI server at {probe_url}")


def _stop_subprocess(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if proc.stdout is not None and not proc.stdout.closed:
        proc.stdout.close()


@dataclass
class ZeroDiskLaunchResult:
    """Handle for a running zero-disk ComfyUI server, reaper, and tunnel."""

    public_url: str
    local_url: str
    output_dir: Path
    temp_dir: Path
    comfy_process: subprocess.Popen[str]
    tunnel: TunnelProcess
    reaper_stop: threading.Event

    def stop(self) -> None:
        """Stop the RAM disk reaper, Cloudflare tunnel, and ComfyUI server."""
        self.reaper_stop.set()
        try:
            self.tunnel.stop()
        finally:
            _stop_subprocess(self.comfy_process)


def launch_comfyui_zero_disk(
    comfy_dir: str | Path = DEFAULT_COMFY_DIR,
    *,
    shm_root: str | Path = DEFAULT_SHM_ROOT,
    host: str = "127.0.0.1",
    port: int = 8188,
    extra_args: Sequence[str] | None = None,
    python_bin: str = sys.executable,
    max_age_sec: float = 300.0,
    reaper_interval_sec: float = 60.0,
    server_timeout_sec: float = 120.0,
    tunnel_timeout_sec: float = 30.0,
    cloudflared_bin: str | None = None,
) -> ZeroDiskLaunchResult:
    """Start ComfyUI in zero-disk mode, prune RAM outputs periodically, and expose via tunnel."""
    dirs = prepare_zero_disk_dirs(shm_root)
    cmd = build_comfyui_command(
        comfy_dir,
        shm_root=shm_root,
        host=host,
        port=port,
        extra_args=extra_args,
        python_bin=python_bin,
    )
    reaper_stop, _ = start_ramdisk_reaper(
        (dirs["output"], dirs["temp"]),
        max_age_sec=max_age_sec,
        interval_sec=reaper_interval_sec,
    )
    comfy_proc = subprocess.Popen(
        cmd,
        cwd=str(comfy_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        local_url = wait_for_local_server(
            port=port,
            host=host,
            timeout_sec=server_timeout_sec,
        )
        tunnel = start_cloudflared_tunnel(
            port,
            timeout_sec=tunnel_timeout_sec,
            cloudflared_bin=cloudflared_bin,
        )
    except Exception:
        reaper_stop.set()
        _stop_subprocess(comfy_proc)
        raise

    return ZeroDiskLaunchResult(
        public_url=tunnel.url,
        local_url=local_url,
        output_dir=dirs["output"],
        temp_dir=dirs["temp"],
        comfy_process=comfy_proc,
        tunnel=tunnel,
        reaper_stop=reaper_stop,
    )
