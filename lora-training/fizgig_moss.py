"""Helpers for training MOSS-Image LoRAs and running Fizgig Web UI on Google Colab."""

from __future__ import annotations

import importlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from shared.hf_download import download_hf_file, download_hf_repo_files
from shared.runtime_env import get_secret, get_secret_status
from shared.tunnel import (
    DEFAULT_TUNNEL_PROTOCOL,
    TunnelProcess,
    normalize_tunnel_protocol,
    start_cloudflared_tunnel,
)

DEFAULT_FIZGIG_DIR = Path("/content/Fizgig")
DEFAULT_SHM_ROOT = Path("/dev/shm")
DEFAULT_WEB_PORT = 8081
MIN_WEB_PASSWORD_LEN = 8
MUON_DEFAULT_ARGS = (
    "adjust_lr_fn=match_rms_adamw weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5"
)
DEFAULT_LOWNOISE_PCT = "11:7:40:8:3"
DEFAULT_SAMPLE_NEGATIVE = "worst quality, bad anatomy, watermark, jpeg artifact"

_SUPPORTED_RECIPES = ("dop_adamw", "muon_baseline")
_SUPPORTED_NETWORK_TYPES = ("lora", "lokr")


def ensure_fizgig_moss_ready(
    repo_dir: str | Path = "/content/colab-workbench",
) -> Path:
    """Ensure ``lora-training`` and ``shared`` are importable from ``repo_dir``."""
    repo_path = Path(repo_dir)
    lora_dir = repo_path / "lora-training"
    module_file = lora_dir / "fizgig_moss.py"
    shared_init = repo_path / "shared" / "__init__.py"
    if not module_file.is_file():
        raise FileNotFoundError(
            f"Missing {module_file}; ensure colab-workbench is checked out on a branch with lora-training."
        )
    if not shared_init.is_file():
        raise FileNotFoundError(
            f"Missing {shared_init}; ensure colab-workbench checkout is complete."
        )

    desired = [str(lora_dir), str(repo_path)]
    sys.path[:] = [*desired, *[p for p in sys.path if p not in desired]]
    importlib.invalidate_caches()

    resolved_shared = shared_init.resolve()
    cached_shared = sys.modules.get("shared")
    if cached_shared is not None:
        cached_file = getattr(cached_shared, "__file__", None)
        if not isinstance(cached_file, (str, os.PathLike)) or Path(cached_file).resolve() != resolved_shared:
            for mod_name in [k for k in list(sys.modules) if k == "shared" or k.startswith("shared.")]:
                del sys.modules[mod_name]

    resolved_module = module_file.resolve()
    cached_mod = sys.modules.get("fizgig_moss")
    if cached_mod is not None:
        cached_mod_file = getattr(cached_mod, "__file__", None)
        if not isinstance(cached_mod_file, (str, os.PathLike)) or Path(cached_mod_file).resolve() != resolved_module:
            del sys.modules["fizgig_moss"]

    return resolved_module


@dataclass(frozen=True)
class MossModelEntry:
    """Descriptor for a MOSS-Image weight file tracked in Fizgig ``prefs.json``."""

    pref_key: str
    repo_id: str
    filename: str
    local_name: str | None = None
    optional: bool = False

    def dest_filename(self) -> str:
        """Return the flat destination filename under ``user_data/models``."""
        return self.local_name or Path(self.filename).name


_STANDARD_MODELS: tuple[MossModelEntry, ...] = (
    MossModelEntry(
        pref_key="moss_image_dit",
        repo_id="Comfy-Org/MOSS-Image",
        filename="diffusion_models/moss_image_base_bf16.safetensors",
    ),
    MossModelEntry(
        pref_key="moss_image_vae",
        repo_id="Comfy-Org/MOSS-Image",
        filename="vae/flux2-vae_small.safetensors",
    ),
    MossModelEntry(
        pref_key="moss_image_text_encoder",
        repo_id="Comfy-Org/Qwen3.5",
        filename="text_encoders/qwen3.5_9b_bf16.safetensors",
    ),
    MossModelEntry(
        pref_key="moss_image_sft_dit",
        repo_id="Comfy-Org/MOSS-Image",
        filename="diffusion_models/moss_image_sft_int8_convrot.safetensors",
    ),
)

_INT8_TE_MODELS: tuple[MossModelEntry, ...] = (
    _STANDARD_MODELS[0],
    _STANDARD_MODELS[1],
    MossModelEntry(
        pref_key="moss_image_text_encoder",
        repo_id="Comfy-Org/MOSS-Image",
        filename="text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    ),
    _STANDARD_MODELS[3],
)

_FAST_PREVIEW_EXTRAS: tuple[MossModelEntry, ...] = (
    MossModelEntry(
        pref_key="moss_image_turbo_lora",
        repo_id="lodestones/Kromoss",
        filename="8-steps-dmd-lora.safetensors",
        optional=True,
    ),
    MossModelEntry(
        pref_key="moss_image_sft_look_lora",
        repo_id="ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA",
        filename="moss_image_sft_look_lora.safetensors",
        optional=True,
    ),
)

MOSS_MODEL_PRESETS: dict[str, tuple[MossModelEntry, ...]] = {
    "standard": _STANDARD_MODELS,
    "int8_te": _INT8_TE_MODELS,
    "with_fast_preview": _STANDARD_MODELS + _FAST_PREVIEW_EXTRAS,
}


def resolve_moss_preset(preset: str = "standard") -> list[MossModelEntry]:
    """Return the model entries matching a MOSS-Image preset."""
    key = preset.strip().lower()
    if key not in MOSS_MODEL_PRESETS:
        valid = ", ".join(sorted(MOSS_MODEL_PRESETS))
        raise ValueError(f"Unknown MOSS preset '{preset}'. Expected one of: {valid}")
    return list(MOSS_MODEL_PRESETS[key])


def wire_fizgig_prefs(
    model_paths: dict[str, Path],
    *,
    fizgig_dir: str | Path = DEFAULT_FIZGIG_DIR,
) -> Path:
    """Merge downloaded model paths into Fizgig's ``user_data/prefs.json`` atomically."""
    user_data_dir = Path(fizgig_dir) / "user_data"
    user_data_dir.mkdir(parents=True, exist_ok=True)
    prefs_path = user_data_dir / "prefs.json"

    prefs: dict[str, object] = {}
    if prefs_path.is_file():
        raw_text = prefs_path.read_text(encoding="utf-8").strip()
        if raw_text:
            parsed = json.loads(raw_text)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object in {prefs_path}, got {type(parsed).__name__}")
            prefs.update(parsed)

    for key, path in model_paths.items():
        prefs[key] = str(Path(path).resolve())

    fd, tmp_name = tempfile.mkstemp(
        prefix=".prefs_",
        suffix=".json.tmp",
        dir=str(user_data_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(prefs, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_name, prefs_path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise

    return prefs_path


def _resolve_hf_token_and_xet(token: str | None) -> str | None:
    raw_token = token if token is not None else get_secret("HF_TOKEN")
    effective_token = raw_token.strip() if isinstance(raw_token, str) and raw_token.strip() else None
    has_secret, secret_reason = get_secret_status("HF_TOKEN")

    if effective_token:
        source_label = "explicit" if token is not None else (secret_reason if has_secret else "env/secret")
        print(f"[fizgig-moss] HF_TOKEN: authenticated ({source_label})", flush=True)
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    else:
        reason_label = "explicitly empty" if token is not None else secret_reason
        print(
            f"[fizgig-moss] warning: HF_TOKEN unavailable ({reason_label}); downloading anonymously. "
            "Shared Colab IPs may hit HTTP 429 rate limits—set HF_TOKEN in Colab Secrets with "
            "'Notebook access' enabled.",
            flush=True,
        )
    return effective_token


def _move_staged_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()
    shutil.move(str(src), str(dest))
    return dest


def _download_repo_entries(
    repo_id: str,
    entries: Sequence[MossModelEntry],
    *,
    models_dir: Path,
    staging_dir: Path,
    token: str | None,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    required = [e for e in entries if not e.optional]
    optional = [e for e in entries if e.optional]
    staging_dir.mkdir(parents=True, exist_ok=True)

    if required:
        print(
            f"[fizgig-moss] batch downloading {len(required)} file(s) from {repo_id}...",
            flush=True,
        )
        got_paths = download_hf_repo_files(
            repo_id,
            [e.filename for e in required],
            local_dir=staging_dir,
            token=token,
        )
        for entry, got_path in zip(required, got_paths):
            staged = Path(got_path) if got_path else (staging_dir / entry.filename)
            resolved[entry.pref_key] = _move_staged_file(staged, models_dir / entry.dest_filename())

    if optional:
        if len(optional) > 1:
            try:
                got_paths = download_hf_repo_files(
                    repo_id,
                    [e.filename for e in optional],
                    local_dir=staging_dir,
                    token=token,
                )
                for entry, got_path in zip(optional, got_paths):
                    staged = Path(got_path) if got_path else (staging_dir / entry.filename)
                    resolved[entry.pref_key] = _move_staged_file(
                        staged,
                        models_dir / entry.dest_filename(),
                    )
                return resolved
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[fizgig-moss] warning: optional batch download failed for {repo_id} ({exc}); retrying individually...",
                    flush=True,
                )

        for entry in optional:
            dest = models_dir / entry.dest_filename()
            if dest.is_file() and dest.stat().st_size > 0:
                resolved[entry.pref_key] = dest
                continue
            try:
                got = Path(
                    download_hf_file(
                        repo_id,
                        entry.filename,
                        local_dir=staging_dir,
                        token=token,
                    )
                )
                resolved[entry.pref_key] = _move_staged_file(got, dest)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[fizgig-moss] warning: optional model skipped ({entry.pref_key}): {exc}",
                    flush=True,
                )

    return resolved


def download_moss_models(
    preset: str = "standard",
    *,
    fizgig_dir: str | Path = DEFAULT_FIZGIG_DIR,
    token: str | None = None,
    wire_prefs: bool = True,
    pause_sec: float = 1.0,
) -> dict[str, Path]:
    """Download MOSS-Image models for Fizgig and wire ``user_data/prefs.json``."""
    if pause_sec < 0:
        raise ValueError("pause_sec must be non-negative")

    entries = resolve_moss_preset(preset)
    effective_token = _resolve_hf_token_and_xet(token)

    fizgig_root = Path(fizgig_dir)
    models_dir = fizgig_root / "user_data" / "models"
    staging_dir = models_dir / ".hf-staging"
    models_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    missing_by_repo: dict[str, list[MossModelEntry]] = {}

    for entry in entries:
        dest = models_dir / entry.dest_filename()
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[fizgig-moss] already present, skip: {dest.name}", flush=True)
            resolved[entry.pref_key] = dest
            continue
        missing_by_repo.setdefault(entry.repo_id, []).append(entry)

    for idx, (repo_id, repo_entries) in enumerate(missing_by_repo.items()):
        if idx > 0 and pause_sec > 0:
            time.sleep(pause_sec)
        resolved.update(
            _download_repo_entries(
                repo_id,
                repo_entries,
                models_dir=models_dir,
                staging_dir=staging_dir,
                token=effective_token,
            )
        )

    if wire_prefs and resolved:
        wire_fizgig_prefs(resolved, fizgig_dir=fizgig_root)

    return resolved


def write_dataset_toml(
    toml_path: str | Path,
    *,
    image_directory: str | Path,
    cache_directory: str | Path,
    resolution: tuple[int, int] = (1024, 1024),
    batch_size: int = 1,
    num_repeats: int = 1,
    enable_bucket: bool = True,
    bucket_no_upscale: bool = True,
    multi_caption: bool = False,
) -> Path:
    """Write a Fizgig-compatible dataset TOML config file."""
    if len(resolution) != 2 or any(int(dim) <= 0 for dim in resolution):
        raise ValueError(f"resolution must be two positive ints, got {resolution!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_repeats < 1:
        raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")

    out_path = Path(toml_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_str = Path(image_directory).as_posix()
    cache_str = Path(cache_directory).as_posix()

    content = "\n".join(
        [
            "[general]",
            f"resolution = [{int(resolution[0])}, {int(resolution[1])}]",
            'caption_extension = ".txt"',
            f"batch_size = {int(batch_size)}",
            f"num_repeats = {int(num_repeats)}",
            f"enable_bucket = {'true' if enable_bucket else 'false'}",
            f"bucket_no_upscale = {'true' if bucket_no_upscale else 'false'}",
            f"multi_caption = {'true' if multi_caption else 'false'}",
            "",
            "[[datasets]]",
            f"image_directory = {json.dumps(img_str)}",
            f"cache_directory = {json.dumps(cache_str)}",
            "",
        ]
    )
    out_path.write_text(content, encoding="utf-8", newline="\n")
    return out_path


def _fizgig_venv_python(fizgig_dir: str | Path) -> str:
    return str(Path(fizgig_dir) / "venv" / "bin" / "python")


def build_moss_cache_commands(
    dataset_toml: str | Path,
    *,
    vae_path: str | Path,
    text_encoder_path: str | Path,
    fizgig_dir: str | Path = DEFAULT_FIZGIG_DIR,
    skip_existing: bool = False,
) -> tuple[list[str], list[str]]:
    """Build Fizgig ``cache.py`` commands for ``latents`` and ``text`` stages."""
    root = Path(fizgig_dir)
    py_bin = _fizgig_venv_python(root)
    cache_script = str(root / "src" / "fizgig" / "families" / "cache.py")

    def _stage_cmd(stage: str, model_path: str | Path) -> list[str]:
        cmd = [
            py_bin,
            cache_script,
            "--family",
            "moss_image",
            "--stage",
            stage,
            "--dataset_config",
            str(dataset_toml),
            "--model",
            str(model_path),
        ]
        if skip_existing:
            cmd.append("--skip_existing")
        return cmd

    return _stage_cmd("latents", vae_path), _stage_cmd("text", text_encoder_path)


@dataclass(frozen=True)
class MossTrainConfig:
    """Configuration for building a Fizgig MOSS-Image ``train.py`` invocation."""

    dataset_config: str | Path
    dit_path: str | Path
    output_dir: str | Path
    output_name: str
    trigger_word: str
    vae_path: str | Path | None = None
    text_encoder_path: str | Path | None = None
    recipe: str = "dop_adamw"
    network_type: str = "lora"
    network_dim: int = 32
    network_alpha: int = 32
    lokr_factor: int = 8
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    max_train_epochs: int = 60
    save_every_n_epochs: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    precision: str = "auto"
    blocks_to_swap: int = -1
    gradient_checkpointing: str = "on"
    compile_blocks: str = "auto"
    ema_decay: float = 0.98
    lownoise_pct: str = DEFAULT_LOWNOISE_PCT
    seed: int = 42
    save_state: bool = True
    save_state_on_train_end: bool = True
    keep_last_n_states: int = 2
    resume: str | Path | None = None
    masked_training: bool | None = None
    mask_floor: float = 0.1
    mask_prior: float | None = None
    dop: bool | None = None
    dop_every: int = 2
    dop_weight: float = 2.0
    dop_class: str = "woman"
    caption_dropout: float | None = None
    multi_caption: bool = False
    preview_checkpoint: str | Path | None = None
    sample_prompts: str | Path | None = None
    sample_every_n_epochs: int = 0
    sample_at_first: bool = False
    sample_width: int = 1024
    sample_height: int = 1024
    sample_steps: int = 30
    sample_cfg_scale: float = 4.0
    sample_seed: int = 17
    sample_negative: str = DEFAULT_SAMPLE_NEGATIVE
    extra_args: Sequence[str] = field(default_factory=tuple)


def build_moss_train_command(
    config: MossTrainConfig,
    *,
    fizgig_dir: str | Path = DEFAULT_FIZGIG_DIR,
) -> list[str]:
    """Build a validated Fizgig ``train.py`` CLI argument list for MOSS-Image."""
    recipe = config.recipe.strip().lower()
    if recipe not in _SUPPORTED_RECIPES:
        valid = ", ".join(_SUPPORTED_RECIPES)
        raise ValueError(f"Unsupported recipe '{config.recipe}'. Expected one of: {valid}")

    network_type = config.network_type.strip().lower()
    if network_type not in _SUPPORTED_NETWORK_TYPES:
        valid = ", ".join(_SUPPORTED_NETWORK_TYPES)
        raise ValueError(
            f"Unsupported network_type '{config.network_type}'. Expected one of: {valid}"
        )

    trigger = config.trigger_word.strip()
    if not trigger:
        raise ValueError("trigger_word must be a non-empty string")
    if not config.output_name.strip():
        raise ValueError("output_name must be a non-empty string")
    if config.max_train_epochs < 1:
        raise ValueError(f"max_train_epochs must be >= 1, got {config.max_train_epochs}")
    if config.save_every_n_epochs < 1:
        raise ValueError(f"save_every_n_epochs must be >= 1, got {config.save_every_n_epochs}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")

    use_dop = (recipe == "dop_adamw") if config.dop is None else config.dop
    use_mask = (recipe == "dop_adamw") if config.masked_training is None else config.masked_training
    mask_prior = (1.0 if recipe == "dop_adamw" else 0.0) if config.mask_prior is None else config.mask_prior
    caption_dropout = (
        (0.05 if recipe == "dop_adamw" else 0.0)
        if config.caption_dropout is None
        else config.caption_dropout
    )

    if use_dop:
        if not config.dop_class.strip():
            raise ValueError("dop_class must be non-empty when DOP is enabled")
        if config.dop_every < 1:
            raise ValueError(f"dop_every must be >= 1, got {config.dop_every}")
        if config.dop_weight <= 0:
            raise ValueError(f"dop_weight must be positive, got {config.dop_weight}")

    root = Path(fizgig_dir)
    cmd = [
        _fizgig_venv_python(root),
        str(root / "src" / "fizgig" / "families" / "train.py"),
        "--family",
        "moss_image",
        "--dataset_config",
        str(config.dataset_config),
        "--dit",
        str(config.dit_path),
        "--output_dir",
        str(config.output_dir),
        "--output_name",
        config.output_name.strip(),
        "--trigger_word",
        trigger,
        "--network_type",
        network_type,
        "--learning_rate",
        f"{config.learning_rate:g}",
        "--lr_scheduler",
        config.lr_scheduler,
        "--lr_warmup_steps",
        str(config.lr_warmup_steps),
        "--max_train_epochs",
        str(config.max_train_epochs),
        "--save_every_n_epochs",
        str(config.save_every_n_epochs),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--max_grad_norm",
        str(config.max_grad_norm),
        "--precision",
        config.precision,
        "--blocks_to_swap",
        str(config.blocks_to_swap),
        "--gradient_checkpointing",
        config.gradient_checkpointing,
        "--compile_blocks",
        config.compile_blocks,
        "--ema_decay",
        str(config.ema_decay),
        "--caption_dropout",
        str(caption_dropout),
        "--seed",
        str(config.seed),
        "--family_option",
        f"lownoise_pct={config.lownoise_pct}",
    ]

    if network_type == "lora":
        cmd.extend(
            [
                "--network_dim",
                str(config.network_dim),
                "--network_alpha",
                str(config.network_alpha),
            ]
        )
    else:
        cmd.extend(["--lokr_factor", str(config.lokr_factor)])

    if recipe == "dop_adamw":
        cmd.extend(["--optimizer_type", "adamw"])
    else:
        cmd.extend(["--optimizer_type", "muon", "--optimizer_args", MUON_DEFAULT_ARGS])

    if use_mask:
        cmd.extend(["--masked_training", "--mask_floor", str(config.mask_floor)])
        if mask_prior > 0:
            cmd.extend(["--mask_prior", str(mask_prior)])

    if use_dop:
        cmd.extend(
            [
                "--dop",
                "--dop_every",
                str(config.dop_every),
                "--dop_weight",
                str(config.dop_weight),
                "--dop_trigger",
                trigger,
                "--dop_class",
                config.dop_class.strip(),
            ]
        )

    if config.multi_caption:
        cmd.append("--multi_caption")

    if config.vae_path is not None:
        cmd.extend(["--vae", str(config.vae_path)])
    if config.text_encoder_path is not None:
        cmd.extend(["--text_encoder", str(config.text_encoder_path)])

    if config.save_state:
        cmd.append("--save_state")
    if config.save_state_on_train_end:
        cmd.append("--save_state_on_train_end")
    cmd.extend(["--keep_last_n_states", str(config.keep_last_n_states)])
    if config.resume is not None:
        cmd.extend(["--resume", str(config.resume)])

    if config.preview_checkpoint is not None:
        cmd.extend(["--preview_checkpoint", str(config.preview_checkpoint)])
    if config.sample_prompts is not None:
        cmd.extend(
            [
                "--sample_prompts",
                str(config.sample_prompts),
                "--sample_every_n_epochs",
                str(config.sample_every_n_epochs),
                "--sample_width",
                str(config.sample_width),
                "--sample_height",
                str(config.sample_height),
                "--sample_steps",
                str(config.sample_steps),
                "--sample_cfg_scale",
                str(config.sample_cfg_scale),
                "--sample_seed",
                str(config.sample_seed),
                "--sample_negative",
                config.sample_negative,
            ]
        )
        if config.sample_at_first:
            cmd.append("--sample_at_first")

    if config.extra_args:
        cmd.extend(config.extra_args)

    return cmd


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    if proc.stdout is not None and not proc.stdout.closed:
        proc.stdout.close()


def _resolve_web_password(password: str | None) -> str:
    candidate = password if password is not None else get_secret("FIZGIG_PASSWORD")
    if candidate is None or not candidate.strip():
        return secrets.token_urlsafe(16)
    pwd = candidate.strip()
    if len(pwd) < MIN_WEB_PASSWORD_LEN:
        raise ValueError(
            f"Fizgig web password must be at least {MIN_WEB_PASSWORD_LEN} characters long."
        )
    return pwd


def _wait_for_web_server(
    proc: subprocess.Popen[str],
    *,
    port: int,
    timeout_sec: float,
    poll_interval_sec: float = 1.0,
) -> str:
    local_url = f"http://127.0.0.1:{port}"
    probe_url = f"{local_url}/"
    deadline = time.monotonic() + timeout_sec
    req_timeout = min(5.0, max(0.5, poll_interval_sec))

    while time.monotonic() < deadline:
        exit_code = proc.poll()
        if exit_code is not None:
            raise RuntimeError(f"Fizgig Web UI exited prematurely with code {exit_code}")
        try:
            with urllib.request.urlopen(probe_url, timeout=req_timeout) as resp:
                if 200 <= resp.status < 400:
                    return local_url
        except urllib.error.HTTPError as exc:
            if exc.code in (200, 302, 401, 403):
                return local_url
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"Timed out waiting for Fizgig Web UI at {probe_url}")


@dataclass
class FizgigWebSession:
    """Running handle for Fizgig Web UI and optional Cloudflare quick tunnel."""

    local_url: str
    public_url: str | None
    port: int
    password: str
    web_process: subprocess.Popen[str]
    tunnel: TunnelProcess | None
    log_file: Path
    tunnel_log_file: Path | None = None

    def stop(self) -> None:
        """Stop the Cloudflare tunnel (if active) and the Fizgig Web UI process."""
        try:
            if self.tunnel is not None:
                self.tunnel.stop()
        finally:
            _stop_process(self.web_process)


def launch_fizgig_web(
    *,
    fizgig_dir: str | Path = DEFAULT_FIZGIG_DIR,
    port: int = DEFAULT_WEB_PORT,
    password: str | None = None,
    start_tunnel: bool = True,
    tunnel_protocol: str | None = DEFAULT_TUNNEL_PROTOCOL,
    cloudflared_bin: str | None = None,
    shm_root: str | Path = DEFAULT_SHM_ROOT,
    startup_timeout_sec: float = 60.0,
    tunnel_timeout_sec: float = 30.0,
) -> FizgigWebSession:
    """Launch password-protected Fizgig Web UI and optionally expose via Cloudflare tunnel."""
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid TCP port: {port}")

    resolved_password = _resolve_web_password(password)
    normalized_protocol = normalize_tunnel_protocol(tunnel_protocol) if start_tunnel else None

    tmp_dir = Path(shm_root) / "fizgig-tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_file = tmp_dir / "fizgig-web.log"
    tunnel_log_file = (tmp_dir / "cloudflared.log") if start_tunnel else None

    root = Path(fizgig_dir)
    cmd = [
        _fizgig_venv_python(root),
        str(root / "fizgig.py"),
        "--web",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    # Pass password via FIZGIG_PASSWORD env var so it stays out of process argv/cmdline.
    env = {**os.environ, "FIZGIG_PASSWORD": resolved_password}

    log_fp = log_file.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        if not log_fp.closed:
            log_fp.close()

    tunnel: TunnelProcess | None = None
    try:
        local_url = _wait_for_web_server(
            proc,
            port=port,
            timeout_sec=startup_timeout_sec,
        )
        if start_tunnel:
            tunnel = start_cloudflared_tunnel(
                port,
                timeout_sec=tunnel_timeout_sec,
                cloudflared_bin=cloudflared_bin,
                protocol=normalized_protocol,
                log_file=tunnel_log_file,
            )
    except Exception:
        if tunnel is not None:
            tunnel.stop()
        _stop_process(proc)
        raise

    return FizgigWebSession(
        local_url=local_url,
        public_url=tunnel.url if tunnel is not None else None,
        port=port,
        password=resolved_password,
        web_process=proc,
        tunnel=tunnel,
        log_file=log_file,
        tunnel_log_file=tunnel_log_file,
    )
