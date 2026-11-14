"""Helpers for downloading models from Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .runtime_env import get_secret


def _resolve_token(token: str | None) -> str | None:
    return token if token is not None else get_secret("HF_TOKEN")


def download_hf_file(
    repo_id: str,
    filename: str,
    *,
    local_dir: str | Path,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a single file from Hugging Face Hub into ``local_dir``."""
    from huggingface_hub import hf_hub_download

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        revision=revision,
        local_dir=str(dest),
        token=_resolve_token(token),
    )
    return Path(downloaded)


def download_hf_snapshot(
    repo_id: str,
    *,
    local_dir: str | Path,
    allow_patterns: Iterable[str] | str | None = None,
    ignore_patterns: Iterable[str] | str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a repository snapshot (or filtered subset) from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(dest),
        allow_patterns=list(allow_patterns) if isinstance(allow_patterns, (list, tuple, set)) else allow_patterns,
        ignore_patterns=list(ignore_patterns) if isinstance(ignore_patterns, (list, tuple, set)) else ignore_patterns,
        token=_resolve_token(token),
    )
    return Path(downloaded)
