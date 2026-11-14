"""Helpers for mounting Google Drive and linking persistent directories."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/colab-workbench")
DEFAULT_SUBDIRS = ("models", "loras", "datasets", "outputs", "configs")


def mount_drive(
    mount_point: str | Path = "/content/drive",
    *,
    force_remount: bool = False,
) -> Path:
    """Mount Google Drive in Colab and return the mount path.

    Outside Colab, raises ``RuntimeError`` unless ``mount_point`` already exists.
    """
    target = Path(mount_point)
    if target.exists() and any(target.iterdir()) and not force_remount:
        return target

    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        if target.exists():
            return target
        raise RuntimeError(
            "google.colab is unavailable; mount_drive() only runs inside Google Colab."
        ) from exc

    drive.mount(str(target), force_remount=force_remount)
    return target


def ensure_workspace_dirs(
    base_dir: str | Path = DEFAULT_DRIVE_ROOT,
    subdirs: Iterable[str] = DEFAULT_SUBDIRS,
) -> dict[str, Path]:
    """Create persistent workspace subdirectories and return a name -> Path map."""
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {"root": root}
    for name in subdirs:
        sub = root / name
        sub.mkdir(parents=True, exist_ok=True)
        result[name] = sub
    return result


def link_dir(
    target_dir: str | Path,
    link_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Create a symlink ``link_path -> target_dir`` safely.

    If ``link_path`` already points to ``target_dir``, nothing is changed.
    Empty directories at ``link_path`` are replaced automatically; non-empty
    directories or conflicting symlinks require ``force=True``.
    """
    target = Path(target_dir)
    link = Path(link_path)
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        if link.resolve() == target.resolve():
            return link
        if not force:
            raise FileExistsError(
                f"Symlink '{link}' already points to '{link.resolve()}'; pass force=True to replace."
            )
        link.unlink()
    elif link.exists():
        if link.is_dir():
            is_empty = not any(link.iterdir())
            if not is_empty and not force:
                raise FileExistsError(
                    f"Directory '{link}' is not empty; pass force=True to replace."
                )
            shutil.rmtree(link)
        else:
            if not force:
                raise FileExistsError(
                    f"File '{link}' already exists; pass force=True to replace."
                )
            link.unlink()

    link.symlink_to(target, target_is_directory=True)
    return link
