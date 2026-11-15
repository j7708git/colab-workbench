"""Helpers for detecting Colab runtime and loading tokens from Secrets or env vars."""

from __future__ import annotations

import os
import sys


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def _validate_secret_name(name: str) -> None:
    if not name or not name.strip():
        raise ValueError("Secret name must be a non-empty string.")


def _lookup_colab_secret(name: str, *, warn: bool) -> tuple[str | None, str | None]:
    """Attempt to read a secret from ``google.colab.userdata``.

    Returns ``(value, failure_reason)`` without printing or exposing secret values.
    """
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return None, None

    try:
        value = userdata.get(name)
    except Exception as exc:  # noqa: BLE001
        exc_name = type(exc).__name__
        exc_lower = exc_name.lower()
        msg_lower = str(exc).lower()

        if "secretnotfound" in exc_lower:
            return None, "not-set"

        if (
            "notebookaccess" in exc_lower
            or "access" in exc_lower
            or "notebook access" in msg_lower
            or "does not have access" in msg_lower
        ):
            if warn:
                print(
                    f"[colab-workbench] warning: Colab secret '{name}' is blocked "
                    f"({exc_name}). Enable the 'Notebook access' toggle for '{name}' "
                    "in Colab's Secrets sidebar (key icon).",
                    file=sys.stderr,
                )
            return None, "notebook-access-disabled"

        if "timeout" in exc_lower or "timeout" in msg_lower or "timed out" in msg_lower:
            if warn:
                print(
                    f"[colab-workbench] warning: Timed out reading Colab secret '{name}' "
                    f"({exc_name}). Check that the 'Notebook access' toggle for '{name}' "
                    "is enabled in Colab's Secrets sidebar (key icon).",
                    file=sys.stderr,
                )
            return None, "colab-secret-timeout"

        if warn:
            print(
                f"[colab-workbench] warning: Failed reading Colab secret '{name}' "
                f"({exc_name}). Check that '{name}' exists and its 'Notebook access' "
                "toggle is enabled in Colab's Secrets sidebar (key icon).",
                file=sys.stderr,
            )
        return None, f"colab-secret-error:{exc_name}"

    if value:
        return value, None
    return None, "not-set"


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from Google Colab userdata first, then os.environ.

    Args:
        name: Secret or environment variable name (e.g. ``HF_TOKEN``).
        default: Fallback value when the secret is missing or inaccessible.
    """
    _validate_secret_name(name)

    colab_value, _ = _lookup_colab_secret(name, warn=True)
    if colab_value:
        return colab_value

    value = os.environ.get(name)
    return value if value else default


def get_secret_status(name: str = "HF_TOKEN") -> tuple[bool, str]:
    """Return ``(is_present, source_or_reason)`` for a secret without exposing its value.

    Returns ``(True, "colab-secret" | "env")`` when available, or
    ``(False, reason)`` such as ``"notebook-access-disabled"`` or ``"not-set"``.
    """
    _validate_secret_name(name)

    colab_value, colab_reason = _lookup_colab_secret(name, warn=False)
    if colab_value:
        return True, "colab-secret"

    if os.environ.get(name):
        return True, "env"

    return False, colab_reason or "not-set"


def require_secret(name: str) -> str:
    """Return a non-empty secret or raise a clear setup error."""
    value = get_secret(name)
    if value:
        return value
    raise RuntimeError(
        f"Missing required secret '{name}'. "
        "Add it in the Colab sidebar (key icon -> Secrets) with Notebook access enabled, "
        f"or set the {name} environment variable."
    )


def export_secret_to_env(
    name: str,
    env_name: str | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Load a secret and export it to ``os.environ`` if present.

    Returns the secret value, or ``None`` when optional and unset.
    """
    value = require_secret(name) if required else get_secret(name)
    if value is not None:
        os.environ[env_name or name] = value
    return value
