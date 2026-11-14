"""Helpers for detecting Colab runtime and loading tokens from Secrets or env vars."""

from __future__ import annotations

import os


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from Google Colab userdata first, then os.environ.

    Args:
        name: Secret or environment variable name (e.g. ``HF_TOKEN``).
        default: Fallback value when the secret is missing or inaccessible.
    """
    if not name or not name.strip():
        raise ValueError("Secret name must be a non-empty string.")

    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        # Outside Colab, or secret not configured / notebook access disabled.
        pass

    value = os.environ.get(name)
    return value if value else default


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
