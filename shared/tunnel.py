"""Helpers for exposing a local Colab WebUI port via cloudflared quick tunnels."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


@dataclass
class TunnelProcess:
    """Running tunnel handle returned by ``start_cloudflared_tunnel``."""

    url: str
    port: int
    process: subprocess.Popen[str]

    def stop(self) -> None:
        """Terminate the background tunnel process if still running."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdout is not None and not self.process.stdout.closed:
            self.process.stdout.close()


def parse_trycloudflare_url(text: str) -> str | None:
    """Extract the first ``*.trycloudflare.com`` URL from tunnel log output."""
    match = TRYCLOUDFLARE_RE.search(text)
    return match.group(0) if match else None


def _cloudflared_download_url() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system != "linux":
        raise RuntimeError(
            f"Auto-install only supports Linux Colab runtimes (got {system}). "
            "Install cloudflared manually on this platform."
        )
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    arch = arch_map.get(machine)
    if not arch:
        raise RuntimeError(f"Unsupported architecture for cloudflared: {machine}")
    return f"https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"


def ensure_cloudflared(bin_path: str | Path = "/usr/local/bin/cloudflared") -> str:
    """Return path to ``cloudflared``, downloading the Linux binary if missing."""
    existing = shutil.which("cloudflared")
    if existing:
        return existing

    target = Path(bin_path)
    if target.exists():
        return str(target)

    url = _cloudflared_download_url()
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    target.chmod(0o755)
    return str(target)


def start_cloudflared_tunnel(
    port: int,
    *,
    timeout_sec: float = 30.0,
    cloudflared_bin: str | None = None,
) -> TunnelProcess:
    """Start a ``cloudflared`` quick tunnel to ``127.0.0.1:<port>`` and return its URL."""
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid TCP port: {port}")

    binary = cloudflared_bin or ensure_cloudflared()
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    deadline = time.monotonic() + timeout_sec
    collected_lines: list[str] = []
    assert proc.stdout is not None

    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                remainder = proc.stdout.read()
                if remainder:
                    collected_lines.append(remainder)
                break
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            collected_lines.append(line)
            url = parse_trycloudflare_url(line)
            if url:
                return TunnelProcess(url=url, port=port, process=proc)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
        if not proc.stdout.closed:
            proc.stdout.close()
        raise

    if proc.poll() is None:
        proc.terminate()
    tail = "".join(collected_lines[-15:])
    if not proc.stdout.closed:
        proc.stdout.close()
    raise TimeoutError(
        f"Timed out waiting for cloudflared tunnel URL on port {port}.\n{tail}"
    )
