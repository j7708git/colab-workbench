"""Helpers for exposing a local Colab WebUI port via cloudflared quick tunnels."""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

TRYCLOUDFLARE_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
DEFAULT_TUNNEL_PROTOCOL = "http2"
SUPPORTED_TUNNEL_PROTOCOLS = ("auto", "http2", "quic")
_TAIL_LOG_LIMIT = 100


@dataclass
class TunnelProcess:
    """Running tunnel handle returned by ``start_cloudflared_tunnel``."""

    url: str
    port: int
    process: subprocess.Popen[str]
    protocol: str | None = None
    log_file: Path | None = None
    _reader_thread: threading.Thread | None = field(default=None, repr=False, compare=False)
    _tail_lines: deque[str] | None = field(default=None, repr=False, compare=False)
    _tail_lock: threading.Lock | None = field(default=None, repr=False, compare=False)

    def tail_logs(self, lines: int = 20) -> str:
        """Return up to the last ``lines`` lines captured from ``cloudflared`` output."""
        if lines <= 0 or self._tail_lines is None:
            return ""
        if self._tail_lock is not None:
            with self._tail_lock:
                snapshot = list(self._tail_lines)
        else:
            snapshot = list(self._tail_lines)
        return "".join(snapshot[-lines:])

    def stop(self) -> None:
        """Terminate the background tunnel process if still running."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            if (
                not self._reader_thread.is_alive()
                and self.process.stdout is not None
                and not self.process.stdout.closed
            ):
                self.process.stdout.close()
        elif self.process.stdout is not None and not self.process.stdout.closed:
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


def normalize_tunnel_protocol(protocol: str | None) -> str | None:
    """Validate and normalize a ``cloudflared --protocol`` option."""
    if protocol is None:
        return None
    normalized = protocol.strip().lower()
    if normalized not in SUPPORTED_TUNNEL_PROTOCOLS:
        supported = ", ".join(SUPPORTED_TUNNEL_PROTOCOLS)
        raise ValueError(
            f"Unsupported cloudflared tunnel protocol: {protocol!r} "
            f"(expected one of: {supported}, or None)."
        )
    return normalized


def start_cloudflared_tunnel(
    port: int,
    *,
    timeout_sec: float = 30.0,
    cloudflared_bin: str | None = None,
    protocol: str | None = DEFAULT_TUNNEL_PROTOCOL,
    log_file: str | Path | None = None,
) -> TunnelProcess:
    """Start a ``cloudflared`` quick tunnel to ``127.0.0.1:<port>`` and return its URL."""
    if port <= 0 or port > 65535:
        raise ValueError(f"Invalid TCP port: {port}")

    normalized_protocol = normalize_tunnel_protocol(protocol)
    log_path = Path(log_file) if log_file is not None else None

    binary = cloudflared_bin or ensure_cloudflared()
    cmd = [binary, "tunnel"]
    if normalized_protocol is not None:
        cmd.extend(["--protocol", normalized_protocol])
    cmd.extend(["--url", f"http://127.0.0.1:{port}", "--no-autoupdate"])

    log_fp = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("w", encoding="utf-8", buffering=1)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
    except Exception:
        if log_fp is not None and not log_fp.closed:
            log_fp.close()
        raise

    assert proc.stdout is not None
    tail_lines: deque[str] = deque(maxlen=_TAIL_LOG_LIMIT)
    tail_lock = threading.Lock()
    url_event = threading.Event()
    eof_event = threading.Event()
    found_url: str | None = None

    def _drain_stdout() -> None:
        nonlocal found_url
        stdout_pipe = proc.stdout
        log_active = log_fp is not None
        try:
            assert stdout_pipe is not None
            while True:
                line = stdout_pipe.readline()
                if not line:
                    break
                with tail_lock:
                    tail_lines.append(line)
                if log_active and log_fp is not None:
                    try:
                        log_fp.write(line)
                        log_fp.flush()
                    except OSError:
                        log_active = False
                if not url_event.is_set():
                    candidate = parse_trycloudflare_url(line)
                    if candidate:
                        found_url = candidate
                        url_event.set()
        except (OSError, ValueError):
            pass
        finally:
            eof_event.set()
            if log_fp is not None and not log_fp.closed:
                try:
                    log_fp.close()
                except OSError:
                    pass
            if stdout_pipe is not None and not stdout_pipe.closed:
                try:
                    stdout_pipe.close()
                except OSError:
                    pass

    reader_thread = threading.Thread(
        target=_drain_stdout,
        name="cloudflared-stdout-drain",
        daemon=True,
    )
    reader_thread.start()

    deadline = time.monotonic() + timeout_sec
    try:
        while not url_event.is_set():
            if eof_event.is_set() and proc.poll() is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            url_event.wait(timeout=min(0.1, remaining))
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        reader_thread.join(timeout=2.0)
        raise

    if url_event.is_set() and found_url is not None:
        return TunnelProcess(
            url=found_url,
            port=port,
            process=proc,
            protocol=normalized_protocol,
            log_file=log_path,
            _reader_thread=reader_thread,
            _tail_lines=tail_lines,
            _tail_lock=tail_lock,
        )

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    reader_thread.join(timeout=2.0)
    with tail_lock:
        tail = "".join(list(tail_lines)[-15:])
    raise TimeoutError(
        f"Timed out waiting for cloudflared tunnel URL on port {port}.\n{tail}"
    )
