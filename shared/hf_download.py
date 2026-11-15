"""Helpers for downloading models from Hugging Face Hub with CLI, retry, and safe HTTP fallback."""

from __future__ import annotations

import codecs
import contextlib
import email.utils
import http.client
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .runtime_env import get_secret

_STDLIB_URLOPEN = urllib.request.urlopen
_MAX_OUTPUT_TAIL_CHARS = 32768
_DOWNLOAD_CHUNK_BYTES = 256 * 1024
_MAX_RETRY_AFTER_SEC = 300.0
_MAX_EXP_BACKOFF_SEC = 60.0
_TRUSTED_HF_HOSTS = frozenset({"huggingface.co", "hf.co"})

_RETRY_AFTER_RE = re.compile(
    r"retry[-_\s]+after\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\b",
    re.IGNORECASE,
)
_RATE_LIMIT_TEXT_RE = re.compile(
    r"(?:(?<![-/])\b429\b(?![-/])|too\s+many\s+requests|rate[-\s]?limit(?:ed)?|throttl(?:e|ed|ing))",
    re.IGNORECASE,
)
_XET_ERROR_TEXT_RE = re.compile(
    r"(?:\bxet\b|cas_client|cas\s+service|cas\s+error)",
    re.IGNORECASE,
)
_TRANSIENT_HTTP_TEXT_RE = re.compile(
    r"(?:"
    r"\b(?:408|500|502|503|504)\b|"
    r"bad\s+gateway|"
    r"service\s+unavailable|"
    r"gateway\s+timeout|"
    r"internal\s+server\s+error|"
    r"temporarily\s+unavailable|"
    r"timed?\s*out|"
    r"timeout|"
    r"connection\s+(?:reset|aborted|refused|error|closed)|"
    r"remote\s+end\s+closed|"
    r"incomplete\s*read|"
    r"chunked\s*encoding|"
    r"broken\s+pipe|"
    r"name\s+resolution|"
    r"transient|"
    r"network\s+error|"
    r"eof\s+occurred"
    r")",
    re.IGNORECASE,
)
_NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 410, 422})
_NON_RETRYABLE_TEXT_RE = re.compile(
    r"(?:"
    r"401\s+unauthorized|"
    r"403\s+forbidden|"
    r"404\s+not\s+found|"
    r"entry\s+not\s+found|"
    r"repository\s+not\s+found|"
    r"gatedrepoerror|"
    r"repositorynotfounderror|"
    r"entrynotfounderror|"
    r"revisionnotfounderror"
    r")",
    re.IGNORECASE,
)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Strip Authorization when redirecting outside trusted Hugging Face domains."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        new_host = (urllib.parse.urlparse(redirected.full_url).hostname or "").lower()
        is_trusted = new_host in _TRUSTED_HF_HOSTS or any(
            new_host.endswith(f".{domain}") for domain in _TRUSTED_HF_HOSTS
        )
        if not is_trusted:
            for hdr_name in list(redirected.headers):
                if hdr_name.lower() == "authorization":
                    del redirected.headers[hdr_name]
            for hdr_name in list(redirected.unredirected_hdrs):
                if hdr_name.lower() == "authorization":
                    del redirected.unredirected_hdrs[hdr_name]
        return redirected


def _resolve_token(token: str | None) -> str | None:
    if token is not None:
        return token if token else None
    return get_secret("HF_TOKEN")


def _is_mocked_huggingface_hub() -> bool:
    hub_mod = sys.modules.get("huggingface_hub")
    if hub_mod is None:
        return False
    return not isinstance(getattr(hub_mod, "__file__", None), (str, os.PathLike))


def _resolve_hf_cli(prefer_cli: bool) -> str | None:
    if not prefer_cli or _is_mocked_huggingface_hub():
        return None
    return shutil.which("hf")


def _build_hf_env(*, token: str | None, degrade_level: int = 0) -> dict[str, str]:
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
    else:
        env.pop("HF_TOKEN", None)
    if degrade_level >= 1:
        env["HF_XET_HIGH_PERFORMANCE"] = "0"
    if degrade_level >= 2:
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


@contextlib.contextmanager
def _temporary_env_overrides(degrade_level: int) -> Iterator[None]:
    if degrade_level <= 0:
        yield
        return

    env_keys = ("HF_XET_HIGH_PERFORMANCE", "HF_HUB_DISABLE_XET")
    prev_env = {k: os.environ.get(k) for k in env_keys}
    const_mod = sys.modules.get("huggingface_hub.constants")
    is_real_const = const_mod is not None and isinstance(
        getattr(const_mod, "__file__", None), (str, os.PathLike)
    )
    prev_const: dict[str, object] = {}

    try:
        if degrade_level >= 1:
            os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
            if is_real_const and hasattr(const_mod, "HF_XET_HIGH_PERFORMANCE"):
                prev_const["HF_XET_HIGH_PERFORMANCE"] = getattr(
                    const_mod, "HF_XET_HIGH_PERFORMANCE"
                )
                setattr(const_mod, "HF_XET_HIGH_PERFORMANCE", False)
        if degrade_level >= 2:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            if is_real_const and hasattr(const_mod, "HF_HUB_DISABLE_XET"):
                prev_const["HF_HUB_DISABLE_XET"] = getattr(
                    const_mod, "HF_HUB_DISABLE_XET"
                )
                setattr(const_mod, "HF_HUB_DISABLE_XET", True)
        yield
    finally:
        for key, old_val in prev_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
        for attr, old_val in prev_const.items():
            setattr(const_mod, attr, old_val)


def _parse_retry_after_header(raw_header: object) -> float | None:
    if raw_header is None:
        return None
    text = str(raw_header).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    return max(0.0, dt.timestamp() - time.time())


def _parse_retry_after(text: str, response: object = None) -> float | None:
    headers = getattr(response, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        for key in ("Retry-After", "retry-after"):
            hdr_val = headers.get(key)
            if hdr_val is not None:
                parsed = _parse_retry_after_header(hdr_val)
                if parsed is not None:
                    return parsed
        if hasattr(headers, "items"):
            for key, val in headers.items():
                if str(key).lower() == "retry-after":
                    parsed = _parse_retry_after_header(val)
                    if parsed is not None:
                        return parsed

    matches = _RETRY_AFTER_RE.findall(text)
    if matches:
        try:
            return max(0.0, float(matches[-1]))
        except ValueError:
            return None
    return None


def _extract_status_code(exc: BaseException | None) -> int | None:
    if exc is None:
        return None
    response = getattr(exc, "response", None)
    for candidate in (
        getattr(response, "status_code", None),
        getattr(exc, "status", None),
        getattr(exc, "code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None


def _is_transient_network_exception(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            socket.gaierror,
            socket.herror,
            ssl.SSLError,
            http.client.HTTPException,
        ),
    ):
        return True
    if isinstance(exc, urllib.error.URLError) and not isinstance(
        exc, urllib.error.HTTPError
    ):
        return True
    type_names = {cls.__name__.lower() for cls in type(exc).__mro__}
    return bool(
        type_names
        & {
            "timeout",
            "readtimeout",
            "connecttimeout",
            "connectionerror",
            "chunkedencodingerror",
            "requestexception",
        }
    )


class _RetryExhaustedError(RuntimeError):
    """Internal marker indicating hub retries exhausted on a retryable failure."""


def _classify_download_failure(
    exc: BaseException | None,
    output_text: str = "",
) -> tuple[bool, bool, float | None]:
    """Return ``(is_retryable, is_rate_limit_or_xet, retry_after_sec)``."""
    combined_text = "\n".join(part for part in (str(exc or ""), output_text) if part)
    response = getattr(exc, "response", None) if exc is not None else None
    retry_after = _parse_retry_after(combined_text, response=response)
    status_code = _extract_status_code(exc)

    if status_code == 429:
        return True, True, retry_after

    if status_code in _NON_RETRYABLE_STATUS_CODES:
        return False, False, None

    if _NON_RETRYABLE_TEXT_RE.search(combined_text) and not re.search(
        r"(?:too\s+many\s+requests|rate[-\s]?limit)",
        combined_text,
        re.IGNORECASE,
    ):
        return False, False, None

    if _RATE_LIMIT_TEXT_RE.search(combined_text):
        return True, True, retry_after

    is_xet = bool(_XET_ERROR_TEXT_RE.search(combined_text))
    if status_code in {408, 500, 502, 503, 504}:
        return True, is_xet, retry_after

    if _is_transient_network_exception(exc) or _TRANSIENT_HTTP_TEXT_RE.search(
        combined_text
    ):
        return True, is_xet, retry_after

    if is_xet:
        return True, True, retry_after

    return False, False, None


def _compute_backoff_seconds(
    attempt: int,
    *,
    initial_backoff_sec: float,
    retry_after_sec: float | None,
) -> float:
    exp_backoff = min(_MAX_EXP_BACKOFF_SEC, max(0.0, initial_backoff_sec) * (2**attempt))
    if retry_after_sec is not None:
        return min(
            _MAX_RETRY_AFTER_SEC,
            max(exp_backoff, max(0.0, retry_after_sec) + 1.0),
        )
    return exp_backoff


def _write_stdout_safe(text: str) -> None:
    if not text:
        return
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace"))
    sys.stdout.flush()


def _stream_process_output(
    cmd: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, str]:
    """Run ``cmd`` while teeing stdout/stderr live to ``sys.stdout`` and keeping tail text."""
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    tail_chunks: deque[str] = deque()
    tail_len = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _append_tail(piece: str) -> None:
        nonlocal tail_len
        if not piece:
            return
        tail_chunks.append(piece)
        tail_len += len(piece)
        while tail_len > _MAX_OUTPUT_TAIL_CHARS and tail_chunks:
            dropped = tail_chunks.popleft()
            tail_len -= len(dropped)

    try:
        if proc.stdout is not None:
            while True:
                if hasattr(proc.stdout, "read1"):
                    chunk = proc.stdout.read1(1024)
                else:
                    chunk = proc.stdout.read(1024)
                if isinstance(chunk, bytes):
                    if not chunk:
                        break
                    text_chunk = decoder.decode(chunk, final=False)
                elif isinstance(chunk, str):
                    if not chunk:
                        break
                    text_chunk = chunk
                else:
                    break
                if text_chunk:
                    _write_stdout_safe(text_chunk)
                    _append_tail(text_chunk)

            final_text = decoder.decode(b"", final=True)
            if final_text:
                _write_stdout_safe(final_text)
                _append_tail(final_text)
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
        raise
    finally:
        if proc.stdout is not None and not getattr(proc.stdout, "closed", False):
            proc.stdout.close()
        returncode = proc.wait()

    return returncode, "".join(tail_chunks)


def _open_url(req: urllib.request.Request, *, timeout_sec: float):
    if urllib.request.urlopen is _STDLIB_URLOPEN:
        opener = urllib.request.build_opener(_SafeRedirectHandler)
        return opener.open(req, timeout=timeout_sec)
    return urllib.request.urlopen(req, timeout=timeout_sec)


def _parse_content_range_total(header_val: str | None) -> int | None:
    if not header_val:
        return None
    match = re.search(r"/(\d+)\s*$", header_val)
    if not match:
        return None
    return int(match.group(1))


def _download_via_urllib_resume(
    url: str,
    part_path: Path,
    target: Path,
    *,
    token: str | None,
    timeout_sec: float,
) -> Path:
    """Download ``url`` into ``part_path`` using HTTP ``Range`` resume, then atomically replace ``target``."""
    for reset_retry in range(2):
        existing_bytes = part_path.stat().st_size if part_path.is_file() else 0
        headers = {"User-Agent": "colab-workbench/1.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            resp = _open_url(req, timeout_sec=timeout_sec)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and existing_bytes > 0 and reset_retry == 0:
                total = _parse_content_range_total(
                    exc.headers.get("Content-Range") if exc.headers else None
                )
                if total is not None and total == existing_bytes:
                    os.replace(part_path, target)
                    return target
                part_path.unlink(missing_ok=True)
                continue
            raise

        with resp:
            status = getattr(resp, "status", None)
            if status is None and hasattr(resp, "getcode"):
                status = resp.getcode()
            if status is None:
                status = 200
            if not (200 <= int(status) < 300):
                raise RuntimeError(f"Unexpected HTTP status {status} downloading {url}")

            mode = "ab" if int(status) == 206 and existing_bytes > 0 else "wb"
            content_len_raw = (
                resp.headers.get("Content-Length")
                if getattr(resp, "headers", None) is not None
                else None
            )
            expected_len: int | None = None
            if content_len_raw is not None:
                try:
                    expected_len = int(content_len_raw)
                except ValueError:
                    expected_len = None

            written_this_req = 0
            with part_path.open(mode) as fp:
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    fp.write(chunk)
                    written_this_req += len(chunk)

        if expected_len is not None and written_this_req < expected_len:
            raise OSError(
                f"Incomplete HTTP download for {url}: "
                f"expected {expected_len} bytes, received {written_this_req}"
            )
        if not part_path.is_file() or part_path.stat().st_size <= 0:
            part_path.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded empty payload from {url}")

        os.replace(part_path, target)
        return target

    raise RuntimeError(f"Failed to resume HTTP download for {url}")


def _download_resolve_url_fallback(
    repo_id: str,
    rel_path: str,
    target: Path | str,
    *,
    revision: str | None = None,
    token: str | None = None,
    timeout_sec: float = 60.0,
) -> Path:
    """Fallback direct HTTPS download from Hugging Face ``/resolve/`` using ``.part`` staging.

    Never writes partial or empty payloads directly to ``target``; ``target`` is only
    created via atomic ``os.replace(part_path, target)`` after verified completion.
    """
    if not repo_id or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string.")
    clean_rel = rel_path.lstrip("/")
    if not clean_rel:
        raise ValueError("rel_path must be a non-empty string.")

    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_name(target_path.name + ".part")

    rev = urllib.parse.quote((revision or "main").strip("/"), safe="")
    quoted_rel = urllib.parse.quote(clean_rel, safe="/")
    url = f"https://huggingface.co/{repo_id}/resolve/{rev}/{quoted_rel}"

    wget_bin = None if token else shutil.which("wget")
    if wget_bin:
        try:
            proc = subprocess.run(
                [wget_bin, "-c", "-O", str(part_path), url],
                check=False,
            )
        except OSError:
            wget_bin = None
        else:
            if proc.returncode == 0 and part_path.is_file() and part_path.stat().st_size > 0:
                os.replace(part_path, target_path)
                return target_path
            if part_path.is_file() and part_path.stat().st_size == 0:
                part_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Direct wget fallback failed (exit {proc.returncode}) for {url}"
            )

    return _download_via_urllib_resume(
        url,
        part_path,
        target_path,
        token=token,
        timeout_sec=timeout_sec,
    )


def _run_cli_download_with_retries(
    hf_bin: str,
    repo_id: str,
    rel_paths: Sequence[str],
    *,
    local_dir: Path,
    revision: str | None,
    token: str | None,
    max_retries: int,
    initial_backoff_sec: float,
) -> bool:
    """Run ``hf download`` with 429/transient retries and Xet degradation.

    Returns ``True`` on CLI success. Returns ``False`` when spawning ``hf`` fails
    with an OS/invocation incompatibility so callers can try Python ``huggingface_hub``.
    Raises on non-retryable errors or when retries are exhausted.
    """
    cmd = [hf_bin, "download", repo_id, *rel_paths, "--local-dir", str(local_dir)]
    if revision:
        cmd.extend(["--revision", revision])

    degrade_level = 0
    for attempt in range(max_retries + 1):
        env = _build_hf_env(token=token, degrade_level=degrade_level)
        try:
            returncode, output_tail = _stream_process_output(cmd, env=env)
        except OSError:
            if attempt == 0:
                return False
            raise

        if returncode == 0:
            return True

        if attempt == 0 and re.search(
            r"(?:no such command|unrecognized arguments|unknown option)",
            output_tail,
            re.IGNORECASE,
        ):
            return False

        is_retryable, is_rate_or_xet, retry_after = _classify_download_failure(
            None, output_tail
        )
        if is_rate_or_xet:
            degrade_level = min(2, degrade_level + 1)

        if not is_retryable:
            raise RuntimeError(
                f"hf download failed (exit {returncode}) for {repo_id}: "
                f"{output_tail[-500:].strip()}"
            )

        if attempt >= max_retries:
            raise _RetryExhaustedError(
                f"hf download exhausted {max_retries} retries (exit {returncode}) for {repo_id}: "
                f"{output_tail[-500:].strip()}"
            )

        wait_sec = _compute_backoff_seconds(
            attempt,
            initial_backoff_sec=initial_backoff_sec,
            retry_after_sec=retry_after,
        )
        print(
            f"[hf_download] retry {attempt + 1}/{max_retries} for {repo_id} "
            f"in {wait_sec:.1f}s (degrade_level={degrade_level})...",
            flush=True,
        )
        if wait_sec > 0:
            time.sleep(wait_sec)

    return False


def _run_python_with_retries(
    fn,
    *,
    label: str,
    max_retries: int,
    initial_backoff_sec: float,
):
    degrade_level = 0
    for attempt in range(max_retries + 1):
        try:
            with _temporary_env_overrides(degrade_level):
                return fn()
        except Exception as exc:  # noqa: BLE001
            is_retryable, is_rate_or_xet, retry_after = _classify_download_failure(exc)
            if is_rate_or_xet:
                degrade_level = min(2, degrade_level + 1)
            if not is_retryable or attempt >= max_retries:
                raise
            wait_sec = _compute_backoff_seconds(
                attempt,
                initial_backoff_sec=initial_backoff_sec,
                retry_after_sec=retry_after,
            )
            print(
                f"[hf_download] retry {attempt + 1}/{max_retries} for {label} "
                f"in {wait_sec:.1f}s (degrade_level={degrade_level})...",
                flush=True,
            )
            if wait_sec > 0:
                time.sleep(wait_sec)

    raise RuntimeError(f"Unreachable retry loop termination for {label}")


def download_hf_file(
    repo_id: str,
    filename: str,
    *,
    local_dir: str | Path,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    max_retries: int = 5,
    initial_backoff_sec: float = 5.0,
    prefer_cli: bool = True,
) -> Path:
    """Download a single file from Hugging Face Hub into ``local_dir``."""
    if not repo_id or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string.")
    if not filename or not filename.strip():
        raise ValueError("filename must be a non-empty string.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_token = _resolve_token(token)

    clean_subfolder = subfolder.strip("/") if subfolder else ""
    rel_path = f"{clean_subfolder}/{filename}" if clean_subfolder else filename
    target = dest / rel_path

    hf_bin = _resolve_hf_cli(prefer_cli)
    if hf_bin is not None:
        try:
            cli_handled = _run_cli_download_with_retries(
                hf_bin,
                repo_id,
                [rel_path],
                local_dir=dest,
                revision=revision,
                token=resolved_token,
                max_retries=max_retries,
                initial_backoff_sec=initial_backoff_sec,
            )
            if cli_handled:
                return target
        except _RetryExhaustedError:
            print(
                f"[hf_download] falling back to direct HTTPS /resolve/ for {repo_id}/{rel_path}...",
                flush=True,
            )
            return _download_resolve_url_fallback(
                repo_id,
                rel_path,
                target,
                revision=revision,
                token=resolved_token,
            )

    from huggingface_hub import hf_hub_download

    try:
        downloaded = _run_python_with_retries(
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                revision=revision,
                local_dir=str(dest),
                token=resolved_token,
            ),
            label=f"{repo_id}/{rel_path}",
            max_retries=max_retries,
            initial_backoff_sec=initial_backoff_sec,
        )
        return Path(downloaded)
    except Exception as exc:  # noqa: BLE001
        is_retryable, _, _ = _classify_download_failure(exc)
        if not is_retryable:
            raise
        print(
            f"[hf_download] falling back to direct HTTPS /resolve/ for {repo_id}/{rel_path}...",
            flush=True,
        )
        return _download_resolve_url_fallback(
            repo_id,
            rel_path,
            target,
            revision=revision,
            token=resolved_token,
        )


def download_hf_repo_files(
    repo_id: str,
    filenames: Sequence[str],
    *,
    local_dir: str | Path,
    revision: str | None = None,
    token: str | None = None,
    max_retries: int = 5,
    initial_backoff_sec: float = 5.0,
    prefer_cli: bool = True,
) -> list[Path]:
    """Download multiple files from one Hugging Face repo into ``local_dir`` in one batch."""
    if not repo_id or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")

    file_list = [str(f) for f in filenames]
    for item in file_list:
        if not item.strip():
            raise ValueError("filenames must not contain blank entries.")

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not file_list:
        return []

    resolved_token = _resolve_token(token)
    expected_paths = [dest / f for f in file_list]

    def _fallback_missing_files() -> list[Path]:
        for rel_path, target in zip(file_list, expected_paths):
            if target.is_file() and target.stat().st_size > 0:
                continue
            print(
                f"[hf_download] falling back to direct HTTPS /resolve/ for {repo_id}/{rel_path}...",
                flush=True,
            )
            _download_resolve_url_fallback(
                repo_id,
                rel_path,
                target,
                revision=revision,
                token=resolved_token,
            )
        return expected_paths

    hf_bin = _resolve_hf_cli(prefer_cli)
    if hf_bin is not None:
        try:
            cli_handled = _run_cli_download_with_retries(
                hf_bin,
                repo_id,
                file_list,
                local_dir=dest,
                revision=revision,
                token=resolved_token,
                max_retries=max_retries,
                initial_backoff_sec=initial_backoff_sec,
            )
            if cli_handled:
                return expected_paths
        except _RetryExhaustedError:
            return _fallback_missing_files()

    from huggingface_hub import snapshot_download

    try:
        _run_python_with_retries(
            lambda: snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=str(dest),
                allow_patterns=list(file_list),
                token=resolved_token,
            ),
            label=repo_id,
            max_retries=max_retries,
            initial_backoff_sec=initial_backoff_sec,
        )
        return expected_paths
    except Exception as exc:  # noqa: BLE001
        is_retryable, _, _ = _classify_download_failure(exc)
        if not is_retryable:
            raise
        return _fallback_missing_files()


def download_hf_snapshot(
    repo_id: str,
    *,
    local_dir: str | Path,
    allow_patterns: Iterable[str] | str | None = None,
    ignore_patterns: Iterable[str] | str | None = None,
    revision: str | None = None,
    token: str | None = None,
    max_retries: int = 5,
    initial_backoff_sec: float = 5.0,
) -> Path:
    """Download a repository snapshot (or filtered subset) from Hugging Face Hub."""
    if not repo_id or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string.")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative.")

    from huggingface_hub import snapshot_download

    dest = Path(local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_token = _resolve_token(token)
    allow_arg = (
        list(allow_patterns)
        if isinstance(allow_patterns, (list, tuple, set))
        else allow_patterns
    )
    ignore_arg = (
        list(ignore_patterns)
        if isinstance(ignore_patterns, (list, tuple, set))
        else ignore_patterns
    )

    downloaded = _run_python_with_retries(
        lambda: snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(dest),
            allow_patterns=allow_arg,
            ignore_patterns=ignore_arg,
            token=resolved_token,
        ),
        label=repo_id,
        max_retries=max_retries,
        initial_backoff_sec=initial_backoff_sec,
    )
    return Path(downloaded)
