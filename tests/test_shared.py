"""Unit tests for shared Colab workbench helpers."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from shared import (
    download_hf_file,
    download_hf_repo_files,
    download_hf_snapshot,
    ensure_workspace_dirs,
    export_secret_to_env,
    get_secret,
    get_secret_status,
    is_colab,
    link_dir,
    mount_drive,
    parse_trycloudflare_url,
    require_secret,
    start_cloudflared_tunnel,
)
from shared.hf_download import _download_resolve_url_fallback, _stream_process_output


class SecretsTests(unittest.TestCase):
    def test_get_secret_falls_back_to_env(self) -> None:
        with mock.patch.dict(os.environ, {"TEST_SECRET_KEY": "env-val"}, clear=False):
            self.assertEqual(get_secret("TEST_SECRET_KEY"), "env-val")

    def test_get_secret_prefers_colab_userdata(self) -> None:
        fake_colab = types.ModuleType("google.colab")
        fake_userdata = types.ModuleType("google.colab.userdata")
        fake_userdata.get = lambda name: "colab-val" if name == "HF_TOKEN" else None
        fake_colab.userdata = fake_userdata
        fake_google = types.ModuleType("google")
        fake_google.colab = fake_colab

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.colab": fake_colab,
                    "google.colab.userdata": fake_userdata,
                },
            ),
            mock.patch.dict(os.environ, {"HF_TOKEN": "env-val"}, clear=False),
        ):
            self.assertEqual(get_secret("HF_TOKEN"), "colab-val")
            self.assertTrue(is_colab())

    def test_require_secret_raises_when_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                require_secret("MISSING_KEY")
            self.assertIn("MISSING_KEY", str(ctx.exception))

    def test_export_secret_to_env(self) -> None:
        with mock.patch.dict(os.environ, {"SRC_KEY": "abc123"}, clear=True):
            val = export_secret_to_env("SRC_KEY", env_name="DEST_KEY")
            self.assertEqual(val, "abc123")
            self.assertEqual(os.environ["DEST_KEY"], "abc123")

    def test_get_secret_rejects_blank_name(self) -> None:
        with self.assertRaises(ValueError):
            get_secret("   ")
        with self.assertRaises(ValueError):
            get_secret_status("   ")

    def test_get_secret_warns_on_notebook_access_error_and_status_reports_disabled(self) -> None:
        class NotebookAccessError(Exception):
            pass

        fake_colab = types.ModuleType("google.colab")
        fake_userdata = types.ModuleType("google.colab.userdata")

        def _raise_access(name: str) -> str:
            raise NotebookAccessError(f"Notebook does not have access to secret {name}")

        fake_userdata.get = _raise_access
        fake_colab.userdata = fake_userdata
        fake_google = types.ModuleType("google")
        fake_google.colab = fake_colab

        stderr_buf = io.StringIO()
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.colab": fake_colab,
                    "google.colab.userdata": fake_userdata,
                },
            ),
            mock.patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stderr(stderr_buf),
        ):
            self.assertIsNone(get_secret("HF_TOKEN"))
            self.assertEqual(
                get_secret_status("HF_TOKEN"),
                (False, "notebook-access-disabled"),
            )

        warn_text = stderr_buf.getvalue()
        self.assertIn("HF_TOKEN", warn_text)
        self.assertIn("Notebook access", warn_text)

    def test_get_secret_silent_on_secret_not_found_and_warns_on_timeout(self) -> None:
        class SecretNotFoundError(Exception):
            pass

        class TimeoutException(Exception):
            pass

        fake_colab = types.ModuleType("google.colab")
        fake_userdata = types.ModuleType("google.colab.userdata")
        fake_colab.userdata = fake_userdata
        fake_google = types.ModuleType("google")
        fake_google.colab = fake_colab

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.colab": fake_colab,
                    "google.colab.userdata": fake_userdata,
                },
            ),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            fake_userdata.get = mock.Mock(side_effect=SecretNotFoundError("Secret HF_TOKEN does not exist"))
            missing_stderr = io.StringIO()
            with contextlib.redirect_stderr(missing_stderr):
                self.assertIsNone(get_secret("HF_TOKEN"))
                self.assertEqual(get_secret_status("HF_TOKEN"), (False, "not-set"))
            self.assertEqual(missing_stderr.getvalue(), "")

            fake_userdata.get = mock.Mock(side_effect=TimeoutException("Request timed out"))
            timeout_stderr = io.StringIO()
            with contextlib.redirect_stderr(timeout_stderr):
                self.assertIsNone(get_secret("HF_TOKEN"))
                self.assertEqual(get_secret_status("HF_TOKEN"), (False, "colab-secret-timeout"))
            self.assertIn("Notebook access", timeout_stderr.getvalue())

    def test_get_secret_status_reports_sources_without_leaking_token(self) -> None:
        fake_colab = types.ModuleType("google.colab")
        fake_userdata = types.ModuleType("google.colab.userdata")
        fake_userdata.get = lambda name: "secret-do-not-leak" if name == "HF_TOKEN" else None
        fake_colab.userdata = fake_userdata
        fake_google = types.ModuleType("google")
        fake_google.colab = fake_colab

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.colab": fake_colab,
                    "google.colab.userdata": fake_userdata,
                },
            ),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            ok, source = get_secret_status("HF_TOKEN")
            self.assertTrue(ok)
            self.assertEqual(source, "colab-secret")
            self.assertNotIn("secret-do-not-leak", source)

        with (
            mock.patch.dict(sys.modules, {"google.colab": None, "google.colab.userdata": None}),
            mock.patch.dict(os.environ, {"HF_TOKEN": "env-do-not-leak"}, clear=True),
        ):
            ok, source = get_secret_status("HF_TOKEN")
            self.assertTrue(ok)
            self.assertEqual(source, "env")
            self.assertNotIn("env-do-not-leak", source)

        with (
            mock.patch.dict(sys.modules, {"google.colab": None, "google.colab.userdata": None}),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(get_secret_status("HF_TOKEN"), (False, "not-set"))


class DriveMountTests(unittest.TestCase):
    def test_ensure_workspace_dirs_creates_expected_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dirs = ensure_workspace_dirs(Path(tmp) / "workbench")
            self.assertTrue(dirs["root"].is_dir())
            for name in ("models", "loras", "datasets", "outputs", "configs"):
                self.assertTrue(dirs[name].is_dir())

    def test_link_dir_replaces_empty_dir_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "drive_models"
            link = root / "app" / "models"
            link.mkdir(parents=True)

            try:
                created = link_dir(target, link)
            except OSError as exc:
                if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                    self.skipTest("Windows symlink privilege not enabled (WinError 1314)")
                raise
            self.assertTrue(created.is_symlink())
            self.assertEqual(created.resolve(), target.resolve())

            # Calling again should be a no-op.
            again = link_dir(target, link)
            self.assertEqual(again.resolve(), target.resolve())

    def test_link_dir_refuses_non_empty_dir_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "drive_models"
            link = root / "app" / "models"
            link.mkdir(parents=True)
            (link / "keep.txt").write_text("data", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                link_dir(target, link)

            try:
                replaced = link_dir(target, link, force=True)
            except OSError as exc:
                if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                    self.skipTest("Windows symlink privilege not enabled (WinError 1314)")
                raise
            self.assertTrue(replaced.is_symlink())
            self.assertEqual(replaced.resolve(), target.resolve())

    def test_mount_drive_returns_existing_non_empty_mount_without_colab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mount = Path(tmp)
            (mount / "MyDrive").mkdir()
            self.assertEqual(mount_drive(mount), mount)


class HfDownloadTests(unittest.TestCase):
    def test_download_hf_file_and_snapshot_forward_secret_token(self) -> None:
        calls: list[tuple[str, dict]] = []
        fake_hf = types.ModuleType("huggingface_hub")

        def fake_hf_hub_download(**kwargs):
            calls.append(("file", kwargs))
            return str(Path(kwargs["local_dir"]) / kwargs["filename"])

        def fake_snapshot_download(**kwargs):
            calls.append(("snapshot", kwargs))
            return kwargs["local_dir"]

        fake_hf.hf_hub_download = fake_hf_hub_download
        fake_hf.snapshot_download = fake_snapshot_download

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
            mock.patch.dict(os.environ, {"HF_TOKEN": "placeholder-token-value"}, clear=False),
        ):
            file_path = download_hf_file("org/repo", "model.gguf", local_dir=tmp)
            snap_path = download_hf_snapshot(
                "org/repo",
                local_dir=tmp,
                allow_patterns=["*.json"],
            )

            self.assertEqual(file_path, Path(tmp) / "model.gguf")
            self.assertEqual(snap_path, Path(tmp))
            self.assertEqual(calls[0][1]["token"], "placeholder-token-value")
            self.assertEqual(calls[1][1]["token"], "placeholder-token-value")
            self.assertEqual(calls[1][1]["allow_patterns"], ["*.json"])

    def test_hf_cli_invocation_uses_env_token_and_batches_repo_files(self) -> None:
        popen_calls: list[tuple[list[str], dict[str, str]]] = []

        def fake_popen(cmd, **kwargs):
            popen_calls.append((list(cmd), dict(kwargs["env"])))
            proc = mock.MagicMock()
            proc.stdout = io.BytesIO(b"downloading\r100%\n")
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            return proc

        stdout_buf = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("shared.hf_download.shutil.which", return_value="/usr/bin/hf"),
            mock.patch("shared.hf_download.subprocess.Popen", side_effect=fake_popen),
            mock.patch.dict(os.environ, {}, clear=True),
            contextlib.redirect_stdout(stdout_buf),
        ):
            single_path = download_hf_file(
                "org/repo",
                "model.safetensors",
                subfolder="/diffusion_models/",
                revision="rev-1",
                local_dir=tmp,
                token="placeholder-cli-token",
            )
            batch_paths = download_hf_repo_files(
                "org/repo",
                ["text_encoders/enc.safetensors", "vae/vae.safetensors"],
                local_dir=tmp,
                token="placeholder-cli-token",
            )

        self.assertEqual(single_path, Path(tmp) / "diffusion_models" / "model.safetensors")
        self.assertEqual(
            batch_paths,
            [
                Path(tmp) / "text_encoders" / "enc.safetensors",
                Path(tmp) / "vae" / "vae.safetensors",
            ],
        )
        self.assertEqual(len(popen_calls), 2)

        cmd1, env1 = popen_calls[0]
        self.assertEqual(
            cmd1,
            [
                "/usr/bin/hf",
                "download",
                "org/repo",
                "diffusion_models/model.safetensors",
                "--local-dir",
                tmp,
                "--revision",
                "rev-1",
            ],
        )
        self.assertNotIn("--token", cmd1)
        self.assertNotIn("placeholder-cli-token", " ".join(cmd1))
        self.assertEqual(env1.get("HF_TOKEN"), "placeholder-cli-token")
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", env1)

        cmd2, env2 = popen_calls[1]
        self.assertEqual(
            cmd2,
            [
                "/usr/bin/hf",
                "download",
                "org/repo",
                "text_encoders/enc.safetensors",
                "vae/vae.safetensors",
                "--local-dir",
                tmp,
            ],
        )
        self.assertNotIn("--token", cmd2)
        self.assertEqual(env2.get("HF_TOKEN"), "placeholder-cli-token")
        self.assertIn("downloading", stdout_buf.getvalue())

    def test_429_retry_parses_retry_after_and_degrades_xet_cli_and_python(self) -> None:
        cli_envs: list[dict[str, str]] = []
        cli_outputs = [
            (1, b"HTTP 429 Too Many Requests\nRetry-After: 7\n"),
            (1, b"Rate limit hit. Retry after 15 seconds.\n"),
            (0, b"done\n"),
        ]

        def fake_popen(cmd, **kwargs):
            cli_envs.append(dict(kwargs["env"]))
            rc, payload = cli_outputs.pop(0)
            proc = mock.MagicMock()
            proc.stdout = io.BytesIO(payload)
            proc.poll.return_value = rc
            proc.wait.return_value = rc
            return proc

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("shared.hf_download.shutil.which", return_value="/usr/bin/hf"),
            mock.patch("shared.hf_download.subprocess.Popen", side_effect=fake_popen),
            mock.patch("shared.hf_download.time.sleep") as sleep_mock,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            got = download_hf_file(
                "org/repo",
                "m.safetensors",
                local_dir=tmp,
                max_retries=5,
                initial_backoff_sec=5.0,
            )

        self.assertEqual(got, Path(tmp) / "m.safetensors")
        self.assertEqual(len(cli_envs), 3)
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", cli_envs[0])
        self.assertNotIn("HF_HUB_DISABLE_XET", cli_envs[0])
        self.assertEqual(cli_envs[1].get("HF_XET_HIGH_PERFORMANCE"), "0")
        self.assertNotIn("HF_HUB_DISABLE_XET", cli_envs[1])
        self.assertEqual(cli_envs[2].get("HF_XET_HIGH_PERFORMANCE"), "0")
        self.assertEqual(cli_envs[2].get("HF_HUB_DISABLE_XET"), "1")
        self.assertEqual(
            [c.args[0] for c in sleep_mock.call_args_list],
            [8.0, 16.0],
        )

        # Verify Python hub path exponential backoff (5 -> 10 -> 20 -> 40 -> 60) + Xet downgrade
        fake_hf = types.ModuleType("huggingface_hub")
        py_envs: list[tuple[str | None, str | None]] = []

        class FakeResponse:
            def __init__(self, status_code: int, headers: dict[str, str] | None = None):
                self.status_code = status_code
                self.headers = headers or {}

        class FakeHubError(Exception):
            def __init__(self, msg: str, status_code: int):
                super().__init__(msg)
                self.response = FakeResponse(status_code)

        def fake_snapshot_download(**kwargs):
            py_envs.append(
                (
                    os.environ.get("HF_XET_HIGH_PERFORMANCE"),
                    os.environ.get("HF_HUB_DISABLE_XET"),
                )
            )
            if len(py_envs) <= 5:
                raise FakeHubError("429 Too Many Requests", 429)
            return kwargs["local_dir"]

        fake_hf.snapshot_download = fake_snapshot_download

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}),
            mock.patch("shared.hf_download.time.sleep") as py_sleep_mock,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            paths = download_hf_repo_files(
                "org/repo",
                ["a.bin", "b.bin"],
                local_dir=tmp,
                max_retries=5,
                initial_backoff_sec=5.0,
            )
            self.assertNotIn("HF_XET_HIGH_PERFORMANCE", os.environ)
            self.assertNotIn("HF_HUB_DISABLE_XET", os.environ)

        self.assertEqual(paths, [Path(tmp) / "a.bin", Path(tmp) / "b.bin"])
        self.assertEqual(
            [c.args[0] for c in py_sleep_mock.call_args_list],
            [5.0, 10.0, 20.0, 40.0, 60.0],
        )
        self.assertEqual(py_envs[0], (None, None))
        self.assertEqual(py_envs[1], ("0", None))
        self.assertEqual(py_envs[2], ("0", "1"))

    def test_download_hf_file_exhausted_429_uses_fallback_and_404_fails_fast(self) -> None:
        def always_429(cmd, **kwargs):
            proc = mock.MagicMock()
            proc.stdout = io.BytesIO(b"429 Too Many Requests\n")
            proc.poll.return_value = 1
            proc.wait.return_value = 1
            return proc

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("shared.hf_download.shutil.which", return_value="/usr/bin/hf"),
            mock.patch("shared.hf_download.subprocess.Popen", side_effect=always_429),
            mock.patch("shared.hf_download.time.sleep"),
            mock.patch(
                "shared.hf_download._download_resolve_url_fallback",
                return_value=Path(tmp) / "m.bin",
            ) as fb_mock,
        ):
            out = download_hf_file(
                "org/repo",
                "m.bin",
                local_dir=tmp,
                max_retries=1,
                initial_backoff_sec=0.0,
                token="tok-abc",
            )
            self.assertEqual(out, Path(tmp) / "m.bin")
            fb_mock.assert_called_once()

        def fail_404(cmd, **kwargs):
            proc = mock.MagicMock()
            proc.stdout = io.BytesIO(b"404 Not Found: https://huggingface.co/org/model-429/resolve/main/m.bin\n")
            proc.poll.return_value = 1
            proc.wait.return_value = 1
            return proc

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("shared.hf_download.shutil.which", return_value="/usr/bin/hf"),
            mock.patch("shared.hf_download.subprocess.Popen", side_effect=fail_404) as popen_mock,
            mock.patch("shared.hf_download._download_resolve_url_fallback") as fb_mock,
        ):
            with self.assertRaises(RuntimeError):
                download_hf_file("org/model-429", "m.bin", local_dir=tmp, max_retries=3)
            self.assertEqual(popen_mock.call_count, 1)
            fb_mock.assert_not_called()

    def test_download_resolve_url_fallback_atomic_part_and_failure_safety(self) -> None:
        class _FakeHTTPResponse:
            def __init__(self, status: int, chunks: list[bytes], headers: dict[str, str]):
                self.status = status
                self._chunks = list(chunks)
                self.headers = headers

            def read(self, _n: int = -1) -> bytes:
                if self._chunks:
                    return self._chunks.pop(0)
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        # 1. Range resume from existing .part -> atomic replace to target
        requests_seen: list[tuple[str, dict[str, str]]] = []

        def fake_urlopen_resume(req, timeout=60.0):
            requests_seen.append((req.full_url, dict(req.header_items())))
            return _FakeHTTPResponse(206, [b"world"], {"Content-Length": "5"})

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "models" / "test.safetensors"
            part = target.with_name(target.name + ".part")
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(b"hello ")

            with (
                mock.patch("shared.hf_download.shutil.which", return_value=None),
                mock.patch("shared.hf_download.urllib.request.urlopen", side_effect=fake_urlopen_resume),
            ):
                got = _download_resolve_url_fallback(
                    "org/repo",
                    "models/test.safetensors",
                    target,
                    revision="main",
                    token="tok-fallback",
                )

            self.assertEqual(got, target)
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_bytes(), b"hello world")
            self.assertFalse(part.exists())
            url, hdrs = requests_seen[0]
            self.assertEqual(url, "https://huggingface.co/org/repo/resolve/main/models/test.safetensors")
            hdrs_lower = {k.lower(): v for k, v in hdrs.items()}
            self.assertEqual(hdrs_lower.get("authorization"), "Bearer tok-fallback")
            self.assertEqual(hdrs_lower.get("range"), "bytes=6-")

        # 2. Truncated stream MUST NOT create final target
        def fake_urlopen_truncated(req, timeout=60.0):
            return _FakeHTTPResponse(200, [b"partial"], {"Content-Length": "100"})

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "broken.safetensors"
            part = target.with_name(target.name + ".part")
            with (
                mock.patch("shared.hf_download.shutil.which", return_value=None),
                mock.patch("shared.hf_download.urllib.request.urlopen", side_effect=fake_urlopen_truncated),
            ):
                with self.assertRaises(OSError):
                    _download_resolve_url_fallback("org/repo", "broken.safetensors", target)

            self.assertFalse(target.exists())
            self.assertTrue(part.is_file())
            self.assertEqual(part.read_bytes(), b"partial")

        # 3. HTTP 429 on fallback MUST NOT create final target
        def fake_urlopen_429(req, timeout=60.0):
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "rate_limited.safetensors"
            with (
                mock.patch("shared.hf_download.shutil.which", return_value=None),
                mock.patch("shared.hf_download.urllib.request.urlopen", side_effect=fake_urlopen_429),
            ):
                with self.assertRaises(urllib.error.HTTPError):
                    _download_resolve_url_fallback("org/repo", "rate_limited.safetensors", target)
            self.assertFalse(target.exists())

        # 4. wget -c path writes to .part and never creates target on failure
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "via_wget.safetensors"
            part = target.with_name(target.name + ".part")

            def fake_wget_ok(cmd, check=False):
                self.assertIn("-c", cmd)
                self.assertEqual(cmd[cmd.index("-O") + 1], str(part))
                self.assertFalse(target.exists())
                part.write_bytes(b"wget-payload")
                return mock.MagicMock(returncode=0)

            with (
                mock.patch("shared.hf_download.shutil.which", side_effect=lambda b: "/usr/bin/wget" if b == "wget" else None),
                mock.patch("shared.hf_download.subprocess.run", side_effect=fake_wget_ok),
            ):
                got = _download_resolve_url_fallback("org/repo", "via_wget.safetensors", target)
            self.assertEqual(got, target)
            self.assertEqual(target.read_bytes(), b"wget-payload")
            self.assertFalse(part.exists())

            target_fail = Path(tmp) / "via_wget_fail.safetensors"
            part_fail = target_fail.with_name(target_fail.name + ".part")

            def fake_wget_fail(cmd, check=False):
                part_fail.write_bytes(b"partial-wget")
                return mock.MagicMock(returncode=8)

            with (
                mock.patch("shared.hf_download.shutil.which", side_effect=lambda b: "/usr/bin/wget" if b == "wget" else None),
                mock.patch("shared.hf_download.subprocess.run", side_effect=fake_wget_fail),
            ):
                with self.assertRaises(RuntimeError):
                    _download_resolve_url_fallback("org/repo", "via_wget_fail.safetensors", target_fail)
            self.assertFalse(target_fail.exists())
            self.assertTrue(part_fail.is_file())

    def test_stream_process_output_real_subprocess(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc, tail = _stream_process_output(
                [sys.executable, "-c", "import sys; sys.stdout.write('step 1\\rstep 2\\n')"],
                env=os.environ.copy(),
            )
        self.assertEqual(rc, 0)
        self.assertIn("step 2", tail)
        self.assertIn("step 2", buf.getvalue())


class TunnelTests(unittest.TestCase):
    @staticmethod
    def _write_fake_cloudflared(tmp_dir: Path, script_body: str, name: str = "cloudflared") -> Path:
        py_script = Path(tmp_dir) / f"{name}.py"
        py_script.write_text(script_body, encoding="utf-8")
        if os.name == "nt":
            fake_bin = Path(tmp_dir) / f"{name}.cmd"
            fake_bin.write_text(
                f'@echo off\r\n"{sys.executable}" -u "{py_script}" %*\r\n',
                encoding="utf-8",
            )
            return fake_bin
        fake_bin = Path(tmp_dir) / name
        fake_bin.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" -u "{py_script}" "$@"\n',
            encoding="utf-8",
        )
        fake_bin.chmod(0o755)
        return fake_bin

    def test_parse_trycloudflare_url(self) -> None:
        line = "2026-11-14T14:00:00Z INF |  https://bright-forest-123.trycloudflare.com  |"
        self.assertEqual(
            parse_trycloudflare_url(line),
            "https://bright-forest-123.trycloudflare.com",
        )
        self.assertIsNone(parse_trycloudflare_url("nothing here"))

    def test_start_cloudflared_tunnel_parses_url_and_stops(self) -> None:
        from shared import DEFAULT_TUNNEL_PROTOCOL, SUPPORTED_TUNNEL_PROTOCOLS

        self.assertEqual(DEFAULT_TUNNEL_PROTOCOL, "http2")
        self.assertEqual(SUPPORTED_TUNNEL_PROTOCOLS, ("auto", "http2", "quic"))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_file = tmp_path / "argv.txt"
            log_file = tmp_path / "logs" / "cloudflared.log"
            fake_bin = self._write_fake_cloudflared(
                tmp_path,
                "import pathlib, sys, time\n"
                f"pathlib.Path({str(argv_file)!r}).write_text('\\n'.join(sys.argv[1:]), encoding='utf-8')\n"
                "print('starting tunnel', flush=True)\n"
                "print('https://demo-unit-test.trycloudflare.com', flush=True)\n"
                "time.sleep(0.5 if sys.platform == 'win32' else 10)\n",
            )

            tunnel = start_cloudflared_tunnel(
                8188,
                timeout_sec=5.0,
                cloudflared_bin=str(fake_bin),
                log_file=log_file,
            )
            try:
                self.assertEqual(tunnel.url, "https://demo-unit-test.trycloudflare.com")
                self.assertEqual(tunnel.port, 8188)
                self.assertEqual(tunnel.protocol, "http2")
                self.assertEqual(tunnel.log_file, log_file)
                self.assertIn("https://demo-unit-test.trycloudflare.com", tunnel.tail_logs())
            finally:
                tunnel.stop()

            argv = argv_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                argv,
                ["tunnel", "--protocol", "http2", "--url", "http://127.0.0.1:8188", "--no-autoupdate"],
            )
            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("starting tunnel", log_text)
            self.assertIn("https://demo-unit-test.trycloudflare.com", log_text)

    def test_start_cloudflared_tunnel_protocol_options_and_validation(self) -> None:
        with self.assertRaises(ValueError):
            start_cloudflared_tunnel(8188, protocol="invalid-proto")
        with self.assertRaises(ValueError):
            start_cloudflared_tunnel(8188, protocol="  ")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            argv_file = tmp_path / "argv.txt"
            fake_bin = self._write_fake_cloudflared(
                tmp_path,
                "import pathlib, sys\n"
                f"pathlib.Path({str(argv_file)!r}).write_text('\\n'.join(sys.argv[1:]), encoding='utf-8')\n"
                "print('https://proto-test.trycloudflare.com', flush=True)\n",
            )

            tunnel_quic = start_cloudflared_tunnel(
                8188,
                timeout_sec=5.0,
                cloudflared_bin=str(fake_bin),
                protocol=" QUIC ",
            )
            try:
                self.assertEqual(tunnel_quic.protocol, "quic")
            finally:
                tunnel_quic.stop()
            self.assertEqual(
                argv_file.read_text(encoding="utf-8").splitlines(),
                ["tunnel", "--protocol", "quic", "--url", "http://127.0.0.1:8188", "--no-autoupdate"],
            )

            tunnel_none = start_cloudflared_tunnel(
                8188,
                timeout_sec=5.0,
                cloudflared_bin=str(fake_bin),
                protocol=None,
            )
            try:
                self.assertIsNone(tunnel_none.protocol)
            finally:
                tunnel_none.stop()
            argv_none = argv_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                argv_none,
                ["tunnel", "--url", "http://127.0.0.1:8188", "--no-autoupdate"],
            )

    def test_start_cloudflared_tunnel_drains_large_stdout_without_pipe_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "cloudflared.log"
            fake_bin = self._write_fake_cloudflared(
                tmp_path,
                "import sys\n"
                "print('https://pipe-drain-test.trycloudflare.com', flush=True)\n"
                "payload = 'X' * 100\n"
                "for idx in range(1600):\n"
                "    sys.stdout.write(f'log-{idx:04d} {payload}\\n')\n"
                "sys.stdout.write('SENTINEL_AFTER_LARGE_BURST\\n')\n"
                "sys.stdout.flush()\n",
            )

            tunnel = start_cloudflared_tunnel(
                8188,
                timeout_sec=5.0,
                cloudflared_bin=str(fake_bin),
                log_file=log_file,
            )
            try:
                self.assertEqual(tunnel.url, "https://pipe-drain-test.trycloudflare.com")
                rc = tunnel.process.wait(timeout=10)
                self.assertEqual(rc, 0)
            finally:
                tunnel.stop()

            self.assertIn("SENTINEL_AFTER_LARGE_BURST", tunnel.tail_logs(lines=5))
            self.assertLessEqual(len(tunnel.tail_logs(lines=200).splitlines()), 100)
            self.assertGreater(log_file.stat().st_size, 128 * 1024)
            self.assertIn("SENTINEL_AFTER_LARGE_BURST", log_file.read_text(encoding="utf-8"))

    def test_start_cloudflared_tunnel_fails_fast_on_early_exit(self) -> None:
        import time

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = self._write_fake_cloudflared(
                tmp_path,
                "import sys\n"
                "print('ERR simulated startup failure', flush=True)\n"
                "sys.exit(1)\n",
            )
            started = time.monotonic()
            with self.assertRaises(TimeoutError) as ctx:
                start_cloudflared_tunnel(
                    8188,
                    timeout_sec=10.0,
                    cloudflared_bin=str(fake_bin),
                )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 5.0)
            self.assertIn("ERR simulated startup failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
