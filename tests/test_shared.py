"""Unit tests for shared Colab workbench helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from shared import (
    download_hf_file,
    download_hf_snapshot,
    ensure_workspace_dirs,
    export_secret_to_env,
    get_secret,
    is_colab,
    link_dir,
    mount_drive,
    parse_trycloudflare_url,
    require_secret,
    start_cloudflared_tunnel,
)


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

            created = link_dir(target, link)
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

            replaced = link_dir(target, link, force=True)
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


class TunnelTests(unittest.TestCase):
    def test_parse_trycloudflare_url(self) -> None:
        line = "2026-11-14T14:00:00Z INF |  https://bright-forest-123.trycloudflare.com  |"
        self.assertEqual(
            parse_trycloudflare_url(line),
            "https://bright-forest-123.trycloudflare.com",
        )
        self.assertIsNone(parse_trycloudflare_url("nothing here"))

    def test_start_cloudflared_tunnel_parses_url_and_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "cloudflared"
            fake_bin.write_text(
                "#!/usr/bin/env python3\n"
                "import sys, time\n"
                "print('starting tunnel', flush=True)\n"
                "print('https://demo-unit-test.trycloudflare.com', flush=True)\n"
                "time.sleep(10)\n",
                encoding="utf-8",
            )
            fake_bin.chmod(0o755)

            tunnel = start_cloudflared_tunnel(
                8188,
                timeout_sec=5.0,
                cloudflared_bin=str(fake_bin),
            )
            try:
                self.assertEqual(tunnel.url, "https://demo-unit-test.trycloudflare.com")
                self.assertEqual(tunnel.port, 8188)
            finally:
                tunnel.stop()


if __name__ == "__main__":
    unittest.main()
