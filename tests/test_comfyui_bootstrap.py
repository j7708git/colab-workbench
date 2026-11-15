"""Unit tests for the zero-disk ComfyUI Colab bootstrapper."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from comfyui import (
    ModelEntry,
    build_comfyui_command,
    cleanup_expired_files,
    download_preset_models,
    ensure_comfyui_package_ready,
    launch_comfyui_zero_disk,
    load_models_from_yaml,
    prepare_zero_disk_dirs,
    resolve_model_preset,
)


class ModelPresetTests(unittest.TestCase):
    def test_resolve_model_preset_variants(self) -> None:
        qwen = resolve_model_preset("qwen21_turbo_int8")
        moss = resolve_model_preset("moss_sft_int8")
        both = resolve_model_preset(" Both ")

        self.assertEqual(len(qwen), 3)
        self.assertTrue(all(m.repo_id == "Comfy-Org/Qwen-Image-2.1" for m in qwen))

        self.assertEqual(len(moss), 4)
        self.assertTrue(
            any(
                m.repo_id == "Kijai/MOSS-image_experimental"
                and m.filename == "loras/moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors"
                for m in moss
            )
        )

        self.assertEqual(len(both), 7)
        with self.assertRaises(ValueError):
            resolve_model_preset("non-existent-preset")

    def test_load_models_from_yaml_parses_example_with_and_without_pyyaml(self) -> None:
        yaml_path = Path(__file__).resolve().parents[1] / "comfyui" / "models.example.yaml"
        entries = load_models_from_yaml(yaml_path)
        self.assertEqual(len(entries), 7)
        self.assertEqual(entries[0].dest_rel_path(), Path("diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors"))

        with mock.patch.dict(sys.modules, {"yaml": None}):
            fallback_entries = load_models_from_yaml(yaml_path)
        self.assertEqual(fallback_entries, entries)


class ModelDownloadTests(unittest.TestCase):
    def test_download_preset_models_downloads_skips_and_handles_optional(self) -> None:
        calls: list[tuple[str, str, str, str | None]] = []

        def fake_download(repo_id: str, filename: str, *, local_dir, token=None, **kwargs):
            calls.append((repo_id, filename, str(local_dir), token))
            if "fail-optional" in filename:
                raise RuntimeError("transient network error")
            out_path = Path(local_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"weights")
            return out_path

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("comfyui.bootstrap.download_hf_file", side_effect=fake_download),
        ):
            comfy_dir = Path(tmp) / "ComfyUI"
            existing = (
                comfy_dir
                / "models"
                / "vae"
                / "qwen_image_2.1_vae_bf16.safetensors"
            )
            existing.parent.mkdir(parents=True, exist_ok=True)
            existing.write_bytes(b"already-downloaded")

            extras = [
                ModelEntry(
                    name="optional-lora",
                    repo_id="org/repo",
                    filename="loras/fail-optional.safetensors",
                    target_subdir="loras",
                    optional=True,
                ),
                ModelEntry(
                    name="relocated-lora",
                    repo_id="org/repo",
                    filename="nested/custom.safetensors",
                    target_subdir="loras",
                ),
            ]

            got = download_preset_models(
                comfy_dir,
                preset="qwen21_turbo_int8",
                extra_entries=extras,
                token="placeholder-token",
            )

            self.assertEqual(len(got), 4)
            self.assertEqual(len(calls), 4)
            self.assertFalse(any("qwen_image_2.1_vae_bf16" in c[1] for c in calls))
            self.assertTrue((comfy_dir / "models" / "loras" / "custom.safetensors").is_file())
            self.assertEqual(calls[0][3], "placeholder-token")


class ZeroDiskLaunchTests(unittest.TestCase):
    def test_prepare_zero_disk_dirs_and_build_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shm_root = Path(tmp) / "shm"
            dirs = prepare_zero_disk_dirs(shm_root)
            self.assertTrue(dirs["output"].is_dir())
            self.assertTrue(dirs["temp"].is_dir())

            cmd = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                port=8188,
                extra_args=["--lowvram"],
                python_bin="python3",
            )
            self.assertEqual(cmd[0], "python3")
            self.assertIn("--output-directory", cmd)
            self.assertEqual(cmd[cmd.index("--output-directory") + 1], str(dirs["output"]))
            self.assertEqual(cmd[cmd.index("--temp-directory") + 1], str(dirs["temp"]))
            self.assertIn("--disable-auto-launch", cmd)
            self.assertEqual(cmd[-1], "--lowvram")

            with self.assertRaises(ValueError):
                build_comfyui_command("/content/ComfyUI", shm_root=shm_root, port=0)

    def test_cleanup_expired_files_removes_only_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "comfy-out"
            nested = out_dir / "sub"
            nested.mkdir(parents=True)

            old_file = nested / "old.png"
            fresh_file = out_dir / "fresh.png"
            old_file.write_bytes(b"old")
            fresh_file.write_bytes(b"fresh")

            now = 1_700_000_000.0
            os.utime(old_file, (now - 600, now - 600))
            os.utime(fresh_file, (now - 30, now - 30))

            removed = cleanup_expired_files([out_dir], max_age_sec=300.0, now=now)
            self.assertEqual(removed, [old_file])
            self.assertFalse(old_file.exists())
            self.assertTrue(fresh_file.exists())

            with self.assertRaises(ValueError):
                cleanup_expired_files([out_dir], max_age_sec=-1.0)

    def test_launch_comfyui_zero_disk_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comfy_dir = root / "ComfyUI"
            comfy_dir.mkdir()
            shm_root = root / "shm"

            fake_proc = mock.MagicMock()
            fake_proc.poll.return_value = None
            fake_proc.stdout = None

            fake_tunnel = mock.MagicMock()
            fake_tunnel.url = "https://zero-disk-test.trycloudflare.com"

            with (
                mock.patch("comfyui.bootstrap.subprocess.Popen", return_value=fake_proc) as popen_mock,
                mock.patch("comfyui.bootstrap.wait_for_local_server", return_value="http://127.0.0.1:8188"),
                mock.patch("comfyui.bootstrap.start_cloudflared_tunnel", return_value=fake_tunnel) as tunnel_mock,
            ):
                result = launch_comfyui_zero_disk(
                    comfy_dir,
                    shm_root=shm_root,
                    port=8188,
                    reaper_interval_sec=60.0,
                )

                self.assertEqual(result.public_url, "https://zero-disk-test.trycloudflare.com")
                self.assertEqual(result.local_url, "http://127.0.0.1:8188")
                self.assertEqual(result.output_dir, shm_root / "comfy-out")
                self.assertEqual(result.temp_dir, shm_root / "comfy-tmp")

                launched_cmd = popen_mock.call_args[0][0]
                self.assertIn(str(shm_root / "comfy-out"), launched_cmd)
                self.assertIn(str(shm_root / "comfy-tmp"), launched_cmd)
                tunnel_mock.assert_called_once()

                result.stop()
                self.assertTrue(result.reaper_stop.is_set())
                fake_tunnel.stop.assert_called_once()
                fake_proc.terminate.assert_called_once()


class NotebookSmokeTests(unittest.TestCase):
    def test_notebook_step1_supports_branch_parameter(self) -> None:
        nb_path = Path(__file__).resolve().parents[1] / "comfyui" / "comfyui.ipynb"
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = {cell.get("id"): cell for cell in notebook.get("cells", [])}
        self.assertIn("step-1-bootstrap-repo", cells)
        self.assertIn("step-3-download-models", cells)
        self.assertIn("step-4-launch-zero-disk", cells)

        step1_source = "".join(cells["step-1-bootstrap-repo"].get("source", []))
        self.assertIn('BRANCH = "feat/comfyui-zero-disk-bootstrap"  #@param {type:"string"}', step1_source)
        self.assertIn("+refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}", step1_source)
        self.assertIn('"checkout", "-B", BRANCH, f"origin/{BRANCH}"', step1_source)
        self.assertIn("importlib.invalidate_caches()", step1_source)
        self.assertIn("sys.modules", step1_source)
        self.assertIn("comfyui/__init__.py", step1_source)

        for cell_id in ("step-3-download-models", "step-4-launch-zero-disk"):
            cell_source = "".join(cells[cell_id].get("source", []))
            self.assertIn("importlib.invalidate_caches()", cell_source)
            self.assertIn("sys.modules", cell_source)

    def test_namespace_package_cache_recovery(self) -> None:
        real_repo_dir = Path(__file__).resolve().parents[1]
        real_repo_resolved = real_repo_dir.resolve()
        original_sys_path = list(sys.path)
        original_modules = {
            k: v
            for k, v in sys.modules.items()
            if k == "comfyui" or k.startswith("comfyui.")
        }

        def _is_real_repo(entry: str) -> bool:
            try:
                return Path(entry or ".").resolve() == real_repo_resolved
            except OSError:
                return False

        try:
            with tempfile.TemporaryDirectory() as tmp:
                fake_repo = Path(tmp) / "fake_repo"
                (fake_repo / "comfyui").mkdir(parents=True)

                for mod_name in [
                    k for k in list(sys.modules) if k == "comfyui" or k.startswith("comfyui.")
                ]:
                    del sys.modules[mod_name]

                sys.path[:] = [
                    str(fake_repo),
                    *[p for p in original_sys_path if not _is_real_repo(p)],
                ]
                importlib.invalidate_caches()

                import comfyui as stale_pkg

                self.assertIsNone(getattr(stale_pkg, "__file__", None))
                with self.assertRaises(ImportError):
                    from comfyui import download_preset_models as _missing  # noqa: F401

                init_path = ensure_comfyui_package_ready(real_repo_dir)
                self.assertEqual(init_path, (real_repo_dir / "comfyui" / "__init__.py").resolve())

                import comfyui as recovered_pkg
                from comfyui import (
                    download_preset_models as recovered_download,
                    load_models_from_yaml as recovered_loader,
                )

                self.assertEqual(
                    Path(recovered_pkg.__file__).resolve(),
                    (real_repo_dir / "comfyui" / "__init__.py").resolve(),
                )
                self.assertTrue(callable(recovered_download))
                self.assertTrue(callable(recovered_loader))
        finally:
            sys.path[:] = original_sys_path
            for mod_name in [
                k for k in list(sys.modules) if k == "comfyui" or k.startswith("comfyui.")
            ]:
                del sys.modules[mod_name]
            sys.modules.update(original_modules)
            importlib.invalidate_caches()


if __name__ == "__main__":
    unittest.main()
