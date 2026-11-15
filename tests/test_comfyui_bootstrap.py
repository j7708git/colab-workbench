"""Unit tests for the zero-disk ComfyUI Colab bootstrapper."""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
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
    def test_download_preset_models_batches_by_repo_skips_existing_and_handles_optional(self) -> None:
        batch_calls: list[tuple[str, list[str], str, str | None]] = []
        file_calls: list[tuple[str, str, str, str | None]] = []

        def fake_repo_download(repo_id: str, filenames, *, local_dir, token=None, **kwargs):
            file_list = list(filenames)
            batch_calls.append((repo_id, file_list, str(local_dir), token))
            paths: list[Path] = []
            for fname in file_list:
                if "fail-optional" in fname:
                    raise RuntimeError("transient network error")
                out_path = Path(local_dir) / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"weights")
                paths.append(out_path)
            return paths

        def fake_file_download(repo_id: str, filename: str, *, local_dir, token=None, **kwargs):
            file_calls.append((repo_id, filename, str(local_dir), token))
            out_path = Path(local_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"relocated-weights")
            return out_path

        clean_env = dict(os.environ)
        clean_env.pop("HF_XET_HIGH_PERFORMANCE", None)
        buf = io.StringIO()

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, clean_env, clear=True),
            mock.patch("comfyui.bootstrap.download_hf_repo_files", side_effect=fake_repo_download),
            mock.patch("comfyui.bootstrap.download_hf_file", side_effect=fake_file_download),
            mock.patch("comfyui.bootstrap.get_secret_status", return_value=(False, "not-set")),
            mock.patch("comfyui.bootstrap.time.sleep") as sleep_mock,
            redirect_stdout(buf),
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
                    name="required-lora",
                    repo_id="org/repo",
                    filename="loras/required.safetensors",
                    target_subdir="loras",
                    optional=False,
                ),
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
                pause_sec=1.0,
            )

            self.assertEqual(len(got), 5)
            # 1 batch call for Comfy-Org/Qwen-Image-2.1 (2 missing files),
            # 1 required batch call for org/repo, 1 optional call for org/repo.
            self.assertEqual(len(batch_calls), 3)
            self.assertEqual(batch_calls[0][0], "Comfy-Org/Qwen-Image-2.1")
            self.assertEqual(
                batch_calls[0][1],
                [
                    "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
                    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                ],
            )
            self.assertEqual(batch_calls[1][0], "org/repo")
            self.assertEqual(batch_calls[1][1], ["loras/required.safetensors"])
            self.assertEqual(batch_calls[2][0], "org/repo")
            self.assertEqual(batch_calls[2][1], ["loras/fail-optional.safetensors"])
            self.assertFalse(
                any("qwen_image_2.1_vae_bf16" in fname for _, files, _, _ in batch_calls for fname in files)
            )

            self.assertEqual(len(file_calls), 1)
            self.assertEqual(file_calls[0][0], "org/repo")
            self.assertEqual(file_calls[0][1], "nested/custom.safetensors")
            expected_staging = comfy_dir / "models" / ".hf-staging"
            self.assertEqual(Path(file_calls[0][2]), expected_staging)
            self.assertTrue(expected_staging.is_dir())
            self.assertTrue((comfy_dir / "models" / "loras" / "custom.safetensors").is_file())

            self.assertEqual(batch_calls[0][3], "placeholder-token")
            self.assertEqual(file_calls[0][3], "placeholder-token")
            self.assertEqual(os.environ.get("HF_XET_HIGH_PERFORMANCE"), "1")
            sleep_mock.assert_called_once_with(1.0)
            self.assertNotIn("placeholder-token", buf.getvalue())

    def test_download_preset_models_skips_xet_high_perf_without_token_and_pauses_only_if_positive(self) -> None:
        batch_calls: list[tuple[str, list[str], str | None]] = []

        def fake_repo_download(repo_id: str, filenames, *, local_dir, token=None, **kwargs):
            file_list = list(filenames)
            batch_calls.append((repo_id, file_list, token))
            paths: list[Path] = []
            for fname in file_list:
                out_path = Path(local_dir) / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"weights")
                paths.append(out_path)
            return paths

        clean_env = dict(os.environ)
        clean_env.pop("HF_TOKEN", None)
        clean_env.pop("HF_XET_HIGH_PERFORMANCE", None)
        buf = io.StringIO()

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, clean_env, clear=True),
            mock.patch("comfyui.bootstrap.get_secret", return_value=None),
            mock.patch(
                "comfyui.bootstrap.get_secret_status",
                return_value=(False, "notebook-access-disabled"),
            ),
            mock.patch("comfyui.bootstrap.download_hf_repo_files", side_effect=fake_repo_download),
            mock.patch("comfyui.bootstrap.time.sleep") as sleep_mock,
            redirect_stdout(buf),
        ):
            comfy_dir = Path(tmp) / "ComfyUI"
            got = download_preset_models(comfy_dir, preset="both", pause_sec=0.0)

            self.assertEqual(len(got), 7)
            self.assertEqual(
                [repo for repo, _, _ in batch_calls],
                [
                    "Comfy-Org/Qwen-Image-2.1",
                    "Comfy-Org/MOSS-Image",
                    "Kijai/MOSS-image_experimental",
                ],
            )
            self.assertTrue(all(tok is None for _, _, tok in batch_calls))
            self.assertNotIn("HF_XET_HIGH_PERFORMANCE", os.environ)
            sleep_mock.assert_not_called()
            output = buf.getvalue()
            self.assertIn("notebook-access-disabled", output)
            self.assertIn("Notebook access", output)

    def test_download_preset_models_multiple_optional_batch_falls_back_per_file(self) -> None:
        def fake_repo_download(repo_id: str, filenames, *, local_dir, token=None, **kwargs):
            file_list = list(filenames)
            if len(file_list) > 1:
                raise RuntimeError("batch 404 on optional file")
            if "bad" in file_list[0]:
                raise RuntimeError("missing optional file")
            out_path = Path(local_dir) / file_list[0]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"ok")
            return [out_path]

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("comfyui.bootstrap.download_hf_repo_files", side_effect=fake_repo_download),
            mock.patch("comfyui.bootstrap.get_secret_status", return_value=(False, "not-set")),
        ):
            comfy_dir = Path(tmp) / "ComfyUI"
            extras = [
                ModelEntry(
                    name="opt-good",
                    repo_id="org/opt",
                    filename="loras/good.safetensors",
                    target_subdir="loras",
                    optional=True,
                ),
                ModelEntry(
                    name="opt-bad",
                    repo_id="org/opt",
                    filename="loras/bad.safetensors",
                    target_subdir="loras",
                    optional=True,
                ),
            ]
            # Use empty existing preset? Make all preset files exist first so only extras download.
            for entry in resolve_model_preset("qwen21_turbo_int8"):
                p = comfy_dir / "models" / entry.dest_rel_path()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"preset")

            got = download_preset_models(
                comfy_dir,
                preset="qwen21_turbo_int8",
                extra_entries=extras,
                token="tok",
                pause_sec=0.0,
            )
            self.assertEqual(len(got), 4)
            self.assertTrue((comfy_dir / "models" / "loras" / "good.safetensors").is_file())
            self.assertFalse((comfy_dir / "models" / "loras" / "bad.safetensors").exists())


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
            cors_idx = cmd.index("--enable-cors-header")
            self.assertEqual(cmd[cors_idx + 1], "*")
            self.assertIn("--enable-compress-response-body", cmd)
            self.assertEqual(cmd[-1], "--lowvram")

            cmd_explicit_sep = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                extra_args=["--enable-cors-header", "https://example.com"],
            )
            self.assertEqual(cmd_explicit_sep.count("--enable-cors-header"), 1)
            self.assertNotIn("*", cmd_explicit_sep)
            self.assertIn("--enable-compress-response-body", cmd_explicit_sep)

            cmd_explicit_eq = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                extra_args=["--enable-cors-header=https://example.com"],
            )
            self.assertNotIn("--enable-cors-header", cmd_explicit_eq)
            self.assertIn("--enable-cors-header=https://example.com", cmd_explicit_eq)
            self.assertNotIn("*", cmd_explicit_eq)

            cmd_explicit_compress = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                extra_args=["--enable-compress-response-body"],
            )
            self.assertEqual(cmd_explicit_compress.count("--enable-compress-response-body"), 1)

            cmd_no_compress = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                enable_compression=False,
            )
            self.assertNotIn("--enable-compress-response-body", cmd_no_compress)

            with self.assertRaises(ValueError):
                build_comfyui_command("/content/ComfyUI", shm_root=shm_root, port=0)

    def test_cleanup_expired_files_removes_only_expired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "comfy-out"
            nested = out_dir / "sub"
            nested.mkdir(parents=True)

            old_file = nested / "old.png"
            fresh_file = out_dir / "fresh.png"
            log_file = out_dir / "comfyui.log"
            old_file.write_bytes(b"old")
            fresh_file.write_bytes(b"fresh")
            log_file.write_bytes(b"server log")

            now = 1_700_000_000.0
            os.utime(old_file, (now - 600, now - 600))
            os.utime(fresh_file, (now - 30, now - 30))
            os.utime(log_file, (now - 900, now - 900))

            removed = cleanup_expired_files(
                [out_dir],
                max_age_sec=300.0,
                now=now,
                exclude_names=("comfyui.log",),
            )
            self.assertEqual(removed, [old_file])
            self.assertFalse(old_file.exists())
            self.assertTrue(fresh_file.exists())
            self.assertTrue(log_file.exists())

            removed_unfiltered = cleanup_expired_files([out_dir], max_age_sec=300.0, now=now)
            self.assertEqual(removed_unfiltered, [log_file])
            self.assertFalse(log_file.exists())
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

            reaper_stop = threading.Event()
            with (
                mock.patch(
                    "comfyui.bootstrap.start_ramdisk_reaper",
                    return_value=(reaper_stop, mock.MagicMock()),
                ) as reaper_mock,
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

                self.assertEqual(
                    reaper_mock.call_args.kwargs.get("exclude_names"),
                    ("comfyui.log", "cloudflared.log"),
                )

                self.assertEqual(result.public_url, "https://zero-disk-test.trycloudflare.com")
                self.assertEqual(result.local_url, "http://127.0.0.1:8188")
                self.assertEqual(result.output_dir, shm_root / "comfy-out")
                self.assertEqual(result.temp_dir, shm_root / "comfy-tmp")
                expected_log = shm_root / "comfy-tmp" / "comfyui.log"
                expected_tunnel_log = shm_root / "comfy-tmp" / "cloudflared.log"
                self.assertEqual(result.log_file, expected_log)
                self.assertEqual(result.tunnel_log_file, expected_tunnel_log)
                self.assertTrue(expected_log.is_file())

                launched_cmd = popen_mock.call_args[0][0]
                self.assertIn(str(shm_root / "comfy-out"), launched_cmd)
                self.assertIn(str(shm_root / "comfy-tmp"), launched_cmd)
                self.assertIn("--enable-cors-header", launched_cmd)
                self.assertIn("--enable-compress-response-body", launched_cmd)
                popen_kwargs = popen_mock.call_args.kwargs
                log_fp = popen_kwargs["stdout"]
                self.assertEqual(Path(log_fp.name), expected_log)
                self.assertTrue(log_fp.closed)
                self.assertEqual(popen_kwargs["stderr"], subprocess.STDOUT)
                tunnel_mock.assert_called_once_with(
                    8188,
                    timeout_sec=30.0,
                    cloudflared_bin=None,
                    protocol="http2",
                    log_file=expected_tunnel_log,
                )

                result.stop()
                self.assertTrue(result.reaper_stop.is_set())
                fake_tunnel.stop.assert_called_once()
                fake_proc.terminate.assert_called_once()

    def test_launch_comfyui_zero_disk_closes_log_fd_when_popen_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comfy_dir = root / "ComfyUI"
            comfy_dir.mkdir()
            shm_root = root / "shm"
            captured_fps: list[object] = []
            reaper_stop = threading.Event()

            def fail_popen(*args, **kwargs):
                captured_fps.append(kwargs.get("stdout"))
                raise OSError("popen failed")

            with (
                mock.patch(
                    "comfyui.bootstrap.start_ramdisk_reaper",
                    return_value=(reaper_stop, mock.MagicMock()),
                ),
                mock.patch("comfyui.bootstrap.subprocess.Popen", side_effect=fail_popen),
                self.assertRaises(OSError),
            ):
                launch_comfyui_zero_disk(
                    comfy_dir,
                    shm_root=shm_root,
                    port=8188,
                    reaper_interval_sec=60.0,
                )

            self.assertEqual(len(captured_fps), 1)
            self.assertTrue(captured_fps[0].closed)
            self.assertTrue(reaper_stop.is_set())

    def test_launch_comfyui_zero_disk_rejects_invalid_protocol_before_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch("comfyui.bootstrap.start_ramdisk_reaper") as reaper_mock,
                mock.patch("comfyui.bootstrap.subprocess.Popen") as popen_mock,
                self.assertRaises(ValueError),
            ):
                launch_comfyui_zero_disk(
                    root / "ComfyUI",
                    shm_root=root / "shm",
                    tunnel_protocol="invalid",
                )
            reaper_mock.assert_not_called()
            popen_mock.assert_not_called()


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
        self.assertIn('get_secret_status("HF_TOKEN")', step1_source)
        self.assertIn("Notebook access", step1_source)

        step3_source = "".join(cells["step-3-download-models"].get("source", []))
        self.assertIn('get_secret_status("HF_TOKEN")', step3_source)
        self.assertIn("hf download", step3_source)
        self.assertIn("429", step3_source)

        step4_source = "".join(cells["step-4-launch-zero-disk"].get("source", []))
        self.assertIn('TUNNEL_PROTOCOL = "http2"  #@param ["http2", "auto", "quic"]', step4_source)
        self.assertIn("tunnel_protocol=TUNNEL_PROTOCOL", step4_source)
        self.assertIn("ComfyUI Log:", step4_source)
        self.assertIn("Tunnel Log:", step4_source)
        self.assertIn("Tunnel Protocol:", step4_source)
        self.assertIn("session.log_file", step4_source)
        self.assertIn("session.tunnel_log_file", step4_source)
        self.assertIn("--enable-cors-header *", step4_source)
        self.assertIn("--enable-compress-response-body", step4_source)
        self.assertIn("403", step4_source)

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
