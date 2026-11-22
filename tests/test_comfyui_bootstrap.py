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
    MINIMAX_H3_REF2VA_8STEP_MODELS,
    ModelEntry,
    build_comfyui_command,
    cleanup_expired_files,
    download_preset_models,
    ensure_comfyui_package_ready,
    install_bundled_workflows,
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

    def test_resolve_minimax_h3_presets_and_no_lora_cross_mixing(self) -> None:
        ref2va_8 = resolve_model_preset("minimax_h3_ref2va_8step")
        ref2va_fp8 = resolve_model_preset("minimax_h3_ref2va_8step_fp8")
        ref2va_full = resolve_model_preset("minimax_h3_ref2va_full")
        fl2va_8 = resolve_model_preset("minimax_h3_fl2va_8step")

        self.assertEqual(len(ref2va_8), 5)
        self.assertEqual(len(ref2va_fp8), 5)
        self.assertEqual(len(ref2va_full), 6)
        self.assertEqual(len(fl2va_8), 5)

        for entries in (ref2va_8, ref2va_fp8, ref2va_full):
            names_and_files = " ".join(e.filename for e in entries)
            self.assertIn("ref2va", names_and_files)
            self.assertIn("minimax_h3_ref2v_turbo_8step", names_and_files)
            self.assertNotIn("fl2va", names_and_files)
            self.assertNotIn("fl2v", names_and_files)

        fl2v_files = " ".join(e.filename for e in fl2va_8)
        self.assertIn("fl2va", fl2v_files)
        self.assertIn("minimax_h3_fl2v_turbo_8step", fl2v_files)
        self.assertNotIn("ref2va", fl2v_files)
        self.assertNotIn("ref2v", fl2v_files)

        lora_8_ref = ref2va_8[-1]
        self.assertEqual(lora_8_ref.repo_id, "lightx2v/Minimax-h3-Turbo")
        self.assertEqual(
            lora_8_ref.dest_rel_path(),
            Path("loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"),
        )

    def test_load_models_from_minimax_h3_example_yaml(self) -> None:
        yaml_path = Path(__file__).resolve().parents[1] / "comfyui" / "models.minimax_h3.example.yaml"
        entries = load_models_from_yaml(yaml_path)
        self.assertEqual(entries, list(MINIMAX_H3_REF2VA_8STEP_MODELS))

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

    def test_download_preset_models_minimax_h3_batches_base_and_relocates_turbo_lora(self) -> None:
        batch_calls: list[tuple[str, list[str]]] = []
        file_calls: list[tuple[str, str, str]] = []

        def fake_repo_download(repo_id: str, filenames, *, local_dir, token=None, **kwargs):
            file_list = list(filenames)
            batch_calls.append((repo_id, file_list))
            paths: list[Path] = []
            for fname in file_list:
                out_path = Path(local_dir) / fname
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b"weights")
                paths.append(out_path)
            return paths

        def fake_file_download(repo_id: str, filename: str, *, local_dir, token=None, **kwargs):
            file_calls.append((repo_id, filename, str(local_dir)))
            out_path = Path(local_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"lora-weights")
            return out_path

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch("comfyui.bootstrap.download_hf_repo_files", side_effect=fake_repo_download),
            mock.patch("comfyui.bootstrap.download_hf_file", side_effect=fake_file_download),
            mock.patch("comfyui.bootstrap.get_secret_status", return_value=(True, "env")),
        ):
            comfy_dir = Path(tmp) / "ComfyUI"
            got = download_preset_models(
                comfy_dir,
                preset="minimax_h3_ref2va_8step",
                token="placeholder-token",
                pause_sec=0.0,
            )

            self.assertEqual(len(got), 5)
            self.assertEqual(len(batch_calls), 1)
            self.assertEqual(batch_calls[0][0], "Comfy-Org/MiniMax-H3")
            self.assertEqual(len(batch_calls[0][1]), 4)
            self.assertEqual(
                file_calls,
                [
                    (
                        "lightx2v/Minimax-h3-Turbo",
                        "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
                        str(comfy_dir / "models" / ".hf-staging"),
                    )
                ],
            )
            self.assertTrue(
                (
                    comfy_dir
                    / "models"
                    / "loras"
                    / "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
                ).is_file()
            )


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

            cmd_reserve = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                reserve_vram_gb=2.0,
            )
            rv_idx = cmd_reserve.index("--reserve-vram")
            self.assertEqual(cmd_reserve[rv_idx + 1], "2")

            cmd_reserve_zero = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                reserve_vram_gb=0.0,
            )
            self.assertNotIn("--reserve-vram", cmd_reserve_zero)

            cmd_reserve_dedup = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                reserve_vram_gb=2.0,
                extra_args=["--reserve-vram", "3.0"],
            )
            self.assertEqual(cmd_reserve_dedup.count("--reserve-vram"), 1)
            self.assertEqual(cmd_reserve_dedup[cmd_reserve_dedup.index("--reserve-vram") + 1], "3.0")

            cmd_reserve_eq = build_comfyui_command(
                "/content/ComfyUI",
                shm_root=shm_root,
                reserve_vram_gb=2.0,
                extra_args=["--reserve-vram=3.0"],
            )
            self.assertNotIn("--reserve-vram", cmd_reserve_eq)
            self.assertIn("--reserve-vram=3.0", cmd_reserve_eq)

            with self.assertRaises(ValueError):
                build_comfyui_command("/content/ComfyUI", shm_root=shm_root, reserve_vram_gb=-0.5)

    def test_install_bundled_workflows_copies_and_respects_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            comfy_dir = Path(tmp) / "ComfyUI"
            installed = install_bundled_workflows(comfy_dir)
            dest = comfy_dir / "user" / "default" / "workflows" / "minimax_h3_ref2va_8step_kj.json"
            self.assertIn(dest, installed)
            self.assertTrue(dest.is_file())

            dest.write_text('{"custom": true}', encoding="utf-8")
            skipped = install_bundled_workflows(comfy_dir, overwrite=False)
            self.assertIn(dest, skipped)
            self.assertEqual(dest.read_text(encoding="utf-8"), '{"custom": true}')

            overwritten = install_bundled_workflows(comfy_dir, overwrite=True)
            self.assertIn(dest, overwritten)
            self.assertNotEqual(dest.read_text(encoding="utf-8"), '{"custom": true}')

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
                    reserve_vram_gb=2.0,
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
                self.assertIn("--reserve-vram", launched_cmd)
                self.assertEqual(launched_cmd[launched_cmd.index("--reserve-vram") + 1], "2")
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
        self.assertIn('FALLBACK_BRANCH = "main"', step1_source)
        self.assertIn('BRANCH = "main"  #@param {type:"string"}', step1_source)
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

    def test_install_sh_uses_lf_and_supports_kjnodes_cu130_and_workflows(self) -> None:
        install_path = Path(__file__).resolve().parents[1] / "comfyui" / "install.sh"
        raw = install_path.read_bytes()
        self.assertNotIn(b"\r\n", raw)
        text = raw.decode("utf-8")
        for needle in (
            "INSTALL_KJNODES",
            "UPGRADE_TORCH_CU130",
            "INSTALL_WORKFLOWS",
            "ComfyUI-KJNodes",
            "ffmpeg",
        ):
            self.assertIn(needle, text)

    def test_minimax_h3_workflow_graph_integrity_and_models(self) -> None:
        wf_path = (
            Path(__file__).resolve().parents[1]
            / "comfyui"
            / "workflows"
            / "minimax_h3_ref2va_8step_kj.json"
        )
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        nodes = {n["id"]: n for n in wf["nodes"]}

        # Validate every link resolves cleanly to matching node slots and types
        seen_links: set[int] = set()
        for link in wf["links"]:
            link_id, src_id, src_slot, dst_id, dst_slot, link_type = link
            self.assertNotIn(link_id, seen_links)
            seen_links.add(link_id)
            src_node = nodes[src_id]
            dst_node = nodes[dst_id]
            src_out = src_node["outputs"][src_slot]
            dst_in = dst_node["inputs"][dst_slot]
            self.assertEqual(dst_in["link"], link_id)
            self.assertIn(link_id, src_out["links"])
            self.assertEqual(src_out["type"], link_type)
            self.assertIn(link_type, str(dst_in["type"]).split(","))

        # Verify model patch chain: UNETLoader -> LoraLoaderModelOnly -> Switch -> SigmaShift -> LowVRAM -> ChunkFF -> BasicScheduler & BasicGuider
        self.assertEqual(nodes[147]["type"], "MiniMaxH3SigmaShift")
        self.assertEqual(nodes[147]["widgets_values"], [12.0, 3.0])
        self.assertEqual(nodes[148]["type"], "MiniMaxLowVRAMAttention")
        self.assertEqual(nodes[148]["widgets_values"], [16])
        self.assertEqual(nodes[149]["type"], "MiniMaxChunkFeedForward")
        self.assertEqual(nodes[149]["widgets_values"], [8, 4096])

        link_by_id = {lk[0]: lk for lk in wf["links"]}
        self.assertEqual(link_by_id[nodes[145]["inputs"][0]["link"]][1], 127)
        self.assertEqual(link_by_id[nodes[141]["inputs"][0]["link"]][1], 127)
        self.assertEqual(link_by_id[nodes[141]["inputs"][1]["link"]][1], 145)
        self.assertEqual(link_by_id[nodes[147]["inputs"][0]["link"]][1], 141)
        self.assertEqual(link_by_id[nodes[148]["inputs"][0]["link"]][1], 147)
        self.assertEqual(link_by_id[nodes[149]["inputs"][0]["link"]][1], 148)
        self.assertEqual(link_by_id[nodes[124]["inputs"][0]["link"]][1], 149)
        self.assertEqual(link_by_id[nodes[126]["inputs"][0]["link"]][1], 149)

        # Verify all 5 model filenames match MINIMAX_H3_REF2VA_8STEP_MODELS
        wf_model_names = {
            nodes[127]["widgets_values"][0],
            nodes[128]["widgets_values"][0],
            nodes[119]["widgets_values"][0],
            nodes[120]["widgets_values"][0],
            nodes[145]["widgets_values"][0],
        }
        preset_model_names = {Path(e.filename).name for e in MINIMAX_H3_REF2VA_8STEP_MODELS}
        self.assertEqual(wf_model_names, preset_model_names)

    def test_minimax_h3_notebook_structure(self) -> None:
        nb_path = Path(__file__).resolve().parents[1] / "comfyui" / "minimax_h3.ipynb"
        notebook = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = {cell.get("id"): cell for cell in notebook.get("cells", [])}
        for cid in (
            "minimax-h3-header",
            "step-1-bootstrap-repo",
            "step-2-install-minimax-h3-env",
            "step-3-download-models",
            "step-4-launch-comfyui",
            "step-5-utils-and-stop",
        ):
            self.assertIn(cid, cells)

        step1 = "".join(cells["step-1-bootstrap-repo"]["source"])
        self.assertIn('BRANCH = "main"  #@param {type:"string"}', step1)
        self.assertIn("importlib.invalidate_caches()", step1)
        self.assertIn("sys.modules", step1)
        self.assertIn("minimax_h3_ref2va_8step_kj.json", step1)

        step2 = "".join(cells["step-2-install-minimax-h3-env"]["source"])
        self.assertIn("INSTALL_KJNODES", step2)
        self.assertIn("UPGRADE_TORCH_CU130", step2)

        step3 = "".join(cells["step-3-download-models"]["source"])
        self.assertIn('PRESET = "minimax_h3_ref2va_8step"', step3)
        self.assertIn("install_bundled_workflows()", step3)

        step4 = "".join(cells["step-4-launch-comfyui"]["source"])
        self.assertIn("RESERVE_VRAM_GB = 2.0", step4)
        self.assertIn("MAX_AGE_SEC = 1800", step4)
        self.assertIn('TUNNEL_PROTOCOL = "http2"', step4)
        self.assertIn("reserve_vram_gb=", step4)


if __name__ == "__main__":
    unittest.main()
