"""Unit tests for the Colab Fizgig MOSS-Image LoRA training workbench."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LORA_DIR = _REPO_ROOT / "lora-training"
if str(_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(_LORA_DIR))

import fizgig_moss  # noqa: E402
from fizgig_moss import (  # noqa: E402
    DEFAULT_LOWNOISE_PCT,
    MIN_WEB_PASSWORD_LEN,
    MOSS_MODEL_PRESETS,
    MossTrainConfig,
    build_moss_cache_commands,
    build_moss_train_command,
    download_moss_models,
    ensure_fizgig_moss_ready,
    launch_fizgig_web,
    resolve_moss_preset,
    wire_fizgig_prefs,
    write_dataset_toml,
)


class InstallScriptTests(unittest.TestCase):
    def test_install_sh_uses_lf_and_isolates_uv_environment(self) -> None:
        install_path = _LORA_DIR / "install.sh"
        raw = install_path.read_bytes()
        self.assertNotIn(b"\r\n", raw)

        text = raw.decode("utf-8")
        self.assertIn("unset UV_SYSTEM_PYTHON", text)
        self.assertIn('export VIRTUAL_ENV="$FIZGIG_DIR/venv"', text)
        self.assertIn('export UV_PYTHON="$FIZGIG_DIR/venv/bin/python"', text)
        self.assertIn("export DISABLE_CUDA=1", text)


class PresetAndPrefsTests(unittest.TestCase):
    def test_resolve_moss_preset_variants(self) -> None:
        std = resolve_moss_preset("standard")
        int8_te = resolve_moss_preset(" Int8_Te ")
        fast = resolve_moss_preset("with_fast_preview")

        self.assertEqual(len(std), 4)
        self.assertEqual({m.pref_key for m in std}, {
            "moss_image_dit",
            "moss_image_vae",
            "moss_image_text_encoder",
            "moss_image_sft_dit",
        })
        self.assertTrue(any("qwen3.5_9b_int8_convrot.safetensors" in m.filename for m in int8_te))
        self.assertEqual(len(fast), 6)
        self.assertEqual(set(MOSS_MODEL_PRESETS), {"standard", "int8_te", "with_fast_preview"})

        with self.assertRaises(ValueError):
            resolve_moss_preset("unknown-preset")

    def test_wire_fizgig_prefs_merges_existing_json_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fizgig_dir = Path(tmp) / "Fizgig"
            user_data = fizgig_dir / "user_data"
            user_data.mkdir(parents=True)
            prefs_file = user_data / "prefs.json"
            prefs_file.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

            dit_path = user_data / "models" / "moss_image_base_bf16.safetensors"
            out_path = wire_fizgig_prefs({"moss_image_dit": dit_path}, fizgig_dir=fizgig_dir)

            self.assertEqual(out_path, prefs_file)
            loaded = json.loads(prefs_file.read_text(encoding="utf-8"))
            self.assertEqual(loaded["theme"], "dark")
            self.assertEqual(loaded["moss_image_dit"], str(dit_path.resolve()))

    def test_download_moss_models_skips_existing_batches_by_repo_and_wires_prefs(self) -> None:
        batch_calls: list[tuple[str, list[str], str, str | None]] = []

        def fake_repo_download(repo_id: str, filenames, *, local_dir, token=None, **kwargs):
            file_list = list(filenames)
            batch_calls.append((repo_id, file_list, str(local_dir), token))
            out_paths: list[Path] = []
            for fname in file_list:
                p = Path(local_dir) / fname
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"weights")
                out_paths.append(p)
            return out_paths

        clean_env = dict(os.environ)
        clean_env.pop("HF_XET_HIGH_PERFORMANCE", None)
        buf = StringIO()

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, clean_env, clear=True),
            mock.patch("fizgig_moss.download_hf_repo_files", side_effect=fake_repo_download),
            mock.patch("fizgig_moss.get_secret_status", return_value=(False, "not-set")),
            mock.patch("fizgig_moss.get_secret", return_value=None),
            mock.patch("fizgig_moss.time.sleep"),
            redirect_stdout(buf),
        ):
            fizgig_dir = Path(tmp) / "Fizgig"
            existing_vae = fizgig_dir / "user_data" / "models" / "flux2-vae_small.safetensors"
            existing_vae.parent.mkdir(parents=True, exist_ok=True)
            existing_vae.write_bytes(b"cached-vae")

            resolved = download_moss_models("standard", fizgig_dir=fizgig_dir, pause_sec=0.0)

            self.assertEqual(len(resolved), 4)
            self.assertEqual(resolved["moss_image_vae"], existing_vae)
            self.assertTrue(resolved["moss_image_dit"].is_file())
            self.assertTrue(resolved["moss_image_text_encoder"].is_file())
            self.assertTrue(resolved["moss_image_sft_dit"].is_file())

            # VAE was skipped; remaining 3 files split across 2 repos.
            self.assertEqual(len(batch_calls), 2)
            self.assertEqual(batch_calls[0][0], "Comfy-Org/MOSS-Image")
            self.assertEqual(len(batch_calls[0][1]), 2)
            self.assertEqual(batch_calls[1][0], "Comfy-Org/Qwen3.5")
            self.assertIsNone(os.environ.get("HF_XET_HIGH_PERFORMANCE"))

            prefs = json.loads((fizgig_dir / "user_data" / "prefs.json").read_text(encoding="utf-8"))
            self.assertEqual(prefs["moss_image_dit"], str(resolved["moss_image_dit"].resolve()))


class CommandBuilderTests(unittest.TestCase):
    def test_write_dataset_toml_supports_multi_caption_and_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "ds" / "dataset.toml"
            out = write_dataset_toml(
                toml_path,
                image_directory="/content/datasets/demo/train",
                cache_directory="/content/Fizgig/user_data/cache/demo",
                multi_caption=True,
            )
            self.assertEqual(out, toml_path)
            content = toml_path.read_text(encoding="utf-8")
            self.assertIn("resolution = [1024, 1024]", content)
            self.assertIn("enable_bucket = true", content)
            self.assertIn("bucket_no_upscale = true", content)
            self.assertIn("multi_caption = true", content)
            self.assertIn('image_directory = "/content/datasets/demo/train"', content)
            self.assertIn('cache_directory = "/content/Fizgig/user_data/cache/demo"', content)

    def test_build_moss_cache_commands(self) -> None:
        lat_cmd, txt_cmd = build_moss_cache_commands(
            "/content/ds.toml",
            vae_path="/content/vae.safetensors",
            text_encoder_path="/content/te.safetensors",
            fizgig_dir="/content/Fizgig",
            skip_existing=True,
        )
        self.assertEqual(Path(lat_cmd[0]).as_posix(), "/content/Fizgig/venv/bin/python")
        self.assertEqual(Path(lat_cmd[1]).as_posix(), "/content/Fizgig/src/fizgig/families/cache.py")
        self.assertIn("latents", lat_cmd)
        self.assertIn("--skip_existing", lat_cmd)
        self.assertIn("text", txt_cmd)
        self.assertIn("/content/te.safetensors", txt_cmd)

    def test_build_moss_train_command_dop_adamw_and_muon_baseline(self) -> None:
        dop_cfg = MossTrainConfig(
            dataset_config="/content/ds.toml",
            dit_path="/content/base.safetensors",
            output_dir="/content/out",
            output_name="char_v1",
            trigger_word="zwxem",
            recipe="dop_adamw",
            network_type="lokr",
            lokr_factor=8,
            dop_class="woman",
            multi_caption=True,
            preview_checkpoint="/content/sft.safetensors",
            sample_prompts="/content/prompts.txt",
            sample_every_n_epochs=5,
            sample_at_first=True,
        )
        dop_cmd = build_moss_train_command(dop_cfg, fizgig_dir="/content/Fizgig")
        self.assertIn("--family", dop_cmd)
        self.assertIn("moss_image", dop_cmd)
        self.assertIn("--optimizer_type", dop_cmd)
        self.assertIn("adamw", dop_cmd)
        self.assertIn("--masked_training", dop_cmd)
        self.assertIn("--mask_prior", dop_cmd)
        self.assertIn("1.0", dop_cmd)
        self.assertIn("--dop", dop_cmd)
        self.assertIn("zwxem", dop_cmd)
        self.assertIn("woman", dop_cmd)
        self.assertIn("--network_type", dop_cmd)
        self.assertIn("lokr", dop_cmd)
        self.assertIn("--lokr_factor", dop_cmd)
        self.assertIn("--multi_caption", dop_cmd)
        self.assertIn(f"lownoise_pct={DEFAULT_LOWNOISE_PCT}", dop_cmd)
        self.assertIn("--sample_at_first", dop_cmd)

        muon_cfg = MossTrainConfig(
            dataset_config="/content/ds.toml",
            dit_path="/content/base.safetensors",
            output_dir="/content/out",
            output_name="char_base",
            trigger_word="zwxem",
            recipe="muon_baseline",
            network_type="lora",
        )
        muon_cmd = build_moss_train_command(muon_cfg, fizgig_dir="/content/Fizgig")
        self.assertIn("muon", muon_cmd)
        self.assertIn("--optimizer_args", muon_cmd)
        self.assertIn("--network_dim", muon_cmd)
        self.assertNotIn("--dop", muon_cmd)
        self.assertNotIn("--masked_training", muon_cmd)

    def test_build_moss_train_command_rejects_invalid_inputs(self) -> None:
        base = dict(
            dataset_config="/content/ds.toml",
            dit_path="/content/base.safetensors",
            output_dir="/content/out",
            output_name="char_v1",
            trigger_word="zwxem",
        )
        with self.assertRaises(ValueError):
            build_moss_train_command(MossTrainConfig(**{**base, "trigger_word": "   "}))
        with self.assertRaises(ValueError):
            build_moss_train_command(MossTrainConfig(**{**base, "recipe": "bad"}))
        with self.assertRaises(ValueError):
            build_moss_train_command(MossTrainConfig(**{**base, "network_type": "bad"}))
        with self.assertRaises(ValueError):
            build_moss_train_command(MossTrainConfig(**{**base, "dop_class": ""}))


class WebLaunchTests(unittest.TestCase):
    def test_launch_fizgig_web_validates_password_closes_log_and_stops_cleanly(self) -> None:
        with self.assertRaises(ValueError):
            launch_fizgig_web(password="short", start_tunnel=False)
        self.assertEqual(MIN_WEB_PASSWORD_LEN, 8)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fizgig_dir = root / "Fizgig"
            fizgig_dir.mkdir()
            shm_root = root / "shm"

            fake_proc = mock.MagicMock()
            fake_proc.poll.return_value = None
            fake_proc.stdout = None
            fake_tunnel = mock.MagicMock()
            fake_tunnel.url = "https://fizgig-demo.trycloudflare.com"

            with (
                mock.patch("fizgig_moss.get_secret", return_value=None),
                mock.patch("fizgig_moss.subprocess.Popen", return_value=fake_proc) as popen_mock,
                mock.patch("fizgig_moss._wait_for_web_server", return_value="http://127.0.0.1:8081"),
                mock.patch("fizgig_moss.start_cloudflared_tunnel", return_value=fake_tunnel) as tunnel_mock,
            ):
                session = launch_fizgig_web(
                    fizgig_dir=fizgig_dir,
                    shm_root=shm_root,
                    port=8081,
                )

                self.assertEqual(session.public_url, "https://fizgig-demo.trycloudflare.com")
                self.assertGreaterEqual(len(session.password), MIN_WEB_PASSWORD_LEN)
                popen_kwargs = popen_mock.call_args.kwargs
                log_fp = popen_kwargs["stdout"]
                self.assertTrue(log_fp.closed)
                self.assertEqual(popen_kwargs["stderr"], subprocess.STDOUT)
                self.assertEqual(popen_kwargs["env"]["FIZGIG_PASSWORD"], session.password)
                # Password should not be passed on CLI argv.
                self.assertNotIn("--password", popen_mock.call_args[0][0])
                tunnel_mock.assert_called_once()

                session.stop()
                fake_tunnel.stop.assert_called_once()
                fake_proc.terminate.assert_called_once()

    def test_launch_fizgig_web_cleans_up_if_tunnel_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fizgig_dir = root / "Fizgig"
            fizgig_dir.mkdir()
            fake_proc = mock.MagicMock()
            fake_proc.poll.return_value = None
            fake_proc.stdout = None

            with (
                mock.patch("fizgig_moss.subprocess.Popen", return_value=fake_proc),
                mock.patch("fizgig_moss._wait_for_web_server", return_value="http://127.0.0.1:8081"),
                mock.patch("fizgig_moss.start_cloudflared_tunnel", side_effect=TimeoutError("tunnel fail")),
                self.assertRaises(TimeoutError),
            ):
                launch_fizgig_web(
                    fizgig_dir=fizgig_dir,
                    shm_root=root / "shm",
                    password="StrongPassword123",
                )

            fake_proc.terminate.assert_called_once()


class NotebookAndBootstrapTests(unittest.TestCase):
    def test_fizgig_moss_notebook_structure(self) -> None:
        nb_path = _LORA_DIR / "fizgig_moss.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        cells = {cell.get("id"): cell for cell in nb.get("cells", [])}

        self.assertIn("step-1-bootstrap-repo", cells)
        self.assertIn("step-2-install-fizgig", cells)
        self.assertIn("step-3-download-moss-models", cells)
        self.assertIn("step-4-launch-or-dry-run", cells)

        step1 = "".join(cells["step-1-bootstrap-repo"]["source"])
        self.assertIn('BRANCH = "main"  #@param {type:"string"}', step1)
        self.assertIn('FALLBACK_BRANCH = "main"', step1)
        self.assertIn("fizgig_moss.py", step1)
        self.assertIn('get_secret_status("HF_TOKEN")', step1)

        step2 = "".join(cells["step-2-install-fizgig"]["source"])
        self.assertIn("lora-training/install.sh", step2)

        step3 = "".join(cells["step-3-download-moss-models"]["source"])
        self.assertIn("download_moss_models", step3)

        step4 = "".join(cells["step-4-launch-or-dry-run"]["source"])
        self.assertIn("launch_fizgig_web", step4)
        self.assertIn("build_moss_train_command", step4)

    def test_ensure_fizgig_moss_ready_recovers_sys_path(self) -> None:
        original_sys_path = list(sys.path)
        try:
            resolved = ensure_fizgig_moss_ready(_REPO_ROOT)
            self.assertEqual(resolved, (_LORA_DIR / "fizgig_moss.py").resolve())
            self.assertEqual(sys.path[0], str(_LORA_DIR))
            self.assertEqual(sys.path[1], str(_REPO_ROOT))
            importlib.invalidate_caches()
        finally:
            sys.path[:] = original_sys_path


if __name__ == "__main__":
    unittest.main()
