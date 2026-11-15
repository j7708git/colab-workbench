# ComfyUI 專區（零磁碟防審核 Colab 部署）

本目錄提供在 Google Colab VM 一鍵啟動 ComfyUI、下載預設公開模型並以記憶體磁碟（`/dev/shm`）零落盤輸出與 Cloudflare Quick Tunnel 暴露 API 的自動化套件。

## 檔案說明

- `comfyui.ipynb`：薄入口 Notebook（支援 `Open in Colab` 一鍵開啟；第 1 格提供 `BRANCH` 表單參數，預設對齊 `feat/comfyui-zero-disk-bootstrap`，若誤填尚未合併的 `main` 會自動 fallback 並清理 Python Namespace Package 快取，免重啟 Colab Runtime）
- `install.sh`：初始化 `/dev/shm/comfy-out` 與 `/dev/shm/comfy-tmp`、安裝／更新 `/content/ComfyUI`（≥ v0.37.0）及 `huggingface_hub` + `hf_xet`
- `bootstrap.py`：預設模型清單下載器、RAM Disk 零磁碟指令組裝、過期暫存定期清理（reaper）、`ensure_comfyui_package_ready()` 與 `cloudflared` 隧道整合
- `models.example.yaml`：預設與選配擴充模型清單範例（僅記錄 repo 與檔名，實際權重不進版控）

## Colab 匯入防呆機制（避免 `unknown location` 錯誤）

若在 Google Colab 先拉取到尚未包含 `comfyui/__init__.py` 的分支，Python 會將 `/content/colab-workbench/comfyui` 快取為空的 PEP 420 Namespace Package（`__file__` 為 `None`），導致後續切換分支後仍出現 `ImportError: cannot import name 'download_preset_models' from 'comfyui' (unknown location)`。
`comfyui.ipynb` 已在第 1、3、4 格加入以下自動修復機制，無須手動重啟 Colab Runtime：

1. **明確遠端 Refspec 同步與自動 Fallback**：第 1 格以 `git fetch origin +refs/heads/<BRANCH>:refs/remotes/origin/<BRANCH>` 與 `git checkout -B <BRANCH> origin/<BRANCH>` 強制對齊遠端；若目標分支缺少 `comfyui/__init__.py`，自動切換回 `feat/comfyui-zero-disk-bootstrap`。
2. **免重啟熱修復快取**：於第 1 格及第 3、4 格開頭主動將 `/content/colab-workbench` 置於 `sys.path[0]`、清除 `sys.modules` 內殘留的 Namespace Package 快取並呼叫 `importlib.invalidate_caches()`。

## 內建模型預設集（Preset）

執行 `comfyui.ipynb` 第 3 格或呼叫 `download_preset_models(preset=...)` 時，可選擇以下公開模型套件（預設為 `both`）：

| Preset | 包含模型 | 來源倉庫 |
| --- | --- | --- |
| `qwen21_turbo_int8` | 主模型 `qwen_image_2.1_turbo_int8_convrot.safetensors`、Text Encoder `qwen3vl_8b_int8_convrot.safetensors`、VAE `qwen_image_2.1_vae_bf16.safetensors` | `Comfy-Org/Qwen-Image-2.1` |
| `moss_sft_int8` | 主模型 `moss_image_sft_int8_convrot.safetensors`、Text Encoder `qwen3.5_9b_int8_convrot.safetensors`、VAE `flux2-vae.safetensors`、4 步蒸餾 LoRA `moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors` | `Comfy-Org/MOSS-Image`、`Kijai/MOSS-image_experimental` |
| `both`（預設） | 同時下載上述兩組全部 7 個檔案（合計約 33.5 GB） | 同上 |

## 零磁碟輸出與前端連線（`dsh-simple-tavern`）

1. `launch_comfyui_zero_disk()` 會強制帶入 `--output-directory /dev/shm/comfy-out --temp-directory /dev/shm/comfy-tmp --disable-auto-launch`，生圖過程與輸出圖檔僅駐留於 Linux `tmpfs` 記憶體，**不寫入 Colab 虛擬硬碟也不掛載 Google Drive**。
2. 背景執行 `start_ramdisk_reaper()`（預設每 60 秒清理超過 300 秒的暫存檔案），避免長時間使用耗盡 Colab RAM。
3. 啟動完成後會印出 `https://*.trycloudflare.com` 網址，可直接貼入瀏覽器或 `dsh-simple-tavern` 設定頁的 `ComfyUI URL`（相容既有 `/prompt`、`/history` 與 `/view` API，前端無須修改程式碼）。

## 資源與顯存注意事項

- `qwen21_turbo_int8` 與 `moss_sft_int8` 皆為中大型模型（合計約 33.5 GB 磁碟暫存，推理建議 Colab L4 24GB 或 A100 40GB 以上執行階段）；若僅在 T4 等較小資源測試，可在第 3 格將 `PRESET` 改選單一模型套件並在 `EXTRA_ARGS` 視需要加入 `--lowvram`。
