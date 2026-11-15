# ComfyUI 專區（零磁碟防審核 Colab 部署）

本目錄提供在 Google Colab VM 一鍵啟動 ComfyUI、下載預設公開模型並以記憶體磁碟（`/dev/shm`）零落盤輸出與 Cloudflare Quick Tunnel 暴露 API 的自動化套件。

## 檔案說明

- `comfyui.ipynb`：薄入口 Notebook（支援 `Open in Colab` 一鍵開啟；第 1 格提供 `BRANCH` 表單參數，預設 `main`，測試未合併分支時可改填如 `feat/comfyui-zero-disk-bootstrap`）
- `install.sh`：初始化 `/dev/shm/comfy-out` 與 `/dev/shm/comfy-tmp`、安裝／更新 `/content/ComfyUI`（≥ v0.37.0）及 `huggingface_hub` + `hf_xet`
- `bootstrap.py`：預設模型清單下載器、RAM Disk 零磁碟指令組裝、過期暫存定期清理（reaper）與 `cloudflared` 隧道整合
- `models.example.yaml`：預設與選配擴充模型清單範例（僅記錄 repo 與檔名，實際權重不進版控）

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
