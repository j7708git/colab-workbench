# ComfyUI 專區（零磁碟防審核 Colab 部署）

本目錄提供在 Google Colab VM 一鍵啟動 ComfyUI、下載預設公開模型並以記憶體磁碟（`/dev/shm`）零落盤輸出與 Cloudflare Quick Tunnel 暴露 API 的自動化套件。

## 檔案說明

- `comfyui.ipynb`：通用生圖薄入口 Notebook（支援 `Open in Colab` 一鍵開啟；第 1 格提供 `BRANCH` 表單參數，預設為 `main`，若輸入尚未包含 `comfyui/__init__.py` 的測試分支會自動 fallback 回 `main` 並清理 Python Namespace Package 快取，免重啟 Colab Runtime）
- `minimax_h3.ipynb`：專為 **MiniMax H3 參考人像生影片（Ref2VA）** 設計的 5-Cell Colab 啟動器（自動啟用 `ComfyUI-KJNodes` 無損分塊、預載 `minimax_h3_ref2va_8step_kj.json`、預設 `--reserve-vram 2.0` 與 30 分鐘 RAM Disk 輸出保留）
- `install.sh`：初始化 `/dev/shm/comfy-out` 與 `/dev/shm/comfy-tmp`、安裝／更新 `/content/ComfyUI`（≥ v0.38.0）並執行 `pip install --upgrade huggingface_hub hf_xet pyyaml`（確保新版 `hf` CLI 與 429 修復就緒）；支援 `INSTALL_KJNODES=1`（安裝 `ComfyUI-KJNodes`）、`UPGRADE_TORCH_CU130=1`（升級 PyTorch `cu130`）、`INSTALL_WORKFLOWS=1`（預設同步 `comfyui/workflows/*.json` 至 `ComfyUI/user/default/workflows/`）與自動安裝 `ffmpeg`
- `bootstrap.py`：依 `repo_id` 批次呼叫 `hf download` 的模型下載器（含 `HF_TOKEN` 狀態檢查與 Xet 並發門檻）、`install_bundled_workflows()`、RAM Disk 零磁碟指令組裝（預設帶入 `--enable-cors-header *` 與 `--enable-compress-response-body`，並支援 `reserve_vram_gb` → `--reserve-vram`）、`/dev/shm/comfy-tmp/comfyui.log` 與 `cloudflared.log` 記憶體內日誌、過期暫存定期清理（reaper）、`ensure_comfyui_package_ready()` 與 `cloudflared`（預設 HTTP/2＋背景 stdout 排空）隧道整合
- `models.example.yaml`：圖片生圖預設與選配擴充模型清單範例（僅記錄 repo 與檔名，實際權重不進版控）
- `models.minimax_h3.example.yaml`：MiniMax H3 Ref2VA 模型下載清單範例（預設 8 步 Turbo + 可選 4 步／FP8／FL2VA／風格 Embedding）
- `workflows/minimax_h3_ref2va_8step_kj.json`：預載 MiniMax H3 Ref2VA 8 步 Turbo + `ModelSamplingMiniMaxH3 (12, 3)` + KJNodes 無損分塊（`head_chunks=16`、`ffn_chunks=8`）工作流 JSON

## Colab 匯入防呆機制（避免 `unknown location` 錯誤）

若在 Google Colab 先拉取到尚未包含 `comfyui/__init__.py` 的分支，Python 會將 `/content/colab-workbench/comfyui` 快取為空的 PEP 420 Namespace Package（`__file__` 為 `None`），導致後續切換分支後仍出現 `ImportError: cannot import name 'download_preset_models' from 'comfyui' (unknown location)`。
`comfyui.ipynb` 已在第 1、3、4 格加入以下自動修復機制，無須手動重啟 Colab Runtime：

1. **明確遠端 Refspec 同步與自動 Fallback**：第 1 格以 `git fetch origin +refs/heads/<BRANCH>:refs/remotes/origin/<BRANCH>` 與 `git checkout -B <BRANCH> origin/<BRANCH>` 強制對齊遠端；若目標分支缺少 `comfyui/__init__.py`，自動切換回 `main`。
2. **免重啟熱修復快取**：於第 1 格及第 3、4 格開頭主動將 `/content/colab-workbench` 置於 `sys.path[0]`、清除 `sys.modules` 內殘留的 Namespace Package 快取並呼叫 `importlib.invalidate_caches()`。

## 內建模型預設集（Preset）

執行 `comfyui.ipynb` 第 3 格或呼叫 `download_preset_models(preset=...)` 時，可選擇以下公開模型套件（預設為 `both`）：

| Preset | 包含模型 | 來源倉庫 |
| --- | --- | --- |
| `qwen21_turbo_int8` | 主模型 `qwen_image_2.1_turbo_int8_convrot.safetensors`、Text Encoder `qwen3vl_8b_int8_convrot.safetensors`、VAE `qwen_image_2.1_vae_bf16.safetensors` | `Comfy-Org/Qwen-Image-2.1` |
| `moss_sft_int8` | 主模型 `moss_image_sft_int8_convrot.safetensors`、Text Encoder `qwen3.5_9b_int8_convrot.safetensors`、VAE `flux2-vae.safetensors`、4 步蒸餾 LoRA `moss_image_sft_4_steps_lora_v2_600_avg_rank_72_bf16.safetensors` | `Comfy-Org/MOSS-Image`、`Kijai/MOSS-image_experimental` |
| `both`（預設） | 同時下載上述兩組全部 7 個檔案（合計約 33.5 GB） | 同上 |
| `minimax_h3_ref2va_8step` | Ref2VA 主模型 `minimax_h3_ref2va_pruned_int8_convrot.safetensors`、Text Encoder `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`、Video VAE `fp16`、Audio VAE `fp32`、8 步 Ref2V Turbo LoRA `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`（5 檔合計約 41.4 GB） | `Comfy-Org/MiniMax-H3`、`lightx2v/Minimax-h3-Turbo` |
| `minimax_h3_ref2va_8step_fp8` | 同上，主模型改為 `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` 備援版（5 檔合計約 41.4 GB） | 同上 |
| `minimax_h3_ref2va_full` | `minimax_h3_ref2va_8step` 另加 4 步草圖 LoRA `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`（6 檔合計約 43.2 GB） | 同上 |
| `minimax_h3_fl2va_8step` | 首尾幀主模型 `minimax_h3_fl2va_pruned_int8_convrot.safetensors`＋共用編碼器／VAE＋8 步 FL2V LoRA `minimax_h3_fl2v_turbo_8step_v1.2_768p_comfyui_bf16.safetensors`（5 檔合計約 41.5 GB） | 同上 |

## 模型下載、HF_TOKEN 與 HTTP 429 防護機制

1. **依倉庫批次下載（`hf download`）**：`download_preset_models()` 會將相對路徑與來源倉庫一致的缺檔依 `repo_id` 分組，每組僅呼叫一次 `download_hf_repo_files()`（底層執行 `hf download <repo_id> <files...> --local-dir ...`），並於不同倉庫間暫停 `pause_sec`（預設 `1.0` 秒）；若需搬移至不同子目錄（`Path(entry.filename) != entry.dest_rel_path()`），會先下載至持久暫存目錄 `ComfyUI/models/.hf-staging` 保留斷點快取，完成後再 `shutil.move` 到目標路徑。
2. **`HF_TOKEN` 與 Colab Secrets「Notebook access」開關**：Colab 新開啟或複製的 Notebook 預設**關閉**各 Secret 的 `Notebook access` 權限，導致 `userdata.get("HF_TOKEN")` 拋出 `NotebookAccessError`。第 1、3 格與 `download_preset_models()` 會透過 `get_secret_status("HF_TOKEN")` 顯示目前狀態（**絕不印出 Token 明文**），若未開啟會明確提示至 Colab 左側鑰匙圖示 Secrets 打開開關，以免在 Colab 共用 IP 下觸發匿名 HTTP 429 限流。
3. **無 Token 不強制高速 Xet**：僅在有效 `HF_TOKEN` 存在時才設定 `HF_XET_HIGH_PERFORMANCE=1`，避免匿名連線在共用 IP 上以高並發請求 Xet CAS 而立刻被限流。
4. **429 自動重試、Xet 降級與 `.part` 安全落回**：遇到 HTTP 429／限流錯誤時會解析 `Retry-After`（或採指數退避）重試；第 1 次重試起自動降級 `HF_XET_HIGH_PERFORMANCE=0`，第 2 次起設定 `HF_HUB_DISABLE_XET=1`。若 `hf` CLI 與 Python Hub 重試皆耗盡，最後會落回 `/resolve/` HTTPS 斷點續傳（寫入 `<filename>.part` 暫存檔，確認完整成功後才以 `os.replace` 取代目的檔，失敗時絕不在正式路徑留下 0-byte 或殘缺檔案）。

## 零磁碟輸出與前端連線（`dsh-simple-tavern`）

1. `launch_comfyui_zero_disk()` 會強制帶入 `--output-directory /dev/shm/comfy-out --temp-directory /dev/shm/comfy-tmp --disable-auto-launch`，生圖過程與輸出圖檔僅駐留於 Linux `tmpfs` 記憶體，**不寫入 Colab 虛擬硬碟也不掛載 Google Drive**。
2. **預設跨來源放行與回應壓縮（修正 403 與 Tunnel 大型 JSON 傳輸）**：`build_comfyui_command()` 預設自動附加 `--enable-cors-header *`（若 `EXTRA_ARGS` 已傳入 `--enable-cors-header` 或 `--enable-cors-header=...` 則不重複附加）與 `--enable-compress-response-body`（可用 `enable_compression=False` 停用；僅壓縮 JSON／純文字回應，如約 2 MB 以上的 `/object_info`，不影響圖片位元組）。這可避免瀏覽器跨站 403，同時大幅縮減透過 Cloudflare Tunnel 傳送節點資訊與歷史紀錄的體積。
3. **記憶體內伺服器／隧道日誌與背景排空**：`launch_comfyui_zero_disk()` 會將 ComfyUI 輸出導向 `/dev/shm/comfy-tmp/comfyui.log`（紀錄於 `session.log_file`），並由背景執行緒持續排空 `cloudflared` 的 stdout／stderr 寫入 `/dev/shm/comfy-tmp/cloudflared.log`（紀錄於 `session.tunnel_log_file`，可透過 `session.tunnel.tail_logs()` 查看近期輸出），避免長時間運行時因 pipe buffer 塞滿而導致 `cloudflared` 卡死。
4. 背景執行 `start_ramdisk_reaper()`（預設每 60 秒清理超過 300 秒的暫存檔案，並自動略過執行中的 `comfyui.log` 與 `cloudflared.log`），避免長時間使用耗盡 Colab RAM。
5. **預設 HTTP/2 隧道協議**：第 4 格提供 `TUNNEL_PROTOCOL`（預設 `"http2"`，可選 `"auto"` 或 `"quic"`），避免 Colab 環境下 UDP/QUIC 遭限流或抖動。啟動完成後會印出 `https://*.trycloudflare.com` 網址，可直接從 Colab 點擊開啟或貼入 `dsh-simple-tavern` 設定頁的 `ComfyUI URL`（相容既有 `/prompt`、`/history` 與 `/view` API，前端無須修改程式碼）。

## 資源與顯存注意事項

- `qwen21_turbo_int8` 與 `moss_sft_int8` 皆為中大型模型（合計約 33.5 GB 磁碟暫存，推理建議 Colab L4 24GB 或 A100 40GB 以上執行階段）；若僅在 T4 等較小資源測試，可在第 3 格將 `PRESET` 改選單一模型套件並在 `EXTRA_ARGS` 視需要加入 `--lowvram`。

## MiniMax H3 參考人像生影片實戰注意事項

- **Colab 執行階段建議**：`minimax_h3.ipynb` 強烈建議使用 **L4（24GB）** 或 **A100（40GB / 80GB）並開啟 High-RAM**；請避免使用 T4（16GB VRAM 與標準系統記憶體難以承載約 41.4 GB 權重與長序列計算）。若遇到 `int8_convrot` 在部分驅動環境推論偏慢，可改選 `minimax_h3_ref2va_8step_fp8` preset 或在安裝格勾選 `UPGRADE_TORCH_CU130`。
- **嚴禁交叉混用 LoRA**：Ref2VA（參考人像生成新鏡頭）務必使用 `minimax_h3_ref2va_*` 主模型配 `minimax_h3_ref2v_turbo_*` LoRA；FL2VA（首尾幀圖生影）則配 `minimax_h3_fl2v_turbo_*` LoRA，切勿混用。預設推薦 **8 步 Turbo LoRA** 以守住臉部紋理與細節。
- **無損防溢流與影片保留**：`minimax_h3_ref2va_8step_kj.json` 內建 `ModelSamplingMiniMaxH3 (12.0, 3.0)` → `MiniMaxLowVRAMAttention (head_chunks=16)` → `MiniMaxChunkFeedForward (chunks=8)`，同時啟動時預設帶入 `--reserve-vram 2.0` 與 `MAX_AGE_SEC=1800`（RAM Disk 保留 30 分鐘），生成完成後請於瀏覽器及時下載 MP4。
- **防變臉與提示詞規範**：近景大臉特寫約在 2.9 秒後較易發生身分漂移，建議特寫鏡頭控制在 3 秒以內或改用中景＋頭部比例空間約束；提示詞請使用英文三段式結構（`integrated_multimodal_description:`、`overall_soundscape:`、`non_diegetic_music:`），口播台詞包在 `<d>[Chinese] ...</d>` 中。
