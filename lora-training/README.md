# LoRA Training 專區

本目錄提供在 Google Colab 上使用 **Fizgig** 訓練 **MOSS-Image 人物 LoRA** 的安裝腳本、Python 自動化模組與薄 Notebook 入口，同時保留後續擴充 LLM QLoRA 的空間。

## 檔案說明

- [`install.sh`](./install.sh)：於 `/content/Fizgig` 安裝固定版本 Fizgig（預設 `v9.7.7`）與獨立 `venv`，自動清除 Colab 的 `UV_SYSTEM_PYTHON` 干擾、設定 `DISABLE_CUDA=1`（`hqq` 相容）並驗證 CUDA 與 `bf16` 運算。
- [`fizgig_moss.py`](./fizgig_moss.py)：提供 MOSS-Image 預設模型清單下載與 `user_data/prefs.json` 自動綁定、資料集 TOML 生成、兩階段快取與訓練命令建構（支援 `dop_adamw` 與 `muon_baseline`、`lora` / `lokr`），以及帶 `FIZGIG_PASSWORD` 認證的 Fizgig Web UI + TryCloudflare HTTP/2 啟動器。
- [`fizgig_moss.ipynb`](./fizgig_moss.ipynb)：4 個 Cell 的 Colab 薄入口（同步倉庫與讀取 Secrets → 執行 `install.sh` → 下載 MOSS-Image 模型 → 啟動 Fizgig Web UI 或預覽 CLI 訓練指令）。

## 使用提醒

1. **執行階段建議**：選用 **L4** 或 **A100（High-RAM）**；避免使用 T4（無原生 `bf16` Tensor Core 加速，且標準記憶體難以承載 19.3 GB `qwen3.5_9b_bf16` text cache）。
2. **Colab Secrets**：於左側鑰匙圖示設定 `HF_TOKEN` 並開啟 **Notebook access**；選填 `FIZGIG_PASSWORD`（至少 8 字元）作為 Web UI 登入密碼，未填時會自動產生隨機密碼。
3. **資料與產出隔離**：訓練資料集、私人設定檔與產出權重一律留在本機、Colab 暫存或個人私有儲存空間，絕不提交至本公開倉庫。
