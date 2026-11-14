# colab-workbench

模組化的 Google Colab 工作台，用來統一管理各類 AI 實驗的啟動筆記本（`.ipynb`）、安裝腳本與共用工具，將「輕量公開設定」與「大型模型／私人資產」徹底分離。

## 目錄結構

| 目錄 | 用途 | 狀態 |
| --- | --- | --- |
| [`shared/`](./shared/) | 跨專區共用模組（Secrets 讀取、Drive 掛載與軟連結、HF 下載、內網穿透） | 已建立 |
| [`comfyui/`](./comfyui/) | ComfyUI 生圖工作流與節點安裝設定 | 骨架就緒 |
| [`llm-inference/`](./llm-inference/) | 開源大語言模型推理（如 `llama.cpp` / `vLLM` / WebUI） | 骨架就緒 |
| [`lora-training/`](./lora-training/) | 影像 LoRA 與 LLM QLoRA 微調訓練環境 | 骨架就緒 |
| [`docs/`](./docs/) | 公開使用指南（[Colab Secrets 與 Google Drive 持久化](./docs/secrets-and-drive.md)） | 已建立 |
| [`tests/`](./tests/) | `shared/` 共用模組單元測試 | 已建立 |

## 設計原則

1. **薄筆記本（Thin Notebook）**：安裝邏輯、依賴清單與共用 Python 工具都放在倉庫裡；`.ipynb` 只負責拉取本倉庫並呼叫對應腳本，方便在本地用編輯器維護與做版本控制。
2. **程式碼與資產分離**：本公開倉庫**只存放純文字設定與程式碼**；API Token 放在 Colab 原生 Secrets，大型模型權重、資料集與產出圖則透過掛載 Google Drive 持久化（或於開機後現抓到 `/content/`）。
3. **按工具獨立分區**：每個實驗方向各自一個資料夾，互不干擾，未來新增工具只需平級新增目錄。

## 在 Colab 中快速引導共用模組

在任意 Colab Notebook 的第一個 Cell 執行以下程式碼，即可載入 `shared` 套件：

```python
import os
import sys

REPO_URL = "https://github.com/<your-username>/colab-workbench.git"
REPO_DIR = "/content/colab-workbench"

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull --ff-only

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
```

接著就能直接呼叫共用工具，例如讀取 Colab Secrets 與掛載 Google Drive：

```python
from shared import export_secret_to_env, mount_drive, ensure_workspace_dirs

export_secret_to_env("HF_TOKEN")
mount_drive()
dirs = ensure_workspace_dirs()
```

詳細的密鑰與硬碟空間配置說明請見 [docs/secrets-and-drive.md](./docs/secrets-and-drive.md)。

## 本地測試

```bash
python3 -m unittest discover -s tests -v
```
