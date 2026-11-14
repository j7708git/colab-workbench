# Colab Secrets 與 Google Drive 持久化指南

本倉庫是**公開（public）**倉庫，任何寫死在 `.ipynb` 或 `.py` 裡的 Token 都會被全世界看到。請依照以下方式安全管理密鑰與大檔案。

---

## 1. 用 Colab 原生 Secrets 存 API 金鑰（絕對不進 Git）

1. 開啟任一 Google Colab Notebook。
2. 點選左側邊欄的 **鑰匙圖示（Secrets）**。
3. 新增需要的金鑰，並將左側的 **Notebook access** 開關打開：
   - `HF_TOKEN`：Hugging Face Access Token（下載 gated 或私有模型用）
   - 其他自訂金鑰（依未來專區需求增補）
4. 在 Notebook 裡透過 `shared.runtime_env` 讀取：

```python
from shared import export_secret_to_env, require_secret

# 自動把 HF_TOKEN 匯入環境變數（不存在時保持靜默；必要時改用 required=True）
export_secret_to_env("HF_TOKEN")

# 若一定要有該密鑰才繼續，會給出明確提示錯誤
token = require_secret("HF_TOKEN")
```

> 原理：Secrets 存在**你自己的 Google 帳號**裡。別人點開你的公開 Notebook 執行時，程式會讀取他自己 Colab 帳號裡的 Secrets，你的金鑰不會外洩。

---

## 2. Google Drive 持久化（模型與輸出保存）

Colab 的 `/content/` 會在斷線或超時後清空，但程式碼本身重新 `git clone` 與 `pip install` 只需 1～2 分鐘，因此建議採 **「程式留本地、大檔放 Drive」** 原則：

| 類型 | 建議放置位置 | 原因 |
| --- | --- | --- |
| 倉庫程式碼、Python 套件（venv） | `/content/`（Colab 本地 SSD） | 細碎小檔極多，放 Drive 會被虛擬檔案系統拖慢 |
| 模型權重（`.safetensors` / `.gguf`） | `/content/drive/MyDrive/colab-workbench/models/` | 單檔體積大，免去每次重下數 GB 的等待 |
| 訓練資料集、LoRA、成品圖 | `/content/drive/MyDrive/colab-workbench/` | 斷線不會遺失心血 |

### 用軟連結（symlink）接回工具目錄

```python
from shared import ensure_workspace_dirs, link_dir, mount_drive

mount_drive()
dirs = ensure_workspace_dirs()

# 例如把 ComfyUI 的輸出資料夾指回 Google Drive
link_dir(dirs["outputs"] / "comfyui", "/content/ComfyUI/output")
```

> 提醒：免費 Google 帳號僅有 15 GB 空間（與 Gmail、相簿共用）。若空間有限，可只把常用 LoRA 與成品圖放 Drive，大型 Base Model 開機時再由 `shared.hf_download` 現抓進 `/content/`。
