# ComfyUI 專區

本目錄用來存放 ComfyUI 在 Google Colab 上的啟動 Notebook、節點安裝腳本與模型清單模板。

## 規劃內容

- `comfyui.ipynb`：薄入口 Notebook（呼叫 `shared/` 掛載 Drive、套用 `extra_model_paths.yaml` 或軟連結、啟動 ComfyUI 並建立 `cloudflared` 瀏覽網址）
- `install.sh`：安裝 ComfyUI 本體、Python 依賴與常用 Custom Nodes
- `models.example.yaml`：模型下載清單範例（真正下載到的 `.safetensors` 不進版控）

## 注意事項

- 免費版 Colab 預設 GPU（如 T4 15GB）較適合中小型工作流或嘗鮮，且不建議長時間掛機跑隧道穿透。
- 建議將程式本體裝在 `/content/ComfyUI`（本地臨時 SSD 速度較快），僅將大型模型與 `output/` 指向 Google Drive。
