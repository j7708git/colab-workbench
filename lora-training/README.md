# LoRA Training 專區

本目錄用來存放影像 LoRA 與語言模型 QLoRA 的微調訓練 Notebook 與配置範例。

## 規劃內容

- `image_lora.ipynb`：影像模型 LoRA 訓練入口（訓練集與輸出 Checkpoint 建議保存於 Google Drive）
- `llm_qlora.ipynb`：語言模型 QLoRA 微調入口（如 `unsloth` / `peft`）
- 訓練資料集、個人設定檔與產出權重一律放在 Google Drive，不提交到本倉庫
