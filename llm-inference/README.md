# LLM Inference 專區

本目錄用來存放開源語言模型（包含量化 GGUF 與原版權重）在 Colab 上的推理實驗與互動介面配置。

## 規劃內容

- `llamacpp_gguf.ipynb`：透過 `llama.cpp` / `llama-cpp-python` 跑 GGUF 量化模型（適合免費版 15GB 顯存）
- `vllm_or_webui.ipynb`：使用 `vLLM` 提供 OpenAI 相容 API 並搭配 WebUI 測試
- 模型下載統一透過 `shared.hf_download` 並讀取 Colab Secrets 中的 `HF_TOKEN`（若為 gated repo）
