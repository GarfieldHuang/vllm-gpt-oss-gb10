# Agent 規範：vLLM gpt-oss-120b 部署專案

本文件定義 LLM Agent 在處理此專案時必須遵守的規範與約束。

---

## 專案概述

- **目標**: 在 ARM64 + NVIDIA GB10 上部署 vLLM 運行 gpt-oss-120b
- **容器映像**: `nvcr.io/nvidia/vllm:25.09-py3`
- **模型**: `openai/gpt-oss-120b`
- **量化**: `mxfp4`

---

## 必須遵守的規範

### 1. Docker 執行規範

**必須包含的參數**:
- `--gpus all` - GPU 存取
- `--ipc=host` - 共享記憶體
- `--ulimit memlock=-1` - 記憶體鎖定
- `--ulimit stack=67108864` - Stack 大小
- `-e TIKTOKEN_ENCODINGS_BASE=/etc/encodings` - tiktoken 路徑
- `-v /vllm/encodings:/etc/encodings:ro` - 編碼檔案掛載
- `-v /vllm/huggingface:/root/.cache/huggingface` - 模型快取掛載

**禁止**:
- 不要省略 `--ipc=host`，會導致 tensor parallelism 失敗
- 不要省略 tiktoken 相關設定，會導致 HarmonyError

### 2. vLLM 參數規範

**必須設定**:
- `--quantization mxfp4` - GB10 記憶體限制需要量化
- `--gpu-memory-utilization 0.85` - 避免 OOM
- `--max-model-len 32768` - GB10 單卡限制

**不要使用**:
- `--trust-remote-code` - gpt-oss 不需要
- `--enable-auto-tool-choice` - gpt-oss 會自動啟用 tool use

### 3. 前置作業規範

**啟動前必須確認**:
1. `/vllm/encodings/o200k_base.tiktoken` 存在
2. `/vllm/encodings/cl100k_base.tiktoken` 存在
3. `/vllm/huggingface/` 目錄存在
4. `$HF_TOKEN` 環境變數已設定
5. Docker 映像已拉取

**若編碼檔案不存在，必須先下載**:
```bash
sudo mkdir -p /vllm/encodings /vllm/huggingface
sudo chown -R $USER:$USER /vllm
curl -L -o /vllm/encodings/o200k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
curl -L -o /vllm/encodings/cl100k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
```

### 4. 錯誤處理規範

| 錯誤 | 處理方式 |
|------|----------|
| `HarmonyError: error downloading or loading vocab file` | 檢查 tiktoken 編碼檔案是否正確掛載 |
| `Free memory (X GiB) less than desired (Y GiB)` | 降低 `--gpu-memory-utilization` 或 `--max-model-len` |
| 容器立即退出 | 檢查 `docker logs` 確認原因 |

### 5. 驗證規範

**服務啟動後必須驗證**:
1. 健康檢查: `curl http://localhost:8000/health`
2. 模型列表: `curl http://localhost:8000/v1/models`
3. 推論測試: 發送簡單的 chat completion 請求

### 6. 容器命名規範

- 容器名稱: `vllm-gpt-oss`
- 埠號: `8000`
- 模型名稱 (API): `gpt-oss-120b`

---

## 已知限制

1. 模型載入時間約 7-8 分鐘，需耐心等待
2. GB10 單卡 max-model-len 上限約 32768
3. MXFP4 量化在此 GPU 使用 Marlin kernel (非原生 FP4)
4. 首次推論會額外編譯 CUDA graphs

---

## 參考資源

- tiktoken 問題: https://github.com/vllm-project/vllm/issues/22525
- vLLM 官方文件: https://docs.vllm.ai/
- NVIDIA NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
