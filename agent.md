# Agent 規範：vLLM 模型部署專案

本文件定義 LLM Agent 在處理此專案時必須遵守的規範與約束。

---

## 專案概述

- **目標**: 在 ARM64 + NVIDIA GB10 上部署 vLLM 運行大型語言模型
- **容器映像**: `nvcr.io/nvidia/vllm:25.09-py3`
- **支援模型**:
  - `openai/gpt-oss-120b` (MXFP4 量化)
  - `google/gemma-3-27b-it` (BF16 或 MXFP4)

---

## 必須遵守的規範

### 1. Docker 執行規範 (通用)

**所有模型必須包含的參數**:
- `--gpus all` - GPU 存取
- `--ipc=host` - 共享記憶體
- `--ulimit memlock=-1` - 記憶體鎖定
- `--ulimit stack=67108864` - Stack 大小
- `-v ~/vllm/huggingface:/root/.cache/huggingface` - 模型快取掛載

**禁止**:
- 不要省略 `--ipc=host`，會導致 tensor parallelism 失敗

### 2. 模型特定規範

#### gpt-oss-120b

**額外必須包含**:
- `-e TIKTOKEN_ENCODINGS_BASE=/etc/encodings` - tiktoken 路徑
- `-v ~/vllm/encodings:/etc/encodings:ro` - 編碼檔案掛載

**vLLM 參數**:
- `--quantization mxfp4` - 必須使用，否則 OOM
- `--gpu-memory-utilization 0.85`
- `--max-model-len 32768`
- `--served-model-name gpt-oss-120b`

**容器名稱**: `vllm-gpt-oss`

#### Gemma 3 27B

**vLLM 參數 (BF16)**:
- `--gpu-memory-utilization 0.90`
- `--max-model-len 8192`
- `--served-model-name gemma-3-27b`

**vLLM 參數 (MXFP4 量化)**:
- `--quantization mxfp4`
- `--gpu-memory-utilization 0.90`
- `--max-model-len 32768`
- `--served-model-name gemma-3-27b`

**容器名稱**: `vllm-gemma3`

**注意**: Gemma 3 不需要 tiktoken 編碼檔案

### 3. 前置作業規範

**啟動任何模型前必須確認**:
1. `~/vllm/huggingface/` 目錄存在
2. `$HF_TOKEN` 環境變數已設定
3. Docker 映像已拉取

**僅 gpt-oss-120b 需要額外確認**:
1. `~/vllm/encodings/o200k_base.tiktoken` 存在
2. `~/vllm/encodings/cl100k_base.tiktoken` 存在

**若編碼檔案不存在，必須先下載**:
```bash
mkdir -p ~/vllm/encodings ~/vllm/huggingface
curl -L -o ~/vllm/encodings/o200k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
curl -L -o ~/vllm/encodings/cl100k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
```

### 4. 錯誤處理規範

| 錯誤 | 處理方式 |
|------|----------|
| `HarmonyError: error downloading or loading vocab file` | 僅 gpt-oss: 檢查 tiktoken 編碼檔案是否正確掛載 |
| `Free memory (X GiB) less than desired (Y GiB)` | 降低 `--gpu-memory-utilization` 或 `--max-model-len` |
| 容器名稱衝突 | 先執行 `docker rm -f <容器名稱>` 移除舊容器 |
| 容器立即退出 | 檢查 `docker logs <容器名稱>` 確認原因 |

### 5. 驗證規範

**服務啟動後必須驗證**:
1. 健康檢查: `curl http://localhost:8000/health`
2. 模型列表: `curl http://localhost:8000/v1/models`
3. 推論測試: 發送簡單的 chat completion 請求

### 6. 切換模型規範

同一時間只能運行一個模型 (共用 port 8000)。切換前必須：
```bash
docker rm -f vllm-gpt-oss 2>/dev/null
docker rm -f vllm-gemma3 2>/dev/null
```

---

## 已知限制

### gpt-oss-120b
- 模型載入時間約 7-8 分鐘
- GB10 單卡 max-model-len 上限約 32768
- MXFP4 量化使用 Marlin kernel (非原生 FP4)

### Gemma 3 27B
- 模型載入時間約 3-4 分鐘
- BF16 時 max-model-len 建議 8192
- MXFP4 量化可支援更長序列

### 通用
- 首次推論會額外編譯 CUDA graphs
- 埠號 8000 被佔用時需先停止現有容器

---

## 參考資源

- tiktoken 問題: https://github.com/vllm-project/vllm/issues/22525
- vLLM 官方文件: https://docs.vllm.ai/
- NVIDIA NGC: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
- Gemma 3: https://huggingface.co/google/gemma-3-27b-it
