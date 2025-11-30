# ARM64 + GB10 部署 vLLM 運行 gpt-oss-120b

本專案提供在 NVIDIA Grace Blackwell (GB10) 上使用 vLLM 部署 OpenAI gpt-oss-120b 模型的完整指南。

## 目錄
- [硬體需求](#硬體需求)
- [軟體需求](#軟體需求)
- [前置準備](#前置準備)
- [啟動服務](#啟動服務)
- [Docker 參數詳解](#docker-參數詳解)
- [vLLM 參數詳解](#vllm-參數詳解)
- [API 使用範例](#api-使用範例)
- [效能數據](#效能數據)
- [疑難排解](#疑難排解)

---

## 硬體需求

| 項目 | 規格 |
|------|------|
| CPU 架構 | ARM64 (aarch64) |
| GPU | NVIDIA GB10 (Grace Blackwell) |
| Compute Capability | 10.0 |
| GPU 記憶體 | ~120 GiB |
| 系統記憶體 | 128 GB RAM (建議) |

---

## 軟體需求

| 項目 | 版本 |
|------|------|
| CUDA | 12.8+ (Blackwell 最低需求) |
| Docker | 已安裝且可存取 GPU |
| NVIDIA Container Toolkit | 已安裝 |

### 環境變數
```bash
# Hugging Face Token (必須有 gpt-oss-120b 模型的存取權限)
export HF_TOKEN="your_huggingface_token"
```

---

## 檔案位置

本專案使用的重要檔案位置：

| 項目 | 路徑 | 說明 |
|------|------|------|
| **模型快取** | `/vllm/huggingface/hub/models--openai--gpt-oss-120b/` | Hugging Face 下載的模型權重，約 240GB (FP16 格式) |
| **tiktoken 編碼檔案** | `/vllm/encodings/` | tokenizer 所需的編碼檔案 |
| - o200k_base.tiktoken | `/vllm/encodings/o200k_base.tiktoken` | 主要編碼檔案 (3.6 MB)，gpt-oss 使用 |
| - cl100k_base.tiktoken | `/vllm/encodings/cl100k_base.tiktoken` | 備用編碼檔案 (1.7 MB) |
| **vLLM 快取** | `~/.cache/vllm/` | torch.compile 和 CUDA graph 快取 |

> **注意**: 模型權重在首次啟動時會自動下載到 `/vllm/huggingface/`，後續啟動會直接使用快取。

---

## 前置準備

### 1. 下載 tiktoken 編碼檔案

gpt-oss 模型使用 `openai_harmony` tokenizer，需要 tiktoken 編碼檔案。由於容器內可能無法直接下載，建議預先準備：

```bash
# 建立編碼檔案目錄
sudo mkdir -p /vllm/encodings
sudo chown -R $USER:$USER /vllm

# 下載 o200k_base 編碼 (gpt-oss 主要使用)
curl -L -o /vllm/encodings/o200k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

# 下載 cl100k_base 編碼 (備用)
curl -L -o /vllm/encodings/cl100k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# 驗證檔案
ls -la /vllm/encodings/
# 預期輸出:
# o200k_base.tiktoken  約 3.6 MB
# cl100k_base.tiktoken 約 1.7 MB
```

### 2. 拉取 Docker 映像

```bash
docker pull nvcr.io/nvidia/vllm:25.09-py3
```

映像資訊：
- **來源**: NVIDIA NGC
- **vLLM 版本**: 0.10.1.1+381074ae
- **平台**: 支援 ARM64 (aarch64)

---

## 啟動服務

### 完整啟動指令

```bash
docker run -d \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -e HF_TOKEN=$HF_TOKEN \
  -e TIKTOKEN_ENCODINGS_BASE=/etc/encodings \
  -v /vllm/huggingface:/root/.cache/huggingface \
  -v /vllm/encodings:/etc/encodings:ro \
  --name vllm-gpt-oss \
  nvcr.io/nvidia/vllm:25.09-py3 \
  vllm serve openai/gpt-oss-120b \
  --quantization mxfp4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --served-model-name gpt-oss-120b
```

### 等待服務就緒

模型載入約需 **7-8 分鐘**，可透過以下方式監控：

```bash
# 查看日誌
docker logs -f vllm-gpt-oss

# 或輪詢健康檢查端點
while ! curl -s http://localhost:8000/health > /dev/null; do
  echo "等待服務啟動..."
  sleep 10
done
echo "服務已就緒!"
```

---

## Docker 參數詳解

### 基礎參數

| 參數 | 說明 |
|------|------|
| `-d` | Detach 模式，容器在背景執行 |
| `--name vllm-gpt-oss` | 指定容器名稱，方便後續管理 |
| `-p 8000:8000` | 將容器的 8000 埠映射到主機的 8000 埠 |

### GPU 相關參數

| 參數 | 說明 |
|------|------|
| `--gpus all` | 讓容器存取所有可用的 GPU。也可指定特定 GPU，如 `--gpus '"device=0"'` |

### 記憶體與 IPC 參數

| 參數 | 說明 |
|------|------|
| `--ipc=host` | 使用主機的 IPC 命名空間。vLLM 需要此設定來進行跨 process 的共享記憶體通訊，對於 tensor parallelism 尤其重要 |
| `--ulimit memlock=-1` | 移除 memory lock 限制 (-1 = 無限制)。允許 GPU 驅動程式鎖定記憶體頁面，防止被 swap 出去，確保 GPU 存取效能 |
| `--ulimit stack=67108864` | 設定 stack size 為 64MB (67108864 bytes)。深度學習推論需要較大的 stack 空間來處理複雜的運算圖 |

### 環境變數

| 參數 | 說明 |
|------|------|
| `-e HF_TOKEN=$HF_TOKEN` | 傳遞 Hugging Face token 到容器，用於下載模型權重 |
| `-e TIKTOKEN_ENCODINGS_BASE=/etc/encodings` | 指定 tiktoken 編碼檔案的基礎路徑。`openai_harmony` tokenizer 會從此路徑載入編碼檔案 |

### Volume 掛載

| 參數 | 說明 |
|------|------|
| `-v /vllm/huggingface:/root/.cache/huggingface` | 掛載 Hugging Face 快取目錄。模型權重約 240GB (FP16)，掛載後可避免每次重新下載 |
| `-v /vllm/encodings:/etc/encodings:ro` | 掛載 tiktoken 編碼檔案。`:ro` 表示唯讀掛載，提高安全性 |

---

## vLLM 參數詳解

### 模型與量化

| 參數 | 說明 |
|------|------|
| `openai/gpt-oss-120b` | Hugging Face 模型 ID。gpt-oss-120b 是 OpenAI 開源的 120B 參數模型，架構為 `GptOssForCausalLM` |
| `--quantization mxfp4` | 使用 MXFP4 (Microscaling FP4) 量化。這是 Blackwell GPU 專屬的 4-bit 量化格式，可將記憶體需求從 ~240GB 降至 ~66GB |

### 記憶體管理

| 參數 | 說明 |
|------|------|
| `--gpu-memory-utilization 0.85` | GPU 記憶體使用率上限 (0.0-1.0)。設為 0.85 表示使用 85% 的 GPU 記憶體，預留 15% 給系統和 KV cache 動態分配。若遇 OOM 可降低此值 |
| `--max-model-len 32768` | 最大序列長度 (輸入 + 輸出 tokens)。gpt-oss-120b 支援最高 200K，但受限於 GB10 記憶體，設為 32K 是較平衡的選擇 |

### 服務設定

| 參數 | 說明 |
|------|------|
| `--served-model-name gpt-oss-120b` | API 中使用的模型名稱。客戶端請求時需指定此名稱 |

### 其他常用參數 (可選)

| 參數 | 說明 |
|------|------|
| `--tensor-parallel-size N` | Tensor Parallelism 的 GPU 數量。單 GPU 時為 1 (預設) |
| `--max-num-seqs 256` | 最大並行處理的序列數 |
| `--enforce-eager` | 禁用 CUDA Graph，改用 eager 模式。除錯時有用，但會降低效能 |
| `--trust-remote-code` | 允許執行模型倉庫中的自訂程式碼 |

---

## API 使用範例

### 健康檢查

```bash
curl http://localhost:8000/health
```

### 查看可用模型

```bash
curl http://localhost:8000/v1/models | jq .
```

回應範例：
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-oss-120b",
      "object": "model",
      "max_model_len": 32768
    }
  ]
}
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 500,
    "temperature": 0.7
  }' | jq .
```

### Streaming 回應

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Write a haiku about programming."}],
    "max_tokens": 100,
    "stream": true
  }'
```

### Python 範例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不需要 API key
)

response = client.chat.completions.create(
    model="gpt-oss-120b",
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
# 回應會包含 reasoning_content (推理過程)
```

---

## 效能數據

以下為 GB10 單卡實測數據：

| 指標 | 數值 |
|------|------|
| 模型載入時間 | ~5.5 分鐘 (權重載入) |
| torch.compile 時間 | ~23 秒 |
| CUDA Graph 捕捉 | 83 graphs, ~14 秒 |
| 總啟動時間 | ~7-8 分鐘 |
| 模型記憶體佔用 | 65.97 GiB |
| KV Cache 可用 | 32.37 GiB |
| KV Cache Token 數 | 471,360 tokens |
| 最大並行請求 (32K tokens/req) | ~27x |

---

## 疑難排解

### 1. HarmonyError: error downloading or loading vocab file

**錯誤訊息**:
```
openai_harmony.HarmonyError: error downloading or loading vocab file
```

**原因**: 容器內 `openai_harmony` 套件無法下載 tiktoken 編碼檔案

**解決方案**: 
- 確認已執行「前置準備」中的 tiktoken 下載步驟
- 確認 Docker 指令包含 `-e TIKTOKEN_ENCODINGS_BASE=/etc/encodings` 和 `-v /vllm/encodings:/etc/encodings:ro`

**參考**: https://github.com/vllm-project/vllm/issues/22525

---

### 2. GPU 記憶體不足 (OOM)

**錯誤訊息**:
```
torch.OutOfMemoryError: CUDA out of memory
# 或
Free memory (X GiB) less than desired (Y GiB)
```

**解決方案**:
```bash
# 降低 GPU 記憶體使用率
--gpu-memory-utilization 0.80  # 從 0.85 降到 0.80

# 或減少最大序列長度
--max-model-len 16384  # 從 32768 降到 16384
```

---

### 3. 容器啟動後立即退出

**檢查方式**:
```bash
# 查看容器狀態
docker ps -a | grep vllm-gpt-oss

# 查看完整日誌
docker logs vllm-gpt-oss
```

**常見原因**:
- HF_TOKEN 未設定或無效
- 模型存取權限不足
- 編碼檔案未正確掛載

---

### 4. 連線被拒絕

**錯誤訊息**:
```
curl: (7) Failed to connect to localhost port 8000
```

**解決方案**:
- 確認容器正在運行: `docker ps`
- 模型可能仍在載入中，請等待 7-8 分鐘
- 查看日誌確認狀態: `docker logs -f vllm-gpt-oss`

---

## 容器管理

### 停止 vLLM 服務

```bash
# 方法 1: 優雅停止 (建議)
# 會等待目前的請求處理完成後再停止，預設等待 10 秒
docker stop vllm-gpt-oss

# 方法 2: 指定等待時間 (秒)
# 適合有長時間運行請求的情況
docker stop -t 30 vllm-gpt-oss

# 方法 3: 強制停止 (不建議)
# 立即終止，不等待請求完成
docker kill vllm-gpt-oss
```

### 完全移除服務

```bash
# 停止並移除容器 (一行完成)
docker stop vllm-gpt-oss && docker rm vllm-gpt-oss

# 如果容器已經停止，直接移除
docker rm vllm-gpt-oss

# 強制移除 (即使正在運行)
docker rm -f vllm-gpt-oss
```

### 其他常用指令

```bash
# 查看容器狀態
docker ps | grep vllm-gpt-oss

# 查看所有容器 (包含已停止的)
docker ps -a | grep vllm-gpt-oss

# 重新啟動已停止的容器
docker start vllm-gpt-oss

# 重啟容器 (停止後立即啟動)
docker restart vllm-gpt-oss

# 查看即時日誌
docker logs -f vllm-gpt-oss

# 查看最後 100 行日誌
docker logs --tail 100 vllm-gpt-oss

# 進入容器 shell
docker exec -it vllm-gpt-oss bash

# 查看容器資源使用情況
docker stats vllm-gpt-oss
```

---

## API 端點列表

| 端點 | 方法 | 說明 |
|------|------|------|
| `/v1/chat/completions` | POST | Chat 補全 (OpenAI 相容) |
| `/v1/completions` | POST | 文字補全 |
| `/v1/models` | GET | 列出可用模型 |
| `/v1/responses` | POST | Response API |
| `/health` | GET | 健康檢查 |
| `/metrics` | GET | Prometheus 指標 |
| `/tokenize` | POST | 文字 tokenization |
| `/detokenize` | POST | Token 還原文字 |

---

## Blackwell GPU 專屬優化

GB10 (Blackwell 架構) 自動啟用以下優化：

| 功能 | 說明 |
|------|------|
| **MXFP4 量化** | 4-bit 量化格式，大幅降低記憶體需求 |
| **FlashInfer** | 優化的 top-p & top-k 取樣 |
| **Triton Attention** | 高效能注意力後端 |
| **CUDA Graphs** | 減少 kernel launch overhead |
| **torch.compile** | 動態圖編譯優化 |

---

## 授權

本部署指南僅供參考。gpt-oss-120b 模型受 OpenAI 使用條款約束。
