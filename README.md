# CZRAG — PDF RAG API + Agent

> 上传 PDF → 自动建立向量索引 → REST API 问答 / Agent 自主检索

---

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                      Client                             │
│              (curl / Swagger UI / 前端)                  │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI (api.py)                       │
│                                                         │
│  POST /upload ──► 解析 PDF ──► 切分 Chunks              │
│                               └──► Chroma 向量数据库     │
│                                                         │
│  POST /query  ──► 固定 RAG 检索 ──► LLM 生成答案         │
│                                                         │
│  POST /agent  ──► Agent (rag_agent.py)                  │
│                     └── LLM 自主决定何时/如何检索         │
│                           └──► Chroma 向量数据库         │
│                                                         │
│  DELETE /session/{id}  ──► 清理会话 + 向量数据库         │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
 SiliconFlow API                    Chroma (本地)
 - Qwen3-30B (LLM)                  - 向量索引
 - BAAI/bge-m3 (Embedding)          - 持久化到磁盘
```

### 两种问答模式对比

| 模式 | 端点 | 行为 |
|------|------|------|
| **RAG** | `POST /query` | 固定流程：检索 top-k → LLM 生成，速度快 |
| **Agent** | `POST /agent` | LLM 自主决定是否检索、检索几次，适合复杂推理 |

---

## 文件说明

```
CZRAG/
├── rag_pipeline.py     # 核心：PDF 解析、切分、Embedding、Chroma 检索
├── rag_agent.py        # Agent：把检索封装成 Tool，由 LLM 自主调用
├── api.py              # FastAPI：暴露 REST 接口
├── Dockerfile          # 容器打包
├── docker-compose.yml  # 一键启动
├── requirements.txt    # Python 依赖
└── .env.example        # 环境变量模板
```

---

## 快速开始

### 方式一：Docker（推荐）

```bash
# 1. 复制环境变量文件并填入 API Key
cp .env.example .env
# 编辑 .env，填入 SILICONFLOW_API_KEY

# 2. 一键启动
docker compose up -d

# 3. 查看日志
docker compose logs -f
```

API 启动后访问：`http://localhost:8000/docs`

---

### 方式二：本地运行

```bash
# 1. 创建虚拟环境
python -m venv langchain_env
langchain_env\Scripts\activate      # Windows
# source langchain_env/bin/activate  # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 SILICONFLOW_API_KEY

# 4. 启动 API
uvicorn api:app --reload
```

---

## API 使用示例

### 1. 上传 PDF

```bash
curl -F "file=@paper.pdf" http://localhost:8000/upload
# 返回：{"session_id": "abc123", "chunks": 87, "filename": "paper.pdf"}
```

### 2. RAG 问答（快速，固定流程）

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "question": "这篇论文的主要贡献是什么？", "top_k": 4}'
```

### 3. Agent 问答（智能，LLM 自主决策）

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123", "input": "第3章和第5章有什么关联？"}'
```

### 4. 删除会话

```bash
curl -X DELETE http://localhost:8000/session/abc123
```

---

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `SILICONFLOW_API_KEY` | SiliconFlow API Key（LLM + Embedding） | ✅ |

获取地址：[https://cloud.siliconflow.cn/](https://cloud.siliconflow.cn/)

---

## 技术栈

- **LLM**：Qwen3-30B-A3B via SiliconFlow
- **Embedding**：BAAI/bge-m3（多语言）via SiliconFlow
- **向量数据库**：Chroma（本地持久化）
- **框架**：LangChain + FastAPI
- **容器**：Docker + Docker Compose
