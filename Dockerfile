# ── Build stage ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装系统依赖（chromadb 需要 sqlite3；pypdf 需要 gcc）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# 拷贝依赖
COPY --from=builder /install /usr/local

# 拷贝源代码（不含 venv、数据库、上传文件）
COPY rag_pipeline.py rag_agent.py api.py ./

# 运行时目录
RUN mkdir -p /app/uploads /app/data

# 暴露 FastAPI 端口
EXPOSE 8000

# 非 root 用户运行
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
