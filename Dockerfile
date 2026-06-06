FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝各模块
COPY rag/ ./rag/
COPY agent/ ./agent/
COPY api/ ./api/

RUN mkdir -p /app/uploads /app/data

EXPOSE 8000

RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
