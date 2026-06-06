from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="CZRAG API",
    description="PDF 上传 + RAG 问答 + Agent（RAG 检索 & Tavily 联网搜索）+ 对话记忆",
    version="2.0.0",
)

app.include_router(router)


@app.get("/health", tags=["系统"])
async def health():
    return {"status": "ok"}
