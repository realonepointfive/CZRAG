import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_pipeline import (
    build_documents_from_pdf,
    split_documents,
    build_vectorstore,
    retrieve_top_chunks,
    dedupe_search_results,
    answer_question_with_llm,
    load_llm,
    load_embeddings,
)
from rag_agent import build_rag_agent
from langchain_chroma import Chroma
from langchain.agents import AgentExecutor

app = FastAPI(title="RAG API", description="PDF 上传 + 问答接口")

# 每个会话对应一个 vectorstore，key = session_id
_sessions: dict[str, Chroma] = {}
# 每个会话对应一个 Agent，key = session_id
_agents: dict[str, AgentExecutor] = {}

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/upload", summary="上传 PDF 并建立向量索引")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    session_id = uuid.uuid4().hex
    save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        documents = build_documents_from_pdf(str(save_path))
        chunks = split_documents(documents)
        embeddings = load_embeddings()
        persist_dir = f"./chroma_db_{session_id}"
        vectorstore = build_vectorstore(chunks, persist_directory=persist_dir, embeddings=embeddings)
        _sessions[session_id] = vectorstore
        _agents[session_id] = build_rag_agent(vectorstore)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引构建失败: {e}")

    return {"session_id": session_id, "chunks": len(chunks), "filename": file.filename}


@app.post("/query", response_model=QueryResponse, summary="对已上传的 PDF 提问")
async def query(req: QueryRequest):
    vectorstore = _sessions.get(req.session_id)
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="session_id 不存在，请先上传 PDF")

    raw_chunks = retrieve_top_chunks(vectorstore, req.question, k=req.top_k * 3)
    top_chunks = dedupe_search_results(raw_chunks)[: req.top_k]

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "chunk_id": doc.metadata.get("chunk_id", "?"),
            "score": round(score, 4),
            "preview": doc.page_content[:120].replace("\n", " "),
        }
        for doc, score in top_chunks
    ]

    try:
        llm = load_llm()
        answer = answer_question_with_llm(llm, req.question, top_chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"LLM 不可用: {e}")

    return QueryResponse(answer=answer, sources=sources)


class AgentRequest(BaseModel):
    session_id: str
    input: str


@app.post("/agent", summary="用 Agent 模式提问（LLM 自主决定是否检索）")
async def agent_query(req: AgentRequest):
    agent = _agents.get(req.session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="session_id 不存在，请先上传 PDF")
    try:
        result = agent.invoke({"input": req.input})
        return {"answer": result["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {e}")


@app.delete("/session/{session_id}", summary="删除会话及向量数据库")
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session_id 不存在")

    _sessions.pop(session_id)
    _agents.pop(session_id, None)
    persist_dir = Path(f"./chroma_db_{session_id}")
    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)

    return {"deleted": session_id}


@app.get("/sessions", summary="列出所有活跃会话")
async def list_sessions():
    return {"sessions": list(_sessions.keys())}


@app.get("/health")
async def health():
    return {"status": "ok"}
