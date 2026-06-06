import uuid
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, List
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.pipeline import build_documents_from_pdf, split_documents, load_llm, load_embeddings
from rag.vectorstore import build_vectorstore, retrieve_top_chunks, dedupe_search_results
from agent.agent import build_agent, chat as agent_chat
from agent.memory import memory_store
from api.schemas import (
    ChatRequest, ChatResponse,
    QueryRequest, QueryResponse, SourceItem,
    HistoryResponse,
)

router = APIRouter()

# ── 全局会话状态 ──────────────────────────────────────────────
_sessions: dict[str, Chroma] = {}        # session_id → vectorstore
_agents: dict[str, object] = {}          # session_id → AgentExecutor

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ── 上传一个或多个 PDF（新建会话）────────────────────────────
@router.post("/upload", response_model=None, summary="上传 PDF（支持多文件），新建会话")
async def upload_pdf(files: List[UploadFile] = File(...)):
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} 不是 PDF 文件")

    session_id = uuid.uuid4().hex
    embeddings = load_embeddings()
    all_chunks = []

    try:
        for file in files:
            save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
            with open(save_path, "wb") as fp:
                fp.write(await file.read())
            documents = build_documents_from_pdf(str(save_path))
            all_chunks.extend(split_documents(documents))

        vectorstore = build_vectorstore(
            all_chunks,
            persist_directory=f"./chroma_db_{session_id}",
            embeddings=embeddings,
        )
        _sessions[session_id] = vectorstore
        _agents[session_id] = build_agent(vectorstore)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引构建失败: {e}")

    return {
        "session_id": session_id,
        "chunks": len(all_chunks),
        "files": [f.filename for f in files],
    }


# ── 向已有会话追加 PDF ────────────────────────────────────────
@router.post("/session/{session_id}/upload", response_model=None, summary="向已有会话追加 PDF")
async def add_pdf_to_session(session_id: str, files: List[UploadFile] = File(...)):
    vectorstore = _sessions.get(session_id)
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="session_id 不存在，请先调用 /upload 创建会话")

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{f.filename} 不是 PDF 文件")

    new_chunks = []
    try:
        for file in files:
            save_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
            with open(save_path, "wb") as fp:
                fp.write(await file.read())
            documents = build_documents_from_pdf(str(save_path))
            new_chunks.extend(split_documents(documents))

        # 直接往已有 vectorstore 追加，无需重建
        vectorstore.add_documents(new_chunks)
        # 重建 agent（vectorstore 对象未变，agent 引用不变，实际可跳过；保险起见重建）
        _agents[session_id] = build_agent(vectorstore)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"追加索引失败: {e}")

    return {
        "session_id": session_id,
        "added_chunks": len(new_chunks),
        "files": [f.filename for f in files],
    }


# ── Agent 多轮对话 ────────────────────────────────────────────
@router.post("/agent/chat", response_model=ChatResponse, summary="Agent 对话（RAG + 联网，带记忆）")
async def agent_chat_endpoint(req: ChatRequest):
    agent = _agents.get(req.session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="session_id 不存在，请先上传 PDF")
    try:
        answer = agent_chat(agent, req.session_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {e}")
    return ChatResponse(session_id=req.session_id, answer=answer)


# ── 查看对话历史 ──────────────────────────────────────────────
@router.get("/agent/history/{session_id}", response_model=HistoryResponse, summary="查看对话历史")
async def get_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session_id 不存在")
    return HistoryResponse(
        session_id=session_id,
        history=memory_store.history_summary(session_id),
    )


# ── 清除对话历史 ──────────────────────────────────────────────
@router.delete("/agent/history/{session_id}", summary="清除对话记忆（保留向量库）")
async def clear_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session_id 不存在")
    memory_store.clear(session_id)
    return {"cleared": session_id}


# ── 简单 RAG 问答（无 Agent，快速）────────────────────────────
@router.post("/query", response_model=QueryResponse, summary="RAG 直接问答（固定流程，速度快）")
async def rag_query(req: QueryRequest):
    vectorstore = _sessions.get(req.session_id)
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="session_id 不存在，请先上传 PDF")

    raw = retrieve_top_chunks(vectorstore, req.question, k=req.top_k * 3)
    top = dedupe_search_results(raw)[: req.top_k]

    sources = [
        SourceItem(
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page", "?"),
            chunk_id=doc.metadata.get("chunk_id", "?"),
            score=round(score, 4),
            preview=doc.page_content[:120].replace("\n", " "),
        )
        for doc, score in top
    ]

    try:
        llm = load_llm()
        context = "\n\n".join(
            f"来源: {doc.metadata.get('source')} 第{doc.metadata.get('page')}页\n{doc.page_content}"
            for doc, _ in top
        )
        prompt = ChatPromptTemplate.from_template(
            "你是一位助手。请根据下面的文档片段回答用户问题。\n\n"
            "文档片段：\n{context}\n\n问题：{question}\n\n请用中文简明回答。"
        )
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": req.question}
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"LLM 不可用: {e}")

    return QueryResponse(answer=answer, sources=sources)


# ── 删除会话 ──────────────────────────────────────────────────
@router.delete("/session/{session_id}", summary="删除会话（向量库 + 记忆）")
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="session_id 不存在")

    _sessions.pop(session_id)
    _agents.pop(session_id, None)
    memory_store.clear(session_id)

    persist_dir = Path(f"./chroma_db_{session_id}")
    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)

    return {"deleted": session_id}


# ── 列出所有会话 ──────────────────────────────────────────────
@router.get("/sessions", summary="列出所有活跃会话")
async def list_sessions():
    return {"sessions": list(_sessions.keys())}
