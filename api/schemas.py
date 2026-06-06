from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    session_id: str
    chunks: int
    filename: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: Optional[int] = 4


class SourceItem(BaseModel):
    source: str
    page: str | int
    chunk_id: str
    score: float
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]
