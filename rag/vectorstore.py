import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.pipeline import load_embeddings


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, 0o700)
        func(path)
    except Exception:
        pass


def _ensure_directory(base_dir: str) -> str:
    target = Path(base_dir)
    if target.exists():
        try:
            shutil.rmtree(target, onerror=_handle_remove_readonly)
            return str(target)
        except Exception:
            fallback = target.parent / f"{target.name}_{uuid.uuid4().hex[:8]}"
            return str(fallback)
    return str(target)


def build_vectorstore(
    chunks: list[Document],
    persist_directory: str = "./pdf_chroma_db",
    embeddings=None,
) -> Chroma:
    if embeddings is None:
        embeddings = load_embeddings()
    persist_directory = _ensure_directory(persist_directory)
    vectorstore = Chroma(
        collection_name="pdf_documents",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


def retrieve_top_chunks(
    vectorstore: Chroma, query: str, k: int = 9
) -> list[tuple[Document, float]]:
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


def dedupe_search_results(
    chunks: list[tuple[Document, float]]
) -> list[tuple[Document, float]]:
    seen = set()
    unique = []
    for doc, score in chunks:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id"),
            doc.page_content[:120],
        )
        if key not in seen:
            seen.add(key)
            unique.append((doc, score))
    return unique
