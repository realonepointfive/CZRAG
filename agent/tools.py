"""
Agent 可用工具：
1. search_documents  — 在用户上传的 PDF 向量库中检索
2. web_search        — Tavily 联网搜索
"""

import os
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma

from rag.vectorstore import retrieve_top_chunks, dedupe_search_results


def make_retrieval_tool(vectorstore: Chroma):
    """生成绑定了特定 vectorstore 的 RAG 检索工具。"""

    @tool
    def search_documents(query: str) -> str:
        """
        在用户上传的 PDF 文档中检索与问题相关的内容片段。
        当问题可能在 PDF 文档中有答案时，优先调用此工具。
        参数 query：用于检索的关键词或问题描述。
        """
        raw = retrieve_top_chunks(vectorstore, query, k=9)
        top = dedupe_search_results(raw)[:4]
        if not top:
            return "PDF 文档中未找到相关内容。"
        parts = []
        for doc, score in top:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(
                f"【来源: {source} 第{page}页 | 相似度: {score:.3f}】\n{doc.page_content}"
            )
        return "\n\n".join(parts)

    return search_documents


def make_web_search_tool() -> TavilySearchResults:
    """创建 Tavily 联网搜索工具。需要环境变量 TAVILY_API_KEY。"""
    return TavilySearchResults(
        max_results=4,
        description=(
            "使用 Tavily 搜索引擎进行联网搜索。"
            "当 PDF 文档中信息不足、问题涉及最新资讯或需要外部知识时调用。"
        ),
    )
