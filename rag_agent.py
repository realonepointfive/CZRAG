"""
RAG Agent — 把向量检索封装成 Tool，由 LLM 自主决定何时调用。
用法：
    from rag_agent import build_rag_agent
    agent = build_rag_agent(vectorstore)
    result = agent.invoke({"input": "什么是RAG？"})
"""

import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag_pipeline import (
    retrieve_top_chunks,
    dedupe_search_results,
    load_llm,
)


def make_retrieval_tool(vectorstore: Chroma):
    """动态生成一个绑定了 vectorstore 的检索 Tool。"""

    @tool
    def search_documents(query: str) -> str:
        """
        在已上传的 PDF 文档中检索与问题相关的内容。
        当需要查找文档里的信息时调用此工具。
        参数 query：用于检索的关键词或问题。
        """
        raw = retrieve_top_chunks(vectorstore, query, k=9)
        top = dedupe_search_results(raw)[:4]
        if not top:
            return "未找到相关内容。"
        parts = []
        for doc, score in top:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            parts.append(
                f"【来源: {source} 第{page}页 | 相似度: {score:.3f}】\n{doc.page_content}"
            )
        return "\n\n".join(parts)

    return search_documents


def build_rag_agent(vectorstore: Chroma) -> AgentExecutor:
    """
    构建一个带有 RAG 检索工具的 Agent。
    LLM 会自主决定是否调用检索工具以及调用几次。
    """
    llm = load_llm()
    tools = [make_retrieval_tool(vectorstore)]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位专业的文档问答助手。"
         "你有一个工具可以检索用户上传的 PDF 文档。"
         "请先思考是否需要检索，再决定如何回答。"
         "回答请使用中文，尽量简洁准确，并注明信息来源页码。"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
