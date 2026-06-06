"""
LangChain Agent 核心。
工具：search_documents（RAG）+ web_search（Tavily）
记忆：从 memory_store 读取 chat_history，回答后写回。
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma

from rag.pipeline import load_llm
from agent.tools import make_retrieval_tool, make_web_search_tool
from agent.memory import memory_store

SYSTEM_PROMPT = """你是一位专业的文档问答助手，拥有两个工具：

1. search_documents：在用户上传的 PDF 文档中检索相关内容
2. tavily_search：联网搜索最新信息

工作策略：
- 优先使用 search_documents 查找 PDF 中的答案
- 若 PDF 信息不足或问题涉及时效性内容，使用 tavily_search 补充
- 可多次调用工具直到获得足够信息
- 回答使用中文，并注明信息来源（PDF 页码 或 网页链接）
- 综合对话历史，保持上下文连贯"""


def build_agent(vectorstore: Chroma) -> AgentExecutor:
    """构建带双工具的 Agent（不含 memory，memory 由调用方注入）。"""
    llm = load_llm()
    tools = [make_retrieval_tool(vectorstore), make_web_search_tool()]

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)


def chat(agent: AgentExecutor, session_id: str, message: str) -> str:
    """
    带记忆的单次对话：
    1. 读取该 session 的历史消息
    2. 调用 Agent
    3. 将本轮对话写入记忆
    4. 返回答案字符串
    """
    history = memory_store.get(session_id)
    result = agent.invoke({"input": message, "chat_history": history})
    answer = result["output"]
    memory_store.add(session_id, message, answer)
    return answer
