"""
每个 session 独立维护对话历史。
存储 HumanMessage / AIMessage 列表，传给 Agent 的 chat_history。
"""

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class SessionMemory:
    def __init__(self, max_turns: int = 10):
        """
        max_turns: 保留最近 N 轮对话（每轮 = 1 human + 1 ai），防止 context 过长。
        """
        self._store: dict[str, list[BaseMessage]] = {}
        self._max_turns = max_turns

    def get(self, session_id: str) -> list[BaseMessage]:
        return self._store.get(session_id, [])

    def add(self, session_id: str, human_msg: str, ai_msg: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(HumanMessage(content=human_msg))
        self._store[session_id].append(AIMessage(content=ai_msg))
        # 只保留最近 max_turns 轮（每轮 2 条）
        max_msgs = self._max_turns * 2
        if len(self._store[session_id]) > max_msgs:
            self._store[session_id] = self._store[session_id][-max_msgs:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def history_summary(self, session_id: str) -> list[dict]:
        """返回可序列化的历史摘要，用于 API 响应。"""
        return [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in self._store.get(session_id, [])
        ]


# 全局单例，被 routes.py 和 agent.py 共用
memory_store = SessionMemory(max_turns=10)
