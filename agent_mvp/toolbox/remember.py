"""记忆工具：由 Agent 注入记忆访问器，模型自主决定持久化。"""

_memory_persist_hook = None


def set_memory_persist_hook(fn):
    """注入一个 callable(content, importance) -> bool，供 remember 工具调用。"""
    global _memory_persist_hook
    _memory_persist_hook = fn


def remember(content: str, importance: float = 0.8) -> str:
    """把用户信息写入长期记忆（是否调用由模型判断）。"""
    if not _memory_persist_hook:
        return "错误: 记忆系统未接入"
    ok = _memory_persist_hook(content, float(importance))
    if ok:
        return f"已记住: {content}"
    return f"未通过记忆治理裁决，未持久化: {content}"