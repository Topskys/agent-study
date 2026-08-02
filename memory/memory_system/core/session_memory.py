"""会话记忆模块。

管理一次独立对话会话（session）期间的记忆：记录会话内产生的记忆条目，
并附带一个可自由读写的会话上下文（如用户偏好、话题状态等临时信息）。
会话结束时，将本会话内的记忆条目整体沉淀（返回）给上层处理。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.memory_item import MemoryItem, MemoryType


class SessionMemory:
    """单次会话的记忆容器。

    绑定一个会话 ID，保存该会话内的记忆条目与上下文信息；
    生命周期从会话开始到会话结束（end_session）。
    """

    def __init__(self, session_id: str):
        # 所属会话的唯一标识
        self.session_id = session_id
        # 会话期间产生的记忆条目列表
        self._items: List[MemoryItem] = []
        # 会话上下文（键值对形式的临时信息）
        self.session_context: Dict[str, Any] = {}

    def add_item(self, item: MemoryItem):
        """写入一条会话记忆：绑定当前会话 ID 并标记为 SESSION 类型。"""
        item.session_id = self.session_id
        item.memory_type = MemoryType.SESSION
        self._items.append(item)

    def get_items(self) -> List[MemoryItem]:
        """返回本会话全部记忆条目的副本。"""
        return list(self._items)

    def get_session_context(self) -> Dict[str, Any]:
        """返回会话上下文的副本。"""
        return dict(self.session_context)

    def update_context(self, key: str, value: Any):
        """写入/更新会话上下文中的某个键值。"""
        self.session_context[key] = value

    def end_session(self) -> List[MemoryItem]:
        """结束当前会话。

        返回本会话内全部记忆条目（即"沉淀项"，交由上层决定是否
        进入长期记忆），并清空条目列表与会话上下文。
        """
        items = list(self._items)
        self._items.clear()
        self.session_context.clear()
        return items

    def size(self) -> int:
        """返回本会话当前的记忆条目数量。"""
        return len(self._items)
