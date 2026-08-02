"""工作记忆模块。

负责管理智能体在短期任务执行过程中的"工作记忆"：
采用有容量上限的 FIFO（先进先出）队列，当内容总量超过 token 上限时，
自动从队头（最早加入的条目）开始裁剪，保证内存占用不超过设定阈值。
"""

from typing import List

from ..models.memory_item import MemoryItem, MemoryType


class WorkingMemory:
    """工作记忆容器。

    以列表形式保存当前工作期的记忆条目，并基于近似 token 数进行
    超容量裁剪（FIFO），是最短生命周期的一层记忆。
    """

    def __init__(self, max_tokens: int = 4096):
        # 最大允许的近似 token 总数，超出即触发裁剪
        self.max_tokens = max_tokens
        # 内部存储的记忆条目列表（队头为最早加入的条目）
        self._items: List[MemoryItem] = []

    def add_item(self, item: MemoryItem):
        """向工作记忆追加一条内容，并触发超容量裁剪。"""
        self._items.append(item)
        self.trim_to_max_tokens()

    def get_items(self) -> List[MemoryItem]:
        """返回全部工作记忆条目的副本（不修改内部列表）。"""
        return list(self._items)

    def trim_to_max_tokens(self):
        """按 FIFO 顺序裁剪超容量条目。

        以各条目内容的字符长度近似 token 数进行累加；当总量超过
        max_tokens 时，持续从队头弹出最早的条目，直到总量不超限。
        """
        total = sum(len(i.content) for i in self._items)
        while total > self.max_tokens and self._items:
            # 移除队头（最早）的条目并同步扣减总量
            removed = self._items.pop(0)
            total -= len(removed.content)

    def clear(self):
        """清空所有工作记忆条目。"""
        self._items.clear()

    def current_tokens(self) -> int:
        """返回当前内容的近似 token 总数（按内容字符长度估算）。"""
        return sum(len(i.content) for i in self._items)

    def size(self) -> int:
        """返回当前工作记忆中的条目数量。"""
        return len(self._items)
