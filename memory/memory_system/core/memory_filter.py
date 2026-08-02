"""记忆过滤（MemoryFilter）治理模块。

负责对记忆条目进行准入过滤：根据记忆类型白名单、内容长度限制
以及敏感/屏蔽关键词，判断一条记忆是否允许进入记忆系统。
"""

from typing import List

from ..models.memory_item import MemoryItem, MemoryType


class MemoryFilter:
    """记忆过滤器：基于类型、内容长度与关键词规则过滤记忆条目。"""

    def __init__(
        self,
        allowed_types: List[MemoryType] = None,
        blocked_keywords: List[str] = None,
        max_content_length: int = 10000,
        min_content_length: int = 1,
    ):
        # 允许的记忆类型集合，未指定时默认放行所有类型
        self.allowed_types = allowed_types or list(MemoryType)
        # 屏蔽关键词列表（大小写不敏感匹配）
        self.blocked_keywords = blocked_keywords or []
        # 内容最大/最小长度限制
        self.max_content_length = max_content_length
        self.min_content_length = min_content_length

    def filter_item(self, item: MemoryItem) -> bool:
        """判断单条记忆是否通过过滤。

        :param item: 待过滤的记忆条目
        :return: True 表示允许进入记忆系统，False 表示被拒绝
        """
        # 记忆类型不在白名单内则拒绝
        if item.memory_type not in self.allowed_types:
            return False

        content = item.content.strip()
        # 内容长度不满足最小/最大限制则拒绝
        if len(content) < self.min_content_length:
            return False
        if len(content) > self.max_content_length:
            return False

        # 内容包含任一屏蔽关键词则拒绝（统一转小写后匹配）
        lower_content = content.lower()
        for kw in self.blocked_keywords:
            if kw.lower() in lower_content:
                return False

        return True
