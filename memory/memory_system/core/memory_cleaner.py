"""记忆清理（MemoryCleaner）治理模块。

负责对记忆库进行清理维护，包括按 TTL 过期清理、按内容去重、
清理低分记忆以及按容量上限裁剪，保持记忆库精炼有效。
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from ..models.memory_item import MemoryItem


class MemoryCleaner:
    """记忆清理器：执行过期清理、去重、低分清理与容量控制。"""

    def __init__(self, ttl_config: Optional[Dict[str, int]] = None):
        # TTL 配置：按记忆类型设置存活时间（秒），未配置时使用默认值
        self.ttl_config = ttl_config or {
            "working": 3600,  # 工作记忆：1 小时
            "session": 86400,  # 会话记忆：1 天
            "long_term": 2592000,  # 长期记忆：30 天
        }

    def clean_expired(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """按 TTL 清除过期记忆。

        以当前时间为基准，根据记忆类型查表得到对应 TTL（秒），
        当记忆的更新时间距今超过 TTL 时判定为过期并移除；
        无更新时间的记忆视为不过期，予以保留。

        :param items: 待清理的记忆列表
        :return: 清除过期项后剩余的记忆列表
        """
        now = time.time()
        keep = []
        for item in items:
            # 按记忆类型取 TTL，枚举类型兼容 value 字符串形式，默认 1 天
            ttl = self.ttl_config.get(
                item.memory_type.value
                if hasattr(item.memory_type, "value")
                else item.memory_type,
                86400,
            )
            if item.updated_at:
                age = now - item.updated_at.timestamp()
                if age < ttl:
                    keep.append(item)
            else:
                keep.append(item)
        return keep

    def clean_duplicates(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """按内容去重，保留评分更高的重复项。

        以（user_id, 去空格后的内容小写）为唯一键识别重复记忆；
        先按评分从高到低排序，评分高者优先保留，相同键只保留一条。

        :param items: 待去重的记忆列表
        :return: 去重后的记忆列表
        """
        seen: Set[str] = set()
        unique = []
        # 按评分降序处理，保证评分高的记录先被保留
        for item in sorted(items, key=lambda x: x.score, reverse=True):
            key = (item.user_id, item.content.strip().lower())
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def clean_low_score(
        self, items: List[MemoryItem], threshold: float = 0.2
    ) -> List[MemoryItem]:
        """清理评分低于阈值的低分记忆。

        :param items: 待清理的记忆列表
        :param threshold: 评分阈值，默认 0.2
        :return: 仅保留评分不低于阈值的记忆列表
        """
        return [it for it in items if it.score >= threshold]

    def enforce_capacity(
        self, items: List[MemoryItem], max_count: int
    ) -> List[MemoryItem]:
        """按最大容量裁剪记忆列表。

        当记忆数量超过上限时，按评分降序保留评分最高的前 max_count 条。

        :param items: 待裁剪的记忆列表
        :param max_count: 容量上限
        :return: 裁剪后的记忆列表
        """
        if len(items) <= max_count:
            return items
        sorted_items = sorted(items, key=lambda x: x.score, reverse=True)
        return sorted_items[:max_count]
