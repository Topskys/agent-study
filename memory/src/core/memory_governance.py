"""记忆治理（MemoryGovernance）模块。

作为记忆系统的核心治理入口，组合过滤（MemoryFilter）、评分（MemoryScorer）、
清理（MemoryCleaner）、版本管理（MemoryVersioner）等子模块，
统一负责记忆的入库裁决、冲突检测、长期记忆准入与周期维护。
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.memory_item import MemoryItem, MemoryType
from ..stores.vector_store import VectorStore
from .memory_cleaner import MemoryCleaner
from .memory_filter import MemoryFilter
from .memory_scorer import MemoryScorer
from .memory_versioner import MemoryVersioner


# 互斥规则表：每条规则定义两个互斥的表达及冲突类别标签，
# 当长期记忆中同时出现同一条规则的两面时判定为记忆冲突。
# 格式：("表述A", "表述B", "冲突类型标签")
CONFLICT_RULES: List[Tuple[str, str, str]] = [
    ("爱吃辣", "忌辣", "food_spicy"),  # 饮食偏好冲突：吃辣与忌辣
    ("喜欢吵闹", "喜欢安静", "environment"),  # 环境偏好冲突：吵闹与安静
    ("外向", "内向", "personality"),  # 性格冲突：外向与内向
    ("喜欢白天", "喜欢夜晚", "active_time"),  # 作息时间冲突：白天与夜晚
]


class MemoryGovernance:
    """记忆治理器：统一协调各治理子模块完成记忆生命周期管理。"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        memory_filter: Optional[MemoryFilter] = None,
        scorer: Optional[MemoryScorer] = None,
        cleaner: Optional[MemoryCleaner] = None,
        versioner: Optional[MemoryVersioner] = None,
        long_term_threshold: float = 0.5,
    ):
        # 向量存储（预留扩展，用于语义检索）
        self.vector_store = vector_store
        # 治理子模块：未显式传入时使用默认实例
        self.filter = memory_filter or MemoryFilter()
        self.scorer = scorer or MemoryScorer()
        self.cleaner = cleaner or MemoryCleaner()
        self.versioner = versioner or MemoryVersioner()
        # 进入长期记忆的评分阈值
        self.long_term_threshold = long_term_threshold

    def should_enter_long_term(self, item: MemoryItem) -> bool:
        """判断一条记忆是否具备进入长期记忆的资格。

        先经过记忆过滤（类型/长度/关键词），再计算评分，
        只有评分不低于 long_term_threshold 才允许进入长期记忆。

        :param item: 待判断的记忆条目
        :return: True 表示允许进入长期记忆
        """
        # 未通过基础过滤直接拒绝
        if not self.filter.filter_item(item):
            return False
        # 计算评分并回写到条目，作为后续判定依据
        item.score = self.scorer.score(item)
        return item.score >= self.long_term_threshold

    def score_memory(self, item: MemoryItem) -> float:
        """计算并返回单条记忆的综合评分。

        :param item: 待评分的记忆条目
        :return: 综合评分（float）
        """
        return self.scorer.score(item)

    def clean_expired_memories(
        self, items: List[MemoryItem], user_id: str = ""
    ) -> List[MemoryItem]:
        """清理过期、重复与低分的记忆。

        :param items: 待清理的记忆列表
        :param user_id: 用户 ID（当前实现未使用，保留为扩展参数）
        :return: 清理后的记忆列表
        """
        # 依次执行过期清理、内容去重、低分清理
        items = self.cleaner.clean_expired(items)
        items = self.cleaner.clean_duplicates(items)
        items = self.cleaner.clean_low_score(items)
        return items

    def version_memory(self, item: MemoryItem) -> str:
        """为记忆创建版本快照并返回版本 ID。

        :param item: 待创建版本快照的记忆条目
        :return: 新版本的 version_id
        """
        version = self.versioner.create_version(item)
        return version.version_id

    def detect_conflicts(self, items: List[MemoryItem]) -> List[Dict[str, Any]]:
        """在长期记忆中检测互斥规则冲突。

        提取所有长期记忆的内容（统一去空格转小写），
        遍历 CONFLICT_RULES：若同一规则的两面表述同时出现，
        则记录一条冲突。

        :param items: 待检测的记忆列表
        :return: 冲突列表，每项含 type（类别标签）、statements（两表述）、severity（严重级别）
        """
        conflicts = []
        # 仅对长期记忆内容做冲突检测
        contents = [
            it.content.strip().lower()
            for it in items
            if it.memory_type == MemoryType.LONG_TERM
        ]

        # 逐一检查互斥规则，若正反两表述同时存在则判定冲突
        for a_text, b_text, label in CONFLICT_RULES:
            has_a = any(a_text in c for c in contents)
            has_b = any(b_text in c for c in contents)
            if has_a and has_b:
                conflicts.append(
                    {
                        "type": label,
                        "statements": [a_text, b_text],
                        "severity": "warning",
                    }
                )

        return conflicts

    def adjudicate_entry(self, item: MemoryItem, existing: List[MemoryItem]) -> bool:
        """裁决一条记忆能否正式入库（长期记忆）。

        入库需同时满足两个条件：
        1. 通过 should_enter_long_term 的资格判定（过滤 + 评分达标）；
        2. 与现有记忆合并检测后不触发互斥规则冲突。

        :param item: 待入库的记忆条目
        :param existing: 已有的记忆列表
        :return: True 表示允许入库
        """
        # 条件一：未达长期记忆资格直接拒绝
        if not self.should_enter_long_term(item):
            return False

        # 条件二：与现有记忆存在互斥冲突则拒绝
        conflicts = self.detect_conflicts(existing + [item])
        if conflicts:
            return False

        # 全部通过后再次确认评分并允许入库
        item.score = self.scorer.score(item)
        return True

    def perform_maintenance(self, items: List[MemoryItem]) -> List[MemoryItem]:
        """执行周期维护，保持记忆库精炼。

        依次执行过期清理、内容去重与低分清理；
        其中低分清理阈值取长期记忆阈值的 50%，相对宽松。

        :param items: 待维护的记忆列表
        :return: 维护后的记忆列表
        """
        items = self.cleaner.clean_expired(items)
        items = self.cleaner.clean_duplicates(items)
        # 低分清理阈值取长期记忆阈值的 50%
        items = self.cleaner.clean_low_score(
            items, threshold=self.long_term_threshold * 0.5
        )
        return items
