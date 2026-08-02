"""记忆条目（MemoryItem）数据模型定义。

包含记忆类型枚举 MemoryType 以及单条记忆的数据结构 MemoryItem，
用于在记忆系统中表示一条具体的记忆及其元数据、评分、版本等信息。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MemoryType(str, Enum):
    """记忆类型枚举，标识记忆的作用范围。"""

    WORKING = "working"  # 工作记忆：当前任务过程中的临时记忆
    SESSION = "session"  # 会话记忆：一次会话范围内的记忆
    LONG_TERM = "long_term"  # 长期记忆：跨会话持久化的长期记忆


@dataclass
class MemoryItem:
    """单条记忆的数据结构。

    存储一条记忆的完整信息，包括内容、元数据、创建/更新时间、
    记忆类型、相关评分、版本号以及所属用户和会话。
    """

    memory_id: str  # 记忆唯一标识
    content: str  # 记忆内容文本
    metadata: Dict[str, Any] = field(default_factory=dict)  # 记忆的扩展元数据
    created_at: Optional[datetime] = None  # 创建时间
    updated_at: Optional[datetime] = None  # 最近更新时间
    memory_type: MemoryType = MemoryType.WORKING  # 记忆类型，默认工作记忆
    score: float = 0.0  # 记忆相关度评分
    version: int = 1  # 记忆版本号
    user_id: str = ""  # 所属用户 ID
    session_id: Optional[str] = None  # 所属会话 ID

    def __post_init__(self):
        """初始化后填充默认时间戳：未指定时使用当前时间。"""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
