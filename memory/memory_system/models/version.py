"""记忆版本（MemoryVersion）数据模型定义。

定义一条记忆历史版本的快照数据结构，用于支持记忆的版本管理与回滚。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MemoryVersion:
    """记忆历史版本快照。

    记录某条记忆在某一版本的完整内容以及对应的创建时间。
    """

    version_id: str  # 版本唯一标识
    memory_id: str  # 所属记忆的 ID
    content: str  # 该版本下的记忆内容快照
    created_at: Optional[datetime] = None  # 版本创建时间

    def __post_init__(self):
        """初始化后填充默认时间戳：未指定时使用当前时间。"""
        if self.created_at is None:
            self.created_at = datetime.now()
