"""事件（Event）数据模型定义。

定义记忆系统中的事件数据结构，用于表示一次事件及其类型、
载荷数据和发生时间。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Event:
    """事件数据结构。

    表示记忆系统中的一个事件，包含事件 ID、类型、载荷数据
    以及发生时间戳。
    """

    event_id: str  # 事件唯一标识
    event_type: str  # 事件类型
    payload: Dict[str, Any] = field(default_factory=dict)  # 事件的载荷数据
    timestamp: Optional[datetime] = None  # 事件发生时间

    def __post_init__(self):
        """初始化后填充默认时间戳：未指定时使用当前时间。"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
