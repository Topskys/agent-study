import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models.event import Event

"""事件流存储引擎：以有序列表保存事件对象（Event），支持按时间范围、
事件类型过滤查询以及按类型计数。

传入 db_path 时启用 SQLite 持久化（写穿透）：事件在内存与磁盘间
保持同步，重启后可恢复；未传 db_path 则纯内存运行。
"""


class EventStreamStore:
    """事件流存储：以有序列表保存事件，追加事件即记录；查询时在内存中对
    事件列表进行时间范围、事件类型过滤，并支持结果数量限制。"""

    def __init__(
        self, store_id: str = "default_event_stream", db_path: Optional[str] = None
    ):
        """初始化存储：记录 store_id 并创建空的事件列表。"""
        self.store_id = store_id
        self.db_path = db_path
        self._events: List[Event] = []
        if self.db_path:
            self._init_db()
            self._load_all()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """创建持久化所需的事件数据表（若不存在）。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_stream (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _load_all(self):
        """启动时从 SQLite 将全部事件载入内存。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT event_id, event_type, payload, timestamp FROM event_stream"
        )
        for event_id, event_type, payload, ts in cursor.fetchall():
            self._events.append(
                Event(
                    event_id=event_id,
                    event_type=event_type,
                    payload=json.loads(payload),
                    timestamp=datetime.fromisoformat(ts) if ts else None,
                )
            )
        conn.close()

    def add_event(self, event: Event):
        """向事件流末尾追加一条事件。"""
        self._events.append(event)
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO event_stream "
                "(event_id, event_type, payload, timestamp) VALUES (?, ?, ?, ?)",
                (
                    event.event_id,
                    event.event_type,
                    json.dumps(event.payload, ensure_ascii=False),
                    event.timestamp.isoformat() if event.timestamp else None,
                ),
            )
            conn.commit()
            conn.close()

    def get_all_events(self) -> List[Event]:
        """返回全部事件的拷贝。"""
        return list(self._events)

    def query_events(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """按条件查询事件：可同时按时间范围（[start, end] 闭区间）和事件类型
        过滤，最后最多返回 limit 条；条件缺省时不作对应过滤。"""
        results = list(self._events)
        if time_range:
            start, end = time_range
            # 按时间范围过滤，事件无时间戳时按 datetime.min 处理（保证落在区间内）
            results = [
                e for e in results if start <= (e.timestamp or datetime.min) <= end
            ]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return results[:limit]

    def count_by_type(self, event_type: str) -> int:
        """统计指定事件类型的出现次数。"""
        return sum(1 for e in self._events if e.event_type == event_type)

    def clear(self):
        """清空全部事件。"""
        self._events.clear()
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM event_stream")
            conn.commit()
            conn.close()
