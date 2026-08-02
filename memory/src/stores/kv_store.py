import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

"""键值存储引擎：内存实现的简单键值数据库（写穿透到 SQLite），
额外支持 TTL（过期时间）管理，用于存储轻量级、可快速读写的键值数据。

传入 db_path 时启用 SQLite 持久化：键值对与 TTL 在内存和磁盘间
保持同步，重启后可恢复；未传 db_path 则纯内存运行。
"""


class KeyValueStore:
    """内存键值存储：以 dict 保存键值对，另用一个 dict 记录每个键的过期
    时间戳；TTL 采用惰性过期策略——只在读写时清理已过期的键，
    不依赖后台定时任务。"""

    def __init__(self, store_id: str = "default_kv", db_path: Optional[str] = None):
        """初始化存储：记录 store_id，并创建数据字典与 TTL 字典。"""
        self.store_id = store_id
        self.db_path = db_path
        self._data: Dict[str, Any] = {}
        self._ttl: Dict[str, float] = {}
        if self.db_path:
            self._init_db()
            self._load_all()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """创建持久化所需的键值数据表（若不存在）。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_items (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                expires_at REAL
            )
        """)
        conn.commit()
        conn.close()

    def _load_all(self):
        """启动时从 SQLite 将全部键值对载入内存。"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, expires_at FROM kv_items")
        now = time.time()
        for key, raw, expires_at in cursor.fetchall():
            if expires_at is not None and expires_at <= now:
                continue
            try:
                value = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                value = raw
            self._data[key] = value
            if expires_at is not None:
                self._ttl[key] = expires_at
        conn.close()

    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """写入一个键值对；若提供 ttl（秒），则记录其绝对过期时间戳。"""
        self._data[key] = value
        expires_at = None
        if ttl is not None:
            # 计算过期绝对时间：当前时间 + ttl
            expires_at = time.time() + ttl
            self._ttl[key] = expires_at
        elif key in self._ttl:
            # 未提供 ttl 时清除该键已有的过期时间（永不过期）
            del self._ttl[key]

        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO kv_items (key, value, expires_at) "
                "VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), expires_at),
            )
            conn.commit()
            conn.close()

    def get(self, key: str) -> Optional[Any]:
        """读取键对应的值；读取前先清理过期键，不存在时返回 None。"""
        self._evict_expired()
        return self._data.get(key)

    def delete(self, key: str):
        """删除键及其 TTL 记录。"""
        self._data.pop(key, None)
        self._ttl.pop(key, None)
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM kv_items WHERE key = ?", (key,))
            conn.commit()
            conn.close()

    def exists(self, key: str) -> bool:
        """判断键是否存在（先清理过期键，已过期的视为不存在）。"""
        self._evict_expired()
        return key in self._data

    def keys(self, pattern: Optional[str] = None) -> List[str]:
        """返回全部键；若提供 pattern，则只返回包含该子串的键。"""
        self._evict_expired()
        if pattern is None:
            return list(self._data.keys())
        return [k for k in self._data if pattern in k]

    def get_all(self) -> Dict[str, Any]:
        """返回全部键值对的拷贝（先清理过期键）。"""
        self._evict_expired()
        return dict(self._data)

    def clear(self):
        """清空全部键值对与 TTL 记录。"""
        self._data.clear()
        self._ttl.clear()
        if self.db_path:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM kv_items")
            conn.commit()
            conn.close()

    def size(self) -> int:
        """返回当前有效键的数量（先清理过期键）。"""
        self._evict_expired()
        return len(self._data)

    def _evict_expired(self):
        """惰性过期清理：找出所有已到期的键并删除对应的数据与 TTL 记录，
        在每次读取类操作前调用。"""
        now = time.time()
        # 收集所有已过期（过期时间 <= 当前时间）的键
        expired = [k for k, t in self._ttl.items() if t <= now]
        for k in expired:
            self._data.pop(k, None)
            self._ttl.pop(k, None)
            if self.db_path:
                conn = self._connect()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM kv_items WHERE key = ?", (k,))
                conn.commit()
                conn.close()
