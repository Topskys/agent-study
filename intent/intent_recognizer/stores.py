"""数据访问层：直连 agent_memory.db 现有记忆模块数据表。

对齐 design/Agent意图识别设计方案v3.md §8 数据表对接设计：
- MemoryStore  -> memory_items   （intent_cache / slot_cache / long_term / session 读）
- ProfileStore -> user_profiles
- EventStore   -> event_stream
- KvStore      -> kv_items

写入约定：长期记忆写入仍走 memory-system 治理裁决（UserMemory.add_long_term_memory），
本层只直接写内部检查点（intent_cache / slot_cache）与审计事件，不绕过记忆治理。
db_path 为 None 时所有读写退化为空操作（无持久化）。

注意：若目标库缺少 memory-system 建的表（如全新测试库），各 Store 会用
CREATE TABLE IF NOT EXISTS 补齐同构表结构；对现有 agent_memory.db 无副作用。
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import FirstStageResult

_INTENT_CACHE = "intent_cache"
_SLOT_CACHE = "slot_cache"


def _now() -> str:
    return datetime.now().isoformat()


def _loads(raw: Optional[str]) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _connect(db_path: Optional[str]) -> Optional[sqlite3.Connection]:
    if not db_path:
        return None
    return sqlite3.connect(db_path)


class MemoryStore:
    """memory_items 读写：长期 / 会话 / 意图检查点 / 槽位缓存。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path) if db_path else None
        if self.db_path:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    version INTEGER NOT NULL DEFAULT 1,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    embedding TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- 读 ----------

    def read_memories(
        self, user_id: str, memory_type: str, limit: int = 20
    ) -> List[dict]:
        conn = _connect(self.db_path)
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT memory_id, content, metadata, memory_type, score, created_at "
                "FROM memory_items WHERE user_id=? AND memory_type=? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, memory_type, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "memory_id": r[0],
                "content": r[1],
                "metadata": _loads(r[2]),
                "memory_type": r[3],
                "score": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def read_long_term(self, user_id: str, limit: int = 20) -> List[dict]:
        return self.read_memories(user_id, "long_term", limit)

    def read_session(self, user_id: str, limit: int = 20) -> List[dict]:
        return self.read_memories(user_id, "session", limit)

    def read_intent_cache(
        self, user_id: str, session_id: str
    ) -> Optional[FirstStageResult]:
        data = self._read_single(self._intent_cache_id(user_id, session_id))
        if not data:
            return None
        return FirstStageResult.from_dict(data)

    def read_slot_cache(self, user_id: str, session_id: str) -> Dict[str, Any]:
        data = self._read_single(self._slot_cache_id(user_id, session_id))
        if not isinstance(data, dict):
            return {}
        return data.get("slots", {})

    def _read_single(self, memory_id: str) -> Optional[Any]:
        conn = _connect(self.db_path)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT content FROM memory_items WHERE memory_id=?", (memory_id,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return _loads(row[0])

    # ---------- 写（检查点） ----------

    def write_intent_cache(
        self, user_id: str, session_id: str, result: FirstStageResult
    ) -> None:
        self._write_cache(
            self._intent_cache_id(user_id, session_id),
            json.dumps(result.to_dict(), ensure_ascii=False),
            {"importance": 0.5, "role": "intent", "source": result.source},
            _INTENT_CACHE,
            user_id,
            session_id,
        )

    def write_slot_cache(
        self, user_id: str, session_id: str, slots: Dict[str, Any]
    ) -> None:
        self._write_cache(
            self._slot_cache_id(user_id, session_id),
            json.dumps({"slots": slots}, ensure_ascii=False),
            {"importance": 0.5, "role": "intent", "source": "slot_extracted"},
            _SLOT_CACHE,
            user_id,
            session_id,
        )

    def _write_cache(
        self,
        memory_id: str,
        content: str,
        metadata: dict,
        memory_type: str,
        user_id: str,
        session_id: str,
    ) -> None:
        conn = _connect(self.db_path)
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT OR REPLACE INTO memory_items "
                "(memory_id, content, metadata, created_at, updated_at, memory_type, "
                "score, version, user_id, session_id, embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory_id,
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                    _now(),
                    _now(),
                    memory_type,
                    0.0,
                    1,
                    user_id,
                    session_id,
                    "[]",
                ),
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _intent_cache_id(user_id: str, session_id: str) -> str:
        return f"intent_cache:{user_id}:{session_id}"

    @staticmethod
    def _slot_cache_id(user_id: str, session_id: str) -> str:
        return f"slot_cache:{user_id}:{session_id}"


class ProfileStore:
    """user_profiles 读：返回用户上下文供消歧 / 槽位默认值。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path) if db_path else None
        if self.db_path:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    base_info TEXT NOT NULL DEFAULT '{}',
                    preferences TEXT NOT NULL DEFAULT '{}',
                    scene_profiles TEXT NOT NULL DEFAULT '{}',
                    config TEXT NOT NULL DEFAULT '{}',
                    lock_keys TEXT NOT NULL DEFAULT '[]',
                    current_version INTEGER NOT NULL DEFAULT 1,
                    last_updated REAL,
                    embedding TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        conn = _connect(self.db_path)
        if conn is None:
            return {}
        try:
            row = conn.execute(
                "SELECT base_info, preferences, scene_profiles, config "
                "FROM user_profiles WHERE user_id=?",
                (user_id,),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return {}
        return {
            "base_info": _loads(row[0]),
            "preferences": _loads(row[1]),
            "scene_profiles": _loads(row[2]),
            "config": _loads(row[3]),
        }


class EventStore:
    """event_stream 写：统一审计入口。"""

    EVENT_INTENT_RECOGNIZED = "intent_recognized"
    EVENT_ASK_PROMPT = "ask_prompt"
    EVENT_SLOT_EXTRACTED = "slot_extracted"
    EVENT_TASK_SCHEDULED = "task_scheduled"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path) if db_path else None
        if self.db_path:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_stream (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    timestamp TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def record(self, event_type: str, payload: dict) -> None:
        conn = _connect(self.db_path)
        if conn is None:
            return
        try:
            conn.execute(
                "INSERT INTO event_stream (event_id, event_type, payload, timestamp) "
                "VALUES (?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    event_type,
                    json.dumps(payload, ensure_ascii=False),
                    _now(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list(self, user_id: str, limit: int = 20) -> List[dict]:
        conn = _connect(self.db_path)
        if conn is None:
            return []
        pattern = f'%"{user_id}"%'
        try:
            rows = conn.execute(
                "SELECT event_id, event_type, payload, timestamp FROM event_stream "
                "WHERE payload LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "event_id": r[0],
                "event_type": r[1],
                "payload": _loads(r[2]),
                "timestamp": r[3],
            }
            for r in rows
        ]


class KvStore:
    """kv_items 读写：规则清单 / 配置覆盖，支持 TTL。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path) if db_path else None
        if self.db_path:
            self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_items (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at REAL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> Optional[Any]:
        conn = _connect(self.db_path)
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT value, expires_at FROM kv_items WHERE key=?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        value, expires_at = row
        if expires_at is not None and expires_at <= time.time():
            return None
        return _loads(value)

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        conn = _connect(self.db_path)
        if conn is None:
            return
        expires_at = (time.time() + ttl) if ttl is not None else None
        try:
            conn.execute(
                "INSERT OR REPLACE INTO kv_items (key, value, expires_at) VALUES (?,?,?)",
                (key, json.dumps(value, ensure_ascii=False), expires_at),
            )
            conn.commit()
        finally:
            conn.close()

    def delete(self, key: str) -> None:
        conn = _connect(self.db_path)
        if conn is None:
            return
        try:
            conn.execute("DELETE FROM kv_items WHERE key=?", (key,))
            conn.commit()
        finally:
            conn.close()
