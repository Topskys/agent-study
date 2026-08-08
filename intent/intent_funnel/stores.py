"""会话状态持久化层（可选 sqlite）。

V3 §八：会话快照持久化，多实例共享槽位缓存与检查点。
db_path 为 None 时退化为内存存储。用例存在 sqlite 表 funnel_session_states。
"""

import json
import sqlite3
from datetime import datetime, timezone

from .models import DialogSessionState, IntentResult


class SessionStore:
    """会话状态存储：内存或 sqlite。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path
        self._memory: dict[tuple[str, str], DialogSessionState] = {}
        if db_path:
            self._init_db()

    # ---------- 建表 ----------

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS funnel_session_states (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- 读写 ----------

    def get(self, user_id: str, session_id: str) -> DialogSessionState:
        conn = self._connect()
        if conn is None:
            return self._memory.get(
                (user_id, session_id),
                DialogSessionState(user_id=user_id, session_id=session_id),
            )
        try:
            row = conn.execute(
                "SELECT state FROM funnel_session_states WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            return DialogSessionState(user_id=user_id, session_id=session_id)
        return self._decode_payload(user_id, session_id, json.loads(row[0]))

    def set(self, state: DialogSessionState) -> None:
        payload = {
            "last_intent_text": state.last_intent_text,
            "filled_slots": state.filled_slots,
            "miss_slots": state.miss_slots,
            "historical_queries": state.historical_queries,
            "checkpoint": state.checkpoint.to_dict() if state.checkpoint else None,
        }
        conn = self._connect()
        if conn is None:
            self._memory[(state.user_id, state.session_id)] = state
            return
        try:
            conn.execute(
                "INSERT OR REPLACE INTO funnel_session_states "
                "(user_id, session_id, state, updated_at) VALUES (?,?,?,?)",
                (
                    state.user_id,
                    state.session_id,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ---------- 内部 ----------

    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path:
            return None
        return sqlite3.connect(self.db_path)

    @classmethod
    def _decode_payload(
        cls, user_id: str, session_id: str, d: dict
    ) -> DialogSessionState:
        state = DialogSessionState(user_id=user_id, session_id=session_id)
        state.last_intent_text = str(d.get("last_intent_text", ""))
        state.filled_slots = dict(d.get("filled_slots", {}))
        state.miss_slots = list(d.get("miss_slots", []))
        state.historical_queries = list(d.get("historical_queries", []))
        cp = d.get("checkpoint")
        state.checkpoint = IntentResult.from_dict(cp) if cp else None
        return state