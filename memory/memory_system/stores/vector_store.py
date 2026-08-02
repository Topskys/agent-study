import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..models.memory_item import MemoryItem, MemoryType
from ..models.user_profile import UserProfile
from ..models.version import MemoryVersion
from ..utils.embeddings import cosine_similarity, generate_embedding

"""向量存储引擎：基于 SQLite 持久化记忆数据（记忆条目、用户画像、记忆版本），
并提供基于余弦相似度的向量检索能力，用于按语义相关度召回记忆。"""


class VectorIndex:
    """向量索引描述对象：用于声明某个字段（或字段组合）可以被建立索引，
    目前仅保存索引的名称与涉及列，实际索引结构由具体的存储实现负责。"""

    def __init__(self, name: str, columns: List[str]):
        """初始化一个索引描述，指定索引名称及其覆盖的列。"""
        self.name = name
        self.columns = columns


class VectorStore:
    """记忆向量存储：使用 SQLite 实现，将记忆条目、用户画像与记忆版本分别
    存入三张表；检索时先在 SQL 层按 user_id 过滤，再加载全部候选行到内存，
    计算查询向量与记忆嵌入向量的余弦相似度后按分数降序取 top_k。"""

    def __init__(self, db_path: str = "memory_system.db"):
        """初始化存储：记录数据库路径并调用 _init_db 确保建表完成。"""
        self.store_id = db_path
        self.db_path = db_path
        self.indices: List[str] = ["user_id", "memory_type", "created_at"]
        self._init_db()

    def _init_db(self):
        """初始化数据库：连接 SQLite 并创建所需的全部数据表（若不存在）。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # memory_items 表：存储核心记忆数据（内容、元数据、时间戳、类型、分数、版本、用户与嵌入向量）
        cursor.execute("""
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
        """)

        # user_profiles 表：存储用户的画像信息（基础信息、偏好、场景画像、配置与锁定的键）
        cursor.execute("""
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
        """)

        # memory_versions 表：存储记忆的历史版本，用于追溯内容变更记录
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def insert_memory(self, memory: MemoryItem):
        """插入或覆盖一条记忆：写入 memory_items 表。

        入库时若 metadata 中尚无嵌入向量，则自动对内容计算真实语义嵌入，
        使后续语义检索（余弦相似度）真正生效。
        """
        if not memory.metadata.get("embedding"):
            try:
                memory.metadata["embedding"] = generate_embedding(memory.content)
            except Exception:
                # 嵌入计算失败时降级：无向量入库，检索回退到 score 字段
                memory.metadata.pop("embedding", None)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO memory_items
            (memory_id, content, metadata, created_at, updated_at, memory_type,
             score, version, user_id, session_id, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                memory.memory_id,
                memory.content,
                json.dumps(memory.metadata),
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                # memory_type 可能为枚举对象，取其 value 以字符串形式入库
                memory.memory_type.value
                if hasattr(memory.memory_type, "value")
                else memory.memory_type,
                memory.score,
                memory.version,
                memory.user_id,
                memory.session_id,
                # 嵌入向量内嵌在 metadata 中，单独提取并 JSON 序列化存储
                json.dumps(memory.metadata.get("embedding", [])),
            ),
        )
        conn.commit()
        conn.close()

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """按 memory_id 查询单条记忆；不存在时返回 None。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_items WHERE memory_id = ?", (memory_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return self._row_to_memory_item(row)
        return None

    def delete_memory(self, memory_id: str):
        """按 memory_id 删除一条记忆。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memory_items WHERE memory_id = ?", (memory_id,))
        conn.commit()
        conn.close()

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """更新记忆内容与元数据，同时刷新 updated_at 时间戳。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            """
            UPDATE memory_items SET content = ?, updated_at = ?, metadata = ?
            WHERE memory_id = ?
        """,
            (new_content, now, json.dumps(metadata or {}), memory_id),
        )
        conn.commit()
        conn.close()

    def search_memories(
        self,
        user_id: Optional[str],
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        """检索与查询向量最相关的记忆：先在数据库层按 user_id 过滤候选，
        再在内存中对每条记忆计算余弦相似度（无嵌入向量时回退到 score），
        最后按相似度降序排序并返回前 top_k 条记忆。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 指定 user_id 时先在 SQL 层过滤，减少后续计算量
        if user_id:
            cursor.execute("SELECT * FROM memory_items WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("SELECT * FROM memory_items")
        rows = cursor.fetchall()
        conn.close()

        # 将数据库行统一转换为内存对象，便于后续过滤与打分
        items = [self._row_to_memory_item(r) for r in rows]

        # 应用附加过滤条件：按 metadata 中的字段进行精确匹配
        if filters:
            for key, value in filters.items():
                items = [it for it in items if it.metadata.get(key) == value]

        # 为每条记忆计算与查询向量的相似度得分
        scored = []
        for item in items:
            emb = item.metadata.get("embedding", [])
            if emb and query_vector:
                # 有嵌入向量时使用余弦相似度作为得分
                sim = cosine_similarity(query_vector, emb)
            else:
                # 缺失嵌入向量时回退使用记忆自身的 score 字段
                sim = item.score
            scored.append((sim, item))

        # 按得分降序排序，截取前 top_k 条
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def query_by_time_range(
        self, user_id: str, start: datetime, end: datetime
    ) -> List[MemoryItem]:
        """按时间范围查询指定用户的记忆，结果按创建时间升序返回。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM memory_items
            WHERE user_id = ? AND created_at BETWEEN ? AND ?
            ORDER BY created_at ASC
        """,
            (user_id, start.isoformat(), end.isoformat()),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_memory_item(r) for r in rows]

    def count_by_user(self, user_id: str) -> int:
        """统计指定用户的记忆条数。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM memory_items WHERE user_id = ?", (user_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def _row_to_memory_item(self, row: Tuple) -> MemoryItem:
        """将数据库查询返回的行（Tuple）转换为 MemoryItem 内存对象，
        完成 JSON 反序列化与 datetime 解析。"""
        return MemoryItem(
            memory_id=row[0],
            content=row[1],
            metadata=json.loads(row[2]),
            created_at=datetime.fromisoformat(row[3]),
            updated_at=datetime.fromisoformat(row[4]),
            memory_type=MemoryType(row[5]),
            score=row[6],
            version=row[7],
            user_id=row[8],
            session_id=row[9],
        )

    def _profile_to_text(self, profile: UserProfile) -> str:
        """将用户画像各字段序列化为文本，用于生成画像语义向量。"""
        parts = [str(profile.user_id)]
        for name in ("base_info", "preferences", "scene_profiles", "config"):
            data = getattr(profile, name)
            if data:
                parts.append(json.dumps(data, ensure_ascii=False))
        return " ".join(parts)

    def insert_user_profile(self, profile: UserProfile):
        """插入或覆盖用户画像：将 UserProfile 序列化后写入 user_profiles 表。
        若画像尚未携带嵌入向量，则自动对画像内容做语义向量化。"""
        if not profile.embedding:
            profile.embedding = generate_embedding(self._profile_to_text(profile))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO user_profiles
            (user_id, base_info, preferences, scene_profiles, config,
             lock_keys, current_version, last_updated, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                profile.user_id,
                json.dumps(profile.base_info),
                json.dumps(profile.preferences),
                json.dumps(profile.scene_profiles),
                json.dumps(profile.config),
                json.dumps(profile.lock_keys),
                profile.current_version,
                profile.last_updated,
                json.dumps(profile.embedding or []),
            ),
        )
        conn.commit()
        conn.close()

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """按 user_id 查询用户画像；不存在时返回 None。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            # 将数据库行反序列化为 UserProfile 对象
            return UserProfile(
                user_id=row[0],
                base_info=json.loads(row[1]),
                preferences=json.loads(row[2]),
                scene_profiles=json.loads(row[3]),
                config=json.loads(row[4]),
                lock_keys=json.loads(row[5]),
                current_version=row[6],
                last_updated=row[7],
                embedding=json.loads(row[8]) if row[8] else None,
            )
        return None

    def insert_version(self, version: MemoryVersion):
        """新增一条记忆版本记录。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory_versions (version_id, memory_id, content, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (
                version.version_id,
                version.memory_id,
                version.content,
                version.created_at.isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def get_versions(self, memory_id: str) -> List[MemoryVersion]:
        """查询某条记忆的全部历史版本，按创建时间倒序返回。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM memory_versions WHERE memory_id = ? ORDER BY created_at DESC
        """,
            (memory_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            MemoryVersion(
                version_id=r[1],
                memory_id=r[2],
                content=r[3],
                created_at=datetime.fromisoformat(r[4]),
            )
            for r in rows
        ]

    def close(self):
        """关闭存储（当前无需要释放的资源，保留接口一致性）。"""
        pass
