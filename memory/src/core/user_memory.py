"""用户记忆模块。

将某个用户（单租户视角）的三层记忆——工作记忆（WorkingMemory）、
会话记忆（SessionMemory）、长期记忆（LongTermMemory）——以及用户画像
（UserProfile）聚合在一个 UserMemory 实例中统一管理，并通过治理层
（MemoryGovernance）裁决记忆是否值得进入长期记忆。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.memory_item import MemoryItem, MemoryType
from ..models.user_profile import UserProfile
from ..stores.vector_store import VectorStore
from ..utils.id_generator import generate_id
from .long_term_memory import LongTermMemory
from .memory_governance import MemoryGovernance
from .session_memory import SessionMemory
from .working_memory import WorkingMemory


class UserMemory:
    """单用户的三层记忆门面（Facade）。

    负责构建/保存用户画像，串联工作记忆、会话记忆、长期记忆，
    并承担会话结束时的记忆沉淀、长期记忆入库裁决、跨层记忆召回
    以及画像更新的锁定字段保护等职责。
    """

    def __init__(
        self,
        user_id: str,
        vector_store: VectorStore,
        governance: MemoryGovernance,
        working_memory: Optional[WorkingMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ):
        # 当前用户唯一标识
        self.user_id = user_id
        # 共享的向量存储（多用户共用同一后端）
        self.vector_store = vector_store
        # 记忆治理层，负责入库裁决与用户画像管理
        self.governance = governance

        # 工作记忆：默认新建；也可注入外部实例
        self.working_memory = working_memory or WorkingMemory()
        # 会话记忆：仅在 start_session 之后才非空
        self.session_memory: Optional[SessionMemory] = None
        # 长期记忆：默认基于同一向量存储新建
        self.long_term_memory = long_term_memory or LongTermMemory(vector_store)
        # 用户画像：惰性加载
        self.profile: Optional[UserProfile] = None

        self._load_profile()

    def _load_profile(self):
        """加载（或在缺失时创建并持久化）当前用户的画像。

        先从向量存储读取用户画像；若不存在，则以当前 user_id
        新建一个默认画像并写回存储。
        """
        self.profile = self.vector_store.get_user_profile(self.user_id)
        if not self.profile:
            self.profile = UserProfile(user_id=self.user_id)
            self.vector_store.insert_user_profile(self.profile)

    def _make_memory(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> MemoryItem:
        """构造一个绑定当前用户的记忆条目（统一字段初始化）。

        参数：content 内容；memory_type 记忆类型；metadata 元数据；
        session_id 可选会话 ID。返回组装完成的 MemoryItem。
        """
        return MemoryItem(
            memory_id=generate_id(),
            content=content,
            metadata=metadata or {},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            memory_type=memory_type,
            user_id=self.user_id,
            session_id=session_id,
        )

    def add_working_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryItem:
        """新增一条工作记忆并返回该条目。"""
        item = self._make_memory(content, MemoryType.WORKING, metadata)
        self.working_memory.add_item(item)
        return item

    def start_session(self, session_id: str):
        """以指定 session_id 开启一个新的会话记忆容器。"""
        self.session_memory = SessionMemory(session_id)

    def add_session_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MemoryItem]:
        """向当前会话写入一条记忆；未开启会话时返回 None。"""
        if not self.session_memory:
            return None
        item = self._make_memory(
            content, MemoryType.SESSION, metadata, self.session_memory.session_id
        )
        self.session_memory.add_item(item)
        return item

    def end_session(self) -> List[MemoryItem]:
        """结束当前会话并将记忆沉淀到长期记忆。

        取出本会话全部条目；逐条经治理层 should_enter_long_term 判定，
        满足条件的写入长期记忆；最后清空会话容器并返回全部沉淀项。
        未开启会话时返回空列表。
        """
        if not self.session_memory:
            return []
        items = self.session_memory.end_session()
        for item in items:
            # 仅当治理层判定该条目值得保留时才入库长期记忆
            if self.governance.should_enter_long_term(item):
                self.long_term_memory.add_memory(item)
        self.session_memory = None
        return items

    def add_long_term_memory(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MemoryItem]:
        """主动写入一条长期记忆（经治理裁决）。

        先构造条目并召回该用户已有的 50 条记忆作为参考，交给治理层
        adjudicate_entry 裁决；裁决不通过则丢弃（返回 None），
        通过则写入长期记忆并返回该条目。
        """
        item = self._make_memory(content, MemoryType.LONG_TERM, metadata)

        # 召回既有记忆作为裁决上下文
        existing = self.vector_store.search_memories(self.user_id, [], 50)
        # 治理裁决：不通过则不入库
        if not self.governance.adjudicate_entry(item, existing):
            return None

        self.long_term_memory.add_memory(item)
        return item

    def retrieve_relevant_memories(
        self, query: str, top_k: int = 5
    ) -> List[MemoryItem]:
        """跨三层记忆召回与查询最相关的内容。

        依次合并工作记忆、会话记忆、长期记忆（按用户语义检索），
        通过 memory_id 去重后按 top_k 截断返回。
        """
        from ..utils.embeddings import generate_embedding

        results = []
        seen_ids = set()

        # 第一层：工作记忆全量并入
        for item in self.working_memory.get_items():
            if item.memory_id not in seen_ids:
                results.append(item)
                seen_ids.add(item.memory_id)

        # 第二层：会话记忆全量并入
        if self.session_memory:
            for item in self.session_memory.get_items():
                if item.memory_id not in seen_ids:
                    results.append(item)
                    seen_ids.add(item.memory_id)

        # 第三层：长期记忆语义检索
        lt_items = self.long_term_memory.retrieve_by_user(self.user_id, query, top_k)
        for item in lt_items:
            if item.memory_id not in seen_ids:
                results.append(item)
                seen_ids.add(item.memory_id)

        return results[:top_k]

    def flush_working_memory(self):
        """清空当前工作记忆。"""
        self.working_memory.clear()

    def update_profile(self, profile_data: Dict[str, Any]):
        """按给定数据更新用户画像，并遵循锁定字段保护。

        更新逻辑：除 "lock_keys" 本身外，凡在 lock_keys 中被锁定的
        字段一律不允许被覆盖；更新成功后刷新 last_updated、递增
        版本号并写回向量存储。
        """
        if not self.profile:
            self._load_profile()

        for key, value in profile_data.items():
            if hasattr(self.profile, key):
                if key == "lock_keys":
                    # lock_keys 属于治理元数据，允许直接更新
                    setattr(self.profile, key, value)
                elif key not in self.profile.lock_keys:
                    # 锁定字段跳过，其余字段正常写入
                    setattr(self.profile, key, value)

        self.profile.last_updated = datetime.now().timestamp()
        self.profile.current_version += 1
        self.vector_store.insert_user_profile(self.profile)
