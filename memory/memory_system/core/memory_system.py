"""记忆系统模块。

作为整个记忆子系统对外的一级入口：面向多租户（多个 user_id）
统一管理各自独立的 UserMemory 实例，共享同一个向量存储与治理层，
并提供用户记忆的懒加载、注册与移除能力。
"""

from typing import Dict, Optional

from ..stores.vector_store import VectorStore
from .memory_governance import MemoryGovernance
from .user_memory import UserMemory


class MemorySystem:
    """多租户记忆系统入口。

    持有唯一的向量存储与记忆治理层，并以字典按 user_id 缓存
    各用户的 UserMemory；首次访问某用户时才惰性构建其记忆实例。
    """

    def __init__(self, db_path: str = "memory_system.db"):
        # 系统实例标识
        self.system_id = "memory_system_01"
        # 所有用户共享的向量存储（SQLite 后端）
        self.vector_store = VectorStore(db_path)
        # 共享的记忆治理层
        self.governance = MemoryGovernance(vector_store=self.vector_store)
        # user_id -> UserMemory 的缓存映射
        self.user_memories: Dict[str, UserMemory] = {}

    def get_user_memory(self, user_id: str) -> UserMemory:
        """获取指定用户的记忆实例（多租户懒加载）。

        首次请求时创建并缓存该用户的 UserMemory；后续直接返回
        缓存中的实例。
        """
        if user_id not in self.user_memories:
            # 懒加载：首次访问才实例化该用户的记忆对象
            self.user_memories[user_id] = UserMemory(
                user_id=user_id,
                vector_store=self.vector_store,
                governance=self.governance,
            )
        return self.user_memories[user_id]

    def remove_user_memory(self, user_id: str):
        """从缓存中移除指定用户的记忆实例（不存在则静默忽略）。"""
        self.user_memories.pop(user_id, None)

    def add_user_memory(self, user_id: str, memory: UserMemory):
        """将用户记忆实例直接注册/覆盖到缓存中。"""
        self.user_memories[user_id] = memory
