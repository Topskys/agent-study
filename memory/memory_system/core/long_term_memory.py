"""长期记忆模块。

管理跨会话持久化的长期记忆：通过向量存储（VectorStore）完成语义检索，
并同步写入事件流（EventStream）用于审计/回放；同时整合图存储（知识图谱）
与键值存储（KV）作为补充能力。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.event import Event
from ..models.graph import GraphEdge, GraphNode
from ..models.memory_item import MemoryItem
from ..stores.event_store import EventStreamStore
from ..stores.graph_store import GraphStore
from ..stores.kv_store import KeyValueStore
from ..stores.vector_store import VectorStore
from ..utils.id_generator import generate_id


class LongTermMemory:
    """长期记忆管理器。

    对外提供记忆入库、语义检索、按用户检索、更新/删除，以及图与
    键值两类附助存储的读写能力。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: Optional[GraphStore] = None,
        kv_store: Optional[KeyValueStore] = None,
        event_store: Optional[EventStreamStore] = None,
    ):
        # 向量存储：长期记忆的主体存储与检索后端
        self.vector_store = vector_store
        # 图存储：无则新建默认实例（与向量库同库持久化）
        self.graph_store = graph_store or GraphStore(db_path=vector_store.db_path)
        # 键值存储：用于保存结构化小数据
        self.kv_store = kv_store or KeyValueStore(db_path=vector_store.db_path)
        # 事件流存储：记录记忆相关事件
        self.event_store = event_store or EventStreamStore(db_path=vector_store.db_path)

    def add_memory(self, item: MemoryItem) -> str:
        """将一条记忆写入长期存储并返回其 memory_id。

        流程：先写入向量库完成入库，再向事件流追加一条
        "memory_added" 事件（含 memory_id 与内容前 50 字符预览），
        以便后续审计与回放。
        """
        self.vector_store.insert_memory(item)
        self.event_store.add_event(
            Event(
                event_id=generate_id(),
                event_type="memory_added",
                payload={
                    "memory_id": item.memory_id,
                    "content_preview": item.content[:50],
                },
            )
        )
        return item.memory_id

    def retrieve_memories(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[MemoryItem]:
        """跨用户（不限定用户）按语义相似度检索记忆。

        user_id 传 None，即向量存储中所有用户的记忆都可被召回；
        filters 用于额外的属性过滤。
        """
        from ..utils.embeddings import generate_embedding

        query_vector = generate_embedding(query, for_query=True)
        return self.vector_store.search_memories(None, query_vector, top_k, filters)

    def retrieve_by_user(
        self, user_id: str, query: str = "", top_k: int = 5
    ) -> List[MemoryItem]:
        """按指定用户检索其长期记忆。

        仅当提供非空 query 时才会生成向量并执行语义检索；
        query 为空时直接返回空列表。
        """
        if query:
            from ..utils.embeddings import generate_embedding

            query_vector = generate_embedding(query, for_query=True)
            return self.vector_store.search_memories(user_id, query_vector, top_k)
        return []

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """按 memory_id 更新长期记忆的内容与元数据。"""
        self.vector_store.update_memory(memory_id, new_content, metadata)

    def delete_memory(self, memory_id: str):
        """按 memory_id 从长期存储中删除记忆。"""
        self.vector_store.delete_memory(memory_id)

    def add_graph_node(self, node: GraphNode):
        """向知识图谱添加一个节点。"""
        self.graph_store.add_node(node)

    def add_graph_edge(self, edge: GraphEdge):
        """向知识图谱添加一条边（节点间关系）。"""
        self.graph_store.add_edge(edge)

    def store_value(self, key: str, value: Any, ttl: Optional[float] = None):
        """以键值形式保存结构化数据，可选设置过期时间 ttl。"""
        self.kv_store.put(key, value, ttl)

    def get_value(self, key: str) -> Optional[Any]:
        """按键读取之前保存的结构化数据。"""
        return self.kv_store.get(key)
