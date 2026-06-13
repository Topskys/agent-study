"""
向量存储——抽象基类

所有向量存储的统一接口。支持多种后端实现。
"""

from abc import ABC, abstractmethod
from rag.datatypes import Chunk, SearchResult


class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def create_collection(self, name: str, dim: int) -> None:
        """创建集合（类似数据库表）"""
        ...

    @abstractmethod
    def insert(self, collection: str, chunks: list[Chunk]) -> None:
        """插入 Chunk 列表到指定集合"""
        ...

    @abstractmethod
    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[SearchResult]:
        """余弦相似度检索，返回 Top-K 结果"""
        ...

    @abstractmethod
    def delete(self, collection: str, chunk_ids: list[str]) -> None:
        """从集合中删除指定 Chunk"""
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """删除整个集合"""
        ...

    @abstractmethod
    def get_all_chunks(self, collection: str) -> list[Chunk]:
        """获取集合中所有 Chunk（用于调试/查看索引）"""
        ...
