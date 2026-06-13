"""
分块器——抽象基类

负责将 RawDocument 切分为多个 Chunk。不同的分块策略实现此接口。
"""

from abc import ABC, abstractmethod
from rag.datatypes import RawDocument, Chunk


class BaseChunker(ABC):
    """分块器抽象基类"""

    @abstractmethod
    def chunk(self, doc: RawDocument) -> list[Chunk]:
        """将 RawDocument 切分为 Chunk 列表"""
        ...
