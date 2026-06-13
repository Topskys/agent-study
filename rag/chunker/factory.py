"""
分块器——工厂

按策略名称选择对应的分块器。semantic 分块需要额外传入 embedder。
"""

from .base import BaseChunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .parent_child import ParentChildChunker


class ChunkerFactory:
    """分块器工厂：按策略名创建分块器实例"""

    def get_chunker(
        self, strategy: str, config: dict | None = None, embedder=None
    ) -> BaseChunker:
        config = config or {}
        if strategy == "recursive":
            return RecursiveChunker(**config.get("recursive", {}))
        elif strategy == "semantic":
            if embedder is None:
                raise ValueError("semantic 分块需要传入 embedder")
            return SemanticChunker(embedder=embedder, **config.get("semantic", {}))
        elif strategy == "parent_child":
            return ParentChildChunker(**config.get("parent_child", {}))
        else:
            raise ValueError(f"未知分块策略: {strategy}")
