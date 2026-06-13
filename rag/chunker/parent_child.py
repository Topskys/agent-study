"""
分块器——父子分块

两级粒度，兼顾检索精度和上下文完整性。
  子 Chunk（小粒度）：用于精确检索
  父 Chunk（大粒度）：包含子 Chunk，用于 LLM 上下文
检索流程：子 Chunk 定位 → 映射到父 Chunk → 返回父 Chunk
"""

import uuid
from rag.datatypes import RawDocument, Chunk
from .base import BaseChunker
from .recursive import RecursiveChunker


class ParentChildChunker(BaseChunker):
    """父子分块器：小粒度检索 + 大粒度生成"""

    def __init__(
        self, child_size: int = 128, parent_size: int = 768, overlap: int = 32
    ):
        self.child_chunker = RecursiveChunker(
            chunk_size=child_size, chunk_overlap=overlap
        )
        self.parent_chunker = RecursiveChunker(
            chunk_size=parent_size, chunk_overlap=overlap
        )

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        # 分别生成父子块
        parents = self.parent_chunker.chunk(doc)
        children = self.child_chunker.chunk(doc)

        # 建立父子映射：子 Chunk 中包含在父 Chunk 文本内的，记录 parent_id
        parent_map = {}
        for p in parents:
            parent_map[p.chunk_id] = p

        for c in children:
            for pid, p in parent_map.items():
                if c.text in p.text or p.text in c.text:
                    c.parent_id = pid
                    break

        return children + parents
