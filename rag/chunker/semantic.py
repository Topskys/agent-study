"""
分块器——语义分块

基于句子嵌入的余弦相似度，在语义断点处切分。
原理：相邻句子语义发生突变的位置就是段落边界。
"""

import uuid
import numpy as np
from rag.datatypes import RawDocument, Chunk
from rag.embed.base import BaseEmbedder
from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """语义分块器：在语义相似度突降的位置切分"""

    def __init__(
        self, embedder: BaseEmbedder, threshold: float = 0.7, min_chunk_size: int = 100
    ):
        self.embedder = embedder  # 用于计算句子向量
        self.threshold = threshold  # 相似度阈值，低于此值切分
        self.min_chunk_size = min_chunk_size  # 最小块大小，防止过碎

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        sentences = self._split_sentences(doc.content)
        if len(sentences) <= 1:
            return self._make_chunks([doc.content], doc, 0)

        # 计算相邻句子间的语义相似度
        embs = self.embedder.embed(sentences)
        sim_matrix = self._cosine_sim(np.array(embs))

        # 在语义断点处切分
        groups = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            if (
                sim_matrix[i - 1, i] < self.threshold
                and len("".join(current)) >= self.min_chunk_size
            ):
                groups.append("".join(current))
                current = []
            current.append(sentences[i])
        if current:
            groups.append("".join(current))

        return self._make_chunks(groups, doc, 0)

    def _split_sentences(self, text: str) -> list[str]:
        """按句号、问号、感叹号、换行符切分句子"""
        import re

        parts = re.split(r"(?<=[。！？.!?\n])\s*", text)
        return [p.strip() for p in parts if p.strip()]

    def _cosine_sim(self, embs: np.ndarray) -> np.ndarray:
        """计算向量间的余弦相似度矩阵"""
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return (embs @ embs.T) / (norms * norms.T)

    def _make_chunks(
        self, texts: list[str], doc: RawDocument, offset: int
    ) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=t,
                metadata={**doc.metadata, "chunk_index": offset + i},
            )
            for i, t in enumerate(texts)
        ]
