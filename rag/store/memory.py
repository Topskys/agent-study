"""
向量存储——内存实现

基于 numpy 的纯内存向量存储。无外部依赖，适合原型开发和小规模数据。
不支持持久化，重启后数据丢失。
"""

import numpy as np
from rag.datatypes import Chunk, SearchResult
from .base import BaseVectorStore


class MemoryStore(BaseVectorStore):
    """内存向量存储（numpy 实现）"""

    def __init__(self):
        self._collections: dict[str, list[Chunk]] = {}

    def create_collection(self, name: str, dim: int = 0) -> None:
        if name not in self._collections:
            self._collections[name] = []

    def insert(self, collection: str, chunks: list[Chunk]) -> None:
        if collection not in self._collections:
            self._collections[collection] = []
        self._collections[collection].extend(chunks)

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[SearchResult]:
        chunks = self._collections.get(collection, [])
        if not chunks:
            return []

        # 提取所有 Chunk 的向量（取第一个嵌入器的向量）
        vecs = []
        valid_chunks = []
        for c in chunks:
            for emb in c.embeddings.values():
                vecs.append(emb)
                valid_chunks.append(c)
                break

        if not vecs:
            return []

        # 余弦相似度计算
        query_vec = np.array(query_vector, dtype=np.float32)
        chunk_vecs = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec)
        norms[norms == 0] = 1e-10
        scores = np.dot(chunk_vecs, query_vec) / norms

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            SearchResult(chunk=valid_chunks[idx], score=float(scores[idx]), rank=rank)
            for rank, idx in enumerate(top_indices)
        ]

    def delete(self, collection: str, chunk_ids: list[str]) -> None:
        if collection in self._collections:
            id_set = set(chunk_ids)
            self._collections[collection] = [
                c for c in self._collections[collection] if c.chunk_id not in id_set
            ]

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)

    def get_all_chunks(self, collection: str) -> list[Chunk]:
        return self._collections.get(collection, [])
