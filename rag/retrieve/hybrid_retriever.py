"""
混合检索器——HybridRetriever

融合向量检索（语义相似度）和关键词检索（BM25）的结果。
默认使用 Reciprocal Rank Fusion (RRF) 算法合并分数。
"""

import math
from rag.datatypes import SearchResult
from rag.store import BaseVectorStore, BM25Index


class HybridRetriever:
    """向量 + BM25 混合检索器"""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_index: BM25Index | None = None,
        config: dict | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self._config = config or {}
        self._rrf_k = self._config.get("rrf_k", 60)
        self._vector_weight = self._config.get("vector_weight", 0.5)
        self._bm25_weight = self._config.get("bm25_weight", 0.5)

    def retrieve(
        self,
        collection: str,
        query_vector: list[float],
        query_text: str = "",
        top_k: int = 10,
    ) -> list[SearchResult]:
        vector_results = self.vector_store.search(collection, query_vector, top_k * 2)

        bm25_results: list[SearchResult] = []
        if self.bm25_index and query_text:
            bm25_results = self.bm25_index.search(collection, query_text, top_k * 2)

        if not bm25_results:
            return vector_results[:top_k]

        return self._rrf_fuse(vector_results, bm25_results, top_k)

    def _rrf_fuse(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(vector_results):
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self._rrf_k + rank + 1)
            chunk_map[cid] = r

        for rank, r in enumerate(bm25_results):
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self._rrf_k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = r

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [
            SearchResult(
                chunk=chunk_map[cid].chunk,
                score=rrf_scores[cid],
                rank=rank,
            )
            for rank, cid in enumerate(sorted_ids)
        ]
