"""
BM25 关键词检索

基于 rank_bm25 + jieba 中文分词的纯 Python 实现。
用于混合检索中的关键词路，弥补向量检索在人名、型号等精确词匹配上的不足。
"""

import jieba
from rank_bm25 import BM25Okapi
from rag.datatypes import Chunk, SearchResult


class BM25Index:
    """BM25 关键词检索索引"""

    def __init__(self):
        self._indexes: dict[str, BM25Okapi] = {}  # 集合名 → BM25 索引
        self._chunk_map: dict[str, list[Chunk]] = {}  # 集合名 → Chunk 列表

    def add_documents(self, collection: str, chunks: list[Chunk]) -> None:
        """增量添加文档，重建 BM25 索引"""
        if collection in self._chunk_map:
            self._chunk_map[collection].extend(chunks)
        else:
            self._chunk_map[collection] = chunks
        self._indexes[collection] = BM25Okapi(
            [self._tokenize(c.text) for c in self._chunk_map[collection]]
        )

    def search(self, collection: str, query: str, top_k: int) -> list[SearchResult]:
        """BM25 检索，返回带分数的 SearchResult 列表"""
        if collection not in self._indexes:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._indexes[collection].get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [
            SearchResult(
                chunk=self._chunk_map[collection][i], score=float(scores[i]), rank=rank
            )
            for rank, i in enumerate(top_indices)
        ]

    def _tokenize(self, text: str) -> list[str]:
        """jieba 中文分词"""
        return jieba.lcut(text)
