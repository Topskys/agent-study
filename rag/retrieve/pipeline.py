"""
检索管线——RetrievalPipeline

编排查询重写 → 混合检索 → 重排序的完整流程。
"""

from rag.datatypes import SearchResult
from rag.embed import BaseEmbedder
from .query_rewriter import QueryRewriter
from .hybrid_retriever import HybridRetriever
from .reranker import Reranker


class RetrievalPipeline:
    """检索管线：编排查询重写 → 混合检索 → 重排序"""

    def __init__(
        self,
        embedder: BaseEmbedder,
        hybrid_retriever: HybridRetriever,
        query_rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
    ):
        self.embedder = embedder
        self.hybrid_retriever = hybrid_retriever
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.reranker = reranker or Reranker()

    def retrieve(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 10,
    ) -> list[SearchResult]:
        rewritten = self.query_rewriter.rewrite(query)
        query_vector = self.embedder.embed([rewritten])[0]
        results = self.hybrid_retriever.retrieve(
            collection, query_vector, rewritten, top_k
        )
        results = self.reranker.rerank(rewritten, results)
        return results
