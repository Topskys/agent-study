"""
重排序器——Reranker

对检索结果进行二次排序，提升精度。
支持多种策略:
  1. identity: 直通，按原分數排序
  2. diversity: 基于来源多样性重排序（MMR 风格）
  3. threshold: 过滤低分结果后重排序
"""

from rag.datatypes import SearchResult


class Reranker:
    """检索结果重排序器"""

    def __init__(self, strategy: str = "identity", config: dict | None = None):
        self.strategy = strategy
        self._config = config or {}
        self._threshold = self._config.get("threshold", 0.0)
        self._diversity_alpha = self._config.get("diversity_alpha", 0.5)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results

        if self.strategy == "identity":
            return results

        if self.strategy == "threshold":
            return self._threshold_filter(results)

        if self.strategy == "diversity":
            return self._diversity_rerank(query, results)

        return results

    def _threshold_filter(self, results: list[SearchResult]) -> list[SearchResult]:
        filtered = [r for r in results if r.score >= self._threshold]
        for rank, r in enumerate(filtered):
            r.rank = rank
        return filtered

    def _diversity_rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        if len(results) <= 1:
            return results

        selected: list[SearchResult] = [results[0]]
        candidate_pool = list(results[1:])

        while candidate_pool and len(selected) < len(results):
            best_idx = 0
            best_score = -float("inf")

            for i, candidate in enumerate(candidate_pool):
                relevance = candidate.score
                max_sim = max(
                    self._jaccard_sim(candidate.chunk.text, s.chunk.text)
                    for s in selected
                )
                mmr_score = (
                    self._diversity_alpha * relevance
                    - (1 - self._diversity_alpha) * max_sim
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidate_pool.pop(best_idx))

        for rank, r in enumerate(selected):
            r.rank = rank
        return selected

    @staticmethod
    def _jaccard_sim(text_a: str, text_b: str) -> float:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
