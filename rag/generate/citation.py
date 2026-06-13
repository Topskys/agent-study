"""
引用追踪器——CitationTracker

追踪生成答案中引用的来源。解析答案中的 [1], [2] 标记，
并关联到对应的 Chunk 来源。
"""

import re
from rag.datatypes import Chunk, RAGResult


class CitationTracker:
    """引用追踪器：将答案中的引用标记映射到原始来源"""

    def __init__(self):
        self._marker_pattern = re.compile(r"\[(\d+)\]")

    def track(self, answer: str, chunks: list[Chunk]) -> list[dict]:
        """从答案文本中提取引用标记并映射到来源"""
        matches = self._marker_pattern.findall(answer)
        seen = set()
        sources = []

        for m in matches:
            idx = int(m) - 1
            if idx in seen or idx < 0 or idx >= len(chunks):
                continue
            seen.add(idx)
            chunk = chunks[idx]
            sources.append(
                {
                    "id": idx + 1,
                    "source": chunk.metadata.get("source", "unknown"),
                    "text": chunk.text[:300],
                }
            )

        if not sources and chunks:
            for i, c in enumerate(chunks):
                sources.append(
                    {
                        "id": i + 1,
                        "source": c.metadata.get("source", "unknown"),
                        "text": c.text[:300],
                    }
                )

        return sources

    def build_result(
        self,
        answer: str,
        chunks: list[Chunk],
        confidence: float = 0.0,
        request_id: str = "",
        latency_ms: float = 0.0,
    ) -> RAGResult:
        sources = self.track(answer, chunks)
        return RAGResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            request_id=request_id,
            latency_ms=latency_ms,
        )
