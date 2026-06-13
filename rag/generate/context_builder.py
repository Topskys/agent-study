"""
上下文构建器——ContextBuilder

将检索到的 Chunk 列表格式化为 LLM 输入的上下文文本。
支持 Token 估算、长度截断、动态压缩。
"""

from rag.datatypes import Chunk


class ContextBuilder:
    """构建器：将 Chunk 列表格式化为 LLM 可用的上下文"""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._max_tokens = self._config.get("max_tokens", 4096)
        self._include_source = self._config.get("include_source", True)

    def build(self, chunks: list[Chunk], query: str = "") -> str:
        if not chunks:
            return ""

        segments: list[str] = []
        total_estimate = 0

        c = self._config
        header = c.get(
            "header",
            "Below are reference documents to help answer the question:",
        )
        segments.append(header)

        for i, chunk in enumerate(chunks):
            source_info = ""
            if self._include_source:
                src = chunk.metadata.get("source", "unknown")
                source_info = f" [Source: {src}]"

            seg = f"[{i + 1}]{source_info}\n{chunk.text}"
            est = len(seg) // 2

            if total_estimate + est > self._max_tokens:
                remaining = self._max_tokens - total_estimate
                if remaining > 20:
                    seg = seg[: remaining * 2]
                    seg += "\n[truncated]"
                    segments.append(seg)
                break

            segments.append(seg)
            total_estimate += est

        return "\n\n---\n\n".join(segments)

    def build_with_markers(self, chunks: list[Chunk]) -> tuple[str, dict[str, Chunk]]:
        """构建带标记的上下文，返回 (context_text, id_to_chunk_map)"""
        id_to_chunk: dict[str, Chunk] = {}
        lines: list[str] = []
        for i, chunk in enumerate(chunks):
            marker = f"[{i + 1}]"
            id_to_chunk[marker] = chunk
            lines.append(f"{marker} {chunk.text}")
        return "\n\n".join(lines), id_to_chunk
