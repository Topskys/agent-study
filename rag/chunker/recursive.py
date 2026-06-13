"""
分块器——递归分块

按分隔符优先级递归切分文本，中英文分隔符交替使用，保证中英文文档均能正确切分。

分隔符优先级：
  ["\n\n\n", "\n\n", "\n", "。", "。 ", ". ", "；", "； ", "，", ",", ""]
"""

import uuid
from rag.datatypes import RawDocument, Chunk
from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    """递归分块器：按分隔符层级递归切分，支持中英文混合"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = [
            "\n\n\n",
            "\n\n",
            "\n",
            "。",
            "。 ",
            ". ",
            "；",
            "； ",
            "，",
            ",",
            "",
        ]

    def chunk(self, doc: RawDocument) -> list[Chunk]:
        texts = self._split_text(doc.content)
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=t,
                metadata={**doc.metadata, "chunk_index": i},
                parent_id=None,
            )
            for i, t in enumerate(texts)
        ]

    def _split_text(self, text: str) -> list[str]:
        """递归切分主逻辑：按 chunk_size 滑动窗口，在每个窗口内找最优分隔点"""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            # 剩余文本不足一个 chunk，直接收尾
            if end >= text_len:
                chunks.append(text[start:].strip())
                break

            # 在当前窗口内找最优分隔位置
            split_pos = self._find_split(text, start, end)

            chunk_text = text[start:split_pos].strip()
            if chunk_text:
                chunks.append(chunk_text)

            # 滑动 start，保留 overlap 部分
            start = split_pos - self.chunk_overlap
            if start < split_pos - self.chunk_size // 2:
                start = split_pos

        return chunks

    def _find_split(self, text: str, start: int, end: int) -> int:
        """在 [start, end] 范围内按分隔符优先级查找最优切分位置"""
        segment = text[start : end + max(len(s) for s in self.separators if s)]
        best_pos = end
        best_priority = len(self.separators) + 1

        for priority, sep in enumerate(self.separators):
            if not sep:
                continue
            pos = segment.rfind(sep, 0, end - start + len(sep))
            if pos != -1:
                actual_pos = start + pos + len(sep)
                # 优先使用高优先级的切分符（序号越小优先级越高）
                if actual_pos <= end and priority < best_priority:
                    best_pos = actual_pos
                    best_priority = priority

        return best_pos
