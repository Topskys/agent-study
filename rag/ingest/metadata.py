"""
元数据提取器——MetadataExtractor

从 RawDocument 中提取/补充元数据，如文件大小、语言、关键词等。
"""

import os
from pathlib import Path

from rag.datatypes import RawDocument


class MetadataExtractor:
    """从文档中提取和补充元数据"""

    def extract(self, doc: RawDocument) -> dict:
        metadata = dict(doc.metadata)
        source = Path(doc.source)
        metadata.setdefault("file_name", source.name)
        metadata.setdefault("file_ext", source.suffix.lower())
        metadata.setdefault("file_size", self._get_file_size(source))
        metadata.setdefault("char_count", len(doc.content))
        if not metadata.get("language"):
            lang = self._detect_language(doc.content)
            if lang:
                metadata["language"] = lang
        return metadata

    def _get_file_size(self, path: Path) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def _detect_language(self, text: str) -> str | None:
        if not text:
            return None
        zh_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        total = len(text.strip())
        if total == 0:
            return None
        return "zh" if zh_chars / total > 0.1 else "en"
