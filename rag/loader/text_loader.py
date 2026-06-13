"""
文档加载器——纯文本类文件加载器

支持 .txt .md .html .vue .json .yaml .css .js .ts .py 等。
- .html：使用 html.parser 去标签，跳过 script/style
- .vue：提取 template/script 中的文本
- 其余文本文件：UTF-8 直接读取
"""

from pathlib import Path
from rag.datatypes import RawDocument
from .base import BaseLoader


class TextLoader(BaseLoader):
    """纯文本类文件加载器"""

    def load(self, file_path: str) -> list[RawDocument]:
        path = Path(file_path)
        ext = path.suffix.lower()
        content = path.read_text(encoding="utf-8")

        if ext == ".html":
            content = self._parse_html(content)
        elif ext == ".vue":
            content = self._parse_vue(content)

        return [
            RawDocument(
                content=content,
                source=str(path),
                metadata={"file_name": path.name, "file_type": ext.lstrip(".")},
            )
        ]

    def supported_extensions(self) -> list[str]:
        return [
            ".txt",
            ".md",
            ".html",
            ".vue",
            ".json",
            ".yaml",
            ".yml",
            ".css",
            ".js",
            ".ts",
            ".py",
        ]

    def _parse_html(self, content: str) -> str:
        """HTML 去标签，只保留可见文本"""
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self._text = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self._text.append(stripped)

            def result(self) -> str:
                return "\n".join(self._text)

        parser = TextExtractor()
        parser.feed(content)
        return parser.result()

    def _parse_vue(self, content: str) -> str:
        """提取 Vue 单文件组件中的 template 和 script 文本"""
        import re

        parts = []
        for match in re.finditer(r"<template>(.*?)</template>", content, re.DOTALL):
            parts.append(match.group(1))
        for match in re.finditer(r"<script(?:.*?)>(.*?)</script>", content, re.DOTALL):
            parts.append(match.group(1))
        combined = "\n".join(parts)
        return self._parse_html(combined) if combined else content
