"""
文档加载器——PDF 加载器

使用 PyMuPDF (fitz) 提取 PDF 文本层。
目前只实现了文字层提取，后续可扩展表格层 (pdfplumber) 和 OCR 层 (PaddleOCR)。
"""

from pathlib import Path
from rag.datatypes import RawDocument
from .base import BaseLoader


class PdfLoader(BaseLoader):
    """PDF 文档加载器（当前仅支持文字层）"""

    def load(self, file_path: str) -> list[RawDocument]:
        path = Path(file_path)
        docs = []
        text_docs = self._extract_text(str(path))
        docs.extend(text_docs)
        return docs

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def _extract_text(self, path: str) -> list[RawDocument]:
        """按页提取 PDF 文字层"""
        try:
            import fitz
        except ImportError:
            raise ImportError("需要安装 PyMuPDF: pip install PyMuPDF")

        doc = fitz.open(path)
        results = []
        for page_num, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if text:
                results.append(
                    RawDocument(
                        content=text,
                        source=path,
                        metadata={
                            "file_name": Path(path).name,
                            "file_type": "pdf",
                            "page_num": page_num,
                            "modality": "text",
                        },
                    )
                )
        doc.close()
        return results
