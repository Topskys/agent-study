"""
文档加载器——工厂

按文件扩展名自动选择对应的加载器。可通过 register_loader 扩展新格式。
"""

from pathlib import Path
from .base import BaseLoader
from .text_loader import TextLoader
from .docx_loader import DocxLoader
from .pdf_loader import PdfLoader
from .video_loader import VideoLoader


class UnsupportedFormatError(Exception):
    """不支持的文件格式"""

    pass


class LoaderFactory:
    """加载器工厂，按扩展名自动路由到对应的加载器"""

    _registry: dict[str, type[BaseLoader]] = {
        ".html": TextLoader,
        ".vue": TextLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
        ".json": TextLoader,
        ".yaml": TextLoader,
        ".yml": TextLoader,
        ".css": TextLoader,
        ".js": TextLoader,
        ".ts": TextLoader,
        ".py": TextLoader,
        ".docx": DocxLoader,
        ".pdf": PdfLoader,
        ".mp4": VideoLoader,
        ".avi": VideoLoader,
        ".mov": VideoLoader,
        ".mkv": VideoLoader,
    }

    def get_loader(self, file_path: str) -> BaseLoader:
        """根据文件路径获取对应的加载器实例"""
        ext = Path(file_path).suffix.lower()
        if ext not in self._registry:
            raise UnsupportedFormatError(f"不支持的文件格式: {ext}")
        return self._registry[ext]()

    def register_loader(self, ext: str, loader_cls: type[BaseLoader]):
        """注册新的加载器"""
        ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        self._registry[ext] = loader_cls
