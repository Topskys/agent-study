"""
文档加载器——抽象基类

所有加载器的统一接口。每种文件格式对应一个加载器实现，
通过 LoaderFactory 按扩展名自动路由。
"""

from abc import ABC, abstractmethod
from rag.datatypes import RawDocument


class BaseLoader(ABC):
    """加载器抽象基类"""

    @abstractmethod
    def load(self, file_path: str) -> list[RawDocument]:
        """加载文件并返回 RawDocument 列表。一个文件可能拆为多页/多模态文档。"""
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """返回该加载器支持的文件扩展名列表"""
        ...
