"""
嵌入层——抽象基类

所有嵌入器的统一接口。无论是本地模型还是云端 API，
都实现 embed/dim/name 三个方法/属性。
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """嵌入器抽象基类"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表批量向量化，返回 list[向量]"""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """嵌入器唯一标识，如 'openai/text-embedding-3-small'"""
        ...
