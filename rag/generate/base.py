from abc import ABC, abstractmethod

from rag.datatypes import Chunk, RAGResult


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context: str, **kwargs) -> str: ...

    @abstractmethod
    def generate_with_sources(
        self, query: str, chunks: list[Chunk], **kwargs
    ) -> RAGResult: ...
