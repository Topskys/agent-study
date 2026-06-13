from abc import ABC, abstractmethod

from rag.datatypes import SearchResult


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[SearchResult]: ...
