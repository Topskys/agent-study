from rag.datatypes import Chunk, SearchResult
from .base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir

    def create_collection(self, name: str, dim: int) -> None:
        raise NotImplementedError("ChromaStore 暂未实现")

    def insert(self, collection: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError("ChromaStore 暂未实现")

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[SearchResult]:
        raise NotImplementedError("ChromaStore 暂未实现")

    def delete(self, collection: str, chunk_ids: list[str]) -> None:
        raise NotImplementedError("ChromaStore 暂未实现")

    def delete_collection(self, name: str) -> None:
        raise NotImplementedError("ChromaStore 暂未实现")

    def get_all_chunks(self, collection: str) -> list[Chunk]:
        raise NotImplementedError("ChromaStore 暂未实现")
