"""
向量存储——Chroma 实现

嵌入式零依赖向量数据库，适合原型开发和小规模数据。
依赖: chromadb>=0.4.0
"""

from rag.datatypes import Chunk, SearchResult
from .base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Chroma 向量存储（嵌入式）"""

    def __init__(self, persist_dir: str = "./data/chroma"):
        self._persist_dir = persist_dir
        self._client = None

    def _lazy_client(self):
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_dir)
        return self._client

    def _collection(self, name: str, dim: int = 0):
        client = self._lazy_client()
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine", "dimension": str(dim)} if dim else {},
            )

    def create_collection(self, name: str, dim: int) -> None:
        self._collection(name, dim)

    def insert(self, collection: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        first_emb = next(
            (v for chunk in chunks for v in (chunk.embeddings or {}).values() if v),
            None,
        )
        if first_emb is None:
            raise ValueError("Chunks 缺少嵌入向量")
        dim = len(first_emb)
        col = self._collection(collection, dim)
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metadatas.append(chunk.metadata)
            emb = next((v for v in (chunk.embeddings or {}).values() if v), None)
            if emb:
                embeddings.append(emb)
        col.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[SearchResult]:
        col = self._collection(collection)
        results = col.query(query_embeddings=[query_vector], n_results=top_k)
        if not results["ids"]:
            return []
        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            chunk = Chunk(
                chunk_id=doc_id,
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            search_results.append(
                SearchResult(
                    chunk=chunk,
                    score=results["distances"][0][i] if results["distances"] else 0.0,
                    rank=i,
                )
            )
        return search_results

    def delete(self, collection: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        col = self._collection(collection)
        col.delete(ids=chunk_ids)

    def delete_collection(self, name: str) -> None:
        client = self._lazy_client()
        try:
            client.delete_collection(name)
        except Exception:
            pass

    def get_all_chunks(self, collection: str) -> list[Chunk]:
        col = self._collection(collection)
        results = col.get()
        if not results["ids"]:
            return []
        return [
            Chunk(
                chunk_id=results["ids"][i],
                text=results["documents"][i] if results["documents"] else "",
                metadata=results["metadatas"][i] if results["metadatas"] else {},
            )
            for i in range(len(results["ids"]))
        ]
