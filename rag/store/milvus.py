"""
向量存储——Milvus 实现

生产级分布式向量数据库，支持 HNSW 索引和 GPU 加速。
依赖: pymilvus>=2.3.0
"""

from rag.datatypes import Chunk, SearchResult
from .base import BaseVectorStore


class MilvusStore(BaseVectorStore):
    """Milvus 向量存储"""

    def __init__(self, host: str = "localhost", port: int = 19530):
        self._host = host
        self._port = port
        self._connected = False
        self._collections: dict[str, bool] = {}

    def _connect(self):
        if self._connected:
            return
        from pymilvus import connections

        connections.connect(host=self._host, port=self._port)
        self._connected = True

    def _ensure_collection(self, name: str, dim: int):
        if name in self._collections:
            return
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        self._connect()
        if utility.has_collection(name):
            self._collections[name] = True
            return
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields)
        col = Collection(name, schema)
        col.create_index(
            "embedding",
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        col.load()
        self._collections[name] = True

    def create_collection(self, name: str, dim: int) -> None:
        self._ensure_collection(name, dim)

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
        self._ensure_collection(collection, dim)
        from pymilvus import Collection

        col = Collection(collection)
        entities = []
        for chunk in chunks:
            emb = next((v for v in (chunk.embeddings or {}).values() if v), None)
            entities.append([chunk.chunk_id, chunk.text, emb])
        col.insert(entities)

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[SearchResult]:
        self._ensure_collection(collection, len(query_vector))
        from pymilvus import Collection

        col = Collection(collection)
        col.load()
        results = col.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["id", "text"],
        )
        if not results or not results[0]:
            return []
        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id=hit.entity.get("id"),
                    text=hit.entity.get("text", ""),
                ),
                score=hit.score,
                rank=rank,
            )
            for rank, hit in enumerate(results[0])
        ]

    def delete(self, collection: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        from pymilvus import Collection

        col = Collection(collection)
        expr = " || ".join(f'id == "{cid}"' for cid in chunk_ids)
        col.delete(expr)

    def delete_collection(self, name: str) -> None:
        from pymilvus import utility

        utility.drop_collection(name)
        self._collections.pop(name, None)

    def get_all_chunks(self, collection: str) -> list[Chunk]:
        from pymilvus import Collection

        col = Collection(collection)
        col.load()
        results = col.query(expr="id != ''", output_fields=["id", "text", "embedding"])
        return [
            Chunk(
                chunk_id=r["id"],
                text=r["text"],
                embeddings={"milvus": r["embedding"]} if "embedding" in r else {},
            )
            for r in results
        ]
