"""
向量存储——PostgreSQL + pgvector 实现

生产级向量存储，支持余弦相似度检索和 IVFFlat 索引。
依赖: psycopg2-binary, pgvector
"""

import time
import uuid
from rag.datatypes import Chunk, SearchResult
from .base import BaseVectorStore


class PGVectorStore(BaseVectorStore):
    """PostgreSQL + pgvector 向量存储"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        dbname: str = "postgres",
        table_prefix: str = "rag_",
        autocommit: bool = True,
        connect_timeout: int = 10,
    ):
        self._conn_params = dict(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname,
            connect_timeout=connect_timeout,
        )
        self._table_prefix = table_prefix
        self._autocommit = autocommit
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg2

            self._conn = psycopg2.connect(**self._conn_params)
            if self._autocommit:
                self._conn.autocommit = True
            self._ensure_pgvector()
        return self._conn

    def _ensure_pgvector(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def _table_name(self, collection: str) -> str:
        safe = collection.replace("-", "_").replace(".", "_")
        return f"{self._table_prefix}{safe}"

    def _ensure_collection(self, name: str, dim: int):
        table = self._table_name(name)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding vector({dim}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    def create_collection(self, name: str, dim: int) -> None:
        self._ensure_collection(name, dim)

    def insert(self, collection: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        table = self._table_name(collection)
        first_emb = next(
            (v for chunk in chunks for v in (chunk.embeddings or {}).values() if v),
            None,
        )
        if first_emb is None:
            raise ValueError("Chunks 缺少嵌入向量")
        dim = len(first_emb)
        self._ensure_collection(collection, dim)

        import json, psycopg2.extras

        conn = self._get_conn()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {table} (id, text, embedding, metadata) VALUES %s "
                f"ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text, "
                f"embedding=EXCLUDED.embedding, metadata=EXCLUDED.metadata",
                [
                    (
                        c.chunk_id,
                        c.text,
                        next(v for v in (c.embeddings or {}).values() if v),
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
                template="(%s, %s, %s::vector, %s::jsonb)",
            )

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[SearchResult]:
        table = self._table_name(collection)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, embedding::text, metadata, "
                f"1 - (embedding <=> %s::vector) AS score "
                f"FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_vector, query_vector, top_k),
            )
            rows = cur.fetchall()

        return [
            SearchResult(
                chunk=Chunk(
                    chunk_id=row[0],
                    text=row[1],
                    metadata=dict(row[3]) if row[3] else {},
                    embeddings={"pgvector": list(map(float, row[2][1:-1].split(",")))},
                ),
                score=float(row[4]),
                rank=rank,
            )
            for rank, row in enumerate(rows)
        ]

    def delete(self, collection: str, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        table = self._table_name(collection)
        conn = self._get_conn()
        import psycopg2.extras

        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"DELETE FROM {table} WHERE id IN (%s)",
                [(cid,) for cid in chunk_ids],
                template="(%s)",
            )

    def delete_collection(self, name: str) -> None:
        table = self._table_name(name)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")

    def get_all_chunks(self, collection: str) -> list[Chunk]:
        table = self._table_name(collection)
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, text, embedding::text, metadata FROM {table} "
                f"ORDER BY created_at"
            )
            rows = cur.fetchall()

        return [
            Chunk(
                chunk_id=row[0],
                text=row[1],
                metadata=dict(row[3]) if row[3] else {},
                embeddings={"pgvector": list(map(float, row[2][1:-1].split(",")))},
            )
            for row in rows
        ]

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
