"""
PGVectorStore 测试

需要本地 PostgreSQL 运行且启用 pgvector 扩展，否则自动跳过。
"""

import sys, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.store import PGVectorStore
from rag.datatypes import Chunk

t_start = time.time()

# 检查 PostgreSQL 是否可用
try:
    store = PGVectorStore(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
        dbname=os.getenv("PGDATABASE", "postgres"),
        connect_timeout=3,
    )
    store.create_collection("test_pgvector", 4)
except Exception as e:
    store = None
    print("SKIP: PostgreSQL 不可用 - %s" % e)
    sys.exit(0)

# 创建测试数据
chunks = [
    Chunk(
        chunk_id="c1",
        text="RAG is retrieval augmented generation",
        metadata={"source": "test"},
        embeddings={"pgvector": [0.1, 0.2, 0.3, 0.4]},
    ),
    Chunk(
        chunk_id="c2",
        text="Vector databases store embeddings",
        metadata={"source": "test"},
        embeddings={"pgvector": [0.5, 0.6, 0.7, 0.8]},
    ),
    Chunk(
        chunk_id="c3",
        text="PostgreSQL with pgvector is a good choice",
        metadata={"source": "test"},
        embeddings={"pgvector": [0.9, 0.8, 0.7, 0.6]},
    ),
]

# 插入
store.insert("test_pgvector", chunks)
print("[1] insert 3 chunks (%.2fs)" % (time.time() - t_start))

# 检索
results = store.search("test_pgvector", [0.1, 0.2, 0.3, 0.4], top_k=2)
assert len(results) == 2, "search 应返回 2 条结果"
print("[2] search top=%.4f (%.2fs)" % (results[0].score, time.time() - t_start))

# 获取全部
all_chunks = store.get_all_chunks("test_pgvector")
assert len(all_chunks) == 3
print("[3] get_all_chunks: %d (%.2fs)" % (len(all_chunks), time.time() - t_start))

# 删除
store.delete("test_pgvector", ["c1"])
remaining = store.get_all_chunks("test_pgvector")
assert len(remaining) == 2
print("[4] delete: %d remaining (%.2fs)" % (len(remaining), time.time() - t_start))

# 清理
store.delete_collection("test_pgvector")
print("[5] cleanup (%.2fs)" % (time.time() - t_start))

store.close()
print("ALL PASSED (%.2fs)" % (time.time() - t_start))
