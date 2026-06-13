"""
示例 4：使用 PGVectorStore 的模块化 RAG 系统

演示：
  - 使用 PostgreSQL + pgvector 作为向量存储
  - 索引文本到知识库
  - 检索 + LLM 生成（带引用来源）

用法：
  uv run python rag/examples/04_pgvector_rag.py
  或 python rag/examples/04_pgvector_rag.py

依赖：
  pip install psycopg2-binary pgvector
  PostgreSQL 需运行并启用 pgvector 扩展
"""

import os
import sys
import tempfile
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / "rag" / ".env")

from rag import RAGSystem
from rag.store import PGVectorStore

PG_HOST = os.getenv("PGHOST", "172.17.19.74")
PG_PORT = int(os.getenv("PGPORT", "5432"))
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASSWORD = os.getenv("PGPASSWORD", "123456")
PG_DB = os.getenv("PGDATABASE", "knowledge")
COLLECTION = "example_docs"

store = PGVectorStore(
    host=PG_HOST,
    port=PG_PORT,
    user=PG_USER,
    password=PG_PASSWORD,
    dbname=PG_DB,
)
store.create_collection(COLLECTION, 1536)

base_url = os.getenv(
    "OPENAI_BASE_URL", "https://models.github.ai/inference/chat/completions"
)
embed_base_url = os.getenv("EMBED_BASE_URL", "")

rag = RAGSystem(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=base_url,
    embed_base_url=embed_base_url,
    embed_model=os.getenv("EMBED_MODEL", "openai/text-embedding-3-small"),
    llm_model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
    vector_store=store,
    config={"collection": COLLECTION},
)

docs = [
    (
        "rag_intro.md",
        "RAG（Retrieval-Augmented Generation）是一种检索增强生成技术。"
        "它通过先检索知识库中的相关文档，再将检索结果作为上下文提供给"
        "大语言模型，从而生成更准确、更有依据的回答。",
    ),
    (
        "pgvector_intro.md",
        "PostgreSQL 配合 pgvector 扩展可以实现高效的向量相似度搜索。"
        "pgvector 支持余弦距离、欧氏距离和内积等多种距离度量，"
        "并支持 IVFFlat 和 HNSW 索引以加速检索。",
    ),
    (
        "modular_rag.md",
        "模块化 RAG 系统由以下几部分组成：索引管线（加载→分块→嵌入→存储）、"
        "检索管线（查询改写→混合检索→重排序）和生成管线（上下文组装→LLM→引用解析）。",
    ),
]

print("=" * 50)
print("索引文档到 PGVectorStore...")

tmpdir = Path(tempfile.mkdtemp(prefix="rag_example_"))
for fname, content in docs:
    fp = tmpdir / fname
    fp.write_text(content, encoding="utf-8")
    n = rag.index_document(str(fp))
    print(f"  {fname} → {n} 个块")

print("\n" + "=" * 50)
print("交互查询模式（输入 /exit 退出）")
print("示例: 什么是 RAG？")

while True:
    q = input("\n问题: ").strip()
    if not q:
        continue
    if q.lower() in ("/exit", "/quit"):
        break
    result = rag.query(q, top_k=3)
    print(f"回答: {result.answer}")
    print(f"置信度: {result.confidence:.2f}")

print("\n" + "=" * 50)
print("清理...")
store.delete_collection(COLLECTION)
store.close()
for f in tmpdir.iterdir():
    f.unlink()
tmpdir.rmdir()
print("完成!")
