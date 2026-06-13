"""
示例 5：将 RAG 全景文章向量化到 PGVectorStore，然后交互查询

用法：
  uv run python rag/examples/05_index_rag_article.py
"""

import os
import sys
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
COLLECTION = "rag_article_2026"

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

article_path = (
    _project_root / "rag" / "assets" / "2026 RAG全景：从大模型基座到Agent记忆中枢.md"
)
print(f"索引: {article_path.name}")
n = rag.index_document(str(article_path))
print(f"  生成 {n} 个块")

print("\n交互查询模式（输入 /exit 退出）")
print("示例: RAG 的架构是什么？")
while True:
    q = input("\n问题: ").strip()
    if not q:
        continue
    if q.lower() in ("/exit", "/quit"):
        break
    result = rag.query(q, top_k=3)
    print(f"\n回答: {result.answer}")
    print(f"\n置信度: {result.confidence:.2f}")

print("\n数据保留在 PGVectorStore 中，collection=%s" % COLLECTION)
print("如需清理，运行: store.delete_collection('%s')" % COLLECTION)
store.close()
