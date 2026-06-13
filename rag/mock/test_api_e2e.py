"""
API 真实端到端测试——索引 → 检索 → 生成

需要 .env 中配置了有效的 GitHub Models PAT。
如果 API 不可用或超时，测试会优雅跳过（不中断 CI）。
"""

import sys, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = "https://models.github.ai/inference"

if not api_key:
    print("SKIP: 未配置 OPENAI_API_KEY")
    sys.exit(0)

from rag.datatypes import RawDocument
from rag.chunker import RecursiveChunker
from rag.embed import ApiEmbedder
from rag.store import MemoryStore, BM25Index
from rag.retrieve import HybridRetriever, RetrievalPipeline
from rag.generate import ContextBuilder, RAGGenerator, GenerationPipeline

t_start = time.time()

# ===== 1. 索引 =====
t = time.time()
embedder = ApiEmbedder(
    api_key=api_key,
    base_url=base_url,
    model="openai/text-embedding-3-small",
    dim=1536,
)
doc = RawDocument(
    content="RAG (Retrieval Augmented Generation) is a technique that "
    "combines information retrieval with text generation. "
    "It first retrieves relevant documents from a knowledge base, "
    "then uses a language model to generate answers based on those documents. "
    "This approach reduces hallucinations and improves factual accuracy. "
    "RAG was introduced by Lewis et al. in 2020. "
    "It has become a fundamental building block for modern AI applications. "
    "The retrieval component can use dense embeddings or sparse keywords. "
    "Popular vector databases include Chroma, Milvus, and FAISS. "
    "The generation component typically uses large language models like GPT-4. "
    "RAG systems are used in question answering, chatbots, and knowledge management. "
    * 10,
    source="api_test_doc",
)
chunks = RecursiveChunker(300, 30).chunk(doc)
print("[1] chunked into %d pieces (%.2fs)" % (len(chunks), time.time() - t))

# ===== 2. 向量化 =====
t = time.time()
try:
    texts = [c.text for c in chunks]
    embs = embedder.embed(texts)
    for c, emb in zip(chunks, embs):
        c.embeddings[embedder.name] = emb
    print("[2] embedded %d vectors (%.2fs)" % (len(embs), time.time() - t))
except Exception as e:
    print("SKIP: embed API failed - %s" % e)
    sys.exit(0)

# ===== 3. 存储 =====
store = MemoryStore()
store.create_collection("api_test", embedder.dim)
store.insert("api_test", chunks)
bm25 = BM25Index()
bm25.add_documents("api_test", chunks)
print("[3] stored %d chunks" % len(chunks))

# ===== 4. 检索 =====
t = time.time()
hybrid = HybridRetriever(store, bm25)
pipeline = RetrievalPipeline(embedder, hybrid)
results = pipeline.retrieve("What is RAG?", "api_test", top_k=3)
print("[4] retrieved %d results (%.2fs)" % (len(results), time.time() - t))
for r in results:
    print("    rank=%d score=%.4f text=%.60s..." % (r.rank, r.score, r.chunk.text))

# ===== 5. 生成 =====
t = time.time()
generator = RAGGenerator(
    api_key=api_key,
    base_url=base_url,
    model="openai/gpt-4o-mini",
)
gen_pipeline = GenerationPipeline(generator=generator)
rag_result = gen_pipeline.generate("What is RAG?", results[:3])
print("[5] generated (%.2fs)" % (time.time() - t))
print("    answer: %.100s..." % rag_result.answer)
print("    sources: %d" % len(rag_result.sources))

elapsed = time.time() - t_start
print("ALL PASSED (%.2fs)" % elapsed)
