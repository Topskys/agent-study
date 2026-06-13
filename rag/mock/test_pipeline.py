"""
管线 Mock 测试——检索 + 生成端到端
"""

import sys, time, uuid, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.datatypes import Chunk, RawDocument, SearchResult
from rag.embed import BaseEmbedder
from rag.chunker import RecursiveChunker
from rag.store import MemoryStore, BM25Index
from rag.retrieve import HybridRetriever, QueryRewriter, Reranker, RetrievalPipeline
from rag.generate import ContextBuilder, CitationTracker, GenerationPipeline
from rag.generate.base import BaseGenerator


class MockEmbedder(BaseEmbedder):
    @property
    def dim(self):
        return 4

    @property
    def name(self):
        return "mock/test"

    def embed(self, texts):
        return np.random.rand(len(texts), 4).tolist()


class MockGenerator(BaseGenerator):
    def generate(self, query, context, **kwargs):
        return "Mock answer based on the provided context."

    def generate_with_sources(self, query, chunks, **kwargs):
        from rag.datatypes import RAGResult

        return RAGResult(
            answer="Mock answer.",
            sources=[
                {
                    "id": i + 1,
                    "source": c.metadata.get("source", ""),
                    "text": c.text[:100],
                }
                for i, c in enumerate(chunks)
            ],
        )


t_start = time.time()

embedder = MockEmbedder()
store = MemoryStore()
store.create_collection("test", 4)
bm25 = BM25Index()

doc = RawDocument(
    content="RAG is retrieval augmented generation. " * 200, source="mock.txt"
)
chunks = RecursiveChunker(200, 20).chunk(doc)
texts = [c.text for c in chunks]
embs = embedder.embed(texts)
for c, emb in zip(chunks, embs):
    c.embeddings[embedder.name] = emb
store.insert("test", chunks)
bm25.add_documents("test", chunks)
print("[1] indexed %d chunks" % len(chunks))

# Test HybridRetriever
hybrid = HybridRetriever(store, bm25)
q_vec = embedder.embed(["RAG retrieval"])[0]
results = hybrid.retrieve("test", q_vec, "retrieval augmented", top_k=3)
assert len(results) > 0, "HybridRetriever returned no results"
print("[2] HybridRetriever: %d results, top=%.4f" % (len(results), results[0].score))

# Test RetrievalPipeline
pipeline = RetrievalPipeline(embedder, hybrid)
results2 = pipeline.retrieve("RAG retrieval", "test", top_k=3)
assert len(results2) > 0, "RetrievalPipeline returned no results"
print(
    "[3] RetrievalPipeline: %d results, top=%.4f" % (len(results2), results2[0].score)
)

# Test Reranker
reranker = Reranker(strategy="diversity")
results3 = reranker.rerank("RAG retrieval", results2)
assert len(results3) > 0
print("[4] Reranker(diversity): %d results" % len(results3))

# Test QueryRewriter
rewriter = QueryRewriter(strategy="identity")
rewritten = rewriter.rewrite("RAG retrieval")
assert rewritten == "RAG retrieval"
print("[5] QueryRewriter(identity): OK")

# Test ContextBuilder
ctx = ContextBuilder()
context = ctx.build(chunks[:3])
assert len(context) > 0
print("[6] ContextBuilder: %d chars" % len(context))

# Test CitationTracker
ct = CitationTracker()
sources = ct.track("According to [1], RAG is great. [2] agrees.", chunks[:3])
assert len(sources) == 2
print("[7] CitationTracker: %d sources" % len(sources))

# Test GenerationPipeline with MockGenerator
gen_pipeline = GenerationPipeline(generator=MockGenerator())
rag_result = gen_pipeline.generate("What is RAG?", chunks[:3])
assert len(rag_result.answer) > 0
print(
    "[8] GenerationPipeline: answer=%d chars, %d sources"
    % (len(rag_result.answer), len(rag_result.sources))
)

elapsed = time.time() - t_start
print("ALL PASSED (%.2fs)" % elapsed)
