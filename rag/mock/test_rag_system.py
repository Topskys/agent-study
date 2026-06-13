"""
RAGSystem Mock 测试——验证统一入口的完整流程
"""

import sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag import RAGSystem
from rag.embed import BaseEmbedder
from rag.generate.base import BaseGenerator
from rag.datatypes import RAGResult


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
        return "Mock answer based on: " + context[:60]

    def generate_with_sources(self, query, chunks, **kwargs):
        return RAGResult(answer="Mock answer.", sources=[])


t_start = time.time()

print("init...")
rag = RAGSystem(
    embedder=MockEmbedder(),
    generator=MockGenerator(),
)
print("init OK")

# Test 1: index_document
test_file = str(Path(__file__).resolve())
n = rag.index_document(test_file, "test")
assert n > 0, "index_document failed"
print("[1] index_document: %d chunks" % n)

# Test 2: batch_index (self directory)
results = rag.batch_index(str(Path(__file__).resolve().parent), "test")
assert len(results) > 0
print("[2] batch_index: %d files" % len(results))

# Test 3: query
result = rag.query("What is RAG?", top_k=3)
assert result is not None
assert len(result.answer) > 0
print(
    "[3] query: answer=%d chars, sources=%d, confidence=%.2f"
    % (len(result.answer), len(result.sources), result.confidence)
)

# Test 4: clear_index
rag.clear_index("test")
print("[4] clear_index: OK")

elapsed = time.time() - t_start
print("ALL PASSED (%.2fs)" % elapsed)
