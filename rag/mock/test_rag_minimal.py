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
        return "mock"

    def generate_with_sources(self, query, chunks, **kwargs):
        return RAGResult()


t = time.time()
rag = RAGSystem(embedder=MockEmbedder(), generator=MockGenerator())
print("[1] init %.2fs" % (time.time() - t))

t = time.time()
n = rag.index_document(str(Path(__file__).resolve()), "test")
print("[2] index %d chunks %.2fs" % (n, time.time() - t))

t = time.time()
r = rag.query("test", top_k=3)
print("[3] query %d chars %.2fs" % (len(r.answer), time.time() - t))

print("ALL PASSED")
