"""
索引管线——Mock 测试
"""

import sys, time, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag import LoaderFactory, RecursiveChunker, MemoryStore, BM25Index, DocumentIndexer
from rag.embed import BaseEmbedder
from rag.datatypes import RawDocument


class MockEmbedder(BaseEmbedder):
    @property
    def dim(self):
        return 4

    @property
    def name(self):
        return "mock/test"

    def embed(self, texts):
        return np.random.rand(len(texts), 4).tolist()


embedder = MockEmbedder()
t_start = time.time()

import os

os.makedirs("rag/mock", exist_ok=True)  # ensure dir exists

t = time.time()
doc = RawDocument(
    content="RAG is retrieval augmented generation. " * 200, source="mock.txt"
)
chunks = RecursiveChunker(200, 20).chunk(doc)
print("[1] %d chunks (%.2f)" % (len(chunks), time.time() - t))

t = time.time()
embs = embedder.embed([c.text for c in chunks])
for c, emb in zip(chunks, embs):
    c.embeddings[embedder.name] = emb
store = MemoryStore()
store.create_collection("test", embedder.dim)
store.insert("test", chunks)
print("[2] %d vectors (%.2f)" % (len(chunks), time.time() - t))

t = time.time()
q = embedder.embed(["RAG retrieval"])[0]
r = store.search("test", q, 3)
print("[3] %d results top=%.4f" % (len(r), r[0].score))

t = time.time()
bm25 = BM25Index()
bm25.add_documents("test", chunks)
r2 = bm25.search("test", "retrieval augmented", 3)
print("[4] %d results top=%.4f (%.2f)" % (len(r2), r2[0].score, time.time() - t))

t = time.time()
indexer = DocumentIndexer(
    LoaderFactory(), RecursiveChunker(128, 16), embedder, store, bm25
)
n = indexer.index_document(
    str(Path(__file__).resolve().parent.parent.parent / "rag/examples/01_basic_rag.py"),
    "code",
)
print("[5] %d chunks (%.2f)" % (n, time.time() - t))

print("ALL PASSED (%.2f)" % (time.time() - t_start))
