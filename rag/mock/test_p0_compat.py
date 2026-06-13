"""
P0 兼容性验证——确认 BasicRAG 与新模块无冲突
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Test 1: 同时导入 BasicRAG 和新模块
from rag import (
    BasicRAG,
    DocumentIndexer,
    RetrievalPipeline,
    GenerationPipeline,
    QueryRewriter,
    Reranker,
    ApiEmbedder,
    RAGGenerator,
    MemoryStore,
    BM25Index,
    RecursiveChunker,
    LoaderFactory,
)

print("[1] 新旧模块同时导入: OK")

# Test 2: BasicRAG 实例化
rag = BasicRAG()
assert rag is not None
assert rag.chunks == []
print("[2] BasicRAG 实例化: OK")

# Test 3: 新模块实例化
store = MemoryStore()
bm25 = BM25Index()
chunker = RecursiveChunker(200, 20)
factory = LoaderFactory()
indexer = DocumentIndexer(factory, chunker, None, store)  # type: ignore
print("[3] DocumentIndexer(无 embedder) 实例化: OK")

# Test 4: 管线类可引用
assert RetrievalPipeline is not None
assert GenerationPipeline is not None
assert QueryRewriter is not None
assert Reranker is not None
print("[4] 管线类引用: OK")

gen = GenerationPipeline()
assert gen is not None
print("[5] GenerationPipeline 实例化: OK")

# Test 5: BasicRAG 的部分能力（无需 API）
from rag.datatypes import RawDocument, Chunk, SearchResult, RAGResult

doc = RawDocument(content="test", source="test.txt")
assert doc.content == "test"
chunk = Chunk(chunk_id="1", text="test")
assert chunk.text == "test"
print("[6] 数据类型共用: OK")

print("ALL PASSED")
