"""
RAGSystem API 真实端到端测试
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

from rag import RAGSystem

t_start = time.time()

rag = RAGSystem(
    api_key=api_key,
    base_url=base_url,
    config={
        "chunk_size": 300,
        "chunk_overlap": 30,
        "collection": "rag_system_test",
    },
)

# Index a test file
test_file = str(Path(__file__).resolve().parent.parent / "examples/01_basic_rag.py")
n = rag.index_document(test_file)
print("[1] indexed %d chunks (%.2fs)" % (n, time.time() - t_start))

# Query
result = rag.query("What is BasicRAG?", top_k=3)
print(
    "[2] query: answer=%.80s... (%d sources, %.2fs)"
    % (result.answer, len(result.sources), time.time() - t_start)
)

rag.clear_index()
print("[3] clear OK")

elapsed = time.time() - t_start
print("ALL PASSED (%.2fs)" % elapsed)
