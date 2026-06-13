"""
示例 3：查看已索引的向量化内容

用法：
  python rag/examples/03_查看索引.py
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

index_file = Path(__file__).resolve().parent.parent / "index_data.pkl"

if not index_file.exists():
    print("暂无索引数据，请先运行:")
    print("  python rag/examples/01_basic_rag.py")
    sys.exit(1)

with open(index_file, "rb") as f:
    chunks = pickle.load(f)

print(f"共 {len(chunks)} 个索引块\n")

for i, c in enumerate(chunks, 1):
    print(f"=== 块 {i} ===")
    print(f"  ID:       {c['id']}")
    print(f"  来源:     {c['metadata'].get('source', '-')}")
    print(f"  向量维度: {len(c['embedding'])}")
    print(f"  向量前8维: {[round(v, 6) for v in c['embedding'][:8]]}")
    print(f"  文本内容:")
    for line in c["text"].split("\n"):
        print(f"    {line}")
    print()
