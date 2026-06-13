"""
示例 2：批量索引目录

演示：
  - 递归扫描目录下所有 .txt / .md 文件
  - 批量索引并持久化
  - 重新加载后继续查询

用法：
  python rag/examples/02_batch_index.py <目录路径>

  不传参数则默认扫描当前目录下的 .txt/.md 文件
"""

import os
import sys
import pickle
from pathlib import Path

# 确保项目根目录在路径中
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from rag import BasicRAG

INDEX_FILE = _project_root / "rag" / "index_data.pkl"


def load_index(rag: BasicRAG):
    if INDEX_FILE.exists():
        with open(INDEX_FILE, "rb") as f:
            rag.chunks = pickle.load(f)
        print(f"加载已有索引: {len(rag.chunks)} 个块")


def save_index(rag: BasicRAG):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(rag.chunks, f)
    print(f"索引已保存: {len(rag.chunks)} 个块")


def index_directory(rag: BasicRAG, root: str):
    root = Path(root)
    if not root.exists():
        print(f"目录不存在: {root}")
        return

    total = 0
    if root.is_file():
        n = rag.index_file(str(root))
        print(f"  已索引: {root.name} → {n} 块")
        total += n
    else:
        for f in sorted(root.rglob("*")):
            if f.suffix.lower() in (".txt", ".md"):
                try:
                    n = rag.index_file(str(f))
                    print(f"  已索引: {f.relative_to(root)} → {n} 块")
                    total += n
                except Exception as e:
                    print(f"  失败: {f.name} → {e}")

    print(f"\n共索引 {total} 个块")
    save_index(rag)


def interactive_query(rag: BasicRAG):
    print("\n输入问题查询（输入 /exit 退出）：")
    while True:
        q = input("\n问题: ").strip()
        if not q:
            continue
        if q.lower() in ("/exit", "/quit"):
            break
        result = rag.query(q, top_k=5)
        print(f"回答: {result['answer']}")
        print(f"置信度: {result['confidence']:.2f}")


if __name__ == "__main__":
    rag = BasicRAG(
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"
        ),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-ada-002"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )

    load_index(rag)

    if len(sys.argv) > 1:
        index_directory(rag, sys.argv[1])
    else:
        print("用法: python examples/02_batch_index.py <目录或文件路径>")
        print("例如: python examples/02_batch_index.py 20260606/")
        print("无参数时进入查询模式")
        interactive_query(rag)
