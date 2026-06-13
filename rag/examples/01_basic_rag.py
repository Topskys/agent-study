"""
示例 1：基础索引与检索

演示：
  - 直接索引文本
  - 索引文件
  - 检索相似内容（不生成）
  - 检索 + LLM 生成

用法：
  python rag/examples/01_basic_rag.py

注意：运行前需配置好环境变量或直接修改下方 API 参数
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 确保项目根目录在路径中
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

# 加载 rag/.env
load_dotenv(_project_root / "rag" / ".env")

from rag import BasicRAG


# ====== 1. 初始化 ======

rag = BasicRAG(
    api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
    embed_base_url=os.getenv("EMBED_BASE_URL", ""),
    embed_model=os.getenv("EMBED_MODEL", "text-embedding-ada-002"),
    llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
)


# ====== 2. 索引 ======

print("=" * 50)
print("索引文档...")

# 直接索引文本
rag.index_text(
    "RAG（Retrieval-Augmented Generation）是一种检索增强生成技术。"
    "它通过先检索知识库中的相关文档，再将检索结果作为上下文提供给"
    "大语言模型，从而生成更准确、更有依据的回答。"
)

rag.index_text(
    "AI Agent 是一种能够自主感知环境、做出决策并执行行动的智能程序。"
    "它以大语言模型为核心推理引擎，结合工具调用完成复杂任务。"
)

# 索引文件（如果存在，从项目根目录查找）
for f in ["测试文档.md", "README.md"]:
    fp = _project_root / f
    if fp.exists():
        n = rag.index_file(str(fp))
        print(f"  已索引: {f} → {n} 个块")

print(f"共 {len(rag.chunks)} 个索引块")


# ====== 3. 检索（不生成） ======

print("\n" + "=" * 50)
print("检索测试（不生成）...")

results = rag.search("什么是 RAG？", top_k=3)
for i, r in enumerate(results, 1):
    print(f"  [{i}] 评分={r['score']:.3f}")
    print(f"       {r['text'][:100]}...")


# ====== 4. 检索 + 生成 ======

print("\n" + "=" * 50)
print("RAG 查询测试（检索 + 生成）...")

result = rag.query("RAG 技术解决了什么问题？", top_k=3)
print(f"回答: {result['answer']}")
print(f"置信度: {result['confidence']:.2f}")
print(f"来源: {[s['source'] for s in result['sources']]}")


# ====== 5. 再次查询（复用已索引的数据） ======

print("\n" + "=" * 50)
print("另一个问题...")

result2 = rag.query("AI Agent 的核心特性有哪些？")
print(f"回答: {result2['answer']}")


# ====== 6. 清空索引 ======

print("\n" + "=" * 50)
print("清空索引...")
rag.clear()
print(f"当前索引块数: {len(rag.chunks)}")
