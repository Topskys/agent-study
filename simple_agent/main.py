import os
import sys
import pickle
from pathlib import Path

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
from simple_agent.agent import SimpleAgent, get_weather, get_time
from rag.basic_rag import BasicRAG

load_dotenv()

RAG_INDEX_FILE = Path(__file__).resolve().parent.parent / "rag" / "index_data.pkl"


def _init_rag() -> BasicRAG:
    rag = BasicRAG(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        embed_base_url=os.getenv("EMBED_BASE_URL", ""),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-ada-002"),
        llm_model=os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4-flash"),
    )
    if RAG_INDEX_FILE.exists():
        with open(RAG_INDEX_FILE, "rb") as f:
            rag.chunks = pickle.load(f)
        print(f"已加载 {len(rag.chunks)} 个索引块")
    return rag


def _save_rag(rag: BasicRAG):
    RAG_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RAG_INDEX_FILE, "wb") as f:
        pickle.dump(rag.chunks, f)
    print(f"已保存 {len(rag.chunks)} 个索引块到 {RAG_INDEX_FILE}")


def index_command(rag: BasicRAG, path: str):
    p = Path(path)
    if p.is_file():
        n = rag.index_file(str(p))
        print(f"已索引 {p.name} → {n} 个块")
    elif p.is_dir():
        total = 0
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in (".txt", ".md"):
                try:
                    n = rag.index_file(str(f))
                    print(f"  + {f.relative_to(p)} → {n} 块")
                    total += n
                except Exception as e:
                    print(f"  x {f.relative_to(p)} → {e}")
        print(f"索引完成，共 {total} 个块")
    else:
        print(f"路径不存在: {path}")
    _save_rag(rag)


def print_usage():
    print("用法:")
    print("  python -m simple_agent.main                  启动交互式 Agent")
    print("  python -m simple_agent.main --index <路径>   索引文档（文件或目录）")
    print("  python -m simple_agent.main --help           显示帮助信息")


def main():
    rag = _init_rag()

    if len(sys.argv) > 1:
        if sys.argv[1] in ("--help", "-h"):
            print_usage()
            return
        if sys.argv[1] == "--index":
            if len(sys.argv) < 3:
                print("用法: python -m simple_agent.main --index <文件或目录路径>")
                return
            index_command(rag, sys.argv[2])
            return

    agent = SimpleAgent(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        model=os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v4-flash"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "1")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        timeout=int(os.getenv("LLM_TIMEOUT", "120")),
    )
    agent.register_tool("get_weather", get_weather, "查询当前天气")
    agent.register_tool("get_time", get_time, "获取当前时间")
    agent.register_tool(
        "rag_query",
        lambda q="", top_k=5: rag.query(q, top_k),
        "基于本地知识库进行检索增强回答",
        args_desc='{"query": "问题", "top_k": 5}',
    )

    print("Simple Agent (输入 /exit 退出；/index <路径> 索引文档)")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ("/exit", "/quit"):
            _save_rag(rag)
            break
        if user_input.strip().startswith("/index "):
            index_command(rag, user_input.strip()[7:])
            continue
        try:
            reply = agent.run(user_input)
            print(f"Agent: {reply}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
