"""工具层：工具实现与分发。

三步接入新工具：
1. 在 tool_schemas.json 里加 function calling 的 JSON schema
2. 在本文件写实现函数
3. 注册进 TOOL_FUNCS
"""

import ast
import json
import math
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parent / "tool_schemas.json"

with open(_SCHEMA_PATH, encoding="utf-8") as f:
    TOOLS = json.load(f)

MAX_READ_CHARS = 20000

# ---------- calculator：eval 实现 + AST 白名单防注入 ----------
_SAFE_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
)
_MATH_FUNC_MAP = {
    "sqrt",
    "log",
    "log2",
    "log10",
    "abs",
    "pow",
    "exp",
    "floor",
    "ceil",
    "round",
    "sin",
    "cos",
    "tan",
    "pi",
    "e",
}


def _check_node(node):
    if not isinstance(node, _SAFE_NODES):
        raise ValueError(f"非法语法: {type(node).__name__}")
    if isinstance(node, ast.Call) and not (
        isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNC_MAP
    ):
        raise ValueError(f"非法函数调用")
    for child in ast.iter_child_nodes(node):
        _check_node(child)


def calculator(expression: str) -> str:
    """用 eval 实现算数，AST 白名单防注入。"""
    tree = ast.parse(expression.strip(), mode="eval")
    _check_node(tree)
    globals_map = {"__builtins__": {}}
    for name in _MATH_FUNC_MAP:
        if hasattr(math, name):
            globals_map[name] = getattr(math, name)
    globals_map["round"] = round
    return str(eval(compile(tree, "<calc>", "eval"), globals_map, {}))


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    """读取文本文件，超长内容截断返回。"""
    p = Path(path).resolve()
    if not p.exists():
        return f"文件不存在: {path}"
    if not p.is_file():
        return f"不是文件: {path}"
    content = p.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_READ_CHARS:
        return (
            content[:MAX_READ_CHARS]
            + f"\n\n...（内容过长，已截断，共 {len(content)} 字符）"
        )
    return content


# ---------- web_search：Bing Web Search API v7（联网搜索） ----------
_BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
_bing_search_api_key = os.environ.get("BING_SEARCH_API_KEY", "")


def set_bing_search_key(key: str):
    """注入 Bing 订阅密钥（宿主从配置读取后调用，未注入则读环境变量）。"""
    global _bing_search_api_key
    _bing_search_api_key = key or _bing_search_api_key


def web_search(query: str, count: int = 5) -> str:
    """联网搜索：Bing Web Search API v7，返回标题 / URL / 摘要。"""
    if not _bing_search_api_key:
        return "错误: 未配置 BING_SEARCH_API_KEY，联网搜索不可用"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "count": min(max(int(count), 1), 20),
            "mkt": "zh-CN",
            "responseFilter": "Webpages",
        }
    )
    req = urllib.request.Request(
        f"{_BING_SEARCH_URL}?{params}",
        headers={"Ocp-Apim-Subscription-Key": _bing_search_api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - 网络/限流等错误统一返回给模型
        return f"错误: 联网搜索失败: {e}"
    pages = data.get("webPages", {}).get("value", [])
    if not pages:
        return f"未搜索到「{query}」的相关结果。"
    lines = []
    for i, p in enumerate(pages[: int(count)], 1):
        lines.append(
            f"{i}. {p.get('name', '')}\n   {p.get('url', '')}\n   {p.get('snippet', '')}"
        )
    return "搜索结果：\n\n" + "\n\n".join(lines)


# ---------- tavily_search：Tavily AI 搜索（联网搜索） ----------
_tavily_search_api_key = os.environ.get("TVLY_API_KEY", "") or os.environ.get("TAVILY_API_KEY", "")


def set_tavily_search_key(key: str):
    """注入 Tavily 订阅密钥（宿主从配置读取后调用，未注入则读环境变量）。"""
    global _tavily_search_api_key
    _tavily_search_api_key = key or _tavily_search_api_key


def tavily_search(query: str, max_results: int = 5, search_depth: str = "basic") -> str:
    """联网搜索：Tavily AI Search API，返回标题 / URL / 内容摘要。
    
    Args:
        query: 搜索关键词
        max_results: 返回结果条数，默认 5
        search_depth: 搜索深度，"basic" 或 "advanced"，默认 "basic"
    """
    if not _tavily_search_api_key:
        return "错误: 未配置 TVLY_API_KEY 或 TAVILY_API_KEY，Tavily 搜索不可用"
    try:
        from tavily import TavilyClient
    except ImportError:
        return "错误: 未安装 tavily-python，请运行 `uv add tavily-python`"
    
    client = TavilyClient(api_key=_tavily_search_api_key)
    try:
        response = client.search(
            query=query,
            max_results=min(max(int(max_results), 1), 20),
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception as e:
        return f"错误: Tavily 搜索失败: {e}"
    
    results = response.get("results", [])
    if not results:
        return f"未搜索到「{query}」的相关结果。"
    
    lines = []
    for i, r in enumerate(results[: int(max_results)], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"{i}. {title}\n   {url}\n   {content}")
    
    answer = response.get("answer", "")
    output = "搜索结果：\n\n" + "\n\n".join(lines)
    if answer:
        output = f"AI 答案：{answer}\n\n" + output
    return output


# ---------- remember：由 Agent 注入记忆访问器，模型自主决定持久化 ----------
_memory_persist_hook = None


def set_memory_persist_hook(fn):
    """注入一个 callable(content, importance) -> bool，供 remember 工具调用。"""
    global _memory_persist_hook
    _memory_persist_hook = fn


def remember(content: str, importance: float = 0.8) -> str:
    """把用户信息写入长期记忆（是否调用由模型判断）。"""
    if not _memory_persist_hook:
        return "错误: 记忆系统未接入"
    ok = _memory_persist_hook(content, float(importance))
    if ok:
        return f"已记住: {content}"
    return f"未通过记忆治理裁决，未持久化: {content}"


TOOL_FUNC_MAP = {
    "calculator": calculator,
    "get_time": get_time,
    "read_file": read_file,
    "remember": remember,
    "web_search": web_search,
    "tavily_search": tavily_search,
}


def run_tool(name: str, args: dict) -> str:
    """工具分发：执行工具并返回可写回 messages 的文本结果。"""
    if name not in TOOL_FUNC_MAP:
        return f"错误: 未知工具 {name}"
    try:
        return str(TOOL_FUNC_MAP[name](**args))
    except Exception as e:
        return f"错误: {name} 执行失败: {e}"
