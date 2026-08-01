"""工具层：工具实现与分发。

三步接入新工具：
1. 在 tool_schemas.json 里加 function calling 的 JSON schema
2. 在本文件写实现函数
3. 注册进 TOOL_FUNCS
"""

import ast
import json
import math
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


TOOL_FUNC_MAP = {
    "calculator": calculator,
    "get_time": get_time,
    "read_file": read_file,
}


def run_tool(name: str, args: dict) -> str:
    """工具分发：执行工具并返回可写回 messages 的文本结果。"""
    if name not in TOOL_FUNC_MAP:
        return f"错误: 未知工具 {name}"
    try:
        return str(TOOL_FUNC_MAP[name](**args))
    except Exception as e:
        return f"错误: {name} 执行失败: {e}"
