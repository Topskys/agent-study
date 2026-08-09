"""计算器工具：eval 实现 + AST 白名单防注入。"""

import ast
import math

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
        raise ValueError("非法函数调用")
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