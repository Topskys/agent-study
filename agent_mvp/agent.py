"""最小 Agent 实现。

四大积木：
- 模型：OpenAI 兼容接口，决定输出内容与工具调用
- 工具：function calling JSON 格式描述，run_tool 执行
- 记忆：messages 列表记录对话历史
- 循环：while 循环判断继续调工具还是直接输出答案

依赖：openai、pyyaml。
"""

import ast
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
from openai import OpenAI

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

SYSTEM_PROMPT = "你是一个可以调用工具来帮助用户的智能助手，请用中文回答。"

# ---------- 1. 工具交付：function calling 的 JSON 格式描述 ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持四则运算、括号及常用数学函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '3 * (2 + 5)' 或 'sqrt(16) + log(100, 10)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文本文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要读取的文件路径，支持绝对路径或相对当前工作目录的路径",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


# ---------- 2. 工具执行：实际逻辑 + 防注入 ----------
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
_MATH_FUNCS = {
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
        isinstance(node.func, ast.Name) and node.func.id in _MATH_FUNCS
    ):
        raise ValueError(f"非法函数调用")
    for child in ast.iter_child_nodes(node):
        _check_node(child)


def calculator(expression: str) -> str:
    """用 eval 实现算数，AST 白名单防注入。"""
    tree = ast.parse(expression.strip(), mode="eval")
    _check_node(tree)
    globals_map = {"__builtins__": {}}
    for name in _MATH_FUNCS:
        if hasattr(math, name):
            globals_map[name] = getattr(math, name)
    globals_map["round"] = round
    return str(eval(compile(tree, "<calc>", "eval"), globals_map, {}))


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


MAX_READ_CHARS = 20000


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


TOOL_FUNCS = {
    "calculator": calculator,
    "get_time": get_time,
    "read_file": read_file,
}


def run_tool(name: str, args: dict) -> str:
    """工具分发：执行工具并返回可写回 messages 的文本结果。"""
    if name not in TOOL_FUNCS:
        return f"错误: 未知工具 {name}"
    try:
        return str(TOOL_FUNCS[name](**args))
    except Exception as e:
        return f"错误: {name} 执行失败: {e}"


# ---------- 3. 配置加载 ----------
def load_config(path: str | os.PathLike = DEFAULT_CONFIG_PATH) -> dict:
    """从 YAML 加载配置，并支持 ${ENV_VAR} 占位符替换。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    import re

    def _sub(m):
        return os.environ.get(m.group(1), "")

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, text)
    cfg = yaml.safe_load(text) or {}
    if not cfg.get("agent", {}).get("system_prompt"):
        cfg.setdefault("agent", {})["system_prompt"] = SYSTEM_PROMPT
    return cfg


# ---------- 4. 循环决策：最小 Agent 主循环 ----------
def _call_with_retry(client: OpenAI, model: str, messages: list, max_retries: int = 3):
    """带指数退避的调用：处理 429/5xx（如 NVIDIA 529 过载）等瞬时错误。"""
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS
            )
        except Exception as e:
            status = getattr(e, "status_code", None)
            if attempt < max_retries and (status == 429 or (status and status >= 500)):
                wait = 2**attempt
                print(
                    f"  [retry] {status} 服务繁忙，{wait}s 后重试（{attempt + 1}/{max_retries}）"
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("重试次数用尽")


class Agent:
    """最小 Agent：记忆（self.memory）随实例存续，跨轮次记住对话。"""

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | os.PathLike = DEFAULT_CONFIG_PATH,
    ):
        cfg = config or load_config(config_path)
        llm = cfg.get("llm", {})
        self.api_key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
        self.model = llm.get("model") or os.environ.get("LLM_MODEL")
        self.temperature = float(llm.get("temperature", 1.0))
        self.top_p = float(llm.get("top_p", 0.95))
        self.max_tokens = int(llm.get("max_tokens", 16384))
        self.timeout = int(llm.get("timeout", 120))
        self.max_retries = int(llm.get("max_retries", 3))
        self.max_rounds = int(cfg.get("agent", {}).get("max_rounds", 10))
        self.system_prompt = cfg.get("agent", {}).get("system_prompt", SYSTEM_PROMPT)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.memory: list = [{"role": "system", "content": self.system_prompt}]

    def reset(self):
        """清空记忆，开始全新会话。"""
        self.memory = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_input: str, max_rounds: int | None = None) -> str:
        self.memory.append({"role": "user", "content": user_input})
        max_rounds = max_rounds or self.max_rounds

        for _ in range(max_rounds):
            resp = _call_with_retry(
                self.client, self.model, self.memory, max_retries=self.max_retries
            )
            msg = resp.choices[0].message
            self.memory.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            )

            if not msg.tool_calls:
                return msg.content or "（无输出）"

            for call in msg.tool_calls:
                fn = call.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(fn.name, args)
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn.name,
                        "content": result,
                    }
                )
                print(
                    f"  [tool] {fn.name}({json.dumps(args, ensure_ascii=False)}) -> {result}"
                )

        return "达到最大迭代轮数，未获得最终答案"


def run_agent(question: str, max_rounds: int = 10) -> str:
    """一次性调用（无状态），每次新建会话。需要跨轮次记忆请用 Agent。"""
    return Agent().run(question, max_rounds)


if __name__ == "__main__":
    agent = Agent()
    print("Agent MVP（/exit 退出；/reset 清空记忆）")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("/exit", "/quit"):
            break
        if question.strip().lower() in ("/reset", "/clear"):
            agent.reset()
            print("记忆已清空")
            continue
        print(f"Agent: {agent.run(question)}")
