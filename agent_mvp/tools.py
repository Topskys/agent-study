"""工具层门面：加载 FunctionCall schema，注册工具实现并统一分发。

工具实现按领域拆分在 toolbox/ 子包：
- calc.py      计算器
- basic.py     时间 / 读文件
- search.py    联网搜索（Bing / Tavily）
- location.py  IP 定位
- weather.py   天气（OpenWeatherMap / wttr.in）
- remember.py  记忆写入

三步接入新工具：
1. 在 tool_schemas.json 里加 function calling 的 JSON schema
2. 在 toolbox/ 下实现（或新增）工具函数
3. 在下方 TOOL_FUNC_MAP 注册
"""

import json
from pathlib import Path

from toolbox import (
    calculator,
    get_location,
    get_time,
    get_weather,
    read_file,
    remember,
    set_bing_search_key,
    set_memory_persist_hook,
    set_openweather_key,
    set_tavily_search_key,
    tavily_search,
    web_search,
)

_SCHEMA_PATH = Path(__file__).resolve().parent / "tool_schemas.json"

with open(_SCHEMA_PATH, encoding="utf-8") as f:
    TOOLS = json.load(f)

TOOL_FUNC_MAP = {
    "calculator": calculator,
    "get_time": get_time,
    "read_file": read_file,
    "remember": remember,
    "web_search": web_search,
    "tavily_search": tavily_search,
    "get_location": get_location,
    "get_weather": get_weather,
}

__all__ = [
    "TOOLS",
    "TOOL_FUNC_MAP",
    "set_bing_search_key",
    "set_memory_persist_hook",
    "set_openweather_key",
    "set_tavily_search_key",
]


def run_tool(name: str, args: dict) -> str:
    """工具分发：执行工具并返回可写回 messages 的文本结果。"""
    if name not in TOOL_FUNC_MAP:
        return f"错误: 未知工具 {name}"
    try:
        return str(TOOL_FUNC_MAP[name](**args))
    except Exception as e:
        return f"错误: {name} 执行失败: {e}"