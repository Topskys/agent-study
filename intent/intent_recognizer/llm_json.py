"""LLM 输出宽松 JSON 解析与带超时调用工具。

应对常见不规整输出：markdown 代码块、前后缀文字、尾逗号、单双引号混用。
统一约定：
- 阶段一输出意图数组（[...]）；
- 阶段二输出槽位数组（[...]）；
- 解析失败返回空结构，交由上层重试 / 规则兜底。
"""

import json
import re
from typing import Any, Dict, List, Optional


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json_block(text: str) -> Optional[str]:
    """提取文本中第一个完整 JSON 块（{...} 或 [...]），跳过字符串内的括号。"""
    text = _strip_code_fence(text)
    in_str = False
    escape = False
    start = -1
    stack: List[str] = []
    pairs = {"}": "{", "]": "["}
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if start == -1:
                start = i
            stack.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
                if not stack:
                    return text[start : i + 1]
    return None


def _loads_loose(text: str) -> Optional[Any]:
    raw = _extract_json_block(text)
    if raw is None:
        return None
    cleaned = _fix_trailing_commas(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 单引号兜底：把字符串内未转义的单引号替换为双引号后再试
    single = re.sub(r"(?<!\\)'(.*?)(?<!\\)'", r'"\1"', cleaned)
    try:
        return json.loads(single)
    except json.JSONDecodeError:
        return None


def parse_intent_array(text: str) -> List[Dict[str, Any]]:
    """解析阶段一意图数组；解析失败返回 []。"""
    obj = _loads_loose(text or "")
    if isinstance(obj, list):
        return [d for d in obj if isinstance(d, dict)]
    if isinstance(obj, dict):
        return [obj] if obj else []
    return []


def parse_slot_results(text: str) -> List[Dict[str, Any]]:
    """解析阶段二槽位数组；解析失败返回 []。"""
    obj = _loads_loose(text or "")
    if isinstance(obj, list):
        return [d for d in obj if isinstance(d, dict)]
    if isinstance(obj, dict):
        return [obj] if obj else []
    return []


def call_with_timeout(fn, timeout: float = 0, default=None):
    """带超时调用。timeout <= 0 表示不设超时（由宿主控制 LLM 超时）。

    超时返回 default；注意线程无法强制终止，仅作调用侧限时。
    """
    if not timeout or timeout <= 0:
        return fn()
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        return default
    finally:
        executor.shutdown(wait=False)
