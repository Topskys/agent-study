"""LLM 输出宽松 JSON 解析 + 带超时调用（自包含，不依赖 intent_recognizer）。

对齐 V3 ComplexIntentParser：FunctionCall 结果解析失败自动重试 2 次，
仍失败降级为规则层兜底。
"""

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _fix_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_json_block(text: str) -> str | None:
    text = _strip_code_fence(text)
    in_str = False
    escape = False
    start = -1
    stack: list[str] = []
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
        elif ch in "}]" and stack and stack[-1] == pairs[ch]:
            stack.pop()
            if not stack:
                return text[start : i + 1]
    return None


def _loads_loose(text: str) -> Any | None:
    raw = _extract_json_block(text)
    if raw is None:
        return None
    cleaned = _fix_trailing_commas(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    single = re.sub(r"(?<!\\)'(.*?)(?<!\\)'", r'"\1"', cleaned)
    try:
        return json.loads(single)
    except json.JSONDecodeError:
        return None


def parse_intent_json(text: str) -> list[dict[str, Any]]:
    """解析 ComplexIntentParser 输出（意图数组）。失败返回 []。"""
    obj = _loads_loose(text or "")
    if isinstance(obj, list):
        return [d for d in obj if isinstance(d, dict)]
    if isinstance(obj, dict):
        return [obj] if obj else []
    return []


def call_with_timeout(fn, timeout: float = 0, default=None):
    """带超时调用；timeout<=0 不设超时（由宿主控制）。"""
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