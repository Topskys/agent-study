"""审计日志：记录每次 LLM 调用（输入摘要 + 输出 + 延迟 + token）。

用途：调试链路、输入输出审计、用量统计。默认输出到标准库 logging（logger 名 llm_client.audit）。
"""

import logging
import time

from .types import LLMRequest, LLMResponse

logger = logging.getLogger("llm_client.audit")


def log_call(
    req: LLMRequest,
    resp: LLMResponse | None = None,
    error: str | None = None,
    duration_ms: float = 0.0,
    model: str = "",
):
    """记录一次 LLM 调用。resp 与 error 至少一个非空。"""
    content_preview = ""
    if resp and resp.content:
        content_preview = resp.content[:200]
    record = {
        "task": req.task,
        "model": model or req.model or "",
        "stream": req.stream,
        "messages": len(req.messages),
        "tools": len(req.tools) if req.tools else 0,
        "duration_ms": round(duration_ms, 1),
        "content_preview": content_preview,
        "error": error or "",
    }
    if resp and resp.usage:
        record["prompt_tokens"] = resp.usage.prompt_tokens
        record["completion_tokens"] = resp.usage.completion_tokens
        record["total_tokens"] = resp.usage.total_tokens
    if error:
        logger.warning("llm call failed: %s", record)
    else:
        logger.info("llm call ok: %s", record)


class Audit:
    """可注入的审计装饰器：包一层 chat / stream / embed 调用。"""

    def __init__(self, enable: bool = True):
        self.enable = enable

    def chat(self, fn, req: LLMRequest):
        if not self.enable:
            return fn()
        start = time.perf_counter()
        try:
            resp = fn()
            log_call(
                req,
                resp,
                duration_ms=(time.perf_counter() - start) * 1000,
                model=resp.model,
            )
            return resp
        except Exception as e:
            log_call(
                req,
                error=f"{type(e).__name__}: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            raise
