"""裸 HTTP Provider：兼容 base_url 指向 .../chat/completions 的场景。

场景：部分服务（如 simple_agent 用的端点）直接给到 /chat/completions，
SDK 拼接会出错；此 Provider 用 requests 直连，行为对齐 OpenAI 兼容协议。
"""

from collections.abc import Iterator
from typing import Any

import requests

from .types import Chunk, LLMRequest, LLMResponse, Message, TokenUsage, ToolCall


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [
        {
            "role": m.role,
            **({"content": m.content} if m.content is not None else {}),
            **({"name": m.name} if m.name else {}),
            **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
            **(
                {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in m.tool_calls
                    ]
                }
                if m.tool_calls
                else {}
            ),
        }
        for m in messages
    ]


def _parse_tool_calls(data: Any) -> list[ToolCall] | None:
    raw_calls = data.get("tool_calls") if isinstance(data, dict) else None
    if not raw_calls:
        return None
    out: list[ToolCall] = []
    for tc in raw_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        out.append(
            ToolCall(
                id=tc.get("id", "") if isinstance(tc, dict) else "",
                name=fn.get("name", "") or "",
                arguments=fn.get("arguments", "{}") or "{}",
            )
        )
    return out


def _to_response(payload: dict[str, Any], req: LLMRequest) -> LLMResponse:
    choices = payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) or {}
    usage = payload.get("usage") or {}
    return LLMResponse(
        content=message.get("content"),
        tool_calls=_parse_tool_calls(message),
        usage=TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        ),
        model=payload.get("model") or req.model or "",
        finish_reason=choice.get("finish_reason"),
        raw=payload,
    )


class HTTPProvider:
    """裸 HTTP Provider。base_url 需完整指向 /chat/completions 或自动补齐。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        timeout: float = 120,
        headers: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.timeout = timeout
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            **(headers or {}),
        }

    def _url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def _payload(self, req: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model or self.default_model,
            "messages": _to_openai_messages(req.messages),
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            **req.extra,
        }
        if req.tools:
            payload["tools"] = req.tools
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice
        return payload

    def chat(self, req: LLMRequest) -> LLMResponse:
        resp = requests.post(
            self._url(),
            headers=self._headers,
            json=self._payload(req),
            timeout=req.timeout or self.timeout,
        )
        resp.raise_for_status()
        return _to_response(resp.json(), req)

    def chat_stream(self, req: LLMRequest) -> Iterator[Chunk]:
        resp = requests.post(
            self._url(),
            headers=self._headers,
            json={**self._payload(req), "stream": True},
            timeout=req.timeout or self.timeout,
            stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            import json as _json

            try:
                chunk = _json.loads(data)
            except ValueError:
                continue
            yield self._chunk_from_payload(chunk)

    @staticmethod
    def _chunk_from_payload(payload: dict[str, Any]) -> Chunk:
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta", {}) or {}
        return Chunk(
            content=delta.get("content", "") or "",
            tool_calls=_parse_tool_calls(delta),
            finish_reason=choice.get("finish_reason"),
        )
