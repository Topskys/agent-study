"""OpenAI 兼容 Provider：基于 openai SDK。

覆盖 ZEN / NVIDIA / DeepSeek / Ollama / vLLM 等一切 OpenAI 兼容端点。
将 SDK 原生对象规整为 llm_client.types 的统一结构。
"""

from collections.abc import Iterator
from typing import Any

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
)

from .types import Chunk, LLMRequest, LLMResponse, Message, TokenUsage, ToolCall


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in m.tool_calls
            ]
        out.append(d)
    return out


def _to_tool_calls(msg: ChatCompletionMessage | None) -> list[ToolCall] | None:
    if not msg or not msg.tool_calls:
        return None
    return [
        ToolCall(
            id=tc.id,
            name=tc.function.name or "",
            arguments=tc.function.arguments or "{}",
        )
        for tc in msg.tool_calls
    ]


def _to_response(cc: ChatCompletion, req: LLMRequest) -> LLMResponse:
    choice = cc.choices[0] if cc.choices else None
    msg = choice.message if choice else None
    usage = cc.usage
    raw = None
    dump = getattr(cc, "model_dump", None)
    if callable(dump):
        try:
            dumped = dump()
            raw = dumped if isinstance(dumped, dict) else None
        except TypeError:
            raw = None
    return LLMResponse(
        content=msg.content if msg else None,
        tool_calls=_to_tool_calls(msg),
        usage=TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        ),
        model=cc.model or req.model or "",
        finish_reason=choice.finish_reason if choice else None,
        raw=raw,
    )


class OpenAIProvider:
    """OpenAI 兼容 Provider。构造后可被 LLMClient 聚合调用。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        model: str = "",
        timeout: float = 120,
        max_retries: int = 0,
        **client_kwargs,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = model
        self.timeout = timeout
        # SDK 自身不重试（max_retries=0），重试交给 RetryPolicy 统一管理
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
            **client_kwargs,
        )

    # ---------- chat ----------

    def chat(self, req: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if req.tools:
            kwargs["tools"] = req.tools
        if req.tool_choice is not None:
            kwargs["tool_choice"] = req.tool_choice
        resp = self._client.chat.completions.create(
            model=req.model or self.default_model,
            messages=_to_openai_messages(req.messages),
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            timeout=req.timeout or self.timeout,
            **kwargs,
        )
        return _to_response(resp, req)

    # ---------- stream ----------

    def chat_stream(self, req: LLMRequest) -> Iterator[Chunk]:
        kwargs: dict[str, Any] = {}
        if req.tools:
            kwargs["tools"] = req.tools
        if req.tool_choice is not None:
            kwargs["tool_choice"] = req.tool_choice
        stream = self._client.chat.completions.create(
            model=req.model or self.default_model,
            messages=_to_openai_messages(req.messages),
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            timeout=req.timeout or self.timeout,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            yield self._chunk_from_stream(chunk)

    @staticmethod
    def _chunk_from_stream(chunk: ChatCompletionChunk) -> Chunk:
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        tool_calls: list[ToolCall] | None = None
        if delta and delta.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id or "",
                    name=(tc.function.name if tc.function else "") or "",
                    arguments=(tc.function.arguments if tc.function else "") or "",
                    index=tc.index or 0,
                )
                for tc in delta.tool_calls
            ]
        return Chunk(
            content=(delta.content if delta else "") or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason if choice else None,
        )

    # ---------- embed ----------

    def embed(self, text: str, model: str) -> list[float]:
        resp = self._client.embeddings.create(model=model, input=text)
        return list(resp.data[0].embedding)
