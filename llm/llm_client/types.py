"""核心数据模型：请求 / 响应 / 消息 / 工具调用。

统一约定：所有 Provider 的原始返回都规整为 LLMResponse，
业务方只依赖本文件定义的类型，不依赖具体 SDK 类型。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# role：system / user / assistant / tool
MessageRole = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """一次工具调用：模型决定调用哪个工具、传什么参数。"""

    id: str = ""
    name: str
    arguments: str = "{}"  # 原始 JSON 字符串（保留原样，解析由工具层负责）


class Message(BaseModel):
    """对话消息，兼容 OpenAI 原生 message 结构。"""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class TokenUsage(BaseModel):
    """token 用量统计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMRequest(BaseModel):
    """统一请求：Agent / RAG / 意图识别等所有调用方共用。"""

    messages: list[Message]
    model: str | None = None  # None 时由 router 按 task 选择
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    tools: list[dict[str, Any]] | None = None  # function calling schemas
    tool_choice: Any | None = None
    task: str = "chat"  # 路由标签：chat / intent / plan / tool / rag ...
    stream: bool = False
    timeout: float | None = None  # 覆盖默认超时（秒）
    extra: dict[str, Any] = Field(default_factory=dict)  # Provider 透传参数

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        system: str | None = None,
        task: str = "chat",
        **kwargs,
    ) -> "LLMRequest":
        """便捷构造：单 prompt / 可选 system。"""
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return cls(messages=messages, task=task, **kwargs)


class LLMResponse(BaseModel):
    """统一响应：所有 Provider 规整为这个结构。"""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    model: str = ""
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None  # Provider 原始返回，供调试/审计


class Chunk(BaseModel):
    """流式响应分片（chat_stream 的产出物）。"""

    content: str = ""
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


ProviderName = Literal["openai_compat", "http"]
