"""llm-client 包：统一 LLM 调用接口与横切能力。

设计目标：
- 统一多 Provider（OpenAI 兼容 / 裸 HTTP）接入；
- chat / chat_stream / embed 三种能力，预留其他模块（agent / rag / memory）接入点；
- 重试、缓存、路由、审计以可插拔职责链组合，不侵入业务调用方。

与其他包的关系：
- agent_mvp / rag / memory 通过依赖注入拿到 LLMClient（Protocol），不强制 import 本包；
- intent 包保持零 SDK 依赖，宿主侧把 4 个注入回调转发到本包即可。
"""

from .base import LLMClient, LLMClientBase
from .cache import LLMCache
from .client import Client, build_client
from .config import LLMConfig, load_config
from .embeddings import EmbeddingsClient
from .http import HTTPProvider
from .openai_compat import OpenAIProvider
from .retry import RetryConfig, RetryPolicy
from .router import LLMRouter, RouteRule
from .types import (
    Chunk,
    LLMRequest,
    LLMResponse,
    Message,
    TokenUsage,
    ToolCall,
)

__all__ = [
    "Chunk",
    "Client",
    "EmbeddingsClient",
    "HTTPProvider",
    "LLMCache",
    "LLMClient",
    "LLMClientBase",
    "LLMConfig",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "Message",
    "OpenAIProvider",
    "RetryConfig",
    "RetryPolicy",
    "RouteRule",
    "TokenUsage",
    "ToolCall",
    "build_client",
    "load_config",
]
