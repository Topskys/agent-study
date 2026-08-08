"""统一 LLM 接口定义（Protocol）与组合基类。

设计原则：
- LLMClient 用 Protocol 定义，其他包（agent_mvp / rag / memory）按类型注解
  依赖、不强 import 本包，保持松散耦合；
- LLMClientBase 是可继承的默认实现，横切能力（重试/缓存/路由/审计）以
  可选组合子形式注入，调用方无感。
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from .types import Chunk, LLMRequest, LLMResponse


@runtime_checkable
class LLMClient(Protocol):
    """统一 LLM 调用接口。其他模块注入本类型即可接入。"""

    def chat(self, req: LLMRequest) -> LLMResponse: ...

    def chat_stream(self, req: LLMRequest) -> Iterator[Chunk]: ...

    def embed(self, text: str, model: str) -> list[float]: ...


class LLMClientBase:
    """可继承的统一客户端：聚合 Provider 与横切能力。

    子类需实现 _do_chat / _do_stream / _do_embed（或直接覆写对外三方法）。
    这里提供对外三方法的默认编排：路由 → 缓存 → 重试 → 审计 → Provider。
    """

    def chat(self, req: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def chat_stream(self, req: LLMRequest) -> Iterator[Chunk]:
        raise NotImplementedError

    def embed(self, text: str, model: str) -> list[float]:
        raise NotImplementedError
