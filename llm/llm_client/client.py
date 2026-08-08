"""组合客户端：把 Provider + 重试 + 缓存 + 路由 + 审计组合成一个 LLMClient。

其他模块只需注入一个 Client 实例：
    client = build_client(LLMConfig.load(...))
    resp = client.chat(LLMRequest(messages=[...], task="intent"))
"""

from collections.abc import Iterator

from .audit import Audit
from .base import LLMClientBase
from .cache import LLMCache
from .http import HTTPProvider
from .openai_compat import OpenAIProvider
from .retry import RetryConfig, RetryPolicy
from .router import LLMRouter
from .types import Chunk, LLMRequest, LLMResponse


class Client(LLMClientBase):
    """统一客户端：编排 Provider 与横切能力。

    组合顺序：router(选模型) → cache(命中即返) → retry → audit → provider。
    stream 与 embed 走 provider 直出（stream 不缓存）。
    """

    def __init__(
        self,
        provider: OpenAIProvider | HTTPProvider,
        router: LLMRouter | None = None,
        retry: RetryPolicy | None = None,
        cache: LLMCache | None = None,
        audit: Audit | None = None,
    ):
        self.provider = provider
        self.router = router or LLMRouter(
            default_model=getattr(provider, "default_model", "")
        )
        self.retry = retry or RetryPolicy()
        self.cache = cache or LLMCache(enabled=False)
        self.audit = audit or Audit(enable=False)

    # ---------- chat ----------

    def chat(self, req: LLMRequest) -> LLMResponse:
        resolved = req.model_copy(update={"model": self.router.resolve(req)})

        cached = self.cache.get(resolved)
        if cached is not None:
            return cached

        def _do():
            return self.provider.chat(resolved)

        resp = self.retry.run(_do)
        resp.model = resp.model or resolved.model
        self.cache.set(resolved, resp)
        return self.audit.chat(lambda: resp, resolved)

    # ---------- stream ----------

    def chat_stream(self, req: LLMRequest) -> Iterator[Chunk]:
        resolved = req.model_copy(update={"model": self.router.resolve(req)})
        return self.provider.chat_stream(resolved)

    # ---------- embed ----------

    def embed(self, text: str, model: str = "") -> list[float]:
        if hasattr(self.provider, "embed"):
            return self.provider.embed(text, model)
        raise NotImplementedError("当前 Provider 不支持 embed")


def build_client(
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    provider: str = "openai_compat",
    timeout: float = 120,
    max_retries: int = 3,
    enable_cache: bool = False,
    default_task_models: dict | None = None,
) -> Client:
    """便捷构造：从配置字段构建统一客户端。

    provider 可选 openai_compat / http（http 用于 base_url 指向 .../chat/completions）。
    default_task_models 形如 {"chat": "m1", "intent": "m2"}，走任务路由。
    """
    if provider == "http":
        p: OpenAIProvider | HTTPProvider = HTTPProvider(
            api_key=api_key, base_url=base_url, model=model, timeout=timeout
        )
    else:
        p = OpenAIProvider(
            api_key=api_key, base_url=base_url, model=model, timeout=timeout
        )
    router = LLMRouter(default_model=model, rules=default_task_models or {})
    return Client(
        provider=p,
        router=router,
        retry=RetryPolicy(config=RetryConfig(max_retries=max_retries)),
        cache=LLMCache(enabled=enable_cache),
        audit=Audit(enable=True),
    )
