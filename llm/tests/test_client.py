"""组合 Client 单元测试：mock Provider，验证 路由→缓存→重试 编排。"""

import pytest
from llm_client.cache import LLMCache
from llm_client.client import Client
from llm_client.retry import RetryConfig, RetryExhaustedError, RetryPolicy
from llm_client.router import LLMRouter
from llm_client.types import LLMRequest, LLMResponse, Message


class _FakeProvider:
    def __init__(self):
        self.calls = 0

    def chat(self, req):
        self.calls += 1
        return LLMResponse(content="ok", model=req.model)


class _FlakyProvider:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times

    def chat(self, req):
        self.calls += 1
        if self.calls <= self.fail_times:
            exc = RuntimeError("boom")
            exc.status_code = 429
            raise exc
        return LLMResponse(content="ok", model=req.model)


def _req(task="chat", model=None, temperature=0.7):
    return LLMRequest(
        messages=[Message(role="user", content="hi")],
        task=task,
        model=model,
        temperature=temperature,
    )


class TestClientOrchestration:
    def test_router_selects_model(self):
        provider = _FakeProvider()
        router = LLMRouter(default_model="default", rules={"intent": "intent-model"})
        client = Client(provider=provider, router=router)
        resp = client.chat(_req(task="intent"))
        assert resp.model == "intent-model"
        assert provider.calls == 1

    def test_retry_wraps_provider(self):
        provider = _FlakyProvider(fail_times=2)
        retry = RetryPolicy(
            config=RetryConfig(max_retries=3, base_delay=0.01, jitter=False)
        )
        client = Client(
            provider=provider, router=LLMRouter(default_model="m"), retry=retry
        )
        assert client.chat(_req()).content == "ok"
        assert provider.calls == 3

    def test_cache_skips_provider(self):
        provider = _FakeProvider()
        client = Client(
            provider=provider,
            router=LLMRouter(default_model="m"),
            cache=LLMCache(enabled=True),
        )
        req = _req(temperature=0)
        assert client.chat(req).content == "ok"
        assert client.chat(req).content == "ok"
        assert provider.calls == 1

    def test_exhausted_raises(self):
        provider = _FlakyProvider(fail_times=99)
        retry = RetryPolicy(
            config=RetryConfig(max_retries=1, base_delay=0.01, jitter=False)
        )
        client = Client(
            provider=provider, router=LLMRouter(default_model="m"), retry=retry
        )
        with pytest.raises(RetryExhaustedError):
            client.chat(_req())
