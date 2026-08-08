"""retry / cache / router 单元测试。"""

from llm_client.cache import LLMCache
from llm_client.retry import RetryConfig, RetryExhaustedError, RetryPolicy
from llm_client.router import LLMRouter
from llm_client.types import LLMRequest, LLMResponse, Message


class TestRetryPolicy:
    def test_success_first_try(self):
        policy = RetryPolicy(config=RetryConfig(max_retries=2, jitter=False))
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        assert policy.run(fn) == "ok"
        assert len(calls) == 1

    def test_retry_on_429(self):
        policy = RetryPolicy(
            config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False)
        )
        calls = []

        class _RateLimited(RuntimeError):
            status_code = 429

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise _RateLimited("limited")
            return "ok"

        assert policy.run(fn) == "ok"
        assert len(calls) == 3

    def test_exhausted(self):
        policy = RetryPolicy(
            config=RetryConfig(max_retries=2, base_delay=0.01, jitter=False)
        )

        class _RateLimited(RuntimeError):
            status_code = 429

        def fn():
            raise _RateLimited("limited")

        try:
            policy.run(fn)
        except RetryExhaustedError:
            return
        raise AssertionError("应当抛出 RetryExhaustedError")

    def test_non_retryable_400_immediately(self):
        policy = RetryPolicy(config=RetryConfig(max_retries=3, base_delay=0.01))
        calls = []

        class _BadRequest(RuntimeError):
            status_code = 400

        def fn():
            calls.append(1)
            raise _BadRequest("bad")

        try:
            policy.run(fn)
        except _BadRequest:
            pass
        assert len(calls) == 1


class TestLLMCache:
    def _req(self, **kw):
        kw.setdefault("temperature", 0)
        return LLMRequest(
            messages=[Message(role="user", content="hi")],
            task="chat",
            **kw,
        )

    def test_deterministic_only(self):
        cache = LLMCache(enabled=True)
        req = self._req()
        assert cache.get(req) is None
        cache.set(req, LLMResponse(content="a"))
        assert cache.get(req).content == "a"

        # temperature != 0 不缓存
        req2 = self._req(temperature=0.7)
        assert cache.get(req2) is None
        cache.set(req2, LLMResponse(content="b"))
        assert cache.get(req2) is None

    def test_key_differs_on_messages(self):
        cache = LLMCache(enabled=True)
        cache.set(self._req(), LLMResponse(content="a"))
        other = self._req()
        other.messages = [Message(role="user", content="other")]
        assert cache.get(other) is None


class TestRouter:
    def test_resolve_order(self):
        router = LLMRouter(default_model="default", rules={"intent": "intent-model"})
        assert router.resolve(self._req()) == "default"
        assert router.resolve(self._req(task="intent")) == "intent-model"

    def test_request_model_wins(self):
        router = LLMRouter(default_model="default", rules={"intent": "intent-model"})
        assert router.resolve(self._req(task="intent", model="explicit")) == "explicit"

    def _req(self, **kw):
        return LLMRequest(messages=[Message(role="user", content="hi")], **kw)
