"""重试策略：指数退避 + 抖动。

吸收 agent_mvp/reason() 里的重试逻辑做通用化：
- 仅对可重试状态码（429 / 5xx）与网络错误重试；
- 4xx 业务错误、超时配置缺失等直接抛错；
- 退避时间 = base * 2**attempt + jitter，attempt 从 0 开始。
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")

# 可重试 HTTP 状态码：限流 + 服务端错误
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# 可重试的底层异常：网络/超时/连接类
_RETRYABLE_EXC = (ConnectionError, TimeoutError, OSError)


class RetryExhaustedError(RuntimeError):
    """重试次数用尽后仍失败。"""


class NonRetryableError(RuntimeError):
    """不可重试错误（4xx 业务错误等），直接上抛。"""


@dataclass
class RetryConfig:
    """重试配置。"""

    max_retries: int = 3
    base_delay: float = 1.0  # 首次退避基数（秒）
    max_delay: float = 30.0  # 退避上限（秒）
    jitter: bool = True  # 是否加随机抖动，避免惊群
    retryable_statuses: set[int] = field(default_factory=lambda: set(_RETRYABLE_STATUS))
    retryable_exceptions: tuple = field(default_factory=lambda: _RETRYABLE_EXC)


@dataclass
class RetryPolicy:
    """重试执行器：包装任意可重试调用。"""

    config: RetryConfig = field(default_factory=RetryConfig)

    def _backoff(self, attempt: int) -> float:
        delay = min(self.config.base_delay * (2**attempt), self.config.max_delay)
        if self.config.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay

    def run(
        self,
        fn: Callable[[], T],
        on_retry: Callable[[int, float], None] | None = None,
    ) -> T:
        """执行 fn，按配置重试。on_retry(attempt, delay) 用于日志回调。"""
        for attempt in range(self.config.max_retries + 1):
            try:
                return fn()
            except NonRetryableError:
                raise
            except Exception as e:
                if self._should_retry(e, attempt):
                    delay = self._backoff(attempt)
                    if on_retry:
                        on_retry(attempt + 1, delay)
                    time.sleep(delay)
                    continue
                # 不可重试（如 4xx）直接上抛；可重试但次数用尽抛 RetryExhaustedError
                if attempt < self.config.max_retries:
                    raise
                raise RetryExhaustedError(f"重试次数用尽: {e}") from e
        raise RetryExhaustedError("重试次数用尽")

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        if attempt >= self.config.max_retries:
            return False
        status = getattr(exc, "status_code", None)
        if status is not None and int(status) in self.config.retryable_statuses:
            return True
        if isinstance(exc, self.config.retryable_exceptions):
            return True
        # openai SDK 的 APIError 等可能没有 status_code，但带 status
        status = getattr(exc, "status", None)
        return status is not None and int(status) in self.config.retryable_statuses
