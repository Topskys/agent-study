"""确定性响应缓存。

规则：仅缓存可确定复用的调用——
- temperature == 0（或 extra 显式带 no_cache 除外）；
- 同一请求（messages + model + temperature + tools）命中即复用；
- 默认进程内 LRU 缓存，可注入任意 get/set 后端（如 Redis）。
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field

from .types import LLMRequest, LLMResponse

_DETERMINISTIC_TEMPERATURE = 0.0


def _cache_key(req: LLMRequest) -> str:
    payload = {
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        "model": req.model,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "tools": req.tools,
        "tool_choice": req.tool_choice,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class LLMCache:
    """确定性响应缓存。key = hash(messages + model + 采样参数 + tools)。"""

    enabled: bool = True
    max_size: int = 1024
    backend: Callable[[str], dict | None] | None = None
    backend_set: Callable[[str, dict], None] | None = None
    _mem: dict[str, dict] = field(default_factory=dict)

    def get(self, req: LLMRequest) -> LLMResponse | None:
        if not self._can_cache(req):
            return None
        key = _cache_key(req)
        data = self.backend(key) if self.backend else self._mem.get(key)
        if data is None and self._mem and not self.backend:
            data = self._mem.get(key)
        if data is None:
            return None
        return LLMResponse.model_validate(data)

    def set(self, req: LLMRequest, resp: LLMResponse) -> None:
        if not self._can_cache(req) or resp is None:
            return
        key = _cache_key(req)
        data = resp.model_dump(exclude_none=True)
        if self.backend_set:
            self.backend_set(key, data)
        else:
            if len(self._mem) >= self.max_size:
                # 简单淘汰：清空重建（进程内场景够用）
                self._mem.clear()
            self._mem[key] = data

    def _can_cache(self, req: LLMRequest) -> bool:
        if not self.enabled:
            return False
        if req.stream:
            return False
        if req.temperature != _DETERMINISTIC_TEMPERATURE:
            return False
        return not req.extra.get("no_cache")
