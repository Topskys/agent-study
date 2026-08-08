"""Embedding 统一入口：供 memory / rag 复用。

支持两种后端：
- api：OpenAI 兼容 embedding 端点（走 openai SDK）；
- 自定义可调用对象（如本地 BGE 模型函数），便于不改造 memory 现有逻辑接入。
"""

from collections.abc import Callable

from openai import OpenAI


class EmbeddingsClient:
    """统一 embedding 调用。backend_fn 非空时直接转发（本地模型场景）。"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        timeout: float = 120,
        backend_fn: Callable[[str, str], list[float]] | None = None,
    ):
        self._backend_fn = backend_fn
        if backend_fn is None:
            self._client = OpenAI(
                api_key=api_key, base_url=base_url or None, timeout=timeout
            )
        else:
            self._client = None

    def embed(self, text: str, model: str = "") -> list[float]:
        if self._backend_fn is not None:
            return self._backend_fn(text, model)
        if not self._client:
            raise ValueError("未配置 embedding 后端")
        resp = self._client.embeddings.create(model=model, input=text)
        return list(resp.data[0].embedding)

    def embed_batch(self, texts: list[str], model: str = "") -> list[list[float]]:
        if self._backend_fn is not None:
            return [self._backend_fn(t, model) for t in texts]
        if not self._client:
            raise ValueError("未配置 embedding 后端")
        resp = self._client.embeddings.create(model=model, input=texts)
        return [list(d.embedding) for d in resp.data]
