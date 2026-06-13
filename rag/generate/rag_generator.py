"""
RAG 生成器——RAGGenerator

基于 LLM API 的答案生成器。
支持 OpenAI / GitHub Models 兼容格式的聊天补全接口。
"""

import json
import requests
from rag.datatypes import Chunk, RAGResult
from rag.generate.context_builder import ContextBuilder
from .base import BaseGenerator


class RAGGenerator(BaseGenerator):
    """基于 LLM API 的答案生成器"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "gpt-4o-mini",
        context_builder: ContextBuilder | None = None,
        config: dict | None = None,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._context_builder = context_builder or ContextBuilder()
        self._config = config or {}
        self._temperature = self._config.get("temperature", 0.3)
        self._max_tokens = self._config.get("max_tokens", 1024)

    def generate(self, query: str, context: str, **kwargs) -> str:
        if not self._api_key or not self._base_url:
            raise ValueError("需要配置 api_key 和 base_url")

        messages = self._build_messages(query, context)
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }

        url = self._base_url + "/chat/completions"
        resp = requests.post(
            url,
            headers=self._build_headers(),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate_with_sources(
        self, query: str, chunks: list[Chunk], **kwargs
    ) -> RAGResult:
        context = self._context_builder.build(chunks, query)
        answer = self.generate(query, context, **kwargs)
        sources = [
            {
                "id": i + 1,
                "source": c.metadata.get("source", "unknown"),
                "text": c.text[:200],
                "score": 0.0,
            }
            for i, c in enumerate(chunks)
        ]
        return RAGResult(answer=answer, sources=sources)

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_messages(self, query: str, context: str) -> list[dict]:
        system_prompt = self._config.get(
            "system_prompt",
            "You are a helpful assistant. Answer the question based on the provided "
            "reference documents. If the documents do not contain enough information, "
            "say so. Cite sources using [1], [2], etc.",
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
