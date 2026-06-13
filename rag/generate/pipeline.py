"""
生成管线——GenerationPipeline

编排上下文构建 → LLM 生成 → 引用追踪的完整流程。
"""

import json
import time
import uuid

import requests
from rag.datatypes import Chunk, RAGResult
from .base import BaseGenerator
from .context_builder import ContextBuilder
from .citation import CitationTracker
from .rag_generator import RAGGenerator


class GenerationPipeline:
    """生成管线：编排上下文构建 → LLM 生成 → 引用追踪"""

    def __init__(
        self,
        generator: BaseGenerator | None = None,
        context_builder: ContextBuilder | None = None,
        citation_tracker: CitationTracker | None = None,
    ):
        self.generator = generator
        self.context_builder = context_builder or ContextBuilder()
        self.citation_tracker = citation_tracker or CitationTracker()

    def generate(
        self,
        query: str,
        chunks: list[Chunk],
        **kwargs,
    ) -> RAGResult:
        t_start = time.time()
        request_id = kwargs.pop("request_id", uuid.uuid4().hex[:12])

        if not self.generator:
            return RAGResult(
                answer="",
                sources=[],
                confidence=0.0,
                request_id=request_id,
                latency_ms=(time.time() - t_start) * 1000,
            )

        context = self.context_builder.build(chunks, query)
        answer = self.generator.generate(query, context, **kwargs)
        latency_ms = (time.time() - t_start) * 1000

        return self.citation_tracker.build_result(
            answer=answer,
            chunks=chunks,
            confidence=kwargs.get("confidence", 0.8),
            request_id=request_id,
            latency_ms=latency_ms,
        )

    def generate_stream(
        self,
        query: str,
        chunks: list[Chunk],
        **kwargs,
    ):
        if not isinstance(self.generator, RAGGenerator):
            yield RAGResult(answer="Streaming requires RAGGenerator")
            return

        context = self.context_builder.build(chunks, query)
        messages = self.generator._build_messages(query, context)
        payload = {
            "model": self.generator._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.generator._temperature),
            "max_tokens": kwargs.get("max_tokens", self.generator._max_tokens),
            "stream": True,
        }

        url = self.generator._base_url + "/chat/completions"
        resp = requests.post(
            url,
            headers=self.generator._build_headers(),
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        full_text = ""
        for line in resp.iter_lines(decode_unicode=True):
            if line:
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            full_text += delta
                            yield self.citation_tracker.build_result(
                                answer=full_text,
                                chunks=chunks,
                                confidence=0.8,
                            )
                    except json.JSONDecodeError:
                        continue
