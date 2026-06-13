"""
查询重写器——QueryRewriter

支持多种查询重写策略:
  1. identity: 直通，不重写
  2. hyde: 生成假设文档（HyDE），用生成的文档替代查询
  3. expansion: 同义词扩展查询
"""

import re
from typing import Callable


class QueryRewriter:
    """查询重写器，支持 HyDE 和查询扩展"""

    def __init__(
        self,
        strategy: str = "identity",
        llm_call: Callable[[str], str] | None = None,
        config: dict | None = None,
    ):
        self.strategy = strategy
        self.llm_call = llm_call
        self._config = config or {}

    def rewrite(self, query: str) -> str:
        if self.strategy == "identity":
            return query
        elif self.strategy == "hyde":
            return self._hyde(query)
        elif self.strategy == "expansion":
            return self._expand(query)
        else:
            return query

    def rewrite_multi(self, query: str) -> list[str]:
        if self.strategy == "expansion":
            return self._expand_multi(query)
        return [self.rewrite(query)]

    def _hyde(self, query: str) -> str:
        if self.llm_call is None:
            return query
        prompt = self._config.get(
            "hyde_prompt",
            "Please write a passage that answers the following question:\n{query}",
        )
        return self.llm_call(prompt.replace("{query}", query))

    def _expand(self, query: str) -> str:
        if self.llm_call is None:
            return query
        prompt = self._config.get(
            "expand_prompt",
            "Expand the following search query with synonyms and related terms. "
            "Return only the expanded query:\n{query}",
        )
        return self.llm_call(prompt.replace("{query}", query))

    def _expand_multi(self, query: str) -> list[str]:
        expanded = self._expand(query)
        # Split on common delimiters, filter empty
        parts = re.split(r"[;，、]", expanded)
        return [q.strip() for q in [query] + parts if q.strip()]
