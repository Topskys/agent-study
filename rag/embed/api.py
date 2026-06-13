"""
嵌入层——云端 API 嵌入器

支持 OpenAI 兼容格式的嵌入 API，包括 GitHub Models。
"""

import requests
from .base import BaseEmbedder


class ApiEmbedder(BaseEmbedder):
    """云端 API 嵌入器"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        provider: str = "openai",
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dim = dim
        self._provider = provider

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"{self._provider}/{self._model}"

    def _is_github_models(self) -> bool:
        """检测是否为 GitHub Models 接口（需要额外请求头）"""
        return "models.github.ai" in self._base_url

    def _build_headers(self) -> dict:
        """构建请求头，GitHub Models 需要额外 Accept 和版本头"""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._is_github_models():
            headers["Accept"] = "application/vnd.github+json"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        return headers

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文本"""
        if not self._api_key or not self._base_url:
            raise ValueError("需要配置 api_key 和 base_url")

        url = self._base_url + "/embeddings"
        payload = {"model": self._model, "input": texts}
        resp = requests.post(
            url, headers=self._build_headers(), json=payload, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]
