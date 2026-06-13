"""
嵌入层——混合路由嵌入器

支持主备双路路由 + 自动 fallback：
- 主嵌入器优先（如本地模型，大批量低成本）
- 主嵌入器失败时自动降级到备选（如云端 API）
- 支持 batch 分片处理大批量文本
"""

from .base import BaseEmbedder


class EmbedRouter(BaseEmbedder):
    """主备路由嵌入器，支持自动 fallback 和 batch 分片"""

    def __init__(
        self,
        primary: BaseEmbedder,
        fallback: BaseEmbedder | None = None,
        config: dict | None = None,
    ):
        self.primary = primary  # 主嵌入器
        self._fallback = fallback  # 备选嵌入器（降级用）
        self._config = config or {}
        self._batch_size = self._config.get("batch_size", 16)

    @property
    def dim(self) -> int:
        return self.primary.dim

    @property
    def name(self) -> str:
        return self.primary.name

    def set_fallback(self, fallback: BaseEmbedder):
        """设置备选嵌入器"""
        self._fallback = fallback

    def embed(self, texts: list[str], mode: str = "auto") -> list[list[float]]:
        """
        向量化入口，支持自动 fallback。

        参数:
            texts: 待向量化文本列表
            mode: 路由模式（auto/indexing/query_fast/query_deep），备用字段

        返回:
            list[list[float]]: 向量列表
        """
        # 按 batch 分片处理
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]
        all_embeddings = []

        for batch in batches:
            try:
                all_embeddings.extend(self.primary.embed(batch))
            except Exception as e:
                # 主嵌入器失败时自动降级
                if self._fallback is not None:
                    all_embeddings.extend(self._fallback.embed(batch))
                else:
                    raise e

        return all_embeddings
