"""
嵌入层——本地嵌入器

使用 sentence-transformers 加载本地模型（如 BAAI/bge-m3）。
采用惰性加载，未安装依赖时抛出清晰提示，不阻塞其他功能。
"""

from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """本地嵌入器，支持 BGE-M3 等 sentence-transformers 模型"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        dim: int = 1024,
        device: str = "cpu",
        quantize: bool = True,
    ):
        self._model_name = model_name
        self._dim = dim
        self._device = device
        self._quantize = quantize
        self._model = None  # 惰性加载，首次 embed 时初始化

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"local/{self._model_name}"

    def _lazy_load(self):
        """首次使用时加载模型"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "本地嵌入需要安装 sentence-transformers: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(self._model_name, device=self._device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._lazy_load()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
