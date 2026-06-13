"""
RAG 系统总入口——RAGSystem

整合索引、检索、生成三管线，提供统一的 query/index 接口。
"""

import uuid

from rag.datatypes import RAGResult
from rag.embed import BaseEmbedder, ApiEmbedder

from rag.loader import LoaderFactory
from rag.chunker import BaseChunker, RecursiveChunker
from rag.store import BaseVectorStore, MemoryStore, BM25Index
from rag.ingest import DocumentIndexer
from rag.retrieve import (
    QueryRewriter,
    HybridRetriever,
    Reranker,
    RetrievalPipeline,
)
from rag.generate import (
    BaseGenerator,
    ContextBuilder,
    RAGGenerator,
    CitationTracker,
    GenerationPipeline,
)


class RAGSystem:
    """RAG 系统总入口，整合索引、检索、生成三管线"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        embed_base_url: str = "",
        embed_model: str = "openai/text-embedding-3-small",
        llm_model: str = "openai/gpt-4o-mini",
        embedder: BaseEmbedder | None = None,
        generator: BaseGenerator | None = None,
        vector_store: BaseVectorStore | None = None,
        chunker: BaseChunker | None = None,
        config: dict | None = None,
    ):
        self._config = config or {}
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._embed_base_url = embed_base_url.rstrip("/") if embed_base_url else ""
        self._embed_model = embed_model
        self._llm_model = llm_model

        self.embedder = embedder or self._init_embedder()
        self.generator = generator or self._init_generator()

        self.vector_store = vector_store or MemoryStore()
        self.bm25_index = BM25Index()

        self.chunker = chunker or RecursiveChunker(
            chunk_size=self._config.get("chunk_size", 500),
            chunk_overlap=self._config.get("chunk_overlap", 50),
        )

        self.indexer = self._init_indexer()
        self.retrieval_pipeline = self._init_retrieval()
        self.generation_pipeline = self._init_generation()

    def _init_embedder(self) -> BaseEmbedder:
        if not self._api_key or not self._base_url:
            from rag.embed.local import LocalEmbedder

            return LocalEmbedder()
        cfg = self._config.get("embedding", {})
        embed_url = self._embed_base_url or self._base_url
        if "chat/completions" in embed_url:
            embed_url = embed_url.replace("chat/completions", "embeddings")
        embed_url = embed_url.rstrip("/")
        if embed_url.endswith("/embeddings"):
            embed_url = embed_url[: -len("/embeddings")]
        return ApiEmbedder(
            api_key=self._api_key,
            base_url=embed_url,
            model=self._embed_model,
            dim=cfg.get("dim", 1536),
        )

    def _init_generator(self) -> BaseGenerator:
        if not self._api_key or not self._base_url:
            from rag.generate.base import BaseGenerator

            class _EmptyGenerator(BaseGenerator):
                def generate(self, query, context, **kwargs):
                    return ""

                def generate_with_sources(self, query, chunks, **kwargs):
                    return RAGResult()

            return _EmptyGenerator()
        gen_cfg = self._config.get("generation", {})
        llm_url = self._base_url.rstrip("/")
        if llm_url.endswith("/chat/completions"):
            llm_url = llm_url[: -len("/chat/completions")]
        return RAGGenerator(
            api_key=self._api_key,
            base_url=llm_url,
            model=self._llm_model,
            context_builder=ContextBuilder(
                config={"max_tokens": gen_cfg.get("max_tokens", 4096)}
            ),
            config=gen_cfg,
        )

    def _init_indexer(self) -> DocumentIndexer:
        return DocumentIndexer(
            loader_factory=LoaderFactory(),
            chunker=self.chunker,
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
        )

    def _init_retrieval(self) -> RetrievalPipeline:
        ret_cfg = self._config.get("retrieval", {})
        hybrid = HybridRetriever(
            vector_store=self.vector_store,
            bm25_index=self.bm25_index,
            config=ret_cfg,
        )

        def _llm_for_rewrite(prompt: str) -> str:
            try:
                return self.generator.generate(prompt, "")
            except Exception:
                return prompt

        rewriter = QueryRewriter(
            strategy=ret_cfg.get("rewrite_strategy", "identity"),
            llm_call=_llm_for_rewrite if hasattr(self.generator, "generate") else None,
        )
        reranker = Reranker(
            strategy=ret_cfg.get("rerank_strategy", "identity"),
            config=ret_cfg,
        )
        return RetrievalPipeline(
            embedder=self.embedder,
            hybrid_retriever=hybrid,
            query_rewriter=rewriter,
            reranker=reranker,
        )

    def _init_generation(self) -> GenerationPipeline:
        return GenerationPipeline(
            generator=self.generator,
            context_builder=ContextBuilder(config=self._config.get("generation", {})),
            citation_tracker=CitationTracker(),
        )

    def query(self, query_text: str, top_k: int = 5) -> RAGResult:
        results = self.retrieval_pipeline.retrieve(
            query_text,
            collection=self._config.get("collection", "default"),
            top_k=top_k,
        )
        if not results:
            return RAGResult(
                answer="知识库中未找到相关信息。",
                confidence=0.0,
                request_id=uuid.uuid4().hex[:12],
            )
        return self.generation_pipeline.generate(query_text, results, top_k=top_k)

    def index_document(self, file_path: str, collection: str | None = None) -> int:
        col = collection or self._config.get("collection", "default")
        return self.indexer.index_document(file_path, col)

    def batch_index(
        self, directory: str, collection: str | None = None
    ) -> dict[str, int]:
        col = collection or self._config.get("collection", "default")
        return self.indexer.batch_index(directory, col)

    def clear_index(self, collection: str | None = None):
        col = collection or self._config.get("collection", "default")
        self.vector_store.delete_collection(col)
