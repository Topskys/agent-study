"""
索引编排器——DocumentIndexer

编排 加载→分块→嵌入→存储 的完整索引流程。
这是索引管线的总入口，DocumentIndexer.index_document() 完成单文件的全部索引工作。
"""

from pathlib import Path

from rag.datatypes import Chunk
from rag.loader import LoaderFactory
from rag.chunker import BaseChunker
from rag.embed import BaseEmbedder
from rag.store import BaseVectorStore, BM25Index


class DocumentIndexer:
    """索引编排器：编排加载→分块→嵌入→存储的完整流程"""

    def __init__(
        self,
        loader_factory: LoaderFactory,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        bm25_index: BM25Index | None = None,
    ):
        self.loader_factory = loader_factory  # 加载器工厂
        self.chunker = chunker  # 分块器
        self.embedder = embedder  # 嵌入器
        self.vector_store = vector_store  # 向量存储
        self.bm25_index = bm25_index  # BM25 索引（可选）

    def index_document(self, file_path: str, collection: str = "default") -> int:
        """
        索引单个文档，返回 Chunk 数量。

        流程:
          1. loader_factory → 获取对应加载器
          2. loader.load(file_path) → list[RawDocument]
          3. chunker.chunk(doc) → list[Chunk]
          4. embedder.embed(texts) → 批量向量化
          5. vector_store.insert(collection, chunks)
          6. bm25_index.add_documents(collection, chunks) [可选]
          7. 返回 Chunk 总数
        """
        loader = self.loader_factory.get_loader(file_path)
        raw_docs = loader.load(file_path)

        # 分块
        all_chunks: list[Chunk] = []
        for doc in raw_docs:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        # 向量化
        texts = [c.text for c in all_chunks]
        embeddings = self.embedder.embed(texts)
        for chunk, emb in zip(all_chunks, embeddings):
            chunk.embeddings[self.embedder.name] = emb

        # 存入向量存储
        self.vector_store.create_collection(collection, self.embedder.dim)
        self.vector_store.insert(collection, all_chunks)

        # 可选：存入 BM25 索引
        if self.bm25_index is not None:
            self.bm25_index.add_documents(collection, all_chunks)

        return len(all_chunks)

    def batch_index(
        self, directory: str, collection: str = "default"
    ) -> dict[str, int]:
        """批量索引目录下所有支持的文件"""
        results = {}
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file():
                try:
                    count = self.index_document(str(file_path), collection)
                    results[str(file_path)] = count
                except Exception:
                    results[str(file_path)] = 0
        return results
