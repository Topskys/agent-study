"""
极简 RAG 系统 - 单文件实现，跑通基础流程

功能:
  - 索引 .txt / .md 文件
  - 递归分块（中英分隔符感知）
  - API 向量化（兼容 OpenAI 格式）
  - 内存向量存储 + numpy 余弦相似度检索
  - LLM 生成带上下文的回答

使用示例:
    from rag import BasicRAG

    rag = BasicRAG(api_key="sk-xxx", base_url="https://api.openai.com/v1/chat/completions")

    # 索引
    rag.index_file("README.md")
    rag.index_text("自定义文本内容")

    # 检索
    results = rag.search("查询关键词", top_k=3)

    # 检索 + 生成
    result = rag.query("问题")
    print(result["answer"])

更多示例:
    python rag/examples/01_basic_rag.py
    python rag/examples/02_batch_index.py <目录>
"""

import numpy as np
import requests
import uuid
import re
from pathlib import Path
from typing import List, Dict, Optional


class BasicRAG:
    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        embed_base_url: str = "",
        embed_model: str = "text-embedding-ada-002",
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        llm_temperature: float = 0.3,
        llm_max_tokens: int = 2048,
    ):
        self.api_key = api_key
        self.llm_base_url = base_url
        self.embed_base_url = embed_base_url or self._infer_embed_url(base_url)
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens

        self.chunks: List[Dict] = []

    # ========== Embedding ==========

    def _is_github_models(self, url: str) -> bool:
        return "models.github.ai" in url

    def _build_headers(self, url: str = "") -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self._is_github_models(url):
            headers["Accept"] = "application/vnd.github+json"
            headers["X-GitHub-Api-Version"] = "2022-11-28"
        return headers

    def _infer_embed_url(self, llm_url: str) -> str:
        if not llm_url:
            return ""
        url = llm_url.rstrip("/")
        if url.endswith("/chat/completions"):
            return url.replace("/chat/completions", "/embeddings")
        return url.rstrip("/") + "/embeddings"

    def _embed(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.api_key or not self.embed_base_url:
            raise ValueError("需要配置 api_key 和 base_url")

        payload = {"model": self.embed_model, "input": texts}
        resp = requests.post(
            self.embed_base_url,
            headers=self._build_headers(self.embed_base_url),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)

    # ========== Chunking ==========

    def _chunk(self, text: str) -> List[str]:
        if not text:
            return []

        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""
        sep_pattern = re.compile(r"[。！？.!?\n]")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 1 <= self.chunk_size:
                current = (current + "\n" + para).strip()
                continue

            if current:
                chunks.append(current)

            if len(para) > self.chunk_size:
                sentences = sep_pattern.split(para)
                current = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) + 1 <= self.chunk_size:
                        current = (current + "。" + sent).strip()
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para

        if current:
            chunks.append(current)

        return chunks

    # ========== Index ==========

    def index_file(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        text = path.read_text(encoding="utf-8")
        return self.index_text(text, {"source": str(path)})

    def index_text(self, text: str, metadata: Optional[Dict] = None) -> int:
        metadata = metadata or {}
        raw_chunks = self._chunk(text)
        if not raw_chunks:
            return 0

        embeddings = self._embed(raw_chunks)

        for i, (chunk_text, emb) in enumerate(zip(raw_chunks, embeddings)):
            self.chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk_text,
                    "metadata": {**metadata, "chunk_index": i},
                    "embedding": emb,
                }
            )

        return len(raw_chunks)

    def clear(self):
        self.chunks.clear()

    # ========== Search ==========

    def search(self, query_text: str, top_k: int = 5) -> List[Dict]:
        if not self.chunks:
            return []

        query_vec = self._embed([query_text])[0]

        chunk_vecs = np.array([c["embedding"] for c in self.chunks])
        norms = np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec)
        norms[norms == 0] = 1e-10
        scores = np.dot(chunk_vecs, query_vec) / norms

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append(
                {
                    "text": self.chunks[idx]["text"],
                    "score": float(scores[idx]),
                    "metadata": self.chunks[idx]["metadata"],
                }
            )
        return results

    # ========== Generation ==========

    def _llm_call(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
        }
        resp = requests.post(
            self.llm_base_url,
            headers=self._build_headers(self.llm_base_url),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def query(self, query_text: str, top_k: int = 5) -> Dict:
        # 1. 检索
        results = self.search(query_text, top_k)

        if not results:
            return {
                "answer": "知识库中暂无相关文档，请先添加文档后重试。",
                "sources": [],
                "confidence": 0.0,
            }

        # 2. 组装上下文
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "未知来源")
            context_parts.append(f"[{i}] 来源: {source}\n{r['text']}")
        context = "\n\n".join(context_parts)

        # 3. 生成
        system_prompt = f"""你是一个基于本地知识库的问答助手。请根据以下参考资料回答问题。
如果参考资料不足以回答问题，请如实说明。

参考资料：
{context}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ]

        answer = self._llm_call(messages)

        # 4. 简单置信度评估
        confidence = min(1.0, max(0.0, results[0]["score"])) if results else 0.0

        sources = [
            {
                "text": r["text"][:200],
                "score": r["score"],
                "source": r["metadata"].get("source", ""),
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }


"""
============================================================
后续优化方向
============================================================

1. 嵌入策略
   - 本地嵌入模型（BGE-M3 / sentence-transformers），减少 API 依赖
   - 混合路由：本地为主，API 降级

2. 分块策略
   - 语义分块（按 embedding 余弦距离切分）
   - 父子分块（粗检索 + 细生成）

3. 检索优化
   - BM25 关键词检索 + RRF 融合（提高命中率）
   - Multi-Query / HyDE 查询增强
   - Cross-encoder 重排序
   - Self-RAG / CRAG 质量判断与修正

4. 向量存储
   - Chroma / Milvus 持久化存储
   - 支持增量更新与删除

5. 文档支持
   - PDF 加载（PyMuPDF 文字层 + pdfplumber 表格 + PaddleOCR）
   - DOCX 加载（python-docx）
   - HTML/Vue 加载（html.parser 去标签）
   - 视频加载（ffmpeg 抽帧 + Whisper 转录）

6. 生成优化
   - 引用溯源（答案标记来源编号）
   - Token 预算管理（长上下文时自动裁剪）
   - 多轮对话中的上下文复用

7. 生产化
   - 全链路日志 PipelineLogger
   - 异步索引与检索
   - 数据飞轮（用户反馈→自动重训）
   - 配置中心 YAML
"""
