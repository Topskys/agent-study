"""
全链路日志——PipelineLogger

记录每次 RAG 查询的完整 Trace，支持写入 JSON 文件。
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from rag.datatypes import LogEntry, RAGResult, SearchResult


class PipelineLogger:
    """记录每次 RAG 查询的完整 Trace"""

    def __init__(self, log_dir: str = "./logs/rag"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, LogEntry] = {}

    def new_request(self, query: str) -> str:
        request_id = f"rag_{uuid4().hex[:12]}"
        self._entries[request_id] = LogEntry(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            user_query=query,
            pipeline={},
            retrieved_chunks=[],
            generation={},
            confidence=0.0,
            latency_ms=0.0,
        )
        return request_id

    def log_rewrite(self, original: str, rewritten: str) -> None:
        for e in self._entries.values():
            if e.user_query == original:
                e.pipeline["rewrite"] = {"original": original, "rewritten": rewritten}
                break

    def log_skip(self, query: str, reason: str) -> None:
        for e in self._entries.values():
            if e.user_query == query:
                e.pipeline["skip"] = {"reason": reason}
                break

    def log_retrieval(self, query: str, results: list[SearchResult]) -> None:
        for e in self._entries.values():
            if e.user_query == query:
                e.retrieved_chunks = [
                    {
                        "chunk_id": r.chunk.chunk_id,
                        "source": r.chunk.metadata.get("source", ""),
                        "score": r.score,
                        "rank": r.rank,
                    }
                    for r in results
                ]
                e.pipeline["retrieval"] = {"count": len(results)}
                break

    def log_generation(self, query: str, result: RAGResult) -> None:
        for e in self._entries.values():
            if e.user_query == query:
                e.generation = {
                    "model": result.answer[:50] if result.answer else "",
                    "sources": result.sources,
                    "confidence": result.confidence,
                }
                e.confidence = result.confidence
                e.latency_ms = result.latency_ms
                break

    def save(self, request_id: str) -> None:
        entry = self._entries.get(request_id)
        if entry is None:
            return
        file_path = self._log_dir / f"{request_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f, ensure_ascii=False, indent=2)
