"""
数据飞轮——DataFlywheel

记录每次查询的完整数据，支持高置信度结果自动入库和低置信度结果待审核。
让系统越用越聪明。
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from rag.datatypes import LogEntry, RAGResult, SearchResult


class DataFlywheel:
    """数据飞轮：记录查询→收集反馈→持续优化"""

    def __init__(self, data_dir: str = "./data/flywheel"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._auto_archive_threshold = 0.85
        self._review_threshold = 0.6

    def record(
        self,
        query: str,
        rewritten_query: str,
        results: list[SearchResult],
        result: RAGResult,
    ):
        entry = {
            "request_id": result.request_id,
            "query": query,
            "rewritten_query": rewritten_query,
            "retrieved_chunks": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "source": r.chunk.metadata.get("source", ""),
                    "score": r.score,
                }
                for r in results
            ],
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "user_feedback": None,
        }
        file_path = self._data_dir / f"{result.request_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

        if result.confidence >= self._auto_archive_threshold:
            self._archive(entry)
        elif result.confidence < self._review_threshold:
            self._mark_for_review(entry)

    def submit_feedback(self, request_id: str, feedback: str) -> bool:
        file_path = self._data_dir / f"{request_id}.json"
        if not file_path.exists():
            return False
        with open(file_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        entry["user_feedback"] = feedback
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
        if feedback == "positive":
            self._archive(entry)
        return True

    def _archive(self, entry: dict):
        archive_dir = self._data_dir / "archived"
        archive_dir.mkdir(exist_ok=True)
        dst = archive_dir / f"{entry['request_id']}.json"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    def _mark_for_review(self, entry: dict):
        review_dir = self._data_dir / "review"
        review_dir.mkdir(exist_ok=True)
        dst = review_dir / f"{entry['request_id']}.json"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)
