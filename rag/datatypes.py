"""
RAG 系统——统一数据类型

所有模块共享的数据结构定义在此文件中。
各模块通过类型注解直接引用这些数据类，不依赖具体实现。
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RawDocument:
    """加载器输出：原始文档"""

    content: str  # 提取的文本内容
    source: str  # 源文件路径
    metadata: dict = field(default_factory=dict)  # 扩展元数据（文件名、类型、页码等）


@dataclass
class Chunk:
    """分块器输出：文本块"""

    chunk_id: str  # 唯一 ID (uuid)
    text: str  # 分块后的文本
    metadata: dict = field(default_factory=dict)  # 继承自 RawDocument 的元数据
    embeddings: dict[str, list[float]] = field(
        default_factory=dict
    )  # {嵌入器名称: 向量}
    parent_id: Optional[str] = None  # 父 Chunk ID（父子分块时使用）


@dataclass
class SearchResult:
    """检索器输出：带分数的检索结果"""

    chunk: Chunk  # 匹配的文本块
    score: float = 0.0  # 相似度分数
    rank: int = 0  # 排序位置


@dataclass
class RAGResult:
    """生成器输出：最终 RAG 结果"""

    answer: str = ""  # 带引用的答案文本
    sources: list[dict] = field(
        default_factory=list
    )  # [{"id":1, "source":"file.pdf", "page":5, "text":"..."}]
    confidence: float = 0.0  # 置信度 0-1
    request_id: str = ""  # 请求追踪 ID
    latency_ms: float = 0.0  # 总耗时


@dataclass
class LogEntry:
    """全链路日志条目"""

    request_id: str = ""
    timestamp: str = ""
    user_query: str = ""
    pipeline: dict = field(default_factory=dict)  # 各阶段耗时和结果
    retrieved_chunks: list[dict] = field(default_factory=list)
    generation: dict = field(default_factory=dict)
    confidence: float = 0.0
    latency_ms: float = 0.0
    user_feedback: Optional[str] = None  # 用户反馈（数据飞轮使用）


__all__ = ["RawDocument", "Chunk", "SearchResult", "RAGResult", "LogEntry", "asdict"]
