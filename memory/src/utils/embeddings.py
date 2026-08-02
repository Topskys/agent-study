"""文本向量（Embedding）相关工具函数。

提供向量相似度计算与语义向量化能力，用于记忆的相似度检索与去重。

默认使用本地 BGE-small-zh-v1.5（ModelScope 下载，512 维）做真实语义嵌入。
模型懒加载、模块级单例缓存，首次调用才加载。

环境变量:
    MEMORY_EMBED_MODE=mock  使用确定性哈希向量（测试环境，秒级，无语义）
    默认 real                使用 BGE-small-zh-v1.5 真实语义嵌入
"""

import math
import os
import random
from pathlib import Path
from typing import List, Optional

# BGE 官方推荐的检索查询前缀（仅在查询时附加，入库时不加）
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 本地模型搜索路径（相对/绝对皆可），找不到回退到 HF 模型名
_MODEL_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "models" / "bge-small-zh-v1.5",
    "BAAI/bge-small-zh-v1.5",
]

_EMBED_DIM = 512


def _is_mock_mode() -> bool:
    return os.environ.get("MEMORY_EMBED_MODE", "real").lower() == "mock"


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """计算两个向量的余弦相似度。

    参数:
        vec_a: 向量 A。
        vec_b: 向量 B。
    返回:
        相似度值（0.0~1.0）；若维度不一致或存在零向量则返回 0.0。
    """
    if len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hash_embedding(text: str, dimensions: int = _EMBED_DIM) -> List[float]:
    """确定性哈希伪向量（mock 模式，仅用于测试速度，无语义）。"""
    seed = hash(text) & 0x7FFFFFFF
    rng = random.Random(seed)
    return [rng.random() for _ in range(dimensions)]


_model = None
_model_loaded = False


def _load_model():
    """懒加载本地 BGE 模型，模块级缓存。"""
    global _model, _model_loaded
    if _model_loaded:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "真实嵌入需要安装 sentence-transformers: uv add sentence-transformers"
        )

    last_err = None
    # 强制离线模式：本地模型加载时不联网检查更新（避免 HF 连通失败阻塞）
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for cand in _MODEL_CANDIDATES:
        try:
            # Windows 下 SentenceTransformer 不接受 Path 对象，须转 str
            _model = SentenceTransformer(str(cand))
            _model_loaded = True
            return _model
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"无法加载本地嵌入模型（{_MODEL_CANDIDATES[0]}），请先用 "
        f"modelscope download --model BAAI/bge-small-zh-v1.5 下载。原因: {last_err}"
    )


def generate_embedding(
    text: str, dimensions: int = _EMBED_DIM, for_query: bool = False
) -> List[float]:
    """将文本向量化。

    参数:
        text: 待向量化文本。
        dimensions: 目标维度（BGE-small-zh-v1.5 固定 512）。
        for_query: 是否为检索查询（查询时附加 BGE 官方前缀，入库文本勿用）。
    返回:
        长度等于 dimensions 的浮点向量列表。
    """
    if _is_mock_mode():
        return _hash_embedding(text, dimensions)

    model = _load_model()
    content = (_QUERY_PREFIX + text) if for_query else text
    emb = model.encode(content, normalize_embeddings=True)
    return emb.tolist()
