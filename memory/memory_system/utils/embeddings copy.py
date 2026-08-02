"""文本向量（Embedding）相关工具函数。

提供向量相似度计算与（占位用的）确定性向量生成能力，
用于记忆的相似度检索与去重。
"""

import math
import random
from typing import List, Optional


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


def generate_embedding(text: str, dimensions: int = 128) -> List[float]:
    """根据文本生成固定维度的确定性向量（占位实现）。

    使用文本的哈希值作为随机种子，保证同一文本生成相同向量。

    参数:
        text: 输入文本。
        dimensions: 生成向量的维度，默认 128。
    返回:
        长度等于 dimensions 的浮点向量列表。
    """
    seed = hash(text) & 0x7FFFFFFF
    rng = random.Random(seed)
    return [rng.random() for _ in range(dimensions)]
