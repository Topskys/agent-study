"""记忆评分（MemoryScorer）治理模块。

负责计算记忆的相关度评分，采用四维加权公式综合衡量记忆的
重要性（importance）、稳定性（stability）、复用率（reuse）与
时效性（recency），评分用于长期记忆筛选与容量淘汰等决策。
"""

import time

from ..models.memory_item import MemoryItem


class MemoryScorer:
    """记忆评分器：对记忆条目进行四维加权评分。"""

    def __init__(
        self,
        importance_weight: float = 0.4,
        stability_weight: float = 0.2,
        reuse_weight: float = 0.2,
        recency_weight: float = 0.2,
    ):
        # 四维评分的权重系数，默认重要性占主导（0.4）
        self.importance_weight = importance_weight
        self.stability_weight = stability_weight
        self.reuse_weight = reuse_weight
        self.recency_weight = recency_weight

    def score(self, item: MemoryItem) -> float:
        """计算记忆的四维加权评分。

        评分公式：
            total = importance_score * importance_weight
                  + stability_score  * stability_weight
                  + reuse_score      * reuse_weight
                  + recency_score    * recency_weight
        四个维度得分均归一化到 [0, 1] 区间，最终结果四舍五入保留 4 位小数。

        :param item: 待评分的记忆条目
        :return: 加权后的综合评分（float）
        """
        # 维度一：重要性（importance），从元数据读取，默认 0.5，并裁剪到 [0, 1]
        importance = item.metadata.get("importance", 0.5)
        if isinstance(importance, (int, float)):
            importance_score = min(1.0, max(0.0, importance))
        else:
            importance_score = 0.5

        # 维度二：稳定性（stability），基于版本号，版本越高越稳定，满分为版本 10
        version = getattr(item, "version", 1)
        stability_score = min(1.0, version / 10.0)

        # 维度三：复用率（reuse），基于历史复用次数，满分为复用 20 次
        reuse_count = item.metadata.get("reuse_count", 0)
        reuse_score = min(1.0, reuse_count / 20.0)

        # 维度四：时效性（recency），按最近更新时间距今的小时数衰减，
        # 30 天内呈线性衰减到 0；无更新时间时取中值 0.5
        if item.updated_at:
            hours_since_update = (time.time() - item.updated_at.timestamp()) / 3600
            recency_score = max(0.0, 1.0 - hours_since_update / (24 * 30))
        else:
            recency_score = 0.5

        # 四维加权求和，得到综合评分
        total = (
            importance_score * self.importance_weight
            + stability_score * self.stability_weight
            + reuse_score * self.reuse_weight
            + recency_score * self.recency_weight
        )
        return round(total, 4)
