"""第二层：SemanticReasoner 会话语义推理层。

按 V3 §5.2：
- 轻量模型分类（注入 classifier 回调）
- Embedding 语义相似度匹配（注入 embedder 回调，可选）
- 双路融合打分：融合分 = alpha*分类分 + (1-alpha)*相似分
- DST：命中时读取/写入会话状态（由上层 DialogStateTracker 完成）
核心职责：承接常规流量，解决同义/指代/多轮省略；置信 <0.9 时让路 LLM 层。

本层保持纯逻辑、无 LLM SDK 依赖；classifier / similarity 由宿主注入。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from .config import FunnelConfig
from .models import IntentRankingItem


@dataclass
class SemanticScore:
    """单意图语义得分（融合后）。"""

    name: str
    cls: float = 0.0
    sim: float = 0.0
    fused: float = 0.0


@dataclass
class SemanticAnnotation:
    """语义推理输出。"""

    text: str
    scores: list[SemanticScore] = field(default_factory=list)

    def ranking(self) -> list[IntentRankingItem]:
        return [
            IntentRankingItem(name=s.name, confidence=round(s.fused, 4))
            for s in sorted(self.scores, key=lambda x: x.fused, reverse=True)
        ]

    @property
    def max_fused(self) -> float:
        return max((s.fused for s in self.scores), default=0.0)

    @property
    def top(self) -> SemanticScore | None:
        if not self.scores:
            return None
        return max(self.scores, key=lambda s: s.fused)


# 注入式回调类型
Classifier = Callable[[str, list[str]], dict[str, float]]  # text, history -> {intent: prob}
Similarity = Callable[[str, str], float]  # text, intent_name -> 相似度


class SemanticReasoner:
    """会话语义推理：分类 + 向量相似度 双路融合。"""

    def __init__(
        self,
        config: FunnelConfig,
        classifier: Classifier | None = None,
        similarity: Similarity | None = None,
        alpha: float = 0.5,
        confidence_threshold: float = 0.9,
    ):
        self.config = config
        self.classifier = classifier
        self.similarity = similarity
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold

    # ---------- 主入口 ----------

    def infer(self, text: str, history: list[str] | None = None) -> SemanticAnnotation:
        """对文本做语义打分（未挂阈值过滤，由上层判断）。"""
        history = history or []
        cls_scores = self._classify(text, history)
        sim_scores = self._similarity(text)

        all_names = set(cls_scores) | set(sim_scores)
        scores: list[SemanticScore] = []
        for name in sorted(all_names):
            if self.config.get_intent(name) is None:
                continue
            c = max(0.0, cls_scores.get(name, 0.0))
            s = max(0.0, sim_scores.get(name, 0.0))
            fused = self.alpha * c + (1.0 - self.alpha) * s
            scores.append(
                SemanticScore(name=name, cls=round(c, 4), sim=round(s, 4), fused=round(fused, 4))
            )
        return SemanticAnnotation(text=text, scores=scores)

    def assess_high(self, annotation: SemanticAnnotation) -> bool:
        """是否达到语义层高可信（>= threshold 的 maxConf）。"""
        return annotation.max_fused >= self.confidence_threshold

    # ---------- 双路打分 ----------

    def _classify(self, text: str, history: list[str]) -> dict[str, float]:
        if not self.classifier:
            return {}
        try:
            raw = self.classifier(text, history) or {}
        except Exception:  # noqa: BLE001 - 注入回调失败按无分类处理
            return {}
        return {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    def _similarity(self, text: str) -> dict[str, float]:
        if not self.similarity:
            return {}
        results: dict[str, float] = {}
        for spec in self.config.all_intents():
            try:
                score = self.similarity(text, spec.name)
            except Exception:  # noqa: BLE001, S112 - 注入回调失败按无相似度处理
                continue
            if isinstance(score, (int, float)):
                results[spec.name] = float(score)
        return results