"""第一层：RuleMatcher 前置规则匹配层。

按 V3 §5.1：
- 文本清洗
- 高危词硬拦截（不进入语义与 LLM）
- 关键词 / 正则 / 固定句式命中 → 生成意图、槽位（静态抽取）+ 别名规范化
- 未命中放行语义层
"""

import re

from .config import FunnelConfig
from .models import EntityItem, IntentItem, IntentResult


class RuleMatch:
    """单条规则命中：意图名 + 置信度 + 静态槽位。"""

    def __init__(self, intent: str, confidence: float = 1.0, entities=None):
        self.intent = intent
        self.confidence = confidence
        self.entities: list[EntityItem] = entities or []


class RuleMatcher:
    """前置规则匹配：高危拦截 + 关键词/正则意图命中。"""

    def __init__(self, config: FunnelConfig, high_risk_keywords: list[str] | None = None):
        self.config = config
        self.high_risk_keywords: list[str] = (
            list(high_risk_keywords) if high_risk_keywords is not None else config.get_risk_keywords()
        )

    # ---------- 主入口 ----------

    def match(self, text: str) -> tuple:
        """返回 (blocked, reason, matches)。

        blocked=True 时直接拦截（matches=[]）；
        否则 matches 为命中列表（可为空 → 放行）。
        """
        text = text.strip()
        if not text:
            return False, "", []

        reason = self._match_high_risk(text)
        if reason:
            return True, reason, []

        matches = self._match_by_keywords(text)
        if matches:
            return False, "", matches
        return False, "", []

    def match_text(self, text: str) -> "RuleMatchResult | None":
        """便捷入口：返回 RuleMatchResult 或 None（未命中）。"""
        blocked, reason, matches = self.match(text)
        return RuleMatchResult(
            blocked=blocked, reason=reason, matches=matches, config=self.config
        )

    # ---------- 敏感词 ----------

    def _match_high_risk(self, text: str) -> str | None:
        for key in self.high_risk_keywords:
            if key and key in text:
                return f"命中高危操作：{key}"
        return None

    # ---------- 关键词/正则匹配 ----------

    def _match_by_keywords(self, text: str) -> list[RuleMatch]:
        matches: list[RuleMatch] = []
        for spec in self.config.all_intents():
            hit = any(k in text for k in spec.keywords)
            if not hit:
                continue
            entities = self._extract_static_slots(spec, text)
            matches.append(RuleMatch(intent=spec.name, confidence=0.9, entities=entities))
        return matches

    def _extract_static_slots(self, spec, text: str) -> list[EntityItem]:
        """基于槽位别名 + 正则做静态抽取；命中并规范化（raw_text/value/normalized_value）。

        按槽位顺序贪婪消费已命中区间，避免如金额正则误吞手机号。
        """
        entities: list[EntityItem] = []
        consumed: list[tuple] = []

        def _overlaps(start: int, end: int) -> bool:
            for s, e in consumed:
                if start < e and end > s:
                    return True
            return False

        for slot_key in spec.all_slot_keys():
            slot = spec.get_slot(slot_key)
            if not slot:
                continue
            found = False
            for target, entity in self._match_alias(slot, text):
                start = text.index(target)
                if not _overlaps(start, start + len(target)):
                    entities.append(entity)
                    consumed.append((start, start + len(target)))
                    found = True
                    break
            if found:
                continue
            for start, end, entity in self._match_regex(slot, text):
                if not _overlaps(start, end):
                    entities.append(entity)
                    consumed.append((start, end))
                    break
        return entities

    @staticmethod
    def _match_alias(slot, text: str):
        """返回 [(raw, EntityItem)] 候选项（按出现先后）。"""
        if not slot.aliases:
            return []
        candidates = []
        for raw, normalized in slot.aliases.items():
            idx = text.find(raw)
            if idx >= 0:
                candidates.append(
                    (
                        raw,
                        EntityItem(
                            intent=slot.slot,
                            value=raw,
                            normalized_value=normalized,
                            raw_text=raw,
                            confidence=1.0,
                            valid=True,
                        ),
                    )
                )
        return sorted(candidates, key=lambda c: text.find(c[0]))

    @staticmethod
    def _match_regex(slot, text: str) -> list[tuple]:
        """正则槽位抽取（去掉 ^/$ 锚点后做子串搜索），按出现先后返回 (start, end, entity)。"""
        pattern = slot.regex
        if not pattern:
            return []
        searchable = re.sub(r"^\^|\$$", "", pattern)
        try:
            it = re.finditer(searchable, text)
        except re.error:
            return []
        results = []
        for m in it:
            raw_val = m.group(0)
            results.append(
                (
                    m.start(),
                    m.end(),
                    EntityItem(
                        intent=slot.slot,
                        value=raw_val,
                        normalized_value=slot.normalize(raw_val),
                        raw_text=raw_val,
                        confidence=1.0,
                        valid=True,
                    ),
                )
            )
        return results


class RuleMatchResult:
    """规则层输出。"""

    def __init__(self, blocked: bool, reason: str = "", matches=None, config: FunnelConfig | None = None):
        self.blocked = blocked
        self.reason = reason
        self.matches: list[RuleMatch] = matches or []
        self.config = config

    def to_result(self, text: str) -> IntentResult:
        if self.blocked:
            return IntentResult(
                source_layer="rule_matcher",
                text=text,
                blocked=True,
                block_reason=self.reason or "高危操作拦截",
                ask_prompt=f"安全拦截：{self.reason or '高危操作'}，已停止执行。如需继续请明确说明并缩小操作范围。",
            )
        intents = [
            IntentItem(
                name=m.intent,
                confidence=m.confidence,
                priority=self._priority(m.intent),
                entities=m.entities,
            )
            for m in self.matches
        ]
        ranking = [self._ranking_item(i) for i in intents]
        return IntentResult(
            source_layer="rule_matcher",
            text=text,
            intents=intents,
            intent_ranking=ranking,
        )

    def _priority(self, name: str) -> int:
        spec = self.config.get_intent(name)
        return spec.priority if spec else 1

    def _ranking_item(self, item: IntentItem):
        from .models import IntentRankingItem

        return IntentRankingItem(name=item.name, confidence=item.confidence)