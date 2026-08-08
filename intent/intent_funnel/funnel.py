"""顶层调度：IntentRecognition 门面（三层漏斗运行）。

按 V3 §七 伪代码定稿实现 recognize()：
1. 预处理 + 会话加载/槽位缓存回填
2. RuleMatcher（高危拦截 → 拒绝；命中 → 完成度/缓存落盘后返回）
3. SemanticReasoner ≥0.9 → 完成度/返回；否则下放
4. ComplexIntentParser → 四分支（执行 / 缺失追问 / 消歧 / 无意图重输）
5. DialogStateTracker 更新会话状态（槽位快照 / 检查点）

本模块不 import 任何 LLM SDK；LLM/语义能力全部注入式回调。
"""

import uuid
from collections.abc import Callable

from .complex_parser import ComplexIntentParser
from .config import FunnelConfig
from .dialog import (
    AskPromptBuilder,
    DialogStateTracker,
    IntentOrchestrator,
    SlotCompletenessChecker,
)
from .models import IntentItem, IntentRankingItem, IntentResult
from .preprocess import normalize
from .rule_matcher import RuleMatcher, RuleMatchResult
from .semantic import SemanticReasoner


class FunnelIntentRecognition:
    """意图识别总入口：三层漏斗路由。"""

    def __init__(
        self,
        config: FunnelConfig | None = None,
        store=None,
        llm_gen: Callable[[str], str] | None = None,
        llm_timeout: float = 0,
        classifier=None,
        similarity=None,
        high_risk_keywords: list[str] | None = None,
        semantic_threshold: float | None = None,
    ):
        self.config = config or FunnelConfig()
        self.slot_checker = SlotCompletenessChecker(self.config)
        self.ask_builder = AskPromptBuilder(self.config)
        self.orchestrator = IntentOrchestrator()
        self.tracker = DialogStateTracker(store)

        self.rule = RuleMatcher(
            self.config, high_risk_keywords=high_risk_keywords
        )
        self.semantic = SemanticReasoner(
            self.config,
            classifier=classifier,
            similarity=similarity,
            confidence_threshold=semantic_threshold or self.config.semantic_high,
        )
        self.llm = ComplexIntentParser(
            self.config, llm_gen=llm_gen, llm_timeout=llm_timeout
        )

    # ---------- 主入口 ----------

    def recognize(
        self,
        text: str,
        history: list[str] | None = None,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> IntentResult:
        text = normalize(text)
        history = history or []
        request_id = uuid.uuid4().hex[:16]
        result: IntentResult

        # ① 前置规则匹配（高危拦截 / 确定性命中）
        rule = RuleMatchResult(*self.rule.match(text))
        if rule.blocked:
            return IntentResult(
                source_layer="rule_matcher",
                text=text,
                blocked=True,
                block_reason=rule.reason,
                session_id=session_id,
                request_id=request_id,
                ask_prompt=f"安全拦截：{rule.reason}，已阻止执行。如需继续请明确说明并缩小操作范围。",
            )
        if rule.matches:
            result = self._from_rule(rule, text)
            return self._finalize(result, text, history, user_id, session_id, request_id)

        # ② 语义层：双路融合
        annotation = self.semantic.infer(text, history)
        if self.semantic.assess_high(annotation):
            result = self._from_semantic(annotation, text)
            return self._finalize(result, text, history, user_id, session_id, request_id)

        # ③ LLM 兜底
        items = self.llm.parse(text, history)
        if not items:
            return self.no_valid(text, session_id, request_id)
        result = IntentResult(
            source_layer="complex_parser",
            text=text,
            intents=items,
            intent_ranking=self._ranking(items),
        )
        return self._finalize(result, text, history, user_id, session_id, request_id)

    # ---------- 分层结果构建 ----------

    def _from_rule(self, rule: RuleMatchResult, text: str) -> IntentResult:
        intents = []
        for m in rule.matches:
            intents.append(
                IntentItem(
                    name=m.intent,
                    confidence=m.confidence,
                    priority=self._priority(m.intent),
                    entities=m.entities,
                )
            )
        return IntentResult(
            source_layer="rule_matcher",
            text=text,
            intents=intents,
            intent_ranking=self._ranking(intents),
        )

    def _from_semantic(self, annotation, text: str) -> IntentResult:
        intents = [
            IntentItem(
                name=s.name,
                confidence=s.fused,
                priority=self._priority(s.name),
                entities=[],
            )
            for s in annotation.scores
            if s.fused >= self.config.semantic_high
        ]
        return IntentResult(
            source_layer="semantic_reasoner",
            text=text,
            intents=intents,
            intent_ranking=annotation.ranking(),
        )

    # ---------- 统一收口（槽位/消歧/无意图 + 缓存落盘） ----------

    def _finalize(
        self,
        result: IntentResult,
        text: str,
        history: list[str],
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> IntentResult:
        result.session_id = session_id
        result.request_id = request_id
        session = self.tracker.get_session(user_id, session_id)
        session.historical_queries = (session.historical_queries + [text])[-20:]

        if not result.intents:
            return self.no_valid(text, session_id, request_id)

        # 槽位完备性：跨意图缺失一次问全（扣除会话缓存）
        missing_all = self.slot_checker.check(result.intents, session)

        invalid = [
            {"slot": e.intent, "new": e.value}
            for it in result.intents
            for e in it.entities
            if not e.valid
        ]

        high = result.high_confidence
        if result.source_layer == "rule_matcher":
            # 确定性命中：仅缺失追问，不做消歧
            if missing_all:
                result.need_ask_slots = True
                result.ask_slots = list(missing_all)
                result.ask_prompt = self.ask_builder.build_slots(missing_all)
            self.tracker.update_after_result(session, result)
            return result

        if result.source_layer == "semantic_reasoner" and high >= self.config.semantic_high:
            if missing_all:
                result.need_ask_slots = True
                result.ask_slots = list(missing_all)
                result.ask_prompt = self.ask_builder.build_slots(missing_all)
            self.tracker.update_after_result(session, result)
            return result

        # LLM 层三段式
        if high >= self.config.llm_high:
            if invalid:
                result.need_ask_slots = True
                result.ask_slots = [e["slot"] for e in invalid]
                result.ask_prompt = self.ask_builder.build_invalid(invalid)
            elif missing_all:
                result.need_ask_slots = True
                result.ask_slots = list(missing_all)
                result.ask_prompt = self.ask_builder.build_slots(missing_all)
            self.tracker.update_after_result(session, result)
            return result
        if high >= self.config.llm_low:
            result.need_disambiguate = True
            result.ask_prompt = self.ask_builder.build_disambiguate(result.intent_ranking)
            return result
        return self.no_valid(text, session_id, request_id)

    def no_valid(self, text: str, session_id: str, request_id: str) -> IntentResult:
        return IntentResult(
            source_layer="complex_parser",
            text=text,
            no_valid_intent=True,
            ask_prompt=self.ask_builder.build_reenter(),
            session_id=session_id,
            request_id=request_id,
        )

    # ---------- 辅助 ----------

    def _priority(self, name: str) -> int:
        spec = self.config.get_intent(name)
        return spec.priority if spec else 1

    def _ranking(self, intents: list[IntentItem]) -> list[IntentRankingItem]:
        return [
            IntentRankingItem(name=i.name, confidence=i.confidence)
            for i in sorted(intents, key=lambda x: x.confidence, reverse=True)
        ]