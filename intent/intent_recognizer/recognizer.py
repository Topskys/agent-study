"""门面编排：IntentRecognizer 全流程（v3 两阶段多意图识别）。

识别流程（recognize）：
1. 预处理净化（TextPreprocessService）：错别字 / 代词消解 / 短提问扩写
2. 规则前置校验（RuleCheckService）：高危拦截 / 系统级硬信号短路
3. 阶段一多意图解析（FirstStageIntentService）：LLM（重试 2 次）→ 规则关键词兜底
4. 槽位缓存合并 + 完备性判定
5. 置信度三档分发：>0.9 执行 / 0.6~0.9 消歧反问 / <0.6 重新输入
6. 未完备 → 聚合追问（AskPromptService）
7. 依赖解析（IntentDependService）→ 阶段二批量抽槽（SecondStageSlotService）
8. 风险分级 + 调度执行（TaskScheduleService）
9. 检查点 / 审计事件持久化（MemoryStore / EventStore）

设计要点：
- LLM 能力全部注入式回调，本包不 import 任何 LLM SDK；
- 数据访问层直连 agent_memory.db 现有记忆模块数据表，不新建会话表。
"""

from dataclasses import asdict
from typing import Any

from .ask_prompt import AskPromptService
from .config import ConfigManager
from .depend import IntentDependService
from .first_stage import FirstStageIntentService
from .models import ExecutionPlan, FirstStageResult, IntentRecognizeItem, RuleHit
from .preprocess import TextPreprocessService
from .protocols import AskUser, LLMExpand, LLMExtractSlots, LLMRecognize, TaskExecutor
from .rule import RuleCheckService
from .scheduler import TaskScheduleService
from .second_stage import SecondStageSlotService
from .stores import EventStore, KvStore, MemoryStore, ProfileStore

_DEFAULT_USER = "default_user"
_DEFAULT_SESSION = "default_session"


class IntentRecognizer:
    """意图识别总编排（v3 两阶段）。

    参数：
    - config: 配置管理器（默认加载 resources/intent_config.json）
    - db_path: agent_memory.db 路径；None 时数据访问层退化为空操作
    - llm_recognize / llm_extract_slots / ask_user / llm_expand: 注入式回调
    - executor: 任务执行回调（可选，调度阶段不注入则不执行）
    - llm_timeout: 单次 LLM 调用超时秒数（<=0 不设超时，由宿主控制）
    - high_risk_keywords: 覆盖默认高危清单
    """

    def __init__(
        self,
        config: ConfigManager | None = None,
        db_path: str | None = None,
        llm_recognize: LLMRecognize | None = None,
        llm_extract_slots: LLMExtractSlots | None = None,
        ask_user: AskUser | None = None,
        llm_expand: LLMExpand | None = None,
        executor: TaskExecutor | None = None,
        llm_timeout: float = 0,
        high_risk_keywords: list[str] | None = None,
    ):
        self.kv = KvStore(db_path)
        self.config = config or ConfigManager(kv=self.kv)
        if high_risk_keywords is not None:
            self.config.high_risk_keywords = list(high_risk_keywords)

        self.store = MemoryStore(db_path)
        self.profile = ProfileStore(db_path)
        self.events = EventStore(db_path)

        self.ask_user = ask_user
        self.executor = executor
        self.llm_timeout = llm_timeout

        self.preprocess = TextPreprocessService(
            business_vocab=self.config.get_vocab(), llm_expand=llm_expand
        )
        self.rule = RuleCheckService(self.config.get_risk_keywords())
        self.first_stage = FirstStageIntentService(
            config=self.config,
            rule=self.rule,
            llm_recognize=llm_recognize,
            llm_timeout=llm_timeout,
        )
        self.ask = AskPromptService(self.config)
        self.depend = IntentDependService()
        self.second_stage = SecondStageSlotService(
            config=self.config,
            llm_extract_slots=llm_extract_slots,
            llm_timeout=llm_timeout,
        )
        self.scheduler = TaskScheduleService()

    # ---------- 主入口 ----------

    def recognize(
        self,
        text: str,
        history: list[str] | None = None,
        user_id: str = _DEFAULT_USER,
        session_id: str = _DEFAULT_SESSION,
    ) -> ExecutionPlan:
        """识别一次输入，返回两阶段执行计划。"""
        history = history or []
        original = str(text or "")
        processed, pre_ambiguous = self.preprocess.process(original, history)

        # ① 规则前置校验：高危拦截
        rule_hit = self.rule.check(processed)
        if rule_hit.blocked:
            self._record_event(
                EventStore.EVENT_INTENT_RECOGNIZED,
                user_id,
                session_id,
                {"blocked": True, "reason": rule_hit.block_reason, "source": "rule"},
            )
            return ExecutionPlan(
                intents=[],
                risk_level="high",
                blocked=True,
                ambiguous=True,
                source="rule",
                original=original,
                processed=processed,
                ask_prompt=self._block_prompt(rule_hit.block_reason),
            )

        # ② 系统级硬信号短路（确定性，不调 LLM）
        if rule_hit.intent_id:
            return self._build_rule_plan(
                rule_hit, original, processed, user_id, session_id
            )

        # ③ 阶段一：多意图识别（LLM → 规则兜底）
        first = self.first_stage.recognize(processed, history)
        self._merge_slot_cache(first, user_id, session_id)

        # ④ 意图集合为空 → 请重新输入
        if not first.intent_list:
            self._record_event(
                EventStore.EVENT_ASK_PROMPT,
                user_id,
                session_id,
                {"reason": "no_intent", "source": first.source},
            )
            return ExecutionPlan(
                intents=[],
                ambiguous=True,
                source=first.source,
                original=original,
                processed=processed,
                ask_prompt="抱歉，我没有识别到你想要的操作，请换一种说法重新输入。",
            )

        # ⑤ 风险折算 + 置信度三档分发
        # 低置信（<0.6）意图已在阶段一被统一阈值过滤；全部被过滤即落入
        # 上一步"无意图 → 请重新输入"（对应调度器 <0.6 分支）。
        actions = self._build_actions(first.intent_list, processed)
        risk = self.rule.assess_risk(processed, actions)
        high = self.config.high_confidence_threshold

        if risk == "high":
            self._record_event(
                EventStore.EVENT_INTENT_RECOGNIZED,
                user_id,
                session_id,
                {"intents": [i.intent_id for i in first.intent_list], "risk": "high"},
            )
            return ExecutionPlan(
                intents=first.intent_list,
                risk_level="high",
                blocked=True,
                ambiguous=True,
                source=first.source,
                original=original,
                processed=processed,
                ask_prompt=self._block_prompt("检测到高危操作"),
            )

        if any(it.confidence <= high for it in first.intent_list):
            # ② 0.6~0.9 → 消歧反问（聚合确认）
            prompt = self.ask.build_confirm_prompt(first.intent_list)
            self._record_event(
                EventStore.EVENT_ASK_PROMPT,
                user_id,
                session_id,
                {
                    "disambiguation": True,
                    "intents": [i.intent_id for i in first.intent_list],
                },
            )
            return ExecutionPlan(
                intents=first.intent_list,
                risk_level=risk,
                ambiguous=True,
                source=first.source,
                original=original,
                processed=processed,
                ask_prompt=prompt,
            )

        # ⑥ 未完备 → 聚合追问（缺失字段一次问全）
        if not first.all_complete:
            prompt = self.ask.build_prompt(first)
            self.store.write_intent_cache(user_id, session_id, first)
            self._record_event(
                EventStore.EVENT_ASK_PROMPT,
                user_id,
                session_id,
                {
                    "missing_slots": first.total_miss_slots,
                    "intents": [i.intent_id for i in first.intent_list],
                },
            )
            return ExecutionPlan(
                intents=first.intent_list,
                risk_level=risk,
                ambiguous=True,
                source=first.source,
                original=original,
                processed=processed,
                ask_prompt=prompt or "请补充必要的信息。",
            )

        # ⑦ 完备且高置信：依赖解析 → 阶段二抽槽 → 调度执行
        ordered = self._order_intents_by_text(first.intent_list, processed)
        task_groups = self.depend.parse(ordered, processed)
        second = self.second_stage.extract(processed, history, ordered)

        if second.invalid_slots:
            prompt = self.ask.build_invalid_prompt(second.invalid_slots)
            self._record_event(
                EventStore.EVENT_ASK_PROMPT,
                user_id,
                session_id,
                {"invalid_slots": second.invalid_slots},
            )
            return ExecutionPlan(
                intents=first.intent_list,
                risk_level=risk,
                ambiguous=True,
                source=first.source,
                original=original,
                processed=processed,
                ask_prompt=prompt,
            )

        slots = self._collect_slots(second, user_id, session_id)
        self.store.write_slot_cache(user_id, session_id, slots)
        self.store.write_intent_cache(user_id, session_id, first)

        self._record_event(
            EventStore.EVENT_INTENT_RECOGNIZED,
            user_id,
            session_id,
            {
                "intents": [i.intent_id for i in first.intent_list],
                "confidence": [i.confidence for i in first.intent_list],
                "source": first.source,
            },
        )
        self._record_event(
            EventStore.EVENT_SLOT_EXTRACTED, user_id, session_id, {"slots": slots}
        )
        self._record_event(
            EventStore.EVENT_TASK_SCHEDULED,
            user_id,
            session_id,
            {"groups": [g.intent_ids for g in task_groups]},
        )

        execution_results = self.scheduler.schedule(task_groups, self.executor, slots)

        return ExecutionPlan(
            intents=first.intent_list,
            slots=slots,
            task_groups=task_groups,
            risk_level=risk,
            source=first.source,
            original=original,
            processed=processed,
            execution_results=execution_results,
        )

    def recognize_debug(
        self,
        text: str,
        history: list[str] | None = None,
        user_id: str = _DEFAULT_USER,
        session_id: str = _DEFAULT_SESSION,
    ) -> dict:
        """调试视图：暴露中间各阶段结果。"""
        history = history or []
        original = str(text or "")
        processed, pre_ambiguous = self.preprocess.process(original, history)
        rule_hit = self.rule.check(processed)
        first = self.first_stage.recognize(processed, history)
        self._merge_slot_cache(first, user_id, session_id)
        plan = self.recognize(original, history, user_id, session_id)
        return {
            "original": original,
            "processed": processed,
            "preprocess_ambiguous": pre_ambiguous,
            "rule_hit": asdict(rule_hit),
            "first_stage": first.to_dict(),
            "plan": plan.to_dict(),
        }

    # ---------- 内部：规则硬信号短路 ----------

    def _build_rule_plan(
        self,
        hit: RuleHit,
        original: str,
        processed: str,
        user_id: str,
        session_id: str,
    ) -> ExecutionPlan:
        item = IntentRecognizeItem(
            intent_id=hit.intent_id,
            name=hit.intent_id,
            confidence=hit.confidence,
            complete=True,
        )
        risk = self.rule.assess_risk(processed, hit.actions)
        self._record_event(
            EventStore.EVENT_INTENT_RECOGNIZED,
            user_id,
            session_id,
            {
                "intents": [hit.intent_id],
                "confidence": hit.confidence,
                "source": "rule",
            },
        )
        return ExecutionPlan(
            intents=[item],
            slots=hit.slots,
            risk_level=risk,
            source="rule",
            original=original,
            processed=processed,
        )

    # ---------- 内部：槽位缓存 / 收集 / 排序 ----------

    def _merge_slot_cache(self, first: FirstStageResult, user_id: str, session_id: str):
        """已填槽位缓存合并：缓存中已有的槽位不再追问。"""
        cached = self.store.read_slot_cache(user_id, session_id)
        if not cached:
            return
        for item in first.intent_list:
            per_intent = cached.get(item.intent_id, {})
            if not per_intent:
                continue
            kept = [s for s in item.miss_slots if s not in per_intent]
            item.miss_slots = kept
            item.complete = not kept
        first.all_complete = all(i.complete for i in first.intent_list)
        dedup: list[str] = []
        for item in first.intent_list:
            for s in item.miss_slots:
                if s not in dedup:
                    dedup.append(s)
        first.total_miss_slots = dedup

    def _collect_slots(self, second, user_id: str, session_id: str) -> dict[str, Any]:
        """阶段二结果 + 会话已缓存槽位合并，结构 {intent_id: {slot_key: value}}。"""
        merged = dict(self.store.read_slot_cache(user_id, session_id))
        for r in second.slot_results:
            merged.setdefault(r.intent_id, {})
            merged[r.intent_id].update(r.slot_kv)
        return merged

    def _order_intents_by_text(
        self, intents: list[IntentRecognizeItem], text: str
    ) -> list[IntentRecognizeItem]:
        """按业务关键词在文本中的最早出现位置排序（供串行依赖解析）。"""
        positions: dict[str, int] = {}
        for it in intents:
            meta = self.config.get_intent(it.intent_id)
            pos = len(text) or 0
            for kw in meta.keywords if meta else [it.intent_id]:
                idx = text.find(kw)
                if idx != -1 and idx < pos:
                    pos = idx
            positions[it.intent_id] = pos
        return sorted(intents, key=lambda it: positions[it.intent_id])

    # ---------- 内部：风险 / 审计 ----------

    def _build_actions(
        self, intents: list[IntentRecognizeItem], text: str
    ) -> list[dict]:
        actions: list[dict] = []
        for i, it in enumerate(intents):
            meta = self.config.get_intent(it.intent_id)
            actions.append(
                {
                    "action": meta.name if meta else it.intent_id,
                    "target": text,
                    "priority": i + 1,
                }
            )
        return actions

    def _record_event(
        self, event_type: str, user_id: str, session_id: str, extra: dict
    ):
        payload = {"user_id": user_id, "session_id": session_id}
        payload.update(extra)
        self.events.record(event_type, payload)

    @staticmethod
    def _block_prompt(reason: str) -> str:
        return f"安全拦截：{reason}，已停止执行。如需继续请明确说明并缩小操作范围。"
