"""阶段一：多意图解析服务。

注入 llm_recognize 回调批量识别意图数组；LLM 输出解析失败自动重试 2 次，
仍失败则降级为规则关键词兜底；逐条过滤低置信意图（<0.6）并做槽位完备性校验。
"""


from .config import ConfigManager
from .llm_json import call_with_timeout, parse_intent_array
from .models import FirstStageResult, IntentRecognizeItem
from .protocols import LLMRecognize
from .rule import RuleCheckService

_MAX_PARSE_RETRIES = 2


class FirstStageIntentService:
    """阶段一：多意图识别 + 置信度过滤 + 槽位完备性校验。"""

    def __init__(
        self,
        config: ConfigManager,
        rule: RuleCheckService | None = None,
        llm_recognize: LLMRecognize | None = None,
        llm_timeout: float = 0,
    ):
        self.config = config
        self.rule = rule or RuleCheckService(config.get_risk_keywords())
        self.llm_recognize = llm_recognize
        self.llm_timeout = llm_timeout

    # ---------- 主入口 ----------

    def recognize(
        self, text: str, history: list[str] | None = None
    ) -> FirstStageResult:
        """阶段一识别。优先 LLM，失败回退规则关键词。"""
        items = self._llm_parse(text, history)
        source = "llm"
        if not items:
            items = self.rule_fallback(text)
            source = "rule" if items else "none"
        return self.check_completeness(items, source)

    # ---------- LLM 解析 ----------

    def _llm_parse(
        self, text: str, history: list[str] | None
    ) -> list[IntentRecognizeItem]:
        if not self.llm_recognize:
            return []
        history = history or []
        prompt = self._build_prompt(text, history)
        raw = call_with_timeout(
            lambda: self.llm_recognize(prompt, history), self.llm_timeout
        )
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            parsed = parse_intent_array(raw or "")
            items = self._normalize(parsed)
            if items:
                return items
            if attempt < _MAX_PARSE_RETRIES:
                raw = call_with_timeout(
                    lambda: self.llm_recognize(prompt, history), self.llm_timeout
                )
        return []

    def _build_prompt(self, text: str, history: list[str]) -> str:
        lines = ["你是一个多意图识别器。根据用户输入与历史上下文，识别出全部意图。"]
        lines.append(
            "输出格式：JSON 数组，每个元素包含 intent_id / name / confidence(0~1) / "
            "complete / miss_slots。"
        )
        lines.append(
            "complete 为 true 表示必填参数齐全；false 时 miss_slots 列出缺失的必填参数键。"
        )
        lines.append("可选意图定义：")
        for m in self.config.get_all_intents():
            req = "、".join(m.required_slots) or "无"
            opt = "、".join(m.optional_slots) or "无"
            lines.append(
                f"- {m.intent_id}（{m.name}）：{m.desc}；必填槽位: {req}；可选槽位: {opt}"
            )
        for h in history[-3:]:
            lines.append(f"历史：{h}")
        lines.append(f"用户输入：{text}")
        lines.append("只输出 JSON 数组，不要输出其他文字。")
        return "\n".join(lines)

    def _normalize(self, parsed: list[dict]) -> list[IntentRecognizeItem]:
        """过滤未知意图 / 低置信 / 重复，规整为 IntentRecognizeItem。"""
        items: list[IntentRecognizeItem] = []
        seen = set()
        threshold = self.config.confidence_threshold
        for d in parsed:
            intent_id = (
                str(d.get("intent_id", "")).strip() or str(d.get("intent", "")).strip()
            )
            if not intent_id:
                continue
            meta = self.config.get_intent(intent_id)
            if not meta:
                # 按名称匹配兜底
                for m in self.config.get_all_intents():
                    if m.name == intent_id:
                        meta, intent_id = m, m.intent_id
                        break
            if not meta:
                continue  # 未知意图丢弃
            try:
                confidence = float(d.get("confidence", 0.6))
            except (TypeError, ValueError):
                confidence = 0.6
            if confidence < threshold:
                continue
            if intent_id in seen:
                continue
            seen.add(intent_id)
            miss = [str(s) for s in (d.get("miss_slots") or []) if s]
            items.append(
                IntentRecognizeItem(
                    intent_id=meta.intent_id,
                    name=meta.name,
                    confidence=round(confidence, 4),
                    complete=bool(d.get("complete", False)),
                    miss_slots=miss,
                )
            )
        return items

    # ---------- 规则兜底 ----------

    def rule_fallback(self, text: str) -> list[IntentRecognizeItem]:
        """规则关键词兜底：命中任一意图关键词即产出该意图。"""
        items: list[IntentRecognizeItem] = []
        seen = set()
        for m in self.config.get_all_intents():
            if any(k in text for k in m.keywords):
                if m.intent_id in seen:
                    continue
                seen.add(m.intent_id)
                items.append(
                    IntentRecognizeItem(
                        intent_id=m.intent_id,
                        name=m.name,
                        confidence=0.7,
                        complete=False,
                        miss_slots=list(m.required_slots),
                    )
                )
        return items

    # ---------- 完备性校验 ----------

    def check_completeness(
        self, items: list[IntentRecognizeItem], source: str = "llm"
    ) -> FirstStageResult:
        """按配置必填槽位对齐完备性，汇总缺失槽位。"""
        total_miss: list[str] = []
        all_complete = True
        for item in items:
            meta = self.config.get_intent(item.intent_id)
            if meta:
                # 仅保留配置中真实存在的必填槽位
                miss = [s for s in item.miss_slots if s in meta.required_slots]
            else:
                miss = list(item.miss_slots)
            item.miss_slots = miss
            item.complete = not miss
            if not item.complete:
                all_complete = False
                total_miss.extend(miss)
        dedup: list[str] = []
        for s in total_miss:
            if s not in dedup:
                dedup.append(s)
        return FirstStageResult(
            intent_list=items,
            all_complete=all_complete,
            total_miss_slots=dedup,
            source=source,
        )
