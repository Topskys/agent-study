"""第三层：ComplexIntentParser LLM 复杂意图解析层。

按 V3 §5.3：
- 组装会话上下文 + 意图 Schema + 工具定义 Prompt
- 调用大模型（超时控制）解析 FunctionCall
- 输出格式解析失败自动重试 2 次，仍失败降级为规则层兜底
- 槽位合法性校验（正则/枚举），非法槽位标记 valid=False
- 未知意图 / 低置信（<llm_low）过滤
"""

import re
from collections.abc import Callable

from .config import FunnelConfig
from .llm_json import call_with_timeout, parse_intent_json
from .models import EntityItem, IntentItem

_LLM_GEN = Callable[[str], str]  # prompt -> 原始输出（期望 JSON 意图数组）
_MAX_PARSE_RETRIES = 2


class ComplexIntentParser:
    """LLM 复杂意图解析：FunctionCall → 意图+槽位，含重试与降级。"""

    def __init__(
        self,
        config: FunnelConfig,
        llm_gen: _LLM_GEN | None = None,
        llm_timeout: float = 0,
    ):
        self.config = config
        self.llm_gen = llm_gen
        self.llm_timeout = llm_timeout

    # ---------- 主入口 ----------

    def parse(
        self, text: str, history: list[str] | None = None
    ) -> list[IntentItem]:
        """解析复杂查询 → 意图列表（已过滤未知/低置信、别名规范化、正则校验）。"""
        if not self.llm_gen:
            return []
        history = history or []
        prompt = self._build_prompt(text, history)
        raw = call_with_timeout(lambda: self.llm_gen(prompt), self.llm_timeout)
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            parsed = parse_intent_json(raw or "")
            items = self._normalize(parsed, text)
            if items:
                return items
            if attempt < _MAX_PARSE_RETRIES:
                raw = call_with_timeout(lambda: self.llm_gen(prompt), self.llm_timeout)
        return []  # 降级由上层（漏斗）触发规则层

    # ---------- 归一化 ----------

    def _normalize(self, parsed: list[dict], text: str) -> list[IntentItem]:
        items: list[IntentItem] = []
        seen = set()
        low = self.config.llm_low
        for d in parsed:
            name = str(d.get("intent", "") or d.get("intent_id", "") or d.get("function", "")).strip()
            if not name:
                name = str(d.get("name", "")).strip()
            spec = self.config.get_intent(name)
            if spec is None:
                continue  # 未知意图丢弃
            if name in seen:
                continue
            seen.add(name)
            try:
                confidence = float(d.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < low:
                continue  # 低置信丢弃

            raw_slots = d.get("slot_info") or d.get("arguments") or {}
            entities = self._build_entities(spec, raw_slots, text)
            items.append(
                IntentItem(
                    name=spec.name,
                    confidence=round(confidence, 4),
                    priority=spec.priority,
                    entities=entities,
                )
            )
        return items

    def _build_entities(self, spec, raw_slots: dict, text: str) -> list[EntityItem]:
        """构建槽位实体：正则校验 + 别名规范化（raw_text → normalized_value）。"""
        if not isinstance(raw_slots, dict):
            return []
        entities: list[EntityItem] = []
        for key, value in raw_slots.items():
            slot = spec.get_slot(key)
            if slot is None:
                continue
            raw_val = str(value) if value is not None else ""
            valid = self._validate(raw_val, slot.regex)
            normalized = slot.normalize(raw_val) if valid else ""
            entities.append(
                EntityItem(
                    intent=slot.slot,
                    value=raw_val,
                    normalized_value=normalized,
                    raw_text=self._find_in_text(raw_val, text) or raw_val,
                    confidence=1.0,
                    valid=valid,
                )
            )
        return entities

    @staticmethod
    def _find_in_text(value: str, text: str) -> str:
        if value and value in text:
            return value
        return ""

    @staticmethod
    def _validate(value: str, regex: str) -> bool:
        if not regex:
            return True
        try:
            return re.fullmatch(regex, value) is not None
        except re.error:
            return True

    # ---------- Prompt ----------

    def _build_prompt(self, text: str, history: list[str]) -> str:
        lines = [
            "你是复杂意图解析器。识别用户完整意图（含多意图、槽位参数抽取）。",
            (
                "输出格式：JSON 数组，每个元素 {intent: 意图名, confidence: 0~1, "
                "slot_info: {槽位键: 值}}。"
            ),
        ]
        if self.config.tool_schemas:
            lines.append("可用工具 Schema：")
            for t in self.config.tool_schemas:
                lines.append(
                    f"- {t.get('function_name') or t.get('function')}: "
                    f"{t.get('description', '')} "
                    f"参数={t.get('parameters', {})}"
                )
        lines.append("支持意图与槽位：")
        for spec in self.config.all_intents():
            slots = {s.slot: (s.slot_desc, "必填" if s.required else "可选") for s in spec.slots.values()}
            lines.append(f"- {spec.name}（{spec.desc}）槽位: {slots}")
        for h in history[-3:]:
            lines.append(f"历史：{h}")
        lines.append(f"用户输入：{text}")
        lines.append("只输出 JSON 数组，不要输出其他文字。")
        return "\n".join(lines)