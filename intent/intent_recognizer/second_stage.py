"""阶段二：批量槽位抽取服务。

已知目标意图集合，注入 llm_extract_slots 分组抽取各意图专属槽位，
再按 SlotMeta.regex 校验参数格式合法性；非法值交由追问层重新收集。
"""

import re
from typing import List, Optional

from .config import ConfigManager
from .llm_json import call_with_timeout, parse_slot_results
from .models import IntentRecognizeItem, SecondStageResult, SlotExtractResult
from .protocols import LLMExtractSlots

_MAX_PARSE_RETRIES = 2


class SecondStageSlotService:
    """阶段二：按意图批量抽取槽位 + 正则合法性校验。"""

    def __init__(
        self,
        config: ConfigManager,
        llm_extract_slots: Optional[LLMExtractSlots] = None,
        llm_timeout: float = 0,
    ):
        self.config = config
        self.llm_extract_slots = llm_extract_slots
        self.llm_timeout = llm_timeout

    # ---------- 主入口 ----------

    def extract(
        self,
        text: str,
        history: Optional[List[str]],
        intents: List[IntentRecognizeItem],
    ) -> SecondStageResult:
        if not intents or not self.llm_extract_slots:
            return SecondStageResult()
        history = history or []
        prompt = self._build_prompt(text, history, intents)
        intent_ids = [i.intent_id for i in intents]

        raw = call_with_timeout(
            lambda: self.llm_extract_slots(prompt, history, intent_ids),
            self.llm_timeout,
        )
        parsed: List[dict] = []
        for attempt in range(_MAX_PARSE_RETRIES + 1):
            parsed = parse_slot_results(raw or "")
            if parsed:
                break
            if attempt < _MAX_PARSE_RETRIES:
                raw = call_with_timeout(
                    lambda: self.llm_extract_slots(prompt, history, intent_ids),
                    self.llm_timeout,
                )

        results: List[SlotExtractResult] = []
        invalid: List[dict] = []
        for d in parsed:
            intent_id = str(d.get("intent_id", "")).strip()
            slot_info = d.get("slot_info")
            if not intent_id or not isinstance(slot_info, dict):
                continue
            meta = self.config.get_intent(intent_id)
            if not meta:
                continue
            valid_kv = {}
            for k, v in slot_info.items():
                slot = meta.get_slot(k)
                if slot is None:
                    continue  # 未知槽位丢弃
                if not self.validate(str(v), slot.regex):
                    invalid.append({"intent_id": intent_id, "slot_key": k, "value": v})
                    continue
                valid_kv[k] = v
            results.append(SlotExtractResult(intent_id=intent_id, slot_kv=valid_kv))
        return SecondStageResult(slot_results=results, invalid_slots=invalid)

    # ---------- 参数校验 ----------

    def validate(self, value: str, regex: str) -> bool:
        """正则全匹配校验；无正则视为合法；正则异常时宽松放行。"""
        if not regex:
            return True
        try:
            return re.fullmatch(regex, value) is not None
        except re.error:
            return True

    # ---------- Prompt ----------

    def _build_prompt(
        self, text: str, history: List[str], intents: List[IntentRecognizeItem]
    ) -> str:
        lines = ["你是槽位抽取器。根据用户输入与历史上下文，为给定意图抽取槽位。"]
        lines.append(
            '输出格式：JSON 数组，每个元素 {"intent_id": "...", "slot_info": {"槽位键": "值"}}。'
        )
        lines.append("槽位定义：")
        for it in intents:
            meta = self.config.get_intent(it.intent_id)
            if not meta:
                continue
            descs = []
            for s in meta.slots:
                req = "必填" if s.required else "可选"
                regex_note = f"，格式: {s.regex}" if s.regex else ""
                descs.append(f"{s.slot_key}（{s.slot_desc}，{req}{regex_note}）")
            lines.append(f"- {meta.intent_id}（{meta.name}）：{'；'.join(descs)}")
        for h in history[-3:]:
            lines.append(f"历史：{h}")
        lines.append(f"用户输入：{text}")
        lines.append("缺失的值留空字符串即可。只输出 JSON 数组，不要输出其他文字。")
        return "\n".join(lines)
