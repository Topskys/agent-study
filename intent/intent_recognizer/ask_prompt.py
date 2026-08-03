"""追问控制层：缺失 / 消歧字段聚合为一条话术一次问全。

对齐 v3 类图 AskPromptService。原则：缺失字段一次问全，禁止逐字段骚扰。
"""

from typing import List, Optional

from .config import ConfigManager
from .models import FirstStageResult, IntentRecognizeItem
from .protocols import AskUser


class AskPromptService:
    """统一追问话术构建 + 主动询问回调。"""

    def __init__(self, config: ConfigManager):
        self.config = config

    # ---------- 话术构建 ----------

    def format_missing(self, result: FirstStageResult) -> str:
        """把各未完备意图的缺失槽位聚合成一句话（带中文描述）。"""
        parts = []
        for item in result.intent_list:
            if item.complete:
                continue
            labels = []
            meta = self.config.get_intent(item.intent_id)
            for slot_key in item.miss_slots:
                slot = meta.get_slot(slot_key) if meta else None
                labels.append(f"{slot_key}（{slot.slot_desc}）" if slot else slot_key)
            parts.append(f"“{item.name}”需要补充：{'、'.join(labels)}")
        if not parts:
            return ""
        return "请提供以下信息：" + "；".join(parts)

    def build_prompt(self, result: FirstStageResult) -> str:
        """缺失槽位聚合追问。"""
        return self.format_missing(result)

    def build_confirm_prompt(self, intents: List[IntentRecognizeItem]) -> str:
        """0.6~0.9 消歧：确认是否执行候选意图。"""
        names = "、".join(f"“{i.name}”" for i in intents) or "未识别到明确意图"
        return f"我理解你可能想执行：{names}。请确认是否继续？(y/n)"

    def build_invalid_prompt(self, invalid_slots: List[dict]) -> str:
        """阶段二参数格式非法 → 聚合重新追问。"""
        parts = [
            f"{inv.get('slot_key')}（当前为：{inv.get('value')}）格式不正确"
            for inv in invalid_slots
        ]
        return "请重新提供以下信息：" + "；".join(parts)

    # ---------- 主动询问 ----------

    def ask(
        self, prompt: str, ask_user: Optional[AskUser], timeout: float = 30
    ) -> Optional[str]:
        """调用注入的 ask_user 询问；无回调 / 异常 / 空回复返回 None。"""
        if not prompt or not ask_user:
            return None
        try:
            return ask_user(prompt, timeout)
        except Exception:
            return None
