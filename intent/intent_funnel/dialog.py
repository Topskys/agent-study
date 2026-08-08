"""V3 §八/§十：会话状态、槽位缓存、完备性检查、聚合追问、意图依赖。

- SlotCompletenessChecker：按必填槽位校验 + 回填缓存，输出缺失槽位 askSlots
- AskPromptBuilder：缺失 / 消歧字段聚合为一条话术一次问全
- DialogStateTracker：会话快照读写（含槽位缓存与检查点），支持可选持久化
- IntentOrchestrator：多意图并行/串行编排（V3 §十一 识别后适配层）
"""

from typing import Any

from .config import FunnelConfig
from .models import DialogSessionState, IntentItem, IntentRankingItem, IntentResult


class SlotCompletenessChecker:
    """槽位完备性检查：额定必填槽位 vs 已填（结果内 + 会话缓存）。"""

    def __init__(self, config: FunnelConfig):
        self.config = config

    def check(
        self, intents: list[IntentItem], session: DialogSessionState
    ) -> list[str]:
        """补齐缺失槽位；返回缺失槽位列表（去重，一次问全）。"""
        miss: list[str] = []
        cached = session.filled_slots
        for item in intents:
            spec = self.config.get_intent(item.name)
            if spec is None:
                item.complete = True
                item.miss_slots = []
                continue
            filled = {e.intent for e in item.entities}
            required = list(spec.required_slots)
            fills = [k for k in required if k in filled]
            cached_fills = [k for k in required if k in cached.get(item.name, {})]
            missing = [k for k in required if k not in fills and k not in cached_fills]
            item.complete = not missing
            item.miss_slots = list(missing)
            miss.extend(missing)
        # 去重保序
        dedup: list[str] = []
        for s in miss:
            if s not in dedup:
                dedup.append(s)
        return dedup


class AskPromptBuilder:
    """追问话术构建：缺失字段一次问全 / 消歧确认 / 重输。"""

    def __init__(self, config: FunnelConfig):
        self.config = config

    def build_slots(self, ask_slots: list[str]) -> str:
        parts = []
        for slot_key in ask_slots:
            desc = self._slot_label(slot_key)
            parts.append(f"{slot_key}（{desc}）" if desc else str(slot_key))
        return f"请补充以下信息：{'、'.join(parts)}"

    def build_disambiguate(self, ranking: list[IntentRankingItem]) -> str:
        names = "、".join(f"“{r.name}”" for r in ranking) or "未识别到明确意图"
        return f"我理解你可能想执行：{names}。请确认是否继续？(y/n)"

    def build_reenter(self) -> str:
        return "抱歉，我没有识别到明确意图，请换一种说法重新输入。"

    def build_invalid(self, invalid: list[dict[str, Any]]) -> str:
        parts = [
            f"{it.get('slot')}（当前为：{it.get('value')}）格式不正确"
            for it in invalid
        ]
        return "请重新提供以下信息：" + "；".join(parts)

    def _slot_label(self, slot_key: str) -> str:
        for spec in self.config.all_intents():
            slot = spec.get_slot(slot_key)
            if slot:
                return slot.slot_desc
        return ""


class DialogStateStore:
    """抽象存储（可替换为 sqlite / redis）。内存实现。"""

    def __init__(self):
        self._data: dict[tuple, DialogSessionState] = {}

    def get(self, user_id: str, session_id: str) -> DialogSessionState:
        key = (user_id, session_id)
        state = self._data.get(key)
        if state is None:
            state = DialogSessionState(user_id=user_id, session_id=session_id)
            self._data[key] = state
        return state

    def set(self, state: DialogSessionState) -> None:
        self._data[(state.user_id, state.session_id)] = state


class DialogStateTracker:
    """会话状态追踪：槽位缓存回填、检查点、历史。"""

    def __init__(self, store: DialogStateStore | None = None):
        self.store = store or DialogStateStore()

    def get_session(self, user_id: str, session_id: str) -> DialogSessionState:
        return self.store.get(user_id, session_id)

    def load_cached_slots(self, session: DialogSessionState) -> dict[str, dict[str, Any]]:
        """导出已确认槽位供上层回填。"""
        return {intent: dict(slots) for intent, slots in session.filled_slots.items()}

    def update_after_result(self, session: DialogSessionState, result: IntentResult) -> None:
        """识别结果落盘：更新槽位缓存/最后意图/历史。"""
        for item in result.intents:
            cached = session.filled_slots.setdefault(item.name, {})
            for entity in item.entities:
                if entity.valid:
                    cached[entity.intent] = entity.value
        session.last_intent_text = result.text
        session.checkpoint = result
        self.store.set(session)


class IntentOrchestrator:
    """多意图串行/并行编排（V4 §十一 适配层）。

    - 无显式依赖标记 → 并行：全部放入同一组
    - 出现条件/顺序标记 → 按意图在文本出现次序排串行依赖
    """

    CONDITIONAL_MARKERS = ("如果", "假如", "若", "要是", "先", "再", "然后", "之后")
    DEPENDENCY_MARKERS = CONDITIONAL_MARKERS

    def group(self, intents: list[IntentItem], text: str = "") -> list[dict[str, Any]]:
        """返回依赖分组 [{group: [intents], depends: int}]。"""
        if not intents:
            return []
        if len(intents) == 1:
            return [{"group": intents, "depends": []}]
        has_order = any(m in (text or "") for m in self.DEPENDENCY_MARKERS)
        if not has_order:
            return [{"group": list(intents), "depends": []}]
        ordered = sorted(intents, key=lambda it: self._first_pos(text, it))
        groups: list[dict[str, Any]] = []
        prev = -1
        for idx, item in enumerate(ordered):
            groups.append({"group": [item], "depends": [prev] if prev >= 0 else []})
            prev = idx
        return groups

    def _first_pos(self, text: str, item: IntentItem) -> int:
        text = text or ""
        for c in (item.name,):
            if c in text:
                return text.index(c)
        return len(text)