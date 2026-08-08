"""意图依赖解析：区分并行 / 串行，输出执行顺序分组。

启发式规则（纯自研，无外部依赖）：
- 文本含条件词（如果/假如/若/只要/当…）或先后词（先/然后/再…）且意图数 >1
  → 视为串行，按意图在文本中的出现次序分组建依赖；
- 否则全量并行（单组，group_id=0）。
意图顺序由上层按业务关键词在文本中的位置预排序后传入。
"""


from .models import IntentRecognizeItem, TaskGroup

_CONDITIONAL_MARKERS = ("如果", "假如", "若", "要是", "当", "就", "只要", "假设")
_SERIAL_MARKERS = ("然后", "接着", "再", "之后", "先", "接下来", "随后")


class IntentDependService:
    """依赖解析：输出有序 TaskGroup 列表（串行分组带前置依赖）。"""

    def parse(
        self, intents: list[IntentRecognizeItem], text: str = ""
    ) -> list[TaskGroup]:
        if not intents:
            return []
        if len(intents) == 1:
            return [TaskGroup(group_id=0, intents=list(intents))]

        has_serial = any(m in (text or "") for m in _CONDITIONAL_MARKERS) or any(
            m in (text or "") for m in _SERIAL_MARKERS
        )
        if not has_serial:
            # 并行无依赖：全部放入同一组
            return [TaskGroup(group_id=0, intents=list(intents))]

        # 串行：按文本出现次序排组，后序组依赖前序组
        ordered = sorted(intents, key=lambda it: self._first_pos(text, it))
        groups: list[TaskGroup] = []
        prev = None
        for idx, it in enumerate(ordered):
            dep = [prev] if prev is not None else []
            groups.append(TaskGroup(group_id=idx, dependency=dep, intents=[it]))
            prev = idx
        return groups

    @staticmethod
    def _first_pos(text: str, item: IntentRecognizeItem) -> int:
        """意图在文本中的最早出现位置（按名称或 intent_id 匹配）。"""
        text = text or ""
        candidates = [item.name, item.intent_id]
        pos = len(text)
        for c in candidates:
            if c and c in text:
                pos = min(pos, text.index(c))
        return pos
