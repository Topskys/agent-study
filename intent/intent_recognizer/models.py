"""v3 两阶段多意图识别核心数据模型。

对齐 docs/Agent意图识别设计方案v2.md §5 类图与 §6 两阶段数据结构。
所有结果模型提供 to_dict / from_dict，便于 intent_cache / slot_cache 持久化。
"""

from dataclasses import asdict, dataclass, field
from typing import Any


class IntentNames:
    """系统级意图常量（规则层硬信号输出）。"""

    CHAT = "chat"
    QUESTION = "question"
    TOOL_USE = "tool_use"
    MEMORY_WRITE = "memory_write"
    MEMORY_QUERY = "memory_query"
    COMMAND = "command"


@dataclass
class SlotMeta:
    """槽位定义：键 / 描述 / 是否必填 / 合法性正则。"""

    slot_key: str
    slot_desc: str
    required: bool = True
    regex: str = ""


@dataclass
class IntentMeta:
    """意图定义：ID / 名称 / 描述 / 关键词 / 必填与可选槽位。"""

    intent_id: str
    name: str
    desc: str = ""
    keywords: list[str] = field(default_factory=list)
    required_slots: list[str] = field(default_factory=list)
    optional_slots: list[str] = field(default_factory=list)
    slots: list[SlotMeta] = field(default_factory=list)

    @property
    def all_slot_keys(self) -> list[str]:
        """全部槽位键（必填 + 可选，保序去重）。"""
        keys = list(self.required_slots)
        for k in self.optional_slots:
            if k not in keys:
                keys.append(k)
        return keys

    def get_slot(self, slot_key: str) -> SlotMeta | None:
        """按槽位键查槽位定义，未定义返回 None。"""
        for s in self.slots:
            if s.slot_key == slot_key:
                return s
        return None


@dataclass
class IntentRecognizeItem:
    """阶段一输出的单条意图识别结果。"""

    intent_id: str
    name: str = ""
    confidence: float = 0.0
    complete: bool = False
    miss_slots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntentRecognizeItem":
        return cls(**data)


@dataclass
class FirstStageResult:
    """阶段一输出：多意图数组 + 完备性 + 缺失槽位汇总。"""

    intent_list: list[IntentRecognizeItem] = field(default_factory=list)
    all_complete: bool = False
    total_miss_slots: list[str] = field(default_factory=list)
    source: str = "none"

    def to_dict(self) -> dict:
        return {
            "intent_list": [i.to_dict() for i in self.intent_list],
            "all_complete": self.all_complete,
            "total_miss_slots": self.total_miss_slots,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FirstStageResult":
        return cls(
            intent_list=[
                IntentRecognizeItem.from_dict(i) for i in data.get("intent_list", [])
            ],
            all_complete=bool(data.get("all_complete", False)),
            total_miss_slots=list(data.get("total_miss_slots", [])),
            source=str(data.get("source", "none")),
        )


@dataclass
class SlotExtractResult:
    """阶段二输出的单意图槽位抽取结果。"""

    intent_id: str
    slot_kv: dict[str, Any] = field(default_factory=dict)


@dataclass
class SecondStageResult:
    """阶段二输出：槽位集合 + 非法参数清单。"""

    slot_results: list[SlotExtractResult] = field(default_factory=list)
    invalid_slots: list[dict] = field(default_factory=list)


@dataclass
class TaskGroup:
    """任务执行分组：group_id / 前置依赖 / 组内意图。"""

    group_id: int
    dependency: list[int] = field(default_factory=list)
    intents: list[IntentRecognizeItem] = field(default_factory=list)

    @property
    def intent_ids(self) -> list[str]:
        return [i.intent_id for i in self.intents]


@dataclass
class ExecutionPlan:
    """一次完整意图识别（两阶段）的执行计划。

    fields 对齐类图 §5；另扩展 execution_results 承载 TaskScheduleService
    调度输出，使识别与执行结果在同一计划内可观测。
    """

    intents: list[IntentRecognizeItem] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    task_groups: list[TaskGroup] = field(default_factory=list)
    risk_level: str = "low"
    blocked: bool = False
    ambiguous: bool = False
    source: str = "none"
    original: str = ""
    processed: str = ""
    ask_prompt: str | None = None
    execution_results: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_intent(self) -> str | None:
        """主意图：第一个意图的 intent_id（无则 None）。"""
        return self.intents[0].intent_id if self.intents else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["primary_intent"] = self.primary_intent
        return d


@dataclass
class RuleHit:
    """规则层输出：硬信号命中 / 高危拦截。"""

    intent_id: str | None = None
    confidence: float = 0.0
    slots: dict[str, Any] = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    blocked: bool = False
    block_reason: str | None = None
