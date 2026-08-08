"""V3 三层漏斗架构统一数据模型。

对齐 docs/Agent意图识别设计方案v3.md §四：
- UserInput 入参
- IntentRecognitionResult 三层同构统一输出（snake_case JSON，含意图/槽位/消歧/追问）
- IntentItem / EntityItem / IntentRankingItem
- DialogSessionState 会话状态（槽位缓存 / 检查点）
三层输出完全同构，调度器无需感知下层实现。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TeacherInput:
    """Tap 输入模型（UserRequest）。"""

    content: str
    user_id: str = "default_user"
    session_id: str = "default_session"
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityItem:
    """槽位结构体：含规范化值（如 北京 → PEK）。"""

    intent: str
    value: Any | None = None
    normalized_value: str = ""
    raw_text: str = ""
    confidence: float = 1.0
    valid: bool = True

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "raw_text": self.raw_text,
            "confidence": round(self.confidence, 4),
            "valid": self.valid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntityItem":
        return cls(
            intent=str(d.get("intent", "")),
            value=d.get("value"),
            normalized_value=str(d.get("normalized_value", "") or ""),
            raw_text=str(d.get("raw_text", "") or ""),
            confidence=float(d.get("confidence", 1.0)),
            valid=bool(d.get("valid", True)),
        )


@dataclass
class IntentItem:
    """单意图单元。"""

    name: str
    confidence: float = 0.0
    priority: int = 1
    entities: list[EntityItem] = field(default_factory=list)
    complete: bool = False
    miss_slots: list[str] = field(default_factory=list)

    @property
    def slot_map(self) -> dict[str, Any]:
        return {e.intent: e.value for e in self.entities}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "priority": self.priority,
            "entities": [e.to_dict() for e in self.entities],
            "complete": self.complete,
            "miss_slots": list(self.miss_slots),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntentItem":
        return cls(
            name=str(d.get("name", "")),
            confidence=float(d.get("confidence", 0.0)),
            priority=int(d.get("priority", 1)),
            entities=[EntityItem.from_dict(e) for e in d.get("entities", [])],
            complete=bool(d.get("complete", False)),
            miss_slots=[str(s) for s in d.get("miss_slots", [])],
        )


@dataclass
class IntentRankingItem:
    """候选意图排序项（供消歧 / 日志 / 评估）。"""

    name: str
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "confidence": round(self.confidence, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> "IntentRankingItem":
        return cls(
            name=str(d.get("name", "")),
            confidence=float(d.get("confidence", 0.0)),
        )


@dataclass
class IntentResult:
    """三层统一输出（V4 §4.2 JSON 定稿 + 扩展字段）。

    扩展字段（可选）：
    - need_ask_slots / ask_slots / ask_prompt：槽位缺失聚合追问
    - blocked / block_reason：高危拦截
    """

    source_layer: str = "none"  # rule_matcher / semantic_reasoner / complex_parser
    text: str = ""
    need_disambiguate: bool = False
    no_valid_intent: bool = False
    intents: list[IntentItem] = field(default_factory=list)
    intent_ranking: list[IntentRankingItem] = field(default_factory=list)
    session_id: str = ""
    request_id: str = ""
    need_ask_slots: bool = False
    ask_slots: list[str] = field(default_factory=list)
    ask_prompt: str = ""
    blocked: bool = False
    block_reason: str = ""

    @property
    def high_confidence(self) -> float:
        return max((i.confidence for i in self.intents), default=0.0)

    @property
    def primary_intent(self) -> str | None:
        return self.intents[0].name if self.intents else None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "from": "bot",
            "source_layer": self.source_layer,
            "need_disambiguate": self.need_disambiguate,
            "no_valid_intent": self.no_valid_intent,
            "intents": [i.to_dict() for i in self.intents],
            "intent_ranking": [r.to_dict() for r in self.intent_ranking],
            "session_id": self.session_id,
            "request_id": self.request_id,
            "need_ask_slots": self.need_ask_slots,
            "ask_slots": list(self.ask_slots),
            "ask_prompt": self.ask_prompt,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "IntentResult":
        return cls(
            source_layer=str(d.get("source_layer", "none")),
            text=str(d.get("text", "")),
            need_disambiguate=bool(d.get("need_disambiguate", False)),
            no_valid_intent=bool(d.get("no_valid_intent", False)),
            intents=[IntentItem.from_dict(i) for i in d.get("intents", [])],
            intent_ranking=[
                IntentRankingItem.from_dict(r) for r in d.get("intent_ranking", [])
            ],
            session_id=str(d.get("session_id", "")),
            request_id=str(d.get("request_id", "")),
            need_ask_slots=bool(d.get("need_ask_slots", False)),
            ask_slots=[str(s) for s in d.get("ask_slots", [])],
            ask_prompt=str(d.get("ask_prompt", "")),
            blocked=bool(d.get("blocked", False)),
            block_reason=str(d.get("block_reason", "")),
        )


@dataclass
class DialogSessionState:
    """会话快照（DialogSessionState）：槽位缓存 / 检查点 / 历史。"""

    session_id: str
    user_id: str
    last_intent_text: str = ""
    filled_slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    miss_slots: list[str] = field(default_factory=list)
    historical_queries: list[str] = field(default_factory=list)
    checkpoint: IntentResult | None = None