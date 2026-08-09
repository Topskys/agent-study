"""配置管理：resources/config.json 加载。

V3 方案：
- intents：意图定义（名称/描述/优先级/必填可选槽位/关键词/别名）
- thresholds：semantic_high（语义层 0.9）/ llm_high（0.9）/ llm_low（0.6）
- high_risk_keywords：高危词，命中直接拦截（不进语义与 LLM）
- tool_schemas：FunctionCall 工具定义（供 ComplexIntentParser 组装 Prompt）
"""

import json
from pathlib import Path
from typing import Any

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCE_DIR / "config.json"


class SlotSpec:
    def __init__(self, raw: dict):
        self.slot = raw["slot"]
        self.slot_desc = raw.get("slot_desc", "")
        self.required = bool(raw.get("required", True))
        self.regex = raw.get("regex", "")
        self.aliases: dict[str, str] = dict(raw.get("aliases", {}))

    def normalize(self, value: Any) -> str:
        return self.aliases.get(str(value), str(value))


class IntentSpec:
    def __init__(self, raw: dict):
        self.name: str = raw["name"]
        self.desc: str = raw.get("desc", "")
        self.priority: int = int(raw.get("priority", 1))
        self.keywords: list[str] = list(raw.get("keywords", []))
        self.required_slots: list[str] = list(raw.get("required_slots", []))
        self.optional_slots: list[str] = list(raw.get("optional_slots", []))
        self.slots: dict[str, SlotSpec] = {
            s.slot: s for s in (SlotSpec(x) for x in raw.get("slots", []))
        }

    def all_slot_keys(self) -> list[str]:
        keys = list(self.required_slots)
        for k in self.optional_slots:
            if k not in keys:
                keys.append(k)
        return keys

    def get_slot(self, slot_key: str) -> SlotSpec | None:
        return self.slots.get(slot_key)


class FunnelConfig:
    """漏斗架构配置门面。"""

    def __init__(self, config_path: Path = _DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.intents: dict[str, IntentSpec] = {}
        self.thresholds: dict[str, float] = {}
        self.high_risk_keywords: list[str] = []
        self.tool_schemas: list[dict] = []
        self.load()

    def load(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.intents = {
            m.name: m for m in (IntentSpec(x) for x in data.get("intents", []))
        }
        self.thresholds = {k: float(v) for k, v in data.get("thresholds", {}).items()}
        self.high_risk_keywords = list(data.get("high_risk_keywords", []))
        self.tool_schemas = list(data.get("tool_schemas", []))

    # ---------- 查询 ----------

    def get_intent(self, name: str) -> IntentSpec | None:
        return self.intents.get(name)

    def all_intents(self) -> list[IntentSpec]:
        return list(self.intents.values())

    def get_risk_keywords(self) -> list[str]:
        return list(self.high_risk_keywords)

    def get_tool_schemas(self) -> list[dict]:
        return list(self.tool_schemas)

    @property
    def semantic_high(self) -> float:
        """语义层阈值：≥0.9 可信直接返回，否则下放 LLM。"""
        return self.thresholds.get("semantic_high", 0.9)

    @property
    def llm_high(self) -> float:
        """LLM 层高置信阈值。"""
        return self.thresholds.get("llm_high", 0.9)

    @property
    def llm_low(self) -> float:
        """LLM 层低置信阈值：<0.6 无有效意图。"""
        return self.thresholds.get("llm_low", 0.6)
