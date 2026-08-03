"""配置管理：resources/intent_config.json + kv_items 动态覆盖。

ConfigManager 加载意图定义 / 业务词库 / 高危关键词 / 阈值；
可选注入 KvStore，支持从 kv_items 覆盖阈值与高危清单（后台可配置）。
键约定：intent:thresholds、intent:high_risk_keywords。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import IntentMeta, SlotMeta

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_DEFAULT_CONFIG_PATH = _RESOURCE_DIR / "intent_config.json"

_KV_PREFIX = "intent:"


class ConfigManager:
    """意图识别配置门面：意图/槽位/词库/高危/阈值 + kv 覆盖。"""

    def __init__(self, config_path: Path = _DEFAULT_CONFIG_PATH, kv=None):
        self.config_path = Path(config_path)
        self.kv = kv
        self.intents: Dict[str, IntentMeta] = {}
        self.business_vocab: List[str] = []
        self.high_risk_keywords: List[str] = []
        self.thresholds: Dict[str, Any] = {"confidence": 0.6, "high": 0.9}
        self.load()

    # ---------- 加载 ----------

    def load(self):
        """从资源 JSON 加载，再应用 kv_items 覆盖。"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.intents = {
            m.intent_id: m for m in self._build_intents(data.get("intents", []))
        }
        self.business_vocab = list(data.get("business_vocab", []))
        self.high_risk_keywords = list(data.get("high_risk_keywords", []))
        self.thresholds.update(data.get("thresholds", {}))
        self._apply_kv_overrides()

    def _build_intents(self, raw: List[dict]) -> List[IntentMeta]:
        metas: List[IntentMeta] = []
        for item in raw:
            slots = [
                SlotMeta(
                    slot_key=s.get("slot_key", ""),
                    slot_desc=s.get("slot_desc", ""),
                    required=bool(s.get("required", True)),
                    regex=s.get("regex", ""),
                )
                for s in item.get("slots", [])
            ]
            metas.append(
                IntentMeta(
                    intent_id=item["intent_id"],
                    name=item.get("name", item["intent_id"]),
                    desc=item.get("desc", ""),
                    keywords=list(item.get("keywords", [])),
                    required_slots=list(item.get("required_slots", [])),
                    optional_slots=list(item.get("optional_slots", [])),
                    slots=slots,
                )
            )
        return metas

    def _apply_kv_overrides(self):
        """从 kv_items 覆盖阈值与高危清单。"""
        if not self.kv:
            return
        th = self.kv.get(_KV_PREFIX + "thresholds")
        if isinstance(th, dict):
            self.thresholds.update(th)
        risk = self.kv.get(_KV_PREFIX + "high_risk_keywords")
        if isinstance(risk, list):
            self.high_risk_keywords = list(risk)

    # ---------- 查询 ----------

    def get_intent(self, intent_id: str) -> Optional[IntentMeta]:
        return self.intents.get(intent_id)

    def get_all_intents(self) -> List[IntentMeta]:
        return list(self.intents.values())

    def get_vocab(self) -> List[str]:
        return list(self.business_vocab)

    def get_risk_keywords(self) -> List[str]:
        return list(self.high_risk_keywords)

    @property
    def confidence_threshold(self) -> float:
        """统一低置信阈值（低于直接丢弃）。"""
        return float(self.thresholds.get("confidence", 0.6))

    @property
    def high_confidence_threshold(self) -> float:
        """高置信阈值（高于视为可执行）。"""
        return float(self.thresholds.get("high", 0.9))
