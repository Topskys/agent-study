"""intent-funnel 包（V3 三层漏斗意图识别）。

架构见 docs/Agent意图识别设计方案v3.md：
- 三层漏斗：RuleMatcher（规则）→ SemanticReasoner（语义）→ ComplexIntentParser（LLM），
  三层输出完全同构（source_layer + intents + intent_ranking）；
- 四分支交互兜底：直接执行 / 缺失槽位聚合追问 / 消歧反问 / 无意图重输；
- 会话状态：槽位缓存 + 检查点 + 历史（可 sqlite 持久化）；
- LLM/语义能力全部注入式回调，本包不 import 任何 LLM SDK。
"""

from .config import FunnelConfig, IntentSpec, SlotSpec
from .funnel import FunnelIntentRecognition
from .models import (
    DialogSessionState,
    EntityItem,
    IntentItem,
    IntentRankingItem,
    IntentResult,
    TeacherInput,
)
from .stores import SessionStore

__all__ = [
    "DialogSessionState",
    "EntityItem",
    "FunnelConfig",
    "FunnelIntentRecognition",
    "IntentItem",
    "IntentRankingItem",
    "IntentResult",
    "IntentSpec",
    "SessionStore",
    "SlotSpec",
    "TeacherInput",
]