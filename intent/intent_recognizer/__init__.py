"""intent-recognizer 包（v3 两阶段多意图识别）。

架构见 design/Agent意图识别设计方案v3.md：
- 两阶段 LLM：阶段一多意图识别（置信度/完备性）→ 阶段二批量槽位抽取；
- LLM 能力全部注入式回调（llm_recognize / llm_extract_slots / ask_user / llm_expand），
  本包不 import 任何 LLM SDK；
- 数据访问层直连 agent_memory.db 现有记忆模块数据表（不新建会话表）；
- 不绑定 LangGraph / LangChain，纯自研编排。
"""

from .config import ConfigManager
from .models import (
    ExecutionPlan,
    FirstStageResult,
    IntentMeta,
    IntentNames,
    IntentRecognizeItem,
    RuleHit,
    SecondStageResult,
    SlotExtractResult,
    SlotMeta,
    TaskGroup,
)
from .preprocess import Preprocessor, TextPreprocessService
from .recognizer import IntentRecognizer
from .stores import EventStore, KvStore, MemoryStore, ProfileStore

__all__ = [
    "ConfigManager",
    "EventStore",
    "ExecutionPlan",
    "FirstStageResult",
    "IntentMeta",
    "IntentNames",
    "IntentRecognizeItem",
    "IntentRecognizer",
    "KvStore",
    "MemoryStore",
    "Preprocessor",
    "ProfileStore",
    "RuleHit",
    "SecondStageResult",
    "SlotExtractResult",
    "SlotMeta",
    "TaskGroup",
    "TextPreprocessService",
]
