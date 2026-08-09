"""agent_mvp 意图漏斗回归测试：验证 resources/intent_config.json 驱动的识别行为。"""

import os
import tempfile

from pathlib import Path

from intent_funnel import FunnelConfig, FunnelIntentRecognition, SessionStore

_CFG = str(Path(__file__).resolve().parent.parent / "resources" / "intent_config.json")


def _recognizer():
    db = os.path.join(tempfile.gettempdir(), "funnel_regression.db")
    if os.path.exists(db):
        os.remove(db)
    return FunnelIntentRecognition(
        config=FunnelConfig(config_path=_CFG),
        store=SessionStore(db_path=db),
        llm_gen=lambda s: "{}",
    )


def test_weather_queries_detect_tool_weather_by_rule():
    r = _recognizer()
    for q in ["当前城市天气", "根据ip查询当前城市天气", "今天天气怎么样", "北京多少度"]:
        m = r.recognize(q, [], "u", "s")
        assert m.primary_intent == "tool_weather", f"{q} -> {m.primary_intent}"
        assert m.source_layer == "rule_matcher"
        assert not m.need_disambiguate
        assert not m.need_ask_slots


def test_memory_write_and_query_still_rule_detected():
    r = _recognizer()
    m = r.recognize("记住我家住在杭州", [], "u", "s")
    assert m.primary_intent == "memory_write"
    assert m.source_layer == "rule_matcher"
    m2 = r.recognize("我之前说过什么，你提醒过我什么", [], "u", "s")
    assert m2.primary_intent == "memory_query"


def test_high_risk_blocked():
    r = _recognizer()
    m = r.recognize("帮我删除这个文件", [], "u", "s")
    assert m.blocked