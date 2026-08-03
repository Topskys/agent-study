"""intent-recognizer 包单测（v3 两阶段多意图识别）。

覆盖（对应 design/Agent意图识别设计方案v3.md）：
- 预处理三子模块：错别字 / 代词消解 / 短提问扩写
- 规则层确定性：高危拦截 / memory_write / memory_query / chat / question / tool_use
- 阶段一：LLM 多意图解析、置信度过滤、完备性校验、规则关键词兜底
- 阶段二：批量槽位抽取 + 正则合法性校验
- 置信度三档分发：>0.9 执行 / 0.6~0.9 消歧反问 / <0.6 重新输入
- 聚合追问（缺失字段一次问全）、串行/并行依赖与调度
- 数据访问层：intent_cache / slot_cache / 事件审计 / kv 覆盖（tmp db）
"""

import os
from pathlib import Path

os.environ.setdefault("MEMORY_EMBED_MODE", "mock")

import pytest

from intent_recognizer import (
    ConfigManager,
    EventStore,
    ExecutionPlan,
    FirstStageResult,
    IntentNames,
    IntentRecognizer,
    IntentRecognizeItem,
    MemoryStore,
)
from intent_recognizer.ask_prompt import AskPromptService
from intent_recognizer.config import ConfigManager as _Cfg
from intent_recognizer.depend import IntentDependService
from intent_recognizer.first_stage import FirstStageIntentService
from intent_recognizer.models import TaskGroup
from intent_recognizer.preprocess import TextPreprocessService
from intent_recognizer.rule import RuleCheckService
from intent_recognizer.scheduler import TaskScheduleService
from intent_recognizer.second_stage import SecondStageSlotService


# ---------- 工具函数 ----------


def make_recognizer(**kwargs) -> IntentRecognizer:
    return IntentRecognizer(**kwargs)


def make_first_service(llm=None, **kw) -> FirstStageIntentService:
    cfg = kw.pop("config", _Cfg())
    return FirstStageIntentService(config=cfg, llm_recognize=llm, **kw)


def make_second_service(llm=None, **kw) -> SecondStageSlotService:
    cfg = kw.pop("config", _Cfg())
    return SecondStageSlotService(config=cfg, llm_extract_slots=llm, **kw)


def item(intent_id, conf=0.96, complete=True, miss=None) -> IntentRecognizeItem:
    return IntentRecognizeItem(
        intent_id=intent_id,
        name=intent_id,
        confidence=conf,
        complete=complete,
        miss_slots=list(miss or []),
    )


def make_db(tmp_path: Path) -> str:
    """返回一个带同构表的空 SQLite 库路径。"""
    db = tmp_path / "intent.db"
    MemoryStore(str(db))._init_db()
    EventStore(str(db))._init_db()
    return str(db)


# ======================================================================
# 预处理：错别字 / 代词 / 扩写
# ======================================================================


def test_typo_correction():
    p = TextPreprocessService()
    assert p.correct_typos("周抱") == "周报"


def test_typo_correction_with_history():
    rec = make_recognizer()
    text, amb = rec.preprocess.process("周抱", history=["上周的周报写得不错"])
    assert text == "周报"
    assert amb is False


def test_typo_no_change_on_correct_word():
    p = TextPreprocessService()
    assert p.correct_typos("合同") == "合同"


def test_pronoun_resolution():
    rec = make_recognizer()
    text, amb = rec.preprocess.process("它有没有风险", history=["昨天那份合同"])
    assert amb is False
    assert text == "昨天那份合同有没有风险"


def test_pronoun_without_history_is_ambiguous():
    rec = make_recognizer()
    text, amb = rec.preprocess.process("它有没有风险", history=[])
    assert amb is True


def test_demonstrative_with_noun_not_duplicated():
    rec = make_recognizer()
    text, _ = rec.preprocess.process("这个合同怎么样", history=["合同"])
    assert "合同合同" not in text


def test_short_query_ambiguous_without_callback():
    rec = make_recognizer()
    _, amb = rec.preprocess.process("处理一下合同", history=[])
    assert amb is True


def test_short_query_expanded_with_callback():
    def fake_expand(text, history):
        return "帮我查一下昨天的合同"

    p = TextPreprocessService(llm_expand=fake_expand)
    text, amb = p.process("处理一下合同", history=[])
    assert amb is False
    assert "合同" in text


def test_complete_short_command_not_ambiguous():
    rec = make_recognizer()
    _, amb = rec.preprocess.process("查周报", history=[])
    assert amb is False


# ======================================================================
# 规则层确定性
# ======================================================================


def test_high_risk_blocked():
    plan = make_recognizer().recognize("把客户资料全删掉")
    assert plan.blocked is True
    assert plan.risk_level == "high"
    assert plan.source == "rule"
    assert plan.ask_prompt and "安全拦截" in plan.ask_prompt


def test_persist_imperative_is_memory_write():
    plan = make_recognizer().recognize("请记住我家在杭州")
    assert plan.primary_intent == IntentNames.MEMORY_WRITE
    assert plan.source == "rule"
    assert plan.risk_level == "low"
    assert not plan.ambiguous
    assert plan.slots.get("content") == "请记住我家在杭州"


def test_query_marker_mixed_regression():
    plan = make_recognizer().recognize("你记住了吗？把从数据库查给我看")
    assert plan.primary_intent == IntentNames.MEMORY_QUERY


def test_remember_question_is_memory_query():
    plan = make_recognizer().recognize("还记得我上次说的吗")
    assert plan.primary_intent == IntentNames.MEMORY_QUERY


def test_greeting_is_chat():
    plan = make_recognizer().recognize("你好")
    assert plan.primary_intent == IntentNames.CHAT


def test_question_words_is_question():
    plan = make_recognizer().recognize("什么是RAG？")
    assert plan.primary_intent == IntentNames.QUESTION


def test_calc_hint_is_tool_use():
    plan = make_recognizer().recognize("帮我算 3*(2+5)")
    assert plan.primary_intent == IntentNames.TOOL_USE


def test_time_hint_not_question():
    plan = make_recognizer().recognize("现在几点了")
    assert plan.primary_intent == IntentNames.TOOL_USE


def test_high_risk_keywords_configurable():
    rec = make_recognizer(high_risk_keywords=["摧毁"])
    assert rec.recognize("把东西摧毁").blocked is True
    assert rec.recognize("把客户资料全删掉").blocked is False  # 默认词库被覆盖


# ======================================================================
# 阶段一：多意图解析
# ======================================================================


def test_first_stage_multi_intent_parse():
    def fake_recognize(prompt, history):
        return (
            '[{"intent_id": "bill_query", "name": "账单查询", "confidence": 0.96, '
            '"complete": true, "miss_slots": []}, '
            '{"intent_id": "phone_recharge", "name": "手机话费充值", "confidence": 0.94, '
            '"complete": false, "miss_slots": ["phone_number", "recharge_amount"]}]'
        )

    svc = make_first_service(fake_recognize)
    result = svc.recognize("查7月账单，充50话费")
    assert result.source == "llm"
    assert [i.intent_id for i in result.intent_list] == ["bill_query", "phone_recharge"]
    assert result.all_complete is False
    assert set(result.total_miss_slots) == {"phone_number", "recharge_amount"}


def test_first_stage_drops_low_confidence_and_unknown():
    def fake_recognize(prompt, history):
        return (
            '[{"intent_id": "bill_query", "confidence": 0.96, "complete": true}, '
            '{"intent_id": "hack", "confidence": 0.99, "complete": true}, '
            '{"intent_id": "phone_recharge", "confidence": 0.3, "complete": false}]'
        )

    svc = make_first_service(fake_recognize)
    result = svc.recognize("随便")
    assert [i.intent_id for i in result.intent_list] == ["bill_query"]


def test_first_stage_garbage_falls_back_to_rule_keywords():
    def fake_recognize(prompt, history):
        return "这段不是JSON，还重复了"

    svc = make_first_service(fake_recognize)
    result = svc.recognize("帮我充50话费")
    assert result.source == "rule"
    assert result.intent_list[0].intent_id == "phone_recharge"


def test_first_stage_without_llm_uses_rule_keywords():
    svc = make_first_service(None)
    result = svc.recognize("查一下这个月的账单")
    assert result.source == "rule"
    assert result.intent_list[0].intent_id == "bill_query"


# ======================================================================
# 阶段二：批量槽位抽取 + 正则校验
# ======================================================================


def test_second_stage_extract_valid_slots():
    def fake_extract(prompt, history, intent_ids):
        return (
            '[{"intent_id": "phone_recharge", "slot_info": '
            '{"phone_number": "13800138000", "recharge_amount": "50"}}]'
        )

    svc = make_second_service(fake_extract)
    result = svc.extract("充50", [], [item("phone_recharge")])
    assert result.slot_results[0].slot_kv == {
        "phone_number": "13800138000",
        "recharge_amount": "50",
    }
    assert result.invalid_slots == []


def test_second_stage_invalid_slots_by_regex():
    def fake_extract(prompt, history, intent_ids):
        return (
            '[{"intent_id": "phone_recharge", "slot_info": '
            '{"phone_number": "12345", "recharge_amount": "50"}}]'
        )

    svc = make_second_service(fake_extract)
    result = svc.extract("充50", [], [item("phone_recharge")])
    assert len(result.invalid_slots) == 1
    assert result.invalid_slots[0]["slot_key"] == "phone_number"
    # 非法槽位不进 valid_kv
    assert result.slot_results[0].slot_kv == {"recharge_amount": "50"}


def test_second_stage_without_llm_returns_empty():
    svc = make_second_service(None)
    result = svc.extract("充50", [], [item("phone_recharge")])
    assert result.slot_results == []
    assert result.invalid_slots == []


def test_validate_regex():
    svc = make_second_service(None)
    assert svc.validate("13800138000", "^1[0-9]{10}$") is True
    assert svc.validate("12345", "^1[0-9]{10}$") is False
    assert svc.validate("随便", "") is True


# ======================================================================
# 置信度三档分发（recognize 门面）
# ======================================================================


def _confident_llm(intent_id, conf, complete=True, miss=None):
    def fake(prompt, history):
        return (
            f'[{{"intent_id": "{intent_id}", "confidence": {conf}, '
            f'"complete": {"true" if complete else "false"}, '
            f'"miss_slots": {miss or "[]"}}}]'
        )

    return fake


def test_full_flow_high_confidence_executes():
    executed = []

    def fake_extract(prompt, history, intent_ids):
        return (
            '[{"intent_id": "bill_query", "slot_info": {"start_time": "2026-07-01"}}]'
        )

    def fake_executor(intent_id, slot_kv):
        executed.append(intent_id)
        return f"{intent_id}:done"

    rec = make_recognizer(
        llm_recognize=_confident_llm("bill_query", 0.96),
        llm_extract_slots=fake_extract,
        executor=fake_executor,
    )
    plan = rec.recognize("查一下7月账单")
    assert isinstance(plan, ExecutionPlan)
    assert plan.primary_intent == "bill_query"
    assert plan.source == "llm"
    assert not plan.ambiguous
    assert not plan.blocked
    assert plan.slots["bill_query"]["start_time"] == "2026-07-01"
    assert plan.execution_results["bill_query"] == "bill_query:done"
    assert executed == ["bill_query"]


def test_fuzzy_confidence_asks_disambiguation():
    rec = make_recognizer(llm_recognize=_confident_llm("bill_query", 0.7))
    plan = rec.recognize("查7月账单")
    assert plan.ambiguous is True
    assert plan.ask_prompt and "确认" in plan.ask_prompt
    assert plan.execution_results == {}


def test_low_confidence_asks_reentry():
    """<0.6 意图被阶段一阈值过滤，且规则兜底也无命中 → 请用户重新输入。"""
    rec = make_recognizer(llm_recognize=_confident_llm("bill_query", 0.4))
    plan = rec.recognize("搞点新花样吧")
    assert plan.ambiguous is True
    assert "重新输入" in plan.ask_prompt
    assert plan.primary_intent is None


def test_incomplete_aggregated_ask():
    rec = make_recognizer(
        llm_recognize=_confident_llm(
            "phone_recharge",
            0.96,
            complete=False,
            miss=["phone_number", "recharge_amount"],
        )
    )
    plan = rec.recognize("帮我充话费")
    assert plan.ambiguous is True
    assert plan.ask_prompt
    assert "phone_number" in plan.ask_prompt
    assert "recharge_amount" in plan.ask_prompt  # 一次问全


def test_high_risk_llm_intent_blocked():
    rec = make_recognizer(llm_recognize=_confident_llm("send_email", 0.96))
    plan = rec.recognize("发送邮件")
    assert plan.risk_level == "mid"  # 发送 → 中风险，不拦截


def test_no_intent_asks_reentry():
    def fake(prompt, history):
        return "[]"

    rec = make_recognizer(llm_recognize=fake)
    plan = rec.recognize("哈哈哈哈哈哈")
    assert plan.ambiguous is True
    assert "没有识别到" in plan.ask_prompt


# ======================================================================
# 追问聚合（AskPromptService）
# ======================================================================


def test_ask_prompt_aggregates_missing_slots():
    cfg = _Cfg()
    ask = AskPromptService(cfg)
    result = FirstStageResult(
        intent_list=[
            item(
                "phone_recharge",
                complete=False,
                miss=["phone_number", "recharge_amount"],
            )
        ],
        all_complete=False,
        total_miss_slots=["phone_number", "recharge_amount"],
    )
    prompt = ask.build_prompt(result)
    assert "手机号" in prompt
    assert "金额" in prompt


def test_ask_user_callback():
    cfg = _Cfg()
    ask = AskPromptService(cfg)

    def fake_ask(prompt, timeout):
        return "13800138000，充50"

    assert ask.ask("请补充", fake_ask, 10) == "13800138000，充50"
    assert ask.ask("请补充", None, 10) is None


# ======================================================================
# 依赖解析 + 调度
# ======================================================================


def test_depend_parallel_when_no_markers():
    svc = IntentDependService()
    groups = svc.parse(
        [item("bill_query"), item("phone_recharge")], "查7月账单，充50话费"
    )
    assert len(groups) == 1
    assert groups[0].dependency == []


def test_depend_serial_with_conditional():
    svc = IntentDependService()
    groups = svc.parse(
        [item("account_balance"), item("phone_recharge")],
        "先查账户余额，余额充足就充值100",
    )
    assert len(groups) == 2
    assert groups[1].dependency == [0]


def test_scheduler_serial_order():
    svc = TaskScheduleService()
    executed = []

    def fake_executor(intent_id, slot_kv):
        executed.append(intent_id)
        return f"{intent_id}:done"

    groups = [
        TaskGroup(group_id=0, dependency=[], intents=[item("account_balance")]),
        TaskGroup(group_id=1, dependency=[0], intents=[item("phone_recharge")]),
    ]
    results = svc.schedule(
        groups, fake_executor, {"account_balance": {}, "phone_recharge": {}}
    )
    assert executed == ["account_balance", "phone_recharge"]
    assert results["phone_recharge"] == "phone_recharge:done"


def test_scheduler_without_executor_returns_pending():
    svc = TaskScheduleService()
    groups = [TaskGroup(group_id=0, intents=[item("bill_query")])]
    results = svc.schedule(groups, None, {"bill_query": {}})
    assert results["bill_query"] == {"status": "pending"}


def test_scheduler_parallel_runs_all():
    svc = TaskScheduleService()
    executed = []

    def fake_executor(intent_id, slot_kv):
        executed.append(intent_id)
        return f"{intent_id}:done"

    groups = [
        TaskGroup(group_id=0, intents=[item("bill_query"), item("phone_recharge")])
    ]
    results = svc.schedule(
        groups, fake_executor, {"bill_query": {}, "phone_recharge": {}}
    )
    assert set(executed) == {"bill_query", "phone_recharge"}
    assert set(results.keys()) == {"bill_query", "phone_recharge"}


# ======================================================================
# 数据访问层（tmp db）
# ======================================================================


def test_memory_store_checkpoints(tmp_path):
    db = make_db(tmp_path)
    store = MemoryStore(db)
    first = FirstStageResult(
        intent_list=[item("bill_query", complete=False, miss=["start_time"])],
        all_complete=False,
        total_miss_slots=["start_time"],
        source="llm",
    )
    store.write_intent_cache("u1", "s1", first)
    store.write_slot_cache("u1", "s1", {"bill_query": {"start_time": "2026-07-01"}})

    cached = store.read_intent_cache("u1", "s1")
    assert cached is not None
    assert cached.intent_list[0].intent_id == "bill_query"
    assert cached.total_miss_slots == ["start_time"]
    assert store.read_slot_cache("u1", "s1") == {
        "bill_query": {"start_time": "2026-07-01"}
    }
    # 按 (user_id, session_id) 唯一：重复写入覆盖不新增
    store.write_intent_cache("u1", "s1", first)
    assert len(store.read_memories("u1", "intent_cache")) == 1
    assert store.read_long_term("u1") == []


def test_slot_cache_merge_skips_filled(tmp_path):
    db = make_db(tmp_path)
    rec = make_recognizer(
        db_path=db,
        llm_recognize=_confident_llm(
            "phone_recharge",
            0.96,
            complete=False,
            miss=["phone_number", "recharge_amount"],
        ),
    )
    # 预填手机号缓存 → 追问只问金额
    rec.store.write_slot_cache(
        "u1", "s1", {"phone_recharge": {"phone_number": "13800138000"}}
    )
    plan = rec.recognize("帮我充话费", user_id="u1", session_id="s1")
    assert plan.ambiguous is True
    assert "phone_number" not in plan.ask_prompt
    assert "recharge_amount" in plan.ask_prompt


def test_events_and_cache_written_on_execute(tmp_path):
    db = make_db(tmp_path)
    rec = make_recognizer(
        db_path=db,
        llm_recognize=_confident_llm("bill_query", 0.96),
        llm_extract_slots=lambda p, h, ids: (
            '[{"intent_id": "bill_query", "slot_info": {"start_time": "2026-07-01"}}]'
        ),
        executor=lambda i, kv: f"{i}:ok",
    )
    rec.recognize("查7月账单", user_id="u1", session_id="s1")

    events = rec.events.list("u1")
    types = [e["event_type"] for e in events]
    assert "intent_recognized" in types
    assert "slot_extracted" in types
    assert "task_scheduled" in types
    assert (
        rec.store.read_slot_cache("u1", "s1")["bill_query"]["start_time"]
        == "2026-07-01"
    )


def test_profile_store_reads_default(tmp_path):
    db = make_db(tmp_path)
    from intent_recognizer import ProfileStore

    assert ProfileStore(db).get_profile("nobody") == {}
    assert ProfileStore(None).get_profile("nobody") == {}


def test_kv_override_thresholds(tmp_path):
    db = make_db(tmp_path)
    from intent_recognizer import KvStore

    kv = KvStore(db)
    kv.set("intent:high_risk_keywords", ["摧毁", "抹除"])
    kv.set("intent:thresholds", {"confidence": 0.5, "high": 0.85})
    cfg = _Cfg(kv=kv)
    assert cfg.get_risk_keywords() == ["摧毁", "抹除"]
    assert cfg.confidence_threshold == 0.5
    assert cfg.high_confidence_threshold == 0.85


def test_recognizer_without_db_is_noop():
    """db_path=None 时数据访问层退化，不报错。"""
    rec = make_recognizer(llm_recognize=_confident_llm("bill_query", 0.96))
    plan = rec.recognize("查7月账单")
    assert plan.primary_intent == "bill_query"


# ======================================================================
# recognize_debug
# ======================================================================


def test_recognize_debug_shape():
    rec = make_recognizer(llm_recognize=_confident_llm("bill_query", 0.96))
    debug = rec.recognize_debug("查7月账单")
    assert "original" in debug
    assert "processed" in debug
    assert "rule_hit" in debug
    assert "first_stage" in debug
    assert debug["first_stage"]["intent_list"][0]["intent_id"] == "bill_query"
    assert debug["plan"]["primary_intent"] == "bill_query"


# ======================================================================
# ConfigManager
# ======================================================================


def test_config_default_thresholds():
    cfg = _Cfg()
    assert cfg.confidence_threshold == 0.6
    assert cfg.high_confidence_threshold == 0.9
    assert cfg.get_intent("phone_recharge").required_slots == [
        "phone_number",
        "recharge_amount",
    ]
    assert cfg.get_intent("phone_recharge").get_slot("phone_number").regex


def test_execution_plan_to_dict():
    cfg = _Cfg()
    plan = make_recognizer(llm_recognize=_confident_llm("bill_query", 0.96)).recognize(
        "查7月账单"
    )
    d = plan.to_dict()
    assert d["primary_intent"] == "bill_query"
    assert d["intents"][0]["intent_id"] == "bill_query"
