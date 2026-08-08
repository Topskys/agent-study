"""intent-funnel 包单测（三层漏斗意图识别，docs/设计方案v3.md）。

覆盖：
- 规则层：高危拦截 / 关键词命中 / 槽位静态抽取（别名+正则、防误吞）
- 语义层：分类+相似度双路融合、≥0.9 放行返回
- LLM 层：高置信执行、缺失槽位追问、无效槽位重新收集、0.6~0.9 消歧、<0.6 无意图
- 四分支收敛：直接执行 / 聚合追问 / 消歧 / 重输
- 会话状态：槽位缓存复用（同一会话二次不重复追问）、SQLite 跨实例恢复、检查点落盘
"""



from intent_funnel import FunnelIntentRecognition, SessionStore

# ---------- 工具 ----------


def make_funnel(store=None, llm_gen=None, classifier=None, similarity=None, **kw) -> FunnelIntentRecognition:
    return FunnelIntentRecognition(
        store=store,
        llm_gen=llm_gen,
        classifier=classifier,
        similarity=similarity,
        **kw,
    )


def fake_llm(payload: str):
    """把固定 JSON 包装成 llm_gen 回调。"""

    def _llm_gen(prompt: str) -> str:
        return payload

    return _llm_gen


# ======================================================================
# ① 规则层
# ======================================================================


def test_high_risk_blocked():
    r = make_funnel()
    res = r.recognize("把项目删了吧", session_id="s1")
    assert res.blocked is True
    assert res.source_layer == "rule_matcher"
    assert "删" in res.block_reason
    assert res.ask_prompt


def test_rule_keyword_hit_with_missing_slots():
    r = make_funnel()
    res = r.recognize("帮我充值话费", session_id="s1")
    assert res.source_layer == "rule_matcher"
    assert res.primary_intent == "phone_recharge"
    assert res.need_ask_slots
    assert set(res.ask_slots) == {"phone_number", "recharge_amount"}
    assert "充值手机号" in res.ask_prompt


def test_rule_static_slot_regex():
    """正则槽位静态抽取，且金额不误吞手机号。"""
    r = make_funnel()
    res = r.recognize("给13800138000充值50元话费", session_id="s1")
    assert res.need_ask_slots is False
    slots = {e.intent: e.value for i in res.intents for e in i.entities}
    assert slots == {"phone_number": "13800138000", "recharge_amount": "50"}


def test_rule_static_slot_aliases():
    """目的地别名规范化（北京 → PEK），缺 departure/date 不计必填。"""
    r = make_funnel()
    res = r.recognize("帮我订一张去北京的机票", session_id="s2")
    assert res.primary_intent == "book_flight"
    entities = {e.intent: e for i in res.intents for e in i.entities}
    assert entities["destination"].normalized_value == "PEK"
    # 仅必填 destination 已满足 → 不再追问
    assert res.need_ask_slots is False


def test_rule_no_match_falls_to_semantic():
    """规则未命中且无语义注入 → 无意图兜底。"""
    r = make_funnel()
    res = r.recognize("随便打一段没有关键词的话啊哈哈", session_id="s3")
    assert res.no_valid_intent
    assert res.ask_prompt


# ======================================================================
# 语义层
# ======================================================================


def test_semantic_high_confidence_returns():
    def clf(text, history):
        return {"book_flight": 0.95}

    def sim(text, name):
        return 0.9 if name == "book_flight" else 0.1

    r = make_funnel(classifier=clf, similarity=sim)
    res = r.recognize("帮我买张明天去北京的票", session_id="s10")
    assert res.source_layer == "semantic_reasoner"
    assert res.primary_intent == "book_flight"
    # destination 未填 → 追问
    assert res.need_ask_slots
    assert res.ask_slots == ["destination"]


def test_semantic_low_confidence_downgrade_to_llm():
    def clf(text, history):
        return {"book_flight": 0.5}

    def sem(text, name):
        return 0.4

    # 语义低置信 + 无 LLM → 无意图
    r = make_funnel(classifier=clf, similarity=sem)
    res = r.recognize("帮我买张票", session_id="s11")
    assert res.source_layer == "complex_parser"
    assert res.no_valid_intent


# ======================================================================
# LLM 层
# ======================================================================


def test_llm_high_conf_complete():
    payload = '[{"intent":"send_email","confidence":0.97,"slot_info":{"recipient":"a@b.com","subject":"hi"}}]'
    r = make_funnel(llm_gen=fake_llm(payload))
    res = r.recognize("帮我通知收信人核对一下资料", session_id="s2")
    assert res.source_layer == "complex_parser"
    assert res.primary_intent == "send_email"
    assert res.need_ask_slots is False
    assert res.need_disambiguate is False


def test_llm_high_conf_missing_slots_ask_once():
    payload = '[{"intent":"send_email","confidence":0.97,"slot_info":{"recipient":"a@b.com"}}]'
    r = make_funnel(llm_gen=fake_llm(payload))
    res = r.recognize("帮我通知收信人", session_id="s2")
    assert res.need_ask_slots
    assert res.ask_slots == ["subject"]
    assert "邮件主题" in res.ask_prompt


def test_llm_invalid_slot_recollect():
    payload = (
        '[{"intent":"send_email","confidence":0.97,'
        '"slot_info":{"recipient":"not-an-email","subject":"hi"}}]'
    )
    r = make_funnel(llm_gen=fake_llm(payload))
    res = r.recognize("帮我通知收信人核对一下资料", session_id="s2")
    assert res.need_ask_slots
    assert "recipient" in res.ask_slots
    assert "格式不正确" in res.ask_prompt


def test_llm_ambiguous_disambiguate():
    payload = '[{"intent":"book_flight","confidence":0.72}]'
    r = make_funnel(llm_gen=fake_llm(payload))
    res = r.recognize("帮我订张票", session_id="s2")
    assert res.need_disambiguate
    assert not res.no_valid_intent
    assert "book_flight" in res.ask_prompt


def test_llm_low_conf_no_valid():
    payload = '[{"intent":"book_flight","confidence":0.3}]'
    r = make_funnel(llm_gen=fake_llm(payload))
    res = r.recognize("随便说说", session_id="s2")
    assert res.no_valid_intent


def test_llm_unknown_intent_filtered():
    payload = '[{"intent":"not_registered","confidence":0.99}]'
    r = make_funnel(llm_gen=fake_llm(payload))
    res = r.recognize("胡思乱想的话", session_id="s2")
    assert res.no_valid_intent


# ======================================================================
# 会话状态 / 槽位缓存
# ======================================================================


def test_slot_cache_fills_in_next_turn():
    """第一轮补给全槽位，第二轮仅问缺失；再补后第三轮直接执行。"""
    r = make_funnel()
    r.recognize("给13800138000充值话费", session_id="s1")
    second = r.recognize("充值", session_id="s1")
    assert second.need_ask_slots
    assert second.ask_slots == ["recharge_amount"]
    third = r.recognize("充50元", session_id="s1")
    assert third.need_ask_slots is False


def test_sqlite_persist_across_instances(tmp_path):
    db = tmp_path / "funnel_state.db"
    s1 = SessionStore(db_path=str(db))
    make_funnel(store=s1).recognize("给13800138000充值50元话费", session_id="sessA")

    # 新实例同会话：已确认槽位从 db 恢复，不应再追问
    s2 = SessionStore(db_path=str(db))
    res2 = make_funnel(store=s2).recognize("话费", session_id="sessA")
    assert res2.need_ask_slots is False

    state = s2.get("default_user", "sessA")
    assert state.filled_slots["phone_recharge"] == {
        "phone_number": "13800138000",
        "recharge_amount": "50",
    }
    assert state.checkpoint is not None


def test_checkpoint_persisted():
    r = make_funnel()
    res = r.recognize("给13800138000充50元话费", session_id="sx")
    state = r.tracker.get_session("default_user", "sx")
    assert state.checkpoint is not None
    assert state.checkpoint.primary_intent == "phone_recharge"
    assert res.source_layer == "rule_matcher"