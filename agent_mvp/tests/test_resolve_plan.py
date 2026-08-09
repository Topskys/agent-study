"""agent._resolve_plan 交互循环回归：chat 不消歧、确认词不重识别。"""

from agent import Agent


def _plan(**over):
    base = {
        "primary_intent": "tool_weather",
        "source_layer": "complex_parser",
        "blocked": False,
        "need_ask_slots": False,
        "need_disambiguate": True,
        "no_valid_intent": False,
        "ask_prompt": "请确认",
        "intents": [object()],
    }
    base.update(over)
    from types import SimpleNamespace

    return SimpleNamespace(**base)


def _agent(recognize_raises=False):
    from types import SimpleNamespace

    rec = SimpleNamespace(
        recognize=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("不应重识别"))
        if recognize_raises
        else object()
    )
    return SimpleNamespace(
        confirm_timeout=30,
        user_id="u",
        session_id="s",
        recognizer=rec,
        ask_user=lambda p, t: "是",
        tracer=SimpleNamespace(ask_user=lambda *a, **k: None),
    )


def test_chat_never_disambiguates():
    a = _agent(recognize_raises=True)
    plan = _plan(primary_intent="chat")
    out = Agent._resolve_plan(a, plan, [], "你好")
    assert out is plan  # 原样放行，未触发任何交互


def test_confirmation_word_proceeds_without_rerecognize():
    a = _agent(recognize_raises=True)
    plan = _plan(primary_intent="tool_weather")
    out = Agent._resolve_plan(a, plan, [], "北京天气")
    assert out.need_disambiguate is False
    assert out.need_ask_slots is False