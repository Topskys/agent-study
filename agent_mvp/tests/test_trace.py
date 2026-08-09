"""trace 模块单测：JSONL 写入、大小轮转、事件语义、agent 挂接。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trace import JsonlRotatingWriter, TraceRecorder


def _read_events(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def test_recorder_writes_full_turn(tmp_path):
    rec = TraceRecorder(enable=True, log_dir=tmp_path, max_bytes=10 * 1024 * 1024)
    rec.turn_start("sess-1", 1, "今天天气怎么样")
    plan = __import__("types").SimpleNamespace(
        primary_intent="tool_weather",
        source_layer="rule_matcher",
        intents=[],
        blocked=False,
        need_ask_slots=False,
        need_disambiguate=False,
        no_valid_intent=False,
        text=None,
    )
    rec.intent(plan)
    rec.tool(name="get_weather", args={"city": "北京"}, result="晴 22.5 度", latency_ms=80.5)
    rec.turn_end(stage="answered", answer="晴", duration_ms=123.4, rounds=2)
    rec.close()

    events = _read_events(tmp_path / "trace.jsonl")
    assert [e["type"] for e in events] == ["turn_start", "intent", "tool", "turn_end"]
    assert events[0]["session_id"] == "sess-1"
    assert events[0]["user_input"] == "今天天气怎么样"
    assert events[1]["primary"] == "tool_weather"
    assert events[1]["layer"] == "rule_matcher"
    assert events[2]["name"] == "get_weather"
    assert events[2]["args"] == {"city": "北京"}
    assert events[2]["latency_ms"] == 80.5
    assert events[3]["stage"] == "answered"
    assert events[3]["duration_ms"] == 123.4
    assert events[3]["rounds"] == 2


def test_recorder_disabled_writes_nothing(tmp_path):
    rec = TraceRecorder(enable=False, log_dir=tmp_path)
    rec.turn_start("s", 1, "hi")
    assert not (tmp_path / "trace.jsonl").exists()


def test_writer_rotates_by_size(tmp_path):
    writer = JsonlRotatingWriter(tmp_path / "trace.jsonl", max_bytes=300, backup_count=2)
    for i in range(30):
        writer.write({"type": "evt", "i": i, "blob": "x" * 40})
    writer.close()

    base = tmp_path / "trace.jsonl"
    assert base.exists()
    assert (tmp_path / "trace.jsonl.1").exists(), "应至少轮转出 .1"
    assert not (tmp_path / "trace.jsonl.3").exists(), "超过 backup_count 的旧文件应被清理"
    assert base.stat().st_size <= 300, "当前文件不应超过 max_bytes"


def test_long_content_is_clipped(tmp_path):
    rec = TraceRecorder(enable=True, log_dir=tmp_path)
    rec.tool(name="echo", args={}, result="x" * 20000)
    rec.close()
    events = _read_events(tmp_path / "trace.jsonl")
    assert len(events[0]["result"]) < 20000
    assert "截断" in events[0]["result"]