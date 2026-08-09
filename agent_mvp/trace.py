"""Agent 运行轨迹（trace）模块：每次会话事件按 JSONL 落盘，文件达到大小上限自动轮转。

格式：一行一个 JSON 对象，UTF-8 编码。事件类型（event.type）：
- turn_start     会话轮开始（session_id / 轮号 / 用户输入）
- intent         意图识别结果（主意图 / 层级 / 槽位 / 拦截 / 交互标记）
- ask_user       交互追问（缺失槽位 / 消歧 / 重述）及用户回复
- llm            LLM 调用（phase / model / content / tool_calls / finish_reason / 延迟）
- tool           工具调用（name / args / result / 延迟 / 错误）
- turn_end       会话轮结束（stage: blocked / memory_query / answered / max_rounds）
- tip            任意结构化备注

大小限制：单文件满 max_bytes（默认 10MB）即轮转，旧文件依次 .1 → .2 → ...
仅保留最近 backup_count 个（默认 5）。写入线程安全，失败不影响主流程。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _clip(text: str | None, limit: int = 4000) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"…[截断 {len(text) - limit} 字符]"


class JsonlRotatingWriter:
    """JSONL 写入器：一行一事件，超过 max_bytes 轮转，只保留 backup_count 个旧文件。"""

    def __init__(self, path: str | os.PathLike, *, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self._lock = threading.Lock()
        self._fh = None
        self._size = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open()

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.seek(0, os.SEEK_END)
        self._size = self._fh.tell()

    def _rotate(self):
        self._fh.close()
        base = self.path
        if self.backup_count > 0:
            oldest = Path(f"{base}.{self.backup_count}")
            if oldest.exists():
                oldest.unlink()
            for i in range(self.backup_count - 1, 0, -1):
                src = Path(f"{base}.{i}")
                if src.exists():
                    os.replace(src, Path(f"{base}.{i + 1}"))
            os.replace(base, Path(f"{base}.1"))
        self._open()

    def write(self, event: dict) -> None:
        if not isinstance(event, dict):
            raise TypeError("trace 事件必须是 dict")
        line = json.dumps(event, ensure_ascii=False) + "\n"
        raw = line.encode("utf-8")
        with self._lock:
            if self.max_bytes > 0 and self._size + len(raw) > self.max_bytes:
                self._rotate()
            try:
                self._fh.write(line)
                self._size += len(raw)
            finally:
                self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


class TraceRecorder:
    """Agent 轨迹记录器：收拢 turn_start / intent / ask_user / llm / tool / turn_end 事件。"""

    def __init__(
        self,
        *,
        enable: bool = False,
        log_dir: str | os.PathLike = "data/trace",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        self.enable = bool(enable)
        self.max_bytes = int(max_bytes)
        self.backup_count = int(backup_count)
        self._writer = (
            JsonlRotatingWriter(
                self._resolve(log_dir) / "trace.jsonl",
                max_bytes=self.max_bytes,
                backup_count=self.backup_count,
            )
            if self.enable
            else None
        )
        self._turns_written = 0

    def close(self):
        if self._writer is not None:
            self._writer.close()

    @staticmethod
    def _resolve(log_dir: str | os.PathLike) -> Path:
        """日志目录：绝对路径直接用，相对路径相对 agent_mvp 包根解析。"""
        p = Path(log_dir)
        if p.is_absolute():
            return p
        return Path(__file__).resolve().parent / p

    def enabled(self) -> bool:
        return self.enable and self._writer is not None

    def _emit(self, type_: str, **fields) -> None:
        if not self.enabled():
            return
        event = {"type": type_, "ts": _now_iso(), **fields}
        try:
            self._writer.write(event)
        except Exception:
            # 日志失败不影响对话主流程
            pass

    def turn_start(self, session_id: str, turn_index: int, user_input: str) -> None:
        self._turns_written = turn_index
        self._emit("turn_start", session_id=session_id, turn=turn_index, user_input=_clip(user_input))

    def intent(self, plan) -> None:
        try:
            slots = {
                item.name: {e.intent: e.value for e in item.entities if e.valid}
                for item in (getattr(plan, "intents", []) or [])
            }
            self._emit(
                "intent",
                turn=self._turns_written,
                primary=getattr(plan, "primary_intent", None),
                layer=getattr(plan, "source_layer", None),
                confidence=[i.confidence for i in (getattr(plan, "intents", []) or [])],
                slots=slots,
                blocked=bool(getattr(plan, "blocked", False)),
                ask_slots=bool(getattr(plan, "need_ask_slots", False)),
                disambiguate=bool(getattr(plan, "need_disambiguate", False)),
                no_intent=bool(getattr(plan, "no_valid_intent", False)),
                text=getattr(plan, "text", None),
            )
        except Exception:
            pass

    def ask_user(self, prompt: str, reply: str | None) -> None:
        self._emit("ask_user", turn=self._turns_written, prompt=_clip(prompt), reply=_clip(reply or ""))

    def llm(
        self,
        *,
        phase: str,
        model: str,
        content: str | None = None,
        tool_calls: list | None = None,
        finish_reason: str | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        calls = []
        for tc in tool_calls or []:
            try:
                args = json.loads(tc.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append({"name": tc.name, "args": args})
        self._emit(
            "llm",
            turn=self._turns_written,
            phase=phase,
            model=model,
            content=_clip(content),
            tool_calls=calls or None,
            finish_reason=finish_reason,
            latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
            error=_clip(error),
        )

    def tool(self, *, name: str, args: dict, result: str, latency_ms: float | None = None) -> None:
        self._emit(
            "tool",
            turn=self._turns_written,
            name=name,
            args=args,
            result=_clip(result, limit=8000),
            latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
        )

    def turn_end(self, *, stage: str, answer: str, duration_ms: float | None = None, rounds: int | None = None) -> None:
        self._emit(
            "turn_end",
            turn=self._turns_written,
            stage=stage,
            answer=_clip(answer, limit=8000),
            duration_ms=round(duration_ms, 1) if duration_ms is not None else None,
            rounds=rounds,
        )