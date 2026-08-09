"""基础工具：时间、文件读取。"""

from datetime import datetime
from pathlib import Path

MAX_READ_CHARS = 20000


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_file(path: str) -> str:
    """读取文本文件，超长内容截断返回。"""
    p = Path(path).resolve()
    if not p.exists():
        return f"文件不存在: {path}"
    if not p.is_file():
        return f"不是文件: {path}"
    content = p.read_text(encoding="utf-8", errors="replace")
    if len(content) > MAX_READ_CHARS:
        return (
            content[:MAX_READ_CHARS]
            + f"\n\n...（内容过长，已截断，共 {len(content)} 字符）"
        )
    return content