# -*- coding: utf-8 -*-
"""一次性端到端测试（mock 嵌入模式，不加载模型）：
验证双通道持久化（关键词 + 模型 remember 工具）与防重复落库。
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["MEMORY_EMBED_MODE"] = "mock"

from agent import Agent
from config import load_config

DB_PATH = Path(__file__).resolve().parent / "_e2e_test.db"
if DB_PATH.exists():
    DB_PATH.unlink()

cfg = load_config()
cfg["memory"]["embed_mode"] = "mock"
cfg["memory"]["db_path"] = str(DB_PATH)
cfg["memory"]["user_id"] = "e2e_user"

agent = Agent(config=cfg)

tests = [
    ("关键词通道", "记住，我家住在杭州"),
    ("模型通道", "我平时喜欢喝冰美式咖啡，不加糖"),
]

for tag, question in tests:
    print(f"\n===== {tag} =====")
    answer = agent.run(question)
    print(f"答案: {answer[:80]}")

print("\n===== 数据库长期记忆检查 =====")
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT memory_type, content, metadata FROM memory_items ORDER BY rowid DESC LIMIT 20"
).fetchall()
for r in rows:
    print(f"- [{r['memory_type']}] {r['content']}  meta={r['metadata']}")
conn.close()
DB_PATH.unlink()
