"""注入式回调类型定义。

intent 包不依赖任何具体 LLM SDK（不 import openai / langchain），
全部 LLM 能力由宿主（agent_mvp）以回调形式注入：
- llm_recognize     阶段一：多意图识别
- llm_extract_slots 阶段二：批量槽位抽取
- ask_user          追问 / 消歧
- llm_expand        短提问扩写
- executor          任务执行（可空，调度阶段不注入则不执行）
"""

from typing import Any, Callable, List, Optional

# 阶段一：输入 (prompt, history)，输出含意图数组的原始文本（期望 JSON）。
LLMRecognize = Callable[[str, List[str]], str]
# 阶段二：输入 (prompt, history, intent_ids)，输出含槽位数组的原始文本（期望 JSON）。
LLMExtractSlots = Callable[[str, List[str], List[str]], str]
# 追问 / 消歧：prompt -> 用户回复（超时/取消 -> None）。
AskUser = Callable[[str, float], Optional[str]]
# 短提问扩写：text + history -> 扩写文本（可 None）。
LLMExpand = Callable[[str, List[str]], Optional[str]]
# 任务执行器：intent_id + 该意图槽位 -> 执行结果。
TaskExecutor = Callable[[str, dict], Any]
