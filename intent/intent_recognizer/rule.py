"""规则前置校验引擎。

对齐 v3 类图 RuleCheckService：
- 高危关键词 / 黑名单硬信号先行，命中 → 直接拦截（不依赖 LLM）；
- 高频硬信号（记住 / 记忆查询 / 问候 / 命令 / 工具线索 / 疑问词）命中 →
  确定性输出系统级意图（source=rule，置信度 1.0/0.9/0.8）；
- assess_risk 供风险分级（high / mid / low），并入置信度折算。
"""


from .models import IntentNames, RuleHit


class RuleCheckService:
    """规则前置校验：高危拦截 + 系统级硬信号 + 风险分级。"""

    # 明确要求"记住某信息"的祈使句式（迁移自 v2 RuleModule）
    PERSIST_IMPERATIVES = (
        "请记住",
        "请记得",
        "帮我记住",
        "帮我记",
        "替我记住",
        "给我记住",
        "记一下",
        "记下",
        "记着",
        "别忘了",
        "别忘",
        "不要忘",
        "千万别忘",
        "写进记忆",
        "写入记忆",
        "存进记忆",
        "存入记忆",
        "保存下来",
    )
    # 询问是否已记住 / 要求回忆记忆：属于查询，不是持久化意图
    PERSIST_QUERY_MARKERS = (
        "记住了吗",
        "记住了么",
        "记住了没",
        "记住没有",
        "还记得吗",
        "还记得么",
        "还记得没",
        "记得吗",
        "记得么",
        "记起来",
        "记得起来",
        "想起来",
        "记忆里",
    )
    _QUESTION_TAIL = ("吗", "么", "呢", "？", "?", "没", "没有")

    GREETINGS = (
        "你好",
        "您好",
        "嗨",
        "早上好",
        "中午好",
        "下午好",
        "晚上好",
        "谢谢",
        "再见",
        "拜拜",
    )
    QUESTION_WORDS = (
        "什么",
        "为什么",
        "怎么",
        "如何",
        "哪些",
        "是哪",
        "是否",
        "为啥",
        "啥是",
    )
    COMMAND_PREFIXES = ("打开文件管理器", "关闭系统", "重启", "关机", "退出系统")

    # 工具线索：命中即 tool_use
    TOOL_HINTS = (
        ("算", "calculator"),
        ("计算", "calculator"),
        ("几点", "get_time"),
        ("时间", "get_time"),
        ("现在几", "get_time"),
        ("打开", "read_file"),
        ("读取", "read_file"),
        ("读", "read_file"),
        ("查看", "read_file"),
        ("文件", "read_file"),
        ("记住", "remember"),
        ("回忆", "memory_query"),
        ("数据库", "memory_query"),
    )

    MID_RISK_ACTIONS = ("修改", "发送", "导出", "写入", "覆盖", "移动")

    def __init__(self, high_risk_keywords: list[str] | None = None):
        self.high_risk_keywords: list[str] = list(high_risk_keywords or [])

    # ---------- 主入口 ----------

    def check(self, text: str) -> RuleHit:
        """规则匹配。高危 → blocked=True；硬信号 → intent_id；否则空 RuleHit。"""
        text = text.strip()
        if not text:
            return RuleHit()

        reason = self._match_high_risk(text)
        if reason:
            return RuleHit(blocked=True, block_reason=reason)

        if self._match_persist(text):
            return RuleHit(
                intent_id=IntentNames.MEMORY_WRITE,
                confidence=1.0,
                actions=[{"action": "记住", "target": text, "priority": 1}],
                slots={"content": text},
            )

        if self._match_query(text):
            return RuleHit(
                intent_id=IntentNames.MEMORY_QUERY,
                confidence=1.0,
                actions=[{"action": "查询", "target": "历史记忆", "priority": 1}],
            )

        if any(text.startswith(p) for p in self.COMMAND_PREFIXES):
            return RuleHit(
                intent_id=IntentNames.COMMAND,
                confidence=1.0,
                actions=[{"action": "执行命令", "target": text, "priority": 1}],
            )

        tool = self._match_tool(text)
        if tool:
            return RuleHit(
                intent_id=IntentNames.TOOL_USE,
                confidence=0.9,
                actions=[
                    {"action": "使用", "target": text, "priority": 1, "tool": tool}
                ],
            )

        if text in self.GREETINGS or text.startswith("你好") or text.startswith("您好"):
            return RuleHit(
                intent_id=IntentNames.CHAT,
                confidence=1.0,
                actions=[{"action": "闲聊", "target": text, "priority": 1}],
            )

        if any(q in text for q in self.QUESTION_WORDS):
            return RuleHit(
                intent_id=IntentNames.QUESTION,
                confidence=0.8,
                actions=[{"action": "问答", "target": text, "priority": 1}],
            )

        return RuleHit()

    def assess_risk(self, text: str, actions: list[dict] | None = None) -> str:
        """风险分级：high（高危关键词）/ mid（写/发/导出等动作）/ low。"""
        if self._match_high_risk(text):
            return "high"
        for a in actions or []:
            combined = str(a.get("action", "")) + str(a.get("target", ""))
            if self._match_high_risk(combined):
                return "high"
            if any(m in str(a.get("action", "")) for m in self.MID_RISK_ACTIONS):
                return "mid"
        return "low"

    # ---------- 私有匹配 ----------

    def _match_high_risk(self, text: str) -> str | None:
        for k in self.high_risk_keywords:
            if k in text:
                return f"命中高危关键词：{k}"
        return None

    def _match_persist(self, text: str) -> bool:
        """是否明确要求记住（询问是否记住的一律不算）。"""
        if any(m in text for m in self.PERSIST_QUERY_MARKERS):
            return False
        if any(k in text for k in self.PERSIST_IMPERATIVES):
            return True
        if "记住" in text:
            return not text.endswith(self._QUESTION_TAIL)
        return False

    def _match_query(self, text: str) -> bool:
        """是否为记忆查询。"""
        if any(m in text for m in self.PERSIST_QUERY_MARKERS):
            return True
        if ("记得" in text or "记住" in text) and text.endswith(self._QUESTION_TAIL):
            return True
        return False

    def _match_tool(self, text: str) -> str | None:
        for hint, tool in self.TOOL_HINTS:
            if hint in text:
                return tool
        return None
