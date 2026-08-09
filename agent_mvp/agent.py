"""ReAct Agent（function calling 版），集成三层记忆 + 意图识别（intent-funnel 三层漏斗）。

统一 LLM 接入层（llm-client 包）：
- 所有模型调用走 `client.chat(LLMRequest(...))`，统一重试/缓存/路由/审计
- ReAct 的 reason / 意图识别的注入回调（llm_gen / ask_user）全部转发到 llm_client

ReAct = Reason + Act + Observe 的循环：
- reason：调模型，让它决定下一步（输出答案或发起工具调用）
- act：模型发起工具调用时，执行工具
- observe：把工具结果写回记忆，进入下一轮推理

记忆能力（memory-system 包）：
- 自动记录：每轮把用户输入与助手回答写入工作记忆与会话记忆
- 相关回忆：推理前按当前输入召回历史记忆，注入上下文
- 沉淀：reset / 新会话时，end_session 将会话记忆沉淀进长期记忆
- 立即持久化（双通道，先关键词后模型）：
  1. 关键词启发式：由 intent 规则层识别 memory_write 意图，回合结束落库
  2. 模型自主：未命中关键词时，由模型判断是否调用 remember 工具
  由 _persisted_this_turn 标志防与 remember 工具双写

意图识别（intent-funnel 包，V3 三层漏斗）：
- 三层串行漏斗：RuleMatcher（规则）→ SemanticReasoner（语义）→ ComplexIntentParser（LLM）
- 四分支交互兜底：直接执行 / 缺失槽位聚合追问(need_ask_slots) / 消歧反问(need_disambiguate) / 重输(no_valid_intent)
- 高风险关键词硬拦截（blocked）；会话槽位经 SessionStore 缓存复用
- 自定义 resources/intent_config.json：内置业务意图 + memory_write / memory_query 记忆意图

模块划分：
- llm-client        统一 LLM 接入层（外部包）
- tool_schemas.json  工具 schema
- tools.py           工具层：实现、分发
- config.py          配置层：YAML 加载
- intent-funnel     意图识别三层漏斗（外部包）
- memory_system      记忆层（外部包）
- agent.py           ReAct 循环 + 入口
"""

import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from intent_funnel import (
    FunnelConfig,
    FunnelIntentRecognition,
    IntentResult,
    SessionStore,
)
from llm_client import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
    ToolCall,
    build_client,
)

from config import DEFAULT_CONFIG_PATH, load_config
from memory_system.core.memory_system import MemorySystem
from tools import (
    TOOLS,
    run_tool,
    set_bing_search_key,
    set_memory_persist_hook,
    set_openweather_key,
    set_tavily_search_key,
)

# 系统级意图常量（与 resources/intent_config.json 的意图名对齐）
class IntentNames:
    MEMORY_WRITE = "memory_write"
    MEMORY_QUERY = "memory_query"
    CHAT = "chat"

# 追问循环上限：防止因 ask_user 一直给空回复而无限循环
_MAX_INTERACT_ROUNDS = 3


class Agent:
    """ReAct Agent：reason → act → observe 循环，接入三层记忆系统与意图识别。"""

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | os.PathLike = DEFAULT_CONFIG_PATH,
        llm_client: LLMClient | None = None,
        llm_gen=None,
        ask_user=None,
    ):
        cfg = config or load_config(config_path)
        llm = cfg.get("llm", {})
        self.api_key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
        self.model = llm.get("model") or os.environ.get("LLM_MODEL")
        self.temperature = float(llm.get("temperature", 1.0))
        self.top_p = float(llm.get("top_p", 0.95))
        self.max_tokens = int(llm.get("max_tokens", 16384))
        self.timeout = float(llm.get("timeout", 120))
        self.max_retries = int(llm.get("max_retries", 3))
        self.stream = bool(llm.get("stream", True))
        self.max_rounds = int(cfg.get("agent", {}).get("max_rounds", 10))
        self.system_prompt = cfg.get("agent", {}).get("system_prompt")

        # 统一 LLM 客户端（可注入，默认按配置 build_client；含重试/缓存/路由/审计）
        self.client = llm_client or build_client(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model or "",
            provider=str(llm.get("provider", "openai_compat")),
            timeout=self.timeout,
            max_retries=self.max_retries,
            enable_cache=bool(llm.get("enable_cache", False)),
        )
        self.memory: list[Message] = [
            Message(role="system", content=self.system_prompt or "")
        ]

        # ---------- 记忆系统 ----------
        mem = cfg.get("memory", {})
        os.environ.setdefault("MEMORY_EMBED_MODE", str(mem.get("embed_mode", "mock")))
        self.memory_top_k = int(mem.get("top_k", 3))
        self.session_importance = float(mem.get("session_importance", 0.8))
        self.user_id = str(mem.get("user_id", "default_user"))
        db_path = str(mem.get("db_path", "agent_memory.db"))
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).resolve().parent / db_path)
        self.memory_system = MemorySystem(db_path=db_path)
        self.user_memory = self.memory_system.get_user_memory(self.user_id)
        self.session_id = str(uuid.uuid4())
        self.user_memory.start_session(self.session_id)
        self._persisted_this_turn = False
        set_memory_persist_hook(self._remember_long_term)
        set_bing_search_key(cfg.get("tools", {}).get("bing_search_api_key", ""))
        set_tavily_search_key(cfg.get("tools", {}).get("tavily_api_key", ""))
        set_openweather_key(cfg.get("tools", {}).get("openweather_api_key", ""))

        # ---------- 意图识别（intent-funnel 三层漏斗） ----------
        intent_cfg = cfg.get("intent", {})
        safety_cfg = cfg.get("safety", {})
        self.confirm_timeout = int(intent_cfg.get("confirm_timeout", 30))
        self.highrisk_timeout = int(intent_cfg.get("highrisk_timeout", 60))
        funnel_cfg_path = cfg.get("intent", {}).get("config_path") or str(
            Path(__file__).resolve().parent / "resources" / "intent_config.json"
        )
        self.recognizer = FunnelIntentRecognition(
            config=FunnelConfig(config_path=funnel_cfg_path),
            store=SessionStore(db_path=self.memory_system.vector_store.db_path),
            llm_gen=llm_gen or self._llm_gen,
            llm_timeout=float(self.timeout),
            high_risk_keywords=safety_cfg.get("high_risk_keywords"),
        )
        self.ask_user = ask_user or self._ask_user
        self._turn_inputs: list[str] = []
        self._stream_last_printed = False

    def reset(self):
        """清空对话与工作记忆，把当前会话沉淀进长期记忆，开启新会话。"""
        self.user_memory.end_session()
        self.user_memory.flush_working_memory()
        self.session_id = str(uuid.uuid4())
        self.user_memory.start_session(self.session_id)
        self.memory = [
            Message(role="system", content=self.system_prompt or "")
        ]
        self._turn_inputs = []

    # ---------- 意图识别：LLM 回调注入 ----------

    def _llm_gen(self, prompt: str) -> str:
        """漏斗 LLM 层回调：已拼好 prompt，直接交模型返回原始文本（期望意图 JSON）。"""
        try:
            resp = self.client.chat(
                LLMRequest.from_prompt(
                    prompt, task="intent", temperature=0, max_tokens=1000
                )
            )
            return resp.content or ""
        except Exception as e:
            return f"[LLM错误: {e}]"

    def _ask_user(self, prompt: str, timeout: float) -> str | None:
        """交互追问回调：REPL 场景注入 input()，API 场景不注入则跳过。"""
        print(prompt)
        try:
            return input("> ")
        except (EOFError, KeyboardInterrupt):
            return None

    # ---------- 记忆：召回与记录 ----------

    def _recall(self, user_input: str) -> str | None:
        """召回与当前输入相关的历史记忆，返回上下文文本；无则返回 None。"""
        memories = self.user_memory.retrieve_relevant_memories(
            user_input, self.memory_top_k
        )
        if not memories:
            return None
        lines = [f"- [{m.memory_type.value}] {m.content}" for m in memories]
        return "回忆到的历史记忆：\n" + "\n".join(lines)

    def _record(self, user_input: str, answer: str):
        """把一轮问答写入工作记忆与会话记忆。

        会话记忆仅在 reset 时沉淀到长期记忆；若意图为 memory_write，
        则由 run() 在回合结束额外调用 _persist 立即落库。
        """
        self.user_memory.add_working_memory(
            f"用户: {user_input}", metadata={"role": "user"}
        )
        self.user_memory.add_working_memory(
            f"助手: {answer}", metadata={"role": "assistant"}
        )
        self.user_memory.add_session_memory(
            f"用户: {user_input}",
            metadata={"importance": self.session_importance, "role": "user"},
        )
        self.user_memory.add_session_memory(
            f"助手: {answer}",
            metadata={"importance": 0.5, "role": "assistant"},
        )

    def _persist(self, user_input: str):
        """把用户要求记住的信息立即写入长期记忆（落库）。"""
        content = f"用户: {user_input}"
        item = self.user_memory.add_long_term_memory(
            content,
            metadata={"importance": self.session_importance, "role": "user"},
        )
        if item:
            print(f"  [memory] 已记住并持久化: {content}")
        else:
            print(f"  [memory] 未通过治理裁决，未持久化: {content}")

    def _remember_long_term(self, content: str, importance: float) -> bool:
        """remember 工具的钩子：模型自主判断时把信息写入长期记忆（落库）。"""
        item = self.user_memory.add_long_term_memory(
            content,
            metadata={"importance": float(importance), "role": "user"},
        )
        if item:
            self._persisted_this_turn = True
            print(f"  [memory] 模型自主记住: {content}")
            return True
        print(f"  [memory] 未通过治理裁决，未持久化: {content}")
        return False

    # ---------- 意图识别：执行计划路由 ----------

    def _recent_user_history(self, n: int = 5) -> list[str]:
        """取最近 N 轮用户原始输入作为识别上下文。"""
        return self._turn_inputs[-n:]

    def _resolve_plan(
        self,
        plan: IntentResult,
        history: list[str],
        user_input: str,
    ) -> IntentResult:
        """追问 / 消歧 / 重述回灌：ask_user 一问，回复后重新识别。

        覆盖漏斗四分支交互兜底：plan.ask_prompt 承载"缺失槽位 / 消歧 /
        重述"等所有主动交互话术；回复作为新一轮输入重走全流程，
        已填槽位经 SessionStore 会话缓存自动累积。

        无意图重述（no_valid_intent）特殊处理：只问一次；重述后仍识别不出
        任何意图，说明输入本就不是受支持的业务操作，停止追问、交由通用
        对话兜底，避免"请换一种说法重新输入"循环把用户卡住。
        """
        no_intent_asked = False
        for _ in range(_MAX_INTERACT_ROUNDS):
            ambiguous = (
                plan.need_ask_slots or plan.need_disambiguate or plan.no_valid_intent
            )
            # chat 是通用兜底意图：绝不向用户做"确认执行 chat"式追问/消歧
            if plan.primary_intent == IntentNames.CHAT:
                return plan
            if plan.blocked or not (ambiguous and plan.ask_prompt and self.ask_user):
                break
            if not plan.intents:
                if no_intent_asked:
                    break
                no_intent_asked = True
            reply = self.ask_user(plan.ask_prompt, float(self.confirm_timeout))
            if not reply or not reply.strip():
                break
            # 把用户补充/确认里的城市等实体并入当前输入的重新识别，
            # 但简单肯定词（是/对/可以/好的）不重新识别，直接按当前意图推进
            if re.sub(r"[\s，。）(。~～！!？?]", "", reply.strip().lower()) in {
                "是", "对", "可以", "行", "好的", "好", "确定", "确认", "执行", "y", "yes", "ok", "嗯", "是的", "对的对的"
            }:
                plan.need_disambiguate = False
                plan.need_ask_slots = False
                plan.no_valid_intent = False
                continue
            plan = self.recognizer.recognize(
                reply.strip(), history, self.user_id, self.session_id
            )
            print(
                f"  [intent] 补全后 primary={plan.primary_intent} "
                f"src={plan.source_layer} amb={plan.need_ask_slots or plan.need_disambiguate}"
            )
            if not plan.intents:
                break
        return plan

    def _block_high_risk(self, plan: IntentResult) -> str:
        """高风险操作拦截回复。"""
        print("  [安全] 已拦截高风险操作")
        return plan.ask_prompt or (
            "安全拦截：该请求包含删除/清空/批量/外发等高风险操作，已停止执行。"
            "如需继续，请确认并明确操作范围。"
        )

    def _answer_memory_query(self) -> str:
        """记忆查询短路：直查库（memory_items 表 long_term），不调 LLM。"""
        db_path = self.memory_system.vector_store.db_path
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT content FROM memory_items WHERE user_id=? AND memory_type='long_term' "
                "ORDER BY created_at DESC LIMIT ?",
                (self.user_id, self.memory_top_k),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return "我当前还没有记住什么长期信息。"
        lines = [f"- {r[0]}" for r in rows]
        return "我记住的信息：\n" + "\n".join(lines)

    def _build_user_content(
        self, context: str | None, plan: IntentResult, user_input: str
    ) -> str:
        """把意图识别结果注入用户消息，帮助模型聚焦执行。"""
        intents_desc = (
            "、".join(f"{i.name}({i.confidence:.2f})" for i in plan.intents) or "未知"
        )
        hint = f"用户意图识别: {intents_desc}（层级: {plan.source_layer}）"
        slots = {
            item.name: {e.intent: e.value for e in item.entities if e.valid}
            for item in plan.intents
        }
        if slots:
            hint += f"；槽位: {json.dumps(slots, ensure_ascii=False)}"
        if plan.text and plan.text != user_input:
            hint += f"；净化后输入: {plan.text}"
        body = f"{hint}\n\n用户问题：{user_input}"
        return f"{context}\n\n{body}" if context else body

    # ---------- ReAct 三步骤 ----------

    def context_stats(self) -> dict:
        """统计当前上下文大小：消息数、字符数、UTF-8 字节数与估算 token 数。

        本项目用代理模型且未装 tiktoken，拿不到精确 tokenizer，
        这里按"中文约 0.6 token/字、其他字符约 0.25 token/字符"估算，仅供参考。
        """
        tool_json = json.dumps(TOOLS, ensure_ascii=False)
        messages_json = json.dumps(
            [m.model_dump(exclude_none=True) for m in self.memory],
            ensure_ascii=False,
        )
        all_text = tool_json + messages_json
        cjk = sum(1 for c in all_text if "\u4e00" <= c <= "\u9fff")
        est_tokens = int(cjk * 0.6 + (len(all_text) - cjk) * 0.25)
        return {
            "messages": len(self.memory),
            "chars": len(all_text),
            "bytes_utf8": len(all_text.encode("utf-8")),
            "est_tokens": est_tokens,
            "max_tokens": self.max_tokens,
        }

    def reason(self) -> LLMResponse:
        """Reason：调模型。返回聚合后的完整 LLMResponse。

        - stream=True：逐 chunk 接收，content 边接收边打印（DCE 输出体验）
        - 流式 tool_calls 按 index 归并增量分片，拼成完整 ToolCall
        - 重试由 llm_client 的 RetryPolicy 统一处理（429/5xx/网络错误指数退避），
          非流式路径走 RetryPolicy；流式不可断点重试，失败直接上抛
        """
        req = LLMRequest(
            messages=self.memory,
            tools=TOOLS,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        if not self.stream:
            resp = self.client.chat(req)
            if resp.content:
                print(f"Agent: {resp.content}")
                self._stream_last_printed = True
            return resp

        content_parts: list[str] = []
        calls: dict[int, ToolCall] = {}
        finish_reason = None
        content_printed = False
        for chunk in self.client.chat_stream(req):
            if chunk.content:
                if not content_printed:
                    print("Agent: ", end="", flush=True)
                    content_printed = True
                content_parts.append(chunk.content)
                print(chunk.content, end="", flush=True)
            if chunk.tool_calls:
                for tc in chunk.tool_calls:
                    cur = calls.get(tc.index)
                    if cur is None:
                        calls[tc.index] = tc
                    else:
                        cur.id = cur.id or tc.id
                        cur.name = (cur.name or "") + (tc.name or "")
                        cur.arguments = (cur.arguments or "") + (tc.arguments or "")
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        if content_printed:
            print()  # 流式输出结束补换行
        tool_calls = [calls[i] for i in sorted(calls)] or None
        self._stream_last_printed = content_printed
        return LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model=self.model or "",
        )

    def act(self, call: ToolCall) -> str:
        """Act：执行一个工具调用，返回观察结果。"""
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        result = run_tool(call.name, args)
        try:
            print(f"  [tool] {call.name}({json.dumps(args, ensure_ascii=False)}) -> {result}")
        except UnicodeEncodeError:
            # Windows 控制台编码问题：截断输出
            print(f"  [tool] {call.name}({json.dumps(args, ensure_ascii=False)}) -> [输出含无法显示字符]")
        return result

    def observe(self, call: ToolCall, result: str):
        """Observe：把工具调用与结果写回记忆。"""
        self.memory.append(
            Message(
                role="tool",
                tool_call_id=call.id,
                name=call.name,
                content=result,
            )
        )

    # ---------- 主循环 ----------

    def run(self, user_input: str, max_rounds: int | None = None) -> str:
        self._turn_inputs.append(user_input)
        self._persisted_this_turn = False
        self._stream_last_printed = False

        # ① 意图识别（三层漏斗：规则 → 语义 → LLM + 主动交互分支）
        history = self._recent_user_history()[:-1]
        plan = self.recognizer.recognize(
            user_input, history, self.user_id, self.session_id
        )
        print(
            f"  [intent] primary={plan.primary_intent} src={plan.source_layer} "
            f"blocked={plan.blocked} ask_slots={plan.need_ask_slots} "
            f"disamb={plan.need_disambiguate} no_intent={plan.no_valid_intent}"
        )

        # ② 主动交互：缺失槽位 / 消歧 / 重述 → ask_user 回灌重识别
        plan = self._resolve_plan(plan, self._recent_user_history(), user_input)

        # ③ 高风险拦截
        if plan.blocked:
            return self._block_high_risk(plan)

        # ④ 记忆查询短路：直查库，不调 LLM
        if plan.primary_intent == IntentNames.MEMORY_QUERY:
            return self._answer_memory_query()

        # ⑤ ReAct 循环（记忆写入回合结束时落库，标志防双写）
        question = plan.text or user_input
        context = self._recall(question)
        self.memory.append(
            Message(
                role="user",
                content=self._build_user_content(context, plan, question),
            )
        )
        max_rounds = max_rounds or self.max_rounds

        for _ in range(max_rounds):
            resp = self.reason()
            self.memory.append(
                Message(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls,
                )
            )

            if not resp.tool_calls:
                answer = resp.content or "（无输出）"
                self._record(user_input, answer)
                if (
                    plan.primary_intent == IntentNames.MEMORY_WRITE
                    and not self._persisted_this_turn
                ):
                    self._persist(user_input)
                return answer

            for call in resp.tool_calls:
                result = self.act(call)
                self.observe(call, result)

        self._record(user_input, "达到最大迭代轮数，未获得最终答案")
        if (
            plan.primary_intent == IntentNames.MEMORY_WRITE
            and not self._persisted_this_turn
        ):
            self._persist(user_input)
        return "达到最大迭代轮数，未获得最终答案"


def run_agent(question: str, max_rounds: int = 10) -> str:
    """一次性调用（无状态），每次新建会话。需要跨轮次记忆请用 Agent。"""
    return Agent().run(question, max_rounds)


if __name__ == "__main__":
    import sys

    # Windows 默认 GBK 控制台打不出 emoji/生僻字，统一用 UTF-8 输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    agent = Agent()
    print(
        "Agent MVP（/exit 退出；/reset 清空对话并沉淀记忆；/context 查看上下文；/intent 查看意图识别）"
    )
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if question.strip().lower() in ("/exit", "/quit"):
            break
        if question.strip().lower() in ("/reset", "/clear"):
            agent.reset()
            print("记忆已沉淀，对话已清空")
            continue
        if question.strip().lower() in ("/context", "/ctx"):
            s = agent.context_stats()
            print(
                f"上下文: {s['messages']} 条消息 | {s['chars']} 字符 | "
                f"{s['bytes_utf8']} 字节(UTF-8) | 估算 {s['est_tokens']} token "
                f"(max_tokens={s['max_tokens']})"
            )
            continue
        if question.strip().lower().startswith("/intent"):
            text = question[len("/intent") :].strip()
            if not text:
                last = agent._turn_inputs[-1] if agent._turn_inputs else None
                if not last:
                    print("用法: /intent <输入>，查看意图识别结果（无参时用上一条用户输入）")
                    continue
                text = last
                print(f"（复用上一条输入: {text}）")
            plan = agent.recognizer.recognize(
                text, agent._recent_user_history(), agent.user_id, agent.session_id
            )
            print(f"意图: {[i.name for i in plan.intents]}")
            print(
                f"层级: {plan.source_layer} | 置信度: {[i.confidence for i in plan.intents]}"
            )
            print(
                f"拦截: {plan.blocked} | 缺槽: {plan.need_ask_slots} | 消歧: {plan.need_disambiguate} | 无意图: {plan.no_valid_intent}"
            )
            slots = {
                item.name: {e.intent: e.value for e in item.entities}
                for item in plan.intents
            }
            print(f"槽位: {slots}")
            ranking = [
                (r.name, r.confidence)
                for r in getattr(plan, "intent_ranking", []) or []
            ]
            print(f"意图排序: {ranking}")
            print(f"追问: {plan.ask_prompt}")
            print(f"净化后: {plan.text}")
            continue
        try:
            answer = agent.run(question)
            if not agent._stream_last_printed:
                print(f"Agent: {answer}")
        except Exception as e:
            print(f"Agent 错误: {e}")
