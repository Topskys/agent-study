"""ReAct Agent（function calling 版），集成三层记忆 + 意图识别（v3 两阶段）。

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

意图识别（intent-recognizer 包，v3 两阶段多意图识别）：
- 阶段一：多意图识别 + 置信度打分 + 槽位完备性校验
- 阶段二：批量槽位抽取 + 参数格式校验
- 缺失参数统一聚合追问（ask_prompt 一次问全），已填槽位缓存复用
- 高风险拦截 / 中风险确认；规则硬信号短路（memory_write / memory_query 等）
- 数据访问层直连 agent_memory.db 现有记忆模块数据表

模块划分：
- tool_schemas.json  工具 schema
- tools.py           工具层：实现、分发
- config.py          配置层：YAML 加载
- intent-recognizer  意图识别（外部包）
- memory_system      记忆层（外部包）
- agent.py           ReAct 循环 + 入口
"""

import json
import os
import time
import uuid
from pathlib import Path

from openai import OpenAI

from config import DEFAULT_CONFIG_PATH, load_config
from intent_recognizer import ExecutionPlan, IntentNames, IntentRecognizer
from memory_system.core.memory_system import MemorySystem
from tools import TOOLS, run_tool, set_bing_search_key, set_memory_persist_hook, set_tavily_search_key

# LLM 短提问扩写 prompt：补全信息、不改变原意（意图预处理阶段使用）
_EXPAND_PROMPT = (
    "用户这句请求不完整或有歧义，请扩写成信息完整的请求。\n"
    "要求：只输出扩写后的句子，补充合理细节、不改变原意；不需要扩写就原样返回。\n"
    "{context}"
    "用户输入：{text}"
)

# 追问循环上限：防止因 ask_user 一直给空回复而无限循环
_MAX_INTERACT_ROUNDS = 3


class Agent:
    """ReAct Agent：reason → act → observe 循环，接入三层记忆系统与意图识别。"""

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | os.PathLike = DEFAULT_CONFIG_PATH,
        llm_recognize=None,
        llm_extract_slots=None,
        ask_user=None,
        llm_expand=None,
    ):
        cfg = config or load_config(config_path)
        llm = cfg.get("llm", {})
        self.api_key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = llm.get("base_url") or os.environ.get("OPENAI_BASE_URL", "")
        self.model = llm.get("model") or os.environ.get("LLM_MODEL")
        self.temperature = float(llm.get("temperature", 1.0))
        self.top_p = float(llm.get("top_p", 0.95))
        self.max_tokens = int(llm.get("max_tokens", 16384))
        self.timeout = int(llm.get("timeout", 120))
        self.max_retries = int(llm.get("max_retries", 3))
        self.max_rounds = int(cfg.get("agent", {}).get("max_rounds", 10))
        self.system_prompt = cfg.get("agent", {}).get("system_prompt")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.memory: list = [{"role": "system", "content": self.system_prompt}]

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

        # ---------- 意图识别（独立模块，v3 两阶段） ----------
        intent_cfg = cfg.get("intent", {})
        safety_cfg = cfg.get("safety", {})
        self.confirm_timeout = int(intent_cfg.get("confirm_timeout", 30))
        self.highrisk_timeout = int(intent_cfg.get("highrisk_timeout", 60))
        self.recognizer = IntentRecognizer(
            db_path=self.memory_system.vector_store.db_path,
            llm_recognize=llm_recognize or self._llm_recognize,
            llm_extract_slots=llm_extract_slots or self._llm_extract_slots,
            ask_user=ask_user or self._ask_user,
            llm_expand=llm_expand or self._llm_expand,
            high_risk_keywords=safety_cfg.get("high_risk_keywords"),
        )
        self._turn_inputs: list[str] = []

    def reset(self):
        """清空对话与工作记忆，把当前会话沉淀进长期记忆，开启新会话。"""
        self.user_memory.end_session()
        self.user_memory.flush_working_memory()
        self.session_id = str(uuid.uuid4())
        self.user_memory.start_session(self.session_id)
        self.memory = [{"role": "system", "content": self.system_prompt}]
        self._turn_inputs = []

    # ---------- 意图识别：LLM 回调注入 ----------

    def _llm_recognize(self, prompt: str, history: list[str]) -> str:
        """阶段一回调：意图模块已拼好 prompt，直接交给模型返回原始文本。"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM错误: {e}]"

    def _llm_extract_slots(
        self, prompt: str, history: list[str], intent_ids: list[str]
    ) -> str:
        """阶段二回调：批量槽位抽取。"""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM错误: {e}]"

    def _llm_expand(self, text: str, history: list[str]) -> str:
        """预处理回调：短提问扩写。"""
        try:
            context = "".join(f"历史：{h}\n" for h in history[-3:])
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": _EXPAND_PROMPT.format(context=context, text=text),
                    }
                ],
                temperature=0,
                max_tokens=300,
            )
            return resp.choices[0].message.content or text
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
        plan: ExecutionPlan,
        history: list[str],
        user_input: str,
    ) -> ExecutionPlan:
        """追问 / 消歧 / 拦截提示回灌：ask_user 一问，回复后重新识别。

        覆盖 v3 三档分发与聚合追问：plan.ask_prompt 承载"缺失槽位 / 消歧 /
        低置信重述"等所有主动交互话术；回复作为新一轮输入重走全流程，
        已填槽位经 slot_cache 自动累积。

        无意图重述（低置信 <0.6）特殊处理：只问一次；重述后仍识别不出
        任何意图，说明输入本就不是受支持的业务操作，停止追问、交由通用
        对话兜底，避免"请换一种说法重新输入"循环把用户卡住。
        """
        no_intent_asked = False
        for _ in range(_MAX_INTERACT_ROUNDS):
            if plan.blocked or not (
                plan.ambiguous and plan.ask_prompt and self.recognizer.ask_user
            ):
                break
            if not plan.intents:
                if no_intent_asked:
                    break
                no_intent_asked = True
            reply = self.recognizer.ask_user(
                plan.ask_prompt, float(self.confirm_timeout)
            )
            if not reply or not reply.strip():
                break
            plan = self.recognizer.recognize(
                reply.strip(), history, self.user_id, self.session_id
            )
            print(
                f"  [intent] 补全后 primary={plan.primary_intent} "
                f"src={plan.source} amb={plan.ambiguous}"
            )
            if not plan.intents:
                break
        return plan

    def _confirm_mid_risk(self, plan: ExecutionPlan) -> bool:
        """中风险操作确认：无交互回调（API 场景）默认执行。"""
        names = "、".join(i.name for i in plan.intents) or plan.primary_intent or "未知"
        prompt = f"  [安全] 中风险操作（{names}），确认执行吗？(y/n) "
        if not self.recognizer.ask_user:
            return True
        reply = self.recognizer.ask_user(prompt, float(self.confirm_timeout))
        return reply is not None and reply.strip().lower() in (
            "y",
            "yes",
            "是",
            "确定",
            "确认",
        )

    def _block_high_risk(self, plan: ExecutionPlan) -> str:
        """高风险操作拦截回复。"""
        print("  [安全] 已拦截高风险操作")
        return plan.ask_prompt or (
            "安全拦截：该请求包含删除/清空/批量/外发等高风险操作，已停止执行。"
            "如需继续，请确认并明确操作范围。"
        )

    def _answer_memory_query(self) -> str:
        """记忆查询短路：直查库（memory_items 表 long_term），不调 LLM。"""
        rows = self.recognizer.store.read_long_term(self.user_id, self.memory_top_k)
        if not rows:
            return "我当前还没有记住什么长期信息。"
        lines = [f"- {r['content']}" for r in rows]
        return "我记住的信息：\n" + "\n".join(lines)

    def _build_user_content(
        self, context: str | None, plan: ExecutionPlan, user_input: str
    ) -> str:
        """把意图识别结果注入用户消息，帮助模型聚焦执行。"""
        intents_desc = (
            "、".join(f"{i.name}({i.confidence:.2f})" for i in plan.intents) or "未知"
        )
        hint = f"用户意图识别: {intents_desc}（风险等级: {plan.risk_level}）"
        if plan.slots:
            hint += f"；槽位: {json.dumps(plan.slots, ensure_ascii=False)}"
        if plan.processed and plan.processed != user_input:
            hint += f"；净化后输入: {plan.processed}"
        body = f"{hint}\n\n用户问题：{user_input}"
        return f"{context}\n\n{body}" if context else body

    # ---------- ReAct 三步骤 ----------

    def context_stats(self) -> dict:
        """统计当前上下文大小：消息数、字符数、UTF-8 字节数与估算 token 数。

        本项目用代理模型且未装 tiktoken，拿不到精确 tokenizer，
        这里按"中文约 0.6 token/字、其他字符约 0.25 token/字符"估算，仅供参考。
        """
        tool_json = json.dumps(TOOLS, ensure_ascii=False)
        messages_json = json.dumps(self.memory, ensure_ascii=False)
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

    def reason(self) -> object:
        """Reason：调模型。返回 ChatCompletionMessage。"""
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.memory,
                    tools=TOOLS,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message
            except Exception as e:
                status = getattr(e, "status_code", None)
                if attempt < self.max_retries and (
                    status == 429 or (status and status >= 500)
                ):
                    wait = 2**attempt
                    print(
                        f"  [retry] {status} 服务繁忙，{wait}s 后重试（{attempt + 1}/{self.max_retries}）"
                    )
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("重试次数用尽")

    def act(self, call) -> str:
        """Act：执行一个工具调用，返回观察结果。"""
        fn = call.function
        try:
            args = json.loads(fn.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        result = run_tool(fn.name, args)
        try:
            print(f"  [tool] {fn.name}({json.dumps(args, ensure_ascii=False)}) -> {result}")
        except UnicodeEncodeError:
            # Windows 控制台编码问题：截断输出
            print(f"  [tool] {fn.name}({json.dumps(args, ensure_ascii=False)}) -> [输出含无法显示字符]")
        return result

    def observe(self, call, result: str):
        """Observe：把工具调用与结果写回记忆。"""
        self.memory.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": result,
            }
        )

    # ---------- 主循环 ----------

    def run(self, user_input: str, max_rounds: int | None = None) -> str:
        self._turn_inputs.append(user_input)
        self._persisted_this_turn = False

        # ① 意图识别（预处理 + 规则前置 + 两阶段 LLM + 风险分级）
        history = self._recent_user_history()[:-1]
        plan = self.recognizer.recognize(
            user_input, history, self.user_id, self.session_id
        )
        print(
            f"  [intent] primary={plan.primary_intent} src={plan.source} "
            f"risk={plan.risk_level} amb={plan.ambiguous} blocked={plan.blocked}"
        )

        # ② 主动交互：缺失槽位 / 消歧 / 重述 → ask_user 回灌重识别
        plan = self._resolve_plan(plan, self._recent_user_history(), user_input)

        # ③ 高风险拦截
        if plan.blocked:
            return self._block_high_risk(plan)

        # ④ 中风险确认
        if plan.risk_level == "mid" and not self._confirm_mid_risk(plan):
            return "已取消本次操作。"

        # ⑤ 记忆查询短路：直查库，不调 LLM
        if plan.primary_intent == IntentNames.MEMORY_QUERY:
            return self._answer_memory_query()

        # ⑥ ReAct 循环（记忆写入回合结束时落库，标志防双写）
        question = plan.processed or user_input
        context = self._recall(question)
        self.memory.append(
            {
                "role": "user",
                "content": self._build_user_content(context, plan, question),
            }
        )
        max_rounds = max_rounds or self.max_rounds

        for _ in range(max_rounds):
            msg = self.reason()
            self.memory.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            )

            if not msg.tool_calls:
                answer = msg.content or "（无输出）"
                self._record(user_input, answer)
                if (
                    plan.primary_intent == IntentNames.MEMORY_WRITE
                    and not self._persisted_this_turn
                ):
                    self._persist(user_input)
                return answer

            for call in msg.tool_calls:
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
                print("用法: /intent <输入>，查看意图识别结果")
                continue
            plan = agent.recognizer.recognize(
                text, agent._recent_user_history(), agent.user_id, agent.session_id
            )
            print(f"意图: {[i.name for i in plan.intents]}")
            print(
                f"来源: {plan.source} | 置信度: {[i.confidence for i in plan.intents]}"
            )
            print(
                f"风险: {plan.risk_level} | 歧义: {plan.ambiguous} | 拦截: {plan.blocked}"
            )
            print(f"槽位: {plan.slots}")
            print(
                f"任务分组: {[(g.intent_ids, g.dependency) for g in plan.task_groups]}"
            )
            print(f"追问: {plan.ask_prompt}")
            print(f"净化后: {plan.processed}")
            continue
        try:
            answer = agent.run(question)
            print(f"Agent: {answer}")
        except Exception as e:
            print(f"Agent 错误: {e}")
