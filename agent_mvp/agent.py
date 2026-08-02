"""ReAct Agent（function calling 版），集成三层记忆。

ReAct = Reason + Act + Observe 的循环：
- reason：调模型，让它决定下一步（输出答案或发起工具调用）
- act：模型发起工具调用时，执行工具
- observe：把工具结果写回记忆，进入下一轮推理

记忆能力（memory-system 包）：
- 自动记录：每轮把用户输入与助手回答写入工作记忆与会话记忆
- 相关回忆：推理前按当前输入召回历史记忆，注入上下文
- 沉淀：reset / 新会话时，end_session 将会话记忆沉淀进长期记忆
- 立即持久化（双通道，先关键词后模型）：
  1. 关键词启发式：用户输入含"记住/记得/别忘"等，直接落库
  2. 模型自主：未命中关键词时，由模型判断是否调用 remember 工具

模块划分：
- tool_schemas.json  工具 schema
- tools.py           工具层：实现、分发
- config.py          配置层：YAML 加载
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
from memory_system.core.memory_system import MemorySystem
from tools import TOOLS, run_tool, set_memory_persist_hook


class Agent:
    """ReAct Agent：reason → act → observe 循环，接入三层记忆系统。"""

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | os.PathLike = DEFAULT_CONFIG_PATH,
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
        db_path = str(mem.get("db_path", "agent_memory.db"))
        if not os.path.isabs(db_path):
            db_path = str(Path(__file__).resolve().parent / db_path)
        self.memory_system = MemorySystem(db_path=db_path)
        self.user_memory = self.memory_system.get_user_memory(
            str(mem.get("user_id", "default_user"))
        )
        self.session_id = str(uuid.uuid4())
        self.user_memory.start_session(self.session_id)
        self._persisted_this_turn = False
        set_memory_persist_hook(self._remember_long_term)

    def reset(self):
        """清空对话与工作记忆，把当前会话沉淀进长期记忆，开启新会话。"""
        self.user_memory.end_session()
        self.user_memory.flush_working_memory()
        self.session_id = str(uuid.uuid4())
        self.user_memory.start_session(self.session_id)
        self.memory = [{"role": "system", "content": self.system_prompt}]

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

    # ---------- 立即持久化：关键词启发式（优先），未命中再由模型 remember 工具兜底 ----------
    # 明确要求"记住某信息"的祈使句式
    _PERSIST_IMPERATIVES = (
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
    _PERSIST_QUERY_MARKERS = (
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

    def _wants_persist(self, user_input: str) -> bool:
        """关键词启发式：判断输入是否明确要求记住某些信息。

        规则：
        1. 询问是否记住 / 要求回忆的疑问句（如"你记住了吗"），一律不算持久化意图；
        2. 命中明确祈使句式（请记住/帮我记住/别忘了…）才算；
        3. 裸"记住"仅当句子不是疑问句结尾时视为祈使。
        """
        text = user_input.strip()
        if not text:
            return False
        if any(m in text for m in self._PERSIST_QUERY_MARKERS):
            return False
        if any(k in text for k in self._PERSIST_IMPERATIVES):
            return True
        if "记住" in text:
            return not text.endswith(("吗", "么", "呢", "？", "?", "没", "没有"))
        return False

    def _record(self, user_input: str, answer: str):
        """把一轮问答写入工作记忆与会话记忆。

        会话记忆仅在 reset 时沉淀到长期记忆；若用户明确要求"记住"，
        则由 run() 额外调用 _persist 立即落库。
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

    # ---------- ReAct 三步骤 ----------

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
        print(f"  [tool] {fn.name}({json.dumps(args, ensure_ascii=False)}) -> {result}")
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
        self._persisted_this_turn = False
        context = self._recall(user_input)
        user_content = f"{context}\n\n用户问题：{user_input}" if context else user_input
        self.memory.append({"role": "user", "content": user_content})
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
                if self._wants_persist(user_input) and not self._persisted_this_turn:
                    self._persist(user_input)
                return answer

            for call in msg.tool_calls:
                result = self.act(call)
                self.observe(call, result)

        self._record(user_input, "达到最大迭代轮数，未获得最终答案")
        if self._wants_persist(user_input) and not self._persisted_this_turn:
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
    print("Agent MVP（/exit 退出；/reset 清空对话并沉淀记忆）")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("/exit", "/quit"):
            break
        if question.strip().lower() in ("/reset", "/clear"):
            agent.reset()
            print("记忆已沉淀，对话已清空")
            continue
        print(f"Agent: {agent.run(question)}")
