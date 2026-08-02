"""function calling + Thought 的 ReAct Agent。

保留原生 function calling（tools 参数），同时要求模型在每次工具调用前，
把推理过程写到 assistant 消息的 content 字段（Thought），以便观察决策链。

ReAct 循环：
- reason：调模型，拿到 assistant 消息（content=Thought，tool_calls=Action）
- act：执行工具调用
- observe：把工具结果写回记忆

模块划分：
- tools.py          工具层：实现、分发
- config.py         配置层：YAML 加载
- react_thought.py  本文件：function calling + Thought 循环
"""

import json
import os
import time

from openai import OpenAI

from config import DEFAULT_CONFIG_PATH, load_config
from tools import TOOLS, run_tool

THOUGHT_PROMPT = """你是一个可以使用工具来解决问题的智能助手。

规则：
1. 在调用工具之前，必须先把推理过程写到响应的 content 字段，格式为：
   Thought: 你的推理（为什么要调用这个工具、打算怎么做）
2. 然后通过 tool_calls 发起工具调用
3. 看到工具结果后，如果还需要更多信息，继续重复上面两步；
   如果已得到答案，直接输出最终答案（可带简短推理）。
"""


class ThoughtAgent:
    """function calling + Thought：模型显式输出推理后调工具。"""

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

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.memory: list = [{"role": "system", "content": THOUGHT_PROMPT}]

    def reset(self):
        self.memory = [{"role": "system", "content": THOUGHT_PROMPT}]

    def reason(self) -> object:
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

    def run(self, user_input: str, max_rounds: int | None = None) -> str:
        self.memory.append({"role": "user", "content": user_input})
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

            if msg.content:
                print(f"  [thought] {msg.content[:200]}")

            if not msg.tool_calls:
                return msg.content or "（无输出）"

            for call in msg.tool_calls:
                fn = call.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(fn.name, args)
                print(
                    f"  [tool] {fn.name}({json.dumps(args, ensure_ascii=False)}) -> {result}"
                )
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn.name,
                        "content": result,
                    }
                )

        return "达到最大迭代轮数，未获得最终答案"


def run_thought_agent(question: str, max_rounds: int = 10) -> str:
    return ThoughtAgent().run(question, max_rounds)


if __name__ == "__main__":
    agent = ThoughtAgent()
    print("Thought Agent（/exit 退出；/reset 清空记忆）")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("/exit", "/quit"):
            break
        if question.strip().lower() in ("/reset", "/clear"):
            agent.reset()
            print("记忆已清空")
            continue
        print(f"Agent: {agent.run(question)}")
