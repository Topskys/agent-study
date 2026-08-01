"""最小 Agent 实现。

四大积木：
- 模型：OpenAI 兼容接口，决定输出内容与工具调用
- 工具：tools.py 中的 function calling JSON 描述 + run_tool 执行
- 记忆：messages 列表记录对话历史
- 循环：while 循环判断继续调工具还是直接输出答案

模块划分：
- tools.py   工具层：schema、实现、分发
- config.py  配置层：YAML 加载
- agent.py   核心：Agent 类与主循环
"""

import json
import os
import time

from openai import OpenAI

from config import DEFAULT_CONFIG_PATH, load_config
from tools import TOOLS, run_tool


def _call_with_retry(client: OpenAI, model: str, messages: list, max_retries: int = 3):
    """带指数退避的调用：处理 429/5xx（如服务过载）等瞬时错误。"""
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS
            )
        except Exception as e:
            status = getattr(e, "status_code", None)
            if attempt < max_retries and (status == 429 or (status and status >= 500)):
                wait = 2**attempt
                print(
                    f"  [retry] {status} 服务繁忙，{wait}s 后重试（{attempt + 1}/{max_retries}）"
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("重试次数用尽")


class Agent:
    """最小 Agent：记忆（self.memory）随实例存续，跨轮次记住对话。"""

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

    def reset(self):
        """清空记忆，开始全新会话。"""
        self.memory = [{"role": "system", "content": self.system_prompt}]

    def run(self, user_input: str, max_rounds: int | None = None) -> str:
        self.memory.append({"role": "user", "content": user_input})
        max_rounds = max_rounds or self.max_rounds

        for _ in range(max_rounds):
            resp = _call_with_retry(
                self.client, self.model, self.memory, max_retries=self.max_retries
            )
            msg = resp.choices[0].message
            self.memory.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                }
            )

            if not msg.tool_calls:
                return msg.content or "（无输出）"

            for call in msg.tool_calls:
                fn = call.function
                try:
                    args = json.loads(fn.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = run_tool(fn.name, args)
                self.memory.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": fn.name,
                        "content": result,
                    }
                )
                print(
                    f"  [tool] {fn.name}({json.dumps(args, ensure_ascii=False)}) -> {result}"
                )

        return "达到最大迭代轮数，未获得最终答案"


def run_agent(question: str, max_rounds: int = 10) -> str:
    """一次性调用（无状态），每次新建会话。需要跨轮次记忆请用 Agent。"""
    return Agent().run(question, max_rounds)


if __name__ == "__main__":
    agent = Agent()
    print("Agent MVP（/exit 退出；/reset 清空记忆）")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("/exit", "/quit"):
            break
        if question.strip().lower() in ("/reset", "/clear"):
            agent.reset()
            print("记忆已清空")
            continue
        print(f"Agent: {agent.run(question)}")
