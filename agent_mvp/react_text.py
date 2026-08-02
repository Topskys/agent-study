"""经典文本协议 ReAct Agent。

ReAct 原文：模型以文本形式输出推理链——
    Thought: 当前推理
    Action: 要调用的工具名
    Action Input: 传给工具的参数
    或
    Final Answer: 最终答案

Agent 用正则解析这段文本，执行工具，把结果作为 Observation 写回，
再让模型继续推理，直到输出 Final Answer。不依赖原生 function calling。

模块划分：
- tools.py          工具层：实现、分发
- config.py         配置层：YAML 加载
- react_text.py     本文件：文本协议 ReAct 循环
"""

import json
import os
import re
import time

from openai import OpenAI

from config import DEFAULT_CONFIG_PATH, load_config
from tools import TOOLS, run_tool

REACT_PROMPT = """你是一个可以使用工具来解决问题的智能助手。

可用工具：
{tools_desc}

请严格按以下格式逐步推理：

Thought: 你当前的想法和推理
Action: 工具名（只能是上面列出的工具之一）
Action Input: 传给工具的 JSON 参数

如果已经得到最终答案，则输出：

Thought: 我已获得足够信息
Final Answer: 你的最终答案

注意：
- 每一步只输出一个 Thought 和 Action
- Action Input 必须是 JSON 格式
- 不调用工具时直接输出 Final Answer
"""


def build_react_prompt() -> str:
    tools_desc = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}" for t in TOOLS
    )
    return REACT_PROMPT.format(tools_desc=tools_desc)


def parse_react_output(reply: str) -> dict:
    """解析模型输出，返回 {type, thought, action, action_input}。

    type: "final"（输出 Final Answer）| "action"（要调工具）| "invalid"
    """
    result = {"type": "invalid", "thought": "", "action": "", "action_input": ""}

    fa_match = re.search(r"Final Answer:\s*(.+)", reply, re.DOTALL)
    if fa_match:
        result["type"] = "final"
        thought = re.search(r"Thought:\s*(.+?)(?=Final Answer:|\Z)", reply, re.DOTALL)
        result["thought"] = thought.group(1).strip() if thought else ""
        result["action_input"] = fa_match.group(1).strip()
        return result

    thought = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", reply, re.DOTALL)
    action = re.search(r"Action:\s*(\w+)", reply)
    action_input = re.search(r"Action Input:\s*(\{.*\})", reply, re.DOTALL)

    if not (thought and action):
        return result

    result["type"] = "action"
    result["thought"] = thought.group(1).strip()
    result["action"] = action.group(1).strip()
    if action_input:
        result["action_input"] = action_input.group(1).strip()
    return result

    thought = re.search(r"Thought:\s*(.+)", reply, re.DOTALL)
    action = re.search(r"Action:\s*(\w+)", reply)
    action_input = re.search(r"Action Input:\s*(\{.*\})", reply, re.DOTALL)

    if not (thought and action):
        return result

    result["type"] = "action"
    result["thought"] = thought.group(1).strip()
    result["action"] = action.group(1).strip()
    if action_input:
        result["action_input"] = action_input.group(1).strip()
    return result


class ReActAgent:
    """文本协议 ReAct：Thought → Action → Observation 循环。"""

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
        self.memory: list = [{"role": "system", "content": build_react_prompt()}]

    def reset(self):
        self.memory = [{"role": "system", "content": build_react_prompt()}]

    def _call_llm(self) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.memory,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
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

    def execute_action(self, action: str, action_input: str) -> str:
        """执行工具：优先解析 JSON 参数，失败则按字符串传给工具。"""
        try:
            args = json.loads(action_input) if action_input else {}
            if isinstance(args, dict):
                return run_tool(action, args)
        except json.JSONDecodeError:
            pass
        return run_tool(
            action,
            {"path": action_input}
            if action == "read_file"
            else {"expression": action_input},
        )

    def run(self, user_input: str, max_rounds: int | None = None) -> str:
        self.memory.append({"role": "user", "content": user_input})
        max_rounds = max_rounds or self.max_rounds

        for _ in range(max_rounds):
            reply = self._call_llm()
            self.memory.append({"role": "assistant", "content": reply})
            parsed = parse_react_output(reply)

            if parsed["type"] == "final":
                return parsed["action_input"]

            if parsed["type"] == "action":
                observation = self.execute_action(
                    parsed["action"], parsed["action_input"]
                )
                print(
                    f"  [react] {parsed['action']}({parsed['action_input']}) -> {observation}"
                )
                self.memory.append(
                    {"role": "user", "content": f"Observation: {observation}"}
                )
                continue

            # 无法解析：把原文给模型看，要求重新格式化
            self.memory.append(
                {
                    "role": "user",
                    "content": "输出格式不正确，请严格按 Thought/Action/Action Input 或 Final Answer 格式重新回答。",
                }
            )

        return "达到最大迭代轮数，未获得最终答案"


def run_react_agent(question: str, max_rounds: int = 10) -> str:
    return ReActAgent().run(question, max_rounds)


if __name__ == "__main__":
    agent = ReActAgent()
    print("ReAct Text Agent（/exit 退出；/reset 清空记忆）")
    while True:
        question = input("You: ")
        if question.strip().lower() in ("/exit", "/quit"):
            break
        if question.strip().lower() in ("/reset", "/clear"):
            agent.reset()
            print("记忆已清空")
            continue
        print(f"Agent: {agent.run(question)}")
