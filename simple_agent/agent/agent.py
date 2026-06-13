import json
import re
import requests


def build_system_prompt(tools: dict) -> str:
    tools_desc = "\n".join(
        f"- {name}: {info['desc']}" + (f" ({info['args']})" if info["args"] else "")
        for name, info in tools.items()
    )
    return f"""You are a helpful assistant with access to these tools:

{tools_desc}

Respond in this exact format:

Thought: your reasoning
Action: tool name
Action Input: input to the tool

When you have the final answer:

Final Answer: your answer"""


def call_llm(
    api_key: str,
    base_url: str,
    model: str,
    messages: list,
    temperature: float = 1,
    top_p: float = 0.95,
    max_tokens: int = 16384,
    timeout: int = 120,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_response(reply: str) -> dict:
    result = {"type": "continue", "thought": "", "action": "", "action_input": ""}

    fa_match = re.search(r"Final Answer:\s*(.+)", reply, re.DOTALL)
    if fa_match:
        result["type"] = "final"
        result["action_input"] = fa_match.group(1).strip()
        return result

    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", reply, re.DOTALL)
    action_match = re.search(r"Action:\s*(\w+)", reply)
    input_match = re.search(r"Action Input:\s*(.+)", reply, re.DOTALL)

    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    if action_match:
        result["action"] = action_match.group(1).strip()
    if input_match:
        text = input_match.group(1).strip()
        for stop in ("\nThought:", "\nFinal Answer:"):
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]
        result["action_input"] = text.strip()

    return result


def execute_action(action: str, action_input: str, tools: dict) -> str:
    if action not in tools:
        return f"Unknown tool: {action}"
    try:
        func = tools[action]["func"]
        try:
            args = json.loads(action_input)
            if isinstance(args, dict):
                return str(func(**args))
        except (json.JSONDecodeError, TypeError):
            pass
        return str(func(action_input))
    except Exception as e:
        return f"Error: {e}"


class SimpleAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/deepseek-v4-flash",
        base_url: str = "",
        temperature: float = 1,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.tools: dict = {}

    def register_tool(
        self, name: str, func: callable, description: str, args_desc: str = ""
    ):
        self.tools[name] = {"func": func, "desc": description, "args": args_desc}

    def run(self, user_input: str) -> str:
        messages = [{"role": "system", "content": build_system_prompt(self.tools)}]
        messages.append({"role": "user", "content": user_input})

        for _ in range(10):
            reply = call_llm(
                self.api_key,
                self.base_url,
                self.model,
                messages,
                self.temperature,
                self.top_p,
                self.max_tokens,
                self.timeout,
            )
            messages.append({"role": "assistant", "content": reply})
            parsed = parse_response(reply)
            if parsed["type"] == "final":
                return parsed["action_input"]

            if not parsed["action"]:
                return reply

            observation = execute_action(
                parsed["action"], parsed["action_input"], self.tools
            )
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "Max iterations reached"
