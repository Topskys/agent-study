import json
import requests


class SimpleAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/deepseek-v4-pro",
        base_url: str = "",
        temperature: float = 1,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        timeout: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.messages = [{"role": "system", "content": "你是助手，可按需调用工具。"}]
        self.tools: dict = {}

    def register_tool(self, name: str, func: callable, description: str):
        self.tools[name] = {"func": func, "desc": description}

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        return self._call()

    def _build_payload(self) -> dict:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info["desc"],
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
                for name, info in self.tools.items()
            ]
        return payload

    def _call(self) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url, headers=headers, json=self._build_payload(), timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        self.messages.append(msg)

        if not msg.get("tool_calls"):
            return msg.get("content", "")

        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = (
                json.loads(tc["function"]["arguments"])
                if tc["function"]["arguments"]
                else {}
            )
            result = self.tools[name]["func"](**args)
            self.messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": str(result)}
            )

        return self._call()
