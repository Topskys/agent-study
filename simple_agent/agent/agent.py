import json

from openai import OpenAI


class SimpleAgent:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/deepseek-v4-pro",
        base_url: str = "",
        temperature: float = 1,
        top_p: float = 0.95,
        max_tokens: int = 16384,
        stream: bool = False,
        timeout: int = 120,
        extra_body: dict | None = None,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.extra_body = extra_body
        self.messages = [{"role": "system", "content": "你是助手，可按需调用工具。"}]
        self.tools: dict = {}

    def register_tool(self, name: str, func: callable, description: str):
        self.tools[name] = {"func": func, "desc": description}

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        response = self._call_llm(self._build_kwargs())

        if self.stream:
            return self._run_stream(response)
        return self._run_sync(response)

    # ------------------------------------------------------------------
    # Internal: LLM call
    # ------------------------------------------------------------------

    def _build_kwargs(self) -> dict:
        kwargs = dict(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=self.stream,
        )
        tool_schema = self._build_tool_schema()
        if tool_schema:
            kwargs["tools"] = tool_schema
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        return kwargs

    def _call_llm(self, kwargs: dict):
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"API call failed: {e}") from e

    # ------------------------------------------------------------------
    # Internal: tool schema
    # ------------------------------------------------------------------

    def _build_tool_schema(self) -> list:
        return [
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

    # ------------------------------------------------------------------
    # Internal: stream path
    # ------------------------------------------------------------------

    def _run_stream(self, response) -> str:
        content, tool_calls_data = self._accumulate_stream(response)
        if not tool_calls_data:
            self.messages.append({"role": "assistant", "content": content})
            return content

        self._append_tool_messages(tool_calls_data)
        final = self._call_llm(self._build_kwargs())
        reply, _ = self._accumulate_stream(final)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _accumulate_stream(self, response) -> tuple:
        content = ""
        tool_calls = {}
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if delta.tool_calls:
                self._merge_tool_chunks(delta.tool_calls, tool_calls)
        return content, list(tool_calls.values()) if tool_calls else None

    @staticmethod
    def _merge_tool_chunks(tool_calls_delta, acc: dict):
        for tc in tool_calls_delta:
            idx = tc.index
            if idx not in acc:
                acc[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
            if tc.id:
                acc[idx]["id"] = tc.id
            if not tc.function:
                continue
            if tc.function.name:
                acc[idx]["function"]["name"] += tc.function.name
            if tc.function.arguments:
                acc[idx]["function"]["arguments"] += tc.function.arguments

    # ------------------------------------------------------------------
    # Internal: sync path
    # ------------------------------------------------------------------

    def _run_sync(self, response) -> str:
        msg = response.choices[0].message
        self.messages.append(msg)

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            result = self.tools[tc.function.name]["func"]()
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                }
            )

        final = self._call_llm(self._build_kwargs())
        reply = final.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _append_tool_messages(self, tool_calls_data: list):
        for tc in tool_calls_data:
            name = tc["function"]["name"]
            args = (
                json.loads(tc["function"]["arguments"])
                if tc["function"]["arguments"]
                else {}
            )
            result = self.tools[name]["func"](**args)
            self.messages.append(
                {"role": "assistant", "content": None, "tool_calls": [tc]}
            )
            self.messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": str(result)}
            )
