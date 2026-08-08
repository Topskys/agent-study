"""OpenAIProvider 单元测试：mock SDK，验证规整为统一响应结构。"""

from unittest import mock

from llm_client.openai_compat import OpenAIProvider
from llm_client.types import LLMRequest, Message


def _fake_completion(content=None, tool_calls=None, finish_reason="stop", model="m"):
    msg = mock.Mock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = mock.Mock()
    choice.message = msg
    choice.finish_reason = finish_reason
    cc = mock.Mock()
    cc.choices = [choice]
    cc.model = model
    usage = mock.Mock()
    usage.prompt_tokens = 3
    usage.completion_tokens = 2
    usage.total_tokens = 5
    cc.usage = usage
    return cc


class TestOpenAIProvider:
    def test_chat_basic(self):
        provider = OpenAIProvider(
            api_key="k", base_url="https://x.example/v1", model="m"
        )
        client = mock.Mock()
        client.chat.completions.create.return_value = _fake_completion(content="你好")
        provider._client = client

        resp = provider.chat(LLMRequest(messages=[Message(role="user", content="hi")]))
        assert resp.content == "你好"
        assert resp.usage.total_tokens == 5
        assert resp.model == "m"

    def test_chat_with_tool_calls(self):
        provider = OpenAIProvider(
            api_key="k", base_url="https://x.example/v1", model="m"
        )
        tc = mock.Mock()
        tc.id = "call_1"
        tc.function.name = "search"
        tc.function.arguments = '{"q": "北京"}'
        client = mock.Mock()
        client.chat.completions.create.return_value = _fake_completion(
            content=None, tool_calls=[tc], finish_reason="tool_calls"
        )
        provider._client = client

        resp = provider.chat(
            LLMRequest(
                messages=[Message(role="user", content="天气")],
                tools=[{"type": "function", "function": {"name": "search"}}],
            )
        )
        assert resp.content is None
        assert resp.tool_calls and resp.tool_calls[0].name == "search"
        # tools 透传给 SDK
        call_kwargs = client.chat.completions.create.call_args[1]
        assert "tools" in call_kwargs

    def test_stream(self):
        provider = OpenAIProvider(
            api_key="k", base_url="https://x.example/v1", model="m"
        )
        chunk = mock.Mock()
        delta = mock.Mock()
        delta.content = "北"
        delta.tool_calls = None
        choice = mock.Mock()
        choice.delta = delta
        choice.finish_reason = None
        chunk.choices = [choice]
        client = mock.Mock()
        client.chat.completions.create.return_value = iter([chunk])
        provider._client = client

        chunks = list(
            provider.chat_stream(
                LLMRequest(messages=[Message(role="user", content="hi")])
            )
        )
        assert "".join(c.content for c in chunks) == "北"
