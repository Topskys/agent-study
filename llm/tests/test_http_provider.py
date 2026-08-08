"""HTTPProvider 单元测试：mock requests.post，不真实调 API。"""

import json
from unittest import mock

import pytest
from llm_client.http import HTTPProvider
from llm_client.types import LLMRequest, Message


def _fake_response(payload: dict):
    resp = mock.Mock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestHTTPProvider:
    def test_chat_basic(self):
        provider = HTTPProvider(api_key="k", base_url="https://x.example/v1", model="m")
        payload = {
            "choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}],
            "model": "m",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        with mock.patch("requests.post", return_value=_fake_response(payload)) as m:
            resp = provider.chat(
                LLMRequest(messages=[Message(role="user", content="hi")])
            )
        assert resp.content == "你好"
        assert resp.usage.total_tokens == 5
        # base_url 未以 /chat/completions 结尾 → 自动补齐
        m.assert_called_once()
        assert m.call_args[0][0] == "https://x.example/v1/chat/completions"

    def test_chat_with_tools(self):
        provider = HTTPProvider(
            api_key="k", base_url="https://x.example/v1/chat/completions", model="m"
        )
        tool_call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"q": "北京"}'},
        }
        payload = {
            "choices": [
                {
                    "message": {"content": None, "tool_calls": [tool_call]},
                    "finish_reason": "tool_calls",
                }
            ],
            "model": "m",
            "usage": {},
        }
        with mock.patch("requests.post", return_value=_fake_response(payload)) as m:
            resp = provider.chat(
                LLMRequest(
                    messages=[Message(role="user", content="天气")],
                    tools=[{"type": "function", "function": {"name": "search"}}],
                )
            )
        assert resp.content is None
        assert resp.tool_calls and resp.tool_calls[0].name == "search"
        assert json.loads(resp.tool_calls[0].arguments) == {"q": "北京"}
        sent = m.call_args[1]["json"]
        assert "tools" in sent
        assert sent["model"] == "m"

    def test_chat_stream(self):
        provider = HTTPProvider(api_key="k", base_url="https://x.example/v1", model="m")
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": "北"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "京"}}]}),
            "data: [DONE]",
        ]
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.iter_lines.return_value = lines
        with mock.patch("requests.post", return_value=resp):
            chunks = list(
                provider.chat_stream(
                    LLMRequest(
                        messages=[Message(role="user", content="hi")], stream=True
                    )
                )
            )
        assert "".join(c.content for c in chunks) == "北京"


class TestHTTPProviderErrors:
    def test_non_200_raises(self):
        provider = HTTPProvider(api_key="k", base_url="https://x.example/v1", model="m")
        resp = mock.Mock()
        resp.raise_for_status.side_effect = RuntimeError("HTTP 500")
        with (
            mock.patch("requests.post", return_value=resp),
            pytest.raises(RuntimeError),
        ):
            provider.chat(LLMRequest(messages=[Message(role="user", content="hi")]))
