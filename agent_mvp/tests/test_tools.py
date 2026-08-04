"""工具层单测：注册、分发、web_search 无 key 容错，tavily_search 同理。"""

import tools


def test_web_search_registered_in_schema():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "web_search" in names
    assert "web_search" in tools.TOOL_FUNC_MAP


def test_web_search_without_key_returns_hint(monkeypatch):
    monkeypatch.setattr(tools, "_bing_search_api_key", "")
    result = tools.run_tool("web_search", {"query": "测试"})
    assert "BING_SEARCH_API_KEY" in result


def test_web_search_with_key_skips_missing_key_branch(monkeypatch):
    monkeypatch.setattr(tools, "_bing_search_api_key", "fake-key")
    result = tools.run_tool("web_search", {"query": "测试"})
    assert "未配置 BING_SEARCH_API_KEY" not in result


def test_set_bing_search_key_injects():
    tools.set_bing_search_key("key-from-config")
    assert tools._bing_search_api_key == "key-from-config"
    tools.set_bing_search_key("")  # 空值不覆盖，保留已注入的 key
    assert tools._bing_search_api_key == "key-from-config"
    tools._bing_search_api_key = ""


def test_run_tool_unknown():
    assert "未知工具" in tools.run_tool("nope", {})


def test_calculator_safe():
    assert tools.run_tool("calculator", {"expression": "3 * (2 + 5)"}) == "21"
    assert "非法" in tools.run_tool("calculator", {"expression": "__import__('os')"})


def test_remember_without_hook():
    assert "记忆系统未接入" in tools.run_tool("remember", {"content": "x"})


def test_tavily_search_registered_in_schema():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "tavily_search" in names
    assert "tavily_search" in tools.TOOL_FUNC_MAP


def test_tavily_search_without_key_returns_hint(monkeypatch):
    monkeypatch.setattr(tools, "_tavily_search_api_key", "")
    result = tools.run_tool("tavily_search", {"query": "测试"})
    assert "TVLY_API_KEY" in result or "TAVILY_API_KEY" in result


def test_tavily_search_with_key_skips_missing_key_branch(monkeypatch):
    monkeypatch.setattr(tools, "_tavily_search_api_key", "fake-key")
    result = tools.run_tool("tavily_search", {"query": "测试"})
    assert "未配置" not in result or "TVLY_API_KEY" not in result


def test_set_tavily_search_key_injects():
    tools.set_tavily_search_key("key-from-config")
    assert tools._tavily_search_api_key == "key-from-config"
    tools.set_tavily_search_key("")  # 空值不覆盖，保留已注入的 key
    assert tools._tavily_search_api_key == "key-from-config"
    tools._tavily_search_api_key = ""
