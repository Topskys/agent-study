"""工具层单测：注册、分发、web_search 无 key 容错，tavily_search 同理。"""

import tools
from toolbox import search as tb_search


def test_web_search_registered_in_schema():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "web_search" in names
    assert "web_search" in tools.TOOL_FUNC_MAP


def test_web_search_without_key_returns_hint(monkeypatch):
    monkeypatch.setattr(tb_search, "_bing_search_api_key", "")
    result = tools.run_tool("web_search", {"query": "测试"})
    assert "BING_SEARCH_API_KEY" in result


def test_web_search_with_key_skips_missing_key_branch(monkeypatch):
    monkeypatch.setattr(tb_search, "_bing_search_api_key", "fake-key")
    result = tools.run_tool("web_search", {"query": "测试"})
    assert "未配置 BING_SEARCH_API_KEY" not in result


def test_set_bing_search_key_injects():
    tools.set_bing_search_key("key-from-config")
    assert tb_search._bing_search_api_key == "key-from-config"
    tools.set_bing_search_key("")  # 空值不覆盖，保留已注入的 key
    assert tb_search._bing_search_api_key == "key-from-config"
    tools.set_bing_search_key("")


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
    monkeypatch.setattr(tb_search, "_tavily_search_api_key", "")
    result = tools.run_tool("tavily_search", {"query": "测试"})
    assert "TVLY_API_KEY" in result or "TAVILY_API_KEY" in result


def test_tavily_search_with_key_skips_missing_key_branch(monkeypatch):
    monkeypatch.setattr(tb_search, "_tavily_search_api_key", "fake-key")
    result = tools.run_tool("tavily_search", {"query": "测试"})
    assert "未配置" not in result or "TVLY_API_KEY" not in result


def test_set_tavily_search_key_injects():
    tools.set_tavily_search_key("key-from-config")
    assert tb_search._tavily_search_api_key == "key-from-config"
    tools.set_tavily_search_key("")  # 空值不覆盖，保留已注入的 key
    assert tb_search._tavily_search_api_key == "key-from-config"
    tools.set_tavily_search_key("")


def test_get_weather_registered_in_schema():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "get_weather" in names
    assert "get_weather" in tools.TOOL_FUNC_MAP


def test_get_location_registered_in_schema():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "get_location" in names
    assert "get_location" in tools.TOOL_FUNC_MAP


def test_get_weather_fallback_error_on_network_failure(monkeypatch):
    """未配 key 走 wttr.in，网络失败时返回错误提示而非抛异常。"""
    from toolbox import weather as tb_weather

    monkeypatch.setattr(tb_weather, "openweather_api_key", "")
    monkeypatch.setattr(
        tb_weather.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("net down"))
    )
    result = tools.run_tool("get_weather", {"city": "北京"})
    assert "获取天气失败" in result


def test_get_location_returns_error_on_network_failure(monkeypatch):
    from toolbox import location as tb_location

    monkeypatch.setattr(
        tb_location.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("net down"))
    )
    result = tools.run_tool("get_location", {})
    assert "获取位置失败" in result


def test_get_weather_openweather_path(monkeypatch):
    """配 key 时走 OpenWeatherMap geocoding → weather。"""
    from toolbox import weather as tb_weather
    import json as _json

    monkeypatch.setattr(tb_weather, "openweather_api_key", "fake-key")

    def fake_urlopen(url, timeout=10):
        url = str(url)
        if "geo/1.0" in url:
            body = _json.dumps([{"lat": 39.9, "lon": 116.4}])
        else:
            body = _json.dumps(
                {
                    "weather": [{"description": "晴"}],
                    "main": {"temp": 22.5, "feels_like": 21.0, "humidity": 40},
                    "wind": {"speed": 2.5},
                }
            )
        import io

        return io.BytesIO(body.encode("utf-8"))

    monkeypatch.setattr(tb_weather.urllib.request, "urlopen", fake_urlopen)
    result = tools.run_tool("get_weather", {"city": "北京"})
    assert "北京" in result
    assert "22.5" in result
    assert "晴" in result
