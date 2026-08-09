"""天气工具：OpenWeatherMap 为主，wttr.in 免费兜底。"""

import json
import os
import urllib.parse
import urllib.request

from .location import city_from_ipapi

_OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
_OPENWEATHER_GEO_URL = "https://api.openweathermap.org/geo/1.0/direct"
openweather_api_key = os.environ.get("OPENWEATHER_API_KEY", "")


def set_openweather_key(key: str):
    """注入 OpenWeatherMap 订阅密钥（宿主从配置读取后调用，未注入则读环境变量）。"""
    global openweather_api_key
    openweather_api_key = key or openweather_api_key


def get_weather(city: str = "") -> str:
    """获取指定城市的实时天气；city 留空则用 IP 粗略定位当前城市。

    优先 OpenWeatherMap（需配置 OPENWEATHER_API_KEY），未配置则走 wttr.in，返回温度/体感/天气/湿度。
    """
    if not openweather_api_key:
        return _wttr_weather(city)
    try:
        target = city if city else city_from_ipapi()
        if not target:
            return "错误: 未指定城市且无法定位当前城市"
        geo = json.loads(
            urllib.request.urlopen(
                f"{_OPENWEATHER_GEO_URL}?q={urllib.parse.quote(target)}&limit=1&appid={openweather_api_key}",
                timeout=10,
            ).read().decode("utf-8")
        )
        if not geo:
            return f"未找到城市「{target}」"
        lat, lon = geo[0]["lat"], geo[0]["lon"]
        data = json.loads(
            urllib.request.urlopen(
                f"{_OPENWEATHER_URL}?lat={lat}&lon={lon}&appid={openweather_api_key}&units=metric&lang=zh_cn",
                timeout=10,
            ).read().decode("utf-8")
        )
        weather = data.get("weather", [{}])[0]
        main = data.get("main", {})
        return (
            f"{target} 当前天气（OpenWeatherMap）：\n"
            f"- 天气: {weather.get('description', '')}\n"
            f"- 温度: {main.get('temp', '?')}°C（体感 {main.get('feels_like', '?')}°C）\n"
            f"- 湿度: {main.get('humidity', '?')}%\n"
            f"- 风: {data.get('wind', {}).get('speed', '?')} m/s"
        )
    except Exception as e:
        return f"错误: 获取天气失败: {e}"


def _wttr_weather(city: str) -> str:
    """wttr.in 免费天气，无需 key。city 留空表示当前 IP 位置。"""
    try:
        url = "https://wttr.in/"
        if city:
            url += urllib.parse.quote(city)
        url += "?format=j1&lang=zh"
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        cur = data.get("current_condition", [{}])[0]
        area = data.get("nearest_area", [{}])[0]
        loc = (
            area.get("areaName", [{}])[0].get("value", city or "当前")
            if area
            else (city or "当前位置")
        )
        desc = cur.get("lang_zh", [{}])[0].get("value", "") if cur.get("lang_zh") else ""
        if not desc:
            desc = cur.get("weatherDesc", [{}])[0].get("value", "")
        return (
            f"{loc} 当前天气（wttr.in）：\n"
            f"- 天气: {desc}\n"
            f"- 温度: {cur.get('temp_C', '?')}°C（体感 {cur.get('FeelsLikeC', '?')}°C）\n"
            f"- 湿度: {cur.get('humidity', '?')}%\n"
            f"- 风: {cur.get('windspeedKmph', '?')} km/h（{cur.get('winddir16Point', '')}）"
        )
    except Exception as e:
        return f"错误: 获取天气失败: {e}"