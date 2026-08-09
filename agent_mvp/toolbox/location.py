"""位置工具：公网 IP 粗略定位。"""

import json
import urllib.request


def get_location() -> str:
    """用公网 IP 粗略定位：返回城市 / 省份 / 国家 / 时区。无 key，免费。"""
    try:
        req = urllib.request.Request(
            "https://ipapi.co/json/",
            headers={"User-Agent": "curl/8.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("status") == "fail":
            return "错误: 无法定位当前位置"
        city = data.get("city", "")
        region = data.get("regionName", "")
        country = data.get("country", "")
        timezone = data.get("timezone", "")
        return f"位置: {city}{region}{country}（时区 {timezone}）".strip()
    except Exception as e:
        return f"错误: 获取位置失败: {e}"


def city_from_ipapi() -> str:
    """IP 定位取城市名，供 get_weather 兜底。"""
    try:
        req = urllib.request.Request(
            "https://ip-api.com/json/",
            headers={"User-Agent": "curl/8.0"},
        )
        data = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        return data.get("city", "") or ""
    except Exception:
        return ""