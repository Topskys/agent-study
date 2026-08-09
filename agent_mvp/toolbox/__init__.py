"""工具箱子包：按领域拆分工具实现，供 tools.py 门面统一注册与分发。"""

from .basic import MAX_READ_CHARS, get_time, read_file
from .calc import calculator
from .location import get_location
from .remember import remember, set_memory_persist_hook
from .search import (
    set_bing_search_key,
    set_tavily_search_key,
    tavily_search,
    web_search,
)
from .weather import get_weather, set_openweather_key

__all__ = [
    "MAX_READ_CHARS",
    "calculator",
    "get_location",
    "get_time",
    "get_weather",
    "read_file",
    "remember",
    "set_bing_search_key",
    "set_memory_persist_hook",
    "set_openweather_key",
    "set_tavily_search_key",
    "tavily_search",
    "web_search",
]