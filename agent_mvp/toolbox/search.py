"""联网搜索工具：Bing Web Search API v7 + Tavily AI Search。"""

import json
import os
import urllib.parse
import urllib.request

_BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
_bing_search_api_key = os.environ.get("BING_SEARCH_API_KEY", "")

_tavily_search_api_key = os.environ.get("TVLY_API_KEY", "") or os.environ.get("TAVILY_API_KEY", "")


def set_bing_search_key(key: str):
    """注入 Bing 订阅密钥（宿主从配置读取后调用，未注入则读环境变量）。"""
    global _bing_search_api_key
    _bing_search_api_key = key or _bing_search_api_key


def set_tavily_search_key(key: str):
    """注入 Tavily 订阅密钥（宿主从配置读取后调用，未注入则读环境变量）。"""
    global _tavily_search_api_key
    _tavily_search_api_key = key or _tavily_search_api_key


def web_search(query: str, count: int = 5) -> str:
    """联网搜索：Bing Web Search API v7，返回标题 / URL / 摘要。"""
    if not _bing_search_api_key:
        return "错误: 未配置 BING_SEARCH_API_KEY，联网搜索不可用"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "count": min(max(int(count), 1), 20),
            "mkt": "zh-CN",
            "responseFilter": "Webpages",
        }
    )
    req = urllib.request.Request(
        f"{_BING_SEARCH_URL}?{params}",
        headers={"Ocp-Apim-Subscription-Key": _bing_search_api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"错误: 联网搜索失败: {e}"
    pages = data.get("webPages", {}).get("value", [])
    if not pages:
        return f"未搜索到「{query}」的相关结果。"
    lines = []
    for i, p in enumerate(pages[: int(count)], 1):
        lines.append(
            f"{i}. {p.get('name', '')}\n   {p.get('url', '')}\n   {p.get('snippet', '')}"
        )
    return "搜索结果：\n\n" + "\n\n".join(lines)


def tavily_search(query: str, max_results: int = 5, search_depth: str = "basic") -> str:
    """联网搜索：Tavily AI Search API，返回标题 / URL / 内容摘要。

    Args:
        query: 搜索关键词
        max_results: 返回结果条数，默认 5
        search_depth: 搜索深度，"basic" 或 "advanced"，默认 "basic"
    """
    if not _tavily_search_api_key:
        return "错误: 未配置 TVLY_API_KEY 或 TAVILY_API_KEY，Tavily 搜索不可用"
    try:
        from tavily import TavilyClient
    except ImportError:
        return "错误: 未安装 tavily-python，请运行 `uv add tavily-python`"

    client = TavilyClient(api_key=_tavily_search_api_key)
    try:
        response = client.search(
            query=query,
            max_results=min(max(int(max_results), 1), 20),
            search_depth=search_depth,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception as e:
        return f"错误: Tavily 搜索失败: {e}"

    results = response.get("results", [])
    if not results:
        return f"未搜索到「{query}」的相关结果。"

    lines = []
    for i, r in enumerate(results[: int(max_results)], 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"{i}. {title}\n   {url}\n   {content}")

    answer = response.get("answer", "")
    output = "搜索结果：\n\n" + "\n\n".join(lines)
    if answer:
        output = f"AI 答案：{answer}\n\n" + output
    return output