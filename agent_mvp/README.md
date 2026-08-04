# Agent MVP

手搓最小 AI Agent（MVP）：仅依赖 `openai` 库，不引入任何 Agent 框架，核心是一个 `while` 循环串联模型、工具、记忆三个模块。

## 项目结构

```
agent_mvp/
├── agent.py          # 全部实现（单文件）
├── pyproject.toml    # uv 项目配置
├── .env              # 环境变量（OPENAI_API_KEY 等）
└── README.md         # 本文件
```

## Agent 的本质：4 块积木

| 积木 | 实现 | 位置 |
|------|------|------|
| **模型** | `OpenAI` 客户端，决定输出内容和工具调用 | `agent.py` 的 `AgentMVP` |
| **工具** | function calling 的 JSON 格式描述 + `run_tool` 分发 | `agent.py` 的 `TOOLS` / `run_tool` |
| **记忆** | `messages` 列表记录对话历史、工具调用与结果 | `agent.py` 的 `AgentMVP.memory` |
| **循环** | `while` 循环判断：继续调工具还是直接输出答案 | `agent.py` 的 `AgentMVP.run` |

## 快速开始

```bash
cd agent_mvp

# 安装依赖（含 openai）
uv sync

# 配置环境变量（编辑 .env 填入 API Key / Base URL）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.example.com/v1   # 若填完整 /chat/completions 端点会自动裁剪

# 运行（交互式，/exit 退出）
uv run python agent.py
```

## 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | API 密钥（必填） |
| `OPENAI_BASE_URL` | API 地址，可选；传完整 `/chat/completions` 端点会自动裁剪为 SDK 所需 base URL |
| `LLM_MODEL` | 模型名，默认 `deepseek-ai/deepseek-v4-flash` |
| `BING_SEARCH_API_KEY` | Bing Web Search API 订阅密钥（联网搜索用，可选） |
| `TVLY_API_KEY` / `TAVILY_API_KEY` | Tavily AI Search API 密钥（联网搜索用，可选） |

## 内置工具

- **calculator(expression)** — 计算数学表达式，支持四则运算、括号及常用数学函数（`sqrt`、`log`、`sin` 等）
- **get_time()** — 获取当前日期和时间
- **read_file(path)** — 读取文本文件，超长内容自动截断
- **remember(content, importance)** — 把值得长期保留的用户信息写入长期记忆（是否调用由模型判断）
- **web_search(query, count)** — 联网搜索实时信息（Bing Web Search API v7，需配置 `BING_SEARCH_API_KEY`）
- **tavily_search(query, max_results, search_depth)** — 联网搜索实时信息（Tavily AI Search，专为 AI 优化，返回结构化结果与 AI 生成答案，需配置 `TVLY_API_KEY` 或 `TAVILY_API_KEY`）

## 防注入

`calculator` 用 `eval` 实现算数，但先对表达式做 AST 白名单校验：仅允许数字字面量、四则运算、括号和受限数学函数，`__import__`、`open`、属性访问等一律拦截。

```python
calculator("3 * (2 + 5)")                # -> 21
calculator("__import__('os').system('dir')")  # -> ValueError 被拦截
```

## 运行原理

```
用户输入 → 追加到 messages → 调模型（带 tools 参数）
  ├─ 模型返回 tool_calls → run_tool 执行 → 结果写回 messages → 继续循环
  └─ 模型无 tool_calls   → 直接输出答案，循环结束
```

## 扩展工具

三步接入一个新工具：

1. 在 `TOOLS` 里加 function calling 的 JSON schema
2. 写实现函数
3. 注册进 `TOOL_FUNCS`

```python
TOOLS.append({
    "type": "function",
    "function": {
        "name": "search",
        "description": "联网搜索",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
})

def search(query: str) -> str:
    return f"搜索结果: {query}"

TOOL_FUNCS["search"] = search
```

可替换为查数据库、读文件、RAG 检索等任意工具。
