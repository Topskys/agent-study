# Simple Agent

一个极简的 LLM 对话 Agent，支持工具调用、流式输出，通过环境变量配置。

## 项目结构

```
simple_agent/
├── agent/                  # 核心包
│   ├── __init__.py         # 包入口，导出 SimpleAgent / get_weather / get_time
│   ├── agent.py            # SimpleAgent 核心逻辑
│   └── tools.py            # 内置工具函数
├── main.py                 # 启动入口
├── pyproject.toml           # uv 项目配置
├── requirements.txt         # 依赖清单
├── .env                     # 环境变量
└── README.md                # 本文件
```

## 核心架构

```
┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ 用户输入 │ → │ LLM 推理 │ → │ 工具调用  │ → │ 结果整合  │ → 输出
└─────────┘   └──────────┘   └──────────┘   └──────────┘
```

组件职责：

| 组件 | 文件 | 职责 |
|------|------|------|
| **SimpleAgent** | `agent/agent.py` | 管理对话状态、调度 LLM 与工具 |
| **工具集** | `agent/tools.py` | 可注册的任意 Python 函数 |
| **启动入口** | `main.py` | 读取 `.env`，组装 Agent 并运行 |

## 快速开始

```bash
cd simple_agent

# 安装依赖
uv sync

# 配置环境变量（编辑 .env 填入 API Key）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.example.com/v1

# 运行
uv run python main.py
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | `""` | API 密钥 |
| `OPENAI_BASE_URL` | `""` | API 地址 |
| `LLM_MODEL` | `deepseek-ai/deepseek-v4-pro` | 模型名 |
| `LLM_TEMPERATURE` | `1` | 采样温度 |
| `LLM_TOP_P` | `0.95` | 核采样 |
| `LLM_MAX_TOKENS` | `16384` | 最大输出 Token |
| `LLM_STREAM` | `false` | 是否流式输出 |
| `LLM_THINKING` | `false` | 是否启用思考模式 |

## 代码示例

```python
from agent import SimpleAgent, get_weather, get_time

agent = SimpleAgent(api_key="sk-xxx", base_url="...")
agent.register_tool("get_weather", get_weather, "查询当前天气")
agent.register_tool("get_time", get_time, "获取当前时间")

print(agent.run("今天天气怎么样？"))
```

## 自定义工具

任何 Python 函数都可注册为工具：

```python
def my_tool(param: str) -> str:
    return f"处理结果: {param}"

agent.register_tool("my_tool", my_tool, "工具描述")
```

## 设计要点

- **工具即函数** — 无需继承或实现接口，普通函数即可注册
- **无状态框架** — 对话上下文由 `messages` 列表维护，可随时存取
- **零侵入扩展** — 新增工具只需一行 `register_tool`，核心逻辑无需改动
