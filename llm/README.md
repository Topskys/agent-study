# llm-client

统一 LLM 接入层独立包（import 名 `llm_client`），供 agent / rag / memory 等模块复用。

解决现状痛点：多包重复"拼 prompt → 调模型 → 解析 → 重试"，无重试/超时兜底、无统一响应类型、无审计/缓存/任务路由，升级 SDK 或换 Provider 需改多处。

- 语言：Python ≥ 3.11
- 依赖：openai / pydantic / pyyaml / python-dotenv / requests
- 完整设计：见 `docs/LLM模块设计方案.md`

## 特性

- **统一三能力**：`chat` / `chat_stream` / `embed`，业务方只依赖 `llm_client.types`，不依赖具体 SDK 类型
- **双 Provider**：
  - `OpenAIProvider`：ZEN / NVIDIA / DeepSeek / Ollama / vLLM 等 OpenAI 兼容端点（`base_url` 指向 `/v1`，走 openai SDK）
  - `HTTPProvider`：`base_url` 已指向 `.../chat/completions` 的场景（requests 直连，自动补齐路径）
- **重试**：仅重试 `429 / 5xx` 与网络错误（`ConnectionError / TimeoutError / OSError`）；4xx 直接上抛；指数退避 + 随机抖动
- **缓存**：仅缓存确定性调用（`temperature == 0`、非流式、未声明 `no_cache`）；key = `sha256(messages + model + 采样参数 + tools)`；支持注入自定义 backend（Redis 等）
- **任务路由**：`task → model` 规则表，请求显式带 `model` 时优先
- **审计**：每次调用记录 task / model / 消息数 / 工具数 / 延迟 / token / 内容预览 / 错误，输出到标准库 `logging`（logger `llm_client.audit`）
- **零 SDK 约束保持**：`LLMClient` 用 `Protocol` 定义，其他包按类型注解依赖不强 import 本包（`intent` 约束不被破坏）

## 目录结构

```
llm/
├── pyproject.toml              # name="llm-client"，hatchling
├── llm_client/
│   ├── __init__.py             # 公共导出
│   ├── types.py                # Message / ToolCall / TokenUsage / LLMRequest / LLMResponse / Chunk
│   ├── base.py                 # LLMClient(Protocol) + LLMClientBase
│   ├── client.py               # Client（组合）+ build_client（便捷构造）
│   ├── openai_compat.py        # OpenAIProvider
│   ├── http.py                 # HTTPProvider
│   ├── embeddings.py           # EmbeddingsClient
│   ├── retry.py                # RetryPolicy / RetryConfig / 异常
│   ├── cache.py                # LLMCache
│   ├── router.py               # LLMRouter / RouteRule
│   ├── audit.py                # Audit / log_call
│   └── config.py               # LLMConfig / load_config
├── config.yaml.example         # 配置模板
├── .env.example                # 环境变量模板
├── tests/                      # 全部 mock，不真实调 API
└── README.md
```

详细类图见 `docs/LLM模块设计方案.md` §2.1。

## 安装

```bash
cd llm
uv sync --group dev        # 或 pip install -e .[dev]
```

## 快速开始

```python
from llm_client import build_client, LLMRequest, Message

client = build_client(
    api_key="...",
    base_url="https://.../v1",
    model="deepseek-v4-flash-free",
    max_retries=3,
)

resp = client.chat(
    LLMRequest(messages=[Message(role="user", content="你好")])
)
print(resp.content)
```

流式：

```python
for chunk in client.chat_stream(LLMRequest(messages=[Message(role="user", content="hi")])):
    print(chunk.content, end="")
```

Embedding：

```python
from llm_client import EmbeddingsClient
emb = EmbeddingsClient(api_key="...", base_url="...")
vec = emb.embed("text", model="text-embedding-3-small")
```

## 配置

两种方式：

**1. YAML + 环境变量（推荐）**——复制模板并填写：

```bash
cp config.yaml.example config.yaml
cp .env.example .env            # 填入 ZEN_API_KEY 等
```

```python
from llm_client import build_client, load_config

cfg = load_config("config.yaml")
client = build_client(
    api_key=cfg.api_key,
    base_url=cfg.base_url,
    model=cfg.model,
    provider=cfg.provider,
    max_retries=cfg.max_retries,
    enable_cache=cfg.enable_cache,
    default_task_models=cfg.task_models,
)
```

**2. 参数直传**（见"快速开始"）。

### 配置项

| 字段 | 说明 | 默认 |
|------|------|------|
| `api_key` | API 密钥，支持 `${ENV_VAR}` | — |
| `base_url` | openai_compat 指向 `/v1`；http 指向 `/chat/completions` | — |
| `provider` | `openai_compat` \| `http` | `openai_compat` |
| `model` | 默认模型 | — |
| `temperature` / `top_p` / `max_tokens` | 采样参数 | 0.7 / 0.95 / 2048 |
| `timeout` | 超时（秒） | 120 |
| `max_retries` | 重试次数 | 3 |
| `enable_cache` | 开启确定性缓存 | false |
| `models.default/chat/intent...` | 任务路由表 `task → model` | — |

## 组合编排

`Client.chat` 的调用顺序：

```
router（选模型）→ cache（命中即返）→ retry（重试）→ audit（审计）→ provider
```

`chat_stream` 与 `embed` 直接透传 provider（流式不缓存）。

## 常用约定

- 所有 Provider 的原始返回一律规整为 `LLMResponse`，业务方不依赖 SDK 对象
- 重试次数用尽抛 `RetryExhaustedError`；4xx 业务错误直接上抛（可被 `RetryPolicy.run` 捕获识别）
- 缓存仅对确定性请求生效；`extra={"no_cache": True}` 可显式跳过
- `intent` 包保持零 SDK 依赖：宿主侧通过注入回调转发到 `client.chat(task="intent")`

## 测试

全部 mock，不真实调 API：

```bash
cd llm && uv run pytest tests/ -q
```

覆盖：`HTTP/openai Provider` 的 chat / tools / stream / 错误抛出；重试成功 / 429 / 用尽 / 400 直抛；缓存仅确定性；路由优先级；`Client` 组合编排。

## 接入其他模块（预留点）

| 模块 | 现状 | 接入方式 |
|------|------|----------|
| `agent_mvp` | openai SDK 四处调用 | `__init__` 注入 `LLMClient`，`reason` / `_llm_*` 改为薄封装转发 |
| `intent` | 注入式回调（零 SDK，保持不变） | 宿主侧 4 个回调转发到 `client.chat(task="intent")` |
| `rag` | 裸 requests | `RAGGenerator` 注入 `llm_client`，`generate()` 换 `chat()` |
| `memory` | 本地哈希 / BGE | 可选接入 `EmbeddingsClient.embed()` |
| `simple_agent` | 裸 requests | `build_client(provider="http")` 直换 |