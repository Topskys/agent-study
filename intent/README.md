# intent-recognizer

意图识别独立模块（v3 两阶段多意图识别）。作为 agent_mvp 的依赖子模块接入，提供：

- **两阶段 LLM 识别**：阶段一多意图识别（置信度打分 + 槽位完备性校验）→ 阶段二批量槽位抽取 + 参数正则校验
- **纯自研编排**：不绑定 LangGraph / LangChain，不 import 任何 LLM SDK，LLM 能力全部注入式回调
- **主动交互**：缺失槽位聚合一次问全、0.6~0.9 消歧确认、低置信重新输入、高危拦截
- **规则硬信号短路**：记忆写入 / 记忆查询 / 闲聊 / 问答 / 工具线索等高确定性意图不调 LLM
- **并行/串行调度**：多意图按依赖分组，组内并行、组间串行
- **数据访问层直连现有记忆库**：复用 agent_memory.db 的 memory_items / user_profiles / event_stream / kv_items，不新建会话表

架构设计见 `docs/Agent意图识别设计方案v3.md`（§5 类图、§8 数据表对接、§10 技术栈）。

## 架构总览

```
┌────────────────────────────── IntentRecognizer（门面编排） ──────────────────────────────┐
│                                                                                            │
│  user_input + history                                                                       │
│      │                                                                                      │
│      ▼                                                                                      │
│  TextPreprocessService   ① 预处理：错别字修正 → 代词消解 → 短提问扩写(llm_expand)             │
│      │                                                                                      │
│      ▼                                                                                      │
│  RuleCheckService        ② 规则前置：高危拦截 / 系统级硬信号短路（不调 LLM）                   │
│      │                                                                                      │
│      ▼                                                                                      │
│  FirstStageIntentService ③ 阶段一：llm_recognize 多意图识别（重试2次→规则关键词兜底）          │
│      │                      + 低置信(<0.6)过滤 + 槽位缓存合并 + 完备性判定                     │
│      ▼                                                                                      │
│  置信度三档分发            >0.9 可执行 / 0.6~0.9 消歧反问 / 无意图 请重新输入                  │
│      ▼                                                                                      │
│  AskPromptService         ④ 未完备 → 缺失槽位聚合一次问全（ask_user）                          │
│      ▼                                                                                      │
│  IntentDependService     ⑤ 依赖解析：文本含条件/先后词且多意图 → 串行，否则并行                 │
│      ▼                                                                                      │
│  SecondStageSlotService  ⑥ 阶段二：llm_extract_slots 批量抽槽 + SlotMeta.regex 校验            │
│      ▼                                                                                      │
│  TaskScheduleService     ⑦ 调度：组内 ThreadPoolExecutor 并行，组间按依赖串行                   │
│      │                        （executor 回调执行，不注入则仅返回 pending）                    │
│      ▼                                                                                      │
│  ExecutionPlan            输出：intents / slots / task_groups / risk / ask_prompt / …         │
│                                                                                            │
│  MemoryStore / EventStore / KvStore / ProfileStore  直连 agent_memory.db 现有表             │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
intent/
├── pyproject.toml                    # 包 intent-recognizer，dep memory-system，dev pytest
├── intent_recognizer/
│   ├── __init__.py                   # 公共导出（IntentRecognizer / ExecutionPlan / …）
│   ├── models.py                     # 全部数据模型（IntentNames / IntentMeta / ExecutionPlan / …）
│   ├── config.py                     # ConfigManager：resources JSON + kv_items 动态覆盖
│   ├── protocols.py                  # 注入式回调类型定义（协议契约）
│   ├── llm_json.py                   # LLM 输出宽松 JSON 解析 + 带超时调用
│   ├── preprocess.py                 # TextPreprocessService（+ Preprocessor 兼容别名）
│   ├── rule.py                       # RuleCheckService：高危拦截 + 系统级硬信号 + 风险分级
│   ├── first_stage.py                # FirstStageIntentService：阶段一
│   ├── ask_prompt.py                 # AskPromptService：聚合追问 / 消歧话术
│   ├── depend.py                     # IntentDependService：并行/串行依赖分组
│   ├── second_stage.py               # SecondStageSlotService：阶段二
│   ├── scheduler.py                  # TaskScheduleService：并行/串行调度
│   ├── stores.py                     # MemoryStore / ProfileStore / EventStore / KvStore
│   ├── recognizer.py                 # IntentRecognizer：全流程门面（recognize / recognize_debug）
│   └── resources/
│       ├── intent_config.json        # 意图定义 + 槽位 + 阈值 + 高危清单
│       └── business_vocab.json       # 业务词库（错别字修正 / 代词消解用）
└── tests/
    └── test_intent.py                # 48 个单元/集成测试
```

## 快速开始

```bash
cd intent
uv sync                          # 安装依赖（memory-system 为 editable 本地路径）
uv run pytest tests -q           # 运行测试（48 passed）
```

## 使用示例

```python
from intent_recognizer import IntentRecognizer

# ① 注入 LLM 回调（宿主负责调模型，本包不 import LLM SDK）
def llm_recognize(prompt: str, history: list[str]) -> str:
    # 阶段一：把 prompt 交给模型，返回含意图数组的原始文本
    return '[]'  # 或 '[{"intent_id":"bill_query","name":"账单查询","confidence":0.95,...}]'

def llm_extract_slots(prompt: str, history: list[str], intent_ids: list[str]) -> str:
    # 阶段二：返回槽位数组
    return '[]'  # 或 '[{"intent_id":"bill_query","slot_info":{"start_time":"2025-07-01"}}]'

def ask_user(prompt: str, timeout: float) -> str | None:
    # 追问 / 消歧：返回用户回复；超时/取消返回 None
    return input(prompt)

# ② 构建识别器（db_path 指向 agent_memory.db；None 时数据层退化为空操作）
recognizer = IntentRecognizer(
    db_path="agent_memory.db",
    llm_recognize=llm_recognize,
    llm_extract_slots=llm_extract_slots,
    ask_user=ask_user,
    llm_expand=None,               # 可选：短提问扩写
    high_risk_keywords=None,       # 可选：覆盖默认高危清单
)

# ③ 识别一次输入
plan = recognizer.recognize("查一下7月账单", history=[], user_id="u1", session_id="s1")

plan.primary_intent    # 主意图 id（无则 None）
plan.intents           # [IntentRecognizeItem(name, confidence, complete, miss_slots)]
plan.slots             # {intent_id: {slot_key: value}}
plan.task_groups       # [TaskGroup(group_id, dependency, intent_ids)]
plan.risk_level        # low / mid / high
plan.blocked           # 高危拦截标记
plan.ambiguous         # 需要主动交互（追问/消歧/重述）
plan.ask_prompt        # 主动交互话术（缺失槽位聚合 / 消歧确认 / 重述 / 拦截说明）
plan.source            # llm / rule / none
plan.execution_results # 调度执行结果 {intent_id: result}
```

> `ask_prompt` 是统一承载所有主动交互话术的字段：缺失槽位聚合追问、0.6~0.9 消歧确认、低置信重述、高危拦截说明。宿主拿到 `ambiguous=True` 且有 `ask_prompt` 时应追问用户并把回复作为新一轮输入重新 `recognize`（已填槽位经 slot_cache 自动累积，不会重复追问）。

### 识别流程与分发规则

| 情形 | 返回计划 |
| --- | --- |
| 命中高危关键词 | `blocked=True, risk_level="high"`，`ask_prompt` 为拦截话术 |
| 规则硬信号（记住/记忆查询/问候/命令/工具线索） | `source="rule"`，不调 LLM |
| 阶段一无意图 / 全部低置信被过滤 | `ambiguous=True`，`ask_prompt` 请用户重新输入 |
| 存在置信度 ≤ 0.9 的意图 | `ambiguous=True`，`ask_prompt` 消歧确认 |
| 槽位未完备 | `ambiguous=True`，`ask_prompt` 聚合缺失字段一次问全 |
| 阶段二参数格式非法 | `ambiguous=True`，`ask_prompt` 重新收集 |
| 完备且高置信（>0.9） | 依赖解析 → 抽槽 → 调度执行，返回完整计划 |

## 注入式回调协议

| 回调 | 签名 | 说明 |
| --- | --- | --- |
| `llm_recognize` | `(prompt: str, history: list[str]) -> str` | 阶段一多意图识别，期望返回意图 JSON 数组 |
| `llm_extract_slots` | `(prompt, history, intent_ids: list[str]) -> str` | 阶段二批量槽位抽取，期望返回槽位 JSON 数组 |
| `ask_user` | `(prompt: str, timeout: float) -> str | None` | 追问 / 消歧；超时或取消返回 None |
| `llm_expand` | `(text: str, history: list[str]) -> str | None` | 短提问扩写（可选，不注入则不扩写） |
| `executor` | `(intent_id: str, slot_kv: dict) -> Any` | 任务执行（可选，不注入则调度仅返回 pending） |

阶段一 / 阶段二返回的 JSON 数组需符合：

```jsonc
// 阶段一（intent_config.json 里已注册的 intent_id / name）
[{"intent_id": "bill_query", "name": "账单查询", "confidence": 0.95, "complete": true, "miss_slots": []}]

// 阶段二
[{"intent_id": "bill_query", "slot_info": {"start_time": "2025-07-01", "bill_type": "消费"}}]
```

解析走 `llm_json.py` 的宽松模式（容忍 markdown 代码块、前后缀文字、尾逗号、单双引号混用）；阶段一解析失败自动重试 2 次，仍失败降级为规则关键词兜底。

## 配置

### resources/intent_config.json

内置 4 个业务意图：

| intent_id | 名称 | 必填槽位 |
| --- | --- | --- |
| `phone_recharge` | 手机话费充值 | `phone_number`（`^1[0-9]{10}$`）、`recharge_amount` |
| `bill_query` | 账单查询 | `start_time`（可选 `end_time` / `bill_type`） |
| `account_balance` | 账户余额查询 | 无（可选 `card_no`） |
| `send_email` | 发送邮件 | `recipient`（邮箱正则）、`subject` |

系统级意图（`IntentNames`，由规则硬信号输出）：`chat` / `question` / `tool_use` / `memory_write` / `memory_query` / `command`。

阈值默认值：`confidence=0.6`（低置信，阶段一直接丢弃）、`high=0.9`（高置信，可直接执行）。

### kv_items 动态覆盖

`ConfigManager` 可选注入 `KvStore`，支持运行时从 kv_items 覆盖阈值与高危清单（键约定 `intent:thresholds` / `intent:high_risk_keywords`），用于后台可配置。

## 数据访问层

数据访问层直连现有记忆库（agent_memory.db），不新建会话 / 检查点表：

| Store | 表 | 用途 |
| --- | --- | --- |
| `MemoryStore` | `memory_items` | 读 long_term / session；写意图检查点 `intent_cache` 与槽位缓存 `slot_cache`（memory_id 形如 `intent_cache:{user_id}:{session_id}`） |
| `ProfileStore` | `user_profiles` | 读用户画像（供消歧 / 槽位默认值） |
| `EventStore` | `event_stream` | 审计事件：`intent_recognized` / `ask_prompt` / `slot_extracted` / `task_scheduled` |
| `KvStore` | `kv_items` | 阈值 / 高危清单动态覆盖（支持 TTL） |

约定：

- **长期记忆写入不绕过记忆治理**：本层只直接写内部检查点与审计事件；长期记忆仍由宿主走 memory-system 的 `UserMemory.add_long_term_memory` 治理裁决
- **建表安全**：目标库缺表时各 Store 用 `CREATE TABLE IF NOT EXISTS` 补齐同构表结构，对现有库无副作用
- **`db_path=None`**：所有读写退化为空操作（返回空值，不报错），便于无库场景使用

## 开发与测试

```bash
uv run pytest tests -q       # 48 passed（全量）
uv run pytest tests/test_intent.py -k "first_stage or second_stage"   # 按模块筛选
```

测试覆盖：预处理三子模块、规则确定性、阶段一多意图解析/置信度过滤/规则兜底、阶段二抽槽与正则校验、置信度三档分发、聚合追问、槽位缓存合并、依赖串行/并行、数据访问层、`recognize_debug` 结构。

## 接入 agent_mvp

`agent_mvp/agent.py` 的 `Agent` 通过注入回调接入本包：

```python
from intent_recognizer import IntentRecognizer

self.recognizer = IntentRecognizer(
    db_path=self.memory_system.vector_store.db_path,   # 与记忆层共用同一个 agent_memory.db
    llm_recognize=llm_recognize or self._llm_recognize,
    llm_extract_slots=llm_extract_slots or self._llm_extract_slots,
    ask_user=ask_user or self._ask_user,
    llm_expand=llm_expand or self._llm_expand,
    high_risk_keywords=safety_cfg.get("high_risk_keywords"),
)
```

宿主在 REPL 场景注入 `ask_user`（`input()`）走追问闭环；API 场景不注入则 `ambiguous` 计划直接交由上层处理。

## 参考

- `docs/Agent意图识别设计方案v3.md`：v3 方案设计（类图 / 两阶段数据结构 / 数据表对接 / 技术栈）
- `memory/`：memory-system 包（嵌入、记忆治理、user_profiles 读写）
- `agent_mvp/agent.py`：宿主接入示例与 ReAct 主循环
