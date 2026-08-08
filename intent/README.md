# intent 意图识别模块

作为 agent_mvp 的依赖子模块接入，包含两个识别包（不同方案版本）：

| 包 | 定位 | 架构 | 设计文档 | 状态 |
| --- | --- | --- | --- | --- |
| `intent_funnel` | 三层漏斗意图识别（低成本优先） | 规则 → 轻量语义 → LLM 兜底 | `docs/Agent意图识别设计方案v3.md` | 推荐 |
| `intent_recognizer` | 两阶段 LLM 多意图识别 | 阶段一识别 → 阶段二批量抽槽 | `docs/Agent意图识别设计方案v2.md` | 并存 |

两者都由以下共同原则约束：
- **纯自研编排**：不绑定 LangGraph / LangChain，不 import 任何 LLM SDK，LLM 能力全部注入式回调
- **主动交互闭环**：缺失槽位聚合一次问全、0.6~0.9 消歧确认、低置信重新输入、高危拦截
- **数据访问层直连记忆库**：复用 agent_memory.db 现有表，不新建会话表

---

# 包一：intent_funnel（三层漏斗，V3 方案）

低成本优先的三层串行漏斗：**RuleMatcher（规则）→ SemanticReasoner（轻量语义）→ ComplexIntentParser（LLM 兜底）**，
三层输出完全同构，调度器无需感知下层实现。规则扛高频、小模型扛主力、大模型扛长尾，极致平衡**成本 / 延迟 / 准确率**。

## 核心流转（分层阈值固定）

```
用户输入 → 预处理/敏感词过滤 → RuleMatcher 命中即返回（置信 1.0）
                              → 未命中 → SemanticReasoner（分类+向量双路融合）
                                           → maxConf≥0.9 返回
                                           → <0.9 下放 → ComplexIntentParser（LLM）
                                                     → ≥0.9 执行 / 0.6~0.9 消歧 / <0.6 无意图
```

四分支交互兜底（任意一层输出收敛到同一套分支）：
1. `maxConf ≥ 阈值` 且槽位完备 → 直接执行
2. `needAskSlots=true` 槽位缺失 → **一次聚合追问**（AskPromptBuilder 跨意图汇聚）
3. `needDisambiguate=true`（0.6~0.9）→ 消歧反问
4. `noValidIntent=true`（<0.6）→ 请用户重新输入

### 会话状态与槽位缓存

- `DialogStateTracker` 维护会话快照：`filled_slots` / `miss_slots` / `historical_queries` / `checkpoint`
- 识别前回填已确认槽位，避免重复询问；识别后落盘已确认槽位与检查点
- **多实例场景**用 `SessionStore` 接 SQLite 持久化（`db_path=None` 退化为内存），保证横向扩展不丢状态
- 槽位合法性校验：`EntityItem.valid` 经正则/枚举校验，非法值转入追问不直接执行

## 快速开始

```bash
cd intent
uv sync                          # 安装依赖（memory-system 为 editable 本地路径）
uv run pytest tests -q           # 运行测试（64 passed）
uv run pytest tests/test_funnel.py -q   # 仅漏斗包
```

## 使用示例

```python
from intent_funnel import FunnelIntentRecognition, SessionStore

store = SessionStore()                       # 内存存储；传 db_path=... 启用 SQLite 持久化
recognizer = FunnelIntentRecognition(store=store)

# 三层符合：规则命中直接返回；未命中走语义层；语义含混走 LLM
res = recognizer.recognize("给13800138000充值50元话费", session_id="s1")

res.source_layer      # rule_matcher / semantic_reasoner / complex_parser
res.primary_intent     # 主意图（无则 None）
res.intents            # [IntentItem(name, confidence, priority, entities, complete, miss_slots)]
res.intent_ranking     # [IntentRankingItem(name, confidence)]
res.blocked            # 高危拦截标记
res.block_reason       # 拦截原因
res.need_ask_slots     # 缺槽位需聚合追问
res.ask_slots          # 缺失必填槽位列表（一次问全）
res.ask_prompt         # 追问 / 消歧 / 重述 统一话术字段
res.need_disambiguate  # 0.6~0.9 消歧
res.no_valid_intent    # <0.6 无意图
```

### 语义层 / LLM 层能力注入（宿主职责）

```python
from intent_funnel import FunnelIntentRecognition, SessionStore

def classifier(text: str, history: list[str]) -> dict[str, float]:
    return {"bill_query": 0.93, "account_balance": 0.05}   # 轻量分类模型

def similarity(text: str, intent_name: str) -> float:
    return 0.9 if intent_name == "bill_query" else 0.0     # Embedding 向量相似度

def llm_gen(prompt: str) -> str:
    return '[{"intent":"bill_query","confidence":0.95,"slot_info":{"start_time":"2025-07-01"}}]'

recognizer = FunnelIntentRecognition(
    store=SessionStore(),
    classifier=classifier,
    similarity=similarity,
    llm_gen=llm_gen,   # LLM 解析（可选，不注入则语义含混时降级为 no_valid）
    llm_timeout=3.0,   # LLM 硬超时（秒），防阻塞
)
```

`llm_gen` 返回 JSON 数组，每个元素：`{"intent": 意图名, "confidence": 0~1, "slot_info": {键: 值}}`。
解析容忍 markdown 代码块 / 尾逗号 / 单双引号混用（`llm_json.py` 宽松解析），失败自动重试 2 次。

## 配置

`intent_funnel/resources/config.json` 定义意图 / 槽位 / 阈值 / 高危词 / 工具 Schema：

| 字段 | 说明 |
| --- | --- |
| `intents` | 意图定义（name / keywords / required_slots / 槽位 regex / 别名） |
| `thresholds.semantic_high` | 语义层可信阈值（默认 0.9） |
| `thresholds.llm_high` | LLM 高置信阈值（默认 0.9） |
| `thresholds.llm_low` | LLM 低置信阈值（默认 0.6） |
| `high_risk_keywords` | 高危词清单，命中直接拦截不进语义与 LLM |
| `tool_schemas` | FunctionCall 工具 Schema（注入 ComplexIntentParser Prompt） |

内置 5 个业务意图：`phone_recharge`（充值）、`bill_query`（账单）、`account_balance`（余额）、
`book_flight`（订机票，带目的地别名归一化 北京→PEK）、`send_email`（邮件，邮箱正校验）。

## 目录结构

```
intent_funnel/
├── __init__.py            # 公共导出（FunnelIntentRecognition / SessionStore / 数据模型）
├── models.py              # 三层同构统一 JSON 模型（IntentResult / IntentItem / EntityItem / DialogSessionState）
├── config.py              # FunnelConfig + IntentSpec / SlotSpec（resources 加载）
├── preprocess.py          # 轻量清洗：全角转半角 / 空白归一化
├── rule_matcher.py        # 第一层：关键词/正则/别名命中 + 高危硬拦截 + 静态槽位抽取
├── semantic.py            # 第二层：分类 + 向量相似度双路融合打分
├── complex_parser.py      # 第三层：LLM 解析 FunctionCall、重试 2 次、槽位正则校验、未知意图过滤
├── llm_json.py            # LLM 输出宽松 JSON 解析 + 带超时调用
├── dialog.py              # SlotCompletenessChecker / AskPromptBuilder / DialogStateTracker / IntentOrchestrator
├── stores.py              # SessionStore：内存或 SQLite 持久化（会话快照）
├── funnel.py              # FunnelIntentRecognition 门面：三层路由 + 四分支收敛 + 缓存落盘
└── resources/config.json   # 业务配置（意图/槽位/阈值/高危词/工具）
```

> 注：`funnel.py` 顶层调度（`FunnelIntentRecognition.recognize`）实现 V3 方案 §七 伪代码，
> `recognize()` 四分支收敛与 `DialogStateTracker` 更新逻辑对齐统一 JSON 模型。

---

# 包二：intent_recognizer（两阶段 LLM，V2）

提供两阶段多意图识别：**阶段一 LLM 多意图识别（置信度打分 + 槽位完备性校验）→ 阶段二批量槽位抽取 + 参数正则校验**。

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
│  ExecutionPlan           输出：intents / slots / task_groups / risk / ask_prompt / …         │
│                                                                                            │
│  MemoryStore / ProfileStore / EventStore / KvStore  直连 agent_memory.db 现有表             │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 快速开始

```python
from intent_recognizer import IntentRecognizer

def llm_recognize(prompt: str, history: list[str]) -> str:
    return '[{"intent_id": "bill_query", "name": "账单查询", "confidence": 0.95, ...}]'

def llm_extract_slots(prompt: str, history: list[str], intent_ids: list[str]) -> str:
    return '[{"intent_id": "bill_query", "slot": {"start_time": "2025-07-01"}}]'

def ask_user(prompt: str, timeout: float) -> str | None:
    return input(prompt)

recognizer = IntentRecognizer(
    db_path="agent_memory.db",    # 指向 agent_memory.db；None 时数据层退化为空操作
    llm_recognize=llm_recognize,
    llm_extract_slots=llm_extract_slots,
    ask_user=ask_user,
    llm_expand=None,               # 可选：短提问扩写
)

plan = recognizer.recognize("查一下7月账单", history=[], user_id="u1", session_id="s1")
plan.source           # llm / rule / none
plan.primary_intent   # 主意图 id（无则 None）
plan.intents          # [IntentData(name, confidence, complete, miss_slots)]
plan.slots            # {intent_id: {slot_key: value}}
plan.task_groups      # [TaskGroup(group_id, dependency, intent_ids)]
plan.blocked          # 高危拦截标记
plan.ambiguous        # 需要主动交互（追问 / 消歧 / 重述）
plan.ask_prompt       # 统一承载主动交互话术
```

> `ask_prompt` 是两包统一的"主动交互话术"出口：聚合追问、消歧确认、重述、拦截说明。
> 宿主拿到 `ambiguous=True` / `need_ask_slots=True` 且有 `ask_prompt` 时，应追问用户并把回复作为新一轮输入重新进入识别（已填槽位经 slot_cache / 会话缓存自动累积，不会重复询问）。

## 注入式回调协议（两包通用设计）

| 回调 | 签名 | 说明 |
| --- | --- | --- |
| `classifier`(语义层) | `(text: str, history: list[str]) -> dict` | 轻量分类模型，返回意图概率 |
| `similarity`(语义层) | `(text: str, intent_name: str) -> float` | Embedding 向量相似度 |
| `llm_gen`(漏斗 LLM 层) | `(prompt: str) -> str` | 返回意图 JSON 数组 |
| `llm_recognize`(两阶段) | `(prompt: str, history: list[str]) -> str` | 阶段一多意图识别 |
| `llm_extract_slots`(两阶段) | `(prompt, history, intent_ids: list[str]) -> str` | 阶段二批量槽位抽取 |
| `ask_user` | `(prompt: str, timeout: float) -> str | None` | 追问 / 消歧；超时或取消返回 None |

各类返回 JSON 均走宽松解析（容忍 markdown 代码块、前后缀文字、尾逗号、单双引号混用）；
解析失败自动重试 2 次 → 仍失败降级为规则层兜底。

## 配置与数据访问层

- `resources/intent_config.json`：内置 `phone_recharge` / `bill_query` / `account_balance` / `send_email` 场景，含正则、枚举，支持 `kv_items` 动态覆盖
- 数据访问：`MemoryStore`(memory_items) 写意图检查点 `intent_cache` 与槽位缓存 `slot_cache`；`ProfileStore`(user_profiles) 画像；`EventStore`(event_stream) 审计；`KvStore`(kv_items) 动态配置
- 长期记忆写入不绕过治理：本层只写内部检查点与审计，长期记忆仍由宿主走 memory-system 的 `UserMemory.add_long_term_memory` 治理裁决

## 目录结构

```
intent_recognizer/
├── __init__.py            # 公共导出（IntentRecognizer / ExecutionPlan / Store 等）
├── config.py              # ConfigManager：resources JSON + kv_items 动态覆盖
├── protocols.py           # 注入式回调类型定义（协议契约）
├── models.py              # 数据模型（IntentMeta / ExecutionPlan / TaskGroup / …）
├── preprocess.py          # TextPreprocessService：错别字 / 代词消解 / 扩写
├── rule.py                # RuleCheckService：高危拦截 + 系统级硬信号 + 风险分级
├── first_stage.py         # FirstStageIntentService：阶段一（多意图 + 置信过滤）
├── second_stage.py        # SecondStageSlotService：阶段二（批量抽槽 + 正则校验）
├── ask_prompt.py          # AskPromptService：聚合追问 / 消歧话术
├── depend.py              # IntentDependService：并行/串行依赖分组
├── scheduler.py           # TaskScheduleService：组内并行、组间串行调度
├── stores.py              # MemoryStore / ProfileStore / EventStore / KvStore
├── recognizer.py          # IntentRecognizer：全流程门面（recognize / recognize_debug）
└── resources/
    ├── intent_config.json # 意图定义 + 槽位 + 阈值 + 高危清单
    └── business_vocab.json   # 业务词库（错别字 / 代词消解用）
```

---

## 如何选择

| 场景 | 推荐包 |
| --- | --- |
| 需要极低 LLM 调用成本、高频标准化意图优先 | `intent_funnel`（推荐） |
| 已有 v2 两阶段、语义长尾需求，微调代价最小 | `intent_recognizer` |

## 参考

- `docs/Agent意图识别设计方案v3.md`：v3 三层漏斗方案（类图 / 四分支 / 会话状态）
- `docs/Agent意图识别设计方案v2.md`：v2 两阶段方案（类图 / 两阶段数据结构 / 数据表对接 / 技术栈）
- `memory/`：memory-system 包（嵌入、记忆治理、user_profiles 读写）
- `agent_mvp/agent.py`：宿主接入示例与 ReAct 主循环