# 2026 RAG 全景：从大模型基座到 Agent 记忆中枢——万字长文吃透全栈落地

> 来源：腾讯云开发者社区
> 作者：烟雨平生
> 原文：https://cloud.tencent.com.cn/developer/article/2654878

**这不是一篇给你讲概念的文章。** 这是一份让你看完就能动手，少走半年弯路的实战指南。

## 为什么你必须搞懂 RAG

2023 年是大模型"百模大战"年，所有人都在刷榜单、比参数。2024 年起，战场转移了——**谁能把大模型真正用起来，谁才有价值。** 而检索增强生成（RAG，Retrieval-Augmented Generation），就是这场"应用落地战"里最核心的武器。

不夸张地说：**没有 RAG 打底，一切 AI 应用都是 PPT。**

文章结构如下：
1. RAG 是什么，为什么需要它
2. RAG 技术的发展迭代历程
3. 落地时如何做技术选型
4. 业界当前的经典实践
5. RAG 未来的发展方向
6. 从零到一的 RAG 实战落地路径

---

## 第一章：RAG 是什么，为什么需要它？

### 1.1 从一个真实的痛点说起

你公司买了 GPT-4 API 权限，花了两周做了一个"企业智能客服"——把公司所有产品文档喂进去，用户提问，AI 作答。演示很完美。上线第一天，用户来问："你们最新出的 Pro 版本，和去年的 Basic 版本相比，具体差在哪里？" AI 答得头头是道。可你看完之后发现——**它在瞎说**。

因为 GPT-4 根本不知道你们公司存在，更不知道你们有什么产品。它给出的答案完全是根据训练数据"编"出来的。

这就是**大模型的两大致命缺陷**：

**① 知识截止（Knowledge Cutoff）**：所有大模型都有训练截止日期。GPT-4 的训练数据截止到某个时间点，之后发生的事情它一概不知。

**② 幻觉（Hallucination）**：大模型在"一本正经地胡说八道"。大模型是在海量数据上训练出来的玩"文字接龙"的概率预测机器。当它被问到不知道的事情时，不会说"我不知道"，而是会"合情合理地编造"一个听起来像真的答案。

**那能不能把知识喂进去训练？** 理论上可以，但：
- 重新微调一个大模型，费用从几万到几百万不等
- 你的文档每天都在更新，不可能每次更新都去重训
- 训练完的知识"固化"在权重里，之后依然存在知识截止问题

**RAG 就是来解决这两个问题的。**

### 1.2 RAG 的核心思路

RAG 的核心思路用一句话概括：**在让大模型作答之前，先去外部知识库找到相关信息，然后把这些信息连同问题一起交给大模型。**

用生活化的比喻来说：你去参加一场开卷考试，不需要把所有知识背进脑子里——你只需要知道去哪里找，以及如何把找到的内容用在答案上。

RAG 全称 **Retrieval-Augmented Generation**，直译是"检索增强生成"，三个词对应三个步骤：
1. **Retrieval 检索** → 去知识库里找相关文档片段
2. **Augmentation 增强** → 把找到的内容拼到 Prompt 里
3. **Generation 生成** → 大模型根据上下文生成答案

### 1.3 RAG 解决了什么，没解决什么

**RAG 解决的问题：**
- ✅ 知识时效性：外部知识库随时可更新，不需要重训模型
- ✅ 幻觉抑制：答案有"依据"可查，减少无依据编造
- ✅ 私有知识接入：企业内部文档、专有数据可安全接入
- ✅ 可追溯性：答案可以附上来源链接，用户可自行核实
- ✅ 成本可控：无需重训大模型，只需维护知识库

**RAG 没有解决的问题：**
- ❌ 复杂推理：需要多步逻辑推导的问题，基础 RAG 依然力不从心
- ❌ 极致实时性：入库、索引构建存在一定延迟
- ❌ 跨文档关联推理：基础 RAG 效果较差

---

## 第二章：RAG 技术的发展迭代

### 2.1 第一代：概念诞生（2020 年）

RAG 这个词最早由 Facebook AI Research 在 2020 年的论文《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》里明确提出。检索器和生成器是一个整体，用联合训练的方式来优化。主要问题：训练成本高、需要标注数据、无法直接使用现成大模型。

### 2.2 第二代：范式确立（2022–2023 年）

ChatGPT 爆火后，大量企业迫切需要把大模型用起来。更务实的 RAG 范式出现：**不做联合训练，直接用 Prompt Engineering 把检索结果塞进上下文。** 检索器和生成器变成松散耦合的两个独立组件。LangChain、LlamaIndex 等框架让"5 分钟搭一个 RAG demo"成为可能。但很快大家发现：**Demo 效果好，生产效果差。**

### 2.3 第三代：Advanced RAG（2023–2024 年）

核心问题出在三个环节：
- **检索前（Pre-Retrieval）**：用户提问质量差、歧义表达
- **检索中（During Retrieval）**：Chunk 策略不当、纯向量检索对精确匹配词效果差
- **检索后（Post-Retrieval）**：召回内容过多淹没关键信息、无质量过滤

**Pre-Retrieval 优化：**
- Query Rewriting（查询改写）：用大模型把模糊问题改写成检索友好格式
- Query Expansion（查询扩展）：一个问题扩展成多个角度子问题
- HyDE（假设文档嵌入）：先让大模型"假设"一个答案，用假设答案去检索

**During Retrieval 优化：**
- 混合检索（Hybrid Search）：向量检索（语义）+ BM25（关键词）并行
- Chunk 策略优化：小块检索、大块喂给 LLM
- 父文档检索（Parent Document Retrieval）：细粒度定位，粗粒度返回上下文

**Post-Retrieval 优化：**
- Re-ranking（重排序）：用 Cross-Encoder 精细打分，提升 Top-K 质量
- 上下文压缩（Context Compression）：剔除无关冗余

### 2.4 第四代：Modular RAG（2024 年）

把每个 RAG 环节抽象成独立模块，根据查询类型、数据源动态组合：
- Search Module：向量、关键词、知识图谱、SQL 查询
- Memory Module：短期上下文记忆、长期知识存储
- Fusion Module：多路召回结果融合
- Routing Module：根据查询类型路由到不同检索策略
- Predict Module：子问题拆分与迭代检索

### 2.5 第五代：Agentic RAG（2025 年起）

**把 RAG 流程里的控制权交给大模型自己决策。** 传统 RAG 是固定单次检索流程；Agentic RAG 让大模型能够：判断当前召回内容是否足够、决定是否需要多轮检索、选择从哪个数据源检索、评估生成答案是否可靠。本质是 **Agent 推理能力 + RAG 知识检索能力** 的结合。

---

## 第三章：落地 RAG 时的技术选型

**技术选型核心原则：匹配场景，简单优先。**

### 3.1 文档解析层

**主要挑战：** PDF 表格/多栏布局/图片、扫描版 PDF 需要 OCR、多格式统一处理。

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| PyMuPDF | 轻量快速，纯文本提取准确 | 文字版 PDF，快速上手 |
| Docling | 支持 GPU，表格/图表识别强 | 复杂排版，生产环境 |
| Unstructured | 格式支持最广（20+ 种） | 多格式混合文档库 |
| LlamaParse | 云服务，专为 RAG 优化 | 不想自建解析基础设施 |
| MinerU | 中文支持好，开源免费 | 中文文档为主的场景 |
| pdfplumber | 轻量、精准，表格提取极强 | 文字版 PDF、精准表格抽取 |

**实践建议：** 先用最简单工具跑通，再根据问题针对性升级。表格是解析难点，大表格拆成"属性-值对"单独存储效果更好。

### 3.2 文本切分层（Chunking）

**常见策略：**
- **固定大小切分**：按 Token/字符截断，可设重叠窗口。参考 300–512 Token，50–100 Token 重叠
- **语义切分**：基于句子嵌入相似度，在语义断点切割
- **递归结构切分**：先按段落、再按句子、再按字符递归切分
- **文档感知切分**：Markdown 按标题层级切分，代码按函数/类切分
- **父子 Chunk**：小 Chunk（128 Token）用于精确检索，大 Chunk（512–1024 Token）用于给 LLM 提供上下文

**推荐策略：** 入门用固定大小 + 重叠；进阶用父子 Chunk；复杂文档用文档感知切分。

### 3.3 Embedding 模型选型

| 模型 | 类型 | 维度 | 特点 |
|------|------|------|------|
| text-embedding-3-large | API | 3072 | OpenAI，英文强，成本低 |
| text-embedding-3-small | API | 1536 | 性价比高，轻量任务首选 |
| BGE-M3 | 开源/本地 | 1024 | 中英双语强，支持密集+稀疏+多向量 |
| BGE-large-zh | 开源/本地 | 1024 | 中文专项优化 |
| Jina Embeddings v3 | API/本地 | 1024 | 多语言，支持长文本 |
| nomic-embed-text | 开源 | 768 | 轻量高效，本地部署友好 |
| m3e-base/m3e-large | 开源/本地 | 768/1024 | 国产中文专属 |

**选型建议：** 中文场景优先 BGE-M3 / BGE-large-zh；纯 API 用 text-embedding-3-small；数据保密要求高则本地部署。

### 3.4 向量数据库选型

| 数据库 | 部署方式 | 特点 | 适用场景 |
|--------|----------|------|----------|
| Milvus | 自建/云服务 | 功能最全，性能强，企业级 | 大规模生产环境 |
| Weaviate | 自建/云服务 | GraphQL 接口，模块化 | 复杂查询、多模态 |
| Qdrant | 自建/云服务 | Rust 编写，高性能 | 高性能要求 |
| Chroma | 本地嵌入 | 简单友好，无需独立服务 | 原型开发、小规模 |
| FAISS | 库（非服务） | Meta 出品，无持久化 | 学习、小项目 |
| pgvector | PostgreSQL 扩展 | 与 PG 深度集成 | 已有 PG 基础设施 |
| Pinecone | 全托管云服务 | 零运维 | 快速上线 |
| Elasticsearch/OpenSearch | 自建/云服务 | 全文+向量检索一体 | 已有 ES 业务 |

### 3.5 LLM 选型

| 需求 | 推荐选型 |
|------|----------|
| 最强效果 | GPT-4o / Claude 3.5 Sonnet |
| 效果成本平衡 | GPT-4o-mini / Gemini 1.5 Flash |
| 中文理解 | Qwen2.5-72B / DeepSeek-V3 |
| 本地部署 | Qwen2.5-32B / Llama 3.3 70B |
| 超长上下文 | Gemini 1.5 Pro（200 万 Token） |

实战常用路径：**GPT-4o 验证 → Qwen-plus 降本。**

### 3.6 RAG 框架选型

- **LangChain**：生态最全，事实标准；缺点是抽象层多
- **LlamaIndex**：专注 RAG，组件粒度更细
- **Haystack**：企业级，Pipeline 设计
- **Dify**：国内最流行，支持工作流编排
- **FastGPT**：知识库问答专项优化
- **Coze**：字节跳动出品，Agent 与工具集成强

**自研 vs 框架的判断标准：** 当你为绕开框架限制写的代码比直接自研还多时，就该自研了。

---

## 第四章：业界经典实践

### 4.1 数据入库流水线

```
原始文档（PDF/Word/网页/数据库）
    │
    ▼
[1] 文档解析（Parser）→ 提取纯文本，处理表格、图片
    │
    ▼
[2] 文本清洗（Cleaner）→ 去噪声：页码、页眉页脚、乱码
    │
    ▼
[3] 文本切分（Chunker）→ 按策略切分，附加元数据（来源、页码）
    │
    ▼
[4] 向量化（Embedder）→ 每个 Chunk 生成向量
    │
    ▼
[5] 写入向量库（Indexer）→ 存入向量数据库，建立索引
    │
    ▼
知识库就绪 ✅
```

**关键细节：** 每个 Chunk 必须附带丰富元数据（source、page、chapter、created_at、doc_type）。生产环境文档持续更新，需要支持增量入库、更新时删除旧 Chunk 写入新 Chunk、删除时清理对应向量。

### 4.2 查询增强

- **查询改写（Query Rewriting）**：去除口语化表达，补全指代不明部分
- **多查询生成（Multi-Query）**：一个问题扩展成 3–5 个不同角度子查询
- **问题分解（Query Decomposition）**：复杂问题拆解为多个子问题分别检索
- **HyDE（假设文档嵌入）**：先用 LLM 生成"假设答案"，再用假设答案去检索

### 4.3 混合检索

单纯向量检索对精确词汇不敏感（如"iPhone 15 Pro Max 电池容量"），BM25 反而能精确命中。混合检索架构：向量检索召回 Top-20 + BM25 检索召回 Top-20 → RRF 融合排序 → Top-10。

RRF（Reciprocal Rank Fusion）融合算法：
```
def reciprocal_rank_fusion(results_list, k=60):
    scores = {}
    for results in results_list:
        for rank, doc_id in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
```

### 4.4 重排序

Bi-Encoder 大范围快速召回，Cross-Encoder 精确重排。常用工具：BGE-Reranker（中文首选）、Cohere Rerank、Jina Reranker、LLM 直接打分。

### 4.5 生成层优化

核心 Prompt 结构：系统角色设定 → 检索结果（带来源）→ 用户问题 → 输出要求（基于文档回答、引用来源、信息不足时说明）。

### 4.6 数据飞轮

用户提问 → RAG 作答 → 置信度高直接回答记录日志 / 置信度低则标准化问题 → 生成候选答案 → 人工审核 → 入库。关键监控指标：置信度分布、问题覆盖率、知识库增长速度、答案准确率。

### 4.7 系统可观测性

"不可观测的系统，无法持续改进。" 生产级 RAG 必须记录完整链路：request_id、user_query、rewritten_query、retrieved_chunks、reranked_results、prompt_tokens、llm_response、confidence、latency_ms、feedback。

---

## 第五章：RAG 未来的发展方向

### 5.1 Agentic RAG

传统 RAG 流程硬编码：一次检索，一次生成。Agentic RAG 赋予 LLM 检索工具调用权，自主决策检索策略。基于 ReAct 框架（Reasoning + Acting），实现"边思考、边行动、边验证"的闭环。

```
Think: 要比较 A 和 B，需分别检索
Act: search("A 产品规格")
Observation: ...
Think: 再查 B
Act: search("B 产品规格")
Think: 信息足够，可以回答
Answer: ...
```

### 5.2 GraphRAG

传统 RAG 根本局限：Chunk 之间没有显式关系。GraphRAG（微软）思路：抽取实体与关系 → 构建知识图谱 → 检索时支持图遍历与多跳查询。优势是关联推理和全局摘要，代价是构建维护成本高。

### 5.3 多模态 RAG

两大路线：文字化（OCR/多模态 LLM 转文本描述）和多模态向量（CLIP 类模型统一编码图文）。

### 5.4 长上下文 vs RAG

长上下文优势：无需检索，避免检索失败。劣势：成本极高、大海捞针、更新困难、延迟高。RAG 不可替代的场景：超大规模知识库（百 GB 以上）、强可溯源要求、频繁实时更新、成本敏感业务。**未来趋势是 RAG + 长上下文融合。**

### 5.5 Self-RAG 与 CRAG

Self-RAG：让 LLM 在生成时判断是否需要检索、内容是否相关、答案是否有依据。CRAG：召回质量不足时自动触发 Web Search 补充信息。代表方向：从人工调优 → 系统自动优化。

### 5.6 RAG 评估体系建设

主流评估框架：RAGAS、TruLens、LangSmith。RAGAS 核心指标：Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Context Precision（上下文精确率）、Context Recall（上下文召回率）。

---

## 第六章：从零到一的 RAG 实战路径

### 6.1 三阶段落地

**阶段一：快速验证（1–2 周）**
目标：跑通流程，验证 RAG 对业务有效
技术栈：PyMuPDF/Unstructured + Chroma + BGE-M3 + GPT-4o-mini + LlamaIndex
验收：测试问题回答率 ≥70%

**阶段二：效果优化（2–4 周）**
目标：准确率从 70% → 85%+
动作：建评估集、查询改写、混合检索、重排序、Prompt 优化

**阶段三：生产化（4–8 周）**
目标：支撑真实用户，持续迭代
动作：迁移生产向量库、全链路日志、数据飞轮、监控仪表盘、定期评估机制

### 6.2 避坑清单

- ❌ 过早优化：没跑通基础 RAG 就玩 GraphRAG/Agentic RAG
- ❌ 忽视数据质量：垃圾进，垃圾出
- ❌ 一刀切 Chunk：不同文档用同一套切分参数
- ❌ 只依赖向量检索：精确词场景必须配合 BM25
- ❌ 不做权限控制：企业知识库必须按角色过滤
- ❌ 只看最终答案，不看召回质量
- ❌ 过度依赖框架：看不到中间日志，出问题无法定位
- ❌ Demo 好就上生产：生产数据远比 Demo 脏

---

## 总结：RAG 的本质与边界

**RAG 是一种思路，不是一项单一技术。** 它的本质是：在需要知识时动态检索，而不是把所有知识固化在模型里。这个思路不会消失。但 RAG 不是银弹：知识库质量、检索精度、上下文组织——每一环都决定最终效果。

**RAG 80% 的问题是数据问题，不是技术问题。** 这句话值得反复读。

---

## 附录：核心工具速查表

### 开源工具

| 类别 | 工具 | 链接 |
|------|------|------|
| 文档解析 | Docling | github.com/DS4SD/docling |
| 文档解析 | Unstructured | github.com/Unstructured-IO/unstructured |
| 文档解析 | MinerU | github.com/opendatalab/MinerU |
| 向量数据库 | Milvus | milvus.io |
| 向量数据库 | Qdrant | qdrant.tech |
| 向量数据库 | Chroma | trychroma.com |
| Embedding | BGE-M3 | huggingface.co/BAAI/bge-m3 |
| Reranker | BGE-Reranker | huggingface.co/BAAI/bge-reranker-large |
| RAG 框架 | LlamaIndex | llamaindex.ai |
| RAG 框架 | LangChain | langchain.com |
| 低代码平台 | Dify | dify.ai |
| 评估框架 | RAGAS | github.com/explodinggradients/ragas |
| GraphRAG | Microsoft GraphRAG | github.com/microsoft/graphrag |

### 关键论文

| 论文 | 核心贡献 |
|------|----------|
| RAG (Lewis et al., 2020) | RAG 概念原始论文 |
| Self-RAG (Asai et al., 2023) | 自反思 RAG |
| CRAG (Yan et al., 2024) | 修正式 RAG |
| GraphRAG (Edge et al., 2024) | 知识图谱增强 RAG |
| HyDE (Gao et al., 2022) | 假设文档嵌入 |
