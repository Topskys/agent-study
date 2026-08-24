# KV Cache（Key-Value Cache，键值缓存）

KV Cache 是大语言模型（LLM）在**自回归推理（文本生成）阶段**最核心的加速技术，本质是**以显存空间换计算时间**。

---

## 一、为什么需要 KV Cache？（痛点背景）

自回归生成时，模型**逐 Token 输出**：

1. 输入："人工智能" → 输出："改变"
2. 输入："人工智能 改变" → 输出："了"
3. 输入："人工智能 改变 了" → 输出："世界"

**没有 KV Cache 时（大量重复计算）：**

- 每生成一个新词，都要把前面所有词重新送入 Transformer 完整前向一次。
- 但因果掩码（causal mask）保证了：前文每个 Token 的 **Key（K）** 和 **Value（V）** 只由它自己及其前缀决定，**与未来无关、一旦算出永不改变**。重算它们是纯浪费。
- 复杂度：无缓存时每步对长度 $S$ 的序列做完整前向，注意力项为 $O(S^2 \cdot d)$；整个 $N$ 步生成的总计算量约 $O(N^3 \cdot d)$ 级，生成越来越慢。[[1]](#ref-1)

---

## 二、KV Cache 的工作原理

Transformer 自注意力公式：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

**为什么只缓存 K 和 V，不缓存 Q？**
每步解码中唯一"新"的计算就是新 Token 的 Q——旧 Q 在各自步骤里乘完 V 即被丢弃，对未来无用；而 K/V 是"被看的一方"，未来每个新 Token 都要反复与全部历史 K/V 打分。

推理因此拆分为两个阶段：

### 1. Prefill（预填充 / 首字生成）

- 一次性并行计算整个 Prompt 所有 Token 的 $Q, K, V$；
- 将所有 Prompt Token 的 $K$、$V$ 写入显存缓存；
- 用最后一个 Token 的 logits 采样出第 1 个生成 Token。

### 2. Decoding（解码 / 后续逐字生成）

- 每步只将**上一步新生成的单个 Token** 作为输入；
- 只计算当前 Token 的 $q_{\text{new}}, k_{\text{new}}, v_{\text{new}}$，并将 $k_{\text{new}}, v_{\text{new}}$ **追加（Append）**到 KV Cache；
- 用 $q_{\text{new}}$ 与**全部历史缓存的 $K$** 做打分，再对**全部历史缓存的 $V$** 加权求和；
- 重复直至终止符。

**复杂度变化：** 单步从「全序列完整前向」的 $O(S^2 \cdot d)$（注意力项）降为「单 Query 对缓存打分」的 $O(S \cdot d)$——由平方级降为线性级；整个 $N$ 步生成由 $O(N^3)$ 级降至 $O(N^2)$ 级。注意单步并非 $O(1)$：注意力打分仍需遍历全部历史缓存。

**缓存的确切内容（每层）：**

```
K_cache[layer]: [B, H_kv, S, D]
V_cache[layer]: [B, H_kv, S, D]    # append-only，随 S 线性增长
```

---

## 三、KV Cache 的代价（显存占用）

KV Cache 解决了**计算瓶颈**，但带来了**显存与带宽瓶颈**。

### 显存占用公式

$$\text{Memory (Bytes)} = 2 \times L \times S \times H_{kv} \times D \times P \times B$$

| 符号 | 含义 | 说明 |
| --- | --- | --- |
| $2$ | K 与 V 各一份 | — |
| $L$ | 模型层数（Layers） | 每层独立缓存 |
| $S$ | 序列总长度 | Prompt + 已生成 Token 数 |
| $H_{kv}$ | KV 注意力头数 | MHA 下 = Q 头数；GQA/MQA 下更小 |
| $D$ | 每头维度（Head Dim） | MHA 下 $H_{kv} \times D = \text{Hidden Size}$ |
| $P$ | 精度字节数 | FP16/BF16 = 2，FP8/INT8 = 1 |
| $B$ | 并发数（Batch Size） | beam search 会成倍放大有效 batch |

> **直观示例：** 13B 模型（40 层，40 个头，头维度 128，FP16），上下文 4096，并发 4：
> $$2 \times 40 \times 4096 \times 40 \times 128 \times 2 \times 4 = 13.4 \text{ GB}$$
> 已接近该模型 FP16 权重（约 26 GB）的一半；**并发或上下文再翻一倍，仅缓存就反超全部权重**。这限制了并发量与长上下文支持，也是 decode 阶段几乎总是 memory-bandwidth bound 的根源。[[1]](#ref-1)[[2]](#ref-2)

---

## 四、常见优化与改进技术

### 1. 结构改进（减少 KV 头数）

| 方案 | 机制 | 缓存缩减 | 代表模型 |
| --- | --- | --- | --- |
| **MHA** | 每个 Q 头独立 K/V 头 | 1×（基准） | LLaMA-2-7B/13B |
| **MQA** | 所有 Q 头共享 1 组 K/V | $H$× | PaLM、Falcon |
| **GQA** | Q 头分组共享 K/V（折中） | $H/H_{kv}$× | LLaMA-3 全系、Mistral；LLaMA-2 仅 70B 版采用 |
| **MLA** | K/V 低秩投影压缩至共享潜空间，按需解压 | 数倍以上 | DeepSeek-V2/V3 |

GQA 论文报告：uptraining 后质量接近 MHA、速度接近 MQA，是当前主流选择。[[4]](#ref-4)

### 2. 系统 / 显存管理优化

- **PagedAttention（vLLM）**：借鉴操作系统虚拟内存分页，将每条序列的 KV Cache 切成固定大小块（block）、经块表映射到非连续物理显存，按需分配。传统方式因碎片化与按最大长度超预留浪费 60%–80% 内存，PagedAttention 将浪费压到 4% 以内，并天然支持块共享（写时复制）。[[3]](#ref-3)
- **Prompt Caching / Prefix Caching**：多个请求含相同 System Prompt 或长前缀时，跨请求复用已计算的 KV 块，省去重复 prefill。

### 3. 量化（Quantization）

将 KV Cache 从 FP16 压缩至 **FP8 / INT8 / INT4**，显存减半甚至更多；通常配合 per-channel/outlier 保护策略以控制精度损失。

### 4. 长文本窗口 / 丢弃策略

- **Sliding Window Attention**：只保留最近窗口 W 内的 K/V，可用滚动缓冲区使缓存封顶于 $\min(S, W)$（Mistral 采用）。
- **StreamingLLM / Attention Sink**：保留最初几个 Attention Sink Token + 滑动窗口，支持无限流式生成且显存常数级。
- **按重要性驱逐/压缩**（如 H2O、SnapKV 等）：依据注意力贡献淘汰低价值 Token 的 K/V。

---

## 参考文献

<a id="ref-1"></a>[1] ["How To Scale Your Model · Inference."](https://jax-ml.github.io/scaling-book/inference/) scaling-book 推理章节（Google DeepMind 系作者），prefill/decode 瓶颈与显存推导。

<a id="ref-2"></a>[2] NVIDIA. ["Mastering LLM Techniques: Inference Optimization."](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) *NVIDIA Developer Blog*, 2023.

<a id="ref-3"></a>[3] W. Kwon et al. ["Efficient Memory Management for Large Language Model Serving with PagedAttention."](https://arxiv.org/abs/2309.06180) *SOSP*, 2023.

<a id="ref-4"></a>[4] J. Ainslie et al. ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoint."](https://arxiv.org/abs/2305.13245) *EMNLP*, 2023.
