---
layout: post
title: "Scaling Book 入门第 10 章：推理基础 — Prefill vs Generation"
date: 2026-05-07
tags: ['LLM', 'Infra', 'Scaling', '推理', 'SGLang']
---

# Scaling Book 入门第 10 章：推理基础 — Prefill vs Generation

> **本章目标**：理解 LLM 推理的两个阶段（Prefill 和 Generation）为何性能特性完全不同，掌握推理中的核心瓶颈和关键指标。
>
> **对应原书**：Chapter 7 (All About Transformer Inference) 上半部分  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 9-10, 约 3 小时

---

## 10.1 推理 vs 训练的本质区别

> 🔗 **与你的联系**
>
> 训练时，你处理的是大 batch、长序列，全部 token 同时参与计算 → 天然 compute-bound。
> 推理时，你一次只生成 1 个 token，但需要重复数百次 → 每次都在加载全部权重但只做极少计算 → 天然 memory-bound。
>
> 这是推理 Infra 和训练 Infra 的根本差异。

| 特性 | 训练 | 推理（Generation） |
|------|------|-------------------|
| Batch size/device | 大（数千 tokens） | 小（1-数百 tokens） |
| 瓶颈 | Compute-bound | Memory-bound |
| 优化目标 | 吞吐量（tokens/s） | 延迟 + 吞吐量 |
| 内存主角 | 参数 + 优化器 + 激活 | 参数 + KV Cache |

---

## 10.2 Autoregressive Generation 的流程

![朴素推理](/assets/scaling-book/img/naive-inference.png)

标准的自回归生成：

```
输入: [token₁, token₂, ..., tokenₙ]（prompt）
输出: 逐个生成 token_{n+1}, token_{n+2}, ...

每步:
1. 输入当前 token（或全部历史）
2. 经过所有 Transformer 层
3. 得到 logits，采样下一个 token
4. 重复
```

**问题**：如果每步都重新计算所有历史 token 的 attention，复杂度是 O(n²)。

**解决方案**：KV Cache。

---

## 10.3 KV Cache

![KV Cache 推理](/assets/scaling-book/img/cached-inference.png)

**核心思想**：Attention 层中，历史 token 的 K 和 V 向量不会变化。只需计算一次并缓存，后续步骤直接使用。

```
步骤1（Prefill）：
  - 一次性处理全部 prompt tokens
  - 计算并缓存所有层的 K, V 矩阵

步骤2+（Generation/Decode）：
  - 每步只输入 1 个新 token
  - 新 token 的 Q × 所有历史的 K → attention scores
  - 将新 token 的 K, V 追加到 cache
```

**KV Cache 大小**（每序列）：

$$\text{KV size} = 2 \times L \times \text{Kv\_heads} \times K \times S \times 2 \text{ (bf16)}$$

LLaMA 70B，序列长度 4096：
- `2 × 80 × 8 × 128 × 4096 × 2 = 1.34 GB/序列`

这很大！如果你想同时 serve 256 个并发请求：256 × 1.34 GB = **343 GB** 仅 KV cache。

---

## 10.4 两个阶段的性能分析

### Prefill 阶段

- **输入**：完整 prompt（数百到数千 tokens）
- **计算模式**：和训练前向传播几乎相同
- **性能特性**：batch_tokens 大 → **Compute-bound**
- **关键指标**：**Time to First Token (TTFT)**

$$T_{\text{prefill}} \approx \frac{2 \times S \times P}{\text{FLOPs/s}}$$

其中 S 是 prompt 长度。

### Generation（Decode）阶段

- **输入**：每步只有 1 个新 token
- **计算模式**：加载全部权重，但只对 1 个 token 做 matmul
- **性能特性**：batch_tokens = 1 → 严重 **Memory-bound**
- **关键指标**：**Time Per Output Token (TPOT)** 或 **Inter-Token Latency (ITL)**

$$T_{\text{decode}} \approx \frac{2P}{\text{HBM bandwidth}}$$

（每步需要从 HBM 加载一次完整的模型权重）

> 📋 **背景知识：为什么 Generation 是 Memory-bound**
>
> 回顾 Roofline：matmul `X[B,D] × W[D,F]` 的算术强度 ≈ B。
> - Prefill：B = prompt_length（数百-数千）→ 远大于临界值 240 → Compute-bound
> - Generation：B = 1（单 token）→ 远小于 240 → Memory-bound
>
> 在 Generation 阶段，模型做的是 `x[1, D] × W[D, F]`，加载了整个 W 但只做了 DF 次 FLOPs（而非训练时的 B×D×F 次）。绝大部分时间花在从 HBM 读取权重上。

---

## 10.5 Attention 的特殊性

在 Generation 阶段，Attention 也有不同的性能特性：

$$Q[1, D] \times K^T[D, S] \rightarrow \text{scores}[1, S]$$

- 这是一个 `[1, D] × [D, S]` 的 matmul，算术强度 ≈ 1
- 加载 K cache 的量 ∝ 序列长度 S
- 随着生成的 token 越来越多，Attention 的开销线性增长

这就是为什么**长序列推理**特别昂贵：KV cache 内存和 attention 计算时间都随 S 线性增长。

---

## 10.6 GQA / MQA 的推理优势

![KV sharing 方案](/assets/scaling-book/img/kv-sharing.png)

- **MHA**（Multi-Head Attention）：每个 head 有独立的 KV → KV cache 最大
- **GQA**（Grouped Query Attention）：多个 Q head 共享一组 KV → KV cache 缩小 H/Kv_heads 倍
- **MQA**（Multi-Query Attention）：所有 Q head 共享一组 KV → KV cache 最小

![GQA/MQA 对比](/assets/scaling-book/img/gmqa.png)

LLaMA 3 使用 GQA（64 Q heads, 8 KV heads）→ KV cache 只有 MHA 的 1/8。

---

## 10.7 延迟和吞吐量建模

**单请求延迟**：

$$\text{Total latency} = T_{\text{prefill}} + n_{\text{output}} \times T_{\text{decode}}$$

**吞吐量**（通过 batching 多请求）：

- Batching 不改变 Generation 的权重加载量（weights 只加载一次供所有请求使用）
- 但增加了 FLOPs：每增加一个请求，增加少量计算
- 直到 batch 大到 compute-bound 为止，吞吐量近似线性增长

$$\text{Throughput} \approx \min\left(\frac{\text{FLOPs/s}}{2P}, \frac{\text{HBM BW}}{2P / B}\right) \times B$$

临界 batch size = HBM FLOPs/s ÷ HBM BW ≈ 240-300（和 Roofline 临界值一致！）

---

## 10.8 Disaggregated Serving（分离式服务）

![分离式推理](/assets/scaling-book/img/disaggregation.png)

由于 Prefill 和 Generation 性能特性完全不同：

| 阶段 | 瓶颈 | 理想硬件 |
|------|------|---------|
| Prefill | Compute | 高 FLOPs/s 的芯片 |
| Generation | Memory BW | 高带宽的芯片 |

**Disaggregated Serving** 将二者分开：
- Prefill 集群：用高算力卡处理 prompt
- Decode 集群：用高带宽/大内存卡做生成
- 中间通过网络传递 KV cache

> 🛠️ **实践：SGLang**
>
> SGLang 在推理方面的核心设计：
>
> 1. **RadixAttention**：SGLang 的核心创新
>    - 将 KV cache 组织为 Radix Tree（前缀树）
>    - 多个请求共享相同的 prompt prefix → KV cache 自动复用
>    - 比传统 prefix caching 更灵活（支持任意共享前缀）
>
> 2. **Prefill/Decode 调度**：
>    - SGLang 使用 chunked prefill：将长 prompt 分 chunk 处理，穿插 decode 步骤
>    - 避免长 prompt 的 prefill 阻塞正在 decode 的请求
>    - `--chunked-prefill-size` 控制 chunk 大小
>
> 3. **Tensor Parallelism 推理**：
>    ```bash
>    python -m sglang.launch_server \
>      --model-path meta-llama/Llama-3-70B \
>      --tp-size 8  # 8 卡张量并行
>    ```
>    - 推理中 TP 也需要 AllReduce，但由于每步只处理少量 token，通信量很小
>    - TP 的主要好处是减少每卡的权重加载量 → 降低 decode 延迟

---

## 关键要点

- [ ] 推理有两个阶段：Prefill（compute-bound）和 Generation（memory-bound）
- [ ] Generation 是 memory-bound 因为每步只处理 1 个 token 但要加载全部权重
- [ ] KV Cache 避免重复计算历史 token，但内存开销大（∝ 序列长度 × 层数）
- [ ] GQA/MQA 大幅减少 KV cache 大小
- [ ] Batching 可以提升 decode 的吞吐量（直到 compute-bound）
- [ ] 临界 batch size ≈ FLOPs/s ÷ HBM BW ≈ 240-300
- [ ] SGLang 的 RadixAttention 实现了高效的前缀共享 KV cache

---

## 进一步阅读

- 原书 Chapter 7: All About Transformer Inference
- [SGLang 论文 (Zheng et al., 2024)](https://arxiv.org/abs/2312.07104)
- [GQA 论文](https://arxiv.org/abs/2305.13245)

