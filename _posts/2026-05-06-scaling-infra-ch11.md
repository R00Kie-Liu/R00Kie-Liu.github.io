---
layout: post
title: "Scaling Book 入门第 11 章：推理优化 — KV Cache / Continuous Batching / 量化"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '推理', 'SGLang']
---

# Scaling Book 入门第 11 章：推理优化 — KV Cache / Continuous Batching / 量化

> **本章目标**：掌握现代 LLM 推理引擎的核心优化技术，理解它们各自解决什么问题。
>
> **对应原书**：Chapter 7 (All About Transformer Inference) 下半部分  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 10-11, 约 3 小时

---

## 11.1 推理的三大挑战

1. **内存**：模型权重 + KV cache 占满 HBM → 限制并发请求数
2. **延迟**：每个 token 都要加载全部权重 → 单请求延迟高
3. **吞吐量**：如何同时服务大量请求 → 利用 batching 提升硬件利用率

以下优化技术分别或同时解决这些问题。

---

## 11.2 Continuous Batching（连续批处理）

![Continuous Batching 动画](/assets/scaling-book/img/continuous-batching.gif)

### 问题：静态 Batching 的浪费

传统做法：将多个请求组成一个 batch，等所有请求都生成完毕才释放。

```
请求 A：生成 10 个 token → 第 10 步完成
请求 B：生成 50 个 token → 第 50 步完成
请求 C：生成 20 个 token → 第 20 步完成

静态 batch：全部等到第 50 步才能接新请求
→ 请求 A 在第 10-50 步白白占着 GPU 资源
```

### 解决方案：Continuous Batching

- 每个 decode 步骤后检查：有请求完成了吗？有新请求到了吗？
- 完成的请求立即释放，新请求立即加入
- batch 大小在每步动态变化

**效果**：GPU 利用率大幅提升（2-10×），因为不再有"等最慢请求"的浪费。

> 📋 **背景知识：为什么 Continuous Batching 在推理中至关重要**
>
> 训练时 batch 中所有序列长度相同（通过 padding），不存在这个问题。
> 推理时每个请求的输出长度不同（有的回答 3 个字，有的写 2000 字），静态 batch 的资源浪费极大。
> Continuous Batching 是所有现代推理引擎（vLLM、SGLang、TensorRT-LLM）的标配。

> 🛠️ **实践：SGLang**
>
> SGLang 原生支持 continuous batching：
> - 每个 decode step 自动检查并调度新请求
> - 支持 **chunked prefill**：长 prompt 的 prefill 被分成多个 chunk，穿插在 decode step 之间
>   ```bash
>   --chunked-prefill-size 8192  # 每次最多 prefill 8192 tokens
>   ```
> - 好处：避免一个长 prompt（如 100K tokens）的 prefill 阻塞所有 decode 请求数秒

---

## 11.3 PagedAttention（分页注意力）

![PagedAttention](/assets/scaling-book/img/paged-attention.png)

### 问题：KV Cache 内存碎片

传统 KV cache 为每个请求预分配固定大小的连续内存：
- 序列长度未知 → 必须按最大长度分配 → 大量浪费
- 请求结束后释放 → 内存碎片

### 解决方案：像操作系统管理虚拟内存一样管理 KV cache

PagedAttention（来自 vLLM）：
- 将 KV cache 分成固定大小的"页"（如 16 tokens/页）
- 每个请求的 KV cache 不要求连续内存
- 使用"页表"映射逻辑位置到物理位置
- 按需分配，用完就释放

**效果**：
- 内存利用率从 ~40% 提升到 ~95%
- 相同 HBM 下可以支持 2-3× 的并发请求

> 🛠️ **实践：SGLang**
>
> SGLang 使用自己的高效内存管理（类似 PagedAttention 的思路）：
> - 基于 token-level 的内存分配
> - 配合 RadixAttention 的前缀共享，共享 prefix 的 KV cache 只存一份
> - `--mem-fraction-static` 参数控制为模型权重预留的 HBM 比例，剩余给 KV cache
>   ```bash
>   --mem-fraction-static 0.8  # 80% 给权重，20% 给 KV cache（默认会自动估算）
>   ```

---

## 11.4 Prefix Caching（前缀缓存）

![Prefix Caching Trie](/assets/scaling-book/img/prefix-caching-trie.png)

### 问题：重复 Prefill 相同的 System Prompt

很多应用场景中，大量请求共享相同的前缀：
- System prompt（如 "You are a helpful assistant..."）
- Few-shot examples
- 文档 context（RAG 场景）

每次重新 prefill 这些共享前缀 → 浪费计算和时间。

### 解决方案：缓存共享前缀的 KV cache

SGLang 的 **RadixAttention**：
- 将所有请求的 token 序列组织为 Radix Tree（基数树）
- 相同前缀的 KV cache 自动共享
- 新请求到来时，先查找树中最长匹配前缀 → 只 prefill 未匹配的部分

**效果**：
- 对共享前缀的场景，TTFT 大幅降低
- 内存节省（共享前缀的 KV cache 只存一份）

> 🛠️ **实践：SGLang**
>
> RadixAttention 是 SGLang 默认启用的，无需额外配置。
>
> 最大化 prefix caching 命中率的技巧：
> - 将 system prompt 放在所有请求的最前面
> - 使用 `sgl.set_default_backend(...)` 设置后端时，prefix caching 自动生效
> - 在 multi-turn 对话中，历史消息自动成为共享前缀
> - 监控：SGLang 的 metrics endpoint 会报告 cache hit rate

---

## 11.5 量化（Quantization）

### 原理

用更少的 bit 表示模型权重：

| 精度 | 每参数 bytes | 70B 模型大小 | 相对速度 |
|------|-------------|-------------|---------|
| fp16/bf16 | 2 | 140 GB | 1× |
| fp8 | 1 | 70 GB | ~2× |
| int8 | 1 | 70 GB | ~2× |
| int4 (GPTQ/AWQ) | 0.5 | 35 GB | ~3-4× |

**为什么量化能加速推理**（尤其是 Generation）：

Generation 是 **memory-bound**。加速的关键是**减少从 HBM 加载的字节数**：
- int4 比 bf16 少 4×字节 → HBM 加载时间减少 4×
- 加上 int4 的 FLOPs/s 可能更高 → 双重加速

**代价**：精度损失，可能影响模型质量。现代量化方法（GPTQ、AWQ、SmoothQuant）通过校准集最小化精度损失。

> 🛠️ **实践：SGLang**
>
> SGLang 支持多种量化方案：
> ```bash
> # FP8 量化推理
> python -m sglang.launch_server \
>   --model-path meta-llama/Llama-3-70B \
>   --quantization fp8 \
>   --tp-size 8
>
> # AWQ 量化模型
> python -m sglang.launch_server \
>   --model-path TheBloke/Llama-3-70B-AWQ \
>   --quantization awq \
>   --tp-size 4  # int4 模型更小，可能只需要 4 卡
> ```
>
> 量化选择建议：
> - **FP8**：精度损失最小，需要 H100/B200 硬件支持
> - **INT8 (SmoothQuant)**：通用性好，大部分硬件支持
> - **INT4 (AWQ/GPTQ)**：最大压缩比，适合资源受限场景

---

## 11.6 Speculative Decoding（投机解码）

![Speculative Decoding](/assets/scaling-book/img/spec-sampling1.png)

### 问题

Generation 每步只生成 1 个 token，但要加载全部权重 → memory-bound。

### 核心思想

![Speculative Decoding 验证](/assets/scaling-book/img/spec-sampling2.png)

1. 用一个**小模型（draft model）** 快速生成 k 个候选 token
2. 用**大模型**一次性验证这 k 个 token（相当于 prefill，compute-bound）
3. 接受所有正确的 token，从第一个错误处截断

![Speculative Decoding 接受](/assets/scaling-book/img/spec-sampling3.png)

**效果**：
- 如果小模型猜对率高（如 70-80%），平均每步验证可接受 3-5 个 token
- 大模型的权重只加载一次就验证了多个 token → 有效算术强度提升
- 总体吞吐量提升 2-3×

---

## 11.7 其他优化技术

### FlashInfer / FlashDecoding

- 优化 decode 阶段的 attention kernel
- 利用 GPU 的多 SM 并行处理 KV cache 的不同 head/层

### CUDA Graphs

- 将多个 GPU kernel 打包成一个"图"
- 消除 kernel launch 开销（每次 launch ~10μs，数百个 kernel 就是 ms 级别）

> 🛠️ **实践：SGLang**
>
> SGLang 内置了这些优化：
> - **FlashInfer**：SGLang 默认使用 FlashInfer 的高效 attention kernel
> - **CUDA Graphs**：SGLang 在 decode 阶段使用 CUDA Graphs 消除 launch 开销
>   ```bash
>   --disable-cuda-graph  # 调试时可禁用
>   ```
> - **Torch Compile**：SGLang 支持用 `--enable-torch-compile` 做 kernel fusion

---

## 关键要点

- [ ] Continuous Batching：动态增删请求，消除"等最慢"的浪费
- [ ] PagedAttention：分页管理 KV cache，内存利用率 40% → 95%
- [ ] Prefix Caching：共享前缀的 KV cache 复用（SGLang 的 RadixAttention）
- [ ] 量化：减少权重字节数 → 减少 HBM 加载量 → decode 加速
- [ ] Speculative Decoding：用小模型 draft + 大模型验证，提升有效吞吐
- [ ] SGLang 内置 continuous batching + RadixAttention + FlashInfer + CUDA Graphs

---

## 进一步阅读

- 原书 Chapter 7: Tricks for Improving Generation Throughput and Latency
- [vLLM / PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [SGLang / RadixAttention 论文](https://arxiv.org/abs/2312.07104)
- [Speculative Decoding 论文 (Leviathan et al., 2023)](https://arxiv.org/abs/2211.17192)

