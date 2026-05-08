---
layout: post
title: "Scaling Book 入门第 12 章：实战 — Serving LLaMA 3"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '推理', 'SGLang']
---

# Scaling Book 入门第 12 章：实战 — Serving LLaMA 3

> **本章目标**：将推理理论应用到 LLaMA 3-70B 的实际 serving 中，分析延迟/吞吐量权衡，给出 SGLang 的实际部署配置。
>
> **对应原书**：[Chapter 8 (Serving LLaMA 3-70B on TPUs)](https://jax-ml.github.io/scaling-book/applied-inference)  
> **优先级**：⭐⭐ 中 | **建议时间**：Day 11-12, 约 2 小时

---

## 12.1 硬件选型

推理的硬件选择通常追求 **FLOPs/$ 最大化**：

| 硬件 | bf16 FLOPs/s | 价格 ($/h) | FLOPs/$ | HBM | HBM BW |
|------|-------------|-----------|---------|-----|--------|
| H100 SXM | 990 TF | ~$10.8 | 3.3e17 | 80 GB | 3.35 TB/s |
| H100 NVL | 990 TF | ~$8.5 | 4.1e17 | 94 GB | 3.35 TB/s |
| A100 80G | 312 TF | ~$3.5 | 3.2e17 | 80 GB | 2.0 TB/s |
| TPU v5e | 197 TF | ~$1.2 | **5.8e17** | 16 GB | 820 GB/s |

TPU v5e 的 FLOPs/$ 最高，但 HBM 只有 16 GB，需要更多芯片来容纳模型。H100 的优势在于大 HBM 和高带宽（对 memory-bound 的 generation 很重要）。

---

## 12.2 LLaMA 3-70B 的推理内存分析

### 模型权重

| 精度 | 总量 | 每卡（TP=8, H100） | 每卡（16 片 TPU v5e） |
|------|------|-------------------|---------------------|
| bf16 | 140 GB | 17.5 GB | 8.75 GB |
| int8 | 70 GB | 8.75 GB | 4.38 GB |
| int4 | 35 GB | 4.4 GB | 2.19 GB |

### KV Cache

每 token：$2 \times K_v\_\text{heads} \times K \times L \times \text{sizeof(dtype)} = 2 \times 8 \times 128 \times 80 = 163,840$ bytes (**160 KB/token** in int8)。

这个数字非常大！32K 序列的 KV cache = 160 KB × 32768 = **5.3 GB/序列**。

| 序列长度 | 每请求 KV (int8) | 每请求 KV (bf16) | 256 并发 (int8) |
|----------|-----------------|-----------------|----------------|
| 2048 | 0.33 GB | 0.67 GB | 85 GB |
| 4096 | 0.67 GB | 1.34 GB | 171 GB |
| 8192 | 1.34 GB | 2.68 GB | 343 GB |

### 最小部署拓扑

> 💡 **Pop Quiz：不同量化下的最小 GPU 数**
>
> LLaMA 3-70B 在不同精度下最少需要多少张 H100（80 GB）？假设 batch=32, 序列长度 8192, KV int8。
>
> <details markdown="1">
> <summary>点击查看答案</summary>
>
> | 精度 | 权重大小 | KV cache (BS=32, 8K) | 总内存 | 最少 H100 |
> |------|---------|---------------------|--------|----------|
> | bf16 | 140 GB | 32×1.34=42.8 GB | 183 GB | 3（但 TP 需偶数 → 4 卡）|
> | int8 | 70 GB | 42.8 GB | 113 GB | 2（但 TP=2 → 2 卡）|
> | int4 | 35 GB | 42.8 GB | 78 GB | 1（勉强放下）|
>
> 注意 int4 权重 + int8 KV 可以在**单张 H100** 上放下 LLaMA 70B！但 batch 只有 32，FLOPs 利用率极低。
>
> </details>

### 内存预算（8 张 H100）

8 张 H100 = 640 GB 总 HBM：
- bf16 权重：140 GB
- 可用于 KV cache + 工作内存：~480 GB
- 序列长度 4096 (int8 KV)：最多 ~710 个并发请求
- 序列长度 8192 (int8 KV)：最多 ~355 个并发请求

---

## 12.3 延迟分析

### 完整的 Decode 延迟公式

每一步 decode 的理论延迟由三部分组成：

$$T_{\text{step}} = \underbrace{\frac{B \times \text{KV size/token} \times S}{\text{Total HBM BW}}}_{\text{Attention（始终 BW-bound）}} + \underbrace{\max\left(\frac{2BP}{\text{Total FLOPs/s}}, \frac{P_{\text{bytes}}}{\text{Total HBM BW}}\right)}_{\text{MLP（可能 compute-bound）}}$$

其中 $S$ 是当前序列长度，$B$ 是 batch size。

### Prefill 延迟（TTFT）

Prefill 是 compute-bound 的：

$$T_{\text{prefill}} = \frac{2 \times S_{\text{prompt}} \times P}{\text{FLOPs/s} \times N_{\text{gpu}} \times \text{MFU}}$$

LLaMA 70B，prompt 2048 tokens，8 张 H100（40% MFU）：

$$T_{\text{prefill}} = \frac{2 \times 2048 \times 70 \times 10^9}{9.9 \times 10^{14} \times 8 \times 0.4} \approx 91 \text{ ms}$$

（理想情况 100% MFU：36 ms）

### Decode 延迟（TPOT）

当 batch size 很小（memory-bound）时：

$$T_{\text{decode}} \approx \frac{P_{\text{bytes}}}{\text{Total HBM BW}}$$

bf16，8 张 H100（总带宽 8 × 3.35 = 26.8 TB/s）：

$$T_{\text{decode}} = \frac{140 \times 10^9}{26.8 \times 10^{12}} \approx 5.2 \text{ ms/token}$$

即约 **192 tokens/s**（单请求 decode 速度）。

> 📋 **关键洞察：延迟的下界**
>
> Decode 延迟的下界 = 从 HBM 加载全部权重的时间。对于 bf16 LLaMA 70B on 8×H100：
> - 最小延迟 = 5.2 ms/token（即使 batch=1）
> - 这就是为什么 **量化是降低延迟的最直接方法**——int8 将此下界减半到 2.6 ms

### 加入 KV Cache 后的延迟

随着序列变长或 batch 变大，KV cache 的加载时间变得显著：

| Batch | 序列长度 | KV 加载时间 | 权重加载时间 | 总延迟 |
|-------|---------|------------|------------|--------|
| 1 | 4096 | 0.025 ms | 5.2 ms | 5.2 ms |
| 32 | 4096 | 0.8 ms | 5.2 ms | 6.0 ms |
| 128 | 4096 | 3.2 ms | 5.2 ms | 8.4 ms |
| 128 | 8192 | 6.4 ms | 5.2 ms | 11.6 ms |
| 256 | 8192 | 12.8 ms | 5.2 ms | 18.0 ms |

**大 batch + 长序列时，KV cache 加载反而主导延迟！**

---

## 12.4 吞吐量分析

### 临界 Batch Size

回顾 Roofline 分析：matmul 从 memory-bound 变为 compute-bound 的临界 batch size 取决于权重精度和算术精度的组合。

| 权重精度 | 算术精度 | 临界 Batch Size (H100) | 临界 Batch Size (TPU v5e) |
|---------|---------|----------------------|-------------------------|
| bf16 | bf16 | B > 295 | B > 240 |
| int8 | bf16 | B > 148 | B > 120 |
| int8 | int8 | B > 295 | B > 240 |
| int4 | bf16 | B > 74 | B > 60 |

> 📋 **背景知识：为什么 int8 权重 + bf16 算术的临界值减半？**
>
> 对于 matmul `X[B,D] × W[D,F]`：
> - **FLOPs** = 2BDF（与权重精度无关，由算术精度决定）
> - **加载字节数** = 权重 DF × sizeof(dtype) + 输入 BD × sizeof(activation)
>
> 当权重从 bf16 变为 int8 时，加载字节数减半，但 FLOPs 不变。算术强度翻倍 → 临界 batch size 减半。这也是为什么 **int8 权重量化在推理中如此有价值**——它不仅减少内存，还让你在更小的 batch 就能达到 compute-bound！

### 最小拓扑与最大 Batch Size

给定拓扑和量化方案，batch size 受限于剩余 HBM。以下分析基于 LLaMA 3-70B + 8192 序列长度：

**H100 (80 GB HBM)**：

| 精度 | 权重大小 | 最小 GPU 数 | 实际配置 | 剩余 HBM | 最大并发 (8K, int8 KV) |
|------|---------|-----------|---------|---------|---------------------|
| bf16 | 140 GB | 2 | 4 卡 TP=4 | 180 GB | ~134 |
| int8 | 70 GB | 1 | 2 卡 TP=2 | 90 GB | ~67 |
| int4 | 35 GB | 1 | 1 卡 | 45 GB | ~33 |

**TPU v5e (16 GB HBM)**：

| 精度 | 权重大小 | 最小芯片数 | 实际拓扑 | 剩余 HBM | 最大 KV 序列数 (8K) |
|------|---------|----------|---------|---------|------------------|
| bf16 | 140 GB | 9 | 4×4=16 片 | 116 GB | ~43 |
| int8 | 70 GB | 5 | 4×2=8 片 | 58 GB | ~43 |
| int4 | 35 GB | 3 | 2×2=4 片 | 29 GB | ~43 |

> 💡 **关键洞察：int4 可以在极少芯片上放下 70B 模型**
>
> TPU v5e 的 2×2 拓扑（4 片）就能放下 int4 的 LLaMA 70B！但 batch size 只有 ~43，FLOPs 利用率极低（远低于临界值 60）。**最小拓扑并非最佳拓扑**——加芯片不是为了放权重，而是为了给 KV cache 腾空间、提升 batch size。

### QPS / 芯片（吞吐量效率）

假设 decode 中位长度 512 tokens，每步延迟 ≈ HBM 满载时间 ≈ 19 ms（TPU v5e 全部 16 GB 加载一遍）。

$$\text{QPS/chip} = \frac{B}{\text{latency/step} \times \text{median\_decode\_length} \times N_{\text{chips}}}$$

| 精度 | 最小拓扑 | Max BS | QPS/chip |
|------|---------|--------|----------|
| bf16 | 4×4 (16) | 43 | 0.27 |
| int8 | 4×2 (8) | 43 | 0.55 |
| int4 | 2×2 (4) | 43 | 1.11 |

### 拓扑翻倍的吞吐量增益

如果将每种精度的拓扑翻倍（2× 芯片），增加的 HBM 全部用于 KV cache → batch size 大幅提升：

| 精度 | 翻倍拓扑 | 新 Max BS | 新 QPS/chip |
|------|---------|----------|------------|
| bf16 | 4×8 (32) | 140 | 0.44 |
| int8 | 4×4 (16) | 140 | 0.90 |
| int4 | 2×4 (8) | 140 | 1.80 |

**翻倍芯片 → QPS/chip 提升 ~60%！** 这是因为额外 HBM 全部变成了 KV cache 空间 → 更大 batch → 更高利用率。继续增加芯片的收益会逐渐递减，直到 batch 达到临界值后，吞吐量趋于饱和。

> 🛠️ **实践：Megatron-LM 推理配置**
>
> Megatron-LM 也支持推理 serving，其核心配置与训练共享 TP/PP 并行度：
>
> ```bash
> # Megatron 推理模式
> python tools/run_text_generation_server.py \
>   --tensor-model-parallel-size 8 \
>   --pipeline-model-parallel-size 1 \
>   --load /path/to/llama-70b-checkpoint \
>   --tokenizer-type HuggingFaceTokenizer \
>   --tokenizer-model meta-llama/Meta-Llama-3-70B-Instruct \
>   --bf16 \
>   --micro-batch-size 1 \
>   --max-tokens-to-oom 8192
> ```
>
> 注意推理中通常 **PP=1**（pipeline 并行增加延迟），主要靠 TP 分摊权重加载。Megatron 的推理引擎不如 SGLang/vLLM 优化（无 continuous batching、无 PagedAttention），但适合与训练共享同一份 checkpoint。

### 理论最大吞吐量

在 compute-bound 极限下（batch 足够大）：

$$\text{Max throughput} = \frac{\text{FLOPs/s} \times N_{\text{chips}}}{2P_{\text{flops}}}$$

| 配置 | 理论最大吞吐量 |
|------|-------------|
| 8×H100, LLaMA 70B | 56,571 tokens/s |
| 16×TPU v5e, LLaMA 70B | 22,514 tokens/s |
| 4×H100, LLaMA 70B (int8) | 28,286 tokens/s |

但实际受 KV cache 内存限制，batch 远达不到临界值，实际吞吐量通常只有理论上限的 **10-30%**。

---

## 12.5 延迟 vs 吞吐量的权衡

### Pareto 前沿

每步 decode 的延迟可分解为三个组成部分：

$$T_{\text{step}} = \underbrace{T_{\text{param}}}_{\text{权重加载}} + \underbrace{T_{\text{KV}}}_{\text{KV cache 加载}} + \underbrace{T_{\text{FLOPs}}}_{\text{计算（MLP 中取 max）}}$$

其中：
- $T_{\text{param}} = \frac{P_{\text{bytes}}}{N \times W_{\text{HBM}}}$，与 batch size 无关
- $T_{\text{KV}} = \frac{B \times \text{KV\_size/token} \times S}{N \times W_{\text{HBM}}}$，∝ batch × 序列长度
- $T_{\text{FLOPs}} = \frac{2BP}{N \times C}$，∝ batch

**延迟分解表**（16 片 TPU v5e, int8 权重, 序列长度 8192）：

| Batch | $T_{\text{param}}$ | $T_{\text{KV}}$ | $T_{\text{FLOPs}}$ | 总延迟 | 瓶颈 |
|-------|-------------------|-----------------|---------------------|--------|------|
| 1 | 5.3 ms | 0.01 ms | 0.04 ms | 5.3 ms | 参数加载 |
| 16 | 5.3 ms | 0.15 ms | 0.71 ms | 5.5 ms | 参数加载 |
| 64 | 5.3 ms | 0.61 ms | 2.84 ms | 5.9 ms | 参数加载 |
| 120 | 5.3 ms | 1.14 ms | 5.33 ms | 6.4 ms | FLOPs ≈ 参数 |
| 240 | 5.3 ms | 2.28 ms | 10.67 ms | 13.0 ms | FLOPs |
| 512 | 5.3 ms | 4.88 ms | 22.76 ms | 27.6 ms | FLOPs |

> 📋 **关键洞察：延迟的三阶段**
>
> 1. **小 batch**（B < 120）：$T_{\text{param}}$ 主导，延迟几乎不变（≈ 5.3 ms），但吞吐量线性增长
> 2. **中 batch**（120 < B < 300）：过渡区，$T_{\text{FLOPs}}$ 开始追上 $T_{\text{param}}$
> 3. **大 batch**（B > 300）：$T_{\text{FLOPs}}$ 主导，延迟 ∝ B，吞吐量趋于常数
>
> **但在长上下文场景下，$T_{\text{KV}}$ 可能在任何阶段都很大！** 序列长度 32K 时，$T_{\text{KV}}$ 是 8K 的 4 倍，可能成为主要瓶颈。

### 吞吐量/延迟 Pareto 前沿

以 batch size 为自变量，可以画出 (延迟, 吞吐量/芯片) 的 Pareto 前沿：

```
吞吐量/芯片 (tokens/s/chip)
    │
1.0 │                          ╭─────── compute-bound 饱和
    │                     ╭───╯
    │                ╭───╯
0.5 │           ╭───╯
    │      ╭───╯
    │  ╭──╯   ← 线性增长区（memory-bound）
0.1 │╯
    └──┬──────┬──────┬──────┬──→ 每步延迟 (ms)
       5     10     15     20
       ↑                    ↑
    小 batch            大 batch
    (低延迟,             (高吞吐,
     低利用率)            高延迟)
```

**核心权衡**：
- **将延迟翻倍（5ms → 10ms）可以将 per-token 成本降低 ~100×**
- 这就是为什么生产系统几乎从不使用 batch=1

### 长上下文的特殊挑战

当序列长度增加时，KV cache 加载时间急剧增长：

| 序列长度 | BS=64 时 $T_{\text{KV}}$ | 占总延迟比例 |
|---------|-------------------------|------------|
| 2K | 0.15 ms | 3% |
| 8K | 0.61 ms | 10% |
| 32K | 2.44 ms | 31% |
| 128K | 9.76 ms | 63% |

> 📋 **关键洞察：长上下文推理中，KV cache 加载始终主导延迟**
>
> 在所有 > 2048 的序列长度下，即使 batch size 未达到 compute-bound 临界值，KV cache 的 HBM 带宽消耗也超过了 FLOPs 计算时间！**这凸显了减少 KV cache 大小对推理性能的核心重要性**——GQA、MQA、KV cache 量化（int8/int4）和 KV cache 压缩都是关键优化方向。

### 场景化配置建议

| 场景 | 优化目标 | 推荐配置 |
|------|---------|---------|
| 聊天机器人 | 低延迟 (TPOT < 30ms) | TP=8, 小 batch, int8 量化 |
| Batch 处理 | 高吞吐 | TP=4, 大 batch, continuous batching |
| API 服务 | 平衡 | TP=8, 中等 batch + chunked prefill |
| 长上下文应用 | KV cache 效率 | GQA + int4 KV + 大 TP |

> 🛠️ **实践：Megatron-LM 的推理 Batch 策略**
>
> Megatron-LM 的推理默认使用静态 batching，即一次接收固定 batch 的请求，全部完成后再接收下一批：
>
> ```python
> # megatron/text_generation/api.py 的简化逻辑
> def generate_and_post_process(model, prompts, tokens_to_generate):
>     # 所有 prompts 组成一个 batch 同时处理
>     # 每步 decode 必须等所有序列完成
>     for step in range(tokens_to_generate):
>         output = model(batch_input)  # 整个 batch 同步推进
> ```
>
> 这种方式的缺点是 **短序列必须等长序列完成**，造成 GPU 空闲。现代推理引擎（SGLang/vLLM）使用 continuous batching 解决了这个问题。

---

## 12.6 Generation 中的分片策略

### 推理只有模型并行

训练中可以使用 DP + TP + PP 等多种并行方式，但 **Generation 阶段几乎只能使用模型并行**（TP）：

- **DP** 在推理中不适用（每个请求独立，不需要梯度同步）
- **PP** 会增加延迟（每步 decode 必须串行经过所有 stage），在延迟敏感的推理中代价太高
- **TP** 是唯一可行的方案：将权重分到多卡 → 减少每卡加载量 → 降低延迟

### 模型并行的上限

TP 不是可以无限扩展的。每次矩阵乘后都需要 AllReduce，通信量为 $2BD$。当模型并行度 $Y$ 增大到一定程度，通信时间会超过计算/加载时间。

对于 compute-bound（大 batch）的情况，ICI-bound 条件为：

$$Y > \frac{F \times M_Y}{C / W_{\text{ICI}}}$$

其中 $M_Y$ 是用于模型并行的 ICI 轴数量，$C/W_{\text{ICI}}$ 是 FLOPs/ICI 带宽比。

| 硬件 | $C/W_{\text{ICI}}$ | 2 轴 TP 上限 | 实际建议 |
|------|-------------------|-------------|---------|
| TPU v5e | 2200 | ~26 | 4×4 = 16 |
| H100 NVLink | ~1300 | ~44 | 8 (节点内) |
| H100 IB | ~16000 | ~3.6 | 不跨节点做 TP |

**GPU 的关键限制**：NVLink 只在节点内部可用（8 卡），跨节点只有 IB（带宽差 10×）。因此 GPU 的 TP 上限通常是 **8**，不像 TPU 可以在整个 pod 上做 TP。

### Memory-bound 下的特殊机会

但在小 batch 时（memory-bound），模型并行的上限可以放宽！原因是：
- Memory-bound 意味着 $T_{\text{HBM}} \gg T_{\text{FLOPs}}$
- 增加 TP 减少 $T_{\text{HBM}}$，即使通信有损耗，总延迟仍然下降
- 粗略估计：memory-bound 下 TP 可以扩展到 $Y = F / (8B)$

| Batch Size | Memory-bound 下 TP 上限 |
|-----------|----------------------|
| 32 | 112 |
| 64 | 56 |
| 128 | 28 |
| 240 | 15 |

> 💡 **Pop Quiz：4×8 拓扑可以做纯 TP 吗？**
>
> LLaMA 3-70B, bf16 权重, 4×8 TPU v5e (32 片)。
>
> <details markdown="1">
> <summary>点击查看答案</summary>
>
> 在 compute-bound 极限下：TP 上限 ≈ 26，4×8 = 32 > 26，**会 ICI-bound**。
>
> 但如果 batch size 较小（如 64），memory-bound 下 TP 上限 ≈ 56 > 32，**可以使用纯 TP！** 代价是 FLOPs 利用率不高，但延迟更低。实际验算（$F=28672, D=8192, B=64$）：
>
> $$T_{\text{ICI}} = \frac{2 \times 64 \times 8192}{9 \times 10^{10}} = 11\,\mu s$$

>
> $$T_{\text{HBM}} = \frac{2 \times 8192 \times 28672}{32 \times 8.1 \times 10^{11}} = 18\,\mu s$$

>
> $$T_{\text{FLOPs}} = \frac{2 \times 64 \times 8192 \times 28672}{32 \times 1.97 \times 10^{14}} = 4.8\,\mu s$$

>
> HBM 加载仍然主导 → 可以用 4×8！但从吞吐量角度，4×4 和 4×8 的 QPS/chip 相同，只是延迟减半。
>
> </details>

### GPU vs TPU 的分片差异

| 方面 | GPU (H100 NVLink) | TPU v5e |
|------|-------------------|---------|
| TP 上限 | 8（节点内 NVLink） | 16-32（pod 内 ICI） |
| 跨节点 TP | 不推荐（IB 太慢） | 可行（ICI 均匀） |
| 典型部署 | 1 节点 TP=8 | 4×4 或 4×8 |
| PP 使用场景 | 跨节点（延迟换内存） | 很少使用 |

---

## 12.7 Prefill 分析与 Disaggregated Serving

### Prefill 延迟

Prefill 是 compute-bound 的，延迟公式简单：

$$T_{\text{prefill}} = \frac{2 \times S_{\text{prompt}} \times P}{\text{FLOPs/s} \times N \times \text{MFU}}$$

| 配置 | Prompt 2048 | Prompt 8192 | Prompt 32768 |
|------|------------|------------|-------------|
| 8×H100 (40% MFU) | 91 ms | 364 ms | 1.5 s |
| 16×TPU v5e (40% MFU) | 227 ms | 910 ms | 3.6 s |
| 4×H100 (40% MFU) | 182 ms | 728 ms | 2.9 s |

**注意 prefill 延迟远大于单步 decode！** 8K prompt 的 prefill 需要 910 ms，而单步 decode 只需 19 ms。

### KV Cache 驱逐率

在 continuous batching 中，每当一个序列完成生成，它的 KV cache 被驱逐，新请求加入 batch。驱逐率：

$$\text{tokens\_evicted/step} = \frac{B \times (S_{\text{prefill}} + S_{\text{decode}})}{S_{\text{decode}}}$$

假设 prefill 中位 8192 tokens，decode 中位 4096 tokens，batch=32：

$$\text{tokens\_evicted/step} = \frac{32 \times (8192 + 4096)}{4096} = 96 \text{ tokens/step}$$

每步有 ~96 tokens 被驱逐 → 每步约 32/4096 ≈ 0.78% 的序列完成。

### Disaggregated Serving 比例计算

> 📋 **背景知识：Disaggregated Serving（分离式推理）**
>
> 传统推理：Prefill 和 Decode 在同一组 GPU 上交替执行。
> 分离式推理：专门的 Prefill 服务器处理 prompt，通过网络将 KV cache 传给专门的 Decode 服务器。
>
> 优势：
> - Prefill 服务器可选高算力卡（如 H100 SXM），Decode 服务器可选高带宽/大内存卡
> - 避免 prefill 的突发计算抢占 decode 的 GPU 时间
> - 两者可以独立扩缩容

设 $P$ 为 prefill 服务器数，$G$ 为 generate 服务器数。要保持流水线平衡：

$$\frac{P}{T_{\text{prefill}}} = \frac{B \times G}{T_{\text{decode}} \times S_{\text{median\_decode}}}$$

代入数值（bf16, 16 片 TPU v5e）：
- $T_{\text{prefill}}$ = 910 ms（8K prompt）
- $T_{\text{decode}}$ = 19 ms/step
- $S_{\text{median\_decode}}$ = 512 tokens
- $B$ = 32

$$\frac{P}{0.91} = \frac{32 \times G}{0.019 \times 512}$$

$$P = 3G$$

**需要 3× 的 prefill 服务器才能匹配 1× 的 generate 服务器！** 这说明 prefill 是整个系统的瓶颈，也解释了为什么 prefill 优化（如 speculative prefilling、prompt caching）如此重要。

---

## 12.8 Roofline 计算代码

以下 Python 代码实现了本章的延迟/吞吐量分析（基于原书代码，改为 GPU 版本）：

```python
import numpy as np

# === 硬件参数 ===
num_chips = 8                # GPU 数量
bytes_per_param = 1          # int8 = 1 byte/param
param_count = 70e9
param_size = bytes_per_param * param_count
sequence_length = 8192       # 可调整

hbm_bandwidth = 3.35e12      # H100 HBM 带宽 (bytes/s)
flops = 9.9e14               # H100 bf16 FLOPs/s

# === KV Cache 大小 ===
def kv_cache_size(bs, seq_len=sequence_length):
    """每 batch 的 KV cache 总字节数（int8）"""
    return 2 * bs * 128 * 8 * 80 * seq_len  # 2 * K_heads * H * L * S * 1byte

# === 最小拓扑 ===
def min_topology(total_bytes, hbm_per_chip=80e9):
    """最少需要多少芯片"""
    return int(2 ** np.ceil(np.log2(total_bytes / hbm_per_chip)))

# === 最大 Batch Size ===
def get_max_batch_size(num_chips, seq_len, param_size, hbm_per_chip=80e9):
    total_hbm = num_chips * hbm_per_chip
    remaining = total_hbm - param_size
    kv_per_seq = 2 * 128 * 8 * 80 * seq_len  # 160KB/token * seq_len
    return int(remaining / kv_per_seq)

max_bs = get_max_batch_size(num_chips, sequence_length, param_size)
batch_sizes = np.arange(1, max_bs + 1)

# === 延迟分解 ===
kv_sizes = np.array([kv_cache_size(b) for b in batch_sizes])

# Attention: 始终 bandwidth-bound
kv_comms_time = kv_sizes / (num_chips * hbm_bandwidth)

# MLP 参数加载时间: 与 batch 无关
param_comms_time = np.full_like(batch_sizes, param_size / (num_chips * hbm_bandwidth), dtype=float)

# MLP FLOPs 时间
flops_time = 2 * param_count * batch_sizes / (num_chips * flops)

# MLP 取 max(FLOPs, 参数加载)
mlp_time = np.maximum(flops_time, param_comms_time)

# 总延迟 = Attention + MLP
latency_ms = 1000 * (mlp_time + kv_comms_time)

# 吞吐量 (tokens/s/chip)
throughput_per_chip = batch_sizes / (latency_ms / 1000) / num_chips

print(f"Max batch size: {max_bs}")
print(f"Min latency: {latency_ms[0]:.1f} ms")
print(f"Max throughput: {throughput_per_chip[-1]:.1f} tokens/s/chip")
```

> 💡 **练习建议**
>
> 修改上面代码的参数，回答以下问题：
> 1. 将 `sequence_length` 从 8192 改为 32768，观察 max batch size 和延迟的变化
> 2. 将 `bytes_per_param` 改为 2（bf16），观察 min latency 和 max throughput 的变化
> 3. 将 `num_chips` 从 8 改为 4，观察 Pareto 曲线的移动

---

## 12.9 SGLang 部署 LLaMA 3-70B

> 🛠️ **实践：SGLang 部署**
>
> ### 基础部署
>
> ```bash
> # bf16，8 卡张量并行
> python -m sglang.launch_server \
>   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
>   --tp-size 8 \
>   --port 30000
> ```
>
> ### 优化部署
>
> ```bash
> python -m sglang.launch_server \
>   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
>   --tp-size 8 \
>   --port 30000 \
>   --mem-fraction-static 0.85 \        # 85% HBM 给权重+cache
>   --chunked-prefill-size 8192 \       # chunked prefill
>   --max-running-requests 256 \        # 最大并发
>   --context-length 8192               # 最大上下文长度
> ```
>
> ### FP8 量化部署（降低成本）
>
> ```bash
> python -m sglang.launch_server \
>   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
>   --tp-size 4 \                       # fp8 更小，4 卡够了
>   --quantization fp8 \
>   --port 30000
> ```

### 关键调优参数

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--tp-size` | 张量并行度 | bf16: 8, fp8: 4, int4: 2 |
| `--mem-fraction-static` | 权重内存比例 | 0.8-0.9 |
| `--max-running-requests` | 最大并发数 | 取决于 KV cache 内存 |
| `--chunked-prefill-size` | Prefill chunk 大小 | 4096-16384 |
| `--context-length` | 最大上下文长度 | 按需设置 |
| `--schedule-policy` | 调度策略 | "lpm" (longest prefix match) |

### Continuous Batching 与 Chunked Prefill

SGLang 使用 continuous batching：当一个序列完成生成时，立即将新请求加入 batch，而不是等整个 batch 完成。结合 chunked prefill：

```
时间轴：
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Decode×32 │  │ Decode×31│  │ Decode×32│
  │           │  │+Prefill  │  │          │
  │ (19ms)    │  │ chunk    │  │ (19ms)   │
  └──────────┘  │ (25ms)   │  └──────────┘
                └──────────┘
                ↑ 序列 #7 完成，
                  新请求 #33 的 prefill
                  chunk 穿插执行
```

**优势**：避免长 prompt 的 prefill 阻塞正在生成的请求。`--chunked-prefill-size` 控制每个 chunk 的 token 数，越小则 decode 延迟抖动越小，但 prefill 总时间越长（更多 chunk 开销）。

### RadixAttention 与前缀共享

SGLang 的 RadixAttention 将 KV cache 组织为前缀树（Radix Tree）：

```
Radix Tree 示例（3 个请求共享系统 prompt）：
                 [System Prompt KV] ← 只存一份
                 /        |        \
         [User A的       [User B的     [User C的
          prompt KV]      prompt KV]    prompt KV]
```

多个请求共享相同前缀的 KV cache → 内存节省巨大。设系统 prompt 2048 tokens，每请求节省 2048 × 160 KB = 320 MB（int8）。100 并发 → 节省 **32 GB** KV cache 内存！

> 🛠️ **实践：Mini-SGLang 架构对照**
>
> Mini-SGLang（`/Users/huabin/mini-sglang-main/`）是 SGLang 的教学简化版，核心文件映射：
>
> | SGLang 概念 | Mini-SGLang 文件 | 对应本章概念 |
> |------------|-----------------|------------|
> | Scheduler | `sglang/scheduler.py` | Prefill/Decode 调度 (12.7) |
> | KV Cache 管理 | `sglang/kvcache/` | KV cache 内存预算 (12.2) |
> | RadixCache | `tutorials/day06-07/` | 前缀共享 |
> | TP Engine | `sglang/engine.py` | TP 分片 (12.6) |
> | Continuous Batching | `sglang/scheduler.py` | Batch 策略 (12.4) |
>
> Mini-SGLang 的 `--tp 4` 部署示例：
>
> ```bash
> python -m sglang.launch_server \
>   --model-path meta-llama/Llama-3-8B-Instruct \
>   --tp 4 \
>   --port 30000
> ```
>
> 对照 12.4 的分析：TP=4 时每卡加载 8B/4 = 2B 参数 = 4 GB（bf16），decode 延迟下界 = 4 GB / 3.35 TB/s ≈ 1.2 ms/token。

### 性能监控

```python
import requests
metrics = requests.get("http://localhost:30000/get_server_info").json()

# 关键指标及其对应的本章分析：
# cache_hit_rate  → 前缀共享命中率，越高越好
# num_running_reqs → 当前 batch size（对照 12.4 的临界值分析）
# token_usage     → KV cache 使用率（对照 12.2 的内存预算）
# avg_prefill_latency → TTFT（对照 12.3 的 prefill 延迟公式）
# avg_decode_latency  → TPOT（对照 12.3 的 decode 延迟公式）
```

### 常见问题排查

| 问题 | 诊断方法 | 解决方案 |
|------|---------|---------|
| OOM | `token_usage` 接近 100% | 减小 `--max-running-requests` 或增大 TP |
| TTFT 高 | `avg_prefill_latency` 大 | 增大 `--chunked-prefill-size` 或增加 GPU |
| 吞吐量低 | `num_running_reqs` 远小于临界 BS | 增加并发请求 |
| Cache hit 低 | `cache_hit_rate` < 50% | 启用 `--schedule-policy lpm` |
| Decode 延迟抖动 | TPOT 方差大 | 减小 `--chunked-prefill-size` |

---

## 12.10 Worked Problems

### Q1：LLaMA 3-405B 的每 token FLOPs 与最小延迟

LLaMA 3-405B 有 405B 参数。

**(a)** 每个 forward pass 的 per-token FLOPs？

<details markdown="1">
<summary>点击查看答案</summary>

$$\text{FLOPs/token} = 2P = 2 \times 405 \times 10^9 = 810 \text{ GFLOPs/token}$$

</details>

**(b)** 假设 FLOPs-bound，在 $N$ 片 TPU v5e 上单次 forward pass 的延迟下界？

<details markdown="1">
<summary>点击查看答案</summary>

$$T_{\text{lower bound}} = \frac{2P}{N \times C} = \frac{810 \times 10^9}{N \times 1.97 \times 10^{14}} = \frac{4.1}{N} \text{ ms}$$

N=16: 0.26 ms；N=32: 0.13 ms。这是理想的 FLOPs-bound 下界。

</details>

**(c)** 假设 comms-bound（纯 HBM 带宽），int8 权重，延迟下界？

<details markdown="1">
<summary>点击查看答案</summary>

$$T_{\text{lower bound}} = \frac{P_{\text{bytes}}}{N \times W_{\text{HBM}}} = \frac{405 \times 10^9}{N \times 8.1 \times 10^{11}} = \frac{500}{N} \text{ ms}$$

N=16: 31 ms；N=32: 15.6 ms。**comms-bound 下界远大于 FLOPs-bound**——405B 的 decode 在小 batch 时严重 memory-bound。

</details>

---

### Q2：LLaMA 3-8B 的内存分析

LLaMA 3-8B，int8 权重 + int8 KV，BS=240，序列长度 8192。

**(a)** 参数内存？

<details markdown="1">
<summary>点击查看答案</summary>

$$P_{\text{bytes}} = 8 \times 10^9 \times 1 = 8 \text{ GB}$$

</details>

**(b)** KV cache 内存？（LLaMA 3-8B: L=32, K_v_heads=8, H=128）

<details markdown="1">
<summary>点击查看答案</summary>

$$\text{KV/token} = 2 \times 32 \times 8 \times 128 = 65,536 \text{ bytes} = 64 \text{ KB/token}$$

$$\text{KV total} = 64 \times 10^3 \times 8192 \times 240 = 126 \text{ GB}$$

</details>

**(c)** 峰值工作激活内存（粗略估计）？

<details markdown="1">
<summary>点击查看答案</summary>

峰值激活大约是一层的中间张量：$B \times D \times F$ bytes ≈ $240 \times 4096 \times 14336 \times 2$ ≈ 28 GB（bf16）。使用 Flash Attention 可以将 attention 的激活压缩到 O(BS) 而非 O(BS²)。

总内存 ≈ 8 + 126 + 28 = **162 GB**。

</details>

**(d)** 最小 TPU v5e 拓扑？

<details markdown="1">
<summary>点击查看答案</summary>

$$\text{min chips} = \lceil 162 / 16 \rceil = 11 \rightarrow \text{向上取到 } 4 \times 4 = 16 \text{ 片}$$

注意 KV cache 主导内存（占 78%）！如果减小 batch 到 64，KV 变为 33 GB，总 69 GB → 4×2 (8 片) 就够了。

</details>

---

### Q3：LLaMA 3-405B Serving 配置设计

假设 int8 权重 + bf16 FLOPs，TPU v5e，延迟硬限制 15 ms/token。

**(a)** 理论最小延迟（batch=1, int8 权重）在 $N$ 片上？

<details markdown="1">
<summary>点击查看答案</summary>

$$T_{\min} = \frac{405 \times 10^9}{N \times 8.1 \times 10^{11}} = \frac{500}{N} \text{ ms}$$

要达到 15 ms 需要 $N > 500/15 = 33.3$ → 至少 **4×8 = 32 片**（还超了一点）。实际需要 **8×8 = 64 片** 才有充裕余量。

</details>

**(b)** 在 64 片上，满足 15 ms 约束的最大吞吐量配置？

<details markdown="1">
<summary>点击查看答案</summary>

64 片总 HBM = 1024 GB。权重 405 GB → 剩余 619 GB 给 KV cache。

KV/token for 405B（L=126, K=8, H=128）：$2 \times 126 \times 8 \times 128 = 258$ KB/token。

8K 序列：每请求 KV = 258 KB × 8192 = 2.1 GB。最大 batch ≈ 619 / 2.1 ≈ **294**。

检查延迟：int8 权重 + bf16 FLOPs 的临界 BS = 120。BS=294 > 120 → compute-bound！

$$T_{\text{FLOPs}} = \frac{2 \times 294 \times 405 \times 10^9}{64 \times 1.97 \times 10^{14}} = 18.9 \text{ ms}$$

超过 15 ms！需要减小 batch。设 $T = 15$ ms：

$$B_{\max} = \frac{T \times N \times C}{2P} = \frac{0.015 \times 64 \times 1.97 \times 10^{14}}{2 \times 405 \times 10^9} = 233$$

$$\text{QPS/chip} = \frac{233}{0.015 \times 512 \times 64} = 0.47$$

</details>

**(c)** 绝对最小每步延迟？

<details markdown="1">
<summary>点击查看答案</summary>

在 batch=1（memory-bound）、最大芯片数时：

$$T_{\min} = \frac{P_{\text{bytes}}}{N \times W_{\text{HBM}}} = \frac{405 \times 10^9}{256 \times 8.1 \times 10^{11}} = 1.95 \text{ ms}$$

理论上 256 片 TPU v5e 可以达到 ~2 ms/token 的延迟。但这非常不经济（QPS/chip 极低）。

</details>

---

### Q4：GPU vs TPU 成本效率比较

LLaMA 3-70B，int8 权重，8K 上下文，目标 1000 QPS。

**(a)** H100 方案需要多少卡？

<details markdown="1">
<summary>点击查看答案</summary>

每个 8 卡节点（TP=8）：decode 延迟 ≈ 5.2 ms（权重加载）。中位 decode 512 tokens → 每节点完成一个请求需 5.2 × 512 = 2.66 s。

8 卡节点的 max batch ≈ (640 - 70) / (8192 × 160e-6) ≈ 435。

QPS/节点 = 435 / 2.66 = **163 QPS**。

1000 QPS 需要 1000/163 ≈ 7 个节点 = **56 张 H100**。

成本：56 × $10.8/h = **$604/h**。

</details>

**(b)** TPU v5e 方案需要多少芯片？

<details markdown="1">
<summary>点击查看答案</summary>

每个 4×4 拓扑（16 片）：max batch ≈ 43。decode 延迟 ≈ 19 ms → 完成一个请求需 19 × 512 = 9.73 s。

QPS/拓扑 = 43 / 9.73 = **4.4 QPS**。

1000 QPS 需要 1000/4.4 ≈ 227 个拓扑 = **3632 片 TPU v5e**。

成本：3632 × $1.2/h = **$4358/h**。

等等——这比 H100 贵 7×？问题在于 TPU v5e 的 HBM 太小（16 GB），batch size 受限。如果用 8×8 拓扑（64 片），max batch ≈ 550，QPS/拓扑 = 550 / 9.73 / 4 = 14.1 QPS → 71 个拓扑 = 4544 片，$5453/h。

**结论**：当 KV cache 主导内存时，HBM/chip 大的 GPU 可能比 FLOPs/$ 高的 TPU 更经济。**选择硬件不能只看 FLOPs/$，还要看 HBM/$！**

</details>

---

## 关键要点

| 概念 | 要点 |
|------|------|
| 硬件选型 | FLOPs/$ 是第一指标，但 HBM/$ 对推理同样关键 |
| KV Cache | LLaMA 70B 每 token 160 KB (int8)，长上下文下主导内存和带宽 |
| Decode 延迟下界 | = 模型字节数 / (HBM 带宽 × GPU 数)，bf16 @8xH100 ≈ 5.2 ms |
| 临界 Batch Size | bf16: ~295, int8 权重 + bf16 FLOPs: ~148, int4: ~74 |
| 最小 ≠ 最优拓扑 | 加芯片增加 KV cache 空间 → 更大 batch → 更高利用率 |
| TP 上限 | GPU: 8（NVLink），TPU: 16-32（ICI）；小 batch 可放宽 |
| Disaggregated Serving | 需要 ~3× 的 prefill 服务器匹配 generate 服务器 |
| 量化 | 同时降低延迟、增大 batch、减少 TP 需求——推理中的万能优化 |
| SGLang 部署 | `--tp-size` + `--mem-fraction-static` + `--chunked-prefill-size` |

---

## 进一步阅读

- [原书 Chapter 8: Serving LLaMA 3-70B on TPUs](https://jax-ml.github.io/scaling-book/applied-inference)
- [SGLang 官方文档](https://sglang.readthedocs.io/)
- [SGLang 论文 (Zheng et al., 2024)](https://arxiv.org/abs/2312.07104)
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180)
- [LLaMA 3 模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
- [Splitwise: Disaggregated Serving](https://arxiv.org/abs/2311.18677)

