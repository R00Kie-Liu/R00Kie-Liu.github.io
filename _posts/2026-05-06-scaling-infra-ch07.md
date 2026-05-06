---
layout: post
title: "Scaling Book 入门第 7 章：Transformer FLOPs/参数量/内存精确计算"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '训练', 'Megatron']
---

# Scaling Book 入门第 7 章：Transformer FLOPs/参数量/内存精确计算

> **本章目标**：精确计算 Transformer 每一层的参数量、FLOPs 和内存占用，建立"Transformer = 一系列已知大小的矩阵乘法"的量化直觉。
>
> **对应原书**：Chapter 4 (All the Transformer Math You Need to Know)  
> **优先级**：⭐⭐ 中 | **建议时间**：Day 6, 约 2.5 小时

---

## 7.1 Transformer 层的解剖

> 🔗 **与你的联系**
>
> 你做模型架构设计时一定计算过 FLOPs（比如用 `6ND` 估算训练 FLOPs）。这一章将精确到 Transformer 的**每一个矩阵乘法**，让你理解 FLOPs 到底花在了哪里，内存被谁占用。这对你设计新架构和评估训练成本都直接有用。

一个标准 Transformer 层由两部分组成：

![Transformer 层结构](/assets/scaling-book/img/transformer-diagram.png)

### 符号约定

| 符号 | 含义 | 典型值 (LLaMA 70B) |
|------|------|-------------------|
| D | 隐藏维度 (d_model) | 8192 |
| F | FFN 中间维度 (d_ff) | 28672 |
| H | 注意力头数 | 64 |
| K | 每头维度 (d_head = D/H) | 128 |
| B | batch 中的 token 数 | 取决于配置 |
| S | 序列长度 | 4096-8192 |
| L | 层数 | 80 |
| V | 词表大小 | 32000 |

---

## 7.2 参数量计算

### Attention 层

每个 Attention 层有 4 个权重矩阵：

| 矩阵 | 形状 | 参数量 |
|------|------|--------|
| Q 投影 (Wq) | [D, D] | D² |
| K 投影 (Wk) | [D, Kv_heads × K] | D × Kv_heads × K |
| V 投影 (Wv) | [D, Kv_heads × K] | D × Kv_heads × K |
| Output 投影 (Wo) | [D, D] | D² |

对于标准 MHA（Multi-Head Attention）：Kv_heads = H，参数量 = **4D²/层**

对于 GQA（Grouped Query Attention，LLaMA 使用）：Kv_heads < H，K/V 投影更小。

### FFN 层

![FFN 结构](/assets/scaling-book/img/transformer-ffw.png)

标准 FFN（SwiGLU，LLaMA 使用）：

| 矩阵 | 形状 | 参数量 |
|------|------|--------|
| Gate 投影 (W_gate) | [D, F] | DF |
| Up 投影 (W_up) | [D, F] | DF |
| Down 投影 (W_down) | [F, D] | DF |

FFN 参数量 = **3DF/层**

### 总参数量

$$P \approx L \times (4D^2 + 3DF) + VD$$

最后的 VD 是 embedding 层和 output head。

> 📋 **背景知识：为什么 F 通常是 D 的 3-4 倍**
>
> 早期 Transformer（如 GPT-2）使用 F = 4D。现代模型使用 SwiGLU 激活函数，由于多了一个 gate 矩阵（3 个 [D,F] 而非 2 个 [D,F]），通常设 F ≈ 8D/3 来保持总 FLOPs 不变。
>
> LLaMA 70B：D=8192, F=28672 ≈ 3.5D

---

## 7.3 FLOPs 计算

> 📋 **背景知识：矩阵乘法的 FLOPs 计算**
>
> `A[M,K] × B[K,N]` 的 FLOPs = `2×M×K×N`
> - M×N 个输出元素，每个需要 K 次乘法和 K-1 次加法 ≈ 2K 次运算

### 前向传播 FLOPs

**每层 Attention**：
- Q/K/V 投影：`3 × 2BD²` = `6BD²`（标准 MHA）
- Attention score：`2B×S×D`（Q×Kᵀ）
- Attention × V：`2B×S×D`
- Output 投影：`2BD²`
- **Attention 总计**：`8BD² + 4BSD`

**每层 FFN**（SwiGLU）：
- Gate + Up + Down：`3 × 2BDF` = `6BDF`
- **FFN 总计**：`6BDF`

**整个模型前向传播**：

$$\text{FLOPs}_{\text{forward}} \approx L \times (8BD^2 + 4BSD + 6BDF) + 2BDV$$

当 S 远小于 D 时（常见于预训练），Attention score 项可忽略：

$$\text{FLOPs}_{\text{forward}} \approx L \times B \times (8D^2 + 6DF) \approx 2BP$$

其中 P 是总参数量。这就是著名的 **"前向 FLOPs ≈ 2 × tokens × params"** 规则。

### 反向传播 FLOPs

反向传播 FLOPs ≈ 前向的 **2×**（计算梯度 + 梯度对权重的乘法）。

### 训练总 FLOPs

$$\text{FLOPs}_{\text{training}} \approx 6 \times N \times P$$

其中 N 是训练的总 token 数，P 是参数量。

> 🔗 **与你的联系**
>
> `C = 6NP` 就是 Scaling Law 论文中常用的训练 FLOPs 估算公式。现在你知道 6 是怎么来的了：前向 2P + 反向 4P = 6P FLOPs/token。

---

## 7.4 内存占用

### 模型权重

| 精度 | 每参数字节数 | 70B 模型 |
|------|-------------|---------|
| fp32 | 4 | 280 GB |
| bf16 | 2 | 140 GB |
| int8 | 1 | 70 GB |
| int4 | 0.5 | 35 GB |

### 训练时的完整内存

| 组件 | 每参数字节数 | 70B 模型 |
|------|-------------|---------|
| 权重 (bf16) | 2 | 140 GB |
| 梯度 (bf16) | 2 | 140 GB |
| 优化器状态 (Adam, fp32) | 8 | 560 GB |
| **小计** | **12** | **840 GB** |

加上激活值（取决于 batch size 和是否使用 gradient checkpointing），总内存可达 **1+ TB**。

### KV Cache（推理时）

每层每 token 的 KV cache：

$$\text{KV cache/token/layer} = 2 \times 2 \times K_v\_\text{heads} \times K = 4 \times K_v\_\text{heads} \times K \text{ bytes (bf16)}$$

对 LLaMA 70B（GQA，Kv_heads=8，K=128）：
- 每 token 每层：4 × 8 × 128 = 4096 bytes = 4 KB
- 80 层，序列长度 4096：4 KB × 80 × 4096 = **1.3 GB/序列**

> 📋 **背景知识：Gradient Checkpointing（梯度检查点/重计算）**
>
> 正常训练：保存所有层的激活值用于反向传播 → 内存 ∝ 层数 × batch size
> Gradient Checkpointing：只保存部分层的激活值，反向时重新计算 → 内存大幅减少，计算增加 ~33%
>
> 在 Megatron 中通过 `--recompute-activations` 或 `--recompute-granularity full` 启用。

---

## 7.5 MoE（Mixture of Experts）的特殊性

![MoE 架构](/assets/scaling-book/img/moe.png)

MoE 将 FFN 替换为多个"expert"，每个 token 只路由到 top-k 个 expert：

- **参数量增加**：如果有 E 个 expert，FFN 参数 × E
- **FLOPs 不变**：每个 token 只经过 k 个 expert
- **通信增加**：需要 AllToAll 将 token 路由到持有对应 expert 的设备

这就是 MoE 模型（如 Mixtral、DeepSeek-V3）能用更少 FLOPs 达到更好效果的原因：更多参数 = 更多知识存储，更少 FLOPs = 每 token 只激活一部分。

---

## 7.6 Flash Attention

> 📋 **背景知识：Flash Attention 的核心思想**
>
> ![Flash Attention 算法](/assets/scaling-book/img/flash-algo.png)
>
> 标准 Attention 的问题：
> 1. 计算 `S = Q×Kᵀ` 产生 [B, H, S, S] 的矩阵 → S=8192 时需要 ~1 GB/head
> 2. 这个大矩阵必须写入 HBM 再读回来做 softmax → 大量 HBM 读写
>
> Flash Attention 的解决方案：
> 1. 将 Q, K, V 分成小 tile
> 2. 在 SMEM/VMEM 中完成 tile 级的 attention 计算
> 3. 使用 online softmax 避免存储完整的 S×S 矩阵
> 4. 结果：**减少 HBM 读写量从 O(S²) 到 O(S)**
>
> 效果：2-4× 加速，大幅减少内存使用。

---

## 关键要点

- [ ] Transformer 参数量 ≈ L × (4D² + 3DF)，FLOPs/token ≈ 2P（前向）
- [ ] 训练总 FLOPs = 6NP（前向 2 + 反向 4）
- [ ] 训练内存 ≈ 每参数 12-20 bytes（权重 + 梯度 + 优化器 + 激活值）
- [ ] KV cache 是推理时的主要内存开销，∝ 序列长度 × 层数
- [ ] MoE：参数多但 FLOPs 少，代价是 AllToAll 通信
- [ ] Flash Attention 通过 tiling 减少 HBM 访问，不改变 FLOPs 总量

---

## 进一步阅读

- 原书 Chapter 4: All the Transformer Math You Need to Know
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [GQA 论文](https://arxiv.org/abs/2305.13245)

