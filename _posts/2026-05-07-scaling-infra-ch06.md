---
layout: post
title: "Scaling Book 入门第 6 章：分片矩阵乘法 — 分布式计算的核心"
date: 2026-05-07
tags: ['LLM', 'Infra', 'Scaling', '硬件基础']
---

# Scaling Book 入门第 6 章：分片矩阵乘法 — 分布式计算的核心

> **本章目标**：理解当矩阵被分片到多个设备上时，如何高效地完成矩阵乘法，以及不同分片方式对通信开销的影响。
>
> **对应原书**：Chapter 3 (Sharded Matrices and How to Multiply Them)  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 5, 约 3 小时

---

## 6.1 什么是分片（Sharding）

LLM 的参数太大，无法放在单个设备的 HBM 中。因此必须将矩阵"切"成多份，分布到多个设备上。

![分片示例](/assets/scaling-book/img/sharding-example.png)

> 🔗 **与你的联系**
>
> 你在 CV 中使用 Data Parallel 时，每张卡持有**完整的**模型参数副本。LLM 太大了，必须把参数本身也切开 — 这就是模型并行（Model Parallelism / Tensor Parallelism）的核心，也是为什么你需要理解分片矩阵乘法。

### 分片记号

原书使用一种简洁的下标记号：

- `A[I, J]`：未分片的矩阵，形状 I×J
- `A[Iₓ, J]`：沿 I 维度在 X 轴的设备上分片
- `A[I, Jᵧ]`：沿 J 维度在 Y 轴的设备上分片
- `A[Iₓ, Jᵧ]`：同时沿两个维度分片

例如，`A[Iₓ, J]` 表示矩阵 A 的行被均匀分配到 X 轴上的各设备，每个设备持有 `I/Nₓ` 行。

---

## 6.2 分片矩阵乘法的四种情况

考虑矩阵乘法 `C[B, F] = X[B, D] × W[D, F]`，其中 D 是收缩维度（contracting dimension）。

根据哪个维度被分片、以及被分片的是否是收缩维度，分为 4 种情况：

### Case 1：无收缩维度被分片

**场景**：`X[Bₓ, D] × W[D, Fₓ]` → 每设备独立计算自己的部分，**无需通信**。

![Case 1 示意](/assets/scaling-book/img/sharding-colored1.png)

```
设备0：X 的前半行 × 完整 W → C 的前半行
设备1：X 的后半行 × 完整 W → C 的后半行
```

这就是 **Data Parallelism** 中前向传播的模式：每设备处理不同的 batch 切片。

### Case 2：一个乘数的收缩维度被分片

**场景**：`X[B, Dₓ] × W[D, F]`（X 沿收缩维度 D 分片，W 未分片）

需要 **AllGather** X 或 **ReduceScatter** 结果。

![Case 2 示意](/assets/scaling-book/img/sharding-colored3.png)

选项 A：先 AllGather X 拼回完整，再本地 matmul：
- 通信：AllGather `2BD` bytes
- 计算：每设备完整 matmul

选项 B：本地 matmul 得到部分和，再 ReduceScatter：
- 计算：每设备 matmul `X_local × W_local`
- 通信：ReduceScatter `2BF` bytes

### Case 3：两个乘数的收缩维度都被分片

**场景**：`X[B, Dₓ] × W[Dₓ, F]`（两者都沿 D 分片到同一组设备上）

![Case 4 示意](/assets/scaling-book/img/sharding-colored4.png)

每设备做局部 matmul 得到**部分和**，然后 **AllReduce** 求总和：
- 每设备计算 `X_local × W_local` → `C_partial[B, F]`
- AllReduce 所有部分和 → `C[B, F]`
- 通信：AllReduce `4BF` bytes（= 2× ReduceScatter）

### Case 4：两个乘数的非收缩维度沿同一轴分片

**场景**：`X[Bₓ, D] × W[D, Fₓ]` → 局部 matmul 即可，**无需通信**！

![Case 1 变体](/assets/scaling-book/img/sharding-colored2.png)

每设备持有 X 的一些行和 W 的一些列，独立计算 C 的对应子块。

> 📋 **背景知识：为什么收缩维度的分片需要通信**
>
> 矩阵乘法 `C[i,j] = Σ_d X[i,d] × W[d,j]`。
> - 如果 d 维度被分片：每个设备只能算部分求和，需要把部分和加起来 → 需要通信
> - 如果 i 或 j 维度被分片：每个设备算的是 C 的不同行/列，彼此独立 → 不需要通信
>
> **核心原则**：分片收缩维度 = 需要 AllReduce/ReduceScatter；分片非收缩维度 = 无需通信。

---

## 6.3 通信代价总结

| 分片方式 | 需要的通信 | 通信量（bytes） |
|----------|-----------|----------------|
| 非收缩维度分片 | 无 | 0 |
| 一个收缩维度分片 | AllGather 或 ReduceScatter | ~2BD 或 ~2BF |
| 两个收缩维度分片 | AllReduce | ~4BF |

**关键洞察**：尽可能沿非收缩维度分片，避免通信开销。

---

## 6.4 Roofline 视角下的分片 matmul

对于 N 个设备的分片 matmul `X[B, D] × W[Dₓ, F]`：

- 计算时间：$T_{\text{math}} = \frac{2BDF}{N \times \text{FLOPs/s}}$
- 通信时间（AllReduce）：$T_{\text{comms}} = \frac{4BF}{B_{\text{link}}}$

Compute-bound 条件：

$$\frac{2BDF}{N \times \text{FLOPs/s}} > \frac{4BF}{B_{\text{link}}}$$

$$D > \frac{2N \times \text{FLOPs/s}}{B_{\text{link}}}$$

即 D 要足够大来"摊平"通信开销。加的设备越多（N 越大），D 需要越大才能保持 compute-bound。

---

## 6.5 Megatron 中的分片矩阵乘法

> 🛠️ **实践：Megatron**
>
> Megatron 的 Tensor Parallelism 本质就是分片矩阵乘法：
>
> ### Column Parallel Linear
>
> FFN 的第一层 `Y = XW₁`：将 W₁ 按**列**切分到 TP 个设备上
> ```
> W₁[D, F] → 每设备持有 W₁[D, F/TP]
> X[B, D] × W₁[D, F/TP] → Y[B, F/TP]（无需通信！）
> ```
> 这是 Case 1/4 — 分片非收缩维度，无需通信。
>
> ### Row Parallel Linear
>
> FFN 的第二层 `Z = YW₂`：将 W₂ 按**行**切分
> ```
> W₂[F, D] → 每设备持有 W₂[F/TP, D]
> Y[B, F/TP] × W₂[F/TP, D] → Z_partial[B, D]
> AllReduce(Z_partial) → Z[B, D]
> ```
> 这是 Case 3 — 收缩维度被分片，需要 AllReduce。
>
> ### 通信次数
>
> 一个 Transformer 层的前向传播中，Megatron 需要 **2 次 AllReduce**：
> 1. FFN 的 Row Parallel 之后
> 2. Attention 的 output projection 之后
>
> 反向传播中还需要额外 2 次，总共 4 次 AllReduce/层。
>
> 这就是为什么 TP 只适合节点内（NVLink 高带宽）：每层都有 4 次 AllReduce。

---

## 6.6 分片策略选择的直觉

| 场景 | 推荐分片方式 | 原因 |
|------|-------------|------|
| Data Parallelism | 沿 B 维度分片 | 每设备处理不同 batch，计算独立 |
| Tensor Parallelism (FFN) | W 按列/行交替分片 | 最小化 AllReduce 次数 |
| Tensor Parallelism (Attention) | 沿 head 维度分片 | 不同 head 独立计算 |
| FSDP | 沿某维度分片参数，计算前 AllGather | 节省内存，用通信换空间 |

---

## 关键要点

- [ ] 分片非收缩维度 → 无需通信（好！）
- [ ] 分片收缩维度 → 需要 AllReduce/ReduceScatter（有开销）
- [ ] Megatron 的 TP 用 Column Parallel（无通信）+ Row Parallel（AllReduce）交替
- [ ] 每个 Transformer 层需要 4 次 AllReduce（前向 2 次 + 反向 2 次）
- [ ] D 越大，TP 越容易保持 compute-bound（通信被计算掩盖）
- [ ] 加更多设备做 TP 需要更大的 D 来维持效率

---

## 进一步阅读

- 原书 Chapter 3: Sharded Matrices and How to Multiply Them
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Megatron-LM v2: Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)

