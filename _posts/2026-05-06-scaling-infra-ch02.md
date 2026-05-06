---
layout: post
title: "Scaling Book 入门第 2 章：性能分析基石 — Roofline 模型"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '硬件基础']
---

# Scaling Book 入门第 2 章：性能分析基石 — Roofline 模型

> **本章目标**：掌握 Roofline 模型——用一个简单框架判断任何算法在硬件上是"算力瓶颈"还是"带宽瓶颈"。
>
> **对应原书**：Chapter 1 (Rooflines)  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 2, 约 2 小时

---

## 2.1 核心问题：时间都去哪了？

一个算法在加速器上运行时，时间消耗只有两大来源：

1. **计算（Compute）**：做浮点运算需要时间
2. **通信（Communication）**：搬运数据需要时间

$$T_{\text{math}} = \frac{\text{总 FLOPs}}{\text{加速器 FLOPs/s}}$$

$$T_{\text{comms}} = \frac{\text{总搬运字节数}}{\text{带宽 (bytes/s)}}$$

**关键洞察**：计算和通信通常可以重叠（pipeline），所以：

$$T_{\text{lower bound}} = \max(T_{\text{math}}, T_{\text{comms}})$$

$$T_{\text{upper bound}} = T_{\text{math}} + T_{\text{comms}}$$

> 🔗 **与你的联系**
>
> 你在做预训练时一定关注过 MFU（Model FLOPs Utilization）。MFU 本质就是在衡量你的训练有多接近 compute-bound 的理想情况。如果 MFU 只有 30%，说明 70% 的时间硬件在等待数据搬运，计算单元空闲。Roofline 模型就是理解这一现象的分析工具。

---

## 2.2 算术强度（Arithmetic Intensity）

**定义**：一个算法每搬运 1 byte 数据能做多少次浮点运算。

$$\text{Arithmetic Intensity} = \frac{\text{总 FLOPs}}{\text{总搬运 Bytes}}$$

这个比值决定了算法是 compute-bound 还是 memory-bound：

- **高算术强度** → 计算多、搬运少 → **Compute-bound**（好！充分利用算力）
- **低算术强度** → 搬运多、计算少 → **Memory-bound**（差！算力浪费）

> 📋 **背景知识：Compute-bound vs Memory-bound 的直觉**
>
> 想象一个厨师（计算单元）和一个服务员（内存带宽）：
> - **Compute-bound**：菜上得很快，厨师忙不过来 → 瓶颈在厨师的手速
> - **Memory-bound**：厨师很快做完了，但要等下一道菜的食材送来 → 瓶颈在食材配送速度
>
> 对于 ML：
> - 大矩阵乘法 → 一次加载权重，做大量计算 → Compute-bound ✓
> - LayerNorm/ReLU → 加载每个元素，只做一次运算 → Memory-bound ✗
> - 小 batch 的矩阵乘法 → 加载所有权重但只做少量计算 → Memory-bound ✗

---

## 2.3 Roofline 图

![Roofline 模型](/assets/scaling-book/img/roofline-improved.png)

Roofline 图（对数坐标）的解读：
- **X 轴**：算法的算术强度（FLOPs/Byte）
- **Y 轴**：算法能达到的实际 FLOPs/s
- **水平线（屋顶）**：硬件的峰值 FLOPs/s
- **斜线（墙壁）**：带宽限制，斜率 = 带宽

**临界算术强度** = 峰值 FLOPs/s ÷ 带宽

- TPU v5e：`1.97×10¹⁴ / 8.1×10¹¹ ≈ 240` FLOPs/Byte
- H100：`9.9×10¹⁴ / 3.35×10¹² ≈ 295` FLOPs/Byte

算法的算术强度超过这个值 → Compute-bound（好）  
低于这个值 → Memory-bound（算力浪费）

---

## 2.4 矩阵乘法的 Roofline 分析

对于 `X[B, D] × Y[D, F] → Z[B, F]`：

- **FLOPs**：`2 × B × D × F`
- **Bytes**：加载 `2BD + 2DF` bytes，写回 `2BF` bytes（bf16 格式）
- **算术强度**：

$$\text{AI} = \frac{2BDF}{2BD + 2DF + 2BF} = \frac{BDF}{BD + DF + BF}$$

当 B 远小于 D 和 F 时（这在 Transformer 中很常见，D 和 F 通常 > 8000）：

$$\text{AI} \approx \frac{BDF}{DF} = B$$

**结论**：对于 TPU v5e，当 per-replica batch size > 240 tokens 时，矩阵乘法就是 compute-bound。

> 📋 **背景知识：为什么 batch size 是关键旋钮**
>
> 这里的 B 是 **token 数**（不是 sequence 数）：
> - batch_size = 32 sequences × 4096 tokens = 131,072 tokens
> - 如果用 128 张卡做 data parallelism，per-replica batch = 1024 tokens
> - 1024 > 240，所以 compute-bound ✓
>
> 但如果你的 per-replica batch 只有 64 tokens（比如做推理时只有单个 token），那就严重 memory-bound 了。这就是为什么推理的 generation 阶段和训练的性能特性完全不同。

---

## 2.5 向量操作的 Roofline

对比：向量点积 `x·y` 其中 x, y ∈ bf16[N]

$$\text{AI}(\text{dot product}) = \frac{2N - 1}{4N + 2} \rightarrow \frac{1}{2}$$

算术强度只有 0.5，远低于临界值 240。**永远是 memory-bound**。

这就是为什么 LayerNorm、ReLU、Softmax 等逐元素操作：
- 占比 FLOPs 很少
- 但可能占比时间不少（如果无法和其他操作 fuse 在一起）

---

## 2.6 网络通信的 Roofline

当模型分布在多芯片上时，出现新的瓶颈：**芯片间通信**。

例：将 `X[B, D] × Y[D, F]` 沿 D 维度切成 2 份，分别在 2 个 TPU 上计算：

- 每个 TPU 的 $T_{\text{math}} = \frac{BDF}{1.97 \times 10^{14}}$（计算量减半，但芯片也变 2 个）
- 通信 $T_{\text{comms}} = \frac{2BF}{4.5 \times 10^{10}}$（需要交换部分结果）

Compute-bound 条件：$\frac{D}{2} > \frac{1.97 \times 10^{14}}{4.5 \times 10^{10}} = 4377$，即 $D > 8755$

**注意**：在芯片间通信场景中，决定是否 compute-bound 的是**模型维度 D**，而非 batch size B！

> 🛠️ **实践：Megatron**
>
> 这直接解释了 Megatron 中 Tensor Parallelism 的一个核心限制：
> - TP 需要在每个 matmul 之后做 AllReduce（通信）
> - 当 hidden_size D 足够大时（> 8K），通信可以被计算掩盖
> - 但对于较小的模型（D < 4K），过多的 TP 会导致通信瓶颈
> - 这就是为什么 Megatron 推荐 TP=8（一个节点内），而不是 TP=64

---

## 2.7 低精度计算的影响

使用 int8 代替 bf16：
- 每个参数只需 1 byte（而非 2 bytes）→ 加载字节数减半
- FLOPs/s 翻倍（TPU v5e：int8 为 `3.94×10¹⁴` vs bf16 的 `1.97×10¹⁴`）
- 临界 batch size 变化不大（约 480 vs 240），但绝对速度提升约 2×

---

## 关键要点

- [ ] Roofline 模型：$T = \max(T_{\text{math}}, T_{\text{comms}})$
- [ ] 算术强度 = FLOPs / Bytes，决定了是 compute-bound 还是 memory-bound
- [ ] 矩阵乘法在 batch size > ~240 tokens 时是 compute-bound（对 TPU v5e）
- [ ] 向量操作（ReLU、LayerNorm）几乎总是 memory-bound
- [ ] 多芯片场景下，模型维度 D 决定 Tensor Parallelism 是否划算
- [ ] 低精度（int8/fp8）可以同时提升算力和减少搬运量
- [ ] MFU 本质是衡量你离 compute-bound 理想情况有多远

---

## 进一步阅读

- 原书 Chapter 1: All About Rooflines
- [Original Roofline Paper (Williams et al., 2009)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyworook.pdf)
- Megatron-LM 论文中的 MFU 计算方法

