---
layout: post
title: "Scaling Book 入门第 5 章：集合通信原语"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '硬件基础']
---

# Scaling Book 入门第 5 章：集合通信原语

> **本章目标**：理解分布式训练/推理中最常用的几种通信操作（AllReduce、AllGather、ReduceScatter、AllToAll），以及它们的时间开销。
>
> **对应原书**：Chapter 3 (Sharded Matrices) 的通信原语部分  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 4, 约 2 小时

---

## 5.1 为什么需要集合通信

> 🔗 **与你的联系**
>
> 你做 CV 分布式训练时一定用过 Data Parallel：每张卡有完整的模型副本，各自算梯度，然后做一次 **AllReduce** 把梯度求平均。这个 AllReduce 就是最简单的集合通信原语。
>
> LLM 训练中，由于模型太大无法放在单卡上，需要更多种类的通信：切分权重后要 AllGather 拼回来，切分激活值后要 ReduceScatter 合并结果。理解这些原语是理解所有并行策略的基础。

---

## 5.2 四种核心通信原语

假设有 N 个设备，每个设备持有一份数据。

### AllGather

**功能**：每个设备持有一个分片，通信后每个设备拥有**完整数据**。

![AllGather 动画](/assets/scaling-book/img/all-gather.gif)

```
通信前：设备0=[A], 设备1=[B], 设备2=[C], 设备3=[D]
通信后：设备0=[A,B,C,D], 设备1=[A,B,C,D], ...（每个设备都有全部）
```

- **通信量**：每设备发送自己的分片（大小 S/N），接收其余 N-1 个分片
- **总字节数/设备**：`S × (N-1)/N ≈ S`（接收端），发送量相同
- **时间**：$T_{\text{AllGather}} = \frac{S}{B_{\text{link}}}$（在带宽为 $B_{\text{link}}$ 的环上）

### ReduceScatter

**功能**：每个设备持有完整数据，通信后每个设备持有**归约后的一个分片**（如求和后的 1/N）。

![ReduceScatter 动画](/assets/scaling-book/img/reduce-scatter.gif)

```
通信前：设备0=[a₀,a₁,a₂,a₃], 设备1=[b₀,b₁,b₂,b₃], ...
通信后：设备0=[a₀+b₀+c₀+d₀], 设备1=[a₁+b₁+c₁+d₁], ...
```

- **通信量和时间**：与 AllGather 相同

### AllReduce

**功能**：每个设备持有一份数据，通信后每个设备拥有**所有设备数据的归约结果**（如总和）。

> 📋 **背景知识：AllReduce = ReduceScatter + AllGather**
>
> ```
> 步骤1 (ReduceScatter)：每设备得到 sum 的一个分片
> 步骤2 (AllGather)：每设备收集到完整的 sum
> ```
>
> 因此 AllReduce 的通信量是 ReduceScatter 的 2 倍。
>
> 在 Ring AllReduce 实现中：
> - N 个设备组成环
> - 数据分 N 份，经过 N-1 步传递完成 ReduceScatter
> - 再经过 N-1 步完成 AllGather
> - 总通信量：`2S × (N-1)/N ≈ 2S`

### AllToAll

**功能**：每个设备将自己的数据分成 N 份，分别发送给 N 个设备。可以理解为"转置"通信。

![AllToAll 动画](/assets/scaling-book/img/all-to-all.gif)

```
通信前：设备0=[A₀,A₁,A₂,A₃], 设备1=[B₀,B₁,B₂,B₃], ...
通信后：设备0=[A₀,B₀,C₀,D₀], 设备1=[A₁,B₁,C₁,D₁], ...
```

- 用途：MoE 模型中将 token 路由到不同 expert
- 通信量：每设备发送 `S × (N-1)/N`

### 所有原语一览

![四种集合通信操作](/assets/scaling-book/img/all-collectives.png)

---

## 5.3 通信时间的计算

### 在 Torus（TPU）上

Ring-based 实现，N 个设备传输总大小 S 的数据：

$$T_{\text{AllGather}} = T_{\text{ReduceScatter}} = \frac{S}{B_{\text{ICI}}}$$

$$T_{\text{AllReduce}} = 2 \times \frac{S}{B_{\text{ICI}}}$$

其中 $B_{\text{ICI}}$ 是 ICI 的单向带宽（TPU v5p 约 90 GB/s/轴）。

注意：这假设了最优的 ring 路径。在 3D torus 中，可以利用多个轴同时通信。

### 在 NVLink（GPU）上

由于 NVSwitch 提供全带宽互联（非 ring），节点内 AllReduce 非常高效：

$$T_{\text{AllReduce}}^{\text{intra-node}} = \frac{2S}{B_{\text{NVLink}}} = \frac{2S}{900 \text{ GB/s}}$$

跨节点时退化为 InfiniBand 带宽，慢得多。

---

## 5.4 通信与计算的重叠

![非重叠 vs 重叠](/assets/scaling-book/img/not-overlapped.png)
![重叠示意](/assets/scaling-book/img/overlapped.png)

关键优化：在做矩阵乘法的同时进行通信。

**AG-matmul**（AllGather + MatMul 重叠）：

![AG-matmul 重叠动画](/assets/scaling-book/img/ag_matmul.gif)

1. AllGather 分 chunk 进行
2. 每收到一个 chunk，立即开始计算该 chunk 的矩阵乘法
3. 通信和计算交错进行

当 $T_{\text{math}} > T_{\text{comms}}$ 时，通信被完全掩盖，总时间 ≈ $T_{\text{math}}$。

> 🛠️ **实践：Megatron**
>
> Megatron 中的通信优化：
>
> 1. **NCCL 后端**：Megatron 使用 NCCL（NVIDIA Collective Communications Library）进行 GPU 间通信
>    - `NCCL_IB_DISABLE=0` 启用 InfiniBand
>    - `NCCL_SOCKET_IFNAME` 指定网络接口
>
> 2. **通信-计算重叠**：Megatron 支持 `--overlap-grad-reduce` 和 `--overlap-param-gather`
>    - 梯度 ReduceScatter 与反向传播计算重叠
>    - 参数 AllGather 与前向计算重叠
>
> 3. **Sequence Parallelism**：Megatron 的 SP 将 LayerNorm/Dropout 的计算分片到 TP 组内的设备上
>    - 使用 AllGather/ReduceScatter 替代 AllReduce
>    - 减少激活值的内存占用（每设备只存 1/TP 的激活值）

---

## 5.5 通信开销的直觉

一些有用的直觉：

**AllReduce 梯度（Data Parallelism）**：
- 通信量 ≈ 2 × 模型大小（与设备数无关！）
- 7B 模型，bf16：通信量 ≈ 2 × 14 GB = 28 GB
- 在 900 GB/s NVLink 上：~31 ms
- 在 50 GB/s InfiniBand 上：~560 ms

**AllGather 权重（FSDP）**：
- 通信量 ≈ 模型大小
- 可以和计算重叠

**AllToAll（MoE）**：
- 通信量取决于 token 路由分布
- 通常不能和计算重叠 → 可能成为瓶颈

---

## 关键要点

- [ ] AllGather：分片 → 每设备拥有完整数据
- [ ] ReduceScatter：完整数据 → 每设备持有归约后的分片
- [ ] AllReduce = ReduceScatter + AllGather，通信量 ≈ 2× 数据大小
- [ ] AllToAll：用于 MoE 的 token 路由
- [ ] 通信可以和计算重叠（AG-matmul），当计算量大于通信量时被掩盖
- [ ] AllReduce 的通信量与设备数无关（只取决于数据大小）
- [ ] Megatron 用 NCCL，支持 `--overlap-grad-reduce` 重叠通信

---

## 进一步阅读

- 原书 Chapter 3: A Deeper Dive into TPU Communication Primitives
- [NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/)
- [Megatron-LM 论文的通信优化部分](https://arxiv.org/abs/2205.05198)

