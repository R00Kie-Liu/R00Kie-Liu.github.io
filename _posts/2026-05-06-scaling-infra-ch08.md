---
layout: post
title: "Scaling Book 入门第 8 章：训练并行策略 — DP / FSDP / TP / PP"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '训练', 'Megatron']
---

# Scaling Book 入门第 8 章：训练并行策略 — DP / FSDP / TP / PP

> **本章目标**：掌握 LLM 训练的四大并行策略，理解每种策略的通信开销、适用场景和组合方式。这是本手册最核心的章节之一。
>
> **对应原书**：Chapter 5 (How to Parallelize a Transformer for Training)  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 7-8, 约 4 小时

---

## 8.1 Scaling 的目标

> 🔗 **与你的联系**
>
> CV 训练中你最熟悉的是 Data Parallelism：每张卡跑不同 batch，梯度 AllReduce 后更新。这在模型能放进单卡时完美工作。但对于 LLM：
> - 70B 参数 + Adam 优化器 ≈ 840 GB → 需要 11 张 H100 仅放参数
> - 想要快速训练 → 需要数千张卡
>
> 因此需要多种并行策略的组合。每种策略在"内存节省"、"通信开销"和"实现复杂度"之间做不同权衡。

**Strong scaling 的定义**：设备数增加 N 倍，训练速度也提升 N 倍。

现实中由于通信开销，实际 scaling 效率 < 100%。我们的目标是让 scaling 效率尽可能接近 100%。

---

## 8.2 Data Parallelism（DP）

![Data Parallelism](/assets/scaling-book/img/data-parallelism.png)

**核心思想**：每张卡持有完整的模型副本，处理不同的数据 batch。

**流程**：
1. 每张卡用不同数据做前向 + 反向，得到本地梯度
2. AllReduce 求梯度平均
3. 每张卡用相同的平均梯度更新参数

**通信分析**：
- 每步需要 AllReduce 全部梯度
- 通信量 = `2 × 模型大小`（不随设备数增加！）
- 时间：$T_{\text{comms}} = \frac{4P}{B_{\text{link}}}$（P 是参数字节数）

**Compute-bound 条件**：

$$\frac{6 \times B_{\text{local}} \times P_{\text{flops}}}{N \times \text{FLOPs/s}} > \frac{4P_{\text{bytes}}}{B_{\text{link}}}$$

其中 $B_{\text{local}}$ 是每设备的 token batch size。

简化：$B_{\text{local}} > \frac{2N \times \text{FLOPs/s}}{3 \times B_{\text{link}}}$

**要点**：
- 实现最简单
- 每卡需要存储完整模型 + 优化器（内存未节省）
- 设备越多，per-device batch 越小，可能变成 memory-bound
- 可以和计算重叠（边算反向边 AllReduce）

---

## 8.3 Fully-Sharded Data Parallelism（FSDP / ZeRO）

![FSDP 示意](/assets/scaling-book/img/fsdp.png)

**核心思想**：不再每卡存完整模型。将参数、梯度、优化器状态按设备数 N 分片，每卡只存 1/N。计算时按需 AllGather 拼回。

**流程**：
1. **前向**：AllGather 当前层权重 → 计算 → 丢弃非本地权重
2. **反向**：AllGather 当前层权重 → 计算梯度 → ReduceScatter 梯度
3. 每卡只更新自己持有的 1/N 参数

**通信分析**：
- 前向：每层 1 次 AllGather（通信量 ~P 字节）
- 反向：每层 1 次 AllGather + 1 次 ReduceScatter（通信量 ~2P 字节）
- **总通信量 ~3P**，比 DP 的 2P 多 50%

**内存节省**：

| 组件 | DP（每卡） | FSDP（每卡） |
|------|-----------|-------------|
| 参数 | 2P | 2P/N |
| 梯度 | 2P | 2P/N |
| 优化器 | 8P | 8P/N |
| **总计** | 12P | 12P/N |

**巨大的内存节省！** 128 卡的 FSDP 让每卡只需 1/128 的参数存储。

> 📋 **背景知识：ZeRO 的三个阶段**
>
> DeepSpeed 的 ZeRO 就是 FSDP 的不同级别：
> - **ZeRO-1**：只分片优化器状态 → 内存 = 4P/N + 4P
> - **ZeRO-2**：分片优化器 + 梯度 → 内存 = 2P/N + 2P
> - **ZeRO-3**：分片所有（= FSDP）→ 内存 = 12P/N
>
> ZeRO-3/FSDP 的通信量最大但内存节省最多。

> 🛠️ **实践：Megatron**
>
> Megatron 使用 `--use-distributed-optimizer` 启用类似 ZeRO-1 的优化器分片：
> - 每张卡只存储 1/DP_size 的优化器状态
> - 对于 70B 模型，优化器从 560 GB 降到 560/DP_size GB
> - 结合 TP+PP 后，实际 DP_size 可能只有 4-16，优化器内存仍显著减少

---

## 8.4 Tensor Parallelism（TP）

![Tensor Parallelism](/assets/scaling-book/img/model-parallelism.png)

**核心思想**：将单个 matmul 切分到多个设备上并行计算（参考第6章的分片矩阵乘法）。

**Megatron 风格的 TP**（FFN 层）：
1. **Column Parallel**：W₁ 按列切分 → `X × W₁_local` → 无需通信
2. **Row Parallel**：W₂ 按行切分 → `Y_local × W₂_local` → **AllReduce**

**通信分析**：
- 每层前向：2 次 AllReduce（FFN 1 次 + Attention 1 次）
- 每层反向：2 次 AllReduce
- 每次 AllReduce 通信量 = `4BD`（B 是 token batch，D 是 hidden dim）
- **总通信量/层** = `16BD`

**Compute-bound 条件**：

需要 D 足够大以掩盖通信。对 TPU v5p ICI：

$$D > \frac{2 \times TP \times \text{FLOPs/s}}{B_{\text{ICI}}}$$

对 GPU NVLink：类似，但带宽更高所以临界 D 更小。

**要点**：
- 通信频率高（每层每 matmul 都要通信）→ 需要**高带宽**互联
- 通常限制在节点内（NVLink），TP ≤ 8
- 减少了内存（每卡只存 1/TP 的参数），也减少了计算
- 天然降低了 per-device batch 的要求

> 🛠️ **实践：Megatron**
>
> ```bash
> # Tensor Parallelism 配置
> --tensor-model-parallel-size 8  # 通常 = 节点内 GPU 数
> ```
>
> **Sequence Parallelism**（Megatron v2 引入）：
> - TP 只分片了 matmul 的计算，但 LayerNorm / Dropout 仍在每卡上对完整激活值计算
> - SP 将这些操作也沿序列维度分片
> - 使用 AllGather（TP → SP 过渡）和 ReduceScatter（SP → TP 过渡）
> - `--sequence-parallel` 启用
> - 好处：激活值内存减少到 1/TP

---

## 8.5 Pipeline Parallelism（PP）

**核心思想**：将模型按层切分，不同设备负责不同层。数据在设备间流水线式传递。

**流程**：
1. 将 L 层分成 PP 组，每组 L/PP 层
2. 将 batch 切成多个 micro-batch
3. Micro-batch 按流水线顺序经过各 stage

![Pipeline Parallelism bubble](/assets/scaling-book/gpu/pipeline-bubble.png)

**Bubble 问题**：

流水线有"填充"和"排空"阶段，期间部分设备空闲。

$$\text{Bubble ratio} = \frac{PP - 1}{M + PP - 1}$$

其中 M 是 micro-batch 数。M 越大，bubble 越小。

**通信分析**：
- 只需点对点传输激活值（一个 `bf16[B_micro, D]` 张量）
- 通信量远小于 TP 和 DP
- 但延迟可能成为问题（尤其跨节点时）

> 📋 **背景知识：为什么需要 Pipeline Parallelism**
>
> PP 解决的核心问题是：当 TP 已经用满节点内所有卡（TP=8），但模型仍然太大时，PP 可以**跨节点**扩展而不需要高带宽互联（因为只传激活值，量很小）。
>
> 代价是 bubble（部分设备空闲），以及内存中需要暂存多个 micro-batch 的激活值。

> 🛠️ **实践：Megatron**
>
> ```bash
> # Pipeline Parallelism 配置
> --pipeline-model-parallel-size 4
> --num-layers-per-virtual-pipeline-stage 1  # Virtual Pipeline（减少 bubble）
> ```
>
> **Virtual Pipeline Parallelism**（Megatron 的优化）：
> - 标准 PP：每个 stage 连续多层
> - Virtual PP：每个 stage 的层交错分布（如 stage 0 负责第 1, 5, 9... 层）
> - 好处：bubble ratio 降低到 `(PP-1) / (M × V + PP - 1)`，其中 V 是虚拟 stage 数
> - `--num-layers-per-virtual-pipeline-stage` 控制每个虚拟 stage 的层数

---

## 8.6 组合策略

实际训练中，通常组合多种并行策略：

![组合并行](/assets/scaling-book/img/mixed-fsdp-model-parallelism.png)

### 经典组合（Megatron 3D 并行）

```
总设备数 = TP × PP × DP
```

例：128 张 H100（16 节点 × 8 卡）训练 70B 模型：

| 策略 | 值 | 位置 | 原因 |
|------|---|------|------|
| TP | 8 | 节点内 | 利用 NVLink 高带宽 |
| PP | 4 | 跨 4 个节点 | 模型太大，8 卡放不下 |
| DP | 4 | 剩余维度 | 128/(8×4) = 4 |

> 🛠️ **实践：Megatron 完整配置示例**
>
> ```bash
> # 3D 并行配置：TP=8, PP=4, DP=4
> --tensor-model-parallel-size 8
> --pipeline-model-parallel-size 4
> # DP 自动计算：world_size / (TP × PP)
>
> # 内存优化
> --use-distributed-optimizer      # ZeRO-1 优化器分片
> --recompute-activations          # Gradient Checkpointing
> --sequence-parallel              # Sequence Parallelism
>
> # 通信优化
> --overlap-grad-reduce            # 梯度 reduce 与反向计算重叠
> --overlap-param-gather           # 参数 gather 与前向计算重叠
>
> # Pipeline 优化
> --num-layers-per-virtual-pipeline-stage 1  # Virtual PP
>
> # Micro-batch
> --micro-batch-size 1
> --global-batch-size 1024         # 总 batch size
> # num_micro_batches = global_batch / (micro_batch × DP)
> ```

### 选择策略的决策树

```
1. 模型能放进单卡？
   → 是：用 DP
   → 否：继续

2. 模型能放进单节点（8卡）？
   → 是：TP=8 + DP
   → 否：继续

3. 加 PP 直到每 stage 能放进 TP 组
   → 设置 PP=需要的 stage 数
   → TP × PP 卡能放下模型

4. 剩余卡数用 DP
   → DP = 总卡数 / (TP × PP)
   → 调整 global batch size 确保 DP 有足够工作

5. 如果 DP 数很大，用 FSDP 替代 DP
   → 进一步节省内存
```

---

## 8.7 跨 Pod 的 Data Parallelism

当训练扩展到多个 Pod 时，Pod 之间通过 DCN 连接（带宽远低于 ICI/NVLink）。

策略：
- Pod 内用 TP + PP
- Pod 间用 DP
- 使用 **gradient accumulation** 增大有效 batch size，减少 AllReduce 频率

$$T_{\text{comms}}^{\text{DP, cross-pod}} = \frac{4P}{B_{\text{DCN}}}$$

如果 DCN 太慢，可以做多步 gradient accumulation 再 AllReduce。

---

## 关键要点

- [ ] **DP**：简单，不省内存，通信量 = 2P（AllReduce 梯度）
- [ ] **FSDP/ZeRO-3**：内存 ÷ N，通信量 = 3P（多 50%）
- [ ] **TP**：每层都通信（AllReduce），需要高带宽 → 限制在节点内
- [ ] **PP**：按层切分，通信量小但有 bubble，Virtual PP 可缓解
- [ ] **组合**：TP（节点内）→ PP（跨少量节点）→ DP（全集群）
- [ ] Megatron 3D 并行：`world_size = TP × PP × DP`
- [ ] `--overlap-grad-reduce` + `--overlap-param-gather` 是关键优化

---

## 进一步阅读

- 原书 Chapter 5: How to Parallelize a Transformer for Training
- [Megatron-LM v1: Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Megatron-LM v2: Sequence Parallelism + Selective Recomputation](https://arxiv.org/abs/2205.05198)
- [Megatron-LM v3: Pipeline Parallelism](https://arxiv.org/abs/2104.04473)
- [ZeRO 论文 (DeepSpeed)](https://arxiv.org/abs/1910.02054)

