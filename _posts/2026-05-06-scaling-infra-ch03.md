---
layout: post
title: "Scaling Book 入门第 3 章：内存层级与带宽 — 数据如何流动"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '硬件基础']
---

# Scaling Book 入门第 3 章：内存层级与带宽 — 数据如何流动

> **本章目标**：理解加速器内部的内存层级（HBM → Cache → 计算单元），以及数据搬运如何成为性能瓶颈。
>
> **对应原书**：Chapter 2 (TPU internals) + Chapter 12 (GPU memory)  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 3, 约 2 小时

---

## 3.1 数据搬运的层级

无论 GPU 还是 TPU，数据在使用前都必须经过多层"搬运"：

```
HBM（主内存，大但慢）
  ↓  HBM 带宽 ~1-9 TB/s
VMEM / L2 Cache（片上缓存，小但快）
  ↓  内部带宽 ~20-40 TB/s
MXU / Tensor Core（计算单元）
```

### TPU 的数据流

![TPU 带宽层级](/assets/scaling-book/img/tpu-bandwidth.png)

1. 权重和激活值存储在 **HBM** 中
2. 计算前，数据从 HBM 流式传入 **VMEM**
3. 从 VMEM 加载到 **MXU** 进行矩阵乘法
4. 结果写回 VMEM，再写回 HBM

关键：这个过程是**流水线化（pipelined）** 的 — 不需要等全部数据加载完才开始计算。

![流水线操作动画](/assets/scaling-book/img/pointwise-product.gif)

> 📋 **背景知识：流水线（Pipelining）的概念**
>
> 如果你需要处理一个很大的矩阵：
> - **不用流水线**：等全部数据从 HBM 搬到 VMEM，然后计算，再全部写回 → 时间 = T_load + T_compute + T_store
> - **用流水线**：边搬第 1 块边计算第 0 块边写回第 -1 块 → 时间 ≈ max(T_load, T_compute, T_store)
>
> 流水线是让计算和通信重叠的关键技术。如果 T_compute > T_load，MXU 不会空闲。

### GPU 的数据流

GPU 的内存层级更复杂：

```
HBM (80-192 GB, 3.35-9 TB/s)
  ↓
L2 Cache (50-126 MB, ~5.5 TB/s on H100)
  ↓
SMEM / L1 Cache (256 kB/SM, 极高带宽)
  ↓
Registers (256 kB/SM)
  ↓
Tensor Core / CUDA Core
```

**GPU vs TPU 的关键差异**：

| 特性 | TPU VMEM | GPU SMEM + L2 |
|------|----------|---------------|
| 容量 | 128 MB | SMEM 32 MB + L2 50 MB |
| 控制方式 | 程序员控制 | SMEM 程序员控制；L2 硬件自动管理 |
| 带宽 | ~22× HBM 带宽 | SMEM 极高；L2 ~1.6× HBM 带宽 |

TPU 的 VMEM 大且快，这让 TPU 在某些推理场景中可以把整层权重放在 VMEM 里，极大减少 HBM 访问。GPU 的 L2 Cache 是硬件自动管理的，程序员无法直接控制，这导致一种"spooky action at a distance" — 你需要调整内存访问模式来利用好 L2 cache，但很难直接控制。

---

## 3.2 VMEM 预取（Prefetching）

TPU 的一个重要优化：**权重预取**。

在 Transformer 的一个层执行 Attention 计算时，可以提前将下一步 FFN 的权重从 HBM 预取到 VMEM。这样当 Attention 完成时，FFN 权重已经在 VMEM 里了，不需要等待 HBM 加载。

前提：权重要足够小（或经过分片）才能放进 VMEM。

---

## 3.3 PCIe：CPU 与加速器之间的桥梁

![PCIe 连接](/assets/scaling-book/img/pcie.png)

加速器通过 **PCIe** 连接到 CPU 主机：

- TPU v4 PCIe 带宽：16 GB/s（每方向）
- 这比 HBM 带宽慢约 **100×**

PCIe 主要用于：
- 从 CPU 内存加载训练数据到加速器
- Host offload（把优化器状态卸载到 CPU 内存以节省 HBM）
- 跨主机通信（DCN 通过 PCIe → 网卡 → 网络）

> 📋 **背景知识：为什么 PCIe 成为瓶颈**
>
> PCIe 5.0 x16 的理论带宽约 64 GB/s，但实际有效带宽通常只有 ~50 GB/s。对比：
> - HBM 带宽：~3 TB/s（H100）
> - NVLink 带宽：~900 GB/s（H100 节点内）
> - PCIe 带宽：~50 GB/s
>
> 这意味着任何需要经过 PCIe 的操作（如从 CPU 加载数据、host offload）都会比在 GPU 内存内操作慢约 60-180×。

> 🛠️ **实践：Megatron**
>
> Megatron 中与内存层级相关的配置：
> - `--recompute-activations`：用重计算代替存储激活值，节省 HBM，但增加计算量（约 33%）
> - `--distribute-saved-activations`：将激活值分布到多张卡上
> - `--use-flash-attn`：Flash Attention 通过减少 HBM 访问大幅加速 Attention（利用 SMEM 做 tiling）
>
> Flash Attention 本质上就是利用内存层级差异的优化：把 Attention 的计算分 tile 在 SMEM 中完成，避免把完整的 N×N attention matrix 写入 HBM。

---

## 3.4 内存容量约束

除了带宽，**内存容量**也是关键约束：

训练一个模型需要在 HBM 中存储：
- **模型参数**：每参数 2 bytes (bf16)
- **梯度**：每参数 2 bytes
- **优化器状态**：Adam 需要每参数 8 bytes (fp32 的一阶/二阶矩)
- **激活值**：取决于 batch size 和序列长度

**总计**：约每参数 16-20 bytes

举例：LLaMA 70B 参数，仅权重 + 优化器就需要 ~1.1 TB。单张 H100 (80 GB) 远远不够，需要至少 14 张卡才能放下——这就是为什么需要分片。

> 🔗 **与你的联系**
>
> 在你训练预训练模型时，你可能遇到过 OOM（Out of Memory）。现在你知道了：
> - 减小 batch size → 减少激活值占用
> - 使用 gradient checkpointing → 用计算换内存
> - 使用 FSDP/ZeRO → 把参数和优化器状态分片到多卡
> - 使用混合精度 → 减少每个值的字节数
>
> 这些都是在内存容量约束下的权衡。

---

## 3.5 关键数字速记

你应该记住的几个量级：

| 链路 | 带宽量级 | 说明 |
|------|----------|------|
| HBM → 计算单元 | ~1-9 TB/s | 取决于芯片代次 |
| VMEM/SMEM → 计算单元 | ~20-40 TB/s | 极快，但容量有限 |
| NVLink（节点内 GPU-GPU） | ~900 GB/s | H100 双向 |
| ICI（TPU 芯片间） | ~90 GB/s/轴 | 直接连接 |
| PCIe | ~50 GB/s | CPU-GPU |
| InfiniBand（节点间） | ~50 GB/s | 以太网级别 |

**记忆法**：从上到下大约每级慢 10×。

---

## 关键要点

- [ ] 数据必须从 HBM → 片上缓存 → 计算单元，层层搬运
- [ ] 流水线化让计算和搬运可以重叠，时间 ≈ max(计算, 搬运)
- [ ] TPU 的 VMEM 大且程序员可控；GPU 的 L2 cache 小且自动管理
- [ ] PCIe 比 HBM 带宽慢约 100×，是 CPU-GPU 通信的瓶颈
- [ ] 训练一个模型需要每参数约 16-20 bytes 的 HBM
- [ ] Flash Attention 的本质是利用 SMEM 替代 HBM 做 tiling 计算

---

## 进一步阅读

- 原书 Chapter 2: What Is a TPU → Memory 部分
- 原书 Chapter 12: What Is a GPU → Memory 部分
- [Flash Attention 论文 (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

