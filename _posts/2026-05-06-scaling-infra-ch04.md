---
layout: post
title: "Scaling Book 入门第 4 章：芯片互联与集群拓扑"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '硬件基础']
---

# Scaling Book 入门第 4 章：芯片互联与集群拓扑

> **本章目标**：理解多芯片如何连接成集群，不同互联方式（ICI/NVLink/InfiniBand/DCN）的带宽差异，以及拓扑结构如何影响并行策略。
>
> **对应原书**：Chapter 2 (TPU Networking) + Chapter 12 (GPU Networking)  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 3-4, 约 2.5 小时

---

## 4.1 为什么需要集群

单张加速器的算力和内存都是有限的。训练一个大模型（如 LLaMA 70B）需要：
- **内存**：远超单卡容量，需要将参数分片到多卡
- **算力**：要在合理时间内完成训练，需要大量并行计算
- **目标**：加 N 倍芯片 → 获得接近 N 倍的加速（strong scaling）

但芯片间需要**通信**（交换参数、梯度、激活值），通信时间如果超过计算时间就无法有效 scale。因此，**互联带宽和拓扑结构直接决定了能用多少芯片、怎么分布计算。**

---

## 4.2 TPU 的互联：ICI 与 Torus 拓扑

### ICI（Inter-Chip Interconnect）

TPU 之间通过 ICI 直接连接，**不经过 CPU 或交换机**。

![ICI 环形连接](/assets/scaling-book/img/ici-wraparound.png)

关键参数（TPU v5p）：
- ICI 带宽：**90 GB/s/轴**，每芯片有 3 个轴
- 总 ICI 带宽：270 GB/s/芯片（3D torus 的 6 个方向）

### Torus 拓扑

TPU 的连接方式是 **Torus（环面）拓扑**：
- TPU v5e/v6e：**2D Torus**，最大 16×16
- TPU v4/v5p：**3D Torus**，如 16×16×16 或 16×20×28

![TPU 集群拓扑](/assets/scaling-book/img/subslices.png)

> 📋 **背景知识：什么是 Torus 拓扑**
>
> 把芯片排成网格（如 4×4），每个芯片只和上下左右 4 个邻居直接相连。
> - **Mesh**（网格）：边缘芯片只有 2-3 个邻居
> - **Torus**（环面）：加上"绕回"连接（首尾相连），每个芯片都有 4 个（2D）或 6 个（3D）邻居
> - 绕回连接将最远距离从 N 缩短到 N/2
>
> 类比：想象一张纸的左右边粘起来（变成圆柱），再把上下粘起来（变成甜甜圈），这就是 2D torus。

**最近邻连接的影响**：两个不相邻的 TPU 通信需要经过中间芯片"转发"，跳数越多延迟越高。这直接影响了 AllReduce 等集合通信的效率。

### TPU Pod 与 Superpod

![TPU Pod 架构](/assets/scaling-book/img/tpu-rack.png)

- **Tray**：4 个 TPU 芯片连接到一个 CPU 主机
- **Pod**：ICI 连接的所有芯片，如 TPU v5p 的 superpod 为 16×20×28 = 8,960 芯片
- 基本构建块：4×4×4 = 64 芯片的"cube"，cube 之间通过光交换机连接
- 小于一个 cube 的拓扑（如 2×2×1）**没有绕回连接**，通信时间翻倍

---

## 4.3 GPU 的互联：NVLink + InfiniBand

GPU 使用完全不同的互联策略：**层级化交换机网络**。

### 节点内：NVLink

![NVLink 节点](/assets/scaling-book/gpu/nvlink-nodes.png)

- H100 DGX 节点：8 张 GPU 通过 NVLink 4.0 互联
- **每 GPU 双向带宽 ~900 GB/s**
- 通过 NVSwitch 实现节点内**任意两张卡的全带宽直连**
- 这比 TPU ICI (~90 GB/s/轴) 快约 10×

### 节点间：InfiniBand / RoCE

- 每节点 8 个 400 Gb/s（50 GB/s）的网口
- 通过交换机实现节点间通信
- 总跨节点带宽：~400 GB/s/节点
- 比 NVLink 慢约 2-4×

### GPU Superpod

![H100 Superpod](/assets/scaling-book/gpu/h100-superpod.png)

典型的 GPU 集群架构：
- DGX 节点（8 GPU）→ NVLink 互联
- 节点间 → InfiniBand 交换机
- Spine-Leaf 网络 → 连接数百到数千节点

> 📋 **背景知识：NVLink vs ICI 的本质差异**
>
> | 特性 | NVLink (GPU) | ICI (TPU) |
> |------|-------------|-----------|
> | 拓扑 | 全互联（交换机） | 最近邻（Torus） |
> | 节点内带宽 | ~900 GB/s/GPU | ~270 GB/s/chip |
> | 扩展性 | 靠增加交换层级 | 靠 torus 自然扩展 |
> | 成本 | 高（NVSwitch 很贵） | 低（直接连接） |
> | 编程复杂度 | 相对简单（像全连接） | 需要考虑拓扑 |
>
> **关键直觉**：NVLink 让节点内 8 张卡"像一张大卡"，但跨节点立刻带宽骤降。TPU 的带宽从近到远是平滑下降的。

---

## 4.4 DCN（Data Center Network）

无论 GPU 还是 TPU，超出一个 Pod/Superpod 的通信走数据中心网络（DCN）：

- TPU DCN 带宽：~6 GB/s/chip（TPU v5p）
- 这比 ICI 慢约 **15×**，比 HBM 带宽慢约 **450×**

DCN 用于：
- **Multi-slice 训练**：多个 ICI-connected 的 TPU slice 通过 DCN 连接
- **跨 Pod 的 Data Parallelism**

---

## 4.5 带宽层级全景

```
计算单元内部（VMEM → MXU）     ~20-40 TB/s
        ↓ (~5-20×)
HBM → 计算单元                  ~1-9 TB/s
        ↓ (~3-10×)
节点内互联 (NVLink/ICI)          ~90-900 GB/s
        ↓ (~2-15×)
节点间互联 (InfiniBand/DCN)      ~6-50 GB/s
        ↓ (~3-10×)
PCIe (CPU ↔ GPU)                 ~50 GB/s
```

**核心原则**：通信密集的操作（如 Tensor Parallelism）应该放在高带宽的互联上（节点内），通信稀疏的操作（如 Data Parallelism）可以放在低带宽的互联上（跨节点）。

> 🛠️ **实践：Megatron**
>
> Megatron 的并行策略映射严格遵循这个带宽层级：
>
> ```
> 节点内 8 卡（NVLink ~900 GB/s）  →  Tensor Parallelism (TP=8)
> 跨节点（InfiniBand ~50 GB/s）    →  Pipeline Parallelism (PP)
> 全集群                           →  Data Parallelism (DP)
> ```
>
> 配置示例（128 张 H100，16 节点）：
> ```bash
> --tensor-model-parallel-size 8    # 节点内
> --pipeline-model-parallel-size 4  # 跨 4 个节点
> --data-parallel-size 4            # 剩余 = 128/(8×4) = 4
> ```
>
> `--tensor-model-parallel-size` 一般不超过单节点的 GPU 数，因为跨节点做 TP 的通信开销太大。

---

## 4.6 拓扑对并行策略的影响

不同拓扑适合不同的并行方式：

| 并行策略 | 通信模式 | 适合的互联 |
|----------|----------|-----------|
| Data Parallelism | AllReduce 梯度（低频、大量数据） | DCN / InfiniBand |
| Tensor Parallelism | AllReduce/AllGather（高频、中量数据） | NVLink / ICI |
| Pipeline Parallelism | 点对点传输（中频、少量数据） | InfiniBand / ICI |
| Expert Parallelism | AllToAll（高频、中量数据） | NVLink / ICI |

---

## 关键要点

- [ ] TPU 用 Torus 拓扑（最近邻直连），便宜但需考虑跳数
- [ ] GPU 用交换机拓扑（NVLink + InfiniBand），节点内高带宽，跨节点骤降
- [ ] 带宽层级：片上缓存 >> HBM >> 节点内互联 >> 节点间互联
- [ ] 通信密集的并行策略放在高带宽互联上（TP → 节点内）
- [ ] 通信稀疏的并行策略放在低带宽互联上（DP → 跨节点）
- [ ] Megatron 的 TP/PP/DP 配置直接映射到这个带宽层级

---

## 进一步阅读

- 原书 Chapter 2: TPU Networking
- 原书 Chapter 12: Networking (GPU)
- [NVIDIA DGX H100 架构](https://www.nvidia.com/en-us/data-center/dgx-h100/)

