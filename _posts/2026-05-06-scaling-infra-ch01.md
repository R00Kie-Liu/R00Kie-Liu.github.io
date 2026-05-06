---
layout: post
title: "Scaling Book 入门第 1 章：硬件基础 — GPU 与 TPU 是什么"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '硬件基础']
---

# Scaling Book 入门第 1 章：硬件基础 — GPU 与 TPU 是什么

> **本章目标**：理解现代 ML 加速器（GPU/TPU）的核心组成部件，建立"芯片 = 计算单元 + 内存"的心智模型。
>
> **对应原书**：Chapter 2 (TPUs) + Chapter 12 (GPUs)  
> **优先级**：⭐⭐⭐ 高 | **建议时间**：Day 1-2, 约 3 小时

---

## 1.1 为什么你需要了解硬件

> 🔗 **与你的联系**
>
> 作为预训练研究员，你关注 Scaling Law（C = 6ND 等公式）和模型架构设计。但 Scaling Law 告诉你"需要多少 FLOPs"，而硬件决定了"这些 FLOPs 要花多少时间和钱"。一个在理论上很好的架构（比如某种新注意力机制），如果在硬件上跑不快，就无法 scale。**理解硬件是连接"模型设计"和"实际训练"的桥梁。**

书中一个核心观点：

> *"A 20% win on benchmarks is irrelevant if it comes at a 20% cost to roofline efficiency."*

---

## 1.2 TPU 的核心组件

一个 TPU 本质上就是一个**超强的矩阵乘法机器**加上一块**快速内存**。

![TPU 芯片结构](/assets/scaling-book/img/tpu-chip.png)

TPU 的核心组件：

### MXU（Matrix Multiply Unit）— 矩阵乘法单元

- TPU 的核心，专门做矩阵乘法
- 使用**脉动阵列**（Systolic Array）架构
- TPU v5e：每个 MXU 每 8 个周期完成一次 `bf16[8,128] × bf16[128,128]` 的矩阵乘法
- 总算力：TPU v5e 约 `2×10¹⁴` bf16 FLOPs/s

![脉动阵列工作原理](/assets/scaling-book/img/systolic-array.gif)

> 📋 **背景知识：什么是脉动阵列（Systolic Array）**
>
> 脉动阵列是一种专门为矩阵乘法设计的硬件结构。想象一个 128×128 的计算单元网格：
> - 矩阵 A 的行从左侧流入
> - 矩阵 B 的列从上方流入
> - 每个计算单元做一次乘加（multiply-accumulate）并将结果传给下一个
> - 数据像心跳一样"脉动"式地流过整个阵列
>
> 关键优势：**每个数据元素被加载一次，但被使用多次**（被整行/列的计算单元共享），这就是为什么矩阵乘法的算术强度可以很高。

### VPU（Vector Processing Unit）— 向量处理单元

- 负责非矩阵乘法的操作：ReLU、LayerNorm、Softmax、逐元素加法等
- 类似 CPU 的 SIMD 单元，对向量做相同操作
- 比 MXU 慢很多（FLOPs/s 低一个数量级）

### VMEM（Vector Memory）— 快速片上缓存

- 大小：TPU v5e 为 128 MiB
- 带宽极高（约为 HBM 的 22×）
- 程序员可控的 scratchpad（不像 CPU cache 那样自动管理）
- 数据必须先从 HBM 搬到 VMEM，MXU 才能使用

### HBM（High Bandwidth Memory）— 主内存

- 存储模型权重、梯度、激活值
- 容量：TPU v5e 为 16 GiB，v5p 为 96 GiB
- 带宽：约 0.8-2.8 TB/s（取决于代次）

> 📋 **背景知识：HBM vs 普通 DDR 内存**
>
> HBM（High Bandwidth Memory）是一种 3D 堆叠的 DRAM：
> - 多层 DRAM 芯片垂直堆叠，通过硅通孔（TSV）连接
> - 带宽比普通 DDR5 高 5-10×（TB/s 级别 vs 几十 GB/s）
> - 容量比 SRAM 大得多，但带宽比片上 SRAM/VMEM 低
> - 物理上紧贴计算芯片（通过 interposer 连接）
>
> 你可以把它想象成：VMEM 是 L1 cache（小但超快），HBM 是主内存（大但相对慢）。

---

## 1.3 GPU 的核心组件

GPU（以 NVIDIA H100 为例）结构类似，但更"模块化"：

![GPU 芯片结构](/assets/scaling-book/gpu/gpu-diagram.png)

### SM（Streaming Multiprocessor）— 流式多处理器

- GPU 由 **132 个 SM** 组成（H100）
- 每个 SM 是一个独立的计算单元，可以并行执行不同任务
- 对比：TPU 只有 1-2 个大的 TensorCore

### Tensor Core — 张量核心（矩阵乘法单元）

- 每个 SM 有 4 个 Tensor Core
- 功能等价于 TPU 的 MXU
- H100 总算力：~990 bf16 TFLOP/s

### CUDA Core — 通用计算核心

- 每个 SM 有 128 个 fp32 CUDA Core
- 负责向量运算（类似 TPU 的 VPU）
- 使用 SIMT（Single Instruction Multiple Threads）模型，比 TPU 的 SIMD 更灵活

### 内存层级

- **Registers**：每 SM 256 kB，最快
- **SMEM/L1 Cache**：每 SM 256 kB，程序员可控的共享内存
- **L2 Cache**：全芯片共享 ~50 MB，硬件管理
- **HBM**：80-192 GB，3.35-9 TB/s 带宽

![SM 内部结构](/assets/scaling-book/gpu/blackwell-sm.png)

---

## 1.4 GPU vs TPU：关键对比

| 组件 | GPU | TPU | 说明 |
|------|-----|-----|------|
| 计算单元 | SM (×132) | TensorCore (×2) | GPU 更模块化 |
| 矩阵乘法 | Tensor Core | MXU | 功能相同 |
| 向量运算 | CUDA Core | VPU | GPU 更灵活 |
| 快速缓存 | SMEM (32MB 总) | VMEM (128MB) | TPU 缓存更大 |
| 主内存 | HBM (80GB) | HBM (96GB) | 容量相近 |
| 编程模型 | SIMT（灵活） | SIMD（简单） | GPU 更通用 |

**核心差异总结**：

1. **TPU 更简单，GPU 更灵活**：TPU 就是一个大矩阵乘法机器，GPU 是由上百个小处理器组成的阵列
2. **TPU 的快速缓存（VMEM）远大于 GPU 的 SMEM**：这让 TPU 在推理时可以把权重放在 VMEM 里
3. **单颗 GPU 通常更强大**：H200 FLOPs/s 约是 TPU v5p 的 2×，但价格也是 2.5×
4. **TPU 靠集群取胜**：TPU 的互联更便宜，可以 scale 到更大的集群

> 🛠️ **实践：Megatron**
>
> Megatron-LM 主要针对 NVIDIA GPU 设计。它假设：
> - 节点内 8 卡通过 NVLink 高速互联（~900 GB/s bidirectional on H100）
> - 节点间通过 InfiniBand 连接（~400 Gb/s per port）
> - Tensor Parallelism 通常限制在节点内（因为需要高带宽）
> - Pipeline/Data Parallelism 可以跨节点
>
> 当你配置 `--tensor-model-parallel-size 8` 时，Megatron 会在同一节点的 8 张卡上做 TP，充分利用 NVLink 的高带宽。

> 🛠️ **实践：SGLang**
>
> SGLang 的推理也需要了解 GPU 内存层级：
> - 模型权重常驻 HBM
> - KV cache 也在 HBM 中，SGLang 通过 RadixAttention 高效管理
> - `--mem-fraction-static` 参数控制为权重预留多少 HBM，剩余给 KV cache 动态分配
> - Tensor Core 的利用率决定了 prefill 的吞吐量

---

## 1.5 为什么矩阵乘法如此特殊

> 📋 **背景知识**
>
> 矩阵乘法 `C[M,N] = A[M,K] × B[K,N]` 的特殊性：
> - **计算量**：O(M×K×N) = O(n³)（对方阵而言）
> - **数据量**：O(M×K + K×N + M×N) = O(n²)
> - **比值**：做 n³ 次运算只需要加载 n² 个数据 → 每加载一个数据可以做 ~n 次运算
>
> 这意味着只要矩阵够大，计算时间远大于数据加载时间 → **compute-bound**。
> 相比之下，逐元素操作（如 ReLU）是 O(n) 计算 / O(n) 数据 → 永远是 memory-bound。
>
> 这就是为什么 TPU/GPU 把绝大部分晶体管面积给了矩阵乘法单元。

---

## 1.6 各代硬件规格速查表

### TPU 规格

| 型号 | HBM 容量 | HBM 带宽 | bf16 FLOPs/s | int8 OPs/s |
|------|----------|----------|-------------|-----------|
| TPU v3 | 32 GB | 900 GB/s | 1.4×10¹⁴ | 1.4×10¹⁴ |
| TPU v5e | 16 GB | 810 GB/s | 2.0×10¹⁴ | 3.9×10¹⁴ |
| TPU v5p | 96 GB | 2.8 TB/s | 4.6×10¹⁴ | 9.2×10¹⁴ |
| TPU v6e | 32 GB | 1.6 TB/s | 9.2×10¹⁴ | 1.8×10¹⁵ |

### GPU 规格

| 型号 | HBM 容量 | HBM 带宽 | bf16 FLOPs/s | fp8 OPs/s |
|------|----------|----------|-------------|----------|
| A100 | 80 GB | 2.0 TB/s | 3.1×10¹⁴ | 6.2×10¹⁴ |
| H100 | 80 GB | 3.4 TB/s | 9.9×10¹⁴ | 2.0×10¹⁵ |
| H200 | 141 GB | 4.8 TB/s | 9.9×10¹⁴ | 2.0×10¹⁵ |
| B200 | 192 GB | 8.0 TB/s | 2.3×10¹⁵ | 4.5×10¹⁵ |

---

## 关键要点

- [ ] TPU = MXU（矩阵乘法）+ VPU（向量运算）+ VMEM（快速缓存）+ HBM（主内存）
- [ ] GPU = 多个 SM × (Tensor Core + CUDA Cores + SMEM) + L2 Cache + HBM
- [ ] 矩阵乘法特殊在于 O(n³) 计算 / O(n²) 数据，天然适合被硬件加速
- [ ] TPU 简单/便宜但依赖编译器和集群；GPU 灵活/强大但复杂
- [ ] 理解硬件是为了判断"我的模型/算法能否高效利用这些计算资源"
- [ ] 在 Megatron 中，硬件拓扑直接决定并行策略的配置

---

## 进一步阅读

- 原书 Chapter 2: How to Think About TPUs
- 原书 Chapter 12: How to Think About GPUs
- [NVIDIA H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [TPU v5e 文档](https://cloud.google.com/tpu/docs/v5e)

