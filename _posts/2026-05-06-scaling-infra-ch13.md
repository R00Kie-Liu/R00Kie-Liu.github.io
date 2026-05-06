---
layout: post
title: "Scaling Book 入门第 13 章：性能调优 — Profiling 与调试"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '工具']
---

# Scaling Book 入门第 13 章：性能调优 — Profiling 与调试

> **本章目标**：了解如何使用 Profiler 定位训练/推理中的性能瓶颈，包括 XLA/JAX profiler 和 NVIDIA 工具。
>
> **对应原书**：Chapter 9 (How to Profile TPU Programs)  
> **优先级**：⭐ 低 | **建议时间**：Day 13, 约 2 小时

---

## 13.1 为什么需要 Profiling

理论 Roofline 分析给出上界，但实际性能通常低于理论值。常见原因：

- **编译器优化不足**：生成的 kernel 没有达到理论最优
- **内存布局问题**：数据在 HBM 中的排列导致额外的 reshape/transpose
- **流水线气泡**：计算和通信没有完美重叠
- **Kernel launch 开销**：大量小 kernel 的调度开销
- **内存碎片**：HBM 利用不充分

**Profiling = 用数据说话**，找到真正的瓶颈。

---

## 13.2 TPU/JAX 软件栈

原书重点讲 TPU/JAX 栈，这里做简要介绍：

```
JAX Python 代码
    ↓ jax.jit / pjit
StableHLO (中间表示)
    ↓ 
XLA 编译器
    ↓ 优化 + 代码生成
TPU/GPU 机器码
```

XLA（Accelerated Linear Algebra）是 Google 的编译器，负责将高级 JAX 代码转化为高效的硬件指令。XLA 会做很多优化：
- **Operator fusion**：将多个小操作合并成一个 kernel
- **Memory layout optimization**：优化数据在 HBM 中的排列
- **Communication scheduling**：安排通信和计算的重叠

---

## 13.3 JAX Profiler

### Trace Viewer

![Trace Viewer](/assets/scaling-book/img/trace-viewer.png)

展示每个 TPU/GPU 在时间轴上做了什么：
- 哪些 kernel 在运行
- 哪些时间段在做通信
- 哪些时间段在空闲（= 潜在优化点）

### XProf Overview

![XProf Overview](/assets/scaling-book/img/xprof-overview.png)

高级概览：
- 总步骤时间
- 各类操作的时间占比（matmul、通信、其他）
- Infeed/outfeed 开销

### Graph Viewer

![Graph Viewer](/assets/scaling-book/img/graph-viewer.png)

可视化 XLA 编译后的计算图：
- 每个节点是一个 XLA op
- 边是数据依赖
- 可以检查编译器是否做了预期的 fusion/优化

### Memory Profile

![Memory Viewer](/assets/scaling-book/img/memory-viewer.png)

展示 HBM 使用随时间的变化：
- 峰值内存使用
- 哪些张量占用最多内存
- 是否有内存泄漏

---

## 13.4 NVIDIA GPU Profiling 工具

对于使用 Megatron + GPU 的场景，更常用的工具是 NVIDIA 生态的：

### NVIDIA Nsight Systems（nsys）

```bash
nsys profile --trace=cuda,nvtx python train.py
```

- 类似 JAX 的 Trace Viewer
- 展示 CUDA kernel 执行时间线
- 可以看到 kernel launch 开销、NCCL 通信时间

### NVIDIA Nsight Compute（ncu）

```bash
ncu --set full python train.py
```

- 单个 kernel 级别的深度分析
- Roofline 分析（自动判断 kernel 是 compute-bound 还是 memory-bound）
- 内存访问模式分析

### PyTorch Profiler

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'),
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

在 TensorBoard 中查看 trace。

> 🛠️ **实践：Megatron**
>
> Megatron 内置的性能监控：
>
> 1. **内置 Timer**：
>    - Megatron 在每个训练步骤输出详细的时间分解
>    - `forward-compute`, `backward-compute`, `optimizer`, `communication` 等
>    - 检查 `backward-compute` 和 `communication` 的比例
>
> 2. **Wandb 集成**：
>    ```bash
>    --wandb-project my-project
>    --wandb-exp-name my-experiment
>    ```
>    监控 MFU、吞吐量、loss 等指标
>
> 3. **NCCL 调试**：
>    ```bash
>    export NCCL_DEBUG=INFO           # 打印通信细节
>    export NCCL_DEBUG_SUBSYS=INIT    # 只看初始化
>    ```
>    排查通信问题（如网络配置错误、带宽不足）
>
> 4. **常见瓶颈排查清单**：
>    - MFU < 30%：检查是否通信瓶颈（TP 跨节点？）
>    - MFU 30-40%：检查 PP bubble（增加 micro-batch 数）
>    - MFU 40-50%：正常范围，可尝试 overlap 优化
>    - MFU > 50%：优秀

> 🛠️ **实践：SGLang**
>
> SGLang 的性能诊断：
>
> 1. **Metrics Endpoint**：
>    ```bash
>    curl http://localhost:30000/get_server_info
>    ```
>    查看当前运行状态、cache 命中率、请求队列等
>
> 2. **Benchmark 工具**：
>    ```bash
>    python -m sglang.bench_serving \
>      --backend sglang \
>      --port 30000 \
>      --dataset-name random \
>      --num-prompts 1000
>    ```
>    测量 TTFT、TPOT、吞吐量等关键指标
>
> 3. **Profiling 模式**：
>    ```bash
>    --enable-torch-compile  # 可以用 torch.compile 的 profiling
>    --log-level debug       # 详细日志
>    ```

---

## 13.5 常见性能问题和解决方案

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| MFU 很低 | TP 跨节点 | 确保 TP ≤ 节点内 GPU 数 |
| Step time 波动大 | Data loading 瓶颈 | 检查 dataloader，增加 worker |
| 通信时间 >> 计算时间 | DP AllReduce 太慢 | 增加 gradient accumulation |
| PP bubble 大 | Micro-batch 太少 | 增加 micro-batch 数 |
| OOM | 激活值占用太多 | 启用 gradient checkpointing |
| Decode 延迟高 | 单请求 memory-bound | 增大 TP 或使用量化 |

---

## 关键要点

- [ ] Profiling 用数据定位真正瓶颈，避免盲目优化
- [ ] TPU 用 JAX Profiler（Trace Viewer + XProf），GPU 用 nsys/ncu
- [ ] Megatron 内置 timer 输出前向/反向/通信时间分解
- [ ] MFU 是最重要的高级指标：< 30% 有严重问题，40-50% 正常，> 50% 优秀
- [ ] SGLang 通过 metrics endpoint 和 bench_serving 工具做推理性能诊断

---

## 进一步阅读

- 原书 Chapter 9: How to Profile TPU Programs
- [NVIDIA Nsight Systems 文档](https://developer.nvidia.com/nsight-systems)
- [PyTorch Profiler 教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

