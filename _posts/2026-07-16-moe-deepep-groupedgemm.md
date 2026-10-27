---
layout: post
title: "MoE我们喜欢你（四）：DeepEP 与 GroupedGEMM 如何拆解通信和计算瓶颈"
series: moe-we-like-you
series_order: 4
last_modified_at: 2026-10-26
date: 2026-07-16
tags: [MoE, Infra, Megatron]
description: "解释 DeepEP/Flex Dispatcher 与 GroupedGEMM/Fused MoE GEMM 如何分别处理 MoE 大规模训练中的通信墙和计算墙。"
---


## 0. 从执行时间看 MoE

第三篇沿源码追踪了 `route -> dispatch -> experts -> combine`。这一篇换到 profiler 上：当 active FLOPs 已经降下来，tokens/s 仍不理想，时间究竟花在哪里？

先区分三个可测量的部分：token 的重排与内存访问、跨 rank 通信、本地专家计算。DeepEP 主要优化 dispatch/combine，GroupedGEMM 主要优化专家矩阵乘；fusion 和 overlap 则减少中间开销或利用独立工作隐藏等待。

![DeepEP 与 GroupedGEMM 的分工](/assets/img/blog/moe-we-like-you/04-deepep-groupedgemm-division.png)
*图 1：沿 MoE layer 的执行路径，分别测量数据移动和专家计算。二者优化后，还需重新检查端到端耗时。*

## 1. 先把 trace 对到执行路径

| 阶段 | 数据发生什么变化 | 重点测量 |
|---|---|---|
| route | hidden states 产生 top-k ids 和 weights | router kernel、负载分布 |
| pack / permute | token 按目的 rank、expert 分组 | 索引构造、内存读写、padding |
| dispatch | 输入到达持有专家的 rank | A2A 时间、SM 占用、rank 间差异 |
| expert compute | 每个本地 expert 执行 MLP | GEMM 形状、kernel 数、GPU 利用率 |
| combine / unpermute | 输出回到原 token 位置并加权合并 | 回传、scatter/reduce、buffer 复用 |

测量时固定模型、batch 和路由结果。若某个 rank 较慢，还要区分它是收到了更多 token，还是相同工作量下执行更慢；平均 A2A 时间可能掩盖这个差别。

## 2. GroupedGEMM：怎样执行不等长的 expert batches

GEMM 是 General Matrix-Matrix Multiplication，即通用矩阵乘法。若 expert `e` 收到 `T_e` 个 token，hidden size 为 `H`、intermediate size 为 `I`，一次 up projection 的形状是：

```text
[T_e, H] @ [H, I] -> [T_e, I]
```

down projection 则是 `[T_e, I] @ [I, H]`。SwiGLU 还包括 gate projection。Dense MLP 通常处理一个较大的 token batch；MoE 则同时出现多个不同 `T_e` 的矩阵乘。

逐 expert launch 时，小矩阵可能无法充分利用 GPU，还要重复支付调度开销。GroupedGEMM 将这些矩阵乘放进一组调度中执行。它并不要求所有 expert 收到相同 token 数，也不会自动消除负载不均。

最值得检查的是本地 expert 数、各自的 `T_e` 和 tile 利用率。全局专家数很大，但每个 rank 只有一个 expert 时，不能仅凭专家总数判断 GroupedGEMM 的收益；某些矩阵已足够大时，收益也会与大量小矩阵的情况不同。

## 3. DeepEP：把通用 A2A 换成 MoE 数据路径

普通 alltoall dispatcher 可以先作为功能和数值基线。EP 跨节点后，MoE 的不规则收发还会遇到 NVLink 与 RDMA 域之间的带宽差异、布局转换、计数同步和通信 kernel 的 SM 占用。

DeepEP 提供面向 token dispatch/combine 的通信路径，将这些 MoE 特有操作与数据移动结合。Megatron 的 Flex Dispatcher 是接入这类后端的框架抽象，DeepEP / HybridEP 是具体实现。切换后端时，需要比较完整 dispatch/combine 阶段，不能只比较通信原语的峰值带宽。

如果单节点 alltoall 已满足吞吐目标，或者瓶颈仍在 expert compute，可先保留简单后端。跨节点 A2A 已占据明显 step time 时，再测 DeepEP/Flex 的收益。normal / low-latency 路径、动态专家副本和 serving 调度在第九篇展开。

## 4. 框架边界与 overlap

Flex Dispatcher 消费 routing metadata，组织重排、通信和 buffer；本地 expert 实现再消费按 expert 分组的 activation 与 token count。Megatron 的 TEGroupedMLP / GroupedLinear 和 serving 中的 fused MoE kernel，分别在这条边界的计算侧工作。

![DeepEP/Flex 与 GroupedGEMM/FusedMoE 的实现边界](/assets/img/blog/moe-we-like-you/04-deepep-flex-groupedgemm-boundary.png)
*图 2：框架通过 dispatcher 与 expert 接口衔接通信和计算；优化后的实现也可能进一步融合重排、量化和 GEMM。*

SGLang 同样分别选择 A2A backend（`--moe-a2a-backend`）和 MoE runner（`--moe-runner-backend`）。前者负责 token 收发，后者负责本地专家执行，二者的组合受硬件、精度格式和 runtime 版本约束。

overlap 还需要独立工作。同一组 token 必须先到达，expert 才能计算；不能仅靠换通信库消除这个依赖。实际调度可利用不同 microbatch、token chunk 或 shared expert path 的工作重叠。要从 timeline 确认等待是否减少，以及通信 kernel 是否挤占了计算资源。

## 5. Profiling：由现象缩小排查范围

### 5.1 一段等待为什么会被记在通信里

下面用事件顺序说明判读过程，不包含实测耗时。假设两个 rank 已完成 dispatch，同一批 token 的输出需要在 combine 时互相回传：

```text
rank 0: dispatch 完成 -> expert GEMM 仍在执行 ---------> 输出就绪 -> combine
rank 1: dispatch 完成 -> expert GEMM 完成 -> 等待所需输出 -------> combine 完成
```

如果只统计 rank 1 的 combine 区间，这段等待容易被归入通信。把两个 rank 的 trace 对齐后，先检查 rank 0 的 GEMM 为什么结束得晚：若它收到更多 token，优先定位负载偏斜；若 token 数接近，再比较 expert 形状、padding 和计算竞争。这样才能区分等待数据就绪与链路传输本身耗时。

这个例子也解释了为什么需要同时记录 per-expert token count 和 per-rank kernel 时间。输出就绪之前的等待对应前序计算与负载；输出就绪之后仍持续的数据传输，再比较通信后端和拓扑。overlap 则检查等待期间是否有可穿插的独立工作。

### 5.2 对照其它现象

| 观察到的现象 | 先核对什么 | 可比较的改动 |
|---|---|---|
| 跨节点后 dispatch/combine 明显变长 | 实际跨节点流量、互联域、收发计数 | Flex/DeepEP、EP 拓扑、group/node-limited routing |
| 少数 rank 拖长整个 step | per-rank token 数和 expert GEMM 时间 | 负载控制、expert placement；动态副本见第九篇 |
| expert compute 有大量小 kernel | 本地 expert 数、每个 `T_e`、ETP 切分 | GroupedGEMM、降低 ETP、融合 activation/量化 |
| GroupedGEMM 已开启但利用率仍低 | tile padding、expert batch 分布、数据类型 | 改 kernel 配置或本次 forward 的 token 数 |
| permute/unpermute 占比高 | layout、索引生成、临时 buffer | permutation fusion、buffer 复用 |
| GPU timeline 有明显空洞 | CPU launch、计数同步、stream/event 依赖 | fusion、可捕获的 CUDA Graph、调度调整 |
| shared expert 与 routed path 串行 | shared 输入是否已就绪、占用多少 SM | shared expert overlap |
| 显存峰值迫使 full recompute | 保存的 activation、staging 和 overlap buffer | selective recompute、降低同时在途的工作量 |

增加 gradient accumulation 步数会增大全局 batch，但不一定增加一次 expert GEMM 的 token 数。若要改善小矩阵形状，应检查 micro-batch、序列长度以及 dispatcher 实际聚合了哪些 token。

负载 CV（标准差 / 均值）用于定位分布差异，仍要结合最大 rank 耗时。低 CV 不保证快：所有 rank 都可能受同一个通信或 kernel 瓶颈限制。

## 6. 一次只改变一组实现

| 阶段 | 比较内容 | 保持不变 |
|---|---|---|
| 功能基线 | alltoall、router 高精度、明确的 drop/balance 配置 | 模型与 batch |
| 本地计算 | expert 实现、GroupedGEMM | routing、dispatcher |
| 重排与小算子 | router / permute fusion | 专家形状和通信后端 |
| 通信 | Flex + DeepEP/HybridEP 等后端 | top-k、专家放置、精度 |
| 调度 | A2A/shared expert overlap | 先固定并行映射 |
| 低精度 | FP8/MXFP8 等 expert 路径 | 保留精度对照，检查 router 行为 |

每轮记录端到端 tokens/s、峰值显存和数值差异，再用 trace 解释变化。overlap 可能缩短等待，也可能增加 staging buffer 或竞争 SM；局部 kernel 更快并不保证整个 step 更快。

## 7. 系统结果怎样反馈到架构

如果固定 active compute 后，更多更细的专家主要增加了小矩阵和重排成本，可以重新比较专家宽度与 top-k。若负担主要来自 expert path 的参数读取或通信维度，再评估 LatentMoE，并把额外 projection 计入成本。这些都是 profiling 后的架构实验，需与原模型做质量对照。

定位瓶颈时，将 kernel 耗时、输入工作量和前后依赖对应起来：较长的通信区间可能包含等待，较低的 GEMM 利用率也可能来自矩阵形状。第五篇继续计算这条路径的显存占用，第九篇讨论通信后端与物理调度。

## 系列导航

上一篇：[第三篇：Megatron Core MoE 源码路径解析]({% post_url 2026-07-11-megatron-core-moe-source %})

下一篇：[第五篇：100B 训练配置、并行策略与显存推演]({% post_url 2026-07-21-moe-100b-parallel-memory %})

## 参考资料

- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [Megatron Core Experts API](https://docs.nvidia.com/megatron-core/developer-guide/0.16.1/apidocs/core/core.transformer.moe.experts.html)
- [DeepEP GitHub Repository](https://github.com/deepseek-ai/DeepEP)
- [SGLang Expert Parallelism Documentation](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
- [SGLang `topk.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
