---
layout: post
title: "MoE我们喜欢你（四）：DeepEP 与 GroupedGEMM 如何拆解通信和计算瓶颈"
date: 2026-07-16
tags: [MoE, Infra, Megatron]
description: "解释 DeepEP/Flex Dispatcher 与 GroupedGEMM/Fused MoE GEMM 如何分别处理 MoE 大规模训练中的通信墙和计算墙。"
---


## 0. 导读

如果把 MoE layer 的执行路径拆开，优先要看两件事：token movement 和 expert compute。DeepEP 主要解决 **communication wall**，GroupedGEMM 主要解决 **compute efficiency wall**。

两者的重要性不同：

| 组件 | 解决的问题 | 实践优先级 |
|---|---|---|
| GroupedGEMM / Fused MoE GEMM | 多 expert 小 batch MLP 导致 GPU 利用率低、kernel launch 多 | 大规模 MoE 训练通常需要 |
| DeepEP / Flex Dispatcher | EP dispatch/combine 的 all-to-all 通信瓶颈 | 单节点/小规模可先不用；跨节点 EP、高专家数、大 batch 下优先考虑 |
| permute/unpermute fusion | token 重排和内存读写开销 | 强烈建议 |
| router fusion | router 小 kernel 和 aux/bias 计算开销 | 强烈建议 |
| EP-A2A overlap | 用计算隐藏 dispatch/combine 通信 | 大模型强烈建议 |

可以把边界简化为：**GroupedGEMM 提高专家计算效率，DeepEP 降低 token movement 成本**。一个大 MoE 训练如果只开了 EP，但没有处理这两件事，常见结果是 active FLOPs 降了，端到端 tokens/s 提升却有限。

![DeepEP 与 GroupedGEMM 的分工](/assets/img/blog/moe-we-like-you/04-deepep-groupedgemm-division.png)
*图 1：DeepEP/Flex Dispatcher 主要处理 dispatch/combine 的 token movement；GroupedGEMM/Fused MoE GEMM 主要处理本地 experts 的小矩阵计算效率。*

## 1. MoE layer 的真实执行路径

MoE layer 不是一个普通 MLP。它的端到端路径通常是：

| 阶段 | 做什么 | 主要系统问题 |
|---|---|---|
| route | router 为每个 token 计算 expert scores，选 top-k | router 精度、top-k 稳定性、load balance |
| permute | 按 expert 分组重排 tokens | 内存读写、索引构造、padding |
| dispatch | 把 token 发到持有目标 expert 的 EP rank | all-to-all、跨节点通信、尾延迟 |
| expert compute | 每个 expert 对自己的 token 执行 MLP | 小 GEMM、负载不均、GroupedGEMM 效率 |
| combine | 把 expert 输出按 router weight 合并 | all-to-all、combine kernel、权重乘加 |
| unpermute | 恢复原始 token 顺序 | 内存读写、索引散射 |

这条路径说明了为什么 MoE 的性能不能只看 active FLOPs。MoE 的 FLOPs 可能比 dense 低，但如果 dispatch/combine 和 token permutation 成为瓶颈，实际吞吐仍然会被通信和内存拖住。

## 2. 为什么 GroupedGEMM 几乎是大规模 MoE 必备

Dense MLP 的 GEMM 形状通常很规整：一个大 batch 乘一个大矩阵。MoE expert MLP 则不同：每个 expert 只接收被 router 分配来的 token，不同 expert 的 token 数还不一样。结果是：

- expert 维度上有很多小 batch GEMM；
- 每个 expert 的矩阵相似但输入 token 数不同；
- 如果逐 expert 单独 launch GEMM，kernel launch 和调度开销较高；
- 某些 expert token 少，GEMM 太小，GPU utilization 差；
- expert load imbalance 会进一步放大尾延迟。

GroupedGEMM 的目标就是把多个 expert 的 GEMM 合并成更高效的一组 kernel 调用，让 GPU 同时处理多组专家矩阵。它不改变 MoE 数学形式，但会显著改变实际吞吐。

### 2.1 GroupedGEMM 解决什么

| 问题 | 没有 GroupedGEMM | 有 GroupedGEMM |
|---|---|---|
| kernel launch | 每个 expert 或每组 expert 频繁 launch | 多 expert 合并调度 |
| 小矩阵效率 | token 少的 expert 利用率差 | 更好地填满 GPU |
| expert imbalance | straggler 更明显 | 仍受 imbalance 影响，但调度更高效 |
| FP8/MXFP8 | 难以在碎片 expert GEMM 上稳定高效 | 更容易和 fused / low-precision kernel 配合 |
| profiling | compute 看起来碎而乱 | compute 段更集中、更可分析 |

Megatron Core 文档把 GroupedGEMM 和 GA fusion、router fusion、token permutation fusion 一起列为 MoE 性能优化项；其 API 文档也把 Experts layer 描述为使用 GroupedGEMM 的高效多专家并行实现。

### 2.2 什么时候 GroupedGEMM 收益最大

GroupedGEMM 的收益通常在这些场景最明显：

- expert 数较多，例如 128、256、512+；
- top-k 较大，例如 top-8 或更高；
- local experts 多，单个 rank 上要处理多个 expert；
- global batch 较大，expert token 数更稳定；
- 使用 FP8/MXFP8 等低精度 expert compute；
- MoE 层占总计算比例高；
- profiler 显示 expert compute 段 kernel 很碎或 GPU utilization 低。

它的边界也要清楚：GroupedGEMM 不能解决 all-to-all 通信慢，也不能解决 router collapse。它只是在 tokens 已经到达本地 expert 后，把 expert compute 做快。

## 3. 为什么 DeepEP 是跨节点大 EP 的关键组件

EP 的基本做法是把 experts 分布在不同 GPU/rank 上。router 选出 expert 后，token 需要被发送到对应 expert 所在 rank，算完后再发送回来。这就是 MoE dispatch/combine。

标准 NCCL all-to-all 可以作为 baseline，但大规模 MoE 会遇到几个问题：

- token dispatch/combine 是每个 MoE layer 都发生的高频通信；
- token 数按 expert 不均匀分布，通信量不规则；
- top-k 越大，token 副本越多，payload 越大；
- EP 跨节点后，NVLink/NVSwitch 和 IB/RDMA 的带宽/延迟差异明显；
- 通信占用 SM、影响 expert compute overlap；
- combine 阶段还要处理 router weights 和 token 恢复顺序。

DeepEP 的定位就是面向 expert parallelism 的高性能通信库，重点提供 MoE dispatch/combine 的高吞吐、低延迟 all-to-all GPU kernels，并支持 FP8 等低精度路径。Megatron Core 通过 Flex Dispatcher 接入 DeepEP / HybridEP，把 token dispatcher 从传统 alltoall 路径中抽象出来。

### 3.1 DeepEP 解决什么

| 问题 | 普通 alltoall 路径 | DeepEP / Flex Dispatcher 路径 |
|---|---|---|
| dispatch/combine | 通用通信原语，MoE 语义不强 | 面向 MoE token dispatch/combine 优化 |
| 跨节点 EP | 容易受 IB/RDMA 延迟和不规则 payload 影响 | 更适合高专家数、跨节点 EP 场景 |
| 通信与计算 overlap | 需要额外调度和流水 | 更容易与 EP-A2A overlap 组合 |
| 低精度 payload | 依赖框架路径 | 支持 FP8 等低精度通信/计算配合 |
| SM 占用 | 通信 kernel 可能挤占 compute | DeepEP 目标之一是低或最小 SM occupation |

### 3.2 什么时候先不用 DeepEP

DeepEP 很重要，但不一定是 bring-up 第一步。建议分阶段：

- 小模型、小 expert 数、单节点实验：先用 alltoall 跑通 correctness。
- 架构实验：先把 router、load balance、GroupedGEMM、dropless 跑稳。
- tokens/s 开始被 EP A2A 卡住：切到 Flex Dispatcher + DeepEP。
- 跨节点 EP、大专家数、top-k 较大：DeepEP 基本应进入默认候选。
- 低延迟 serving 或 prefill：再区分 high-throughput 和 low-latency 通信路径。

这样做的原因是：DeepEP 改的是通信路径，调试时会和 routing、permutation、expert load、overlap、precision 交织在一起。先用简单路径确认模型数学和 load balance，再切高性能 dispatcher，排查边界会更清晰。

## 4. DeepEP 和 GroupedGEMM 如何配合

MoE 性能优化的目标不是单独让某个 kernel 更快，而是让 dispatch、expert compute、combine 形成更好的流水。

可以把 MoE layer 看成两段：

| 段 | 代表组件 | 目标 |
|---|---|---|
| token movement | permute、dispatch、combine、unpermute、DeepEP | 减少和隐藏 token movement 成本 |
| expert compute | GroupedGEMM、Fused MoE GEMM、FP8/MXFP8 | 提高专家 MLP 的 GPU 利用率 |

对应到系统路径，大概是：

```python
tokens = permute_by_expert(hidden, topk_ids)
tokens = ep_all_to_all_dispatch(tokens)
expert_out = grouped_gemm(tokens, tokens_per_expert)
expert_out = ep_all_to_all_combine(expert_out)
hidden = unpermute_and_weighted_sum(expert_out, topk_weights)
```

### 4.1 实现思路：一个管 token movement，一个管 expert kernel

可以把实现边界看得更清楚一点：Flex Dispatcher 是 Megatron 里的 dispatcher 抽象层，负责把 `routing_map/topk_ids/topk_weights` 这类 MoE metadata 转成后端可执行的通信计划；DeepEP/HybridEP/NCCL-EP 是具体后端，重点优化 dispatch/combine 这两次 token movement。它们关心的是 splits、permute、all-to-all、unpermute、buffer shape、通信 stream、SM 占用和 overlap。

GroupedGEMM / FusedMoE 则站在另一侧：token 已经按 expert 分好组，接下来要把多个 expert 的 FC1、activation、FC2 尽量合成少量高利用率 GPU kernel。训练侧常见口径是 TEGroupedMLP / GroupedLinear / fused grouped MLP；推理侧的 FusedMoE 还可能把 routing metadata、量化、permute 和 GEMM 进一步融合进 serving kernel。

![DeepEP/Flex 与 GroupedGEMM/FusedMoE 的实现边界](/assets/img/blog/moe-we-like-you/04-deepep-flex-groupedgemm-boundary.png)
*图 2：Flex/DeepEP 主要处理 token 从原 rank 到 expert rank、再从 expert rank 回原 token rank 的移动；GroupedGEMM/FusedMoE 主要处理本地 expert batches 上的 FC1/activation/FC2 计算。*

### 4.2 推理侧：以 SGLang 为例

推理侧的 MoE 加速也可以沿着同一条边界理解。SGLang 把 EP serving 拆成两个可替换后端：

| 后端 | 命令行口径 | 负责什么 |
|---|---|---|
| A2A backend | `--moe-a2a-backend` | token dispatch/combine，例如 `deepep`、`flashinfer`、`mooncake`、`nixl` |
| MoE runner backend | `--moe-runner-backend` | 本地 expert compute，例如 `triton`、`deep_gemm`、`flashinfer_cutlass`、`flashinfer_trtllm` |

所以 SGLang 的 serving 路径大概是：

```python
topk = router_topk(hidden, correction_bias, grouped_topk)
dispatch_out = a2a_backend.dispatch(hidden, topk)
expert_out = moe_runner(dispatch_out, quantized_expert_weights)
hidden = a2a_backend.combine(expert_out, topk)
```

这里的加速点和训练侧类似，但目标不同：

- **top-k / routing kernel**：尽量把 grouped top-k、`correction_bias`、renormalize、logical-to-physical expert id 映射做成高效 kernel-friendly metadata。
- **A2A backend**：prefill 更看重高吞吐，decode 更看重低延迟和 CUDA Graph 兼容；DeepEP 这类后端会区分 normal / low_latency 模式。
- **MoE runner**：根据硬件和量化格式选 Triton、DeepGEMM、FlashInfer/CUTLASS/TRT-LLM 等 fused MoE kernel，把 expert GEMM、activation、量化/反量化尽量放进高效路径。
- **EPLB / expert placement**：serving 侧还要根据真实请求的 expert hotness 做 expert 放置或冗余，降低热门 expert 造成的尾延迟。
- **overlap**：SGLang 这类 serving runtime 会进一步做 two-batch / single-batch overlap，把 attention、dispatch、expert compute、combine 尽量交错起来。

这也是为什么推理侧的 MoE 优化不只是“把训练 kernel 拿来跑 serving”。Serving 有 continuous batching、prefill/decode 分离、KV cache、量化权重、EPLB 和 p99 latency 约束；MoE runner 和 A2A backend 必须分开选，才有空间针对不同阶段做不同优化。

理想状态是：

- dispatch 尽量快，并和前后计算 overlap；
- tokens 到达本地后，GroupedGEMM 高效执行 expert MLP；
- combine 尽量被后续计算隐藏；
- shared expert compute 可以与 routed expert path overlap；
- router/permute/unpermute 不产生太多小 kernel gap；
- expert/device load balance 不制造 straggler。

这也是为什么只开一个组件经常不够：如果 GroupedGEMM 很快，但 dispatch 慢，GPU 还是等通信；如果 DeepEP 很快，但 expert GEMM 太碎，GPU 还是算不满。

## 5. Megatron Core 里的相关能力

Megatron Core 的 MoE 文档和 README 已经把这条路径拆得比较清楚。相关能力可以按下面理解：

| 能力 | 作用 | 典型开关/口径 |
|---|---|---|
| Flex Dispatcher | 抽象 token dispatcher backend | `moe-token-dispatcher-type=flex` |
| DeepEP backend | 高性能 EP dispatch/combine | `moe-flex-dispatcher-backend=deepep` 或相关 DeepEP 开关 |
| HybridEP backend | NVIDIA 优化 dispatcher，面向新硬件拓扑 | `moe-flex-dispatcher-backend=hybridep` |
| GroupedGEMM | 多 expert GEMM 合并执行 | `moe-grouped-gemm` |
| router fusion | 融合 router 相关小 kernel | `moe-router-fusion` |
| permute fusion | 融合 token permute/unpermute | `moe-permute-fusion` |
| EP-A2A overlap | 隐藏 EP all-to-all 通信 | `overlap-moe-expert-parallel-comm`、`delay-wgrad-compute` |
| selective recompute | 控制 MoE/MLA/MLP activation memory | `recompute-granularity=selective` 与模块列表 |

这里不建议一上来把所有开关全打开。更稳的顺序是：先 alltoall + GroupedGEMM 跑通，再加 fusion，再切 Flex/DeepEP，最后调 overlap 和低精度。

## 6. Profiling：怎么判断瓶颈在 DeepEP 还是 GroupedGEMM

### 6.1 如果是通信墙

典型现象：

- EP all-to-all 时间占比高；
- dispatch/combine 周期性出现在每个 MoE layer；
- GPU timeline 上 expert compute 前后有明显等待；
- EP 从单节点扩到跨节点后 tokens/s 明显下降；
- device load CV 高，某些 EP rank 成为 straggler；
- 增大 batch 后通信吞吐改善，但 latency 仍显著。

优先动作：

| 动作 | 目的 |
|---|---|
| 检查 expert load CV 和 device load CV | 先排除负载不均 |
| 确认 EP × TP 是否跨高速域 | 避免不必要跨节点 EP/TP |
| 切 Flex Dispatcher + DeepEP | 优化 dispatch/combine |
| 开 EP-A2A overlap | 隐藏通信 |
| 调整 PP/VPP | 避免盲目扩大跨节点 EP |
| 评估 group/node-limited routing | 减少跨节点 token movement |

### 6.2 如果是计算效率墙

典型现象：

- expert compute 段 GPU utilization 低；
- timeline 上很多小 GEMM 或小 kernel；
- 单个 expert token 数不足；
- TP 把 expert MLP 切得太碎；
- GroupedGEMM utilization 低；
- FP8/MXFP8 未启用或 scale 不稳定。

优先动作：

| 动作 | 目的 |
|---|---|
| 开 GroupedGEMM | 合并多 expert GEMM |
| 降低 expert TP / ETP | 保留更大的本地 GEMM |
| 增大 micro-batch 或 global batch | 提高每 expert token 数 |
| 检查 top-k 与 expert 数 | 避免专家过细导致每 expert token 太少 |
| 开 FP8/MXFP8 expert compute | 提高吞吐、降低带宽 |
| 保留 router/norm 等敏感模块高精度 | 避免低精度导致路由不稳 |

### 6.3 如果是内存/重排墙

典型现象：

- permute/unpermute 占比不小；
- token index 构造、scatter/gather 产生明显开销；
- activation memory 压力导致 full recompute；
- CPU launch gap 明显；
- shared expert 和 routed expert path 串行化。

优先动作：

| 动作 | 目的 |
|---|---|
| 开 permute/unpermute fusion | 减少 token 重排开销 |
| 用 memory-efficient token permutation | 降低中间内存读写 |
| selective recompute | 避免 full recompute 过重 |
| overlapped shared expert | 让 shared expert 不阻塞 routed path |
| CUDA Graph scoped capture | 减少 CPU launch gap |

## 7. 推荐调优顺序

一个实用 bring-up 顺序：

| 阶段 | 做什么 | 避免同时修改 |
|---|---|---|
| correctness | alltoall dispatcher、router FP32、dropless、基础 load balance | 暂不引入 DeepEP/FP8/overlap |
| compute baseline | 打开 GroupedGEMM，确认 expert compute 稳定 | 暂不改 top-k/expert 数 |
| fusion | 打开 router fusion、permute fusion、cross entropy fusion | 暂不切换 dispatcher |
| communication | 切 Flex Dispatcher + DeepEP/HybridEP | 暂不改路由策略 |
| overlap | 打开 EP-A2A overlap、shared expert overlap | 暂不改 pipeline layout |
| precision | 逐步打开 FP8/MXFP8 | 暂不低精度化 router/norm/output |
| topology | 调 EP/TP/PP/CP/Parallel Folding | 暂不改模型结构 |

这个顺序的目的，是先确认模型数学和专家负载，再分别优化计算、通信和内存。一次改太多会让 tokens/s 的变化难以归因。

## 8. 和架构设计的反向约束

DeepEP 和 GroupedGEMM 不是纯系统实现，它们会反过来影响架构选择。

### 8.1 对 expert granularity 的约束

更多、更细的 experts 提高组合空间，但会让每个 expert 的 token 数下降，GroupedGEMM 压力上升。如果专家过细，系统上可能出现：

- expert GEMM 太小；
- load balance 统计噪声更大；
- dispatch payload 更碎；
- device load 更难平衡；
- top-k 需要提高来维持质量，但通信又变贵。

因此，fine-grained MoE 不能只按模型质量选 expert 数，还要看每 step 每 expert token 数和 GroupedGEMM 利用率。

### 8.2 对 top-k 的约束

top-k 提高会增加专家组合能力，但也线性或近似线性增加 dispatch/combine payload 和 expert compute。没有 DeepEP/overlap 时，top-k 从 4 到 8 或从 8 到 16，系统代价可能比 loss 改善更明显。

所以 top-k 的选择应同时看：

- validation loss；
- expert load CV；
- EP A2A 时间；
- GroupedGEMM utilization；
- end-to-end tokens/s；
- serving latency。

### 8.3 对 LatentMoE 的启发

LatentMoE 的系统意义在这里会更明确：如果 MoE 的瓶颈已经从 FLOPs 转向 dispatch bytes、expert parameter bytes 和 all-to-all payload，那么把 expert path 压到 latent space 就不是小修小补，而是直接减少通信墙和计算墙的架构方案。

也就是说，DeepEP 和 GroupedGEMM 是把普通 MoE 系统做快；LatentMoE 是改变 MoE path 本身，让系统需要移动和计算的数据变少。

## 9. 什么时候这篇里的优化没那么关键

不是所有 MoE 实验都需要一开始就启用完整系统优化。

可以暂时不用 DeepEP 的情况：

- 单节点小 MoE；
- expert 数较少；
- top-k 较小；
- 只做 architecture ablation；
- EP A2A 不是 profiler 主瓶颈；
- 使用 allgather/alltoall 已经能满足实验吞吐。

GroupedGEMM 在 bring-up 初期不一定必须打开，但只要进入正式训练或规模稍大，它通常很快会变成必要项。没有 GroupedGEMM 的 MoE 实验可以用于功能验证，不太适合代表真实大规模训练效率。

## 10. 最小实验矩阵

如果要系统评估 DeepEP 和 GroupedGEMM，可以做下面这个矩阵：

| 维度 | baseline | variant A | variant B |
|---|---|---|---|
| dispatcher | alltoall | Flex + DeepEP | Flex + HybridEP |
| expert compute | standard expert GEMM | GroupedGEMM | GroupedGEMM + FP8/MXFP8 |
| permutation | baseline permute | permute fusion | memory-efficient permutation |
| overlap | none | EP-A2A overlap | EP-A2A + shared expert overlap |
| routing topology | unrestricted | group-limited | node-limited |
| parallel mapping | EP in-node | EP cross-node | Parallel Folding |

核心指标：

- end-to-end tokens/s；
- MFU / HFU；
- EP dispatch time；
- EP combine time；
- GroupedGEMM utilization；
- expert load CV；
- device load CV；
- permutation/unpermutation time；
- CPU kernel gap；
- activation memory；
- validation loss 是否受系统路径变化影响。

## 11. 结论

DeepEP 和 GroupedGEMM 需要单独讲，是因为它们代表了 MoE 训练里两个绕不开的事实：

1. **MoE 的 token 需要跨 rank 移动**。只要 experts 分布在不同 rank，dispatch/combine 就会成为一等系统问题。DeepEP/Flex Dispatcher 解决的是 token movement 的通信墙。
2. **MoE 的 expert compute 是很多不规则小 GEMM**。GroupedGEMM/Fused MoE GEMM 解决的是 expert compute 的 GPU 利用率问题。

所以实践口径不是“只开 EP 就足够”，而是：

**用 EP 放专家，用 DeepEP/Flex 降低 token movement 成本，用 GroupedGEMM 提高专家计算效率，用 fusion/overlap 把重排、通信和计算串成流水，再用 profiling 判断下一步该调通信、计算还是内存。**

## 系列导航

上一篇：[第三篇：Megatron Core MoE 源码剖析]({% post_url 2026-07-11-megatron-core-moe-source %})  
下一篇：[第五篇：100B MoE 训练配置推演：并行策略与显存估算]({% post_url 2026-07-21-moe-100b-parallel-memory %})

## 参考资料

- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [Megatron Core Experts API](https://docs.nvidia.com/megatron-core/developer-guide/0.16.1/apidocs/core/core.transformer.moe.experts.html)
- [DeepEP GitHub Repository](https://github.com/deepseek-ai/DeepEP)
- [SGLang Expert Parallelism Documentation](https://docs.sglang.io/docs/advanced_features/expert_parallelism)
- [SGLang `topk.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
