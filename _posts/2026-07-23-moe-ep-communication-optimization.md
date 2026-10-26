---
layout: post
title: "MoE我们喜欢你（九）：DeepEP v1/v2、MoonEP 与 EP 通信优化"
series: moe-we-like-you
series_order: 9
last_modified_at: 2026-10-26
date: 2026-07-23
tags: [MoE, Infra, Inference]
description: "从 DeepEP v1/v2、MoonEP、UltraEP、LLEP、SGLang Waterfill/LPLB 出发，解释 MoE expert parallelism 的 dispatch/combine、prefill/decode 分流、负载均衡和静态 shape 优化。"
---


## 0. 通信之外，还要决定在哪里执行

第四篇讨论如何定位 MoE 的通信、计算和重排瓶颈。这一篇进一步区分两层系统工作：底层后端负责移动 token，上层 planner 决定 logical expert 的工作落到哪个 physical rank 或 replica。

二者可以组合。通信库换得更快，不会自动消除 hot rank；增加副本也可能引入权重同步、额外 GEMM 和 buffer。下面先说明数据量与阶段，再分别讨论通信路径、副本规划和 serving 调度。

## 1. 先固定数据量和阶段

设每个 EP rank 输入 `T` 个 token，每个 token 选择 `K` 个 experts，hidden size 为 `H`，每元素 `b` 字节。逐 token–expert 分配展开后的逻辑输入体积是：

```text
T * K * H * b
```

这不是实际跨网络字节数的下界。本地 expert 不必跨网络访问；若多个目标 expert 在同一 rank，backend 也可能复用传输。低精度、padding、转发和 metadata 又会改变实际链路流量。比较后端时，要注明统计的是逻辑 activation、跨 rank payload，还是物理链路 bytes。

一次 dispatch 通常包括 pack/layout、A2A 和接收端 metadata 处理，再将输入交给 expert kernel。combine 反向回传输出并恢复 token 顺序。通信时间以外，还应记录 rank 尾部耗时、通信 SM 占用、计数同步和 buffer 峰值。

| 阶段 | 常见工作形态 | 优先测量 |
|---|---|---|
| 预训练 / 继续训练 | 大 token batch，重复执行各 MoE layer | step time、dispatch/combine、SM 竞争、rank load |
| SFT / RL | 数据分布更窄，可能出现专家热点 | 除吞吐外，检查路由状态、overflow 和峰值显存 |
| prefill | prompt token 多 | TTFT、吞吐、通信与 expert compute 分解 |
| decode | 单步 token 较少，反复执行 | TPOT/p99、固定开销、host sync、CUDA Graph |

后续用 `T_e` 表示一个 expert 收到的 token 数，`T_bar` 表示均衡分配时的数量。例如 `max_vio = max_e(T_e / T_bar) - 1` 描述最大过载，但相同 token 数不保证相同执行时间。

## 2. DeepEP：通信路径、metadata 与接口复用

[DeepEP][deepep] v1 区分 normal 与 low-latency kernels。normal 面向训练和 prefill 的较大 payload，处理 NVLink 到 RDMA 域的转发，并允许控制通信 SM 数；low-latency 面向小 token batch，重点是固定延迟和避免 host 同步。

[v1 legacy 文档][deepep-v1] 的训练型测试采用 4096 tokens/batch、hidden 7168、top-8、FP8 dispatch 与 BF16 combine。decode 型测试的 token batch 更小。两个场景的 payload、精度和目标不同，不适合只拿一个带宽或延迟数字横向比较。

DeepEP v2 改用 NCCL Gin backend，以 `ElasticBuffer` 统一 high-throughput / low-latency API，并按模型和硬件参数计算 SM / QP 配置。README 报告的最高 1.3x peak performance 与最多 4x SM count 节省，属于其测试设置；迁移时应验证完整 MoE layer 的耗时，尤其是释放出的 SM 是否改善了 expert compute。

![DeepEP v1/v2 EP 通信路径](/assets/img/blog/moe-we-like-you/09-deepep-v1-v2-ep.png)
*图 1：v1 区分 normal / low-latency 路径，v2 用 ElasticBuffer 与 EPHandle 统一接口。*

接口还要提供 expert kernel 所需的布局与计数。下面省略具体参数和版本分支，只展示 handle 的生命周期：

```python
recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(...)
expert_out = grouped_gemm(recv_x, handle.num_recv_tokens_per_expert_list)
combined_x, _, event = buffer.combine(expert_out, handle=handle)
```

框架可能还需按 expert 重排接收数据；不能仅凭 dispatch 完成就假定可直接送入任意 GroupedGEMM。combine 复用 handle 中的路由关系，避免重复构造 metadata。

在固定 routing mask 下，dispatch 是将 token 复制/散射到 experts，反向则将各分支梯度收回并累加，所以呈 combine 的通信模式；combine 的反向相应呈 dispatch 模式。router weights 的梯度和加权位置还需按具体实现处理。

serving 接入时，SGLang 可以按阶段选择 normal / low_latency；[vLLM][vllm-ep] 也区分 high-throughput 与 low-latency EP backend。mode、handle caching 和 CUDA Graph 应一起验证，特别要检查获取收发计数是否引入隐式 CPU/GPU 同步。

## 3. 动态副本：把规划放在 gate 与 dispatch 之间

这类方法保留 logical top-k，但改变物理执行位置。它们需要将 planner、weight movement 和训练梯度回收都计入 critical path。

### 3.1 MoonEP：约束每个 rank 的总工作量

[MoonEP][moonep] 根据当前 routing 结果在线规划少量 redundant experts，将工作分给主副本与临时 slots。输入满足其接口约定时，每个 rank 每层固定处理 `T × K` 条 expert-token 分配；训练副本的梯度再归并回 home rank。

![MoonEP static-shape 物理执行](/assets/img/blog/moe-we-like-you/09-moonep-static-shape.png)
*图 2：MoonEP 调整 physical mapping，使 rank 级总 token 数固定；logical expert choice 保持不变。图中的 S 对应本文的输入 token 数 T。*

固定 rank 总量有利于 activation shape 和 buffer 规划，但各 expert 的 token count 与 `cu_seqlens` 仍可随 routing 改变。实现还要求连续的 symmetric-memory 权重、prefetch slots 和训练 grad/reduce buffer。这是一套执行约定，需要框架适配，不能视作普通 alltoall 的无条件替换。

### 3.2 UltraEP：按本批次负载共同规划副本与配额

[UltraEP][ultraep] 在 gate 后获得当前 microbatch 的 exact load，再求 quota table `U = {u_{e,r}}`：expert `e` 在 rank `r` 承担多少 token。非零配额决定哪里需要实例，dispatch 按配额分配 token，训练时再归并副本梯度。

![UltraEP quota-driven exact-load balancing](/assets/img/blog/moe-we-like-you/09-ultraep-quota-driven-exact-load-balancing.png)
*图 3：quota planner 同时决定 expert replica 与 token split。*

它与 MoonEP 都使用当前 batch 的路由信息，侧重点不同：MoonEP 强调固定 rank 总量与静态执行形状，UltraEP 强调 rack-scale node 上副本和配额的联合规划。后者依赖较快的 scale-up fabric，让即时复制和状态通信有机会抵得上减少的尾部等待。

### 3.3 LLEP 与训练时的回收成本

[LLEP][llep] 从过载 GPU 向较空 GPU spill 多余 token 和对应 expert weights。它需要选择 spill 阈值与最小 GEMM 粒度：拆得太细，权重转移和额外小 GEMM 可能超过平衡收益。

训练使用任何动态副本时，都要确认梯度回到正确的原参数，optimizer 状态的归属不因临时复制而改变。建议分别计时 planner、weight sync、dispatch、GEMM 与 grad reduce；仅报告最终 load CV，无法判断整个 step 是否受益。

## 4. Serving 调度：已有副本怎样分工作

serving 的短时热点未必适合通过更新 router 来修正。若保持模型语义，可调整 shared work、已有 replicas 或物理放置，但这些动作有不同前提。

[Waterfill / LPLB][waterfill-lplb] 分别处理两种工作。Waterfill 把 shared expert work 分配给 routed load 较低的 rank，避免它叠加在已较忙的 rank 上；LPLB 假设 replicated routed experts 已存在，用当前 batch 的 token count 求一个小 LP，决定 token 分到哪些副本，使最大 rank load 尽量小。LPLB 不移动权重；没有副本时也就没有这类选择空间。

[TEMPO][tempo] 再把目标从 token count 改成执行时间。小 batch 激活一个副本，就要付出读取 expert weights 和 GEMM tile padding 的成本；把相同 token 数均分到更多副本，可能增加总开销。这里的冷副本指本批次尚未参与计算的副本，不等同于权重尚未部署。

![TEMPO makespan-aware dispatch](/assets/img/blog/moe-we-like-you/09-tempo-makespan-aware-dispatch.png)
*图 4：TEMPO 通过测量的 expert/GEMM cost model 优化最大 rank 耗时。*

因此 TEMPO 用 measured cost model 选择 split，优化 `max_r t_r`，而非只优化最大 token count。评价时也要检查 cost model 是否适用于当前 batch、量化与 grouped GEMM 实现。

[EPIC][epic] 的范围更广：在线 placement、migration 和 expert activation 同时考虑计算与网络负载，再结合传输消重和通信/计算 overlap。它提醒我们，负载调度的收益仍受物理拓扑和 transport 约束。

## 5. 后端接口、弹性与特定拓扑

下列工作改变的主要是通信接口或执行基础设施，应与上面的 planner 分层比较：

| 工作 | 主要需求 | 使用时要核对 |
|---|---|---|
| [NCCL EP][nccl-ep] | 将 dispatch/combine 做成统一 primitive | HT/LL mode、后端支持和框架适配 |
| [UCCL-EP][uccl-ep] | 异构 GPU/NIC 上可移植的 EP 通信 | CPU proxy、RDMA 路径及额外控制开销 |
| [NIXL-EP][nixl-ep] | 动态 add/remove ranks 的连接与通信 | 弹性成员变化与 serving 状态管理 |
| [UBEP][ubep] | superpod 内更细的依赖调度 | topology、flag/synchronization 和任务粒度 |
| [Mixture-of-Kittens][mok] | NVL72 上融合 MoE training 的执行路径 | megakernel、ring buffers 及对应硬件约束 |

其中 megakernel 还融合了通信与专家计算，范围已超过单个 A2A backend。FlashInfer MNNVL 等路径也有明确的互联前提。选择时先确定高速域、RDMA 路径和成员是否动态变化，再判断接口与 kernel 能否接入当前 runtime。

## 6. 由 trace 决定下一项实验

| 当前现象 | 优先验证 |
|---|---|
| 大 batch A2A 占比高 | high-throughput 后端、通信 SM 占用与 overlap |
| rank token 总量动态变化引起 buffer 压力 | MoonEP 等约束执行形状的路径及适配成本 |
| 即时热点明显，weight movement 足够快 | 动态副本与 quota/spill 规划的端到端收益 |
| decode 固定开销高 | low-latency mode、host sync、handle/CUDA Graph 复用 |
| 已有 replicas，但 batch 内仍不均 | LPLB；若 token 均衡仍不能缩短尾部，再检查 TEMPO 式时间模型 |
| shared compute 加重热点 | shared fusion 与 Waterfill 的收益及开销 |
| 后端不匹配网络或拓扑 | 可移植、弹性或拓扑特化路径的接口支持 |

每次实验记录完整 MoE layer 或 step 的耗时，并把新增 planner、buffer 和同步成本算进去。这样可以判断是在改善通信、降低 rank tail，还是仅把等待转移到了另一个阶段。

## 系列导航

上一篇：[第八篇：稀疏计算与参数复用的新方向]({% post_url 2026-07-23-moe-future-sparse-architecture %})

下一篇：[第十篇：视觉生成中的专家分工]({% post_url 2026-07-23-moe-visual-generation %})

## 延伸阅读

读 DeepEP / MoonEP 时，最好直接对着官方 README 和接口看：[DeepEP README][deepep] 解释 high-throughput / low-latency dispatch-combine kernel，[MoonEP README][moonep] 则把 dynamic redundant experts 和固定 rank 总 token 数的执行约定 写得很清楚。

如果更关心 serving runtime，[SGLang EP 文档][sglang-ep] 和 [vLLM EP deployment][vllm-ep] 可以放在一起读：前者更贴近 backend 选择和 DeepEP / Mooncake / NIXL-EP 模式切换，后者更贴近 vLLM 里的 EP backend、disaggregated serving 和 DeepEP kernel 依赖。

[Waterfill / LPLB 博客][waterfill-lplb] 分别解释 shared expert dispatch 和 replicated expert dispatch，并给出 serving 侧的实现与测量。AMD 的 [vLLM MoE Playbook][amd-vllm-moe-playbook] 可用于补充分布式部署中 TP/DP/EP 组合及延迟、吞吐的取舍。

[MoonEP / UltraEP 等 EP 优化介绍][moe-ep-zhihu] 是中文综述入口，适合先了解方法关系；实现时可结合对应 README、论文和 runtime 文档。

## 参考资料

- [DeepEP GitHub Repository][deepep]
- [DeepEP v1 legacy documentation][deepep-v1]
- [MoonEP GitHub Repository][moonep]
- [SGLang Expert Parallelism Documentation][sglang-ep]
- [Improving DeepEP MoE Load Balance in SGLang with Waterfill and LPLB][waterfill-lplb]
- [Least-Loaded Expert Parallelism][llep]
- [UltraEP: Unleash MoE Training and Inference on Rack-Scale Nodes with Near-Optimal Load Balancing][ultraep]
- [Balancing and Beyond: Communication-Centric Optimizations in Expert Parallelism][epic]
- [TEMPO: Makespan-Aware Expert-Parallel Load Balancing Across Memory- and Compute-Bound Regimes][tempo]
- [NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL][nccl-ep]
- [NIXL EP: Expert-Parallel Communication Example][nixl-ep]
- [UCCL-EP: Portable Expert-Parallel Communication][uccl-ep]
- [UBEP: Re-architecting Expert Parallelism Communication Library for Production Superpods][ubep]
- [vLLM Expert Parallel Deployment][vllm-ep]
- [Mixture-of-Kittens: our open-source MoE megakernel for NVL72s][mok]
- [DeepSeek EPLB Repository][eplb]
- [Megatron Core MoE Documentation][megatron-moe]
- [DeepSeek-V3 Technical Report][deepseek-v3]
- [MoonEP / UltraEP 等 EP 优化介绍][moe-ep-zhihu]

[deepep]: https://github.com/deepseek-ai/DeepEP
[deepep-v1]: https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md
[moonep]: https://github.com/MoonshotAI/MoonEP
[sglang-ep]: https://docs.sglang.io/docs/advanced_features/expert_parallelism
[waterfill-lplb]: https://www.lmsys.org/blog/2026-06-26-waterfill-lplb
[llep]: https://arxiv.org/abs/2601.17111
[ultraep]: https://arxiv.org/abs/2606.04101
[epic]: https://dl.acm.org/doi/10.1145/3789240.3829201
[tempo]: https://arxiv.org/abs/2608.13057
[nccl-ep]: https://arxiv.org/abs/2603.13606
[nixl-ep]: https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep
[uccl-ep]: https://arxiv.org/abs/2512.19849
[ubep]: https://arxiv.org/abs/2607.06202
[vllm-ep]: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
[amd-vllm-moe-playbook]: https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html
[mok]: https://cursor.com/blog/mixture-of-kittens
[eplb]: https://github.com/deepseek-ai/EPLB
[megatron-moe]: https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html
[deepseek-v3]: https://arxiv.org/abs/2412.19437
[moe-ep-zhihu]: https://zhuanlan.zhihu.com/p/2067703296500954047
