---
layout: post
title: "长话长说（四）：超长上下文训练的并行、显存与调度"
date: 2026-08-18
last_modified_at: 2026-10-21
tags: [LLM, Long Context, Pretraining, Distributed Training]
description: "讨论 128K、256K 到 1M token 上下文训练中的系统瓶颈：attention 计算、activation 显存、SP/CP、Ring Attention、Ulysses、linear attention CP、varlen packing、MoE/EP 负载、pipeline、attention disaggregation 与 compiler 调度。"
---

## 写在前面

> 这是长上下文 attention 架构系列的第四篇。第一篇从 [attention 与 KV cache 总览]({% post_url 2026-07-27-long-context-attention-architecture-survey %}) 开始，第二篇单独拆 [linear attention]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})，第三篇讨论 [长上下文 attention 设计趋势]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %})。这一篇不再问模型该长成什么样，而是问训练侧：结构基本定了以后，怎样让 128K、256K，甚至 1M context 在大规模集群上跑得动、跑得稳。 完整目录见[系列导航]({% post_url 2026-07-27-long-context-attention-architecture-survey %}#series-map)。

长上下文训练有个容易误导人的说法：把 `max_position_embeddings` 拉大，换上更省显存的 attention kernel，再把 batch 调小一些，就可以继续训练。

真正上手时，麻烦通常不是这样排队出现的。sequence length 变长以后，attention core 的计算、activation 保存、跨卡通信、变长样本 packing、MoE router 负载、梯度累积和调度策略会一起变形。单看某一个优化点，往往都能讲得通；把它们放到同一次训练里，就会发现一个地方省下来的显存，可能立刻被另一个地方的通信 buffer 或 expert token buffer 吃掉。

所以这篇把问题收窄到训练 infra：

> 当上下文从 32K 推到 128K、256K、1M 后，训练系统到底要同时解决哪几类开销？

下面的讨论默认模型仍包含 full softmax attention 或周期性的全局 attention。纯 linear attention、pure state-space 或极端 sparse attention 会改变成本主项；但公开的大规模 LLM recipe 多数仍是 hybrid，训练系统也必须和这些 hybrid 结构一起设计。

文章先建立成本模型，再解释 kernel 的硬件瓶颈和 softmax/recurrent CP；随后讨论显存生命周期、varlen 动态分配、pipeline 与 attention disaggregation，最后处理 MoE 的耦合约束和落地检查。FA4、DCP、Zeppelin、DistCA 分别对应 kernel、attention 任务放置、模块布局和跨任务负载均衡。

## 1. 先估训练侧的开销

设 micro-batch 为 $$B$$，序列长度为 $$L$$，hidden size 为 $$d$$，query heads 数为 $$H$$，每个 head 维度为 $$d_h$$，其中 $$d=Hd_h$$。单层 full attention 的核心计算可以粗略写成：

$$
\mathrm{MACs}_{QK} \approx BHL^2d_h = BL^2d,
$$

$$
\mathrm{MACs}_{AV} \approx BHL^2d_h = BL^2d.
$$

如果把 multiply-add 记成 2 FLOPs，那么 attention core 约为 $$4BL^2d$$ FLOPs。严格的 causal mask 只用下三角，常数会小一些；FlashAttention 一类 kernel 又会改变 IO 和实际吞吐。但路线判断不变：只要是 full softmax attention，训练和 prefill 的 attention 主项仍随 $$L^2$$ 增长。

显存这边更麻烦。训练要为反向传播保存 activation，至少会碰到几类对象：

| 对象 | 随 $$L$$ 的关系 | 备注 |
|---|---:|---|
| hidden states、residual、LayerNorm 输入 | 线性 | 层数多时很可观 |
| Q/K/V、中间 projection | 线性 | 是否保存取决于 recompute 策略 |
| attention logits/probs | 传统实现近似二次 | FlashAttention 避免显式物化 $$L^2$$ 矩阵 |
| MLP/MoE 中间激活 | 线性 | MoE 还会引入 router、per-expert buffer |
| 通信 overlap buffer | 线性或与分片有关 | CP、EP、FSDP/ZeRO 都可能引入 |

这就是为什么“把 micro-batch 降到 1”经常只是开始。micro-batch 变小以后，单步 sample 数少了，但训练关心的通常是 tokens per optimizer step：

$$
\mathrm{global\ tokens/step}
= B \times L \times \mathrm{DP} \times \mathrm{grad\_accum}.
$$

这里的 DP 是数据并行副本数。Context Parallelism 只是把同一条长序列切到多张卡上，不应该再额外乘进 global batch。改变长度后，应先计算新的 $$B\times L\times\mathrm{DP}$$，再按目标 tokens/step 调整 gradient accumulation；它可能需要增加，也可能减少。比较不同长度阶段时，还应检查学习率 schedule 的 token 口径。

换句话说，长上下文训练不是简单地“同样 batch，更长 sequence”。它更像把原来的一个 batch 拆成很多跨卡协作的长样本，再按新的每个 micro-step token 数调整梯度累积。

## 2. Attention kernel 与硬件瓶颈迁移

CP、调度和 offload 都建立在单个 kernel 的执行代价之上。先区分 HBM 中间矩阵、片上数据搬运与计算流水线，后面才能判断通信是否有足够计算可以重叠。

[FlashAttention](https://arxiv.org/abs/2205.14135) 的位置要放准。它是 exact attention kernel，通过 tiling 减少 HBM 和 SRAM 之间的数据搬运，并避免显式保存完整 attention matrix。它能显著降低 attention 的 IO 和 memory footprint，也能提高实际吞吐；但它没有把 full attention 的数学计算从 $$O(L^2)$$ 变成 $$O(L)$$。[FlashAttention-3](https://arxiv.org/abs/2407.08608)继续利用 Hopper 上 Tensor Cores 和 TMA 的异步能力，以及 FP8 低精度路径；这说明 kernel 层还在快速演进，但 full attention 的二次交互语义没有被它改掉。所以在 128K、256K 训练里，FlashAttention 往往是入场券，不是终点。

可以把 FlashAttention 理解成这样：普通实现像是先把整张 $$L \times L$$ 注意力表摊在桌上，再做 softmax 和乘 $$V$$；FlashAttention 不摊这张大表，而是每次拿一小块 $$Q$$ 和一小块 $$K,V$$ 到片上 SRAM 里算，边扫边维护 online softmax 的最大值和归一化项，最后只把输出 $$O$$ 写回 HBM。省下来的主要是 HBM 往返和 $$L^2$$ 中间矩阵的保存，不是少算了所有 token 对。

[FlashAttention-4](https://arxiv.org/abs/2603.05451)处理的是 Blackwell 上的新失衡：矩阵乘变快后，shared memory 搬运和 softmax 指数运算没有同比加速，Tensor Cores 可能开始等待它们。forward 因此不把“算完一块 QK → 做完 softmax → 再算 PV”串行排满，而让不同块的 matmul、softmax 与数据搬运交错执行。论文还把部分指数计算用多项式近似分担给乘加单元，并减少不必要的归一化重缩放。

backward 的问题则是中间量在片上反复读写，以及多块向同一梯度累加。FA4 利用 tensor memory（TMEM）和两个 CTA 协作的 MMA 模式，减少一部分 shared-memory 访问和原子加法。这里应测的是“matmul 在等谁”：指数单元、片上带宽还是梯度写回，而非只比较 GPU 峰值 FLOPs。异步流水与反向路径见下图；具体 tile 和指令细节留给原文。

![FlashAttention、Varlen packing 与 DCP 的直观解释](/assets/img/blog/long-context-training/flashattention_varlen_dcp_explainer_gpt_image2.png)

*图：上半部分对照 FlashAttention 的 IO 优化与 FA4 的 Blackwell MMA/TMEM 流水线；下半部分用每篇文档两个示意块说明 varlen 与 DCP。mask 和计算块使用同一组因果依赖，跨文档及未来位置不生成任务；块数不代表文档实际长度。*

![FlashAttention-4 在 Blackwell 上的 kernel pipeline 与瓶颈迁移示意](/assets/img/blog/long-context-training/fa4_kernel_pipeline_gpt_image2.png)

<a id="2-sp-和-cp-不是一回事"></a>

## 3. SP 和 CP 不是一回事

讨论训练并行时，最容易混在一起的是 Sequence Parallelism 和 Context Parallelism。它们都切 sequence 维，但解决的问题不同。

Megatron 语境里的 Sequence Parallelism 通常和 Tensor Parallelism 绑定：TP 负责把线性层权重和 GEMM 切开，SP 则把 LayerNorm、Dropout 等原本在 TP ranks 上冗余保存的非 TP activation 沿序列维切开。它能明显缓解 activation memory，也能减少一部分冗余计算。

但 SP 本身并不等价于“把 attention 的全序列计算分摊到多卡”。full attention 仍然需要让每个 query 看见合法的历史 key/value。长到单卡放不下时，才轮到 Context Parallelism 出场：把一个长 sequence 分给多个 ranks，每个 rank 只持有一段 token，但 attention 语义仍要等价于完整序列。

可以用一张表先钉住差别：

| 并行方式 | 主要切什么 | 主要省什么 | attention 是否需要特殊通信 |
|---|---|---|---|
| TP | hidden/head/MLP 维 | 参数、GEMM、部分 activation | 取决于 attention 实现 |
| SP | 非 TP 区域的 sequence activation | LayerNorm/Dropout 等冗余 activation | 不能单独解决 full attention 长序列 |
| CP | attention 看到的 sequence/context | 长序列 attention activation 与单卡长度 | 必须处理跨 rank 的 KV、softmax 或 recurrent state |
| DP/FSDP/ZeRO | batch 维和训练状态 | 参数、梯度、优化器状态 | 与 CP/SP 互补 |

这个区分很重要。SP、CP、Ulysses、Ring Attention 不是同一层概念。SP 是一类 activation sharding；CP 是长上下文 attention 的语义切分；Ulysses 和 Ring Attention 是 softmax attention 下实现 CP 的两种通信方案。到了 linear attention 或 KDA 这类 recurrent state attention，CP 的通信对象又会变。

还有一个细节常被忽略：CP rank 上的位置不能从 0 重新开始。如果一个 256K sequence 被切成 8 段，第 5 个 rank 的第一个 token 位置应该是全局位置 128K 附近，而不是本地位置 0。前面写 RoPE 实现时已经提到过这个问题；在训练侧，它会和 packed sequence、`cu_seqlens`、block-diagonal mask 一起出现。

<a id="3-softmax-attention-的-cp交换-kv还是重排-head"></a>

## 4. Softmax attention 的 CP：交换 KV，还是重排 head

[Ring Attention](https://arxiv.org/abs/2310.01889) 和 [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) 是 softmax attention 下两条常见的 CP 思路。

Ring Attention 的直觉很朴素。每个 rank 固定持有一段 query 和本地 KV block，然后沿 ring 轮转 KV block。每来一个 KV block，rank 就计算自己 query 对这段 KV 的局部 attention；所有 KV block 轮完以后，当前 rank 得到了自己这段 query 对完整上下文的输出。

这个设计能成立，靠的是 softmax 的在线合并。对每个 query，在各 KV shard 上先求局部最大值 $$m_j$$、相对该最大值的指数和 $$l_j$$，以及已除以该指数和的局部输出 $$o_j$$。全局最大值为：

$$
m = \max_j m_j.
$$

全局归一化项和输出可以按：

$$
l = \sum_j \exp(m_j-m)l_j,
$$

$$
o = \frac{\sum_j \exp(m_j-m)l_j o_j}{l}
$$

来合并。这样就不需要把完整 $$L \times L$$ attention matrix 放到某一张卡上。Ring Attention 的论文把 blockwise attention 和 KV block 通信重叠起来，目标正是让超长序列在设备数增加时继续扩展。

causal mask 会给 Ring 带来一个现实问题：不同 query block 可见的 KV 数不一样。靠前的 query 只看很少历史，靠后的 query 看很多历史；如果按连续片段切，后面的 ranks 工作更重。于是实际系统会考虑 zigzag、striped 或更细的 block 分配，把三角形负载摊平。这个优化不改变 attention 语义，但会显著影响 step time。

Ulysses 走的是另一条路。它先沿 sequence 维切输入，通过 all-to-all 把数据重排到 head 维，使每个 rank 在 attention 计算时拿到一部分 heads 的完整序列。算完 attention 后，再通过 all-to-all 切回 sequence 维。Ulysses 论文强调，当序列长度和设备数按比例增加时，这种 all-to-all 的通信量可以保持相对稳定；工程上，它也比较容易和现有 attention kernel 拼起来。

Ulysses 的限制也来自这里：如果按 head 维重排，CP size 通常要受 head 数整除约束。head 数不多、GQA/MLA/KV head 已经被压缩、或者想把 CP 扩到很大时，这个约束就会冒出来。

![Ring Attention 与 Ulysses 的通信视角对比](/assets/img/blog/long-context-training/ring_vs_ulysses_gpt_image2.png)

*图：Ring 的直觉是“query 留在本地，KV block 轮流送过来”；Ulysses 的直觉是“先把 sequence shard 通过 all-to-all 换成 head shard，算完再换回去”。图为机制示意，不是论文原图。*

所以 softmax CP 不是选一个名字就结束。Ring 更像“query 不动，KV block 轮转”；Ulysses 更像“sequence shard 先换成 head shard”。前者要处理 ring overlap、causal 负载和 block 调度；后者要处理 all-to-all、head divisibility 和并行网格。哪一个更好，取决于模型 head 配置、网络拓扑、kernel 支持和目标 sequence length。

2024 以后的一批工作，基本都在补这两个原型的工程缺口。[Striped Attention](https://arxiv.org/abs/2311.09431)把连续 sequence shard 改成条纹式分布，专门处理 causal attention 的三角形负载不均；论文报告在 256K causal transformer 训练上相对 Ring Attention 有最高 1.45x end-to-end throughput 提升，在 16 个 TPUv4 芯片、786K sequence 上有 1.65x speedup。这个数字不该跨硬件外推，但它说明“怎么切 token”本身就是性能问题。

[DISTFLASHATTN](https://arxiv.org/abs/2310.03294)则沿着 FlashAttention 的思路做分布式 exact attention：token-level workload balancing、KV communication overlap、以及 rematerialization-aware gradient checkpointing 是它的三个核心点。它的评估覆盖 Llama-7B 及其变体、32K 到 512K sequence，并把 Ring、Ulysses、Megatron-LM + FlashAttention 都作为对照。它值得借鉴的地方，是把优化对象从“attention 算子”扩展到了 workload balance、通信隐藏和 checkpoint 策略。

[USP](https://arxiv.org/abs/2405.07719)和 [LoongTrain](https://arxiv.org/abs/2406.18485)可以放在一起看。USP 把 Ulysses 和 Ring 组合成统一的 sequence parallelism，并讨论和 DP/TP/PP/ZeRO 组成 4D parallelism 的实践；LoongTrain 进一步提出 2D-Attention，把 head parallel 和 context parallel 组合起来，用 Double-Ring-Attention、hybrid ZeRO 和 Selective Checkpoint++ 改善扩展性。它们共同说明一件事：当上下文继续拉长，单一维度的 CP 很快会碰到 head 数、网络拓扑或通信重叠的限制，系统需要在 head、context、optimizer state 和 checkpoint 之间一起分账。

<a id="4-linear-attention-不能照搬-softmax-cp"></a>

## 5. Linear attention 不能照搬 softmax CP

第二篇讲过，linear attention 把历史从 KV list 改成 recurrent state。这个变化到了训练系统里，会直接改变 CP 的通信对象。

softmax attention 的局部结果能靠 online softmax 合并，是因为每个 KV shard 对当前 query 的贡献可以先独立算出，再用全局归一化修正。recurrent attention 的形状不同。以抽象形式写，状态更新大致是：

$$
s_t = F_t(s_{t-1}, x_t), \qquad o_t = G_t(s_t, x_t).
$$

如果 $$F_t$$ 只是简单加法，有时还能把各段局部 state 做 prefix-sum 式合并；但一旦出现 token-dependent transition，例如 KDA 里的 delta rule，后一个 segment 的效果就依赖进入该 segment 的初始 state。把每段都从 $$s=0$$ 开始算再相加，已经不够。

对 GDN/KDA 这类更新，固定输入投影产生的读写信号后，可以把一段递推表示成仿射映射。令 $$A_j$$ 为第 $$j$$ 段累计的状态转移算子，$$B_j$$ 为该段从零状态生成的写入，两段合并就是：

$$
S_1=A_0S_0+B_0,\qquad
S_2=A_1S_1+B_1=(A_1A_0)S_0+A_1B_0+B_1.
$$

各设备可以先并行求自己的 $$A_j,B_j$$；跨段组合后，才知道本段真正的 incoming state。段内某位置的输出，也要把 incoming state 经段起点到该位置的累计转移、再由 query 读出的贡献，加到本地零状态输出上。只交换 $$B_j$$ 再相加，就漏掉了 $$A_1B_0$$ 这样的传播。

[Kimi K3](https://arxiv.org/abs/2607.24653)的 KDA CP 利用这类 transition/state 分解。[Ling/Ring 2.6](https://arxiv.org/abs/2606.15079)的 Lightning Attention CP 则利用更简单的衰减结构：先求 local state/output，AllGather 形如 $$[B,H,D,D]$$ 的段状态，再按位置衰减补上跨段贡献；状态通信可与本地输出 kernel 重叠。两者都需要跨段修正，但通信对象与组合代价不能混为一谈。

<a id="41-stateflow把递推边界接进-sequence-pipeline"></a>

### 5.1 StateFlow：把递推边界接进 sequence pipeline

[StateFlow](https://arxiv.org/abs/2608.06838)（2026-08）从流水线调度处理同一个依赖问题。传统 PP 以 microbatch 为主要调度单位；StateFlow 进一步把每条序列分成 chunks，把 chunk 作为前向、反向和 activation 释放的单位。它并不替换 Ring/Ulysses 的 KV 通信算法，而是在保留递推语义的前提下，让 sequence chunks 更早进入流水线。

以某个递推层为例，chunk $$C_i$$ 除了接收上一模型层的 activation，还需要前一个 chunk 留下的边界状态 $$S_i^{\mathrm{in}}$$。前向产生 $$S_i^{\mathrm{out}}$$，供下一个 chunk 使用；反向则必须接收后一个 chunk 传回的状态梯度。短卷积等模块也有跨 chunk 边界状态，不能只传主 memory matrix。

~~~text
同一递推层的前向：S_0 -> C_0 -> S_1 -> C_1 -> S_2 -> C_2 -> S_3
状态梯度的反向： dS_0 <- C_0 <- dS_1 <- C_1 <- dS_2 <- C_2 <- dS_3
~~~

每个 chunk 仍与 PP 的前后模型 stage 交换 activation / gradient。调度器需同时满足模型深度和序列递推两组依赖；满足依赖并完成局部 backward 后，该 chunk 的 activation 才能释放。省显存的来源是缩短张量存活时间，而不是截断跨 chunk 的梯度。

Hybrid 模型还需要非均匀切分。设 chunk $$i$$ 长度为 $$s_i$$，它前面已有 $$p_i$$ 个 token。把固定模型宽度下的线性工作合并为系数 $$a$$，全局 causal softmax 的每对 query/key 成本合并为 $$b$$，可以近似写成：

$$
\operatorname{Cost}_i\approx a s_i+b\left(p_i s_i+\frac{s_i^2}{2}\right).
$$

后一项同时包含跨 chunk 的 prefix attention 和块内 causal attention。假设把序列均分成四块，只考虑 softmax 的二次项，四块工作量大致是 $$1:3:5:7$$。后面的块虽然 token 数相同，却会让流水线空等更久。StateFlow 因而结合 profiling 搜索非均匀边界，让后面的 chunk 更短；在线性层和 softmax 层比例变化时，边界也要重新评估。

系统还对并行度有限的 state-transition kernel 调整 launch grid，并与周围独立计算重叠。论文在最高 32B 模型、256K context 上，相对对应原生 PP 配置报告最高 2.22× 吞吐提升、峰值显存最高降至约 $$1/2.45$$。结果依赖 chunk 数、模型结构和并行配置；切得过细也会增加调度与小 kernel 开销。


<a id="5-显存优化不是只开-zero"></a>

## 6. 显存生命周期：训练状态、activation 与临时峰值

长上下文训练的显存通常由三类东西叠在一起：训练状态、activation、临时 buffer。

ZeRO、FSDP、distributed optimizer 主要处理第一类：参数、梯度和优化器状态。模型越大，这部分越关键。但当 $$L$$ 拉长以后，activation 会变成另一条主线。即使参数已经被很好地 shard 了，单层 attention、MLP/MoE、residual path 和通信 overlap buffer 仍然会把显存顶满。

Activation recompute 控制的是保存与重算的取舍。full activation recompute 可以省显存，但会让反向传播重跑很多前向计算。[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) 之所以强调 sequence parallelism 和 selective activation recomputation，是因为很多 activation 不值得全量保存，也不值得全量重算。长上下文阶段通常更关心“重算哪几块最划算”：attention 的某些中间量、MLP 激活、MoE expert path，和通信 overlap buffer 的生命周期。

一个比较实用的判断是：先把 optimizer state 和参数 shard 做到位，再用 SP 减少非 TP activation 的冗余，用 FlashAttention 降 attention IO，随后用 selective recompute 处理峰值。如果这些还不够，才把 CP 放进训练网格，重新评估每层的 local tokens、通信路径和 overlap。

这里不能只看 samples/s。长上下文阶段 micro-batch 经常降到 1，吞吐要看 tokens/s、tokens/step、MFU 以及 step time 的尾部波动。尤其是 MoE 模型，router 的一次偏斜就可能让某些 ranks 的 expert buffer 暴涨，平均吞吐很好看，稳定性却不够。

<a id="51-压低-attention-峰值后下一处-oom-在哪里"></a>

### 6.1 压低 attention 峰值后，下一处 OOM 在哪里

[Flattening Every Memory Peak in Long-Context Mixture-of-Experts Training](https://arxiv.org/abs/2609.14306)（2026-09）把显存问题拆成四个会先后暴露的峰值。单独降低最大的一项，往往只是让下一项成为限制。

| 峰值来源 | 为什么会随规模增长 | 论文的处理 |
|---|---|---|
| expert dispatch | 路由后的接收 token 数和临时 buffer 可能很大 | PipelinedLLEP 限制每个源 rank 对一个 dispatch chunk 的贡献，约束接收端工作集 |
| 词表投影 / logits | 未分块时要物化 token 数 × 词表大小 | Ring-DTP 分块流转 activation 或词表权重，用 online log-sum-exp 累积精确 loss |
| checkpoint 边界 | 开了 recompute，边界输入仍会跨层长期保存 | SCO 把边界中长期存活的张量 offload 到 CPU，反向前恢复 |
| optimizer state / update | Adam 状态随参数量增长，CPU offload 又可能拖慢 step | OffloadStreamAdamW 将主机状态按 bucket 传入 GPU 更新并回写，流水重叠传输与计算 |

词表投影是个容易漏掉的例子。仅 32K 个 token、200K 词表，完整 BF16 logits 就约为 12.2 GiB，尚未计入其他张量。FlashAttention 不处理这部分开销。Ring-DTP 因此把词表分块，用全局归一化的在线合并得到 loss 和梯度，避免完整 logits 同时驻留。

四种处理共同约束的是“某一时刻有多少数据必须在 GPU 上”，让更多 token 变成更多批次的流式工作，而非更大的单次分配。它们保持原 loss 和梯度的数学语义，也仍需要足够的主机内存与传输带宽。论文的组合实验覆盖 120B、241B、667B，达到 1M context；这是指定系统与 FSDP2 调优基线之间的容量、吞吐比较，不代表任意同规模模型都能获得相同倍数。

实际排查时，应记录峰值发生在哪个算子、哪个 rank、哪些张量当时仍存活。只记录一个整步的 peak memory 数字，很难判断下一次扩长度该改 attention、dispatch、loss，还是 checkpoint 生命周期。


<a id="6-varlen-packing-变成了数据语义问题"></a>

## 7. 从 varlen 语义到动态负载分配

### 7.1 扁平 token buffer 必须保留样本边界

短上下文训练里，packing 常被理解成“把 batch 填满”。到了 128K 以后，它更像数据组织方式。

一个 packed batch 里可能拼了多个文档、代码仓库片段、对话轨迹或检索样本。对于要求样本隔离的训练，attention mask 应保持 block-diagonal causal 语义；若配方允许文档间连续建模，则应明确采用另一种 mask，不能由 packing 偶然决定。position IDs 也要和样本语义一致：有些任务需要每个子序列从 0 开始，有些长文档切片需要保留全局 offset；CP 切分以后，本地 rank 还要知道自己在全局 sequence 里的位置。

这里的 varlen 可以先按最朴素的方式理解：不再把每条样本都 pad 到同一个长度，而是把真实 token 拼成一条扁平 token buffer，再用 `cu_seqlens` 标出每条子序列的起止位置。

```text
doc A:  8K tokens
doc B: 64K tokens
doc C:128K tokens

flat_tokens = [A tokens][B tokens][C tokens]
cu_seqlens  = [0, 8K, 72K, 200K]
mask        = block diagonal causal mask
```

这个格式本身不难，难的是它携带了数据语义。`cu_seqlens` 一错，错误不会总是立刻 OOM 或 NaN。更危险的情况是训练还能跑，但样本之间发生信息泄漏，或者 RoPE/NoPE 之外的位置假设被悄悄破坏。

大 CP size 会放大另一个问题：短子序列太多。每个 rank 拿到的是全局 packed sequence 的一段，本地可能包含大量很短的 fragments。朴素实现会触发很多小 kernel launch，GPU 没忙起来，CPU 先被 launch overhead 卡住。[Ling/Ring 2.6 技术报告](https://arxiv.org/abs/2606.15079)在 256K context 的 varlen 场景下，把 per-sub-sequence correction 融合成 Triton kernel，并报告了约 68% 的 end-to-end speedup。这个数字属于他们报告里的实验条件，不应直接外推；但它提示了一个方向：长上下文训练的数据管线、mask 表示和 kernel fusion 要一起看。

### 7.2 DCP：按实际 attention 依赖放置计算

变长问题也不只发生在 kernel launch 上。静态 CP 配置默认每个 batch 的长度和 attention pattern 足够相似；真实数据里，长短样本、document mask、局部窗口和稀疏块会让不同 ranks 的计算/通信比例不断变化。[DCP](https://arxiv.org/abs/2510.10620)把这个问题称为 input dynamism。它关心的不是新的 ring 形态；更关键的是把“attention 里谁依赖谁”显式表示出来。

DCP 把输入和输出切成 data blocks，把一个 query block 对某个 KV block 的注意力计算切成 computation blocks。mask 里没有依赖的地方，就不生成对应的 computation block。这样一来，问题从“sequence 均匀切几段”变成“这些 data/computation blocks 应该放到哪些设备上”。论文把放置写成 hypergraph partitioning：computation block 带 FLOPs 权重，data block 带 size 权重，hyperedge 表示某个数据块被哪些计算块使用。优化目标也比较直接：减少跨设备通信，同时保持计算和显存平衡。

可以拿上面的 packed batch 做例子。若 query block $$Q_i$$ 只允许看同一篇文档内部的 KV block $$KV_j$$，那么跨文档的 $$C_{ij}$$ 根本不该生成；如果是 sliding window 或 lambda mask，也只生成窗口、sink 或共享问题相关的 $$C_{ij}$$。DCP 的 data block 是“数据放在哪里”，computation block 是“哪一块 Q 要和哪一块 KV 发生 attention”。这两个东西分开以后，planner 才能把长文档、短文档和稀疏 mask 造成的不均匀负载重新摊到多张 GPU 上。

执行时，planner 把放置结果编译成 block 计算、归约、copy 和通信 launch/wait 指令，executor 按依赖运行。论文的 attention micro-benchmark 可达数倍加速，但端到端 causal 训练为 0.94–1.16×、sparse mask 为 1.00–1.46×，部分设置并不加速。收益依赖静态切分究竟浪费多少，不能把算子倍率当作整步收益。

![DCP 把变长样本和 mask pattern 转成 data blocks / compute blocks 再动态放置](/assets/img/blog/long-context-training/dcp_dynamic_cp_gpt_image2.png)

### 7.3 Zeppelin：attention、linear 与网络分别调度

[Zeppelin](https://arxiv.org/abs/2509.21841)看的是另一层。DCP 主要在 CP 内部动态切 attention，Zeppelin 则从 data-parallel 训练里的变长 workload 出发，问一个更系统的问题：同一批样本，对 attention 模块、linear 模块和网络设备的压力并不一样，能不能用一套静态分配同时满足它们？

它给出的答案是分层调度。对 attention，Zeppelin 先按序列长度和通信/计算比，把样本分到 local、intra-node、inter-node 三类队列：短序列尽量本地算，中等序列在节点内切，长序列才跨节点。对网络，它不把 GPU-NIC affinity 当成固定路径，而是用 local dispatch、multi-NIC inter-node transfer、local combine 三步路由，让更多 NIC 参与传输。对 linear module，它又加了 remapping layer：attention 最优的布局不一定让 MLP/linear 模块负载均衡，所以进入 linear 前后要重新排一次。论文报告平均 2.80x speedup，这个数字依赖硬件和长度分布；更值得借鉴的是它的分解方式：变长训练已经超过普通 packing 范围，attention 计算、linear 计算和网络拓扑要同时调度。

![Zeppelin 按 local / intra-node / inter-node 队列调度变长样本，并在 attention 与 linear 布局之间 remap](/assets/img/blog/long-context-training/zeppelin_varlen_scheduler_gpt_image2.png)

因此 packing 不能只问“padding 浪费了多少”。更该问：

| 问题 | 为什么重要 |
|---|---|
| 子序列边界是否可靠 | 防止跨文档、跨对话泄漏 |
| position IDs 是否符合任务 | 防止 CP rank 或 packed 样本重置错位置 |
| block-diagonal mask 是否被 kernel 高效支持 | 防止理论 token 利用率被 launch overhead 吃掉 |
| 长短样本是否混得过碎 | 防止 rank 间和 kernel 间负载不均 |

长上下文不是把所有东西拼成长串就行。拼接本身已经是训练配方的一部分。

## 8. Pipeline、attention disaggregation 与框架调度

### 8.1 序列 chunk 改变流水线的调度单位

[Seq1F1B](https://arxiv.org/abs/2406.03488)把 batch-level unit 继续切成 sequence-level unit，减小 pipeline bubble 和 activation footprint；[InfiniPipe](https://arxiv.org/abs/2509.21275)在 token-level PP 与 batch-level PP 之间做 elastic 调度，专门处理真实数据里的变长分布。含 GDN/KDA 或 state-space 层时，还要检查 §5.1 StateFlow 所处理的边界状态、反向依赖和 hybrid chunk 负载。

### 8.2 CAD/DistCA：将 core attention 拆成可重分配任务

如果 attention 已经成为明显的 straggler，[CAD/DistCA](https://arxiv.org/abs/2510.18121)这种 attention disaggregation 就值得单独看。它先把 transformer layer 拆得更细：QKV projection、O projection、LayerNorm、FFN 这些算 context-independent 或近线性的部分；真正二次增长、又没有训练参数的，是 core attention，也就是 $$\mathrm{softmax}(QK^\top)V$$。当长文档和短文档混在同一个 batch 里，core attention 的 $$l^2$$ 项会让某些 DP rank 或 PP stage 变成 straggler，后面的 pipeline stage 跟着空等。

CAD 的关键观察有两个。第一，core attention 没有参数，临时数据也少，天然适合从原来的 layer 里拆出来单独调度。第二，现代 attention kernel 可以吃 arbitrary-length 的 fused token shards；只要 batch 足够大，切成 CA-tasks 再 rebatch，不一定会把 kernel efficiency 打碎。DistCA 就把这些 CA-tasks 发给 attention servers，由 scheduler 按 FLOPs 重新分配任务，尽量让所有 attention servers 同时结束。

实现上还有两个小心思。一个是 in-place attention server：不是真的另划一批 GPU 只做 attention，否则 core attention compute-heavy 但 memory-light，会浪费显存；它让 GPU 在普通模型计算和 attention server 角色之间 time-share。另一个是 ping-pong overlap：一边计算当前 batch 的 core attention，一边传下一批 CA-task 的输入/输出，尽量把 disaggregation 带来的通信藏起来。论文在最多 512 张 H200、512K context 的设置下报告最高 1.35x end-to-end throughput 提升。这个方案不适合作为第一步优化，因为调度器、跨池通信和容错都会复杂；但当 CP 已经不能很好处理 DP/PP straggler 时，它提供了一条很清楚的系统路线：把二次增长的 core attention 从其余模块里拆出来，单独负载均衡。

![CAD/DistCA 将 core attention 拆成 CA-tasks，由 attention servers 动态 rebatch 并用 ping-pong overlap 隐藏通信](/assets/img/blog/long-context-training/cad_distca_disaggregation_gpt_image2.png)

### 8.3 AutoSP：将并行与 checkpoint 搜索放进编译器

更往框架层看，手写 CP/SP 组合会越来越难维护。[AutoSP](https://arxiv.org/abs/2604.27089)把 sequence parallelism 和 long-context-aware activation checkpointing 放进编译器优化，目标是减少训练库为了长上下文反复改造的成本。这类工作未必替代手调 kernel，但它说明一部分长上下文优化正在进入框架和编译器层，而不只是少数系统团队的定制脚本。

<a id="kv-sharing-pipeline"></a>

### 8.4 KV sharing 会约束 pipeline 层切分

训练系统还必须处理共享组的放置。若 producer 和 consumer 跨 pipeline stage，就需要传输共享的 KV activations；可以把一组放在同一 stage，但这又限制层切分。反向传播时，共享 KV 接收多个 consumer 的梯度，不能无意中 detach，只留下 provider 本层的梯度。

[KV-Pipe](https://arxiv.org/abs/2608.15943) 进一步把共享层的布置作为 pipeline 负载调节手段，先减少瓶颈 stage 的投影工作，再重新评估瓶颈。其收益不随共享层数单调增长：原瓶颈被减轻过头，别的 stage 又会成为瓶颈。主要训练实验覆盖 4K–16K、PP=2/4/8，并未在主实验里同时扫 TP/CP，因此不能直接把数字搬到百万上下文训练。

共享 bank 的生成来源与读取层见[第六篇]({% post_url 2026-10-13-cross-layer-kv-reuse %})。这里需要把 producer/consumer 的通信与梯度依赖计入 stage 切分，而非只按每层参数量均分。

<a id="7-moe-会把长上下文的波动放大"></a>

## 9. MoE 会把长上下文的波动放大

MoE 模型里，长上下文还有一层额外放大器：expert routing。

假设每个 token 选 top-k experts。序列从 8K 到 128K，单个 step 里的 token 数增加很多，router 的统计表面上更平滑；但在 packed 数据、block-diagonal mask、不同任务混合和 CP 切分之后，局部 rank 上的 expert token 分布仍然可能很不均匀。EP 的 all-to-all、expert GEMM、dispatch/combine buffer 都会跟着波动。

CP 和 EP 叠在一起时，问题更难直觉判断。CP 把同一条长序列切到多个 ranks；EP 又把 routed tokens 发到不同 experts；TP/PP/DP 还在旁边切 hidden、layer 和 batch。一个配置在短上下文 pretrain 里很稳，到了长上下文 post-training 不一定还能靠同样的显存余量撑住。

[Ling/Ring 2.6 技术报告](https://arxiv.org/abs/2606.15079)里还有两个很工程化的观察，值得放进这里。第一，long-context MoE 训练会遇到 router load-balancing fluctuation 和 block-diagonal causal mask 造成的 rank 间负载不均。第二，pre-training 可以更激进地吃满 GPU memory，但 long-context post-training 需要更保守的显存策略，为突然的 router imbalance 留余量。

排查时可以直接跟踪一批 tokens：CP 切分后各源 rank 路由到哪些 experts，dispatch 后哪个接收端 buffer 最大，最慢的 expert GEMM 又让哪些 ranks 等待 combine。全局 router loss 较平衡，不保证每个 CP shard 或每次通信都平衡。expert 重布局、节点内重平衡和专用 dispatch/combine 分别处理其中的热点、搬运与等待，相关工作集中列在延伸阅读，不在这里逐项展开。

还有一个不太显眼但很真实的坑：token count 和 index 的整数类型。长上下文加 MoE 后，一些 kernel 里原本默认 fit in int32 的 expert token count 可能溢出。报告里提到他们审计并把关键路径改成 int64。这个问题很小，却很能说明长上下文训练的性质：模型公式没有变，变大的是那些过去被当作安全常数的工程维度。

<a id="8-落地时按内圈到外圈排"></a>

## 10. 落地检查：先定位瓶颈，再扩大并行网格

把这些问题串起来，长上下文训练通常不应该从“开 CP=8”开始。可以按依赖关系排查：先保证单卡、单 rank 内的路径正确高效，再把显存峰值压住，最后才扩到更复杂的并行网格。

第一圈是 kernel 和语义。FlashAttention、fused rotary、varlen attention、MLP/MoE kernel、mask 和 position IDs 都要先对。这里一旦错了，后面的 CP、PP、EP 只会把错误放大；尤其是 packed sequence 和 block-diagonal mask，训练能跑不代表语义没泄漏。

第二圈是 activation 峰值。Optimizer state 和参数 shard 只能解决一部分显存；长上下文阶段真正容易顶满的，往往是 attention、MLP/MoE 激活、通信 overlap buffer 和临时 workspace。SP、selective recompute、FlashAttention/FlashAttention-4 这类 kernel 优化，应该先在这个层面把余量做出来。

第三圈才是 CP 和完整并行网格。CP 加进去以后，TP 会影响 head 数和 Ulysses 约束，EP 会影响 MoE dispatch/combine，PP 会影响 bubble 和 activation 保存，DP/ZeRO/FSDP 又决定训练状态怎么 shard。网络拓扑也会进入问题：ring、all-to-all、all-gather 和 expert dispatch 最好不要全挤在同一条最慢链路上。

最后，监控指标也要跟着变。不同长度 bucket 的 step time、OOM retry、router imbalance、CP 通信 overlap、packing 碎片率，都应该和 tokens/s 一起看。这里可以借鉴 [Long-Context Attention Benchmark](https://arxiv.org/abs/2510.17896) 这类工作的问题设定：同一个 attention/CP 方案，在不同 mask pattern、sequence length 和 distributed scale 下可能表现完全不同。对于 128K、256K、1M 这种级别的训练，系统方案不能只给出“能跑”的配置，还要说明哪条链路在吃时间、哪类样本在制造尾部延迟、哪块 buffer 需要留安全余量。

如果只用一句话收束这篇：长上下文训练的核心，是把“每个 token 看见什么历史”这件事，在 kernel、并行、显存、通信和调度上同时说清楚。

## 延伸阅读

这一篇的材料比较偏系统，不建议从“大而全”的 distributed training 综述开始。更有效的读法，是挑几条最容易落到工程改造的链路往里钻。

- kernel 线先读 FlashAttention-3/4。FA-3 把异步和低精度放到 Hopper 上讨论；FA-4 继续把问题推到 pipeline 与硬件瓶颈迁移。读这两篇时，重点放在 bottleneck 怎样从 HBM I/O 移到 TMA、WGMMA、寄存器和调度。
- softmax CP 线可以把 Ring Attention、Striped Attention、Ulysses、DISTFLASHATTN、USP 和 LoongTrain 放在一起看。它们都在拆 $$L^2$$ attention，只是拆法分别偏 KV 交换、head 维重排、统一 SP/CP 或 head-context 混合。
- 变长调度线读 DCP、Zeppelin、Seq1F1B 和 InfiniPipe。长上下文训练很少是整齐的固定长度矩阵乘，真实 batch 里的长度分布、document packing、pipeline bubble 和 straggler 会直接决定吞吐。
- 递推训练线接着读 StateFlow 与 [E²-TTT](https://arxiv.org/abs/2608.21308)。StateFlow 调整 chunk 调度及 activation 生命周期；E²-TTT 推导 chunk-level fast-weight / momentum 转移，在“梯度于 chunk 起点权重计算”的近似下保留逐 token 更新的时序结构，不能理解成任意 TTT 都能无近似并行化。
- [BASP](https://arxiv.org/abs/2609.03151)利用 microbatch 结构把设备分成较小的序列并行组，减少 all-to-all 的参与范围。这里的 sequence parallelism 是长序列 attention 切分语境，区别于 §3 的 Megatron SP。论文在 A100 集群报告约 1.17–1.31× 训练加速，可作为批结构与通信域共同设计的补充案例。
- 显存线读 *Flattening Every Memory Peak*：将 dispatch、logits、checkpoint 边界和 optimizer update 分别 profiling，再分析组合后的最大工作集。
- 如果 attention 已经成了全局尾部，CAD/DistCA 值得单独读。它把 core attention 从 transformer layer 里拆出来做动态 rebatch，适合用来理解“为什么有时继续加 CP 也不够”。
- MoE 线不必只看 router loss。LAER-MoE、FEPLB、Hybrid-EP 和 NCCL EP 更接近训练系统里的真实痛点：expert 热点、dispatch/combine、Copy Engine、以及 EP 通信怎样和 CP/PP 叠在一起。
- 最后读 AutoSP 这类编译器工作。手写并行策略还能覆盖一部分主流结构，但当 mask、packing、checkpointing 和 sequence parallelism 一起变化时，编译器开始接管一部分搜索空间。

## 参考资料

### Attention kernel 与 softmax CP

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451)
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
- [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
- [DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training](https://arxiv.org/abs/2310.03294)
- [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)
- [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485)

### Linear/recurrent attention CP

- [Ling and Ring 2.6 Technical Report](https://arxiv.org/abs/2606.15079)
- [Kimi K3 Technical Report](https://arxiv.org/abs/2607.24653)
- [StateFlow: Sequence Pipeline Parallelism for Long-Context Modeling with Linear Recurrence](https://arxiv.org/abs/2608.06838)
- [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)

### 显存、packing、pipeline 与 benchmark

- [KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs](https://arxiv.org/abs/2608.15943)

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism](https://arxiv.org/abs/2510.10620)
- [Zeppelin: Balancing Variable-length Workloads in Data Parallel Large Model Training](https://arxiv.org/abs/2509.21841)
- [Efficient Long-context Language Model Training by Core Attention Disaggregation](https://arxiv.org/abs/2510.18121)
- [Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism](https://arxiv.org/abs/2510.17896)
- [Seq1F1B: Efficient Sequence-Level Pipeline Parallelism for Large Language Model Training](https://arxiv.org/abs/2406.03488)
- [InfiniPipe: Elastic Pipeline Parallelism for Efficient Variable-Length Long-Context LLM Training](https://arxiv.org/abs/2509.21275)
- [AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism](https://arxiv.org/abs/2604.27089)
- [BASP: Communication-Efficient Batch-Aware Sequence Parallelism for LLM Training](https://arxiv.org/abs/2609.03151)
- [Flattening Every Memory Peak in Long-Context Mixture-of-Experts Training](https://arxiv.org/abs/2609.14306)

### MoE 与 expert parallel

- [LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training](https://arxiv.org/abs/2602.11686)
- [FEPLB: Exploiting Copy Engines for Nearly Free MoE Load Balancing in Distributed Training](https://arxiv.org/abs/2604.19654)
- [Optimizing Communication for Mixture-of-Experts Training with Hybrid Expert Parallel](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)
- [NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL](https://arxiv.org/abs/2603.13606)
