---
layout: post
title: "MoE我们喜欢你（五）：100B 训练配置、并行策略与显存推演"
date: 2026-07-21
tags: [MoE, Infra, Scaling]
description: "给定 100B MoE 规格后，推演并行策略、显存估算、短上下文与长上下文训练配置迁移。"
---


## 0. 导读：定义“100B MoE”

讨论 100B MoE 训练配置时，不能只给出 EP/TP/PP 的组合。更稳妥的做法是先明确参数构成、active params、并行切分和显存来源。下面定义会反复用到的量：

| 符号 | 含义 |
|---|---|
| `P_total` | 总参数量，例如 100B |
| `P_dense` | 非 routed expert 参数：attention、embedding、norm、router、dense warmup layers 等 |
| `P_routed` | routed expert 总参数 |
| `P_shared` | shared expert 参数，通常 always-on |
| `E` | 每个 MoE layer 的 routed expert 数 |
| `k` | 每 token 激活的 routed expert 数 |
| `rho_e = k / E` | expert activation ratio |
| `P_active` | 每 token active params |
| `L` | 总层数 |
| `L_moe` | MoE 层数 |
| `S` | context length |
| `mbs` | micro-batch size |

一个基本关系是：

| 量 | 粗略公式 |
|---|---|
| 总参数 | `P_total = P_dense + P_shared + P_routed` |
| 每 token active 参数 | `P_active ~= P_dense + P_shared + (k / E) * P_routed` |
| routed expert 稀疏度 | `1 - k / E` |

需要注意：`k/E` 不是 active parameter ratio。attention、embedding、dense warmup、shared expert 都是 always-on，所以 active params 往往明显高于 `P_total * k/E`。


![100B MoE 显存构成](/assets/img/blog/moe-we-like-you/05-100b-moe-memory-breakdown.png)
*图 1：100B MoE 的显存估算可以拆成四类：weights、gradients/optimizer、activations、MoE dispatch buffers。active params 主要影响 FLOPs，但完整 routed weights 和 MoE buffer 仍要按并行策略估算。*

## 1. Ling-flash-2.0：100B total params 怎么算出来

先用 Ling-flash-2.0 做一个更接近真实配置的参数账。它的公开口径是 100B total parameters、6.1B activated parameters，其中 non-embedding activated parameters 是 4.8B。这个例子适合说明：MoE 的 total params 主要来自“所有 routed experts 的参数池”，active params 则只计算每个 token 实际走到的专家。

### 1.1 先看关键配置

Ling-flash-2.0 的 `config.json` 里，和参数计算最相关的是这些字段：

| 字段 | 值 | 说明 |
|---|---:|---|
| `num_hidden_layers` | 32 | 总 Transformer 层数 |
| `hidden_size` | 4096 | hidden dimension |
| `vocab_size` | 157184 | 词表大小 |
| `first_k_dense_replace` | 1 | 前 1 层是 dense MLP |
| `num_experts` | 256 | 每个 MoE layer 的 routed experts 数 |
| `num_experts_per_tok` | 8 | 每个 token 选 8 个 routed experts |
| `num_shared_experts` | 1 | 每个 MoE layer 有 1 个 shared expert |
| `intermediate_size` | 9216 | dense MLP intermediate size |
| `moe_intermediate_size` | 1024 | 每个 routed/shared expert 的 intermediate size |
| `num_attention_heads` / `num_key_value_heads` | 32 / 4 | GQA attention 配置 |
| `head_dim` | 128 | attention head dimension |

所以层结构可以粗略理解为：

| 部分 | 数量 |
|---|---:|
| dense MLP layer | 1 层 |
| MoE layer | 31 层 |
| routed experts per MoE layer | 256 个 |
| active routed experts per token | 8 个 |
| shared experts per MoE layer | 1 个 |

这里的 routed activation ratio 是 `8 / 256 = 1 / 32`。但这不等于全模型 active parameter ratio，因为 attention、embedding、第一层 dense MLP 和 shared expert 都是 always-on。

### 1.2 每个模块先算单元参数量

为了便于手算，先忽略 norm、QK norm、router bias 这类百万级以下的小项，只看主要矩阵。

Ling-flash-2.0 使用 SwiGLU 风格 MLP / expert，可以按三组矩阵估算：gate、up、down。因此一个 FFN 或 expert 的参数量近似是 `3 x hidden_size x intermediate_size`。

| 模块 | 公式 | 参数量 |
|---|---|---:|
| 单个 routed expert | `3 x 4096 x 1024` | 12.58M |
| 单个 shared expert | `3 x 4096 x 1024` | 12.58M |
| 第一层 dense MLP | `3 x 4096 x 9216` | 113.25M |
| 单层 attention | `Q + K + V + O` | 37.75M |
| embedding | `157184 x 4096` | 643.83M |
| LM head | `157184 x 4096` | 643.83M |
| 单层 router | `4096 x 256` | 1.05M |

attention 这里因为是 GQA，`K/V` 的输出维度是 `num_key_value_heads x head_dim = 4 x 128 = 512`，所以单层 attention 近似为：

`4096 x 4096 + 4096 x 512 + 4096 x 512 + 4096 x 4096 = 37.75M`

### 1.3 total params：为什么接近 103B

把上面的单元参数量乘上层数，就能得到 total params 的主要构成：

| 部分 | 计算 | 参数量 |
|---|---|---:|
| embedding + LM head | `2 x 157184 x 4096` | 1.29B |
| all attention | `32 x 37.75M` | 1.21B |
| 第一层 dense MLP | `1 x 113.25M` | 0.11B |
| routed experts | `31 x 256 x 12.58M` | 99.86B |
| shared experts | `31 x 1 x 12.58M` | 0.39B |
| routers | `31 x 4096 x 256` | 0.03B |
| 合计 | 主要矩阵相加 | 102.89B |

这就解释了为什么一个 hidden size 4096、只有 32 层的模型，也能做到 100B total params：**每个 MoE layer 都有 256 个 routed experts，虽然每个 token 只激活 8 个，但 total params 要把 256 个专家全部算进去。**

更具体地看，单个 MoE layer 的 routed expert 参数池就是：

`256 x 12.58M = 3.22B`

31 个 MoE layer 合起来就是：

`31 x 3.22B = 99.86B`

也就是说，Ling-flash-2.0 的 100B total params 几乎都在 routed expert pool 里。attention、embedding、shared experts、dense warmup 和 router 加起来只占几 B。

### 1.4 active params：为什么又只有 6.1B

active params 只看一个 token 前向时实际用到哪些参数。对每个 MoE layer，Ling-flash-2.0 激活：

- 8 个 routed experts；
- 1 个 shared expert；
- 该层 attention 和 norm/router 等 always-on 部分。

因此每个 MoE layer 的 active expert 参数量约为：

`(8 + 1) x 12.58M = 113.25M`

31 个 MoE layer 合起来：

`31 x 113.25M = 3.51B`

再加上所有 attention、第一层 dense MLP 和 router：

| active 部分 | 参数量 |
|---|---:|
| active routed experts | `31 x 8 x 12.58M = 3.12B` |
| active shared experts | `31 x 1 x 12.58M = 0.39B` |
| all attention | 1.21B |
| 第一层 dense MLP | 0.11B |
| routers | 0.03B |
| non-embedding active 合计 | 约 4.86B |
| 加 embedding + LM head | 约 6.15B |

这和官方给出的 6.1B activated parameters / 4.8B non-embedding activated parameters 基本对齐。这里也能看到 small-active MoE 的关键：total params 由 `E=256` 放大，active params 由 `k=8` 控制；shared expert、attention 和 embedding 则构成 active params 的下限。

一个值得注意的小设计是：`8` 个 routed experts 加 `1` 个 shared expert，一共相当于 9 个 1024 intermediate 的 expert path；`9 x 1024 = 9216`，刚好等于 dense MLP 的 `intermediate_size`。这意味着每个 MoE layer 的 active expert FFN 宽度，和第一层 dense MLP 的 FFN 宽度在数量级上是对齐的，但 total expert pool 被 256 个 routed experts 放大到了 100B 级别。

### 1.5 抽象成一个 100B MoE 配置

后文为了推演并行和显存，会继续用一个更抽象的 100B total MoE。它不完全等同于 Ling-flash-2.0，而是方便把 dense/shared/routed 三部分拆开讨论：

| 项 | 假设 |
|---|---:|
| `P_total` | 100B |
| `P_dense` | 16B |
| `P_shared` | 4B |
| `P_routed` | 80B |
| `E` | 64 experts/layer |
| `k` | top-4 |
| `rho_e` | 6.25% |

那么每 token active params 约为：

`P_active ~= 16B + 4B + 80B * 4 / 64 = 25B`

这说明一个容易忽略的问题：即使 routed expert activation ratio 只有 6.25%，active params 也不是 6.25B，而是 25B，因为 dense trunk 和 shared expert 是 always-on。

如果想把 active params 压低，可以调这些旋钮：

| 方法 | 影响 | 风险 |
|---|---|---|
| 增加 `E`，保持 `k` | 降低 routed active ratio | expert 更细，通信/GroupedGEMM 压力上升 |
| 降低 `k` | 降低 active compute | 路由更硬，质量/稳定性风险上升 |
| 减小 `P_dense` | 降低 active params 下限 | attention/通用能力可能受损 |
| 减小 `P_shared` | 降低 always-on compute | routed experts 更容易重复学习通用知识 |
| 使用 LatentMoE | 降低 expert path compute/bytes | 架构和框架复杂度上升 |

## 2. 并行维度怎么分工

100B MoE 训练里，各并行维度的职责不同：

| 并行 | 主要作用 | 对 100B MoE 的建议 |
|---|---|---|
| EP | 切 routed experts | MoE 层优先用 EP，避免 expert 权重全落在每张卡 |
| PP/VPP | 切 layers | 多节点扩展的主力，避免 EP/TP 跨太多节点 |
| DP | 扩 batch，配合 distributed optimizer | 尽量保留足够 DP，稳定 load balance 统计并 shard optimizer states |
| TP | 切 attention/dense matmul | expert MLP 上不宜过度 TP，否则 expert GEMM 会被切得过碎 |
| SP | 配合 TP 降 activation | TP > 1 时通常一起考虑 |
| CP | 长上下文切 sequence | 短上下文阶段通常不启用；长上下文 midtrain 再加入 |
| ETP | expert tensor parallel | 只有 expert 太大或 local expert shard 放不下时再考虑 |

核心原则：

1. MoE expert MLP 优先 EP，而不是 TP。
2. `EP x TP` 尽量放在单节点高速互联域内。
3. 跨节点扩展优先 PP/VPP，而不是盲目扩大跨节点 EP。
4. 短上下文阶段优先稳定 EP、GroupedGEMM 和 load balance。
5. 长上下文阶段再引入 CP，主要解决 attention activation，而不是替代 EP。

## 3. 短上下文配置候选

假设硬件是 80GB 级 GPU，短上下文 `S = 4K-8K`，目标为 pretrain bring-up。

### 3.1 256 GPU 起步配置

| 维度 | 建议起点 | 说明 |
|---|---:|---|
| PP | 8 | 每个 pipeline stage 约承担 1/8 层 |
| EP | 8 | 每个 MoE layer 的 64 experts 分到 8 个 EP ranks，每 rank 约 8 experts |
| TP | 1 | 保留 expert GEMM 大小 |
| DP | 4 | `8 x 8 x 1 x 4 = 256 GPUs` |
| CP | 1 | 短上下文阶段先不加 |
| SP | 关 | TP=1 时不需要 |

这个配置的分工是：PP 切层，EP 切专家，DP 提供 optimizer sharding 和 batch 统计，TP 暂不切 expert。

### 3.2 512 GPU 扩展配置

| 维度 | 建议起点 | 说明 |
|---|---:|---|
| PP | 8 |
| EP | 8 |
| TP | 1 或 2 |
| DP | 8 或 4 |
| CP | 1 |

如果 TP=2，需要检查 expert GEMM 是否被切得太碎；如果 attention/dense 部分显存或 compute 压力较高，可以让 TP=2，并用 Parallel Folding 让 MoE path 仍保持较低 ETP。

### 3.3 128 GPU 起步配置

| 维度 | 起点 | 风险 |
|---|---:|---|
| PP | 8 | pipeline bubble 更敏感 |
| EP | 8 | 每层专家能切开 |
| TP | 1 | dense 部分可能偏重 |
| DP | 2 | optimizer sharding 和 global batch 弱一些 |

128 GPU 可以作为小 batch bring-up 配置，但不适合作为长期训练的推荐起点。DP 太小会限制 load balance 统计、optimizer sharding 和吞吐。

## 4. 长上下文配置候选

当 context 从 4K/8K 扩到 32K/128K，瓶颈会明显转向 activation 和 attention。此时优先引入 CP，尽量保持 MoE 架构和 EP 映射稳定。

| 阶段 | 配置变化 | 目标 |
|---|---|---|
| 4K-8K pretrain | PP=8, EP=8, DP=4, TP=1, CP=1 | 稳定 MoE 路径 |
| 32K midtrain | CP=2 或 4，适当调低 micro-batch | 控制 attention activation |
| 128K continued pretrain | CP=4 或 8，配合 selective recompute | 控制 sequence memory 和 attention 通信 |

长上下文阶段的关键是：CP 主要服务 attention/sequence 侧，MoE expert path 仍然主要依赖 EP。加入 CP 后仍需保持 expert MLP 的本地 GEMM 规模，避免 CP/TP 组合把 expert compute 切得过碎。

## 5. 显存估算：参数、优化器、activation、buffer

### 5.1 参数显存

每张 GPU 上的参数量近似由三部分组成：

| 部分 | 粗略分片方式 |
|---|---|
| dense/attention/embedding | 通常按 PP、TP 分片 |
| routed experts | 通常按 PP、EP、ETP 分片 |
| shared experts | 取决于实现，可按 PP/TP 或 expert path 分片 |

对 1.5 里的抽象 100B 示例，如果使用 `PP=8, EP=8, TP=1`：

| 部分 | 总量 | 每 GPU 近似 |
|---|---:|---:|
| `P_dense` | 16B | `16B / 8 = 2B` |
| `P_shared` | 4B | `4B / 8 = 0.5B` |
| `P_routed` | 80B | `80B / 8 / 8 = 1.25B` |
| 合计 local params | 100B | 约 3.75B |

BF16 权重本身约 `3.75B x 2 bytes = 7.5GB`。这只是权重，不包括梯度、optimizer states、activation 和 buffer。

### 5.2 训练状态显存

Adam 类优化器的粗略字节估算：

| 项 | 常见字节/参数 |
|---|---:|
| BF16 weights | 2 |
| gradients | 2-4 |
| FP32 master weights | 4 |
| Adam m | 4 |
| Adam v | 4 |

不分片时，训练状态可能接近 16 bytes/param 甚至更高。Megatron 的 distributed optimizer / ZeRO-1 类策略会把 optimizer states 按 DP 分片，因此 DP 对显存非常重要。

以 local params 约 3.75B、DP=4 粗算：

| 项 | 估算 |
|---|---:|
| BF16 local weights | 约 7.5GB |
| local gradients | 约 7.5GB-15GB |
| optimizer states，按 DP=4 shard | 约 `3.75B x 12 bytes / 4 = 11.25GB` |
| 小计 | 约 26GB-34GB |

这是一个乐观下界，不是可直接落地的显存承诺。真实峰值还要加上 param/grad bucket、param gather buffer、distributed optimizer 的具体实现、expert data parallel 分组、master weight 是否保留、gradient accumulation、通信 overlap staging、CUDA allocator fragmentation 等。这里也尚未计入 activation 和 MoE buffer。80GB GPU 上能否运行，主要取决于 micro-batch、context、recompute、MoE buffer 和 fragmentation。

### 5.3 Activation 显存

Activation 近似与下面这些量成正比：

| 因素 | 影响 |
|---|---|
| micro-batch size | 线性增加 |
| context length | 近似线性增加，attention 相关项更敏感 |
| hidden size | 线性或平方影响部分 attention kernel |
| layers per PP stage | 线性增加 |
| TP/SP/CP | 会切分一部分 activation |
| recompute | 用额外计算换显存 |
| MoE top-k | 增加 dispatched token copies 和 expert activation |

因此 activation 的实用估算不是一个固定公式，而是一个流程：

1. 用目标 `mbs`、`S`、`PP` 算每 stage token 数。
2. 先不开 full recompute，观察 OOM 或 profiler。
3. 优先开 selective recompute，而不是 full recompute。
4. 长上下文时加 CP，再重新估算 activation。
5. 观察 MoE dispatch buffer 是否成为额外峰值。

### 5.4 MoE dispatch / combine buffer

MoE 额外 buffer 的核心来自 token copies：

| 量 | 粗略关系 |
|---|---|
| local tokens | 近似为 `mbs_local x S / CP`，其中 `mbs_local` 是当前并行 rank 上的 micro-batch |
| routed token copies | `local tokens x k` |
| 每个 token hidden bytes | `hidden_size x dtype_bytes` |
| dispatch/combine buffer | 与 `local tokens x k x hidden_size` 成正比 |

这只是最小 payload 下界。真实实现里还可能有 capacity padding、send/recv staging buffer、permute/unpermute index、combine output buffer、router weights、overlap double buffer，以及 TP/SP 对 hidden 或 sequence 的切分影响。如果 top-k 从 4 提到 8，dispatch/combine payload 也会相应上升。DeepEP/Flex Dispatcher 和 EP-A2A overlap 主要优化的就是这部分开销。

## 6. 如何判断配置是否合理

### 6.1 显存侧

| 症状 | 优先处理 |
|---|---|
| 权重/optimizer 放不下 | 增大 PP/EP/DP sharding，或用 precision-aware optimizer |
| activation 爆 | selective recompute、SP、CP、降低 micro-batch |
| MoE buffer 峰值高 | 降 top-k、改善 balance、检查 dispatcher buffer |
| fragmentation 高 | expandable segments、manual GC、减少动态 shape |
| full recompute 太慢 | 换 selective recompute，只重算 attention/MoE/MLP 关键模块 |

### 6.2 通信侧

| 症状 | 优先处理 |
|---|---|
| EP A2A 时间高 | Flex Dispatcher + DeepEP / HybridEP |
| 跨节点 EP 慢 | 限制 EP x TP 在节点内，跨节点用 PP/VPP |
| device load CV 高 | loss-free balance、device-level balance、node/group routing |
| PP bubble 高 | 调 VPP、micro-batch、pipeline layout |
| CP 通信拖慢 | 检查 CP overlap 和 attention kernel |

### 6.3 计算侧

| 症状 | 优先处理 |
|---|---|
| expert GEMM utilization 低 | GroupedGEMM，降低 expert TP，增大每 expert token 数 |
| router/permute 小 kernel 多 | router fusion、permute fusion |
| GPU kernel gap 大 | CUDA Graph scoped capture、减少 CPU launch overhead |
| FP8 不稳 | router/norm/final projection 保持高精度 |

## 7. 100B MoE 的推荐 bring-up 顺序

| 阶段 | 目标 | 配置倾向 |
|---|---|---|
| 单层原型 | 验证 router、GroupedGEMM、dispatch buffer | 少量 GPU 即可 |
| 小模型等比例 | 验证 loss-free balance、top-k、shared expert | 10B-20B scale |
| 100B 短上下文 | 跑通 PP+EP+DP，CP=1 | alltoall baseline，再 DeepEP |
| 100B 性能优化 | 提升 tokens/s | GroupedGEMM、fusion、DeepEP、overlap |
| 100B 长上下文 | 加 CP 和 selective recompute | 尽量保持 EP 映射不变 |
| 后训练 | freeze expert bias update，监控 route drift | SFT/RL 阶段减小路由扰动 |

## 8. 配置推演流程

设计 100B MoE 训练配置时，可以按下面的顺序推进：

1. **拆分参数**：多少 dense，多少 shared，多少 routed。
2. **确定 active params**：`P_active ~= P_dense + P_shared + (k/E) * P_routed`。
3. **用 EP 切 experts**：让每张卡只持有部分 routed experts。
4. **用 PP 切 layers**：跨节点扩展优先 PP/VPP。
5. **保留 DP**：给 global batch 和 optimizer sharding 留空间。
6. **TP 谨慎使用**：attention 可以使用 TP，expert MLP 不宜切得过碎。
7. **短上下文 CP=1**：优先稳定 MoE 路径。
8. **长上下文引入 CP**：解决 attention/activation memory。
9. **显存不足先评估 recompute/CP/DP shard**：优先调整训练策略和并行切分，再考虑改模型结构。
10. **tokens/s 不足以 profiler 定位**：通信墙对应 DeepEP，计算墙对应 GroupedGEMM，内存墙对应 recompute/fusion。

## 9. 结论

100B MoE 的训练配置没有单一固定答案。更可复用的方法是拆成四类估算：

- 参数估算：total params 如何分到 dense/shared/routed。
- 激活估算：active params 和 context/micro-batch 如何决定每步显存。
- 并行估算：EP、PP、DP、TP、CP 各自解决什么。
- 系统估算：DeepEP 降低 token movement 成本，GroupedGEMM 提高 expert compute 效率，recompute 降 activation，distributed optimizer 降训练状态。

可以归纳为：

**短上下文使用 PP+EP+DP 稳定 MoE 路径；长上下文阶段加入 CP；显存按 weights、grads、optimizer、activation、MoE buffer 估算；性能按 communication wall、compute wall、memory wall 分别定位。**

## 系列导航

上一篇：[第四篇：DeepEP 与 GroupedGEMM，MoE 大规模训练的通信墙和计算墙]({% post_url 2026-07-16-moe-deepep-groupedgemm %})  
下一篇：[第六篇：端侧/本地小 active MoE 的挑战、推理瓶颈与设计思路]({% post_url 2026-07-26-edge-moe-inference-design %})

## 参考资料

- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [Ling-flash-2.0 Model Card](https://huggingface.co/inclusionAI/Ling-flash-2.0)
- [Ling-flash-2.0 `config.json`](https://huggingface.co/inclusionAI/Ling-flash-2.0/blob/main/config.json)
- [Ling Scaling Laws](https://arxiv.org/abs/2507.17702)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- [DeepEP GitHub Repository](https://github.com/deepseek-ai/DeepEP)
