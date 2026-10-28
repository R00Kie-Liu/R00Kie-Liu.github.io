---
layout: post
title: "MoE我们喜欢你（五）：100B 训练配置、并行策略与显存推演"
series: moe-we-like-you
series_order: 5
last_modified_at: 2026-10-26
date: 2026-07-21
tags: [MoE, Infra, Scaling]
description: "给定 100B MoE 规格后，推演并行策略、显存估算、短上下文与长上下文训练配置迁移。"
---


## 0. 参数构成、并行维度与显存口径

讨论 100B MoE 训练配置时，不能只给出 EP/TP/PP 的组合。更稳妥的做法是先明确参数构成、active params、并行切分和显存来源。下面定义会反复用到的量：

| 符号 | 含义 |
|---|---|
| `P_total` | 总参数量，例如 100B |
| `P_dense` | 除 routed/shared experts 外的参数：attention、embedding、norm、router、前置 dense 层等 |
| `P_routed` | routed expert 总参数 |
| `P_shared` | shared expert 参数，通常 always-on |
| `E` | 每个 MoE layer 的 routed expert 数 |
| `k` | 每 token 激活的 routed expert 数 |
| `rho_e = k / E` | expert activation ratio |
| `P_active` | 每 token active params |
| `L` | 总层数 |
| `L_moe` | MoE 层数 |
| `S` | context length |
| `mbs` | 每个数据并行 rank 一次 forward 的序列数 |

按上述不重叠的分类，一个基本关系是：

| 量 | 粗略公式 |
|---|---|
| 总参数 | `P_total = P_dense + P_shared + P_routed` |
| 每 token active 参数 | `P_active ~= P_dense + P_shared + (k / E) * P_routed` |
| routed expert 稀疏度 | `1 - k / E` |

需要注意：`k/E` 不是 active parameter ratio。attention、前置 dense 层、shared expert 等不随 routed top-k 稀疏化。这里沿用模型报告对 embedding / LM head 的参数计数口径；它不等于逐 token 实际读取了整张输入 embedding 表。因此 active parameter ratio 通常高于 `k/E`，也不能直接替代 FLOPs 或权重流量。


## 1. Ling-flash-2.0 的总参数与激活参数计算

先用 Ling-flash-2.0 做一个更接近真实配置的参数账。它的公开口径是 100B total parameters、6.1B activated parameters，其中 non-embedding activated parameters 是 4.8B。这个例子适合说明：MoE 的 total params 主要来自“所有 routed experts 的参数池”，active params 则只计算每个 token 实际走到的专家。

### 1.1 Ling-flash-2.0 的模型配置

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

### 1.2 各模块的参数量公式

为了便于手算，先忽略 norm、QK norm、router bias 等小项，只算主要矩阵。后续乘法用原始整数维度计算，表中的 M/B 值取四舍五入显示。

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

### 1.3 总参数量：逐模块汇总

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

也就是说，Ling-flash-2.0 的 100B total params 几乎都在 routed expert pool 里。attention、embedding、shared experts、前置 dense 层和 router 加起来只占几 B。

### 1.4 激活参数量：每 token 的专家与公共路径

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

### 1.5 并行与显存估算使用的配置假设

后面的并行和显存估算继续沿用这套约 102.89B total / 6.15B active 配置，`E=256`、`k=8`、32 层。按前面的矩阵计数，公共参数（attention、embedding / LM head、前置 dense MLP、router、shared expert）约 3.03B，routed expert 参数约 99.86B。

假设 embedding 与 LM head 不绑权，参数为 BF16，采用 Adam 类 distributed optimizer。下文先讨论 TP=ETP=CP=1 的配置；所有 GB 按十进制字节计算，MiB/GiB 另行标注。

## 2. 并行分组：普通参数与专家参数分别计算

| 符号 | 含义 |
|---|---|
| `W` | GPU 总数 |
| `PP` | pipeline stage 数 |
| `TP` / `ETP` | 普通矩阵 / expert 矩阵的 tensor parallel 大小 |
| `CP` | context parallel 大小 |
| `DP_dense` | 普通参数的数据并行组大小 |
| `DP_expert` | 同一 expert 参数分片的副本数，即 expert-data-parallel 大小 |

Megatron 的 EP 从数据并行维度中划分，不能将普通 `DP` 与 `EP` 当成完全独立的维度重复相乘。对于本篇 TP=ETP=CP=1 的配置：

```text
DP_dense  = W / PP
DP_expert = W / (PP * EP)
```

公共参数在各 EP rank 上复制，routed experts 才按 EP 分片。因此 optimizer states 也要分别按 `DP_dense` 和 `DP_expert` 估算。TP、ETP、CP 不为 1 或使用 Parallel Folding 时，应以框架实际建立的 process groups 为准。

EP 负责分配专家，PP 切层，distributed optimizer 利用各自的 DP 组分片状态。TP/SP 和 CP 可进一步处理普通矩阵与长序列，但会改变 activation、通信和 expert batch 的形状，需要重新计算。

## 3. 8K 上下文：256 卡的 PP、EP 与 DP 分组

以 256 张标称 80GB GPU、每节点 8 卡高速互联为示例，序列长度先取 8K。下面是一套用于推演的映射，不表示它已经达到最优吞吐：

| 维度 | 示例值 | 含义 |
|---|---:|---|
| `PP` | 8 | 32 层均分，每 stage 4 层 |
| `EP` | 8 | 每个 MoE layer 的 256 个 experts 分到 8 个 ranks，每 rank 32 个 |
| `TP` / `ETP` | 1 / 1 | 此轮不切矩阵内部维度 |
| `CP` | 1 | 先计算短上下文 |
| `DP_dense` | 32 | `256 / 8` |
| `DP_expert` | 4 | `256 / (8 * 8)` |

这需要把 EP group 放进节点内，PP 跨节点连接；还应核对框架生成的 rank 映射。因为每个 stage 都有 embedding 或层数差异，显存不能只用全模型参数平均除以 PP。

同样保持 PP=8、EP=8 时，128 卡对应 `DP_dense=16、DP_expert=2`，512 卡对应 `64、8`。卡数改变了 optimizer 分片和数据并行规模，是否值得扩展仍取决于 global batch、pipeline bubble、网络和实际 step time，不能只由参数量判断。

## 4. 参数、梯度与优化器状态的显存占用

### 4.1 各 pipeline stage 的本地参数与权重占用

以下假设首 stage 放 embedding 和第一层 dense MLP，末 stage 放 LM head，每 stage 四层；忽略 norm 等小项。shared expert 放在公共参数中计算。

| stage | 公共 local params | routed local params（已除 EP=8） | BF16 local weights |
|---|---:|---:|---:|
| 首 stage：1 dense + 3 MoE | 0.949B | 1.208B | 4.31GB |
| 中间 stage：4 MoE | 0.206B | 1.611B | 3.63GB |
| 末 stage：4 MoE + LM head | 0.849B | 1.611B | 4.92GB |

例如中间 stage 的 routed 参数是 `4 × 256 × 12.58M / 8 ≈ 1.611B`。这套分配中，末 stage 的参数与训练状态更重；实际 pipeline layout 可以调整，但必须重新核算。

### 4.2 公共参数与专家参数的训练状态分片

此处假设 BF16 权重、未分片的 FP32 梯度，并把 FP32 master weights、Adam m/v 共 12 bytes/param 按 distributed optimizer 分片。不同实现若连梯度也分片或去掉 master weights，需要替换对应项。

设公共 local 参数为 `P_c`，routed local 参数为 `P_e`，则：

```text
weights   = 2 * (P_c + P_e)
gradients = 4 * (P_c + P_e)
optimizer = 12 * P_c / DP_dense + 12 * P_e / DP_expert
```

在 256 卡配置下：

| stage | BF16 weights | FP32 gradients | optimizer（两组分别分片） | 小计 |
|---|---:|---:|---:|---:|
| 首 stage | 4.31GB | 8.63GB | 3.98GB | 16.92GB |
| 中间 stage | 3.63GB | 7.26GB | 4.91GB | 15.81GB |
| 末 stage | 4.92GB | 9.84GB | 5.15GB | 19.91GB |

这些数字尚未包含 activation、通信 staging、参数 gather/bucket、kernel workspace 和 allocator 预留。某些实现会复用表中张量的存储，也可能引入额外副本，应在实际 optimizer 配置下核对。

## 5. Activation 与通信 buffer 的显存占用

### 5.1 单层 forward 的 activation 形状与容量

取 `mbs=1、S=8192、H=4096、I=1024、k=8、CP=1`，activation 用 BF16。假设各 EP rank 输入 token 数相同、负载大致均衡，则一个 rank 接收并展开后的 expert input 约有 `S × k` 行。

| 张量 | 形状 | 容量 |
|---|---|---:|
| 一层原始 hidden | `[8192, 4096]` | 64MiB |
| 展开后的 expert input | `[8192 × 8, 4096]` | 512MiB |
| gate projection 输出 | `[8192 × 8, 1024]` | 128MiB |
| up projection 输出 | `[8192 × 8, 1024]` | 128MiB |

若未融合实现同时保存后三块张量，仅这三项就是每层 768MiB，四个 MoE layer 为 3GiB。实际保存量取决于 activation fusion 和 recompute；还有 attention、SwiGLU 中间值、输出和 autograd 状态，因此 3GiB 不是完整 activation 峰值。

pipeline 中一个 stage 可能同时保留多个 microbatch 的 activation。应把该 stage 同时存活的 microbatch 数、调度方式和 selective recompute 的保存边界一起列出，不能只用“每 stage 四层”乘一次就结束。

### 5.2 通信 staging 与 activation 的复用和计数

以同样的逻辑 token 展开量估算，一块 BF16 expert input 大约 512MiB。若 dispatcher 为 dispatch/combine 各保留独立 send/recv staging，四块相同大小 buffer 对应约 2GiB；双缓冲会进一步增加驻留量。真实布局可能复用 buffer、按目的 rank 去重，或使用低精度传输，必须以实际 backend 为准。

如果接收 buffer 本身就是前面计算的 expert input，就只能记一次。负载偏斜、capacity padding、索引和 router weights 也需要另算；这些因素使不同 rank 的峰值可能高于均衡估计。

### 5.3 1F1B 调度与重算下的 activation 驻留量

现在选定一个可复算的保存策略：PP=8、无 VPP、非交错 1F1B 调度，microbatch 数足以进入稳态；对每个 Transformer block 做 activation checkpoint，保留 block 输入，反向时逐 block 重算。仍使用前面的 BF16、`mbs=1、S=8192、H=4096`。

一份 block 输入为 64MiB。每 stage 四层，因此每个尚未完成反向的 microbatch 保留 `4 × 64 = 256MiB` 的 block 输入。stage 从 0 编号，1F1B 下 stage `i` 最多同时保留 `PP-i` 个 microbatch：

| stage 示例 | 在途 microbatch 上限 | 保存的 block 输入 |
|---|---:|---:|
| 首 stage，i=0 | 8 | 2GiB |
| 中间 stage，i=3 | 5 | 1.25GiB |
| 末 stage，i=7 | 1 | 0.25GiB |

反向重算到一个 MoE block 时，若当前实现同时物化第 5.1 节列出的 expert input、gate 和 up 输出，还要加 768MiB。末 stage 的这部分 activation 预算就是 `256MiB + 768MiB = 1GiB`；它包括保存边界与当前重算块的这三项，没有乘上其它已经释放的层。

再采用第 5.2 节的四块 512MiB staging 假设。其中一块接收 buffer 就是已计入的 expert input，额外 staging 为 `2GiB - 0.5GiB = 1.5GiB`。末 stage 已列出的项目合计：

```text
训练状态                     19.91GB
保存的 block 输入 + 当前三项   1GiB
独占 staging                  1.5GiB
分项合计                     ≈ 22.59GB
```

这是上述保存和 buffer 布局下的分项预算。剩余项目逐项补入：attention 的保存/重算张量、激活反向临时量、pipeline 收发、bucket、kernel workspace 和 allocator 预留。末 stage 还要检查 LM head：若显式物化 BF16 logits `[8192, 157184]`，单块约 2.40GiB；fused/chunked loss 的存储与生命周期会不同。

这样已确定的 activation 和 staging 有具体数值，未知项也有对应张量或实现位置。更换 checkpoint 粒度、调度或 dispatcher 时，重新列同时存活的张量即可；不同阶段不同时存活的缓冲区不重复累加。

## 6. 长上下文：CP 对 DP 分组与显存的影响

从 8K 扩到 32K/128K 时，可以用 CP 分担 sequence activation 和 attention 工作，但 CP 会占用原来的数据并行维度。以 256 卡、PP=8、TP=ETP=1 为例，普通 DP 为：

```text
DP_dense = 256 / (8 * CP)
```

若 expert 侧的映射把 CP ranks 折入专家计算，`EP=8` 时仍可能保持 `DP_expert=4`；若采用其它映射，则必须读取实际 expert process groups。不要把 CP 额外乘到前面的 256 卡配置上，同时假定所有 DP 组都不变。

长上下文的验证顺序是：确定 CP/EP 的映射，重算本地 token 数和 optimizer 分片，检查同时存活的 activation，再比较 selective recompute 和 attention kernel。CP 减少的存储与增加的通信都要进入同一份 step-time 和显存记录。

## 7. 显存估算的实测校验与 OOM 归因

先用单层或小模型验证 dispatcher、GroupedGEMM 和状态分片，再运行目标模型。每个 pipeline stage 至少记录参数/optimizer 驻留、activation 峰值、通信 buffer 和最大 allocated/reserved memory；时间上分别看 pipeline bubble、A2A 和 expert compute。

如果 OOM，先判断超出预算的是哪一项：参数与状态需要调整分组或层分配，activation 需要调整 micro-batch、调度或 recompute，buffer 则要检查偏斜和 backend。若显存充足但吞吐不足，沿第四篇的 profiling 路径处理。

这份算例的作用是让每个数字都能追溯到模型配置、张量形状和 process group。换模型、换 optimizer 或加入 CP 后，从这些定义重新计算，比沿用一个固定的 EP/PP 推荐组合更可靠。

## 系列导航

上一篇：[第四篇：DeepEP 与 GroupedGEMM 如何拆解通信和计算瓶颈]({% post_url 2026-07-16-moe-deepep-groupedgemm %})

下一篇：[第六篇：端侧 Active MoE 的推理瓶颈与设计取舍]({% post_url 2026-07-26-edge-moe-inference-design %})

## 参考资料

- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [Ling-flash-2.0 Model Card](https://huggingface.co/inclusionAI/Ling-flash-2.0)
- [Ling-flash-2.0 `config.json`](https://huggingface.co/inclusionAI/Ling-flash-2.0/blob/main/config.json)
- [Ling Scaling Laws](https://arxiv.org/abs/2507.17702)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- [DeepEP GitHub Repository](https://github.com/deepseek-ai/DeepEP)
- [Megatron Bridge Parallelisms Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html)
