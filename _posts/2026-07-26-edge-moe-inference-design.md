---
layout: post
title: "MoE我们喜欢你（六）：端侧 Active MoE 的推理瓶颈与设计取舍"
date: 2026-07-26
tags: [MoE, Infra, Inference]
description: "讨论端侧和本地小 active MoE 的专家访问不规则性、active set、量化一致性、真实设备 latency 和设计 recipe。"
---


## 0. 导读

端侧 MoE 的常见误区是只看“active params 很小”。如果从真实设备看，更核心的问题是：**能不能减少权重访问、不规则访问、小 kernel 和 latency 抖动，同时保持足够好的质量。**

这和数据中心训练 MoE 的目标函数不同。训练侧关心 EP、all-to-all、GroupedGEMM、tokens/s 和显存分片；端侧/本地推理更关心：

- 完整权重是否常驻内存；
- active expert set 是否稳定；
- batch=1 decode 下小 GEMM 是否吃掉收益；
- expert cache / offload 是否带来尾延迟；
- KV cache 是否反而成为主要内存瓶颈；
- 量化后 router、experts 和 shared expert 是否仍稳定；
- 手机 SoC、Apple Silicon、消费级 GPU、企业边缘服务的瓶颈是否完全不同。

所以看一个端侧 MoE claim，只看 total params 和 active params 是不够的。更关键的是 **full weight memory、active params、prefill/decode latency、KV cache、expert cache hit rate、energy、p95/p99 latency**。


![端侧 MoE 专家访问不规则性](/assets/img/blog/moe-we-like-you/06-edge-moe-irregular-expert-access.png)
*图 1：dense/shared path 的权重访问更规则；MoE routed path 虽然 active experts 少，但连续 token 的 active set 会变化，端侧更容易遇到 cache miss、小 GEMM 和 p99 抖动。*

## 1. 先划清范围：手机端、本地边缘、研究基线不是一回事

“端侧 MoE”这个词经常被混用。可以先按四类拆开：

| 类型 | 典型设备 | 代表 | 主要问题 |
|---|---|---|---|
| 严格手机端 | 手机 SoC / NPU / 移动 GPU | Meta MobileMoE | 内存带宽、电量、batch=1、小 kernel、专家访问不规则 |
| 本地个人设备 | Apple Silicon、PC CPU/GPU、consumer GPU | Liquid LFM2.5、Gemma 4、gpt-oss-20b | 完整权重驻留、KV cache、量化格式、runtime 支持 |
| 企业边缘/低资源服务 | 小 GPU、CPU 集群、本地私有部署 | IBM Granite MoE、gpt-oss、Phi/GRIN | 吞吐、延迟、并发、成本、稳定性 |
| 开放研究基线 | 可复现实验环境 | OLMoE | router、专家专门化、训练/后训练行为可分析 |

这四类都可能叫“edge / on-device / local MoE”，但设计重点不同。严格手机端要尽量减少真实设备上的权重访问和调度开销；本地 PC/GPU 更能容忍完整权重驻留；研究基线则更看重可复现性和可观测性。

## 2. 手机 SoC 上，MoE 推理到底在跑什么

如果把手机端 MoE 推理落到硬件上，瓶颈通常不是单一的“算力不够”，而是 CPU / GPU / NPU / DRAM / cache / runtime 之间的配合问题。A18 Pro、Snapdragon 8 Elite 这类旗舰手机 SoC 都会强调 on-device AI，但它们的硬件组织仍然是异构的：CPU 适合控制流，GPU 适合较灵活的并行矩阵计算，NPU/DSP 适合静态、量化、规则形状的算子，权重和 KV cache 主要从统一内存或 LPDDR 里读。

对 MoE 来说，一次 decode step 可以拆成：

| 阶段 | 主要硬件 | MoE 特有问题 |
|---|---|---|
| router / top-k | CPU / GPU / NPU | logits 小，但 top-k、group routing、bias、量化一致性要稳定 |
| expert selection | CPU runtime / GPU kernel | 需要把 token 映射到动态专家集合 |
| expert weight access | DRAM / system cache / on-chip SRAM | 当前 token 访问哪些 experts 由 router 决定，预取和 cache 复用更难 |
| expert compute | GPU / NPU / CPU fallback | batch=1 时是很多小 GEMV / 小 GEMM，难以吃满峰值 TOPS |
| combine | GPU / NPU | routed outputs 要按 top-k weights 加权合并 |
| attention KV read | DRAM / cache / GPU/NPU | MoE 不减少 KV cache，长上下文 decode 仍然读大量 KV |

这里最关键的区别是：dense 模型每层都访问固定权重，runtime 可以把权重布局、kernel 和预取做得很规则；MoE 每层会根据 token 动态选择 experts，active params 变少了，但访问路径变得不规则。

### 2.1 CPU、GPU、NPU 分别卡在哪里

手机端不是“小号服务器 GPU”。不同后端的限制不同：

| 硬件 | 适合做什么 | 跑 MoE 的主要问题 |
|---|---|---|
| CPU | router 控制流、top-k、少量 fallback、调度 | 大矩阵吞吐和能效不够，长时间跑会影响功耗和交互 |
| GPU | 相对灵活的 GEMM/GEMV、Metal/Vulkan/OpenCL kernel | 小 expert kernel 多，launch / dispatch overhead 和 memory bandwidth 很快变成瓶颈 |
| NPU / DSP | INT8/INT4/QAT 后的静态图、规则矩阵乘 | 动态 top-k、scatter/gather、变长 active set、custom op 支持可能导致 fallback |
| DRAM / unified memory | 存完整权重、KV cache、activation buffer | 权重流式读取和 KV cache 读取共享带宽；cache miss 会直接变成 latency |
| on-chip SRAM / cache | 缓存 tile、热点权重、局部 activation | 容量远小于完整 expert pool，active set 抖动时命中率下降 |

NPU 的峰值 TOPS 对 MoE 不是充分条件。只有当 expert path 形状固定、量化格式支持、top-k / gather / combine 能被编译进同一条稳定执行图时，NPU 才容易发挥作用。否则常见路径会变成：router 或部分 custom op 在 CPU/GPU 上跑，expert matmul 在 GPU/NPU 上跑，中间反复同步，p95/p99 latency 被放大。

### 2.2 prefill 和 decode 是两种瓶颈

端侧 MoE 要分开看 prefill 和 decode。

| 阶段 | 特点 | 主要瓶颈 |
|---|---|---|
| prefill | 一次处理 prompt 中很多 token，并行度较高 | expert 分组、token permutation、KV cache 写入、GPU/NPU 是否能做较大 GEMM |
| decode | 每步生成 1 个或少量 token，batch 很小 | 权重读取、小 GEMV、kernel launch、active set churn、KV cache 读取 |

prefill 的 token 多，runtime 还能把同一 expert 的 token 聚起来，形成较大的 batch。decode 往往是 batch=1，每层 top-k 之后只有少量 expert 被访问，计算形状更像多个小 GEMV。对手机助手、聊天、agent 这类场景，decode p95/p99 往往比平均 prefill tokens/s 更重要。

### 2.3 active params 主要省 FLOPs，不自动省访问成本

MoE 的常见判断是：total params 大，active params 小，所以推理便宜。但在端侧部署里，这个判断只对了一半。

首先，为了低延迟推理，runtime 通常希望完整模型权重常驻在 DRAM / unified memory / 显存里。否则每步 decode 都可能需要从 SSD、文件映射或慢速内存换入专家权重，尾延迟会很差。一个 5B total、0.8B active 的 INT4 MoE，理论每 token 只算 0.8B active params，但设备仍要能容纳约 2.5GB 量级的量化权重，再加 KV cache、runtime buffer、系统占用和应用本身。

其次，active weights 即使已经常驻，也不等于访问免费。一个粗略下界是：

| active params | INT4 weight bytes | 如果有效带宽 60GB/s，仅权重读取下界 |
|---:|---:|---:|
| 0.3B | 约 150MB | 约 2.5ms |
| 0.8B | 约 400MB | 约 6.7ms |
| 1.0B | 约 500MB | 约 8.3ms |

这只是理论下界，还没有算 dequant、activation、KV cache、kernel launch、cache miss、CPU/GPU/NPU 同步和 thermal throttling。真实手机上，decode latency 通常会比这个下界高不少。因此 MobileMoE 把 active params 放在 0.3B-0.9B 这一档，并不只是为了 FLOPs，也是为了控制每步权重访问和能耗。

### 2.4 KV cache 不会因为 MoE 自动变小

MoE 主要减少 FFN / expert path 的 active compute，不直接减少 attention KV cache。长上下文下，KV cache 可能比 expert compute 更先成为瓶颈。

这意味着：

- 32K / 128K context 的本地 MoE，内存不只由 active params 决定。
- decode 阶段每步都要读长 KV cache，MoE 的 FFN 节省可能被 attention memory bandwidth 抵消。
- 如果模型还带多模态或长上下文，端侧 profiling 必须同时报 KV cache 和 expert 权重占用。

## 3. 端侧 MoE 的真实瓶颈排序

### 3.1 第一瓶颈：权重访问和 active set churn

MobileMoE 里提到的专家访问不规则性，可以理解为：连续 token 激活的专家集合不断变化，导致权重访问、cache 命中、kernel shape 和执行路径都不稳定。

普通 dense 模型每层都走同一组权重，内存访问和 kernel 形状相对固定。MoE 则不同：

- token A 可能访问 experts 3、17、42；
- token B 可能访问 experts 1、17、60；
- 下一层又是另一组 experts；
- 如果 top-k 或 active experts 动态变化，形状更不稳定。

active set 指一次推理中实际需要访问的专家集合。它可以按不同粒度定义：

| 粒度 | active set 含义 | 为什么重要 |
|---|---|---|
| 单 token 单层 | 当前 token 在当前 MoE 层选中的 top-k experts | 决定这一层即时 compute 和权重访问 |
| 单 token 全模型 | 当前 token 在所有 MoE 层访问过的专家集合 | 决定一次 decode step 的专家覆盖面 |
| 一个请求窗口 | 一个 prompt / generation window 内访问过的专家集合 | 决定 expert cache 是否能复用 |
| 一批请求 | 多个用户请求合并后的专家集合 | 决定 batching 和专家驻留策略 |

手机端通常希望 active set 小且稳定。小 active set 降低权重访问，稳定 active set 提高 cache 命中。问题是：过度限制 active set 会伤害模型质量，尤其是多语、代码、数学和工具调用这类长尾能力。

### 3.2 第二瓶颈：小 expert GEMM / GEMV

数据中心训练可以靠大 batch、GroupedGEMM 和多 token 并行提高利用率；手机端交互式 decode 常常是 batch=1。MoE top-k 会把一个 token 分给多个专家，结果变成很多小 GEMV / 小 GEMM。

这会带来三个问题：

- kernel launch / runtime 调度开销占比高；
- 小矩阵利用率低，理论 FLOPs 节省没有完全变成 latency 节省；
- 不同 token 的专家组合不同，runtime 难以稳定复用执行路径。

因此端侧 MoE 不宜盲目追求极高专家数或很大的 top-k。专家更细能提升模型质量和组合空间，但过细会让端侧执行更碎。MobileMoE 所说的 moderate sparsity，本质上就是在模型质量、权重访问规则性和小 kernel 效率之间折中。

### 3.3 第三瓶颈：NPU 静态图和动态路由的冲突

手机 NPU 通常更喜欢固定 shape、固定算子、固定内存布局和量化后的静态图。MoE 的动态路由会带来：

- top-k expert ids 每 token 变化；
- 每个 expert 的 token 数变化；
- gather / scatter / combine 需要 custom kernel；
- zero-expert / adaptive top-k 会让每层 compute 量变化；
- 不同 runtime 对 MoE op 的支持程度不一样。

如果这些动态部分不能被 NPU delegate 接住，就会发生 CPU/GPU fallback。fallback 本身不一定慢，但 CPU/GPU/NPU 之间同步会打断流水，造成 p95/p99 latency 抖动。严格手机端要优先追求“规则可编译”的 MoE，而不是最复杂的动态 MoE。

### 3.4 第四瓶颈：量化和 routing consistency

端侧部署通常离不开 INT4、INT8、FP8、MXFP4/MXFP8 或 GGUF 量化。但 MoE 量化比 dense 模型更复杂：

- router score 对精度敏感，低精度可能改变 top-k 选择；
- 不同 experts 的激活分布可能不同，统一量化 scale 不一定合适；
- shared expert 是常驻路径，量化误差可能影响所有 token；
- 稀疏专家的校准数据可能不足，长尾 experts 更容易被量化伤害；
- QAT 比 PTQ 更有价值，但训练成本和工程复杂度更高。

MobileMoE 把 QAT 纳入训练闭环的意义就在这里：端侧 MoE 不应把量化当作最后一步压缩，而应该在架构和训练阶段一起设计。MoE 量化比 dense 模型多一个关键指标：量化前后 router 选出的 top-k experts 是否一致。dense 模型的量化误差通常表现为激活值偏移；MoE 的量化误差还可能改变专家选择，一旦 top-k 变了，后续访问的专家权重、cache 命中和输出路径都会变。

建议把量化评估拆成三层：

| 指标 | 为什么重要 |
|---|---|
| router top-k 一致率 | 判断量化是否改变专家选择 |
| per-expert activation / weight error | 判断长尾专家是否被量化伤害 |
| downstream + latency 联合指标 | 判断精度收益和真实设备收益是否同时成立 |

实操上，router / gate、norm、embedding、final projection 等敏感模块应优先保留高精度；专家权重量化时使用 expert-balanced calibration，避免校准集只覆盖热门专家。对端侧模型来说，QAT 的价值不只是恢复 perplexity，还包括让 router 在 INT4/INT8 部署口径下保持稳定。

## 4. 端侧 MoE 的设计思路

### 4.1 moderate sparsity 比极高稀疏度更稳

数据中心大 MoE 可以追求 256、512、896 experts 这种大专家池，再靠 EP 和通信库处理；手机端不一定适合。

端侧更常见的 sweet spot 可能是 moderate sparsity：

- experts 避免多到 cache 和调度失控；
- top-k 避免大到每 token 访问太多权重；
- routed experts 要足够细，但不能细到每个 expert GEMM 太小；
- shared expert 保留通用路径，减少 routed experts 重复学习。

MobileMoE 给出的方向是 moderate sparsity + fine-grained experts + shared experts。这里的重点不是某个固定专家数，而是端侧需要在模型质量和执行规则性之间折中。

### 4.2 shared expert 是端侧稳定器

shared expert 在端侧尤其有意义：

- 它为所有 token 提供稳定公共路径；
- 它减少 routed experts 重复学习通用知识；
- 它可以降低 router 错选对 common tokens 的伤害；
- 它让 routed experts 更专注于长尾能力。

但 shared expert 也是 always-on compute。端侧设计里，shared expert 不能太大，否则小 active MoE 会退化成“dense + 少量专家增强”，latency 收益会变小。

### 4.3 前置 dense 层仍然有价值

很多 MoE 设计会在 MoE layer 前保留 1-2 层 dense 层，这个经验在端侧也成立。底层 token 表示还不稳定时过早路由，容易导致 router collapse、专家负载不稳和通用模式重复学习。

端侧模型更看重稳定和可预测 latency，前置 dense 层可以让 router 看到更稳定的表示，同时避免每一层都发生专家调度。

### 4.4 layer/expert sharing 适合端侧，但要谨慎验证

Layer-shared MoE 或 cross-layer expert sharing 的动机很适合端侧：多个层共享部分 expert pool，减少总权重、提升专家复用率、降低存储压力。

潜在收益：

- 降低完整权重内存；
- 提高 expert cache 复用；
- 减少不同层重复学习类似专家；
- 更适合低显存或端侧 runtime。

主要风险：

- 不同层 hidden distribution 不同，共享专家难以同时适配；
- 共享过强会降低层间功能分化；
- 同一个专家跨层复用会让 router / adapter 设计更复杂；
- runtime 需要处理跨层专家缓存和调度。

layer/expert sharing 更适合放在端侧 MoE 的“值得探索但需要实测”方向，而不是默认主线。相比之下，moderate sparsity、shared expert、dense warmup、QAT 的模型级证据更强。

### 4.5 LatentMoE 对端侧值得评估

LatentMoE 把 expert path 放到低维 latent space，会同时减少：

- expert compute；
- expert parameter bytes；
- dispatch payload；
- expert cache pressure；
- memory bandwidth。

这些正好是端侧 MoE 的痛点。因此，LatentMoE 不只是大模型训练的通信优化，也可以看作端侧 MoE 的结构压缩方案。

但它是否适合小模型，要看两个条件：

- latent projection 的额外开销是否小于节省的 expert path 成本；
- runtime 是否能高效支持 latent expert 的执行和量化。

如果端侧瓶颈主要是 full weight memory 和 expert cache，LatentMoE 值得评估；如果模型已经较小，瓶颈主要是 attention / KV cache，收益可能没有预期中大。

### 4.6 Zero-expert / adaptive compute 可以做，但不应先上

Zero-computation expert 或 token-adaptive top-k 很适合端侧愿景：简单 token 少算，困难 token 多算。

端侧 runtime 对动态 shape 通常不友好。zero-expert 会让每层真实 compute 变化，可能进一步放大执行路径不稳定。它更适合作为第二阶段能力：

- 先做固定 top-k 的稳定 MoE；
- 再引入 zero-expert，并约束平均 active budget；
- 对数学、代码、工具调用等 hard tokens 单独检查 zero rate；
- 保持 shape 尽量静态，避免系统收益被调度开销吃掉。

## 5. 模型案例怎么读

下面这些模型都值得放进端侧/本地 MoE 讨论，但读法不同。

| 模型 | 适合回答的问题 | 不适合过度解读的地方 |
|---|---|---|
| Meta MobileMoE | 严格手机端 MoE scaling law、QAT、真实设备 profiling | 不应默认当作 open-weight 生态样本 |
| Liquid LFM2.5-8B-A1B | 厂商开放权重、本地/边缘小 active MoE、tool calling | 更偏 laptop / consumer hardware / edge，不等同手机 SoC 常驻 |
| IBM Granite MoE | 企业本地、低资源、开放许可、低 active 部署 | 不是 frontier 能力模型 |
| Gemma 4 26B-A4B | 本地多模态 MoE、active params vs full weight memory | 不能只看 4B active 推断显存 |
| gpt-oss-20b | 本地 reasoning MoE、低延迟 specialized use cases | 不等同手机端 MoE scaling law |
| OLMoE | 完整开放研究基线，适合看 router / expert 行为 | 不是端侧硬件专门优化 |
| Phi-3.5-MoE / GRIN-MoE | 小 active MoE、routing-gradient 类训练思路 | 更适合作为对照，不是严格手机端模型 |

这张表的重点是证据分层。端侧 MoE 的研究不能只堆 model names，而要问：这个模型提供的是手机端实测证据、开放权重部署证据，还是训练/路由可复现实验证据？

## 6. 评估口径：端侧 MoE 应该报什么

### 6.1 模型指标

| 指标 | 为什么要报 |
|---|---|
| total params | 决定存储、下载、完整权重内存 |
| active params | 决定每 token 理论 compute |
| expert count / top-k | 决定 active set 和调度复杂度 |
| shared expert size | 决定 always-on compute |
| MoE layer count | 决定每次推理发生多少次专家调度 |
| dense warmup layers | 影响 router 稳定性和底层通信/调度 |
| context length | 决定 KV cache 上限 |

### 6.2 内存指标

| 指标 | 为什么要报 |
|---|---|
| full weight memory | 判断模型是否能常驻设备 |
| quantized weight memory | 判断真实部署占用 |
| KV cache memory | 长上下文下可能是主瓶颈 |
| expert cache size | 判断专家换入换出频率 |
| dispatch/combine buffer | 判断 MoE 额外中间内存 |
| peak memory | 比静态权重更接近真实 OOM 风险 |

### 6.3 延迟指标

| 指标 | 为什么要报 |
|---|---|
| TTFT | 用户感知首 token 延迟 |
| prefill tokens/s | 长 prompt / agent trace 的输入处理能力 |
| decode tokens/s | 交互式生成速度 |
| p50 / p95 / p99 latency | 判断专家 cache miss 和动态 routing 抖动 |
| energy per token | 手机端和移动设备尤其关键 |
| cold start latency | 权重加载、专家缓存初始化成本 |

### 6.4 路由指标

| 指标 | 为什么要报 |
|---|---|
| expert load histogram | 看 router 是否 collapse |
| active set size | 看每个请求访问多少专家 |
| active set churn | 看连续 token 的专家集合变化有多快 |
| expert cache hit rate | 判断专家驻留策略是否有效 |
| per-domain routing | 看代码、数学、多语是否依赖不同专家 |
| quantization 前后 top-k 一致率 | 判断量化是否破坏路由 |

如果只能选一个最小评估集，建议至少包括：full weight memory、KV cache memory、TTFT、prefill tokens/s、decode tokens/s、p95 latency、energy/token、active set churn、expert cache hit rate。

## 7. 分设备的设计建议

### 7.1 手机 SoC / NPU

手机端最严苛，核心约束是 DRAM 带宽、可用内存、NPU/GPU 支持的算子集合、以及持续功耗。这里要分两条路径看：

| 路径 | 适合条件 | 主要风险 |
|---|---|---|
| GPU 路径 | MoE op 动态、runtime 需要灵活 kernel、模型还在快速迭代 | 小 kernel 多，权重读取和 kernel launch 成本高 |
| NPU 路径 | QAT/INT8/INT4 完整闭环，shape 静态，top-k/combine 可编译 | 动态 routing、custom gather/scatter、fallback 同步导致尾延迟 |

建议：

- active params 控制在 1B 以下或接近 1B；
- moderate sparsity，不追求极高 expert count；
- top-k 尽量小，优先 top-1/top-2/top-4 的实测对比；
- 1 个较小 shared expert；
- 前置 1-2 层 dense；
- QAT 优先于纯 PTQ；
- router / norm / embedding 等敏感模块优先保留更高精度；
- expert weights 要按端侧 runtime 的 tile / cache 友好格式重新打包；
- 如果目标是 NPU，尽量避免 token-adaptive top-k 和过多动态 shape；
- 优先优化 decode p95/p99，而不是平均 FLOPs；
- 真实设备 profiling 必须进主表。

### 7.2 Apple Silicon / laptop / PC

本地个人设备可以容忍更大的 full weight memory，Apple Silicon / laptop 也有更高的散热和内存预算，所以能跑的 MoE 规模会比手机大很多。但这类结果不能直接外推到手机：laptop 可以靠更大的 unified memory、更高持续功耗和更成熟的 runtime 承担专家池，手机端更容易被电量、温度和后台系统占用限制。

建议：

- 可以评估 8B-30B total、1B-4B active 的 MoE；
- 关注 MLX、llama.cpp、ONNX、vLLM/SGLang 等 runtime 支持；
- 同时看 active params、完整权重和 KV cache；
- 如果模型面向工具调用或长上下文，单独测 prefill 和 long-context memory；
- 量化格式要和目标 runtime 一起选；
- 如果在 Apple Silicon 上验证端侧 MoE，至少要另外在真机手机 SoC 上测一轮 decode p95/p99 和能耗。

### 7.3 本地 GPU / 企业边缘服务

这类场景可以用更大模型，但要关注并发和尾延迟。建议：

- 允许 20B-30B+ total、3B-10B active；
- 如果 batch 能做大，MoE 的小 GEMM 问题会缓和；
- 多请求 batching 可以提高专家复用，但会增加调度复杂度；
- 需要监控 expert load、active set overlap、p95/p99 latency；
- 对低延迟服务，expert cache 策略比平均 tokens/s 更重要。

## 8. 一个实用 recipe

如果今天要做一个端侧/本地小 active MoE，可以按这个顺序推进：

| 阶段 | 做什么 | 退出标准 |
|---|---|---|
| 目标定义 | 明确是手机端、laptop、consumer GPU 还是边缘服务 | 设备、内存、延迟和上下文目标明确 |
| dense baseline | 先训练或选一个同 active compute 的 dense baseline | 有质量、latency、memory 基线 |
| MoE baseline | 固定 top-k、moderate sparsity、1 shared expert、1-2 dense warmup layers | router 稳定，无明显 expert collapse |
| 量化闭环 | 做 INT4/INT8 或目标 runtime 量化，必要时 QAT | top-k 一致率、质量和 latency 过线 |
| runtime profiling | 分 prefill/decode 测 full memory、KV cache、p95/p99 | 找出真实瓶颈是权重、KV、kernel 还是 cache miss |
| 结构增强 | 评估 layer sharing、LatentMoE、zero-expert、expert caching | 只保留真实设备收益明确的改动 |

这个顺序的目的，是先让固定形状 MoE 稳定，再做动态和压缩增强。端侧系统噪声较大，一开始就同时上 adaptive routing、offloading、QAT 和 layer sharing，难以判断收益来源。

## 9. 常见风险

- 只报 active params，不报 full weight memory。
- 用 FLOPs 代替真实设备 latency。
- 把 laptop / consumer GPU 结果说成手机端结果。
- 只看平均 decode tokens/s，不看 p95/p99。
- 忽略 KV cache，尤其是长上下文模型。
- 专家数开得太多，导致 active set churn 和 cache miss 失控。
- 量化只测 perplexity，不测 router top-k 一致率。
- 没有 dense baseline，无法判断 MoE 的成本收益是否成立。
- 用开放权重等同于可复现实验，忽略数据、训练日志和 router 统计缺失。

## 10. 结论

端侧/本地 MoE 的关键问题不是“能不能把 active params 做小”，而是能不能让小 active compute 在真实设备上转化为稳定的低延迟、低内存和低功耗。

从预训练和部署一起看，比较稳的判断是：

1. 严格手机端优先学习 MobileMoE 的方法论：moderate sparsity、fine-grained experts、shared experts、QAT、真实设备 profiling。
2. 本地/边缘开放权重模型要重点看 full weight memory、KV cache、runtime 和量化，而不是只看 active params。
3. 端侧 MoE 的下一步很可能来自结构和系统一起设计：shared expert + dense warmup + QAT 是底座，LatentMoE / layer sharing / expert caching / zero-expert 是后续增强。

可以概括为：

**端侧 MoE 不是把大 MoE 缩小，而是把“每 token 少算”落实为设备上的低权重访问、低等待和低抖动。**

## 系列导航

上一篇：[第五篇：100B MoE 训练配置推演：并行策略与显存估算]({% post_url 2026-07-21-moe-100b-parallel-memory %})  
下一篇：[第七篇：为什么 MoE RL 容易出现训推不一致]({% post_url 2026-07-31-moe-rl-train-serving-mismatch %})

## 参考资料

- [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/abs/2605.27358)
- [Apple Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [Apple iPhone 16 Pro / A18 Pro Press Release](https://www.apple.com/newsroom/2024/09/apple-debuts-iphone-16-pro-and-iphone-16-pro-max/)
- [Qualcomm Snapdragon 8 Elite Mobile Platform](https://www.qualcomm.com/smartphones/products/8-series/snapdragon-8-elite-mobile-platform)
- [Android Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/)
- [ONNX Runtime NNAPI Execution Provider](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html)
- [Liquid LFM2.5-8B-A1B Documentation](https://docs.liquid.ai/lfm/models/lfm25-8b-a1b)
- [IBM Granite 3.1 3B-A800M Base](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-base)
- [gpt-oss-20b Model Card](https://openai.com/index/gpt-oss-model-card/)
- [Gemma 4 Model Overview](https://ai.google.dev/gemma/docs/core)
- [OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
- [Phi-3.5-MoE-instruct](https://ai.azure.com/catalog/models/Phi-3.5-MoE-instruct)
- [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2306.06189)
- [ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Expert Caching and Token Scheduling](https://arxiv.org/abs/2410.17954)
