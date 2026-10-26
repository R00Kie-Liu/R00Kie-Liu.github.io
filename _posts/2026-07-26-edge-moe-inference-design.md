---
layout: post
title: "MoE我们喜欢你（六）：端侧 Active MoE 的推理瓶颈与设计取舍"
series: moe-we-like-you
series_order: 6
last_modified_at: 2026-10-26
date: 2026-07-26
tags: [MoE, Infra, Inference]
description: "讨论端侧和本地小 active MoE 的专家访问不规则性、active set、量化一致性、真实设备 latency 和设计 recipe。"
---


## 0. 导读

端侧部署一个小 active MoE，需要同时回答两个问题：完整模型能否常驻设备，每步选中的权重又能多快被读取和计算？前者取决于总权重、量化和 KV cache，后者还取决于 batch、专家组合与 CPU/GPU/NPU 执行路径。

下面先区分目标设备，再用一次 decode 的数据流估算权重访问，最后讨论结构和量化应该怎样验证。

![端侧 MoE 专家访问不规则性](/assets/img/blog/moe-we-like-you/06-edge-moe-irregular-expert-access.png)
*图 1：连续 token 选择的专家集合可能变化，影响权重复用、kernel 调度和尾延迟。固定 top-k 时，专家 id 变化不等于 tensor shape 一定变化。*

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

### 2.3 常驻权重与每步读取量分别估算

相对完整执行同样的专家参数池，稀疏激活可以减少每步计算和权重读取；但完整模型的驻留容量没有按 active ratio 同比缩小，实际读取效率也受不规则访问影响。

首先，为了低延迟推理，runtime 通常希望完整模型权重常驻在 DRAM / unified memory / 显存里。否则每步 decode 都可能需要从 SSD、文件映射或慢速内存换入专家权重，尾延迟会很差。一个 5B total、0.8B active 的 INT4 MoE，理论每 token 只算 0.8B active params，但设备仍要能容纳约 2.5GB 量级的量化权重，再加 KV cache、runtime buffer、系统占用和应用本身。

其次，常驻 DRAM 的 active weights 仍需要读取。假设一次 decode 将下表中的权重各读取一次、没有跨 token 的 cache 复用，INT4 权重的纯读取时间可按 bytes / bandwidth 估算：

| active params | INT4 weight bytes | 按有效带宽 60GB/s 估算的读取时间 |
|---:|---:|---:|
| 0.3B | 约 150MB | 约 2.5ms |
| 0.8B | 约 400MB | 约 6.7ms |
| 1.0B | 约 500MB | 约 8.3ms |

这是上述流式读取假设下的量级估算，还没有算量化元数据、dequant、activation、KV cache、kernel launch、CPU/GPU/NPU 同步和 thermal throttling。片上 cache 命中会减少 DRAM 流量，带宽利用率不足则会延长时间。MobileMoE 的 0.3B-0.9B active 配置也需要与这些设备约束一起理解。

### 2.4 KV cache 不会因为 MoE 自动变小

MoE 主要减少 FFN / expert path 的 active compute，不直接减少 attention KV cache。长上下文下，KV cache 可能比 expert compute 更先成为瓶颈。

这意味着：

- 32K / 128K context 的本地 MoE，内存不只由 active params 决定。
- decode 阶段每步都要读长 KV cache，MoE 的 FFN 节省可能被 attention memory bandwidth 抵消。
- 如果模型还带多模态或长上下文，端侧 profiling 必须同时报 KV cache 和 expert 权重占用。

## 3. 用 active set 解释专家权重复用

active set 是某个窗口内实际访问的专家集合。必须注明窗口与层号：每层 expert id 通常是局部编号，跨层统计时应使用 `(layer, expert_id)`，不能把不同层的 expert 17 当成同一块权重。

| 统计粒度 | 能回答的问题 |
|---|---|
| 单 token、单层的 top-k | 该层即时读取哪些专家 |
| 同一层相邻 decode steps 的集合 | 专家切换有多频繁，是否能复用缓存 |
| 一个请求窗口的 expert union | 缓存容量需要覆盖多大的工作集 |
| 一批并发请求的 expert union | batching 带来多少复用，又扩大了多少工作集 |

例如连续 token 分别选择 `{3,17,42}` 和 `{1,17,60}`，只有 expert 17 被复用。可以按相邻集合的 overlap 记录变化，但还要结合每个 expert 的字节量与真实 cache hit rate。集合相同不保证权重仍在 cache，集合不同也不一定导致 tensor shape 变化。

对比 profiling 后，才适合给瓶颈排序：小 batch decode 可能主要受权重读取限制，prefill 可能受 expert 分组和 GEMM 影响，NPU fallback 较多时同步又可能占主导。设备和 workload 不同，不能预先给出固定的“第一到第四瓶颈”。

## 4. 由设备约束决定结构和量化实验

### 4.1 先验证一个固定 top-k 的基线

第一、二篇已讨论专家宽度、shared expert 和前置 dense 层的模型侧动机。在端侧，需要额外比较它们是否减少真实的权重访问或执行开销。专家切得更细，可能扩大组合空间，也可能增加小 GEMV 和调度次数；shared expert 提供规则的公共路径，同时增加每步固定成本。

因此，先固定总权重或 active compute 其中一个预算，比较一组可运行的配置。不要把 moderate sparsity 简化成通用的专家数上限，也不要仅凭服务器结果判断手机上的最佳 top-k。

### 4.2 压缩和动态结构分别改变什么

跨层 expert sharing 可以减少唯一权重，但要检查不同层是否能共享功能，以及 cache 是否真的复用了这些权重。LatentMoE 压缩 routed path 的维度，需把额外 projection 和精度损失计入结果；若瓶颈已在 attention / KV cache，它未必解决主要问题。

zero-expert 或 adaptive top-k 改变每步计算量，也可能增加执行分支。只有固定 top-k 的基线已经可测量，才能判断动态计算节省的时间是否超过新增调度成本。

### 4.3 量化还要检查路由变化

端侧部署常使用 INT4、INT8 等量化格式；本地 GPU 还可能采用 FP8、MXFP4/MXFP8。GGUF 则是文件格式，需要注明其中使用的具体量化方式。MoE 量化还要额外检查：

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

## 5. 模型证据怎样分层

MobileMoE 提供手机端 scaling、量化和设备 profiling 的研究证据；Liquid LFM2.5、Granite MoE、Gemma 4、gpt-oss 等开放权重模型，则适合检验特定本地 runtime 的部署能力；OLMoE 提供更完整的训练与路由研究入口。它们回答的问题不同。

比较时应同时注明设备、量化格式、context、batch 和模型总参数。一个模型能在 laptop 上运行，不能直接证明它适合手机；开放了权重，也不代表能完整复现训练中的专家分工。

## 6. 一份可比较的评估记录

| 记录项 | 最少应注明什么 |
|---|---|
| 模型与权重 | total / active params、专家数、top-k、shared 大小、量化格式和文件大小 |
| 运行条件 | 设备、runtime/version、batch、prompt/生成长度、context、冷/热启动 |
| 内存 | 权重驻留、KV cache、expert cache、peak allocated/reserved memory |
| 延迟 | TTFT、prefill tokens/s、decode TPOT，及对应的 p50/p95/p99 |
| 持续运行 | energy/token、温度和降频情况，说明测量时长 |
| 专家访问 | 按层统计 active set、churn、cache hit rate、权重换入字节量 |
| 质量与量化 | 下游结果、量化前后 top-k overlap、必要时的 per-expert 误差 |

冷启动与热缓存、短时峰值与持续运行应分开报告。若提出“减少不规则访问”的改动，需要用后两类访问指标解释延迟变化，而不是只给 active params。

top-k 一致率是诊断量。量化后路由有变化，不等于质量一定下降；应同时观察输出质量与延迟，避免为了维持完全相同的专家选择而忽视其它误差。

## 7. 不同设备优先验证什么

| 目标设备 | 优先验证的约束 | 结构或 runtime 实验 |
|---|---|---|
| 手机 SoC | 可用内存、持续功耗、LPDDR 流量、GPU/NPU 算子覆盖 | 固定形状路径、目标格式量化、减少 fallback 与同步；在真机测持续 decode |
| Apple Silicon / laptop / PC | 完整专家池与 KV 是否常驻、量化与 runtime 支持 | 先测现有模型，比较 cache、expert packing 与长上下文；不把结果外推到手机 |
| 本地 GPU / 企业边缘 | 并发、batching、专家工作集与尾延迟 | 比较不同 batch 的专家复用、吞吐和 p99，必要时再做 offload |

参数规模由这些预算反推。前面的 INT4 权重访问算例可以估出量级，具体延迟仍需目标设备实测。

## 8. 一个实用 recipe

如果今天要做一个端侧/本地小 active MoE，可以按这个顺序推进：

| 阶段 | 做什么 | 退出标准 |
|---|---|---|
| 目标定义 | 明确是手机端、laptop、consumer GPU 还是边缘服务 | 设备、内存、延迟和上下文目标明确 |
| dense baseline | 先训练或选一个同 active compute 的 dense baseline | 有质量、latency、memory 基线 |
| MoE baseline | 固定 top-k，选定专家宽度和公共路径配置 | router 稳定，无明显 expert collapse |
| 量化闭环 | 做 INT4/INT8 或目标 runtime 量化，必要时 QAT | top-k 一致率、质量和 latency 过线 |
| runtime profiling | 分 prefill/decode 测 full memory、KV cache、p95/p99 | 找出真实瓶颈是权重、KV、kernel 还是 cache miss |
| 结构增强 | 评估 layer sharing、LatentMoE、zero-expert、expert caching | 只保留真实设备收益明确的改动 |

每次结构增强都回到同一组质量、内存和持续延迟测量。先确认瓶颈，再引入 offloading、QAT、sharing 或 adaptive routing，才能解释收益来自哪里。

## 系列导航

上一篇：[第五篇：100B 训练配置、并行策略与显存推演]({% post_url 2026-07-21-moe-100b-parallel-memory %})

下一篇：[第七篇：RL 训推不一致与路由对齐问题]({% post_url 2026-07-31-moe-rl-train-serving-mismatch %})

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
