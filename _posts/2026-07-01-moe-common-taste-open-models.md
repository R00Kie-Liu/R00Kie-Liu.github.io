---
layout: post
title: "MoE我们喜欢你（一）：开源模型里的共识设计"
series: moe-we-like-you
series_order: 1
last_modified_at: 2026-10-26
date: 2026-07-01
tags: [MoE, Pretraining, LLM]
description: "从最新开放模型出发，整理大规模 MoE 已经趋于稳定的架构设计实践，以及 LatentMoE、zero expert、shared-expert sink 等仍偏前沿的探索。"
---


## 0. 写在前面

DeepSeekMoE / V3 之后，大规模开放 MoE 的设计开始出现一些重复选择：细粒度专家、shared expert、负载均衡控制，以及为专家计算和通信配套的系统实现。但不同模型并没有收敛到唯一方案。shared expert 是否保留、router 用什么打分、哪些层使用 MoE，都仍有差异。

这篇先整理这些选择的证据：哪些在多个模型中反复出现，哪些已有大模型验证，哪些仍值得当作实验方向。具体如何给预训练项目定配置，留到第二篇。

## 1. 系列路线

前七篇沿着架构、训练实现和部署展开；后三篇是架构研究、EP 系统和视觉生成专题，可以按问题单独阅读。

| 篇章 | 主题 | 重点问题 |
|---|---|---|
| 1 | [共识设计]({% post_url 2026-07-01-moe-common-taste-open-models %}) | 开放模型反复采用了什么，证据有多强 |
| 2 | [架构与训练策略]({% post_url 2026-07-06-moe-scaling-design-training %}) | 在计算、参数和通信预算下怎样选择配置 |
| 3 | [Megatron Core 源码]({% post_url 2026-07-11-megatron-core-moe-source %}) | router、负载控制、dispatcher 和 experts 如何衔接 |
| 4 | [DeepEP 与 GroupedGEMM]({% post_url 2026-07-16-moe-deepep-groupedgemm %}) | 怎样从 profiling 判断通信、计算和重排瓶颈 |
| 5 | [100B 配置推演]({% post_url 2026-07-21-moe-100b-parallel-memory %}) | 把参数、并行分组和显存放在同一个算例中 |
| 6 | [端侧推理]({% post_url 2026-07-26-edge-moe-inference-design %}) | 设备如何执行小 active MoE，应该测什么 |
| 7 | [RL 训推一致性]({% post_url 2026-07-31-moe-rl-train-serving-mismatch %}) | 怎样区分后端偏差、路由变化和策略更新 |
| 8 | [MoE 的架构扩展]({% post_url 2026-07-23-moe-future-sparse-architecture %}) | 稀疏计算如何扩展到深度、attention、memory 和参数共享 |
| 9 | [EP 通信与调度]({% post_url 2026-07-23-moe-ep-communication-optimization %}) | 怎样选择通信路径，何时需要专家副本和物理负载均衡 |
| 10 | [视觉生成]({% post_url 2026-07-23-moe-visual-generation %}) | timestep、token、head 和控制接口分别怎样组织专家 |

## 2. 区分常见设计、候选结构与探索方案

讨论 MoE 时，一个常见误区是把“某个新模型用了”直接等同于“应该默认采用”。这在 MoE 里尤其危险，因为 MoE 的收益很依赖系统、并行拓扑、训练数据规模和推理 runtime。

更稳的分类方式是：

| 类型 | 判断标准 | 例子 |
|---|---|---|
| 常见设计 | 在多个模型中反复采用，且有对应的训练或系统经验 | shared expert、fine-grained experts、前置 dense 层、loss-free balance、router FP32、dropless 优先 |
| 候选结构 | 有清晰动机和大模型证据，但实现/训练/部署成本更高 | LatentMoE、group/node-limited routing、device-level balance、FP8/MXFP8 expert compute |
| 探索方案 | 少数模型报告中效果突出，但尚未成为通用 recipe | LongCat zero-computation expert、Inkling shared-expert sink、Kimi K3 Stable LatentMoE、quantile/PID controller 变体 |

这个分类用于确定实验优先级。预训练阶段同时修改 router、balance、专家形状和 dispatcher，会显著增加问题归因难度。

## 3. 最新开放模型里的 MoE 地图

下表按“对架构选择的启发”整理。模型名旁标出主要来源及材料类型。技术报告可以追踪训练和消融，官方博客、model card 或 API 文档则按其实际披露的范围使用。

| 模型 | 参数/激活 | MoE 设计信号 | 对架构选择的启发 |
|---|---:|---|---|
| DeepSeek-V3 · [报告](https://arxiv.org/abs/2412.19437) | 671B / 37B active | DeepSeekMoE、前 3 层 dense、256 routed experts、top-8、shared expert、aux-loss-free balance | loss-free balance 和 shared expert 成为大规模 MoE baseline 的强信号 |
| Qwen3 · [报告](https://arxiv.org/abs/2505.09388) | 235B-A22B、30B-A3B | dense + MoE 同步发布，MoE 进入通用开放生态 | MoE 不只服务 500B+ 模型，小 active 版本也值得系统化 |
| Kimi K2 · [报告](https://arxiv.org/abs/2507.20534) | 1T / 32B active | 大总参、小 active、agentic data synthesis、MuonClip、joint RL | MoE 和 agentic 后训练开始深度绑定 |
| Kimi K3 · [官方博客](https://www.kimi.com/blog/kimi-k3) | 2.8T，16/896 experts | Stable LatentMoE、Kimi Delta Attention、Attention Residuals | 极高专家数需要 latent expert 和更强稳定性设计配合；完整训练细节仍要继续核验 |
| GLM-4.5 · [报告](https://arxiv.org/abs/2508.06471) | 355B / 32B active | loss-free balance routing、sigmoid gate、agentic/reasoning/coding 统一 | sigmoid + bias balance 已经不再是小众选择 |
| MiniMax-M2 · [报告](https://arxiv.org/abs/2605.26494) | 229.9B / 9.8B active | 256 fine-grained experts、很小 active budget、agentic deployment | 10B active 级别也能承载 agent 能力，说明 active budget 可以继续压 |
| LongCat-Flash · [报告](https://arxiv.org/abs/2509.01322) | 560B / dynamic active | zero-computation experts、PID expert bias、shortcut-connected MoE | token-dependent compute 有吸引力，但不适合作为初始 baseline |
| Nemotron 3 Ultra · [官方研究页](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/) | 550B / 55B active | Hybrid Mamba-Attention、LatentMoE、NVFP4、MTP | 架构、低精度和系统吞吐一起 co-design |
| Inkling · [发布说明](https://thinkingmachines.ai/news/introducing-inkling/) | 975B / 41B active | 256 routed experts、2 shared experts、top-6、sigmoid router、aux-loss-free bias、shared-expert sink | shared expert 开始从 always-on path 走向更细的 mixture 设计 |
| Meta MobileMoE · [论文](https://arxiv.org/abs/2605.27358) | 1.3B-5.3B total / 0.3B-0.9B active | on-device scaling law、moderate sparsity、fine-grained + shared experts、QAT、真实手机 profiling | 端侧 MoE 不能只看 active params，必须看真实设备上的访问规则性和 latency |
| Liquid LFM2.5-8B-A1B · [模型文档](https://docs.liquid.ai/lfm/models/lfm25-8b-a1b) | 8B / 1.5B active | on-device / edge MoE、128K context、tool calling | 本地/边缘 MoE 已经有开放权重样本，但不等同手机 SoC 上的严格端侧 |
| IBM Granite MoE · [model card](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-base) | 1B-A400M、3B-A800M | fine-grained experts、dropless routing、128K context、Apache 2.0 | 企业本地/低资源 MoE 的实用基线 |
| OLMoE-1B-7B · [项目与训练材料](https://github.com/allenai/OLMoE) | 约 7B / 1B-1.3B active | 64 experts、top-8、代码/数据/logs 开放 | 适合做可复现路由分析，而不是追 frontier 能力 |

这张表的重点不在“谁的 total params 最大”，而在几个重复出现的设计：

- active params 被压到 0.3B、1B、3B、10B、22B、32B、55B 等多个档位；
- shared expert 反复出现；
- fine-grained routed experts 反复出现；
- loss-free / bias-based balance 反复出现；
- sigmoid routing 和 top-k normalization 出现频率上升；
- 前置 dense 层仍然被很多大 MoE 保留；
- 推理侧的 communication、memory bandwidth、专家访问规则性越来越像一等约束。

这些重复出现的选择构成了后文讨论的起点。

## 4. 常见设计 一：前 1-3 层 dense，别太早开始路由

很多 MoE 设计会在底部保留 1-2 层，甚至 3 层 dense MLP，再进入 MoE layer。这个设计不显眼，但很实用。

一种设计解释是，底层表示仍偏局部，较早引入路由可能让通用模式分散到不同专家，同时增加底层调度开销：

- 通用 pattern 被拆到多个专家里重复学习；
- 早期 layer 的 load 统计噪声大，router 更容易 collapse 或形成脆弱分工。

前置 dense 层让所有 token 共享底层变换，也减少了这些层的专家调度。这里列出的是结构与设计动机；稳定性收益由逐层负载、质量和相同预算的架构消融判断。

## 5. 常见设计 二：fine-grained routed experts + shared expert

DeepSeekMoE 之后，fine-grained experts 和 shared expert 逐渐成为大规模 MoE 的常见起点。

fine-grained experts 扩大了给定 active compute 下可选择的专家组合。DeepSeekMoE 将专家拆细，再同时激活多个小专家；其设计动机是减少冗余、提高专门化。专家最终按什么分工，仍需通过路由分析和消融确认。

shared expert 为所有 token 提供公共计算路径，设计动机是承接通用变换，减少 routed experts 的重复学习。它与 fine-grained experts 是两个可分别消融的选择，并非所有 MoE 都需要同时采用。

这组设计的代价也要记住：

| 设计 | 结构与设计意图 | 代价 |
|---|---|---|
| fine-grained experts | 更细组合与更小的单个 expert，目标是提高专门化 | router/load balance 更敏感，dispatch 更碎 |
| shared expert | 所有 token 共用一条计算路径，用于承接通用变换 | always-on compute，过大时会削弱 sparse 收益 |

shared expert 的宽度直接占用 active compute。比较不同 shared 大小时，应固定总计算预算，检查公共路径带来的质量收益是否抵得上减少的 routed capacity。

## 6. 常见设计 三：loss-free / bias-based balance 变成主线

MoE load balance 的目标存在张力：系统希望每个专家、每个 EP rank 的负载接近均衡；模型又希望 router 有自由度，让专家保持专门化。

早期 auxiliary loss 的做法直接：把负载均衡作为额外 loss 加到 LM objective 上。但大规模训练里，这个 loss 的系数、schedule、统计粒度和 capacity factor 都会变成隐性超参。系数太小，专家过载；系数太大，router 为了均衡牺牲主目标，专家变得平均而不专门。

loss-free / bias-based balance 通常把两件事拆开：

- 主 LM loss 负责语言建模；
- expert bias / controller 负责负载调节；
- 过载专家降低 bias，欠载专家提高 bias；
- 很小的 sequence/global aux loss 可以保留为稳定器，但不再承担主要平衡压力。

对于延续预训练路由的 SFT/RL 项目，冻结 bias update 是一个值得先验证的方案。expert bias 更像控制器状态，不是一个普通可学习参数；后训练数据分布更窄，继续让它按小数据分布重平衡，可能破坏预训练阶段形成的专家分工。

后训练阶段是否继续更新 bias，需要结合数据规模和路由偏移判断。第二篇会区分 bias 控制器与 router weights，并讨论冻结或重新校准的条件。

## 7. 常见设计 四：sigmoid gating + top-k normalization 值得优先考虑

MoE router 的 score function 过去常从 softmax 开始。softmax 的好处是全专家分数具有概率解释；选出 top-k 后，是否再次归一化仍取决于实现；但它也会让专家之间强耦合，一个 expert 的分数变化会影响所有专家概率。

sigmoid gating 的趋势在于：每个 expert 的 score 更像独立打分，再从中选 top-k。选完之后通常再对 top-k weights 做 normalization。这样 router 的表达更自由，也更适合高专家数场景。

但 sigmoid 不是总是更好。它对几个细节很敏感：

- top-k 之后是否 renormalize；
- routing score 和 combine weight 是否使用同一套值；
- expert bias 是加在 score 前还是用于 choice-only；
- 推理 runtime 是否和训练 runtime 完全对齐；
- 低精度或量化是否改变 top-k 排序。

MoE 的训推偏差可能出现在专家选择、混合权重和最终 token logprob 上。第七篇讨论如何沿这些环节定位差异。

## 8. 常见设计 五：稀疏度需要和系统能力一起定

MoE 设计常会被推向极高稀疏度：total params 更大、active ratio 更低，理论效率更高。但实际训练里，稀疏度还会影响：

- 每个 expert 的 token 数；
- GroupedGEMM 的矩阵形状；
- EP all-to-all payload；
- device-level load balance；
- router 统计稳定性；
- serving p95/p99 latency。

按总参数规模整理，前述开放模型提供了以下配置样本范围：

| total 级别 | 常见 active 区间 | 常见专家设计 | 备注 |
|---|---:|---|---|
| 30B-100B | 1B-10B active | 64-128 experts，top-4/top-8 | 小 active、本地/边缘、高吞吐常见 |
| 100B-300B | 10B-22B active | 128-256 experts，top-8 左右 | 通用开源 MoE 主战场 |
| 500B-1T | 27B-55B active | 256-512 experts，top-8 左右 | 需要强 EP、GroupedGEMM、通信优化 |
| 2T-3T | 本文未核验 active 参数量 | Kimi K3 披露 896 experts、top-16，即 `k/E ≈ 1.79%` | 专家激活比例与 active 参数量分别记录 |

这不是 scaling law 结论，只是开放模型给出的可训练区间。真正做设计时，active params、top-k、expert granularity 和 EP 拓扑要一起 sweep。

## 9. 探索方向：LongCat 的 zero-computation expert

LongCat-Flash 里一个比较特殊的设计是 zero-computation experts。它的基本假设是：不是每个 token 在每一层都需要相同的 FFN compute。对简单 token，可以让它走一个 no-op / identity 风格的专家；对困难 token，再分配更多真实专家计算。

这等于把 MoE 从 fixed active budget 推向 token-dependent active budget。

它试图解决的问题很明确：

- 简单 token 少算；
- 困难 token 多算；
- 平均 active params 可以被控制在目标附近；
- 计算预算开始跟 token 难度绑定。

但它也带来非常现实的工程成本：

| 风险 | 说明 |
|---|---|
| 动态 shape | 每层真实计算量变化，CUDA Graph、dispatcher、buffer 规划更麻烦 |
| 负载控制更难 | 不只是专家之间均衡，还要控制真实 compute 总量 |
| router 更敏感 | router 同时决定 expert choice 和是否计算 |
| serving tail latency | 某些 batch 里困难 token 集中时，尾延迟可能抖动 |

所以 zero expert 更适合作为第二阶段能力。先把固定 top-k MoE 训练稳定，再引入 zero expert，会比第一版就动态化所有计算更便于调试。

## 10. 候选结构：LatentMoE / Stable LatentMoE

LatentMoE 把 routed expert path 放到低维 latent space，同时改变计算和数据移动的维度：

| 普通 MoE | LatentMoE |
|---|---|
| hidden state 直接进入专家 | hidden state 先降维到 latent |
| expert 权重和通信按 hidden dim 付费 | expert compute、参数和 payload 都按 latent dim 付费 |
| 通信墙/内存墙仍然可能很重 | 结构上减少 expert path 的 bytes 和 compute |

这对大 MoE 很关键，因为 MoE 的瓶颈经常不是理论 FLOPs，而是 dispatch/combine、expert parameter bytes、memory bandwidth 和 serving latency。

Nemotron 3 Ultra 把 LatentMoE 放进 550B-A55B 的公开模型里，Kimi K3 又用 Stable LatentMoE 配合 16/896 experts 把极高稀疏度推到更前台。这说明 latent expert path 不只是论文结构，也开始出现在前沿开放模型中。

这两条线的侧重点可以先这样区分：

| 路线 | 更核心的问题 |
|---|---|
| Nemotron LatentMoE | 如何让 routed expert path 的 compute、参数 bytes 和 dispatch payload 更便宜 |
| Kimi-K3 Stable LatentMoE | 在 2.8T、16/896 experts 这种极高稀疏度下，如何让 routing 和训练仍然稳定 |

LatentMoE 增加降维/升维 projection，并影响 shared path、精度和 runtime 兼容。通信或参数读取已成为瓶颈时，可优先比较这些额外成本与 latent path 节省的字节量。

## 11. 探索方向：Inkling 的 shared-expert sink

Inkling 的 shared-expert sink 需要单独讨论，因为它改变了 shared expert 的角色。

传统 shared expert 更像 always-on 公共路径：每个 token 都走，输出再和 routed experts 合并。Inkling 的公开材料里强调 shared experts 和 selected routed experts 一起做 normalization，让 shared path 可以在每个 token 上参与 mixture weight 的竞争。

这带来一个有意思的视角：shared expert 不一定只是固定补丁，也可以是 router mixture 里的一个“稳定吸收槽”。

可能收益是：

- common tokens 可以更多走 shared path；
- routed experts 更专注于 specialized tokens；
- shared path 不再只靠固定残差比例工作；
- 可能缓解某些 token 被硬塞到 routed experts 的问题。

但它还不是广泛验证的默认 recipe。要真正评估，需要看 shared expert weight 分布、route entropy、专家负载、长上下文和多模态 token 上的行为，以及 serving runtime 是否能把这种 mixture 高效实现。

## 12. 端侧/本地 MoE：active 小不等于推理快

端侧 MoE 常见的误区是：active params 小，所以推理一定快。

现实会复杂很多。MoE 的完整权重通常仍要存储或可访问；每个 token 的专家集合会变；batch=1 下小 expert GEMM 难以吃满硬件；router、dispatch、权重访问和 cache miss 都会变成显性成本。

Meta MobileMoE 联合研究 on-device scaling law、moderate sparsity、fine-grained/shared experts、INT4/INT8 QAT，并在真实手机上 profiling。结构、量化和设备测量由此有了对应的实验设置。

除此之外，Liquid LFM2.5-8B-A1B、IBM Granite MoE、gpt-oss-20b、OLMoE 这些开放模型/研究基线也值得关注，但要分层理解：

| 层级 | 代表 | 应该怎么用 |
|---|---|---|
| 严格手机端 | MobileMoE | 学设计原则和评估口径，不默认当 open-weight 生态样本 |
| 本地/边缘开放权重 | Liquid LFM2.5、Granite MoE、gpt-oss-20b | 看小 active MoE 的部署和工具调用形态 |
| 开放研究基线 | OLMoE | 做 router、expert load、后训练行为的可复现实验 |

第六篇会专门讲端侧 MoE 的不规则专家访问、active set、量化一致性和真实设备评估。

## 13. 从模型地图到配置选择

这些开放模型提供了设计起点，也暴露了不同选择的代价。细专家增加组合空间，shared expert 增加公共计算，负载控制影响专门化与系统均衡；LatentMoE 和 zero expert 又会改变参数、通信或执行形状。

第二篇从预算出发讨论这些取舍，给出需要固定的实验条件和可调整的变量。

## 系列导航

下一篇：[第二篇：Scaling MoE 的架构选择与训练策略]({% post_url 2026-07-06-moe-scaling-design-training %})

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- [Kimi K3 Tech Blog: Open Frontier Intelligence](https://www.kimi.com/blog/kimi-k3)
- [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471)
- [MiniMax-M2 Series Technical Report](https://arxiv.org/abs/2605.26494)
- [LongCat-Flash Technical Report](https://arxiv.org/abs/2509.01322)
- [NVIDIA Nemotron 3 Ultra](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/)
- [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)
- [Inkling: Our Open-Weights Model](https://thinkingmachines.ai/news/introducing-inkling/)
- [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/abs/2605.27358)
- [Liquid LFM2.5-8B-A1B](https://docs.liquid.ai/lfm/models/lfm25-8b-a1b)
- [IBM Granite 3.1 3B-A800M Base](https://huggingface.co/ibm-granite/granite-3.1-3b-a800m-base)
- [gpt-oss-20b Model](https://developers.openai.com/api/docs/models/gpt-oss-20b)
- [OLMoE-1B-7B](https://github.com/allenai/OLMoE)
