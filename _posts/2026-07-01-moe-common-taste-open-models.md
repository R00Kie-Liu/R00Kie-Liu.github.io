---
layout: post
title: "MoE我们喜欢你（一）：开源模型里的共识设计"
date: 2026-07-01
tags: [MoE, Pretraining, LLM]
description: "从最新开放模型出发，整理大规模 MoE 已经趋于稳定的架构设计实践，以及 LatentMoE、zero expert、shared-expert sink 等仍偏前沿的探索。"
---


## 0. 写在前面


从Deepseek成功scaling MoE并惊艳全球后，MoE已经是所有想要“训个大的”的公司默认选择（Dense已经路边一条），目前各家（Deepseek、GLM、Kimi & others）的MoE架构设计其实没有太大的区别，基本都收敛到了一个较为固定的范式。截止目前，一个大规模MoE的路线选择大致是 **fine-grained routed experts + shared expert + 前置 dense 层 + loss-free / bias-based balance + sigmoid/top-k normalization + dropless + router 高精度 + EP-friendly 静态执行路径**。

然而，LatentMoE、zero-computation expert、shared-expert sink、quantile/PID controller 这些稍微新一些的trick值得关注，但在基模架构设计阶段，是否直接应用，还是需要做足够充足的消融。

## 1. 一个预训练研究员对于MoE的scope

| 篇章 | 主题 | 重点问题 |
|---|---|---|
| 第一篇 | 开放模型里的 MoE common taste | 哪些 MoE 设计已经趋于稳定，哪些还属于少数模型创新 |
| 第二篇 | Scaling MoE 的架构与训练 recipe | 稀疏度、shared expert、LatentMoE、zero-expert、loss-free balance 怎么选 |
| 第三篇 | Megatron Core MoE 源码剖析 | router、load-balance loss、expert bias、token drop、dispatcher 和 experts 如何实现 |
| 第四篇 | DeepEP 与 GroupedGEMM | MoE 大规模训练的通信墙和计算墙 |
| 第五篇 | 100B MoE 训练配置推演 | 给定模型规格后如何配置并行策略、估算显存、迁移短/长上下文训练 |
| 第六篇 | 端侧/本地小 active MoE | 推理瓶颈、active set、expert cache、量化、真实设备评估口径 |
| 第七篇 | MoE 后训练/RL 稳定性 | 为什么 MoE 在 RL 里容易出现训推不一致，如何监控和缓解 |

## 2. 先把 common taste 和 frontier experiment 分开

讨论 MoE 时，一个常见误区是把“某个新模型用了”直接等同于“应该默认采用”。这在 MoE 里尤其危险，因为 MoE 的收益很依赖系统、并行拓扑、训练数据规模和推理 runtime。

更稳的分类方式是：

| 类型 | 判断标准 | 例子 |
|---|---|---|
| common taste | 多个开放模型或训练框架都支持，失败模式相对清楚 | shared expert、fine-grained experts、前置 dense 层、loss-free balance、router FP32、dropless 优先 |
| strong candidate | 有清晰动机和大模型证据，但实现/训练/部署成本更高 | LatentMoE、group/node-limited routing、device-level balance、FP8/MXFP8 expert compute |
| frontier experiment | 少数模型报告中效果突出，但尚未成为通用 recipe | LongCat zero-computation expert、Inkling shared-expert sink、Kimi K3 Stable LatentMoE、quantile/PID controller 变体 |

这个分类用于给 recipe 排优先级。预训练阶段同时修改 router、balance、专家形状和 dispatcher，会显著增加问题归因难度。

## 3. 最新开放模型里的 MoE 地图

下表按“对架构选择的启发”整理。这里会区分完整 technical report、model card、官方 blog/API 文档和 framework day-0 支持材料，避免把证据强度不一样的东西混在一起。

| 模型 | 参数/激活 | MoE 设计信号 | 对 common taste 的启发 |
|---|---:|---|---|
| DeepSeek-V3 | 671B / 37B active | DeepSeekMoE、前 3 层 dense、256 routed experts、top-8、shared expert、aux-loss-free balance | loss-free balance 和 shared expert 成为大规模 MoE baseline 的强信号 |
| Qwen3 | 235B-A22B、30B-A3B | dense + MoE 同步发布，MoE 进入通用开放生态 | MoE 不只服务 500B+ 模型，小 active 版本也值得系统化 |
| Kimi K2 | 1T / 32B active | 大总参、小 active、agentic data synthesis、MuonClip、joint RL | MoE 和 agentic 后训练开始深度绑定 |
| Kimi K3 | 2.8T，16/896 experts | Stable LatentMoE、Kimi Delta Attention、Attention Residuals | 极高专家数需要 latent expert 和更强稳定性设计配合；完整训练细节仍要继续核验 |
| GLM-4.5 | 355B / 32B active | loss-free balance routing、sigmoid gate、agentic/reasoning/coding 统一 | sigmoid + bias balance 已经不再是小众选择 |
| MiniMax-M2 | 229.9B / 9.8B active | 256 fine-grained experts、很小 active budget、agentic deployment | 10B active 级别也能承载 agent 能力，说明 active budget 可以继续压 |
| LongCat-Flash | 560B / dynamic active | zero-computation experts、PID expert bias、shortcut-connected MoE | token-dependent compute 有吸引力，但不适合作为初始 baseline |
| Nemotron 3 Ultra | 550B / 55B active | Hybrid Mamba-Attention、LatentMoE、NVFP4、MTP | 架构、低精度和系统吞吐一起 co-design |
| Inkling | 975B / 41B active | 256 routed experts、2 shared experts、top-6、sigmoid router、aux-loss-free bias、shared-expert sink | shared expert 开始从 always-on path 走向更细的 mixture 设计 |
| Meta MobileMoE | 1.3B-5.3B total / 0.3B-0.9B active | on-device scaling law、moderate sparsity、fine-grained + shared experts、QAT、真实手机 profiling | 端侧 MoE 不能只看 active params，必须看真实设备上的访问规则性和 latency |
| Liquid LFM2.5-8B-A1B | 8B / 1.5B active | on-device / edge MoE、128K context、tool calling | 本地/边缘 MoE 已经有开放权重样本，但不等同手机 SoC 上的严格端侧 |
| IBM Granite MoE | 1B-A400M、3B-A800M | fine-grained experts、dropless routing、128K context、Apache 2.0 | 企业本地/低资源 MoE 的实用基线 |
| OLMoE-1B-7B | 约 7B / 1B-1.3B active | 64 experts、top-8、代码/数据/logs 开放 | 适合做可复现路由分析，而不是追 frontier 能力 |

这张表的重点不在“谁的 total params 最大”，而在几个重复出现的设计：

- active params 被压到 0.3B、1B、3B、10B、22B、32B、55B 等多个档位；
- shared expert 反复出现；
- fine-grained routed experts 反复出现；
- loss-free / bias-based balance 反复出现；
- sigmoid routing 和 top-k normalization 出现频率上升；
- 前置 dense 层仍然被很多大 MoE 保留；
- 推理侧的 communication、memory bandwidth、专家访问规则性越来越像一等约束。

这就是 common taste 的来源。

## 4. Common taste 一：前 1-3 层 dense，别太早开始路由

很多 MoE 设计会在底部保留 1-2 层，甚至 3 层 dense MLP，再进入 MoE layer。这个设计不显眼，但很实用。

底层 token representation 还偏局部、偏表层，router 太早介入容易出现两个问题：

- 通用 pattern 被拆到多个专家里重复学习；
- 早期 layer 的 load 统计噪声大，router 更容易 collapse 或形成脆弱分工。

前置 dense 层相当于让底层先做稳定的 lexical / local feature processing，再把更抽象的 hidden states 交给 router。对大规模训练来说，这是一种低成本稳定器；对端侧模型来说，它还减少了最底层就发生专家调度带来的不规则访问。

所以第二篇会把“前置 1-2 层 dense”写进默认 recipe。它不是定律，但如果没有强反证，不建议一上来把每一层 FFN 都换成 MoE。

## 5. Common taste 二：fine-grained routed experts + shared expert

DeepSeekMoE 之后，fine-grained experts 和 shared expert 逐渐成为大规模 MoE 的常见起点。

fine-grained experts 解决的是组合空间问题。给定同样 active compute，与其让 token 从少数大专家里选，不如拆成更多更小的专家，让 top-k 组合表达更细的分工。这样可以减少专家之间的冗余，也让 router 更容易形成语言、领域、任务、格式等隐式 specialization。

shared expert 解决的是通用知识重复问题。每个 token 都需要一些通用变换，如果完全交给 routed experts，多个专家可能重复学习同一类 common pattern。shared expert 提供一条 always-on 的公共路径，让 routed experts 更专注于差异化部分。

这组设计的代价也要记住：

| 设计 | 收益 | 代价 |
|---|---|---|
| fine-grained experts | 更细组合、更强 specialization、更小 expert GEMM | router/load balance 更敏感，dispatch 更碎 |
| shared expert | 通用知识不必在专家中重复，router 错选伤害更小 | always-on compute，过大时会削弱 sparse 收益 |

一个实用判断是：shared expert 应该是稳定器，不应该变成第二个 dense MLP。它要足够强，能承接通用 token；也要足够小，避免削弱 MoE 的 latency 优势。

## 6. Common taste 三：loss-free / bias-based balance 变成主线

MoE load balance 的目标存在张力：系统希望每个专家、每个 EP rank 的负载接近均衡；模型又希望 router 有自由度，让专家保持专门化。

早期 auxiliary loss 的做法直接：把负载均衡作为额外 loss 加到 LM objective 上。但大规模训练里，这个 loss 的系数、schedule、统计粒度和 capacity factor 都会变成隐性超参。系数太小，专家过载；系数太大，router 为了均衡牺牲主目标，专家变得平均而不专门。

loss-free / bias-based balance 通常把两件事拆开：

- 主 LM loss 负责语言建模；
- expert bias / controller 负责负载调节；
- 过载专家降低 bias，欠载专家提高 bias；
- 很小的 sequence/global aux loss 可以保留为稳定器，但不再承担主要平衡压力。

这也是为什么 SFT/RL 阶段通常要 freeze bias update。expert bias 更像控制器状态，不是一个普通可学习参数；后训练数据分布更窄，继续让它按小数据分布重平衡，可能破坏预训练阶段形成的专家分工。

从 common taste 角度，今天的大 MoE baseline 可以这样理解：

| 项 | 更推荐的默认值 |
|---|---|
| 主 balance | loss-free / expert-bias controller |
| aux loss | none 或很小 seq/global aux stabilizer |
| router dtype | FP32 优先，避免随意低精度化 |
| token drop | dropless 优先，capacity/drop 作为保护机制 |
| SFT/RL | freeze expert bias update，监控 top-k drift |

## 7. Common taste 四：sigmoid gating + top-k normalization 值得优先考虑

MoE router 的 score function 过去常从 softmax 开始。softmax 的好处是概率解释清楚，top-k weights 自然归一化；但它也会让专家之间强耦合，一个 expert 的分数变化会影响所有专家概率。

sigmoid gating 的趋势在于：每个 expert 的 score 更像独立打分，再从中选 top-k。选完之后通常再对 top-k weights 做 normalization。这样 router 的表达更自由，也更适合高专家数场景。

但 sigmoid 不是总是更好。它对几个细节很敏感：

- top-k 之后是否 renormalize；
- routing score 和 combine weight 是否使用同一套值；
- expert bias 是加在 score 前还是用于 choice-only；
- 推理 runtime 是否和训练 runtime 完全对齐；
- 低精度或量化是否改变 top-k 排序。

这也是第七篇要重点讲的地方：MoE 的训推不一致很多时候不是 logits 层面的问题，而是 router 选专家这一步细节不一致。

## 8. Common taste 五：稀疏度需要和系统能力一起定

MoE 设计常会被推向极高稀疏度：total params 更大、active ratio 更低，理论效率更高。但实际训练里，稀疏度还会影响：

- 每个 expert 的 token 数；
- GroupedGEMM 的矩阵形状；
- EP all-to-all payload；
- device-level load balance；
- router 统计稳定性；
- serving p95/p99 latency。

从开放模型看，比较稳的区域大致是：

| total 级别 | 常见 active 区间 | 常见专家设计 | 备注 |
|---|---:|---|---|
| 30B-100B | 1B-10B active | 64-128 experts，top-4/top-8 | 小 active、本地/边缘、高吞吐常见 |
| 100B-300B | 10B-22B active | 128-256 experts，top-8 左右 | 通用开源 MoE 主战场 |
| 500B-1T | 27B-55B active | 256-512 experts，top-8 左右 | 需要强 EP、GroupedGEMM、通信优化 |
| 2T-3T | 16/896 这类极高稀疏度开始出现 | 更依赖 latent expert / 强 balance / supernode 推理 | 仍属于 frontier 设计区 |

这不是 scaling law 结论，只是开放模型给出的可训练区间。真正做设计时，active params、top-k、expert granularity 和 EP 拓扑要一起 sweep。

## 9. Frontier experiment 一：LongCat 的 zero-computation expert

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

## 10. Frontier experiment 二：LatentMoE / Stable LatentMoE

LatentMoE 的核心不是单纯减少 FLOPs，而是把 routed expert path 放到低维 latent space：

| 普通 MoE | LatentMoE |
|---|---|
| hidden state 直接进入专家 | hidden state 先降维到 latent |
| expert 权重和通信按 hidden dim 付费 | expert compute、参数和 payload 都按 latent dim 付费 |
| 通信墙/内存墙仍然可能很重 | 结构上减少 expert path 的 bytes 和 compute |

这对大 MoE 很关键，因为 MoE 的瓶颈经常不是理论 FLOPs，而是 dispatch/combine、expert parameter bytes、memory bandwidth 和 serving latency。

Nemotron 3 Ultra 把 LatentMoE 放进 550B-A55B 的公开模型里，Kimi K3 又用 Stable LatentMoE 配合 16/896 experts 把极高稀疏度推到更前台。这说明 latent expert path 不只是论文结构，也开始进入 frontier open-model narrative。

这两条线的侧重点可以先这样区分：

| 路线 | 更核心的问题 |
|---|---|
| Nemotron LatentMoE | 如何让 routed expert path 的 compute、参数 bytes 和 dispatch payload 更便宜 |
| Kimi-K3 Stable LatentMoE | 在 2.8T、16/896 experts 这种极高稀疏度下，如何让 routing 和训练仍然稳定 |

但 common taste 角度仍要保守一点。LatentMoE 会引入额外的降维/升维 projection、精度保护、shared expert 交互和 checkpoint/runtime 兼容问题。它值得作为 100B+ 或通信/带宽瓶颈明显时的 strong candidate，不一定是所有小 MoE 的默认起点。

## 11. Frontier experiment 三：Inkling 的 shared-expert sink

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

Meta MobileMoE 的价值在于，它不是只报 FLOPs，而是把 on-device scaling law、moderate sparsity、fine-grained experts、shared experts、INT4/INT8 QAT 和真实手机 profiling 放进同一个闭环。

除此之外，Liquid LFM2.5-8B-A1B、IBM Granite MoE、gpt-oss-20b、OLMoE 这些开放模型/研究基线也值得关注，但要分层理解：

| 层级 | 代表 | 应该怎么用 |
|---|---|---|
| 严格手机端 | MobileMoE | 学设计原则和评估口径，不默认当 open-weight 生态样本 |
| 本地/边缘开放权重 | Liquid LFM2.5、Granite MoE、gpt-oss-20b | 看小 active MoE 的部署和工具调用形态 |
| 开放研究基线 | OLMoE | 做 router、expert load、后训练行为的可复现实验 |

第六篇会专门讲端侧 MoE 的不规则专家访问、active set、量化一致性和真实设备评估。

## 13. 一套更稳的默认 recipe

如果只从这篇里带走一个版本，可以是下面这张表。

| 组件 | 默认建议 | 什么时候再升级 |
|---|---|---|
| MoE layer placement | 前 1-2 层 dense，后续按固定频率使用 MoE | 底层表示很稳、训练规模足够大时再尝试更激进替换 |
| Expert granularity | fine-grained routed experts | 如果 GroupedGEMM 太碎，再回调 expert size/top-k |
| Shared expert | 小 shared expert 默认考虑 | shared 占比过大或 serving latency 受影响时缩小 |
| Router score | sigmoid + top-k normalization 值得优先试 | 如果实现栈 softmax 更成熟，可先 softmax baseline |
| Load balance | loss-free / expert-bias balance 主导 | 小 seq/global aux 作为稳定器，避免 aux loss 接管主目标 |
| Token drop | dropless 优先 | 只把 capacity/drop 当保护或极端内存约束 |
| Router precision | FP32 或至少高精度保护 | 量化前必须测 top-k consistency |
| LatentMoE | strong candidate | 通信/带宽/参数 bytes 成为瓶颈时优先评估 |
| Zero expert | 第二阶段探索项 | 固定 top-k MoE 跑稳后再引入 |
| Shared-expert sink | 前沿探索项 | 需要能监控 shared/routed mixture 权重和 runtime 支持 |

## 14. 结论

今天的 MoE common taste 已经比较清楚：先把稳定性和系统可执行性做好，再追极高稀疏度和动态计算。

更具体地说：

1. **稳定底座**：前置 dense 层、fine-grained experts、shared expert、router 高精度、dropless 优先。
2. **负载控制**：loss-free / bias-based balance 是主线，aux loss 更适合作为轻量稳定器。
3. **稀疏度选择**：active ratio 需要和 expert token 数、GroupedGEMM、EP 拓扑、serving latency 一起评估。
4. **前沿结构**：LatentMoE 可以作为 strong candidate；zero expert、shared-expert sink 先作为探索项。
5. **后训练和推理**：MoE 不只是训练架构，router 一致性会一路影响 SFT、RL 和 serving。

下一篇会把这些 common taste 收敛成更具体的 scaling recipe：给定模型规模和稀疏度，MoE layer 怎么放、top-k 怎么选、shared expert 多大、LatentMoE 何时值得上，以及 loss-free balance 在预训练和 SFT/RL 阶段怎么处理。

## 系列导航

下一篇：[第二篇：如果今天要 scaling MoE，架构和训练策略该怎么选]({% post_url 2026-07-06-moe-scaling-design-training %})

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
