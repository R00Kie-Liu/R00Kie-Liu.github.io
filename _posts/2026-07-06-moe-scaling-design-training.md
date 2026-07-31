---
layout: post
title: "MoE我们喜欢你（二）：Scaling MoE 的架构选择与训练策略"
date: 2026-07-06
tags: [MoE, Pretraining, Scaling]
description: "讨论大规模 MoE 的稀疏度、expert granularity、shared expert、LatentMoE、zero expert、loss-free balance 与训练策略选择。"
---


## 0. 导读

如果今天要给一个较大规模 MoE 预训练项目定方案，不应该先从“总参要做多大”开始，而应该先反推 active budget、通信预算和后训练风险。一个默认方案大概会是：

- **fine-grained routed experts + 1 个 shared expert**，先保证通用知识和专家专门化分开。
- **前 1-2 层先保留 dense FFN，再进入 MoE layers**。很多开放 MoE 会在底部插入 dense warmup layers，让底层 token 表示先稳定下来；如果 router 早期负载仍不稳，可以扩到前 3-4 层 dense。
- **active params 按部署目标反推**，再确定 total params。通用大模型可从 5%-10% active ratio 附近起步；高吞吐/端侧/agent serving 可以尝试 1%-5%，但要付出更强 routing 和系统优化成本。
- **top-k 以 4/8 为主，超大专家池或 LatentMoE 下再考虑 16+**。top-k 越大，质量和稳定性可能更好，但通信、combine、专家计算更贵。
- **router 使用 FP32，优先 sigmoid scoring + top-k normalization 或已验证的 softmax baseline**。高专家数下 router 精度和 score scale 会变成稳定性问题。
- **load balance 默认用 loss-free / expert-bias 方法**，必要时加很小的 sequence/global aux loss 作为稳定器。
- **LatentMoE 适合通信/内存/延迟成为瓶颈的中大型 MoE**，尤其是专家数很高、EP 跨节点、在线 serving 压力大的场景。
- **zero-computation expert 是可选增强，不是默认起点**。它适合做 adaptive compute，但会显著增加 routing、静态 shape 和负载控制难度。
- **训练中尽量避免 token dropping**。drop 掉的往往不是无害 token，而可能是代码、数学、多语长尾或 hard token。

## 1. 先定义预算：MoE scaling 不是只看 FLOPs

立项阶段应先把四个预算写清楚：

| 预算 | 含义 | 设计影响 |
|---|---|---|
| total parameter budget | 总参数容量 | 决定知识容量、存储、checkpoint、权重加载 |
| active compute budget | 每 token 激活参数/FLOPs | 决定训练吞吐和推理成本 |
| token budget | 训练 token 总量或阶段 token 预算 | 决定 scaling law 最优点、router 统计稳定性和专家是否充分训练 |
| communication budget | EP dispatch/combine 和设备拓扑预算 | 决定 MoE 在真实系统里是否比 dense 更快 |

很多 MoE 方案在论文 FLOPs 指标上表现不错，但落到训练系统时，all-to-all、expert imbalance、token permutation 和小 GEMM 效率都会进入主瓶颈。因此 MoE scaling 的第一原则是：**active budget 和通信预算要一起设计**。

## 2. 稀疏度怎么选

MoE 里有两个容易混淆的稀疏度：

- expert activation ratio：每个 token 激活多少专家，例如 8/256、16/896。
- active parameter ratio：每 token active params / total params，例如 37B/671B。

这两个比例相关但不等价，因为 shared expert、attention、embedding、latent projection、每层是否 MoE 都会改变 active params。

### 2.1 从开放模型看可行区间

| 模型类型 | 代表配置 | 可借鉴的 active 区间 |
|---|---|---|
| 通用大 MoE | DeepSeek-V3 671B/37B、Qwen3 235B/22B | 约 5%-10% active params 是成熟区间 |
| agentic / 高吞吐 MoE | Kimi K2 1T/32B、MiniMax-M2 229.9B/9.8B、LongCat-Flash dynamic active | 约 3%-5% active params 更强调 serving 成本 |
| 小激活/端侧/本地 MoE | Meta MobileMoE、Liquid LFM2.5-8B-A1B、Granite MoE、OLMoE、Qwen3-30B-A3B、Nemotron 3 Nano 30B-A3B | active 绝对值很小；手机端、边缘本地和开放研究基线要分层比较 |
| 3T-class ultra-sparse | Kimi K3 16/896 experts | expert activation ratio 约 1.8%，但 active params 需要完整报告进一步核验 |

一个实用建议：

- 第一版大模型不宜盲目追求极低 active ratio。5%-10% 通常更利于训练稳定。
- 如果目标是 serving 成本，先把 active params 降到 10B-30B，再通过 total params 扩容量。
- 如果目标是 2T+ total params，极高专家数几乎不可避免，但必须搭配 latent expert、强 balance 和 EP 静态形状优化。

### 2.2 top-k 的取舍

top-k 越大，每个 token 可以组合更多专家，路由更平滑，专家遗漏风险更小；但也会增加 expert compute、dispatch payload、combine cost 和 load balance 难度。

| top-k | 适用场景 | 风险 |
|---:|---|---|
| 1-2 | 极低延迟、小模型、Switch-like baseline | expert capacity 容易不足，专家专门化和稳定性压力大 |
| 4 | 中等专家数、较强 serving 约束 | 对 router 质量更敏感 |
| 8 | 当前开放大 MoE 的常见甜点 | 通信和专家 compute 可控，质量/稳定性较均衡 |
| 16+ | 超大专家池、LatentMoE、3T-class 稀疏模型 | 必须解决 dispatch、combine、静态 shape 和 balance |

默认选择是：**100B-300B 级先试 top-4/top-8，500B+ 先试 top-8；如果用了 LatentMoE 或专家极多，再评估 top-16。**

### 2.3 MoE layer placement：前置 dense 层通常值得保留

一个常被忽略的架构细节是：很多开放 MoE 并不是从第 1 个 Transformer block 就开始 MoE 化，而是在底部先放若干 dense FFN 层，再进入稀疏 MoE。这个设计可以理解成 router 的“表示预热”。

底层 hidden states 更接近局部 token、词形、位置和浅层组合特征。如果太早路由，router 看到的表示还不够语义化，容易出现几个问题：

- 早期专家负载更不稳定，router collapse 更容易发生。
- 专家会被迫学习过多底层通用模式，削弱后续专家专门化。
- token dispatch 在所有层都发生，增加不必要的 EP 通信。
- 底层 MoE 的收益未必抵得过路由和通信成本。

因此，比较稳的 recipe 是：

| 规模/目标 | 前置 dense 层建议 | 说明 |
|---|---:|---|
| 小激活/端侧/本地 MoE | 1 层 | 控制 active compute 和实现复杂度 |
| 100B-300B 通用 MoE | 1-2 层 | 作为默认起点，给 router 更稳定的输入 |
| 500B+ 或 router 不稳定 | 2-4 层 | 牺牲少量稀疏化比例，换训练稳定性和更少底层通信 |
| hybrid attention / 多模态 MoE | 视前端 encoder/adapter 而定 | 如果前端已做较强表示抽取，可减少 dense warmup |

从现有开放模型看，这个模式并不少见：有的模型只保留首层 dense，有的保留前 3 层 dense，也有模型把前 4 层或最后一层排除在 MoE 之外。这里不必机械追求“所有 FFN 都 MoE 化”。对 scaling 来说，前置 dense 层减少的是一小部分参数容量，但经常能换来更好的 router 稳定性、底层表示质量和 EP 训练效率。

默认起点可以设为：**前 1-2 层 dense，后续大部分 FFN 使用 MoE；如果 early-layer expert load CV 明显偏高或 router entropy 过早下降，再增加 dense warmup 层数。**

## 3. Expert granularity：为什么要细专家

Fine-grained experts 的意义不是“专家越多越好”，而是让同样 active compute 能组合出更丰富的专家子集。

DeepSeekMoE 的思路可以拆成两部分：

- routed experts 做细，让不同 token 可以组合更精细的知识片段；
- shared experts 隔离出来，负责通用知识，减少 routed experts 之间重复学习 common knowledge。

Scaling-law 视角下，granularity 是一个独立超参。它和训练 token、模型大小、active ratio 一起决定最优点。过粗的专家容易冗余，过细的专家会让每个专家 token 数不足、GEMM 变小、通信更碎。

实操建议：

| 总规模 | 专家数起点 | top-k 起点 | 备注 |
|---:|---:|---:|---|
| 1B-80B | 32-128 | 2-8 | 面向小激活/端侧/本地时要关注专家驻留、缓存、量化和 kernel 利用率 |
| 100B-300B | 128-288 | 4-8 | 当前比较稳的工程区间 |
| 500B-1T | 256-512 | 8 | 需要强 EP 和通信 overlap |
| 2T+ | 512-896+ | 8-16+ | 建议同时考虑 LatentMoE 或其他通信压缩结构 |

这些值不应当机械套用。真正的选择应由目标 active params、每层 MoE 比例、hidden size、FFN expansion、global batch、EP 拓扑和 token budget 一起决定。

## 4. Shared expert：现在几乎应该默认考虑

Shared expert 的作用是把所有 token 都需要的通用变换从 routed expert 中拿出来。它解决两个问题：

- routed experts 不必重复学习 common knowledge；
- router 可以把专家容量更多用于长尾、任务、语言、代码、数学、工具调用等差异化模式。

什么时候需要 shared expert：

- 专家数较多，routed experts 容易冗余。
- 多语/代码/数学/工具数据混合较复杂。
- top-k 较小，担心通用知识被错路由。
- 后训练阶段分布变化大，需要一个稳定公共路径。

什么时候可以不加或减小：

- 模型规模较小，shared expert 增加的 active compute 成本收益不明显。
- 使用了很强的 latent expert 或共享低秩结构。
- 实验发现 shared expert 抢走 routed experts 的学习信号。

建议做法：

- 第一版使用 1 个 shared expert。
- shared expert intermediate size 可以小于完整 FFN，先从 routed expert 总 active compute 的一小部分起步。
- 单独监控 shared expert gate/activation 对输出的贡献，避免它变成“第二个 dense FFN”。

## 5. LatentMoE：什么时候值得上

LatentMoE 的基本做法是把 routed expert 计算放到低维 latent space。输入 hidden state 先降维为 latent 表示，router 和 experts 主要在 latent space 里工作，再投回原 hidden dimension。

可以用一个很短的伪代码区分普通 MoE 和 LatentMoE：

```python
topk_ids, weights = router(x)
normal_y = combine(experts(dispatch(x, topk_ids)), weights)

topk_ids, weights = router(x)          # routing can still read full hidden states
z = down_proj(x)                       # hidden -> latent
y_latent = combine(experts(dispatch(z, topk_ids)), weights)
latent_y = up_proj(y_latent)           # latent -> hidden
```

这里可以把 Nemotron 和 Kimi-K3 的口径先分开：Nemotron 的 LatentMoE 更像结构压缩，回答“routed expert path 能不能按 latent dim 付费”；Kimi-K3 的 Stable LatentMoE 更像在这个结构上叠加极高专家数的稳定化 recipe，回答“896 experts、top-16 这种稀疏度下 routing 怎么不崩”。后者的具体训练细节仍要等技术报告核验。

它值得上的场景：

- EP all-to-all 已经成为主要瓶颈。
- 专家数很高，dispatch payload 和 expert cache pressure 明显。
- online serving 受 memory bandwidth / latency 限制，而不是单纯受 FLOPs 限制。
- 希望扩大 expert pool，但不想线性增加专家参数 bytes 和通信量。
- 目标模型进入 500B+ 或 T 级，普通 MoE 的系统成本开始压过 FLOPs 优势。

它不一定适合的场景：

- 小模型或小专家数，瓶颈还在 dense attention / MLP compute。
- 训练框架尚无成熟支持，调试成本不可接受。
- 主要目标是快速复现已有 dense/MoE baseline，而不是做系统-架构协同优化。

LatentMoE 的关键指标不只是 perplexity，还要看：

- accuracy per FLOP；
- accuracy per parameter；
- dispatch bytes per token；
- expert cache miss；
- end-to-end latency；
- high-throughput batch 下 EP bandwidth；
- low-latency batch 下 memory bandwidth。

对 Kimi-K3 这类 Stable LatentMoE，当前更适合把它当作方向信号；完整 technical report 出来前，Quantile Balancing、QAT 等训练细节不应写成已确认实现。

## 6. Zero-computation expert：好想法，但不应一开始就滥用

Zero-computation expert 可以理解为 MoE 里的 no-op / identity 路径：某些 token 在某些 MoE 层被判断为“不值得做完整 FFN expert compute”。LongCat-Flash 把它作为 dynamic computational budget allocation 的核心组件，并配合 PID expert bias 做负载控制。

它的价值：

- easy tokens 少算；
- hard tokens 多算；
- active params 可以随上下文需求变化；
- 推理吞吐有机会明显改善；
- MoE 从“固定 top-k sparse compute”变成“token-budget-aware compute”。

它的代价：

- 动态 active params 让静态 shape 更难；
- load balance 不再只是专家间 token 数均衡，还要控制总 compute；
- router 更容易学到捷径，过多 token 走 no-op 会伤害能力；
- CUDA Graph、GroupedGEMM、EP overlap 都更难调；
- 后训练/RL 中策略分布漂移会影响 zero-expert 使用率。

建议把 zero-expert 作为第二阶段实验：

- 先训练固定 top-k 的稳定 MoE baseline。
- 加入 zero-expert 后固定平均 active budget。
- 监控每层 zero-expert rate、任务相关 zero rate、hard token zero rate。
- 对数学、代码、长链推理、工具调用样本单独做 no-op 分析。
- 保持 shape 静态或接近静态，避免动态调度抵消系统收益。

## 7. Load balance 训练策略

### 7.1 不建议纯靠 auxiliary loss

Auxiliary loss 的问题不是不能用，而是它直接改变主优化目标。系数太大，专家被迫平均；系数太小，负载还是崩。对大规模 MoE，balance 策略应该尽量减少对 LM objective 的干扰。

更好的默认组合：

- loss-free / expert-bias balance 作为主平衡机制；
- 很小的 sequence-level 或 global-batch aux loss 作为稳定器；
- router logits 和 router score 用 FP32；
- 不做 token dropping，或只作为最后保护机制；
- 持续记录 expert load、device load、router entropy、top-k margin 和 bias drift。

### 7.2 Loss-free balance 的工作方式

Loss-free balance 的机制直接：如果某个 expert 一直过载，就降低它的 routing bias；如果某个 expert 一直欠载，就提高它的 routing bias。这样主 loss 仍然优化语言建模，负载控制器单独管理使用率。

需要调的关键项：

| 项 | 作用 | 风险 |
|---|---|---|
| bias update rate | 控制负载调节速度 | 太大震荡，太小反应慢 |
| load window | 统计专家负载的时间窗口 | 太短噪声大，太长适应慢 |
| target load | 每个专家目标 token/compute | 动态 routing 下可能不再相等 |
| sequence/global stabilizer | 防止局部 batch 偏斜 | 过强会干扰专家专门化 |
| device-level balance | 控制 EP rank 负载 | 只平衡 expert 不等于设备平衡 |

### 7.3 Quantile Balancing：把专家负载变成阈值问题

Quantile Balancing 可以理解为另一类 loss-free balance：给每个 expert 维护一个动态价格 `beta`，routing 时选 `score - beta` 最高的 experts。热门 expert 的 `beta` 变大，被扣分更多；冷门 expert 的 `beta` 变小，更容易被选中。

“分位数”在这里就是 cutoff threshold。如果希望某个 expert 只接收 `target` 个 token，就把所有 token 对它的竞争力分数排序，取刚好能留下 `target` 个 token 的那个阈值。

```python
adjusted = scores - beta
topk_ids = topk(adjusted, k)

row_cutoff = kth_value(adjusted, k + 1, dim="experts")
competitiveness = scores - row_cutoff
target = num_tokens * k // num_experts
beta_next = per_expert_quantile(competitiveness, target)
```

它的优点是 balance 信号更直接，不需要把强 aux loss 加进 LM objective；代价是 batch 要足够大，`beta` 更新要稳定，并且要确认 serving 侧也能复现同样的 choice-only routing 语义。对 512/896 experts 这类大专家池，它比“只靠 aux loss 慢慢拉平”更像一个显式负载控制器。

### 7.4 SFT/RL 阶段是否 freeze expert bias

对于 DeepSeek-style loss-free balance，SFT/RL 阶段更稳的默认做法是 **freeze expert bias update**，也就是保留预训练/continued pretrain 末期的 expert bias 值，但把 bias update rate 设为 0。这里的 freeze 不是普通意义上的 `requires_grad=False`，因为这类 bias 通常不是通过主 loss 反向传播学习出来的参数，而是根据专家负载动态更新的控制器状态。

这样做的原因是：SFT 数据量通常远小于预训练，且 domain / instruction / long-CoT / agent 任务分布更偏。如果继续用 SFT batch 的负载统计去更新 expert bias，router 可能会为了这批偏分布数据重新平衡专家，带来 route drift，破坏预训练阶段已经形成的专家专门化。RL 阶段还会多一个 rollout / learner 分布不一致的问题，动态改 bias 更容易制造 off-policy routing mismatch。

比较稳的策略是：

| 阶段 | expert bias update | 说明 |
|---|---|---|
| pretraining 前中期 | 开启，使用较小 update rate | 建立全局专家负载平衡 |
| pretraining 后期 / long-context midtrain | 逐步降到 0 或直接设为 0 | 固化路由控制器，避免后期震荡 |
| SFT | 默认 freeze bias update | 保留预训练路由结构，避免小数据偏分布重写负载 |
| RL | 更应谨慎，默认 freeze | 减少 rollout/learner 路由漂移和 off-policy mismatch |
| domain continued pretrain | 可小幅开启或短期重校准 | 仅当数据规模足够大且确实出现严重 load imbalance |

需要区分两件事：**freeze expert bias update 不等于必须 freeze router weights**。如果做全参数 SFT，router projection 仍可能随主 loss 更新；如果 SFT 数据很小、路由很脆，才进一步考虑冻结 router 或只做 LoRA/adapter。对 loss-free balance 而言，关键是避免负载控制器在小规模 SFT/RL 数据上继续大幅移动。

### 7.5 新开源模型里的实践信号

| 模型 | 训练/路由信号 | 可借鉴点 |
|---|---|---|
| DeepSeek-V3 | aux-loss-free balance、shared expert、no token dropping、sequence-wise aux stabilizer；预训练后段 bias update speed 设为 0 | loss-free balance 是大规模 MoE 强 baseline；SFT 前固化 bias 是合理默认 |
| Qwen3 | dense + MoE 同步发布，global-batch load balance 口径 | MoE 可进入多尺寸模型族，而非单旗舰 |
| GLM-4.5 | sigmoid gate、loss-free balance bias、sequence aux loss；前 15T 更新 bias，后续 update rate 设为 0 | loss-free 与辅助稳定项可以共存，后期/SFT 前 freeze bias update 很自然 |
| MiniMax-M2 | 256 fine-grained experts、top-8、expert-specific bias、MTP | 小 active agent MoE 可用 expert bias 替代强 aux |
| LongCat-Flash | zero-computation experts、PID expert bias、device-level balance | adaptive compute 需要控制系统级 balance |
| Kimi K3 | Stable LatentMoE、16/896 experts；Quantile Balancing / QAT from SFT 等细节待技术报告核验 | 极高稀疏度下 balance 和静态 EP shape 是一等问题 |
| Inkling | 256 routed experts、top-6，再加 2 个 shared experts；shared experts 作为 routing-score sink，但不进入 top-k routed candidate pool | shared expert 不一定只是 always-on FFN，也可以参与 router normalization，吸收一部分概率质量 |
| Megatron Core | aux/seq/global/sinkhorn/loss-free 均支持，router FP32 建议，Flex/DeepEP dispatcher | 框架层已把多种 balance 与 EP 路径纳入标准接口 |

## 8. Router 设计细节

### 8.1 Sigmoid vs softmax

Softmax router 的好处是概率归一化天然清晰，适合早期 baseline。Sigmoid scoring 的好处是每个专家的得分更独立，和 top-k normalization 配合时可以减少专家之间的强竞争。DeepSeek、GLM、MiniMax 等路线都显示 sigmoid gate 正在变得更常见。

实操上不应把这个问题抽象成“sigmoid 一定更好”。更重要的是：

- router logits 保持 FP32；
- top-k 后的 combine weights 归一化；
- 监控 top-k margin，避免 routing 选择过于随机或过于尖锐；
- 分阶段观察 router entropy 是否过早塌缩；
- 长上下文和 RL 阶段重新评估路由分布。

### 8.2 Shared-expert sink / expert sink：Inkling 的做法

常规 shared expert 通常不参与 top-k，直接作为 always-on FFN path 加到 routed experts 输出上。Inkling 的 shared-expert sink 稍微不同：shared experts 仍然是每个 token 都会计算的公共路径，但也参与 routing-score normalization，吸收一部分概率质量。

以 Inkling 的 256 routed experts、top-6 routed experts、2 个 shared experts 为例，shared experts 不挤占 routed top-k 名额，但会改变 routed weights 的归一化口径。它相当于给 common / easy / high-frequency token 一个稳定的 probability sink，同时让 routed experts 继续保持细粒度组合。

伪代码上可以这样理解：

```python
routed_scores = sigmoid(router_routed(x))     # [num_tokens, 256]
sink_scores = sigmoid(router_shared(x))       # [num_tokens, 2]

top6_ids = topk(routed_scores, k=6)           # shared experts are not candidates
denom = sum(gather(routed_scores, top6_ids)) + sum(sink_scores)
routed_weights = gather(routed_scores, top6_ids) / denom
sink_weights = sink_scores / denom

y = combine(routed_experts(top6_ids, x), routed_weights)
y = y + combine(shared_experts(x), sink_weights)
```

因此它适合作为 shared expert 的增强变体来评估，而不是直接替代基础 shared expert。实现时要注意训练和推理必须使用同一套 sink normalization；如果 serving 侧只对 top-6 routed scores 做 normalization，或者把 shared experts 当成额外 top-k candidate，就会和训练语义不一致。

### 8.3 Grouped routing 和 node-limited routing

专家数变大后，不可能只从模型质量角度选专家，还必须考虑设备拓扑。Grouped routing / node-limited routing 的目标是减少跨节点通信，让 token 尽量路由到更近的专家组。

代价是 router 自由度下降。建议只在 EP 通信明显成为瓶颈时启用，并做三组对比：

- 不限制 routing；
- expert group 限制；
- node/device-aware 限制。

评估指标不能只看 loss，还要看端到端 tokens/s、EP A2A 时间、专家负载 CV 和下游任务。

## 9. 推荐的三档配置

下面是用于启动实验的配置思路，不是固定答案。

| 目标 | 架构起点 | 训练起点 | 系统注意点 |
|---|---|---|---|
| 1B-80B 小激活/端侧/本地 MoE | 前 1 层 dense；32-128 experts，top-2/top-4/top-8，1 shared expert，可选部分层 MoE | aux/global aux baseline + loss-free 对照，router FP32；MobileMoE/Granite/OLMoE 这类小 active 配置应分层对照 | 关注专家驻留、推理 batch、量化、端侧缓存和小 batch kernel 利用率 |
| 100B-300B 通用 MoE | 前 1-2 层 dense；128-288 experts，top-8，fine-grained routed + shared expert | loss-free balance + 小 seq/global aux，no token dropping | EP 尽量留在节点内，GroupedGEMM/permute fusion 默认打开 |
| 500B-3T frontier MoE | 前 2-4 层 dense；256-896+ experts，top-8/top-16，LatentMoE 优先评估，zero-expert 后置 | loss-free/quantile/PID 类 balance，阶段化 router 监控，QAT/FP8/MXFP8 早规划 | DeepEP/Flex/HybridEP、静态 shape、EP overlap、PP/VPP、CP/MLA 联合设计 |

## 10. 最小实验矩阵

如果资源允许，可以做一个递进矩阵：

| 维度 | baseline | variant A | variant B |
|---|---|---|---|
| expert granularity | 128 experts top-8 | 256 experts top-8 | 256 experts top-4 |
| MoE layer placement | 前 1 层 dense | 前 2 层 dense | 前 3-4 层 dense |
| shared expert | none | 1 shared small | 1 shared medium |
| balance | aux/global aux | loss-free bias | loss-free + small seq aux |
| router score | softmax | sigmoid | sigmoid + group routing |
| latent | normal MoE | latent bottleneck small | latent bottleneck medium |
| adaptive | fixed top-k | zero-expert late stage | layer/token adaptive |

核心观测指标：

- validation loss 和 downstream；
- expert load CV；
- device load CV；
- router entropy；
- top-k margin；
- shared expert contribution；
- token drop rate；
- per-layer zero-expert rate；
- EP all-to-all 时间；
- tokens/s；
- end-to-end latency；
- post-training/RL 后路由分布漂移。

## 11. 常见风险

- 只报 total params，不报 active params 和 top-k。
- 只看训练 FLOPs，不看 EP 通信和端到端 latency。
- 用过强 aux loss 把专家训练成平均副本。
- 从第一层就 MoE 化，但没有检查 early-layer router/load 是否稳定。
- token dropping 没有按任务/语言/代码/数学分布分析。
- 小 batch 下调出的 balance 超参直接迁移到大 global batch。
- pretrain 路由稳定就以为 SFT/RL 也稳定。
- 只做 expert-level balance，不做 device/EP-level balance。
- 用动态 routing 但系统仍按静态固定 top-k 的假设优化。

## 12. 结论

可以归纳成一个工程决策：

**用前置 dense 层稳定底层表示，用 fine-grained experts 扩组合空间，用 shared expert 承担通用知识，用 loss-free balance 降低训练干扰，用 LatentMoE 降低专家路径的系统成本；如果还想进一步提高计算效率，再小心引入 zero-expert / adaptive routing。**

下一篇会把这个 recipe 落到 Megatron Core 源码：MoE layer 如何执行，router/gating、load-balance loss、expert bias、token dropping、dispatcher 和 experts 如何串起来。第四篇再单独展开 DeepEP/Flex Dispatcher/GroupedGEMM 等性能组件，第七篇会重点讨论 MoE RL 的训推不一致。

## 系列导航

上一篇：[第一篇：最新开放模型里的 MoE Common Taste]({% post_url 2026-07-01-moe-common-taste-open-models %})  
下一篇：[第三篇：Megatron Core MoE 源码剖析]({% post_url 2026-07-11-megatron-core-moe-source %})

## 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
- [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871)
- [Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models](https://arxiv.org/abs/2507.17702)
- [LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts](https://arxiv.org/abs/2601.18089)
- [LongCat-Flash Technical Report](https://arxiv.org/abs/2509.01322)
- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Kimi K3 API Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)
- [Inkling Model Card](https://thinkingmachines.ai/model-card/inkling/)
- [TML Inkling on vLLM: Day-0 Support with Optimized Performance](https://vllm.ai/blog/2026-07-15-inkling)
- [SGLang and Miles Add Day-0 Support for Inkling](https://www.lmsys.org/blog/2026-07-15-inkling-day0-support)
- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
