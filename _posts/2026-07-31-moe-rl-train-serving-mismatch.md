---
layout: post
title: "MoE我们喜欢你（七）：RL 训推不一致与路由对齐问题"
date: 2026-07-31
tags: [MoE, RL, Megatron, SGLang]
description: "以 Megatron learner 和 SGLang rollout 为例，解释 MoE RL 为什么不只要对齐 logprob，还要对齐每个 token 背后的 expert path。"
---

## 0. 写在前面：路线 A 生成，路线 B 更新

Dense 模型做 RL，也会遇到训推不一致：rollout 和 training 的 batch 不同，dtype、sampling kernel、权重同步节奏都可能不一样，最后表现为 logprob 或 policy distribution 对不上。

MoE 把这个问题再往前推了一层。它不只是生成 token，还会在每一层为每个 token 选择专家。一条 rollout，其实还隐含了一条 expert path：

| token | layer 1 experts | layer 2 experts | ... | layer L experts |
|---|---|---|---|---|
| `x_t` | top-k ids / weights | top-k ids / weights | ... | top-k ids / weights |

RL 里的 reward 是根据 rollout 文本算出来的。如果 learner 反向更新时，同一段 prompt/response 在训练引擎里走了另一条专家路径，就会出现一个更隐蔽的问题：

**reward 是路线 A 生成的，梯度却更新了路线 B。**

这才是 MoE RL 里比较麻烦的训推不一致。它不是普通数值误差，而是 policy 的隐含计算路径变了。

![MoE RL train-serving mismatch loop](/assets/img/blog/moe-we-like-you/07-moe-rl-train-serving-mismatch-loop.png)

*图 1：MoE RL 闭环里，SGLang rollout、reward/verifier、Megatron learner 和 checkpoint sync 之间会反复传递状态。router math、expert bias、token dropping、quantization 和 expert placement 都可能成为 mismatch 注入点。*

**Dense RL 主要在对齐 token 分布；MoE RL 还要对齐每个 token 背后的 expert path。**

## 1. 一个 token 例子：路径错了，credit 就错了

把上面的判断放到一个 token 上就更清楚了。假设某个 token 在 rollout 时，SGLang 选了 experts `{3, 17, 42, 88}`，这条 response 最后拿到高 reward。

![Reward on Path A, Gradient on Path B](/assets/img/blog/moe-we-like-you/07-reward-on-path-a-gradient-on-path-b.png)


*图 2：同一条样本在 rollout 和 learner 中如果走了不同 expert path，reward 和梯度的归因对象就会错位。图里只表达这个核心问题，具体 expert id 以正文例子为准。*

learner 用 Megatron 重新 forward 同一段 prompt/response，准备计算 logprob 和 policy gradient。只要 router dtype、expert bias、top-k renormalize 或 expert id mapping 有一点偏差，同一个 token 就可能变成 `{3, 17, 42, 91}`。

这个变化看起来很小，credit assignment 已经变了：88 参与了生成却没拿到正向 credit，91 没参与生成却被更新；combine weight 也变了，token logprob 不再对应 rollout path。若这种翻转集中在 math/code 关键 token 上，RL 会把能力更新推向错误专家；若高 reward 样本集中激活少数 experts，还会推高 load imbalance。

所以这里麻烦的不是 serving/training 有一点数值误差，而是行为策略和被更新策略可能在不同 expert path 上解释同一条样本。

## 2. 一个 MoE RL 闭环：rollout engine vs learner engine

实际系统里，大模型 RL 通常至少有两套引擎：

| 角色 | 常见引擎 | 做什么 |
|---|---|---|
| rollout / serving engine | SGLang、vLLM、自研 serving runtime | 用当前或滞后的 policy 生成 response，追求吞吐、低延迟、continuous batching |
| learner / training engine | Megatron、Megatron-Core、DeepSpeed、FSDP stack | 用 rollout 样本、logprob、advantage/reward 做反向更新 |

在 dense 模型里，两套系统边界主要是 policy distribution 的边界。到 MoE 里，这个边界会穿过 router、dispatcher、expert placement 和 serving kernel：

- 每个 token 选中哪些 experts；
- top-k weights 如何归一化；
- expert bias 是否参与 choice；
- shared expert 是否 fused 进 top-k；
- group-limited routing 是否使用相同分组；
- experts 在物理 GPU 上的 placement 是否一致；
- 训练时是否有 token dropping / capacity，推理时是否 dropless；
- rollout 量化是否改变 router 排序。

下面按三类看。

## 3. 不一致从哪里来：三类问题

第一类是 **router semantics mismatch**：同一份 router logits，两个 engine 到底应该怎样选专家。这里最容易错的是 score function、top-k、renormalize、expert bias、shared expert、group-limited routing 和 tie-breaking。

Megatron learner 侧的 router 往往不是一个纯 top-k 函数。它可能涉及 `moe_router_dtype`、`topk_routing_with_score_function()`、`expert_bias`、aux/z-loss、token dropping、router trace/replay 等训练逻辑。SGLang serving 侧则更关心从 router logits 快速得到 `topk_ids` 和 `topk_weights`，再走 fused MoE 或 grouped expert kernel。两个方向都合理，但一旦放进 RL 闭环，就必须确认它们对“同一份 logits”给出同一套 top-k 语义。

其中 expert bias 特别容易被低估。loss-free balance 里的 bias 通常存在 checkpoint 中，影响 top-k choice，但不一定参与 combine weight。如果 checkpoint 转换漏掉 bias、bias 符号反了、dtype 降低，或者 serving 把 bias 混进 combine weight，expert path 都会变。

第二类是 **runtime/state mismatch**。数学语义可能一致，但 runtime 状态把 path 改了。常见问题包括：

| 问题 | 为什么会影响 MoE RL |
|---|---|
| router dtype / 量化 | top-k margin 小的 token 容易被翻转 |
| token dropping / capacity | rollout dropless，learner drop，会让 reward path 被裁剪 |
| logical/physical expert id mapping | id 看起来都对，实际可能指向不同 physical slot |
| expert placement / EPLB | 不一定改变数学结果，但会改变 device load、尾延迟和部分 fused path |
| shared expert fuse 方式不同 | shared expert 是独立 path 还是 fused top-k slot，语义要对齐 |

第三类是 **RL system mismatch**。rollout 系统和 learner 系统本来就不在同一时间、同一 batch、同一请求分布上。RL rollout 往往用 serving engine，是因为它吞吐高；learner batch 则按训练 step 组织。continuous batching、prefix cache、chunked prefill、异步 checkpoint sync、replay buffer、serving quantization lag，都会让 rollout policy 和 training policy 拉开。

R3 论文还补了一刀：即便在同一个 Megatron 训练框架里，对同一批序列做两次 forward，MoE 输出概率也可能出现可测差异。问题不只是“SGLang vs Megatron”两个系统不同，MoE router 本身的敏感性也会放大 runtime 里的小扰动。

对应的处理方式也不同：router semantics 要对齐配置和实现；runtime/state 要做 probe 和 route audit；RL system mismatch 要靠 correction、filter、weight sync 和 staleness 控制。

## 4. 主流 RL 框架怎么处理：slime 和 verl

放到具体框架里看，slime、verl Rollout Correction、VeXact 和 [R3](https://arxiv.org/abs/2510.11370) 处理的是不同层面的 mismatch。

| 方法 | 主要解决 | 对 MoE 的缺口 |
|---|---|---|
| slime | Megatron training、SGLang rollout、Data Buffer、reward/verifier 的系统集成 | 不自动保证每个 token 的 expert path 一致 |
| verl Rollout Correction | 用 IS / RS 修正 rollout policy 和 training policy 的 logprob 偏差 | 修的是 token 分布，不知道偏差来自哪个 expert path |
| verl VeXact | 让 actor 和 rollout engine 尽量产生 bitwise-aligned logprob | 覆盖面取决于支持的 backend、模型和 MoE kernel |
| [R3 / routing replay](https://arxiv.org/abs/2510.11370) | 把 rollout 时的 top-k mask 带回 training 侧审计或回放 | 成本更高，适合诊断、ablation 或重点样本稳定化 |

slime 的价值在系统层：把 Megatron 和 SGLang 接成一条更明确的数据流，减少 checkpoint、tokenizer、chat template、sampling config 或 rollout id 被外部 glue code 接错的概率。但 MoE 还要单独问：SGLang 记录了哪些 `topk_ids` / `topk_weights`？Megatron 重算 route 后 overlap 如何？expert bias 和 expert id mapping 是否真的同步？

verl Rollout Correction 是统计层修正，适合处理异步 staleness、replay buffer、backend 精度差异带来的 off-policy 偏差。它能看到概率比值，却不一定知道这个比值为什么变了；在 MoE 里，原因可能是 router dtype，也可能是 expert bias、shared expert、量化或 id mapping。

VeXact 更接近实现层对齐：如果同一模型、同一输入、同一权重下的 rollout logprob 能 bitwise-aligned，很多 TIM 会从源头上变小。只是到 MoE 场景，最终仍要看具体 kernel 和 backend 是否覆盖了 router、dispatcher、expert path 这些细节。

所以，与其说某个框架“已经解决”MoE RL 训推不一致，不如说它们各自压住了一部分误差：slime 降低系统集成误差，verl 降低 token/logprob 层面的偏差，VeXact 提供实现层 baseline；MoE 特有的 expert path mismatch 仍然要单独记录、审计和过滤。

## 5. 怎么观测：只看 reward 不够

MoE RL 只看 reward/loss 不够。固定 probe 要先有：同一批 prompts 分别跑 Megatron 和 SGLang，比较 top-k overlap、top-1 flip、expert load、device load、router entropy 和量化前后的 route drift。

按 R3 论文的口径，还可以同时看 train-inference KL 和 extreme-token ratio：前者看整体 policy 偏移，后者看“少数 token 概率差得离谱”的尾部风险。MoE RL 崩的时候，尾部 token 往往比平均 reward 更早报警。

在线 rollout 也要抽样留一点 route metadata：`topk_ids`、`topk_weights` 或 margin、per-request expert hotness、overflow/drop stats，以及 logical-to-physical expert mapping。这样 reward、expert load 和 p95/p99 latency 才能放到同一张图里看。

如果日志成本太高，最低限度也要对采样 batch 或 probe prompts 记录 top-k ids。没有 route metadata，MoE RL 的很多问题只能看到“训练不稳”，看不到到底是哪条 path 变了。

## 6. R3：把 route path 也当成 rollout 数据

仅靠“训练和推理对齐同一套 MoE 逻辑”还不够。真实 RL 系统里，rollout engine 和 learner engine 往往就是两套 runtime：SGLang 负责高吞吐生成，Megatron 负责反向训练。差异很难完全消掉，所以更实用的做法是把 routing metadata 纳入数据闭环。

这里说的 replay 不是泛泛的 data replay，而是 **routing replay**。这里的 R3 指的是 [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)。它关心的不是简单复用旧样本，而是把 inference 侧的 top-k mask 带回 training 侧，减少训练和推理 router 行为的偏差。

![R3 rollout routing replay](/assets/img/blog/moe-we-like-you/07-r3-rollout-routing-replay.png)

*图 3：R3 论文中的 Rollout Routing Replay 示意。左侧展示把 inference engine 的 route selection 回放到 old/update policy 的 training engine；右侧展示 R3 前后的 train/inference probability 对齐和验证分数变化。图源：[Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)。*

R3 做的事很具体：rollout 时不仅保存 prompt、response、logprob、reward，还保存每层 router 选出的 top-k mask。训练侧在 recompute old policy 和 update policy 时复用这个 inference mask，让两次训练 forward 都沿着 rollout 时的专家集合走。

一个容易写错的细节是：R3 不是把 inference 侧的 gate weight 原样拿来训练。它复用的是 top-k mask；training 侧仍然用自己的 router logits，在这个 mask 内重新 softmax 得到 combine weight。这样 expert selection 对齐了，router 的梯度流也还在。

记录粒度也不用一开始拉满。更现实的起点，是对固定 probe 和抽样 rollout 记录 `topk_ids`、`topk_weights`、margin 或 routing distribution。很多 mismatch 不是所有 token 都翻，而是集中在 router score 很接近的边界 token 上；只记录 top-k ids 能知道翻没翻，记录 margin 才能知道这次翻转是不是本来就在不稳定边界。

Megatron 里的 `router_replay.py` 已经提供了类似能力：可以 record 某次 forward 的 top-k indices，也可以在 forward/backward recompute 时使用给定 top-k indices。对应到 RL，就是把 rollout engine 记录的 route path 带回 learner，让 learner 在实验中沿着 rollout 路径计算 logprob 和梯度。

R3 论文还提到 multi-turn / agent 场景：如果 serving 侧用了 prefix KV cache，同一段 prefix 的 routing mask 也可以跟着 cache 复用，避免为了补 route metadata 重新 prefill。

但 full router replay 不适合作为第一默认项。长 CoT RL 记录完整 route metadata 成本很高，rollout route 又对应生成时刻的 checkpoint；如果 learner 已经更新了权重，长期强制旧 route 还可能把模型本来应该学习的新 router 边界固定住。R3 最有价值的地方，不是“永远回放旧 path”，而是提醒我们：route path 也是 rollout 数据的一部分。

## 7. 实际怎么稳定 MoE RL 训练

实际训练时，不建议一开始就上 full routing replay。先守住路由语义：RL 开始前，freeze 或放慢会直接改变路由的状态，router/gate 保高精度，score function、top-k、renormalize、expert bias、shared expert 和 expert id mapping 逐项对齐。RL 阶段尽量 dropless；如果必须用 capacity 或 drop，overflow/drop stats 要和 reward、expert load 放在一起看。

然后再处理样本。某条 rollout 在 learner 侧重算后 top-k overlap 很低，或者关键 token 的 top-1 flip 很高，就不要让它高权重更新 expert/router。更稳的做法是 route consistency filter、downweight，或者先放进 diagnostics buffer。full router replay 留给 ablation：它适合回答“如果强制沿着 rollout path 学，问题是不是会消失”，但不适合一开始就当成主路径。

## 8. 结论

MoE RL 麻烦的地方，不只是 router 会不会漂移，而是 rollout 和 learner 可能在不同专家路径上解释同一条样本。SFT/RL 阶段少动 router 相关状态，训练和 rollout 的 MoE 语义逐项对齐，route-level probe 和 reward、load、latency 一起看；出了明显 mismatch，先把样本权重收住。

**MoE RL 里，reward 不能只和 token 对齐，也要和当时走过的专家路径对齐。**

## 系列导航

上一篇：[第六篇：端侧/本地小 active MoE 的挑战、推理瓶颈与设计思路]({% post_url 2026-07-26-edge-moe-inference-design %})  

## 参考资料

- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-LM `router.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py)
- [Megatron-LM `moe_utils.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py)
- [Megatron-LM `token_dispatcher.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py)
- [Megatron-LM `router_trace.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router_trace.py)
- [Megatron-LM `router_replay.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router_replay.py)
- [SGLang `topk.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/topk.py)
- [SGLang `fused_moe_native.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/fused_moe_native.py)
- [slime Documentation](https://thudm.github.io/slime/)
- [THUDM/slime](https://github.com/THUDM/slime)
- [verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html)
- [verl-project/vexact](https://github.com/verl-project/vexact)
- [Diagnosing Training Inference Mismatch in LLM Reinforcement Learning](https://arxiv.org/abs/2605.14220)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)
