---
layout: post
title: "MoE我们喜欢你（七）：RL 训推不一致与路由对齐问题"
series: moe-we-like-you
series_order: 7
last_modified_at: 2026-10-26
date: 2026-07-31
tags: [MoE, RL, Megatron, SGLang]
description: "以 Megatron learner 和 SGLang rollout 为例，解释 MoE RL 为什么不只要对齐 logprob，还要对齐每个 token 背后的 expert path。"
---

## 0. 先区分后端偏差与策略更新

MoE RL 通常由 serving engine 生成 response，再由 learner 重算 logprob 并更新参数。两套引擎既可能存在实现偏差，也可能因为权重同步滞后而使用不同版本的策略。这两件事需要分开排查。

本文先比较同一 checkpoint、同一 token 序列在两套引擎中的输出。此时 router dtype、expert bias、归一化或数值扰动可能改变专家选择，放大 token probability 的偏差。随后再讨论新旧策略之间的差异，以及 routing replay 在其中改变了什么。

learner 更新权重之后，专家选择发生变化可以是正常学习结果。route overlap 因此是定位问题的线索，不能单独作为“更新错误”的判据。

![MoE RL train-serving mismatch loop](/assets/img/blog/moe-we-like-you/07-moe-rl-train-serving-mismatch-loop.png)
*图 1：rollout engine、learner 和 checkpoint sync 共同决定实际使用的策略。诊断时先固定 checkpoint、输入和路由配置，再观察引擎间的偏差。*

## 1. 同一个 checkpoint，为什么会走出另一条路径

为了与图中层级对应，先用 top-1 简化：某个 token 在四层分别走专家 `3、17、88、42`。learner 用同一 checkpoint 重算同一段文本，第三层却选中 `91`。真实 top-k 模型也会出现专家集合部分替换的情况。

![Reward on Path A, Gradient on Path B](/assets/img/blog/moe-we-like-you/07-reward-on-path-a-gradient-on-path-b.png)
*图 2：固定 checkpoint 后，第三层的 expert 88 与 91 仍因引擎差异发生替换。图示仅突出一条分支；reward 属于整条生成样本，不能直接视为单个 expert 的监督标签。*

如果两个专家输出不同，这个替换可能使 learner 算出的 token probability 偏离实际 rollout policy。RL 使用的概率比值就可能混入后端误差，影响优化的稳定性。反过来，若两个专家输出很接近，一次翻转未必造成显著影响，因此还需要检查 logprob/KL 和路由 margin。

这个例子关注的是同权重条件下不应有的额外偏差。策略更新后的路由变化、异步 rollout 的权重滞后，则要放到新旧策略与 off-policy correction 的问题里分析。

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
- logical expert id 是否映射到正确权重，副本是否同步；物理 GPU 位置可以不同；
- 训练时是否有 token dropping / capacity，推理时是否 dropless；
- rollout 量化是否改变 router 排序。

下面按三类看。

## 3. 不一致从哪里来：三类问题

第一类是 **router semantics mismatch**：同一份 router logits，两个 engine 到底应该怎样选专家。这里最容易错的是 score function、top-k、renormalize、expert bias、shared expert、group-limited routing 和 tie-breaking。

当前 forward 的选择与权重由 score function、choice-only bias、分组、top-k、drop 和归一化决定。把相同 logits 交给两套实现，可以单独测试这层语义。dtype/casting 则属于下一类数值执行问题。

aux/z-loss 对训练梯度的影响另行区分。第三篇采用的 Megatron 版本通过 `MoEAuxLossAutoScaler` 附加这类梯度，forward 原样返回 logits/probs；相同 checkpoint 下，附加 loss 本身不会改写该次 top-k。router weights 或控制器状态更新后引起的变化，归到新旧策略比较。

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

R3 论文还观察到：即便在同一个 Megatron 训练框架里，对同一批序列做两次 forward，MoE 输出概率也可能出现可测差异。问题不只是“SGLang vs Megatron”两个系统不同，路由选择的离散性也可能放大 runtime 里的小扰动。

对应的处理方式也不同：router semantics 要对齐配置和实现；runtime/state 要做 probe 和 route audit；RL system mismatch 要靠 correction、filter、weight sync 和 staleness 控制。

## 4. 框架与修正方法分别处理哪一层

| 方法或系统 | 处理的问题 | 使用时需确认 |
|---|---|---|
| slime | 连接 Megatron training、SGLang rollout、数据与 reward 流水线 | checkpoint、模板、采样配置和路由状态是否正确转换 |
| verl Rollout Correction | 用重要性采样（IS）/拒绝采样（RS）等处理行为策略与训练策略的概率差异 | 实际使用哪两个概率、是否包含 staleness 与后端误差 |
| VeXact | 在支持的模型和后端上对齐引擎输出 | 当前模型、量化和 MoE kernel 是否覆盖 |
| R3 / routing replay | 回放 rollout 的 expert selection，减少路由不一致 | mask 的记录、缓存和训练侧使用方式 |

系统集成、概率修正和路由回放可以组合。比较时应固定各自处理的误差来源，分别观察 token probability 与 expert selection；仅凭用了某个框架，无法判断后端已经对齐。

## 5. 怎么观测：只看 reward 不够

固定 probe 时，先用同一引擎生成一份 token 序列，再让 Megatron 和 SGLang 对这份相同序列计算 logprob 和 routing trace。若分别采样生成不同文本，就不能把逐 token 差异直接归因于后端。诊断项包括 top-k overlap、top-1 flip、expert load、device load、router entropy 和量化前后的变化。

按 R3 论文的口径，还可以同时看 train-inference KL 和 extreme-token ratio：前者看整体 policy 偏移，后者看“少数 token 概率差得离谱”的尾部风险。它们分别描述整体偏移和概率比值的尾部，适合与 reward 曲线一起观察。

在线 rollout 也要抽样留一点 route metadata：`topk_ids`、`topk_weights` 或 margin、per-request expert hotness、overflow/drop stats，以及 logical-to-physical expert mapping。这样 reward、expert load 和 p95/p99 latency 才能放到同一张图里看。

如果日志成本太高，最低限度也要对采样 batch 或 probe prompts 记录 top-k ids。没有 route metadata，MoE RL 的很多问题只能看到“训练不稳”，看不到到底是哪条 path 变了。

## 6. R3：回放选择，重新计算混合权重

[R3](https://arxiv.org/abs/2510.11370) 在 rollout 时保存每层、每个 token 的 top-k mask，训练侧计算 old policy 和 update policy 时复用该 mask。combine weights 仍由训练侧当前 logits 计算。按论文采用的 softmax gate，可以写成：

```text
mask = rollout_topk_mask
weights = softmax(train_logits restricted to mask)
output = sum(weights[e] * expert[e](hidden) for e in mask)
```

固定的是分支选择，权重与分支内计算仍参与求导。用于 sigmoid 等 router 时，也应沿用该模型的打分与归一化语义。

![R3 rollout routing replay](/assets/img/blog/moe-we-like-you/07-r3-rollout-routing-replay.png)

*图 3：R3 论文中的 Rollout Routing Replay 示意。左侧展示把 inference engine 的 route selection 回放到 old/update policy 的 training engine；右侧展示 R3 前后的 train/inference probability 对齐和验证分数变化。图源：[Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370)。*

诊断只需抽样 trace；完整 replay 则需要被回放各层、各 token 的 mask。Megatron 的 `router_replay.py` 提供 record/replay indices 的接口，接入 RL 时还需将 serving 记录的逻辑 expert id 正确传回 learner。多轮对话中，prefix mask 可以与 KV cache 一起复用，减少重复 prefill。

多个 update step 使用同一份 rollout 时，这个 mask 会约束训练中的分支选择。实验应同时记录稳定性、质量以及 metadata 的传输和缓存成本，判断回放是否适合当前训练配置。

## 7. 按误差来源安排实验

先做同 checkpoint 的 backend probe，对齐 score function、top-k、renormalize、bias、shared expert 和 logical-to-physical id mapping，保留 router 精度对照。若有 capacity/drop，单独比较它对同一 token 序列的影响。物理 placement 可以变化，但对应权重和副本状态必须正确。

然后记录权重同步时间与 rollout staleness，比较新旧策略的概率差异。冻结 expert bias update 可以减少一个变化源，但是否冻结 router weights，应由训练方案和消融决定。

若问题集中在路由翻转，可在相同训练预算下比较 replay 与非 replay，观察 KL、概率比值尾部、质量和吞吐。对低 overlap 样本做 filter/downweight 只能作为另一组实验：它可能系统性移除小 margin 或特定任务样本，需要报告被过滤比例及样本分布，不能单凭 route overlap 判定数据无效。

## 8. 小结

先回答“同一份策略在两套引擎里是否计算一致”，再回答“策略更新后变化是否受控”。routing trace 能帮助定位概率偏差，R3 则提供了约束专家选择的一种干预方式。二者都需要与 logprob、质量和系统成本联合评价。

## 系列导航

上一篇：[第六篇：端侧 Active MoE 的推理瓶颈与设计取舍]({% post_url 2026-07-26-edge-moe-inference-design %})

下一篇：[第八篇：稀疏计算与参数复用的新方向]({% post_url 2026-07-23-moe-future-sparse-architecture %})

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
