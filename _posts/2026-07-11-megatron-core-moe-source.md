---
layout: post
title: "MoE我们喜欢你（三）：Megatron Core MoE 源码路径解析"
date: 2026-07-11
tags: [MoE, Megatron, Pretraining]
description: "结合 Megatron Core 源码，拆解 MoE layer、router/gating、load-balance loss、expert bias、token dropping、dispatcher 和 experts 的实现路径。"
---


## 0. 写在前面

MoE 论文和模型报告里常见的词包括 top-k routing、shared expert、loss-free balance、dropless、EP、GroupedGEMM、DeepEP。落到训练框架里，这些词会变成一条更具体的执行链：

**hidden states 先经过 router 选专家，再被 token dispatcher 重排和跨 rank 发送，随后进入本地 experts 做 GroupedGEMM，最后再 combine 回原 token 顺序。**

这篇基于 Megatron-LM `bb5647a` 附近的 `megatron/core/transformer/moe/` 源码整理。目的不是逐行讲源码，而是把几个关键部件的实现口径讲清楚：

- MoE layer 的主执行路径在哪里；
- gating 和 top-k routing 如何产生 `routing_map` 和 `probs`；
- aux / seq aux / global aux / z-loss 分别在哪一步接入；
- loss-free expert bias 在训练时如何统计、如何更新；
- token dropping / capacity 是怎么发生的；
- allgather / alltoall / flex dispatcher 各自承担什么；
- latent MoE 和 shared expert 在 Megatron 里怎么接入。

从实现结构看，Megatron Core 里的 MoE 不是一个单独 layer，而是一组 router、dispatcher、expert kernel、balance controller 和并行通信协议拼起来的系统。

![Megatron Core MoE 执行链](/assets/img/blog/moe-we-like-you/03-megatron-core-moe-execution-chain.png)


*图 1：读 Megatron Core MoE 源码时，可以先按 `route -> dispatch -> experts -> combine` 这条主路径定位，再分别展开 router loss、expert bias、token capacity、shared/latent expert 等旁路状态。*

后面的代码块都是按 Megatron-LM `bb5647a` 附近源码做的删节版，只保留关键分支和状态流转。这样比整段源码更容易看出设计意图，也避免把正文变成逐行注释。

## 1. 源码地图

先把关键文件放在一张表里。

| 文件 | 主要角色 |
|---|---|
| `moe_layer.py` | MoE layer 主入口，负责 route、preprocess、dispatch、expert compute、combine |
| `router.py` | `Router` / `TopKRouter` / `InferenceTopKRouter`，实现 gating、routing、aux loss、expert bias 统计 |
| `moe_utils.py` | top-k routing、load-balance loss、z-loss、token dropping、permute/unpermute、expert bias update |
| `token_dispatcher.py` | allgather、alltoall、flex/hybrid EP token dispatcher |
| `experts.py` | `TEGroupedMLP`、`SequentialMLP`、inference grouped MLP |
| `shared_experts.py` | shared expert MLP、overlap 和 fused shared expert path |
| `router_replay.py` | 记录/回放 routing 决策，用于调试或一致性分析 |
| `router_trace.py` | 捕获 top-k ids、weights、logits 等 routing trace |

这几个文件的依赖关系大致是：

| 阶段 | 入口 | 产物 |
|---|---|---|
| route | `MoELayer.route()` -> `TopKRouter.forward()` | `probs`, `routing_map` |
| preprocess | `MoELayer.preprocess()` | latent projection、dispatcher metadata |
| dispatch | `token_dispatcher.token_dispatch()` | expert-major tokens、tokens per expert |
| expert compute | `TEGroupedMLP.forward()` | local expert outputs |
| combine | `token_dispatcher.token_combine()` + `combine_postprocess()` | 回到原 token 顺序的 hidden states |

理解 Megatron MoE，建议按这条链读源码，而不是从某一个 flag 开始看。

## 2. MoE layer 主路径：route -> dispatch -> experts -> combine

`moe_layer.py` 里的 `MoELayer` 是最适合开始的地方。

初始化阶段主要做几件事：

| 组件 | 源码位置 | 说明 |
|---|---|---|
| router | `self.router = ...` | 选择 top-k experts，生成 routing map |
| latent projection | `fc1_latent_proj` / `fc2_latent_proj` | `moe_latent_size` 打开后，把 routed expert path 放到 latent dim |
| token dispatcher | `MoEAllGatherTokenDispatcher` / `MoEAlltoAllTokenDispatcher` / `MoEFlexTokenDispatcher` | 决定 token 如何重排、通信、送到本地 experts |
| experts | `self.experts` | 通常是 `TEGroupedMLP` 或其它 expert MLP |
| shared experts | `self.shared_experts` | always-on 或 overlap 的 shared expert path |

forward 的逻辑可以概括成：

| 步骤 | 做什么 | 关键状态 |
|---|---|---|
| route | router 读 hidden states，输出每个 token 的 top-k experts | `probs`, `routing_map` |
| preprocess | latent projection、dispatcher 预处理、统计 tokens per expert | dispatcher metadata |
| dispatch | token 按 expert-major 顺序排列，并跨 EP rank 发送 | permuted tokens |
| shared experts | 可与 routed path overlap | shared expert output |
| routed experts | 本地 experts 做 MLP/GroupedGEMM | expert output |
| combine | all-to-all 回传、unpermute、按 router probs 加权合并 | final hidden states |

这条路径说明，MoE 的瓶颈不一定在 expert MLP。router、permute、all-to-all、combine、shared expert overlap、latent projection 都可能成为实际吞吐里的关键段。

先把完整 forward 压成一段伪代码：

```python
hidden_in = hidden
probs, routing_map = router(hidden)
hidden, probs = dispatch_preprocess(hidden, routing_map, probs)
permuted_hidden, permuted_probs = dispatch(hidden, probs)
expert_out = experts(permuted_hidden, permuted_probs)
hidden = combine(expert_out, probs)
hidden = shared_expert(hidden_in) + hidden if use_shared_expert else hidden
```

可以先看 `MoELayer` 的删节版主路径。注意：router 只负责产生 `probs` 和 `routing_map`，真正把 token 发到各个 expert rank 的是 dispatcher。

```python
class MoELayer(BaseMoELayer):
    def route(self, hidden_states, padding_mask=None):
        probs, routing_map = self.router(hidden_states, padding_mask)
        return probs, routing_map

    def preprocess(self, hidden_states, probs, routing_map):
        if self.config.moe_latent_size:
            hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs

    def dispatch(self, hidden_states, probs):
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def routed_experts_compute(self, hidden_states, probs):
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        return self.token_dispatcher.combine_preprocess(expert_output), mlp_bias

    def combine(self, output):
        return self.token_dispatcher.token_combine(output)
```

## 3. Gating：router logits 从哪里来

`router.py` 里最基础的抽象是 `Router.gating()`。它做的事情直接：把输入 hidden states 喂给一个 linear gate，得到每个 token 对每个 expert 的 logits。

关键代码不长，但信息量很大：router 计算可以强制转到 FP32/FP64，再通过 `router_gating_linear` 得到 logits。

```python
def gating(self, input: torch.Tensor):
    if self.weight.device.type == "cpu":
        self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
    if self.bias is not None and self.bias.device.type == "cpu":
        self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

    router_dtype = input.dtype
    if self.config.moe_router_dtype == "fp32":
        router_dtype = torch.float32
    elif self.config.moe_router_dtype == "fp64":
        router_dtype = torch.float64

    logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
    return logits
```

几个实现细节值得注意。

第一，router dtype 是显式配置的。Megatron 会把 router 输入转到 `moe_router_dtype`，常见选择是 FP32 或 FP64。大专家数、sigmoid/top-k、expert bias 混用时，router logits 的微小差异可能改变 top-k，所以 router 精度通常要比普通 FFN 更谨慎。

第二，`router_gating_linear` 可以走 fused autograd function。MoE router 是小算子，但每层每步都跑，kernel launch 和精度都重要，所以 Megatron 把 router fusion 放进性能优化路径。

第三，`Router` 本身只定义接口，真正的 routing 逻辑在 `TopKRouter` 和 `InferenceTopKRouter`。

| 类 | 用途 |
|---|---|
| `Router` | 基类，负责 gating linear 和 dtype |
| `TopKRouter` | 训练路径，包含 z-loss、load-balance loss、token dropping、expert bias 统计 |
| `InferenceTopKRouter` | 推理路径，去掉训练侧 aux loss、z-loss、token dropping 和 bias update 等开销 |

这一点对第七篇很重要：训练和推理不是同一条 router path。即使权重一样，训练 runtime 和 serving runtime 也可能因为 router dtype、bias 处理、renormalize、top-k tie-breaking 而产生不一致。

## 4. TopKRouter：从 logits 到 routing_map

`TopKRouter.routing()` 是训练时最核心的函数。它大致做下面几件事：

| 顺序 | 逻辑 | 相关函数 |
|---|---|---|
| 1 | flatten logits，处理 padding mask | `routing()` |
| 2 | 训练时可附加 router z-loss | `apply_z_loss()` / `z_loss_func()` |
| 3 | 根据 balance 类型选择 routing 方法 | `sinkhorn_load_balancing()` / `quantile_balancing()` / `topk_routing_with_score_function()` |
| 4 | 如果配置 capacity factor，应用 token dropping | `apply_router_token_dropping()` |
| 5 | 训练时附加 aux / seq aux / global aux loss | `_apply_aux_loss()` 等 |
| 6 | 统计本 batch 的 expert load，用于 expert bias 更新 | `_apply_expert_bias()` |

最终输出两个东西：

| 输出 | 含义 |
|---|---|
| `probs` | token 对选中 experts 的 combine weights |
| `routing_map` | token-expert 的布尔/稀疏选择矩阵 |

`routing_map` 是后面 dispatcher 的核心输入。dispatcher 不关心论文里叫 top-8 还是 top-6，它关心的是哪些 token 要发到哪些 expert 上。

把 `TopKRouter.routing()` 压缩以后，主线可以概括为：先算 routing，再可选 token dropping，再把训练侧 aux loss 和 expert bias 统计挂上去。

```python
def routing(self, logits, padding_mask=None):
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1)

    logits = self.apply_z_loss(logits, padding_mask=padding_mask)

    if self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "quantile_balancing":
        probs, routing_map = self.quantile_balancing(logits)
    else:
        probs, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
        )

    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        probs = self._apply_aux_loss(...)
        probs = self._apply_seq_aux_loss(...)
        probs = self._apply_global_aux_loss(...)

    self._apply_expert_bias(routing_map, padding_mask=padding_mask)
    return probs, routing_map
```

## 5. Score function：softmax、sigmoid、sqrtsoftplus

Megatron 的 `topk_routing_with_score_function()` 支持多种 score function，包括 softmax、sigmoid 和 sqrtsoftplus。这里的关键不是名字，而是“选择专家的分数”和“合并输出的权重”是否一致。

常见口径如下：

| 口径 | 选择专家 | combine weight |
|---|---|---|
| softmax | 对所有 experts softmax 后 top-k | top-k 概率，可 renormalize |
| sigmoid | 每个 expert 独立 sigmoid 后 top-k | top-k 分数再 normalization |
| expert bias | 用 score + bias 参与选择 | combine weight 通常仍来自原 score |
| group-limited routing | 先选 expert group，再在 group 内选 top-k | group 内 top-k weights |

expert bias 的 choice-only 语义很关键。Megatron 在 `moe_utils.py` 中会把 `expert_bias` 加到 `scores_for_routing` 上参与 top-k 选择，但输出权重仍然来自原始 scores。这和 DeepSeek-style loss-free balance 的设计目标一致：bias 用来控制负载，不直接改变专家输出的 mixture weight。

这段实现是理解 loss-free balance 和 serving 对齐的关键。对于 sigmoid / sqrtsoftplus，bias 参与的是 `top_indices` 的选择，`scores` 会重新从原始 score 里 gather 出来，再归一化成 combine weights。

```python
if score_function == "softmax":
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
    else:
        scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)

elif score_function in ("sigmoid", "sqrtsoftplus"):
    if score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
    else:
        scores = torch.nn.functional.softplus(logits.float()).sqrt()

    if expert_bias is not None:
        scores_for_routing = scores + expert_bias.float()
        _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
        scores = torch.gather(scores, dim=1, index=top_indices)
    else:
        scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)

    probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
```

如果 serving runtime 把 bias 直接混进 combine weights，或者 renormalize 口径不同，就会出现训练-推理不一致。

## 6. Load balance：aux、seq aux、global aux、z-loss 和 QB

Megatron 同时支持多种 router 相关 loss。它们听起来相近，但目的不同。

| 机制 | 源码入口 | 解决什么 |
|---|---|---|
| aux loss | `_apply_aux_loss()` / `switch_load_balancing_loss_func()` | micro-batch 粒度的 expert load balance |
| seq aux loss | `_apply_seq_aux_loss()` | 以 sequence 为统计单位，让每条样本内部更平滑 |
| global aux loss | `_apply_global_aux_loss()` | 跨 micro-batch / global 统计更稳定 |
| z-loss | `apply_z_loss()` / `z_loss_func()` | 抑制 router logits 过大，提升 router 数值稳定性 |
| sinkhorn | `sinkhorn_load_balancing()` | 用近似 assignment 的方式做均衡路由 |
| quantile balancing | `quantile_balancing()` | 用 dual / quantile 状态调节专家选择，禁用 aux loss |

`switch_load_balancing_loss_func()` 的核心形式可以理解为 `num_experts * sum(f_i * P_i)`：`f_i` 是 tokens 实际分配到 expert 的比例，`P_i` 是 router 概率质量。它想惩罚“概率给很多但实际不均”或“实际集中到少数专家”的情况。

但从 pretraining recipe 看，aux loss 不应越强越好。它能救 router collapse，也可能干扰主 LM objective。现在更稳的用法是：loss-free expert bias 做主 balance，aux/seq/global loss 做轻量稳定器。

Quantile Balancing 是另一条路：它不用 aux loss 接管主目标，而是在 router 里维护 `qb_beta`。当前 batch routing 用的是 `scores - qb_beta`，随后根据每个 expert 的分位数阈值累计下一轮 `qb_beta`。

Megatron 里的实现要点很集中：

| 状态/函数 | 作用 |
|---|---|
| `moe_router_load_balancing_type="quantile_balancing"` | 选择 QB 路由，要求 aux loss coeff 为 0 |
| `qb_beta` | persistent buffer，每个 expert 一个动态价格 |
| `qb_beta_accum` / `qb_beta_count` | 非 persistent buffer，累计 microbatch quantile |
| `qb_dual_update()` | 用 `scores - beta` 选 top-k，并估计每列 quantile |
| `topk_routing_with_score_function(..., precomputed_indices=indices)` | QB 只决定 top-k ids，combine weights 仍按普通 score function 计算 |

伪代码大概是：

```python
indices, beta_local = qb_dual_update(scores, topk, qb_beta)
qb_beta_accum += beta_local
probs, routing_map = score_function(scores, precomputed_indices=indices)
```

这个设计和 choice-only expert bias 的精神类似：负载控制影响“选谁”，但不直接把控制器状态混进 combine weights。区别在于普通 expert bias 往往按实际 load 加减一步，QB 则用每个 expert 的分位数阈值估计下一轮价格。

## 7. Expert bias：loss-free balance 在代码里是什么状态

`TopKRouter.__init__()` 里如果 `moe_router_enable_expert_bias` 打开，会注册两个状态：

| 状态 | persistent | 用途 |
|---|---:|---|
| `expert_bias` | 是 | 保存每个 expert 的 routing bias，参与 checkpoint |
| `local_tokens_per_expert` | 否 | 当前 step / micro-batch 统计的 expert token count |

几个细节很重要。

第一，`expert_bias` 会被保持为 FP32。源码里有 `_maintain_float32_expert_bias()`，原因很明确：低精度 bias 更新会放大 routing error。

第二，`_apply_expert_bias()` 只在 `torch.is_grad_enabled()` 时累计 `local_tokens_per_expert`。也就是说，它是训练路径里的统计状态，推理路径不会继续更新。

第三，真正更新 bias 的逻辑在 `moe_utils.get_updated_expert_bias()`。具体规则是：如果某个 expert 负载高于均值，就沿一个方向调 bias；如果低于均值，就反向调。更新幅度由 `moe_router_bias_update_rate` 控制。

这里可以把 expert bias 当成一个轻量的负载控制器：

| 专家状态 | bias 行为 | 目标 |
|---|---|---|
| 过载 | 降低被选中概率 | 把 token 推给其它专家 |
| 欠载 | 提高被选中概率 | 吸引更多 token |
| 接近均衡 | 小幅或不更新 | 保持稳定 |

这也解释了为什么后训练阶段通常 freeze bias update：SFT/RL 的 batch 分布太窄，继续更新控制器，等于让专家负载按小数据重新定标。

代码里也能看出它为什么更像 controller state，而不是普通反向传播参数。`_apply_expert_bias()` 只累计 token count；真正的 bias 更新按专家负载和平均负载的符号方向走一步。

```python
def _apply_expert_bias(self, routing_map, padding_mask=None):
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            if padding_mask is not None:
                routing_map = routing_map & (~padding_mask)
            self.local_tokens_per_expert += routing_map.sum(dim=0)


def get_updated_expert_bias(tokens_per_expert, expert_bias, update_rate, group):
    with torch.no_grad():
        torch.distributed.all_reduce(tokens_per_expert, group=group)
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True)
        average_tokens = average_tokens / tokens_per_expert.shape[-1]
        offset = average_tokens - tokens_per_expert
        return expert_bias + torch.sign(offset) * update_rate
```

## 8. Token dropping：capacity factor 什么时候会丢 token

MoE 里 dropless 是常见目标，但框架里通常仍保留 capacity / drop 逻辑。Megatron 的 token dropping 主要由 `moe_expert_capacity_factor` 和 `moe_pad_expert_input_to_capacity` 控制。

在 `TopKRouter.routing()` 中，如果 `moe_expert_capacity_factor` 不为空，会调用 `apply_router_token_dropping()`。它会根据每个 expert 的 capacity 裁掉超出部分的 token routing，再让后续 dispatcher 只处理保留下来的 token。

删节版实现如下。`drop_policy="probs"` 时，每个 expert 保留 routing probability 最高的前 `capacity` 个 token；`drop_policy="position"` 时，则按 routing_map 的位置保留。

```python
def apply_router_token_dropping(
    routing_probs,
    routing_map,
    router_topk,
    capacity_factor,
    drop_policy="probs",
    pad_to_capacity=False,
):
    num_tokens, num_experts = routing_probs.shape
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    if expert_capacity > num_tokens:
        capacity_mask = torch.ones_like(routing_probs).bool()
    elif drop_policy == "probs":
        _, capacity_indices = torch.topk(
            routing_probs, k=expert_capacity, dim=0, sorted=False
        )
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
    elif drop_policy == "position":
        _, capacity_indices = torch.topk(
            routing_map.int(), k=expert_capacity, dim=0, sorted=False
        )
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()

    final_map = capacity_mask if pad_to_capacity else routing_map & capacity_mask
    final_probs = routing_probs * final_map
    return final_probs, final_map
```

在 all-to-all dispatcher 里，`drop_and_pad` 又会影响 buffer shape：

| 模式 | 行为 | 影响 |
|---|---|---|
| dropless | output size 通常是 `num_tokens * topk` | 不丢 token，但 shape 可能随路由动态变化 |
| drop | 超过 expert capacity 的 token 被裁掉 | 稳定显存/通信，但会改变训练信号 |
| drop-and-pad | 丢掉超 capacity token，并 pad 到固定 capacity | 更适合静态 shape / CUDA Graph，但会浪费 padding compute |
| expert-rank capacity | 给每个 EP rank 设置 upper bound | 控制 rank-level overflow 和 HybridEP buffer |

训练 recipe 上，通常建议先 dropless 跑稳。token drop 适合作为保护机制或极端内存约束下的选择，不应该在还没搞清楚 router/load 之前就打开。

第七篇里还会讲：如果训练时用了 capacity/drop，而 serving 时是 dropless，这本身就是一种训推不一致。

## 9. Token dispatcher：allgather、alltoall、flex

Megatron 的 token dispatcher 负责把 token 从原始 sequence-major layout 变成 expert-major layout，并在 EP ranks 之间通信。

| Dispatcher | 适用口径 | 主要特点 |
|---|---|---|
| `MoEAllGatherTokenDispatcher` | 小规模或简单 correctness path | gather 后本地筛 token，逻辑直观但扩展性弱 |
| `MoEAlltoAllTokenDispatcher` | 常规 EP 训练 | 先 permute，再 all-to-all，experts 计算后再 combine |
| `MoEFlexTokenDispatcher` | DeepEP / HybridEP / 高性能路径 | 把 dispatch/combine、permute、通信、静态 buffer 做更深融合 |

all-to-all 路径可以拆成：

| 阶段 | 做什么 |
|---|---|
| preprocess | 根据 `routing_map` 计算 `tokens_per_expert`、input/output splits |
| token permutation | 把 token 重排成 expert-major |
| dispatch all-to-all | token 发到 expert 所在 EP rank |
| dispatch postprocess | 本地再按 expert 排序，给 experts 准备输入 |
| combine preprocess | expert 输出准备回传 |
| combine all-to-all | 输出发回 token 原 rank |
| unpermute | 恢复原 token 顺序，并按 router weights 合并 |

![Megatron MoE all-to-all dispatch/combine 路径](/assets/img/blog/moe-we-like-you/03-megatron-moe-all-to-all-dispatch-combine.png)


*图 2：all-to-all dispatcher 可以拆成两次数据移动：dispatch 阶段把 token 按 expert 重新分组并发到 expert 所在 EP rank；combine 阶段再把 expert 输出发回原 token rank，最后 unpermute 并按 router weights 合并。*

这条路径就是 MoE 通信墙的来源。只要 expert load 不均、跨节点 EP 太多、token permutation 太碎，active FLOPs 再低也会被 all-to-all 和 memory movement 拖住。

## 10. Experts：TEGroupedMLP 为什么重要

Megatron 的 routed experts 常走 `TEGroupedMLP`。它用 Transformer Engine 的 `GroupedLinear` 把多个本地 experts 的 MLP 合并成 grouped GEMM。

这比 naive “每个 expert 一个小 MLP 循环跑”重要得多。MoE 的每个 expert 只拿到一部分 token，如果逐 expert 单独 launch，小 GEMM 会非常低效。GroupedGEMM 的意义是把一批不同 expert 的矩阵乘组织成更适合 GPU 的执行方式。

`TEGroupedMLP` 里还有一个和 LatentMoE 相关的细节：如果 `moe_latent_size` 打开，expert 输入输出维度会使用 latent size，而不是完整 hidden size。这意味着 latent expert path 不只是 layer 外面套一个投影，experts 内部的参数和 compute 形状也随之改变。

| 专家实现 | 使用场景 |
|---|---|
| `TEGroupedMLP` | 大规模训练默认关注，高效本地 grouped experts |
| `SequentialMLP` | correctness / fallback / 小规模调试 |
| `InferenceGroupedMLP` | 推理路径，可接不同 inference grouped GEMM backend |

## 11. Shared expert 和 LatentMoE 的接入位置

shared expert 在 `moe_layer.py` 里不是 router 选出来的普通 expert，而是单独的 shared path。

常规路径下，shared expert 会直接读完整 hidden states，输出再和 routed experts 的结果相加。为了性能，Megatron 支持 shared expert overlap：shared expert 可以和 routed expert dispatch/compute 重叠，减少串行开销。

LatentMoE 打开后，路径更微妙：

| 部件 | 维度 |
|---|---|
| shared expert | 通常读完整 hidden dim |
| routed experts | 可在 latent dim 上计算 |
| `fc1_latent_proj` | hidden dim -> latent dim |
| `fc2_latent_proj` | latent dim -> hidden dim |

源码里对 inference + NVLS dispatcher + latent MoE 的 shared expert overlap 做了专门处理：shared expert 需要在降维前运行，routed expert 输出需要在升维后再和 shared expert output 相加。这个细节说明，LatentMoE 不是单纯改 expert MLP hidden size，它会影响 shared path、dispatcher、推理 overlap 和 checkpoint 兼容。

## 12. 训练路径和推理路径并不完全一样

Megatron 里有 `InferenceTopKRouter`，这一点对训推一致性很关键。

训练时 `TopKRouter` 会处理：

- z-loss；
- aux / seq aux / global aux loss；
- token dropping；
- expert bias load 统计；
- padding mask 和训练态 logging；
- quantile balancing 的状态累计；
- router trace / replay 等调试逻辑。

推理时 `InferenceTopKRouter` 会尽量去掉这些训练开销，只保留 top-k routing 需要的东西。对 serving 来说这是合理的；对后训练/RL 来说，这也埋下了训推不一致的入口。

因此，debug MoE RL 或 serving 一致性时，不能只问“checkpoint 权重是不是一样”，还要问：

| 问题 | 为什么重要 |
|---|---|
| expert bias 是否被保存并正确加载 | bias 影响 top-k choice |
| 推理是否使用同样 score function | softmax/sigmoid/sqrtsoftplus 不可混 |
| top-k 后是否 renormalize | combine weights 会变 |
| token dropping 训练时是否打开 | serving 通常不会丢 token |
| router dtype 是否一致 | 低精度可能翻转 top-k |
| group-limited routing 是否一致 | expert group 选择不同会改变路径 |

第七篇会以 Megatron + SGLang 为例展开这一点。

## 13. Router replay / trace：把路由指标纳入监控

Megatron 里有 `router_replay.py` 和 `router_trace.py`。这两个工具类模块常被忽略，但对 MoE 研究有价值。

| 工具 | 用途 |
|---|---|
| router trace | 捕获 top-k ids、weights、logits、tokens-per-expert 等 routing 信息 |
| router replay | 复用或回放某次 routing 决策，用来排查 router 变化造成的训练差异 |

这给 MoE 调试提供了一个直接入口：除了 loss 曲线，还要看 route entropy、top-k overlap、expert load CV、device load CV。很多 MoE 问题在 loss 爆之前，已经会先反映到这些路由指标里。

对 SFT/RL 阶段尤其如此。固定一套 probe corpus，每个 checkpoint 跑 router trace，比只看 benchmark 更早发现 route drift。

## 14. 读源码时常见的混淆点

| 容易混淆的点 | 更准确的理解 |
|---|---|
| expert bias 是普通参数吗 | 更像负载控制器状态，参与 checkpoint，但更新不靠主 loss 反传 |
| loss-free 是完全不用 aux loss 吗 | 不一定。主平衡靠 bias，轻量 seq/global aux 仍可作为稳定器 |
| dropless 是没有 capacity 逻辑吗 | 不是。框架通常仍保留 capacity/drop/pad 逻辑，只是默认不丢 |
| shared expert 是一个普通 expert 吗 | 通常不是 router top-k 选出来的 expert，而是单独 always-on path |
| EP 只是把 expert 切到不同 GPU 吗 | 还包括 token dispatch/combine、load balance、straggler、buffer 和通信 overlap |
| LatentMoE 只是 low-rank MLP 吗 | 不是。它改变 routed expert path 的维度、通信 payload、expert 参数和 shared path 交互 |
| 推理 router 等同训练 router 吗 | 不等同。训练 path 有 loss/bias/drop 统计，推理 path 通常只保留 choice/weights |

## 15. 和下一篇的衔接

这篇讲的是 Megatron Core 里 MoE 部件怎么实现。下一篇会把其中两个系统瓶颈单独放大：

- token dispatch/combine 的通信墙，对应 DeepEP / Flex Dispatcher；
- expert MLP 的计算效率墙，对应 GroupedGEMM / fused MoE GEMM。

源码层面看，DeepEP 和 GroupedGEMM 不只是普通加速开关。当 MoE 模型进入大 EP、多节点和高专家数后，它们决定的是 active FLOPs 能不能转化成 tokens/s。

## 系列导航

上一篇：[第二篇：如果今天要 scaling MoE，架构和训练策略该怎么选]({% post_url 2026-07-06-moe-scaling-design-training %})  
下一篇：[第四篇：DeepEP 与 GroupedGEMM，MoE 大规模训练的通信墙和计算墙]({% post_url 2026-07-16-moe-deepep-groupedgemm %})  

## 参考资料

- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
- [Megatron-LM `moe_layer.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_layer.py)
- [Megatron-LM `router.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py)
- [Megatron-LM `moe_utils.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/moe_utils.py)
- [Megatron-LM `token_dispatcher.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/token_dispatcher.py)
- [Megatron-LM `experts.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/experts.py)
