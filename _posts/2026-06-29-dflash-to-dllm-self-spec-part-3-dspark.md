---
layout: post
title: "从 DFlash 到 dLLM Self-Spec（三）：DSpark 的半自回归 Draft 与负载感知 Verify"
date: 2026-06-29
tags: [Infra, Speculative Decoding]
description: "结合 DeepSeek DSpark、DeepSpec 代码和论文，解释半自回归 draft 与 confidence-scheduled verification 如何服务化。"
---

## TL;DR

- DSpark 不是替代 DFlash，而是在 DFlash 的 parallel block draft 上补两件事：半自回归 token 依赖，以及负载感知的 verify budget。
- Markov/RNN head 让 block 内后续 token 显式依赖前面已采样 token，目标是缓解 DFlash 的 suffix acceptance decay。
- Confidence head 给 scheduler 提供 prefix survival probability，让 target verify 从固定 block 成本变成可调度资源。
- 如果从部署角度读，DSpark 最关键的问题不是“draft 怎么更像 target”，而是“在当前并发和 target capacity 下，哪些 verify token 值得送进 target”。
- DeepSpec 能证明 draft module、Markov head、confidence loss 等算法侧实现；DeepSeek 线上 serving scheduler 和 kernel 细节仍主要来自论文与公开配置。

## 相关链接

- [DeepSeek-V4-Pro-DSpark model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)
- [DeepSpec DSpark implementation](https://github.com/deepseek-ai/DeepSpec/tree/main/deepspec/modeling/dspark)
- [DSpark paper](https://arxiv.org/abs/2606.19348)

## 第三章：DeepSeek DSpark 的算法与 infra 融合解读

### 3.1 DSpark 解决的两个线上瓶颈

DFlash 已经用 block draft + target verify + KV injection 降低了大模型 decode 的逐 token 成本，SGLang Spec V2 又用 overlap、FutureMap、KV over-allocation 等机制减少了 serving 热路径里的等待。DSpark 不是推翻 DFlash，而是继续处理两个更偏线上部署的短板：

| DFlash 已经解决的问题 | DSpark 继续补的问题 |
| --- | --- |
| draft 侧一次生成 block，避免逐 token draft loop。 | block 后半段 token 因为缺少 token 间依赖，acceptance tail 容易下降。 |
| target 一次 verify 多个 draft token。 | 固定长度 verify 在高并发时会浪费 target batch capacity。 |
| KV injection 提高 draft 对上下文的理解。 | confidence head + scheduler 根据负载动态决定每个请求该 verify 多长。 |

### 3.2 算法信号如何进入 serving 调度

DSpark 可以概括为两个增强：draft 更连贯，verify 更节制。前者用半自回归生成缓解 DFlash block 后半段 token 的接受率衰减；后者用 confidence-scheduled verification 只验证“值得花 target compute”的 prefix。

如果用前文的性能模型：

`speedup ~= baseline target decode per-token cost * average committed tokens per round / speculative round total cost`

那么 DSpark 的三个直接目标是：

| 目标 | 对性能模型的影响 |
| --- | --- |
| 提高 accepted length | 增大分子里的 `average committed tokens per round`。 |
| 保持 draft cost 接近 DFlash | 避免 `T_draft` 因自回归 draft 变大。 |
| 减少无效 verify | 降低高并发下的有效 `T_verify` 和 target batch pressure。 |

这几个动作对应到 serving 层是：

| DSpark 动作 | 算法收益 | Infra / serving 含义 | DeepSpec 能证明什么 |
| --- | --- | --- | --- |
| DFlash-style parallel backbone | 一次产生 block hidden/base logits，保留低 draft cost。 | block shape 相对固定，更容易沿用 CUDA Graph、批处理和固定 attention metadata 思路。 | `Qwen3DSparkModel` 里有 block draft、mask/noise token、target hidden feature 输入。 |
| Markov / RNN sequential head | 给 block 内 token 加入局部自回归依赖，缓解 suffix decay。 | 重计算仍并行，只有很轻的 logits bias / sampling 左到右执行；不会退化成完整逐 token draft。 | `markov_head.py` 里 `sample_block_tokens` 左到右更新 `prev_token_ids`。 |
| Confidence head | 估计 prefix survival probability，知道哪些 suffix 值得 verify。 | verify length 从固定 block 变成动态 budget，需要 scheduler 能处理 per-request 变长 verify。 | `loss.py` 用 draft/target 分布距离构造 soft acceptance label。 |
| Hardware-aware scheduler | 在吞吐曲线和负载约束下选择 verify prefix。 | 需要 engine-side SPS profile、异步调度、变长 token flatten、attention marker / metadata 支持。 | DeepSpec 不包含完整线上 scheduler；这部分主要来自论文与公开配置。 |

核心问题链可以进一步拆成：

| Serving 问题 | DSpark 给出的信号或机制 | 线上落地时真正难的部分 |
| --- | --- | --- |
| verify token 太多，高并发下 target capacity 被低价值 suffix 占掉。 | confidence head 输出 prefix survival probability。 | scheduler 要把 confidence 转成 per-request verify length，而不是固定 verify 全 block。 |
| 变长 verify 破坏固定 block 的整齐形状。 | confidence-scheduled verification。 | token flatten、positions、attention marker / metadata、KV slot 映射都要支持不规则 token。 |
| scheduler 决策本身可能卡住 GPU。 | hardware-aware scheduler + 异步调度。 | 决策要和 GPU forward overlap，不能每轮同步等 CPU 算完最优 prefix。 |
| draft 更强会不会拖慢单轮 latency。 | 半自回归 head 只在轻量 logits / sampling 层做 sequential correction。 | 保持重计算并行，只把少量 token 依赖放在便宜路径上。 |

![DSpark 算法与 serving 调度融合图](/assets/dflash-self-spec/dspark-algo-infra.svg)

### 3.3 为什么 DFlash 之后还需要 DSpark

#### 3.3.1 DFlash 的 suffix acceptance decay

DFlash 的优点是一次 forward 产生一个 block，但这个优点也带来一个问题：block 内多个位置主要是并行预测的，后面的位置没有充分利用前面已经采样出的 draft token。

直观例子：

| 位置 | DFlash 更像在做什么 | 风险 |
| --- | --- | --- |
| 第 1 个 draft token | 基于当前上下文预测下一词。 | 通常比较准。 |
| 第 2-5 个 draft token | 基于上下文和 mask/noise 表示预测后续词。 | 对前面 draft token 的具体取值依赖不够强。 |

因此，block 越长，后半段越容易出现“语义大致对，但 token 级前缀对不上”的情况。由于 speculative decoding 是严格 prefix acceptance，只要前面某个 token 被拒，后面的 token 即使本来不错也不能提交。

这就是 DSpark 论文里反复强调的 suffix decay：DFlash 把 draft 成本压下来了，但 long block 的后半段 acceptance tail 仍然是上限瓶颈。

#### 3.3.2 固定长度 verify 会浪费 target batch capacity

前面我们讨论过：target verify 可以并行，是因为 draft tokens 已经给出来了，target 只需要像短 prefill/extend 一样一次评估多个已知位置。

但并行不等于免费。高并发时，target verify 的实际 token 数大致是：

`sum(batch 中每个请求的 verify_length)`

如果每个请求都固定 verify 5 个 token，而其中很多请求后两三个 token 被接受的概率很低，那么这些 token 会消耗 target attention、MLP、KV write、logits 和调度容量，却不会转化为 committed tokens。

这就是为什么类似投机解码算法在高并发下容易提速有限：baseline decode 已经通过 batching 把 GPU 喂得比较满，投机解码再额外引入一批低收益 verify token，就可能把 batch capacity 花在“马上会被拒绝的 suffix”上。

DSpark 的系统贡献正在这里：它不是简单让 verify kernel 更快，而是让 scheduler 少送低价值 token 进入 target verify。

### 3.4 半自回归 draft：重计算并行，轻采样串行

DSpark 的 draft 阶段保留了 DFlash 的并行 backbone：用 anchor token + mask/noise tokens 一次产生多个位置的 hidden states / base logits。

不同点是，DSpark 在最终采样 draft token 时加了一个轻量的 sequential head。论文默认使用 Markov head，DeepSpec 代码里也能看到这个结构：

```python
class VanillaMarkov(nn.Module):
    def __init__(self, *, vocab_size: int, markov_rank: int):
        self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
        self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)

    def apply_step_logits(self, logits, *, token_ids, hidden_states):
        return logits + self.markov_w2(self.markov_w1(token_ids.long()))
```

这段逻辑可以翻译成：

- DFlash backbone 给出每个位置的 `base_logits`。
- Markov head 根据“上一个已采样 token”生成一个 vocab bias。
- 当前 token 的最终 logits 是 `base_logits + transition_bias`。
- 然后从这个 logits 采样当前 token，并把它作为下一个位置的 previous token。

DeepSpec 的 `sample_block_tokens` 是左到右循环的：

```python
for step_idx in range(proposal_len):
    step_logits = self.apply_step_logits(
        base_logits[:, step_idx, :],
        token_ids=prev_token_ids,
        hidden_states=step_hidden,
    )
    next_token_ids = sample_tokens(step_logits.unsqueeze(1), temperature=temperature)
    prev_token_ids = next_token_ids
```

所以，回答一个容易混淆的问题：DSpark / DFlash 不是都“完全一次性直接吐出 8 个/16 个 token”。

更准确的说法是：

| 方法 | 重计算部分 | token 采样部分 |
| --- | --- | --- |
| DFlash | block hidden/base logits 一次并行算出。 | 通常可直接从各位置 logits 取 token。 |
| DSpark | block hidden/base logits 仍然一次并行算出。 | 用 Markov/RNN head 做很轻的左到右修正采样。 |

DSpark 只把很轻的 logits bias / sampling 做成 sequential，而不是把整个 draft model forward 变成逐 token 自回归。因此它能提高 block 内 token 依赖，又不显著增加 draft round latency。论文里报告的现象也是：draft length 从 4 增到 16 时，sequential head 只带来很小的单轮延迟增量，但 accepted length 明显改善。

### 3.5 Confidence-scheduled verification：verify 不是越长越好

DSpark 的另一个核心是 confidence head。它为每个 draft 位置预测一个条件 survival probability，可以理解为：

| 记号 | 含义 |
| --- | --- |
| `c_k` | 在前面 draft tokens 已经被接受的条件下，第 `k` 个 draft token 继续被接受的概率。 |
| `a_k = prod(c_1 ... c_k)` | prefix 能活到第 `k` 个 token 的累计概率。 |

训练时，confidence target 来自 draft distribution 和 target distribution 的距离。DeepSpec 的 loss 里对应代码是：

```python
draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
```

这相当于用 total variation distance 估计“draft 分布和 target 分布有多接近”。越接近，token 被 target 接受的概率越高。

有了 `a_k` 后，scheduler 不再固定验证每个请求的完整 block，而是问一个更工程化的问题：

在当前负载和硬件吞吐曲线下，应该把有限 target verify capacity 分给哪些请求、哪些 prefix token，才能最大化 expected committed tokens per second？

论文里的抽象目标可以简化为：

| 量 | 含义 |
| --- | --- |
| `B = sum_r(1 + l_r)` | 这一轮实际送进 target verify 的 token 规模。 |
| `tau = sum_r(1 + sum_j a_{r,j})` | 这一轮期望提交的 token 数。 |
| `SPS(B)` | 真实 serving engine 在 batch size `B` 下的 steps/sec。 |
| `Theta = tau * SPS(B)` | 系统期望 token throughput。 |

scheduler 会优先选择累计 survival probability 高的 prefix extension，并在 throughput 继续改善时增加 verify budget。

一句话：DSpark 把 `block_size=5` 从“每个请求都一定 verify 5 个”变成“最多可以 verify 5 个，但是否真的 verify 到第 5 个，要看 confidence 和系统负载”。

### 3.6 这如何回答 verify 瓶颈和高并发问题

前文说过，verify 的时间成本类似短 prefill/extend：可以并行，但 target model 仍然要对 `batch_size * verify_length` 个位置做完整 forward。

DSpark 正好把这个讨论往前推进一步：

| 场景 | 固定 block verify 的问题 | DSpark 的处理 |
| --- | --- | --- |
| 低并发 | target 有空余计算，verify 长一点也不一定伤吞吐。 | scheduler 可以给高 confidence 请求更长 verify budget。 |
| 中等并发 | 需要在 per-user speed 和 aggregate throughput 之间折中。 | 把 verify budget 分给更可能被接受的 prefix。 |
| 高并发 | target batch capacity 已经紧，低 confidence suffix 会拖垮吞吐。 | 动态缩短 verify length，保护 batch capacity。 |

所以 DSpark 的加速并不是“verify 终于不是瓶颈了”。更准确地说，它承认 verify 会成为瓶颈，并把 verify 从固定开销变成可调度资源。

这也是它和普通 DFlash 的关键区别：DFlash 主要优化 draft 侧和 block verify 形态；DSpark 进一步优化“哪些 draft token 值得被 verify”。

### 3.7 生产部署里的两个重要工程细节

#### 3.7.1 异步调度：为什么不用同步算完再决定

论文里的理想 scheduler 会根据当前轮 confidence 和硬件 `SPS(B)` 曲线计算最优 verify prefix。但生产 serving 里还有 CUDA Graph、Zero-Overhead Scheduling 等约束：下一轮 batch size 往往要尽早确定，否则 GPU pipeline 会被 scheduler 卡住。

DeepSeek 的做法是异步调度：

- 当前轮仍然按最新 confidence 对 candidate tokens 排序。
- 但动态容量 `K` 使用两步之前的 confidence 输出来估计。
- 这样 scheduler 的延迟可以被隐藏，GPU 不必等 CPU 同步决策。

这和 SGLang Spec V2 的 overlap 思路很像：收益不是某个数学公式更漂亮，而是避免“每轮 forward 后必须等 scheduler 完整想明白”的硬同步点。

#### 3.7.2 变长 verify：逻辑变长，物理展平

confidence scheduler 会让不同请求拥有不同 verify length。例如同一个 batch 中：

| 请求 | verify length |
| --- | --- |
| A | 5 |
| B | 2 |
| C | 0 |
| D | 4 |

如果把它们 padding 成统一长度再跑，低长度请求会浪费大量 padding compute。论文提到的生产做法是把不同请求的 verify tokens flatten 成一个物理 token 列表，让 kernel 把它们当成独立元素处理；真实的序列依赖通过 marker tensor / sparse attention metadata 表达。

这点和当前 SGLang DFlash 文档里的 `req_to_token`、`positions`、`cache_loc`、attention metadata 是同一类问题：投机解码的核心难点不只是“多预测几个 token”，而是让这些不规则 token 能被高效组织进 GPU kernel。

### 3.8 DeepSpec 代码侧证据：算法如何落到 draft module

DeepSpec 是 DeepSeek 开源的 speculative draft model 训练和评测仓库，支持 DSpark、DFlash、Eagle3。它提供的是训练 / 数据准备 / 离线评测代码，不是 DeepSeek 线上 serving engine 的完整实现。

因此，DeepSpec 的价值是把 DSpark 的算法部分钉牢：draft module 长什么样、Markov head 怎么加、confidence target 怎么训练、公共配置如何选择。它不能直接证明 DeepSeek 线上 scheduler、kernel 和服务调度的完整实现。

#### 3.8.1 Draft module：target hidden feature + block backbone + heads

从 `deepspec/modeling/dspark/qwen3/modeling.py` 可以看到，`Qwen3DSparkModel` 的配置显式要求这些字段：

| 字段 | 作用 |
| --- | --- |
| `target_layer_ids` | 选择 target model 哪些层的 hidden states 作为 draft module 的上下文特征。 |
| `mask_token_id` | 构造 DFlash/DSpark block draft 里的 mask/noise token。 |
| `num_anchors` | 训练时从长序列中采样多个 anchor block，提高训练吞吐。 |
| `enable_confidence_head` | 是否训练 confidence head。 |
| `markov_rank` | 是否启用 Markov head，以及低秩转移维度。 |

模型内部还有三个关键组件：

| 组件 | 代码含义 | 和 infra 的关系 |
| --- | --- | --- |
| `fc + hidden_norm` | 把多个 target layer hidden states 拼接后投到 draft hidden size。 | draft 不从零理解上下文，而是消费 target 已经算出的表示；serving 侧需要把这些 hidden states 或等价 KV/feature 接到 draft path。 |
| `markov_head` | 在 base logits 上叠加 token transition bias。 | 重的 backbone 仍然 block 并行，轻的 token 依赖放在 head 里，尽量不破坏 fixed-block 执行形态。 |
| `confidence_head` | 输出每个 draft 位置的 confidence。 | 这个输出不是单纯评估指标，而是后续 scheduler 做动态 verify budget 的输入。 |

这说明 DSpark 的 draft module 天然就是算法和系统的交界面：它既要提高 acceptance tail，又要产出 scheduler 能用的 confidence 信号。

#### 3.8.2 Markov head：为什么叫“半自回归”

`deepspec/modeling/dspark/markov_head.py` 里的 `VanillaMarkov` 很小，但它正是 DSpark 区分于纯 DFlash 的关键：

```python
self.markov_w1 = nn.Embedding(self.vocab_size, self.markov_rank)
self.markov_w2 = nn.Linear(self.markov_rank, self.vocab_size, bias=False)
```

它把上一个 token id 映射成一个低秩向量，再投回 vocab 维度，作为当前位置 logits 的 bias：

```python
return logits + self.compute_step_bias(token_ids, hidden_states)
```

采样时，`sample_block_tokens` 不是一次性独立采样所有位置，而是左到右更新 `prev_token_ids`：

```python
prev_token_ids = first_prev_token_ids.long()
for step_idx in range(proposal_len):
    step_logits = self.apply_step_logits(
        base_logits[:, step_idx, :],
        token_ids=prev_token_ids,
        hidden_states=step_hidden,
    )
    next_token_ids = sample_tokens(step_logits.unsqueeze(1), temperature=temperature)
    prev_token_ids = next_token_ids
```

这就是“半自回归”的具体含义：

| 部分 | 是否自回归 | 成本含义 |
| --- | --- | --- |
| backbone hidden/base logits | 否，block 并行。 | 保留 DFlash 的低 draft round cost。 |
| Markov/RNN head sampling | 是，轻量左到右。 | 引入 token 间依赖，但只多付很小的 logits 修正和采样成本。 |

算法上，它缓解 suffix decay；infra 上，它避免把 draft 重新变成完整逐 token forward。这也是 DSpark 能同时追求 accepted length 和 serving latency 的关键折中。

#### 3.8.3 Confidence loss：scheduler 信号从哪里来

`deepspec/modeling/dspark/loss.py` 里，confidence target 来自 draft distribution 和 target distribution 的距离：

```python
draft_probs = torch.softmax(outputs.draft_logits.float(), dim=-1)
target_probs = torch.softmax(aligned_target_logits.float(), dim=-1)
accept_rate_3d = 1.0 - 0.5 * (draft_probs - target_probs).abs().sum(dim=-1)
```

这等价于用 total variation distance 估计 draft token 被 target 接受的软概率。后面 confidence head 用 BCE 去拟合这个 soft target：

```python
confidence_errors = F.binary_cross_entropy_with_logits(
    outputs.confidence_pred.float(),
    confidence_targets,
    reduction="none",
)
```

这部分把算法和 infra 接起来了：

- 算法侧：confidence 不是简单的 top-1 概率，而是尽量贴近 target/draft 分布一致性。
- Scheduler 侧：只有 confidence 校准得比较可靠，`a_k = prod(c_1 ... c_k)` 才能用于估算“多 verify 一个 token 是否划算”。
- Serving 侧：confidence 最终要变成 per-request verify length，而不是只作为离线 metric。

#### 3.8.4 配置：公共 DeepSpec 和 V4-Pro-DSpark 的边界

从 DeepSpec 公开代码能确认的 DSpark 结构包括：

| 路径 | 说明 |
| --- | --- |
| `deepspec/modeling/dspark/qwen3/modeling.py` | Qwen3 DSpark draft model，实现 target hidden feature 输入、block backbone、confidence head。 |
| `deepspec/modeling/dspark/markov_head.py` | Markov / Gated Markov / RNN head，实现半自回归 token 修正采样。 |
| `deepspec/modeling/dspark/loss.py` | CE、distribution matching、confidence loss 和 acceptance 统计。 |
| `config/dspark/dspark_qwen3_4b.py` | 公共 Qwen3 DSpark 训练配置。 |

公共 Qwen3 配置里有这些关键参数：

```python
model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=512,
    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
    loss_decay_gamma=4.0,
)
```

这些配置和 DeepSeek-V4-Pro-DSpark 的公开生产配置不完全一样。公共 Qwen3 训练配置是 `block_size=7`、`markov_rank=256`；V4-Pro-DSpark 公开配置里是 `dspark_block_size=5`、`dspark_markov_rank=512`。这说明 DeepSpec 是算法和训练 recipe 的参考，而不是把 V4-Pro 线上参数逐项公开成可复现 production stack。

更重要的是，公共配置本身也体现了算法/infra 折中：

| 配置 | 算法含义 | 系统含义 |
| --- | --- | --- |
| `block_size=7` | 训练一个能提出较长 block 的 draft module。 | block 越大，verify 和 KV/headroom 压力越高，生产未必照搬。 |
| `num_draft_layers=5` | draft model 不能太浅，否则 acceptance 会掉。 | draft model 越深，`T_draft` 越高，需要和 accepted length 一起算账。 |
| `target_layer_ids=[1, 9, 17, 25, 33]` | 多层 target hidden 给 draft 更丰富的上下文特征。 | target cache / hidden feature 存储和传输成本更高。 |
| `markov_rank=256` | 用低秩 transition bias 注入 token 间依赖。 | rank 越高，head 成本越高；rank 太低，suffix 修正能力不足。 |
| `confidence_head_alpha=1.0` | confidence 是训练目标的一等公民。 | serving 侧需要消费这个信号，否则只训练不用就浪费了。 |

### 3.9 性能数字应该怎么读

论文里的结果可以分两类看：

| 类型 | 该怎么理解 |
| --- | --- |
| 离线 accepted length | 主要衡量 draft model 本身有没有更长的可接受前缀。 |
| 线上 throughput / TPS frontier | 衡量 draft、verify、scheduler、kernel、负载分布和硬件 profile 的整体系统效果。 |

离线结果里，DSpark 相比 DFlash 在 Qwen3-4B/8B/14B 上的 macro-average accepted length 分别提升约 16.3%、18.4%、18.3%。这说明半自回归 head 确实改善了 DFlash 的 suffix decay。

生产结果里，DSpark 在 DeepSeek-V4-Flash 和 DeepSeek-V4-Pro preview serving 中，相比 MTP-1 baseline 在相同实用吞吐水平下提升 per-user generation speed。严格 SLA 点上的 661% / 406% throughput 数字更应该理解为“把可服务 frontier 往外推”，不是在一个已经充分利用的 baseline 上稳定乘了 6 倍。

Figure 8 最值得关注的现象是：并发较低时 scheduler 给 4-6 token/request 的 verify budget；并发上升、target capacity 变紧时，verify budget 自动下降。这把“高并发下投机解码收益有限”和“verify 可能成为瓶颈”两个问题放到了同一个调度框架里。

### 3.10 实践启发

对 SGLang 使用者来说，DSpark 的启发不是“把 block size 拉大就完事”，而是：

- 先看 acceptance length，不要只看 draft 延迟。
- verify length 要和并发、context length、target batch capacity 一起看；高并发下更要看 accepted tokens / verified tokens。
- 高并发下固定 verify 全 block 可能浪费 target 计算，动态 prefix scheduling 会越来越重要。
- 变长 verify 不是只改策略，runtime 还要支持 token flatten、attention metadata、KV slot 映射和异步 scheduler。
- draft module 的训练域、target hidden layer 选择、Markov head rank 都会影响 acceptance tail。
- 公开启动方式只能验证模型能不能被 serving 框架拉起；线上级别的 speedup 还依赖 runtime 是否真正支持 DSpark 的动态 verify 调度。

## 结语

> DFlash 解决“如何便宜地提出一批 token”；Spec V2 解决“如何少等调度和拷贝”；DSpark 进一步解决“在当前系统负载下，哪些 token 值得送去 verify”。

## 参考资料

1. DeepSeek-V4-Pro-DSpark Hugging Face model card
   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark

2. DeepSpec GitHub repository
   https://github.com/deepseek-ai/DeepSpec

3. DSpark paper: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
   https://arxiv.org/abs/2606.19348

4. DeepSeek-V4-Pro-DSpark inference folder
   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/blob/main/inference/README.md

5. SGLang Spec V2 / DFlash overview
   https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/
