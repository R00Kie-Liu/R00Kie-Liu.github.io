---
layout: post
title: "Why Muon Works"
date: 2026-07-07
tags: [Pretraining, Optimizer, Megatron]
description: "调研 Muon 优化器为什么能 work、相比 AdamW 的优劣、开源大模型如何使用 Muon，以及在 Megatron 中落地 Muon 的工程要点。"
---

## 写在前面

过去很长一段时间，AdamW 几乎是大模型预训练的默认优化器。很多训练 recipe 的讨论会集中在 learning rate schedule、weight decay、batch size、warmup、gradient clipping 和混合精度，而 optimizer 本身似乎已经没有太多悬念。

Muon 的出现让这个问题重新变得有意思。

它不是一个简单的 AdamW 变体，也不是“把 beta 改一改”的调参技巧。Muon 的核心想法是：Transformer 里占主要参数量的是二维权重矩阵，优化这些矩阵时不一定要把它们当成一堆独立标量；可以先对梯度做 momentum，再对矩阵更新做近似正交化，让更新方向具有更好的谱结构。

围绕 Muon，最值得先弄清楚的是四个问题：

1. 为什么有人想从 AdamW 迁移到 Muon？
2. Muon 相比 AdamW 的优势和风险在哪里？
3. Moonlight、Kimi K2、GLM、DeepSeek、Megatron/NeMo 这类公开实践到底怎么用 Muon？
4. 如果要在 Megatron 里真训一次，工程上应该先改哪里、先看哪些指标？

## TL;DR

- Muon 更适合被理解成 **matrix optimizer**，而不是 AdamW 的全量替代品。
- 最稳的迁移方式是 **Hybrid Optimizer**：2D hidden matrices 走 Muon，embedding、output head、bias、norm、scalar/vector 参数继续走 AdamW 或 Lion。
- Muon 能 work 的主线解释是：它在矩阵谱范数几何下构造一个正交化更新方向，类似一种非欧 trust-region / spectral steepest descent。
- 大模型实践里，Muon 真正重要的配套是 weight decay、update scale 校准、QK logits 稳定性监控，以及分布式训练中的 matrix-preserving param sharding。
- Megatron 里不能只是把 `optimizer=adam` 改成 `optimizer=muon`。Muon 需要 layer-wise distributed optimizer、QKV/MLA split、TP mode、NS steps、QK-Clip 和 AdamW fallback 一起设计。

一句话版本：

> Muon 值得作为预训练和持续预训练的 optimizer candidate，但不应该在 SFT/LoRA 或短训任务里无脑替换 AdamW。

## 1. 为什么大家开始关心 Muon

Muon 被 LLM 圈真正关注，主要来自 [Moonlight](https://arxiv.org/abs/2502.16982) 的结果。Moonshot AI 在 Moonlight-16B-A3B MoE 上报告，Muon 相比 AdamW 能带来约 `2x` compute efficiency。这个数字很吸引人，因为它不是一个小型 benchmark 上的 optimizer 花活，而是指向预训练最贵的部分：用同样 compute 达到更低 loss，或者用更少 compute 达到同样 target loss。

随后 Kimi K2 把 Muon 和 QK-Clip 组合成 MuonClip，GLM 系列提到 Muon Split，DeepSeek-V4 技术报告也把 Muon 放进 optimizer recipe。NVIDIA 则在 Megatron/NeMo 生态里做了 emerging optimizers 支持，让 Muon 不再只是单机 PyTorch 代码，而是开始进入真正的大规模训练框架。

这背后的直觉很简单：Transformer 的大多数参数不是孤立标量，而是矩阵。

例如：

- attention 的 Q/K/V/O projection；
- MLP 的 up/gate/down projection；
- MoE expert FFN 矩阵；
- 某些 MLA / GQA / latent projection 矩阵。

AdamW 对这些矩阵做的是逐元素一阶/二阶矩自适应。Muon 则显式利用矩阵结构：先做 momentum，再对更新矩阵做 Newton-Schulz 近似正交化。

这就是 Muon 和 AdamW 最大的概念差异。

## 2. AdamW 和 Muon 到底差在哪

AdamW 的更新可以粗略写成（这里省略 bias correction）：

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(g_t \odot g_t), \\
\Delta_t &= \frac{m_t}{\sqrt{v_t}+\epsilon}, \\
\theta_t &= \theta_{t-1} - \eta \Delta_t - \eta \lambda \theta_{t-1}.
\end{aligned}
$$

它的基本单位是元素。每个参数元素都有自己的二阶矩估计，优点是稳定、鲁棒、工程成熟；缺点是没有显式利用矩阵整体结构，optimizer state 也比较重。

Muon 的更新更像是先对矩阵梯度做 momentum，再把 momentum 矩阵映射到近似 polar factor：

$$
\begin{aligned}
M_t &= \mu M_{t-1} + (1-\mu)G_t, \\
O_t &\approx \operatorname{Polar}(M_t) \quad \text{via Newton--Schulz}, \\
W_t &= W_{t-1} - \eta \, s(W_{t-1}) \, O_t .
\end{aligned}
$$

这里的 \(G_t\) 和 \(M_t\) 是矩阵。Muon 不再问“每个元素应该缩放多少”，而是问“这个矩阵更新方向的谱结构是否合理”。

这也解释了为什么实际训练里一般不会让所有参数都走 Muon。Muon 适合 2D hidden matrices；embedding、output layer、bias、norm 这类参数形态和训练动力学不同，继续交给 AdamW/Lion 更稳。

一个更实用的分类可以写成：

| 参数类别 | 常见参数 | 形状特征 | 推荐处理 |
|---|---|---|---|
| Attention hidden matrices | `q_proj`、`k_proj`、`v_proj`、`o_proj`、fused `qkv_proj` | 2D 矩阵 | 优先走 Muon；fused QKV 要按 Q/K/V split |
| MLP hidden matrices | `gate_proj`、`up_proj`、`down_proj`、SwiGLU 的 `w1/w2/w3` | 2D 矩阵 | 优先走 Muon |
| MoE expert matrices | expert FFN 的 gate/up/down projection | 2D 矩阵，通常按 expert 分组 | 可以走 Muon，但要按 expert 保持矩阵边界 |
| MLA / latent projection | \(W^{UQ}\)、\(W^{UK}\)、\(W^{UV}\)、latent up/down projection | 2D 矩阵，可能融合多个 head | 可以走 Muon，但更适合按 head 或 latent block split |
| Embedding | token embedding、learned position embedding | 2D 表查找矩阵 | 通常保留 AdamW；它们的行更新和 token 频率强相关 |
| Output head | `lm_head`、tied embedding/output weight | 2D 矩阵 | 通常保留 AdamW；直接影响 logits，稳定性更敏感 |
| Norm 参数 | LayerNorm/RMSNorm 的 weight/bias | 1D 向量 | AdamW/Lion |
| Bias / scalar 参数 | linear bias、MoE expert bias、router bias、logit scale、LayerScale | 0D/1D 标量或向量 | AdamW/Lion |
| Router / gate 参数 | MoE router weight、task/router projection | 常见为 2D，但控制离散路由 | 需要单独 ablate；保守起点是 AdamW |

这张表里最容易踩坑的是 embedding、output head 和 router。它们可能也是 2D，但不等于就应该走 Muon。Muon 的“矩阵结构”假设更适合 hidden transformation matrix；而 embedding 的更新受 token 频率影响很强，output head 会直接改变 logits 尺度，router 则会改变 MoE 的离散分配。工程上更稳的做法是先把它们排除，等主干矩阵的 Muon 路径稳定后再单独 ablate。

![AdamW 到 Muon 的 Hybrid Optimizer 路由](/assets/img/blog/muon-optimizer/muon-hybrid-routing.svg)

*图 1：实际迁移更推荐 Hybrid Optimizer，而不是全量替换 AdamW。*

## 3. Muon 的收益和代价

Muon 最有吸引力的收益是 token / FLOPs 效率。Moonlight 的主张是：在 compute-optimal training 下，Muon 比 AdamW 有明显更好的计算效率。即使后续不同模型上的收益未必都能复现到 `2x`，它至少说明 optimizer 仍然是 scaling recipe 里的显性变量。

第二个收益是 optimizer state。AdamW 需要一阶矩和二阶矩；Muon 管理的矩阵参数通常只需要 momentum buffer，不需要 AdamW 的 second moment。对大模型预训练来说，optimizer state 是显存、checkpoint 和分布式状态管理的重要压力来源。

第三个收益是矩阵归纳偏置。Transformer 的主要权重本来就是矩阵，Muon 的正交化更新会把梯度奇异谱拉平，避免更新被少数大 singular value 主导。这一点和 Shampoo、matrix whitening、spectral preconditioning 一类方法有共同的味道，只是 Muon 的工程形态更轻，更容易塞进大模型训练。

但 Muon 的代价也很明确。

第一，每步多了 Newton-Schulz iteration。常见设置是 `num_ns_steps=5`，每一步都是矩阵乘。它省了二阶矩 state，但不是省了所有计算。

第二，分布式训练更麻烦。AdamW 可以把参数 flatten 成大 buffer 后做 fused update；Muon 必须知道每个 2D 权重矩阵的形状，因为正交化发生在矩阵层面。把所有参数拍平成一长条再做 Muon，会破坏它想利用的矩阵语义。

第三，大规模稳定性需要额外监控。Kimi K2 引入 MuonClip/QK-Clip，就是因为 attention Q/K 权重变化可能让 QK logits 变大，训练中需要控制 attention logit 爆炸和 loss spike。

因此可以先按训练场景粗略划分优先级：

| 场景 | 是否优先尝试 Muon | 原因 |
|---|---|---|
| 大规模预训练 | 是 | 潜在收益是 compute efficiency |
| 持续预训练 | 是，但要保守 A/B | 可复用 baseline，验证 same-token / same-wall-clock loss |
| 长上下文继续训练 | 可以尝试 | 需要特别监控 QK logits |
| SFT / LoRA | 不优先 | 工程收益可能覆盖不了风险 |
| 小模型短训 | 不优先 | AdamW 的稳定性和成熟度更重要 |

## 4. 为什么 Muon 能 work

Muon 的理论解释还在快速发展，但到 2026 年已经形成了几条比较清晰的脉络。

### 4.1 一个更直观的推导：稳、快与谱范数

苏神的 blog 里给了一个很通俗易懂的解释：理想优化器每一步都想同时满足“稳”和“快”。“稳”是指不要过度扰动模型，“快”是指让 loss 尽可能下降。

对矩阵参数 \(W\) 和梯度 \(G\)，一阶近似下 loss 变化可以写成：

$$
\Delta \mathcal{L}
\approx
\operatorname{Tr}(G^\top \Delta W).
$$

于是优化器的一步更新可以被理解成一个约束优化问题：

$$
\min_{\Delta W}
\operatorname{Tr}(G^\top \Delta W)
\quad \text{s.t.}\quad
\rho(\Delta W) \le \eta .
$$

不同优化器的关键差别，某种程度上就是选择了不同的“稳”的度量 \(\rho\)。如果选择 Frobenius 范数，就会得到类似 SGD 的梯度反方向；如果选择谱范数，就会自然导向 matrix sign / polar factor，也就是 Muon 试图近似的方向。

这条推导不需要预设“正交化更高级”，而是先问一个更基础的问题：对一个线性层 \(y=xW\)，什么样的 \(\Delta W\) 才算对输出扰动更可控？谱范数 \(\lVert \Delta W\rVert_2\) 恰好直接约束了线性映射的最大放大倍数，所以它比逐元素度量更贴近矩阵作为线性变换的角色。

### 4.2 谱范数约束下的 steepest descent

[A Note on the Convergence of Muon](https://arxiv.org/abs/2502.02900) 和 [Muon Optimizes Under Spectral Norm Constraints](https://arxiv.org/abs/2506.15054) 都把 Muon 放到谱范数几何里理解。对矩阵参数 \(W\)，如果我们考虑：

$$
\min_U \langle G, U\rangle
\quad \text{s.t.} \quad
\lVert U\rVert_2 \le 1,
$$

那么最优下降方向会和 \(G\) 的 polar factor / matrix sign 有关。Muon 用 Newton-Schulz 近似正交化 momentum matrix，本质上是在构造一个谱范数受控的矩阵更新方向。

由此可以得到一个关键直觉：Muon 并不是“把梯度 normalize 一下”，而是在矩阵空间里选择一个被谱范数约束的下降方向。

### 4.3 非欧 trust-region 视角

[Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization](https://arxiv.org/abs/2503.12645) 给了一个更统一的解释：orthogonalized gradient 可以看成一种 non-Euclidean trust-region method，其中 trust region 由矩阵谱范数定义。Muon 是这个框架下的一个具体实例。

这个视角也解释了为什么 weight decay 和 update scale 对大模型 Muon 很关键。Muon 改变了更新方向的几何约束，如果不重新校准步长和正则，直接替换 AdamW 可能会让某些层的 update RMS 变得不合理。

### 4.4 Polar decomposition / matrix preconditioning

[PolarGrad](https://arxiv.org/abs/2505.21799) 把 Muon、Shampoo 等方法放进更大的 matrix-gradient preconditioning 家族里看。Muon 可以理解成一种轻量 matrix preconditioner：它没有像 Shampoo 那样维护复杂的 Kronecker preconditioner，而是通过 polar / sign 近似改变更新矩阵的奇异值结构。

从工程角度看，Muon 不是神秘的新 optimizer，它更像矩阵预条件路线里更便宜、更 GPU-friendly 的一支。

### 4.5 少量 Newton-Schulz 为什么够用

精确做 SVD 或 QR 太慢，不适合大模型每步训练。Muon 实践中通常只做 3-5 步 Newton-Schulz。[Convergence of Muon with Newton-Schulz](https://arxiv.org/abs/2601.19156) 和 [Beyond the Ideal: Analyzing the Inexact Muon Update](https://arxiv.org/pdf/2510.19933) 都在讨论一个现实问题：我们不需要完美正交化，只需要足够好的 polar factor 近似。

这也是为什么后续会出现 Polar Express、CANS、Turbo-Muon、CacheMuon 这类工作。它们的目标不是把数学上最精确的 orthogonalization 搬进训练，而是用更少 matmul、更低精度、更好的系数或缓存机制，得到“训练上够用”的正交化更新。

### 4.6 边界条件：Muon 并非总有优势

我不建议把 Muon 理解成“理论上更高级，所以一定更好”。近期几篇论文也在提醒边界：

- [What Really Matters in Matrix-Whitening Optimizers?](https://arxiv.org/abs/2510.25000) 认为 matrix-whitening 类方法的收益不只来自 spectral descent，variance adaptation 可能同样重要。
- [Isotropic Curvature Model](https://arxiv.org/abs/2511.00674) 认为梯度正交化方向通常合理，但未必严格最优。
- [AdaGrad Meets Muon](https://arxiv.org/abs/2509.02981) 和 [Adam Improves Muon](https://arxiv.org/abs/2602.17080) 都在尝试把 adaptive stepsize / noise adaptation 重新接回 Muon。

所以更稳妥的说法是：

> Muon 给矩阵参数提供了一个强有力的新归纳偏置；但它是否优于 AdamW，仍然要回到具体模型、数据、并行方式和训练阶段里验证。

## 5. 开源模型和系统怎么用 Muon

公开资料里，Muon 已经不是孤立案例。放到 [2025-2026 开源 LLM 演进]({% post_url 2026-06-13-open-llm-survey %}) 这条线里看，它已经从“一个新优化器”变成了 frontier-scale sparse MoE recipe 的一部分。

| 项目 | Muon 用法 | 关键看点 |
|---|---|---|
| [Moonlight](https://arxiv.org/abs/2502.16982) | Muon + weight decay + per-parameter update scale | 第一个把 Muon 大规模 LLM 训练讲清楚的代表案例；重点是把 AdamW baseline 迁移到 Muon 时要做 update scale 对齐。 |
| [Kimi K2 / K2.5](https://github.com/MoonshotAI/Kimi-K2) | MuonClip = Muon + weight decay + RMS matching + QK-Clip | K2 用 MuonClip 处理 attention logits explosion；QK-Clip 将 query/key projection weights 按 head rescale，报告口径里阈值 \(\tau=100\)。K2.5 复用 K2 lineage，并把这套稳定性 recipe 延伸到 visual-agentic / multimodal RL 场景。 |
| DeepSeek-V4 | 大多数模块使用 Muon；embedding、output head、mHC、RMSNorm 等保留 AdamW | 这是“Muon 不是全量替代 AdamW”的典型案例。V4 还为 Muon 做 hybrid ZeRO bucket assignment，以处理 Muon 需要完整梯度矩阵的问题。 |
| GLM-4.5 / GLM-5 | GLM-4.5 使用 Muon + cosine decay；GLM-5 引入 Muon Split | Muon Split 的重点是稳定 MLA：把 projection weights 切成更符合语义的小矩阵再正交化。GLM-5 还把 Muon 和 MTP loss、MoE balance、long-context mid-training 放进同一套 recipe。 |
| NVIDIA Megatron / NeMo | dist_muon / layer-wise distributed optimizer / QK-Clip | 真正解决 DP/TP/param sync 的工程路径；适合把 Muon 从论文实现推进到 Megatron 训练栈。 |
| DeepSpeed | Muon 集成与 GLM 实践 | 更贴近 ZeRO/FSDP 训练栈的使用方式，重点是 optimizer state、sharding 和完整矩阵更新之间的折中。 |
| PyTorch | `torch.optim.Muon` | 通用实现入口，适合理解算法和小规模实验；frontier-scale 训练仍需要 Megatron/DeepSpeed 这类分布式框架。 |

这些实践有三个共同点。

第一，Muon 没有成为直接替代 AdamW 的答案。Kimi K2 用 QK-Clip 约束 attention logits，DeepSeek-V4 保留 AdamW exceptions，GLM-5 用 Muon Split 处理 MLA。真正可扩展的 recipe 都是 hybrid。

第二，Muon 通常和 LR schedule、weight decay、MTP loss、MoE balance 一起出现。换句话说，Muon 不是独立插件，而是训练控制面板的一部分。

第三，模型结构越复杂，Muon 越需要结构感知。高稀疏 MoE、MLA/GQA、长上下文 attention、agentic long trajectories 都会放大 optimizer 的稳定性问题，因此 Muon 的落地往往伴随着 QK-Clip、split_qkv、Muon Split、hybrid ZeRO 或 layer-wise distributed optimizer。

## 6. Muon 加速：不是只有 NS steps 一个旋钮

Muon 的系统瓶颈主要来自两件事：

1. Newton-Schulz 正交化本身要做额外 matmul；
2. 分布式训练里必须保持矩阵语义，不能随便 flatten。

围绕这两个瓶颈，最近的加速论文大致可以分成四类。

第一类是 **更快的 polar / matrix sign 近似**。例如 [The Polar Express](https://arxiv.org/abs/2505.16932)、[CANS](https://arxiv.org/abs/2506.10935)、[Turbo-Muon](https://arxiv.org/abs/2512.04632)。它们关注 Newton-Schulz 系数、预条件和低精度下的收敛速度。

第二类是 **减少通信或全矩阵同步**。例如 [MuonBP](https://arxiv.org/abs/2510.16981) 使用 block-periodic orthogonalization：大多数 step 做本地 blockwise orthogonalization，周期性做 full orthogonalization。这和 Megatron 里的 `blockwise` TP mode 很贴近。

第三类是 **复用时间相关性**。[CacheMuon](https://arxiv.org/abs/2606.16371) 的想法是，训练相邻 step 的 momentum / polar factor 往往变化平滑，可以缓存前序 step 的信息来近似当前 polar factor，减少重复正交化计算。

第四类是 **压低 optimizer state**。例如 [Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106)、4-bit-Muon-GRASP、MuonQ 等工作，重点不是梯度通信，而是 Muon momentum state 的量化。这里不能简单套 AdamW 的 state quantization，因为 Muon 对奇异向量方向误差更敏感。

我对这些加速方向的优先级判断是：

1. 先在 Megatron 里做好 blockwise / layer-wise 路径。
2. 再 ablate `num_ns_steps=5` vs `3`，以及 NS coefficient。
3. 如果 optimizer state 或 checkpoint 成本成为瓶颈，再考虑 8-bit Muon state。
4. adaptive / row-wise / neuron-wise Muon variants 放到第二阶段，不要和第一版迁移一起叠太多变量。

## 7. Megatron 里真正要改什么

在 Megatron 里做 Muon，最危险的误解是：把 optimizer 名字换掉就行。

Muon 的核心约束是：**每个 Muon update 要知道自己对应的 2D 矩阵形状**。这和 Megatron 里常见的 flatten buffer、distributed optimizer、overlap param gather 路径天然有冲突。

因此比较合理的工程路径是：

1. 先给参数打标签：2D hidden matrix vs non-2D / special params。
2. matrix params 走 Muon。
3. embedding、output、bias、norm 走 AdamW/Lion。
4. Muon 管理的矩阵按 layer-wise / bucket layout 分配到 DP ranks。
5. 每个 rank 更新自己负责的完整矩阵或完整 block。
6. 更新后 all-gather / buffer sync，让下一轮 forward 看到一致参数。

![Megatron 中 Muon 的工程路径](/assets/img/blog/muon-optimizer/megatron-muon-path.svg)

*图 2：Megatron 中 Muon 的难点是保持矩阵语义，同时和 DDP、TP、param sync 共存。*

### 7.1 参数路由

推荐规则很简单：

```python
if param.dim() == 2 and not is_embedding_or_output_parameter:
    optimizer = Muon
else:
    optimizer = AdamW  # or Lion
```

这条规则看起来粗糙，但很实用。实际代码里不能只看 `dim()==2`，还要结合参数名和模块类型排除 embedding、output head、router、norm、bias 等特殊参数。Muon 的主要收益来自 hidden matrices，不应该为了“全量使用 Muon”把这些参数也卷进去。

### 7.2 QKV / MLA split

QKV 经常是 fused weight。按常见行切分记法，可以写成：

$$
W_{qkv} \in \mathbb{R}^{3d_{\text{hidden}} \times d_{\text{hidden}}}.
$$

如果直接对整个 fused QKV 做正交化，Q/K/V 的更新会被绑在一起。实际更建议：

- 打开 `muon_split_qkv`；
- 对 Q、K、V 子矩阵分别做 Muon；
- 对 MLA up-projection 或 per-head projection，考虑按 head / latent block split。

GLM-5 里提到的 Muon Split 就是这个方向，但它更准确地说是 **optimizer-side split**，不是改变 MLA 的前向结构。MLA 的 KV cache 和推理路径仍然是原来的 latent 表示；变化发生在 Muon 对哪些矩阵做正交化。

以 MLA 的 up-projection 为例，报告中点名的是：

$$
W^{UQ},\quad W^{UK},\quad W^{UV}.
$$

如果把 \(W^{UQ}\) 看成不同 attention head 的行块拼接：

$$
W^{UQ}
=
\begin{bmatrix}
W^{UQ}_1 \\
W^{UQ}_2 \\
\cdots \\
W^{UQ}_H
\end{bmatrix},
\qquad
M^{UQ}
=
\begin{bmatrix}
M^{UQ}_1 \\
M^{UQ}_2 \\
\cdots \\
M^{UQ}_H
\end{bmatrix},
$$

原始做法是对整个 momentum 矩阵做一次正交化：

$$
\Delta W^{UQ} = \operatorname{Orth}(M^{UQ}).
$$

Muon Split 则是对每个 head 的小矩阵分别做正交化，再拼回去：

$$
\Delta W^{UQ}
=
\begin{bmatrix}
\operatorname{Orth}(M^{UQ}_1) \\
\operatorname{Orth}(M^{UQ}_2) \\
\cdots \\
\operatorname{Orth}(M^{UQ}_H)
\end{bmatrix}.
$$

\(W^{UK}\) 和 \(W^{UV}\) 同理。这样做的好处是，不同 head 不再共享同一个正交化尺度和谱预算。对于 MLA 这种从 compressed latent 展开到多头表示的结构来说，每个 head 的 Q/K/V 子空间可能承担不同功能；整矩阵正交化会把这些 head 绑在一起，而 per-head split 更接近 attention 本身的语义边界。GLM-5 报告里也提到，Muon Split 让 MLA 的效果接近 GQA-8，并且在预训练中 attention logits 可以保持稳定，不需要额外 clipping。

### 7.3 TP mode

Tensor Parallel 下，一个矩阵可能被切成多个 shard。Muon 的正交化可以有几种模式：

| 模式 | 含义 | 优点 | 风险 |
|---|---|---|---|
| duplicated | 多个 TP rank 重复做更接近完整矩阵的更新 | 简单，结果更接近 full Muon | 额外计算多 |
| distributed | 跨 TP rank 协同做正交化 | 更全局 | 通信复杂 |
| blockwise | 每个 shard/block 本地正交化 | 吞吐友好，工程可控 | 和 full Muon 有偏差 |

实践起点可以先选 `blockwise`。如果 loss 明显劣化，再考虑 periodic full orthogonalization 或更复杂的 distributed mode。

### 7.4 QK-Clip 和稳定性监控

迁移 Muon 时，以下日志应该被当成一等公民：

- max QK logits；
- Q/K weight norm；
- attention entropy；
- grad norm；
- Muon update RMS；
- per-layer update norm；
- loss spike 和 NaN/inf。

如果 max QK logits 持续增长，应该先启用 log-only QK-Clip 观察，再决定是否真正 clip。Kimi K2 的 MuonClip 很重要的一点就在这里：optimizer 和 attention stability 不是两个孤立问题。

为什么 Muon 下要特别看 QK logits？对单个 attention head，有：

$$
S =
\frac{QK^\top}{\sqrt{d_h}}
=
\frac{(XW_Q)(XW_K)^\top}{\sqrt{d_h}}.
$$

因此 logit scale 会同时受 \(W_Q\) 和 \(W_K\) 影响。粗略地说：

$$
|S_{ij}|
\lesssim
\frac{\lVert x_i\rVert \lVert x_j\rVert}{\sqrt{d_h}}
\lVert W_Q\rVert_2
\lVert W_K\rVert_2.
$$

也就是说，Q/K projection 的 scale drift 会以乘法形式放大到 attention logits 上。Muon 的矩阵正交化更新并不像 AdamW 的二阶矩那样对每个元素提供自适应阻尼，因此在长训或大规模 MoE 场景中更容易暴露 QK logits 漂移问题。

QK-Clip 的作用就是把这个漂移显式约束住。如果某个 head 的 max logit 为 \(m\)，阈值为 \(\tau\)，同时缩放 Q/K：

$$
W_Q \leftarrow \alpha W_Q,\quad
W_K \leftarrow \alpha W_K,\quad
\alpha = \sqrt{\frac{\tau}{m}},
$$

则 logits 会近似从 \(m\) 拉回到 \(\tau\)。

### 7.5 推荐起始配置

不同 Megatron / NeMo 封装层字段名会变，但第一轮可以从类似配置开始：

```yaml
optimizer: muon

lr: <reuse_adamw_baseline_lr>
min_lr: <reuse_adamw_baseline_min_lr>
weight_decay: 0.1
decoupled_weight_decay: true

muon_momentum: 0.95
muon_nesterov: false
muon_num_ns_steps: 5
muon_scale_mode: spectral
muon_coefficient_type: quintic
muon_tp_mode: blockwise
muon_split_qkv: true
muon_extra_scale_factor: 1.0
muon_scalar_optimizer: adam

use_layer_wise_distributed_optimizer: true
use_precision_aware_optimizer: false

log_max_qk_logits: true
qk_clip: log_only_first
```

第一轮不要同时改 batch size、sequence length、data mixture 和 LR schedule。否则你很难知道 Muon 到底有没有贡献。

这里尤其要注意 update scale。苏神在 Muon 快速上手指南里提醒过，不同 Muon 实现的主要区别之一就是正交化更新前面的缩放因子，例如 naive、KellerJordan、MuP、Moonlight 版本的 scale 并不完全一样。工程上这意味着：迁移 AdamW baseline 时，`muon_scale_mode` 和 `muon_extra_scale_factor` 不是装饰参数，而是决定 update RMS 是否接近原 recipe 的关键旋钮。

## 8. 第一轮实验怎么做

如果要在一个已有 AdamW baseline 上迁移 Muon，第一轮实验可以这样设计：

| 实验 | 改动 | 目标 |
|---|---|---|
| A | AdamW baseline | 保留同一代码路径和日志 |
| B | matrix-only Muon + AdamW fallback | 看是否稳定、loss/token 是否改善 |
| C | B + `num_ns_steps=3` | 看 optimizer step 加速是否值得 |
| D | B + QK-Clip log-only | 看 max QK logits 是否异常 |
| E | B + QK-Clip enabled | 在 logits 异常时验证稳定性 |
| F | B + `split_qkv=false` | 验证 QKV split 是否真的必要 |
| G | B + MLA per-head split ablation | 验证 \(W^{UQ}, W^{UK}, W^{UV}\) 是否需要按 head 独立正交化 |

指标不要只看 loss。至少看：

- same-token loss；
- same-wall-clock loss；
- tokens/s；
- optimizer step time；
- peak memory；
- checkpoint save/load；
- resume 后 loss continuity；
- max QK logits；
- per-layer update norm。

成败标准也要提前定清楚。如果 Muon 的 loss/token 更好，但 wall-clock 被 NS 和通信吃掉，那么它不一定是更好的工程方案。反过来，如果 tokens/s 稍慢但 same-wall-clock loss 更好，Muon 仍然值得继续投入。

## 9. 结论

综合来看，Muon 的位置可以概括成五点：

第一，Muon 是预训练 optimizer recipe 的一个真实变量，不是小模型 benchmark 上的局部 trick。Moonlight、Kimi K2、GLM、DeepSeek、Megatron/NeMo 的实践已经足够说明它值得认真对待。

第二，Muon 的正确打开方式是 hybrid。只让矩阵参数走 Muon，非矩阵和特殊参数继续走 AdamW/Lion，是更工程化、也更符合它设计初衷的路线。

第三，Muon 的理论解释正在收敛到矩阵谱几何、非欧 trust-region、polar decomposition 和 matrix preconditioning 这些关键词上。这些解释不保证它总是更有优势，但能帮助我们知道该看哪些失败模式。

第四，Megatron 里的难点主要不是 optimizer 数学，而是参数布局和分布式语义。只要 flatten buffer 破坏矩阵形态，Muon 的核心假设就没了。

第五，下一阶段真正有价值的方向可能不是“Muon vs AdamW 哪个更好”，而是：

- Muon + adaptive scaling；
- Muon + QK stability control；
- blockwise / periodic / cached orthogonalization；
- low-bit Muon state；
- 针对 MLA/GQA/MoE expert 的结构化 split。

## 推荐阅读

如果只读几篇，可以按这个顺序：

1. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/)
2. [苏剑林：Muon优化器赏析：从向量到矩阵的本质跨越](https://spaces.ac.cn/archives/10592)
3. [苏剑林：Muon续集：为什么我们选择尝试Muon？](https://spaces.ac.cn/archives/10739)
4. [苏剑林：Muon优化器指南：快速上手与关键细节](https://spaces.ac.cn/archives/11416)
5. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
6. [Understanding Gradient Orthogonalization via Non-Euclidean Trust-Region Optimization](https://arxiv.org/abs/2503.12645)
7. [Convergence of Muon with Newton-Schulz](https://arxiv.org/abs/2601.19156)
8. [The Polar Express](https://arxiv.org/abs/2505.16932)
9. [MuonBP: Faster Muon via Block-Periodic Orthogonalization](https://arxiv.org/abs/2510.16981)
10. [Effective Quantization of Muon Optimizer States](https://arxiv.org/abs/2509.23106)
11. [NVIDIA: Advancing Emerging Optimizers for Accelerated LLM Training with Megatron](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)
