---
layout: post
title: "长话长说（二）：Linear Attention，从 KV cache 到递推状态"
date: 2026-07-29
last_modified_at: 2026-10-21
tags: [LLM, Attention, Long Context, Pretraining]
description: "从 Softmax Attention 的 KV cache 表示出发，解释 linear attention 如何把历史压成递推状态，以及它和 RWKV、RetNet、Mamba、Gated DeltaNet、Kimi Linear/KDA 的关系。"
---

## 写在前面

> 本文是长上下文 attention 架构系列的第二篇。第一篇先做 [attention 与 KV cache 总览]({% post_url 2026-07-27-long-context-attention-architecture-survey %})；这一篇只讲 linear attention 这条线；第三篇继续看 [长上下文 attention 设计趋势]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %})。 完整目录见[系列导航]({% post_url 2026-07-27-long-context-attention-architecture-survey %}#series-map)。

在长上下文 attention 综述里，linear attention 是最容易讲糊的一条线。

GQA/MQA 很直观：少存几组 K/V。MLA 也还在同一个框架里：把 K/V 压成 latent。SWA、DSA、MSA 也可以理解成“少看一些历史 token 或 block”。但 linear attention 不是这个思路。它问的是另一个问题：

> 历史一定要以 token-level KV list 的形式存在吗？

标准 causal attention 里，历史是：

```text
[K_1,V_1], [K_2,V_2], ..., [K_t,V_t]
```

每生成一个新 token，都要拿当前 query 去读这串不断增长的 KV。linear attention 想把历史改写成：

```text
state_t
```

每一步只更新一个状态，再用当前 query 读这个状态。这个变化看上去只是把公式换了个括号，但对长上下文推理来说，它等于把“不断增长的 KV cache”换成了“固定形状的递推记忆”。

下文从 softmax attention 的历史表示出发，用一次矩阵乘法换序推到 recurrent state；然后解释有限状态为什么需要 gating 和 delta rule，再回到近期开源模型的 hybrid 配方。这样几条容易混在一起的路线，至少能在公式和系统接口上分开。

## 1. Full attention 的历史表示：KV list

先从普通 causal self-attention 写起。忽略多头和缩放系数，单个位置 $$t$$ 的输出可以写成：

$$
o_t = \frac{\sum_{i \le t}\exp(q_t^\top k_i)v_i}{\sum_{i \le t}\exp(q_t^\top k_i)}.
$$

这个式子有两个特点。

第一，当前 token 和每个历史 token 都显式打分。$$q_t^\top k_i$$ 是一个内容相关的相似度，softmax 会把权重集中到当前 query 认为重要的位置上。

第二，历史不能简单合并。因为每个新 query 都会重新决定“哪些历史 token 重要”，所以系统必须保留每个历史 token 的 K/V。decode 时 KV cache 随上下文线性增长；prefill 和训练时 attention score 矩阵随序列长度二次增长。

换句话说，full attention 的历史记忆是 **可随机访问的 token 表**：

| 问题 | Full attention 的答案 |
|---|---|
| 历史怎么存 | 每个 token 一份 K/V |
| 当前 token 怎么读历史 | query 对所有历史 key 打分 |
| 是否能精确找远处证据 | 能，前提是模型学会打分 |
| decode cache 是否随长度增长 | 是 |
| prefill/训练 attention 是否有 $$L^2$$ 项 | 是 |

这也是 full attention 很强的原因：它保留了细粒度的内容寻址能力。代价是长上下文时越来越贵。

## 2. Linear attention 的第一步：把 softmax 换成可交换的相似度

linear attention 最常见的推导从一个替换开始。假设 attention 的相似度可以写成两个 feature map 的内积：

$$
\mathrm{sim}(q,k)=\phi(q)^\top \phi(k),
$$

其中 $$\phi(\cdot)$$ 可以是简单的非负激活，也可以是随机特征近似。把 softmax attention 改写成这种形式后：

$$
o_t =
\frac{\sum_{i \le t}\phi(q_t)^\top \phi(k_i)v_i}
{\sum_{i \le t}\phi(q_t)^\top \phi(k_i)}.
$$

关键在于矩阵乘法可以换顺序：

$$
o_t =
\frac{\phi(q_t)^\top \left(\sum_{i \le t}\phi(k_i)v_i^\top\right)}
{\phi(q_t)^\top \left(\sum_{i \le t}\phi(k_i)\right)}.
$$

于是我们可以定义两个递推状态：

$$
\begin{aligned}
S_t &= S_{t-1} + \phi(k_t)v_t^\top, \\
z_t &= z_{t-1} + \phi(k_t).
\end{aligned}
$$

当前输出变成：

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
$$

这个式子是理解 linear attention 的抓手。它把历史上所有 token 的 K/V 合进了 $$S_t$$ 和 $$z_t$$。decode 时，新 token 只需要更新状态，不需要回读完整 KV list。

```text
Full attention:
  q_t -> scan [k_1,v_1] ... [k_t,v_t] -> o_t

Linear attention:
  state_t = update(state_{t-1}, k_t, v_t)
  q_t -> read state_t -> o_t
```

所以，“linear” 的重点不只是复杂度从 $$L^2$$ 变成 $$L$$，而是历史表示变了：从 token list 变成 recurrent state。

## 3. 它到底省了什么

在 causal decode 阶段，full attention 每一步都要读越来越长的 KV cache：

$$
\mathrm{cache}_{\mathrm{full}} \propto L \times H_{kv} \times d_h.
$$

linear attention 层则只需要保留状态。以最朴素的外积状态为例，单层单 head 的状态近似是：

$$
S \in \mathbb{R}^{d_\phi \times d_v}, \quad z \in \mathbb{R}^{d_\phi}.
$$

状态大小和历史长度 $$L$$ 无关，但和 feature 维度、value 维度有关。也就是说，它不是免费午餐，只是把增长方向从“随 token 数增长”换成了“固定状态容量”。

这有三个直接后果：

| 维度 | Full attention | Linear attention |
|---|---|---|
| 历史表示 | token-level KV list | fixed-shape recurrent state |
| decode 内存 | 随上下文长度增长 | 对每层状态近似固定 |
| 精确检索 | 可以重新访问任意 token | 依赖状态里是否保留下来 |
| prefill/训练 | attention core 有 $$L^2$$ 项 | 理想情况下按 $$L$$ 增长，但需要 scan/chunk kernel |
| 主要瓶颈 | KV bandwidth、attention score 矩阵 | 状态容量、kernel 并行化、数值稳定 |

这也是它和 GQA/MLA 的本质差别。GQA/MLA 仍然保留“每个 token 有一份历史表示”，只是把这份表示变小。linear attention 则试图让大部分层不再保存 token-level KV。

<a id="7-为什么-2024-2026-又重新热起来"></a>

## 4. 为什么早期 linear attention 没有成为主流

2020 年左右，linear attention 的核心数学已经很清楚了。Katharopoulos 等人的 Linear Transformer 明确指出，自注意力可以通过 kernel feature map 和矩阵乘法结合律改写成线性复杂度，并展示了它和 RNN 的关系。Performer 则走另一条路，用 FAVOR+ 随机特征去近似 softmax attention。

当时的任务和系统条件，还没有把它的长处逼出来。主战场很大一部分仍是 encoder / BERT 类任务，输入通常不够长，也没有 decoder-only 模型持续生成时反复读取 KV cache 的压力。把历史改成状态，带来的系统收益并不突出。

早期方法又常把目标定成“近似 softmax”。近似粗了会损失质量，近似复杂了则会吃掉速度与稳定性收益。更棘手的是 query-dependent selection：softmax attention 允许每个 query 重新挑选历史位置，而固定状态必须提前决定保留什么，细粒度证据难免发生压缩、混叠或遗忘。

线性复杂度也不自动等于更快的 GPU 实现。full attention 后来有 FlashAttention 这类高度优化的 kernel，精确 attention 的可用边界被推远；linear attention 若缺少 chunkwise scan、合适的 state layout 和 fused kernel，纸面复杂度很难兑现成 wall-clock 收益。

科学空间的 linear attention 系列把这段变化概括得很清楚：研究重点从模仿 softmax，逐渐转向设计独立的递推记忆，再反过来启发 softmax attention。按这个脉络看，今天的 KDA、Gated DeltaNet、RWKV 和 RetNet 已经不能只当作 softmax 的廉价近似。

长上下文和长输出扩大了 KV 容量与重复读取的成本，chunkwise kernel、状态缓存和 hybrid recipe 又改善了可实现性。这两方面共同推动了近年的应用，而非仅靠线性复杂度的数学结论。

<a id="6-rwkvretnetmamba-和-linear-attention-是什么关系"></a>

### 4.1 RWKV、RetNet、Mamba 和 linear attention 是什么关系

这几个名字经常被混在一起。它们确实有关联，但不要简单画等号。

| 路线 | 和 linear attention 的关系 | 适合放在哪个坐标里 |
|---|---|---|
| Linear Transformer / Performer | 标准 kernelized linear attention 路线 | 从 softmax attention 出发，用 feature map 和结合律降复杂度 |
| RetNet | 显式连接 recurrence、attention、chunkwise recurrence | 把 attention-like mixing 写成可并行训练、可递推推理的 retention |
| RWKV | 不是标准 kernelized attention，但目标接近 | 用 time-mixing、decay、receptance gate 做 RNN 形态的 token mixing |
| Mamba / SSM | 不是 attention，但同属 state-based sequence model | 用选择性状态空间替代大量 token-to-token attention |
| Gated DeltaNet / KDA | gated/delta linear attention 新一代 | 把有限状态的写入、遗忘、读取做得更强，更贴近 LLM 训练和推理 |

RWKV 和 linear attention 的关系尤其值得单独说一下。RWKV 不是“把 softmax attention 换个 feature map”这种标准 linear attention，但它和 linear attention 共享一个大方向：训练时尽量保留 Transformer 的并行性，推理时像 RNN 一样维护状态。

因此，RWKV 可以放进 broader linear/recurrent attention family，但不应等同于某个具体的 linear attention 公式。它们面对同一个工程约束：能否避免每层 KV cache 随上下文无限增长。

Mamba 属于 selective state space model。在 LLM 架构讨论里，它仍常和 linear attention 相邻出现，因为二者都把一部分 token mixing 交给固定形状的状态。linear attention 保留 query/key/value 式的读写，Mamba 则从动态系统和卷积/递推角度建模序列。

## 5. 从“累加状态”到“可控记忆”

前面的 normalized kernel attention 同时维护 `S_t`、`z_t`，读出时用后者构造分母。下面的 GDN/KDA 转向独立设计的关联记忆模块：直接读矩阵状态，q/k 与输出另做归一化，不再保留同一形式的 `z_t` 分母。这里改变了模型的更新与读出规则，只沿用递推记忆的视角。

最朴素的 linear attention 状态是累加：

$$
S_t = S_{t-1} + \phi(k_t)v_t^\top.
$$

这个式子有个明显问题：它只会往里加，不太会忘。长序列下，旧信息、新信息、噪声信息都挤在同一个状态里。full attention 可以通过 softmax 在每个 query 上重新选择重点；朴素 linear attention 则更像把历史搅进同一个记忆桶。

后来的改进大多在解决两个问题：

1. 什么信息应该写入状态？
2. 旧状态什么时候应该遗忘或更新？

后面几节会反复出现同一组符号，先固定一下：

| 符号 | 含义 |
|---|---|
| $$x_t$$ | 第 $$t$$ 个 token 的 hidden state |
| $$q_t,k_t,v_t$$ | 由 $$x_t$$ 投影得到的 query、key、value |
| $$S_t$$ | 到第 $$t$$ 个 token 为止累计出的 recurrent state |
| $$\alpha_t$$ | retention / forget gate，控制旧状态保留多少 |
| $$\beta_t$$ | write strength，控制当前 token 对状态写入多强 |
| $$\gamma_t$$ | output gate，控制读出的状态有多少进入输出 |
| $$B$$ | 伪代码中的 batch size；§6.3 的计算块长度统一记作 $$C$$ |

### 5.1 Gating：给状态加遗忘门

gating 的直觉很像 RNN/LSTM/GRU：状态不应该无条件累加，而应该有选择地保留、衰减、覆盖。

一个简化写法是：

$$
S_t = \alpha_t \odot S_{t-1} + \phi(k_t)v_t^\top.
$$

$$\alpha_t$$ 可以控制哪些通道应该保留、哪些应该衰减。这样模型不再被迫把所有历史等权塞进状态，而是可以学习不同时间尺度的记忆。

### 5.2 Delta rule：把写入变成有目标的修正

delta rule 的直觉是：如果当前 key 在旧状态里已经能读出某个 value，那么新 value 不一定要完整写进去；更合理的是写入“预测误差”。

简化地看，可以把更新理解成：

$$
\begin{aligned}
\mathrm{error}_t &= v_t - \operatorname{read}(S_{t-1}, k_t), \\
S_t &= S_{t-1} + \beta_t \cdot \operatorname{write}(k_t, \mathrm{error}_t).
\end{aligned}
$$

这样状态更新从“把新东西加进去”变成“修正当前记忆”。这对有限状态很关键：状态容量有限时，盲目累加很快会互相干扰；有目标的更新更像在维护一张可学习的快速权重表。

### 5.3 Gated DeltaNet：把遗忘门放进 delta rule

Gated DeltaNet 把 gating 和 delta rule 结合起来：gating 负责快速遗忘和时间尺度控制，delta update 负责更精细地修改状态。一个简化版递推可以写成：

$$
S_t = (I-\beta_t k_t k_t^\top)(\alpha_t S_{t-1}) + \beta_t k_t v_t^\top,
\qquad
\tilde{o}_t = S_t^\top q_t.
$$

这里把 $$S_t$$ 写成 key-dim $$\times$$ value-dim 的状态表，读出时用 $$S_t^\top q_t$$。$$\alpha_t$$ 是 data-dependent forget gate，$$\beta_t$$ 是 write gate。先令 $$\bar S_t=\alpha_tS_{t-1}$$，再从这张已经衰减的表里读出旧 value、计算误差并写入。展开后，擦除项也带有 $$\alpha_t$$：$$\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}$$。这个顺序与下面的伪代码一致。

这里的公式是单头视角。放到多头实现里，Gated DeltaNet 的 $$\alpha_t$$ 通常是 per-head 的 scalar retention/log-decay：每个 head 在每个 token 上有自己的遗忘强度，但同一个 head 内的 key channel 共享这个衰减系数。后面说 KDA 更细，是指 KDA 把它推进到 per-head、per-key-channel 的 retention vector。

Gated DeltaNet 的实际 block 还包含 q/k/v 投影、短卷积、SiLU、q/k 的 L2 normalization、输出归一化和 output gate。公式只描述了状态更新这一环；放进 Transformer 以后，它已经是一个完整 token-mixing block。

![Gated DeltaNet hybrid architecture and block design](/assets/img/blog/linear-attention/gated_deltanet_fig1_architecture.png)

*图源：Gated Delta Networks, Figure 1。左侧给出两种 hybrid pattern：Gated DeltaNet-H1 在 Gated DeltaNet 与 SWA 之间交替，H2 则把 Mamba2、Gated DeltaNet 和 SWA 叠在同一个循环里。右侧是 Gated DeltaNet block design：$$q,k$$ 经过 Linear、short convolution 和 L2 normalization，$$v$$、$$\alpha$$、$$\beta$$ 也由输入分支生成，再进入 Gated Delta Rule；读出的状态经过 normalization、output gate 和 Linear 输出。*

这里最容易漏掉的一点是：$$\alpha_t$$、$$\beta_t$$、$$\gamma_t$$ 不是手工规则，也没有单独的“遗忘标签”或“写入标签”。它们和普通 attention 里的 $$q,k,v$$ 一样，都是当前 token hidden state 经过 projection 生成的连续值，然后通过 next-token loss 反传学习。不过三者的激活函数和语义不完全一样：$$\alpha_t$$ 是 retention factor，Gated DeltaNet / Kimi Linear 里通常先生成 negative-Softplus 形式的 log-decay，再取指数；$$\beta_t$$ 是 bounded write strength，常用 sigmoid；$$\gamma_t$$ 是输出侧的 activation gate，常用 SiLU/Swish，不一定是 $$(0,1)$$ 里的概率门。

为什么 `old_value` 可以写成 $$S_{t-1}^\top k_t$$？可以先把状态近似理解成一张已经写过的 key-value memory：

$$
S_{t-1}\approx \sum_{i<t} k_i v_i^\top .
$$

于是用当前 key 读旧状态时：

$$
S_{t-1}^\top k_t
=
\sum_{i<t} v_i(k_i^\top k_t).
$$

这和一次没有 softmax 的 attention 很像：当前 $$k_t$$ 和哪些历史 $$k_i$$ 更相似，哪些历史 $$v_i$$ 就贡献更多。因此 `old_value` 指的是旧状态沿当前 key 方向读出的 value mixture，不对应某个具体历史 token。delta rule 后面的 `error = v_t - old_value`，问的就是当前 token 想写入的 $$v_t$$，旧状态已经能读出多少，还差多少。

先按单层写一个带 shape 的朴素版本。实际实现会把它改成 chunkwise recurrent 或 parallel scan，但学习逻辑不变：

```python
def gated_delta_forward(x):
    # x: [B, L, D_model]
    # H: num heads, Dk: key dim per head, Dv: value dim per head
    # 这些 W_* 都是普通可学习参数，靠 LM loss 训练
    q = normalize(project_q(x))              # [B, L, H, Dk]
    k = normalize(project_k(x))              # [B, L, H, Dk]
    v = project_v(x)                         # [B, L, H, Dv]

    # retention gate / decay:
    # 先生成 log-decay，再取 exp 得到 alpha；越接近 1，旧状态保留越多
    log_alpha = -exp(A_alpha)[None, None, :] * softplus(project_alpha(x))
    alpha = exp(log_alpha)                   # [B, L, H], in (0, 1)

    # write gate: 控制当前 token 对状态修正的幅度
    beta = sigmoid(project_beta(x))          # [B, L, H]

    # output gate: 调制读出的状态；通常是 SiLU/Swish gate，不是概率门
    gamma = silu(project_gamma(x))           # [B, L, H, Dv]

    # S 把 key 方向映射到 value；形状是 [d_key, d_value]
    S = zeros([B, H, Dk, Dv])
    y = []

    for t in range(L):
        # 1. 旧状态先按 head 级别衰减
        S_decay = alpha[:, t, :, None, None] * S

        # 2. 在当前 key 方向读旧状态
        old_value = einsum("b h k v, b h k -> b h v", S_decay, k[:, t])

        # 3. delta rule: 不直接写 v[t]，而是写“旧记忆没预测对的部分”
        error = v[:, t] - old_value           # [B, H, Dv]

        # 4. write gate + delta update
        S = S_decay + beta[:, t, :, None, None] * einsum(
            "b h k, b h v -> b h k v", k[:, t], error
        )

        # 5. 用当前 query 读取更新后的状态，再过 output gate
        o_t = einsum("b h k v, b h k -> b h v", S, q[:, t])
        y_t = project_o(gamma[:, t] * norm(o_t))
        y.append(y_t)

    return stack(y, dim=1)                    # [B, L, D_model]
```

为什么三个 gate 不统一写成 sigmoid？原因不在名字，而在它们插入计算图的位置。

| 量 | 插入位置 | 需要满足什么约束 | 常见写法 |
|---|---|---|---|
| $$\alpha_t$$ | 直接乘在 recurrent state 上 | 长序列里会反复累乘，最好在 log-space 控制 decay，且保证 $$\alpha_t\in(0,1)$$ | $$g_t=-e^A\operatorname{Softplus}(z_t),\ \alpha_t=\exp(g_t)$$ |
| $$\beta_t$$ | 乘在当前 token 的 delta 写入上 | 像一个 bounded learning rate，控制这次 correction 写多重 | $$\beta_t=\operatorname{Sigmoid}(W_\beta x_t)$$ |
| $$\gamma_t$$ | 作用在已经读出的 $$\tilde{o}_t$$ 上 | 只做输出通道调制，不参与状态递推的稳定性 | $$\gamma_t=\operatorname{SiLU}(W_\gamma x_t)$$ 或 Swish gate |

这里的 $$A$$ 可以理解成每个 head 的可学习 log-scale。它不负责根据当前 token 选择忘什么，那是 $$z_t=W_\alpha x_t+b_\alpha$$ 的工作；它主要控制这个 head 整体偏快忘还是偏慢忘。实际计算时，projection 先给出一个非正的 $$g_t=\log\alpha_t$$，最后用 $$\alpha_t=\exp(g_t)$$ 得到真正乘在 state 上的 retention factor。这样还有一个好处：多个 token 的 decay 连乘时，可以先在 log-space 累加，数值上更容易处理。

log-space 如何避免中间累计乘积过早下溢，在 §6.1 单独推导。

这段伪代码里有两层“学习”。

第一层是 gate 的学习。若某类 token 后面还会反复用到旧状态里的信息，loss 的梯度会推动对应的 log-decay 不那么负，等价于让 $$\alpha_t$$ 更大、状态衰减更慢；若上下文切到新话题、旧状态开始干扰预测，梯度会把 log-decay 推得更负，对应 $$\alpha_t$$ 变小。类似地，$$\beta_t$$ 会学习“这个 token 值不值得写、写多重”：某种写入能降低后续 token 的 loss，相关参数会被推向更强写入；写入带来干扰时，梯度会压低它。

第二层是 delta 的学习。`error = v[t] - old_value` 看起来像有一个监督目标，其实这个目标也是模型内部生成的：$$v_t$$ 来自 $$W_vx_t$$，$$old\_value$$ 来自旧状态和 $$k_t$$。如果旧状态在当前 key 方向上已经能读出接近 $$v_t$$ 的内容，error 就小，写入自然变弱；如果当前 token 带来新的关联，error 变大，$$\beta_t$$ 又允许写入，那么状态会沿 $$k_t$$ 这个方向修正。最终这些投影矩阵和 gate 参数，仍然一起由语言模型的交叉熵 loss 训练。

把上面的 update 展开，就能对应到前面的矩阵公式。这里的旧内容取自已经衰减的 $$\bar S_t$$：

$$
\begin{aligned}
S_t
&= \bar S_t + \beta_t k_t(v_t - \bar S_t^\top k_t)^\top \\
&= (I-\beta_t k_tk_t^\top)(\alpha_t S_{t-1})
 + \beta_t k_tv_t^\top .
\end{aligned}
$$

这个展开式也说明了 gating 和 delta 的分工。gating 控制整张状态表的时间尺度；delta rule 只在当前 key 相关的方向上做低秩修正。前者像“这段旧记忆还要不要留”，后者像“这个地址下的内容要不要改”。

拿到 $$S_t$$ 以后，还差一步把 memory 读回 token representation。最小形式就是用当前 query 去读更新后的状态：

$$
\tilde{o}_t = S_t^\top q_t.
$$

这一步和前面的 `old_value = S_{t-1}^\top k_t` 很像，只是用途不同：$$k_t$$ 用来检查当前 token 对应的旧记忆，$$q_t$$ 用来生成当前层要输出的 token mixing 结果。实际 Gated DeltaNet block 里，$$\tilde{o}_t$$ 通常还会经过 normalization、output gate 和 output projection：

$$
y_t = W_o\left(\gamma_t \odot \operatorname{Norm}(\tilde{o}_t)\right).
$$

这里的 $$\gamma_t$$ 也是由输入投影得到的 gate，但它和 $$\alpha_t,\beta_t$$ 的位置不同。它不改变 $$S_t$$ 本身，而是在状态读出以后做逐通道调制；$$W_o$$ 则把读出的 value 表示投回模型 hidden size。再往外才是标准 Transformer block 里的 residual connection 和 MLP。因此，Gated DeltaNet 里可以把三类 gate 分开理解：$$\alpha_t$$ 管旧状态保留，$$\beta_t$$ 管当前误差写入，$$\gamma_t$$ 管读出的内容如何暴露给下一层。

### 5.4 KDA：Kimi 线上的 gated delta attention

Kimi Delta Attention (KDA) 适合单独拿出来讲。它和 Gated DeltaNet 共享“gating + delta update”的思想，但目标不只是提出一个 token-mixing block，而是把这类状态更新接到 Kimi Linear / Kimi K3 的 hybrid 架构、FlashKDA kernel、NoPE/MLA 分工和推理 state cache 里。

先给一个紧凑对照，再逐项展开：

| 问题 | Gated DeltaNet | KDA / Kimi Linear → K3 |
|---|---|---|
| 状态更新 | gated delta rule | gated delta rule 的更细粒度版本 |
| 遗忘门 | per-head scalar retention | per-head / per-key-channel retention，位置和近因偏置更细 |
| 训练 kernel | chunkwise recurrent / parallel scan | FlashKDA，把 diagonal/off-diagonal tile 尽量统一到 dense Tensor Core 路径 |
| 模型 recipe | 可与 SWA、Mamba2 组成 hybrid | 3 KDA + 1 MLA/Gated MLA，MLA 层承担周期性全局读取 |
| 推理接口 | recurrent state cache | KDA state 与 MLA KV cache 一起进入 paged pool，还要处理 speculative decoding replay |

从状态更新开始。KDA 论文里把单头递推写成：

$$
S_t
=
(I-\beta_t k_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
+
\beta_t k_tv_t^\top,
\qquad
\tilde{o}_t=S_t^\top q_t.
$$

和 5.3 的 Gated DeltaNet 相比，它沿用同一套“先衰减、再做 delta correction”的顺序。不同的是，KDA 把标量 $$\alpha_t I$$ 换成 $$\operatorname{Diag}(\alpha_t)$$，沿 key/channel 维度分别衰减。这样一来，不同 channel 可以学习不同的时间尺度：有些维度更像短期缓冲，遇到话题切换就快速清空；有些维度衰减慢一些，用来保留更长距离的统计或结构线索。

如果写成伪代码，这个差别会更直观：

```python
# Gated DeltaNet：同一个 head 内共享 scalar retention
S_decay = alpha_t * S
old_value = S_decay.T @ k_t
S = S_decay + beta_t * outer(k_t, v_t - old_value)

# KDA：按 key/channel 衰减，再做同样的 delta correction
S_decay = alpha_vec_t[:, None] * S
old_value = S_decay.T @ k_t
S = S_decay + beta_t * outer(k_t, v_t - old_value)
```

这里的 `alpha_vec_t` 是 $$d_k$$ 维的 retention vector。也因此，KDA 的 decay 不再只是“这层记不记旧信息”的开关，它会在每个 head 内给不同 key channel 分配不同半衰期。

把 batch、head 和 channel 维度也写出来，KDA 里 $$\alpha$$ 大概是这样生成和使用的。下面只是 shape 示意，真实 FlashKDA 会改成 chunkwise kernel：

```python
# x: [B, L, D_model]
# H: num heads, Dk: key dim per head, Dv: value dim per head

q = project_q(x)                  # [B, L, H, Dk]
k = project_k(x)                  # [B, L, H, Dk]
v = project_v(x)                  # [B, L, H, Dv]

# decay logits: low-rank projection + head-specific bias
u = x @ W_alpha_down              # [B, L, R]
z_shared = u @ W_alpha_up         # [B, L, Dk]
z = z_shared[:, :, None, :] + b_alpha[None, None, :, :]
# z: [B, L, H, Dk], b_alpha: [H, Dk]

# Kimi Linear style：unbounded negative-Softplus log-decay
g = -exp(A)[None, None, :, None] * softplus(z)

# Kimi K3 style：lower-bounded log-decay；和上一行二选一
# g = g_min * sigmoid(exp(A)[None, None, :, None] * z)

alpha = exp(g)                    # [B, L, H, Dk]
beta = sigmoid(project_beta(x))    # [B, L, H]

S = zeros([B, H, Dk, Dv])
for t in range(L):
    S_decay = alpha[:, t, :, :, None] * S
    old_value = einsum("b h k v, b h k -> b h v", S_decay, k[:, t])
    error = v[:, t] - old_value
    S = S_decay + beta[:, t, :, None, None] * einsum("b h k, b h v -> b h k v", k[:, t], error)
```

这里最关键的是最后两维：`alpha[:, t]` 的形状是 `[B, H, Dk]`，乘到 `S: [B, H, Dk, Dv]` 上时，会沿 value 维广播。也就是说，KDA 在每个 head 里的每个 key channel 上都有自己的 retention factor。

Kimi Linear 与 K3 的 decay 参数化、FlashKDA tile 路径见 §6.2。

KDA 与 Qwen 使用的 Gated DeltaNet 同属 gated/delta family。Kimi 的 hybrid 配置、两类缓存与 speculative decoding replay 见 §9.2。

到这里，linear attention 的主语已经从“近似 softmax”转成了 **finite-state memory layer**：输入 token 生成读写信号，更新 recurrent state，再读回 token representation。RWKV、RetNet、Mamba 和 gated/delta 系列公式各不相同，但都在围绕这条状态路线做取舍。

### 5.5 Kalman Delta Networks：写入强度也可以依赖记忆的不确定性

[Kalman Delta Networks（KDN）](https://arxiv.org/abs/2609.07816)（2026-09）继续追问 $$\beta_t$$ 的来源。GDN/KDA 用当前 hidden state 投影出写入门；KDN 额外维护一份随历史更新的 uncertainty，决定现有记忆和新观测各应占多大权重。这里的 hidden state 本来就可能包含上下文，新增的是显式的不确定性递推。

沿用 $$S\in\mathbb R^{d_k\times d_v}$$ 的记法，令 $$D_t\in\mathbb R^{d_k\times d_k}$$ 为衰减矩阵，先预测 $$\bar S_t=D_tS_{t-1}$$。KDN 的基本读写式为：

$$
e_t=v_t-\bar S_t^\top k_t,\qquad
\kappa_t=\frac{P_t^-k_t}{r_t+k_t^\top P_t^-k_t},\qquad
S_t=\bar S_t+\kappa_te_t^\top.
$$

$$P_t^-\in\mathbb R^{d_k\times d_k}$$ 是写入前预测的 key-space 协方差，$$r_t>0$$ 表示新观测的噪声尺度；$$\kappa_t\in\mathbb R^{d_k}$$ 是 Kalman gain，$$e_t\in\mathbb R^{d_v}$$ 是 residual。协方差也经过预测与观测更新，因此后续写入能依赖此前积累的可靠性信息。

若不跟踪完整协方差，只以 $$P_t^-\approx b_t^-I$$ 近似它，那么：

$$
\kappa_t=\beta_tk_t,\qquad
\beta_t=\frac{b_t^-}{r_t+b_t^-\|k_t\|^2}.
$$

这就回到了 delta rule 熟悉的写入形式。与直接用 sigmoid 预测 $$\beta_t$$ 相比，区别在于 gain 由不确定性状态和观测可靠性共同决定；它也不等于手工指定一个固定学习率。

完整协方差更新昂贵；Diagonal / Isotropic KDN 因而分别保留 channel / head 级不确定性。论文在 750M、1.3B 受控预训练中验证语言建模和检索，吞吐图只测 mixer layer。并行化所需的 Riccati / Möbius 推导留给原文，本文重点是 gain 多依赖了一份历史可靠性状态。

## 6. 数值范围与 chunkwise 计算

状态更新式明确以后，还要解决两个实现问题：如何稳定地累计衰减，以及如何将递推变成 GPU 上可并行的块计算。

### 6.1 Log-decay：先累计对数，再求相对衰减

具体地说，状态衰减常会出现一串小于 1 的数连乘：

$$
\Gamma_t=\prod_{i=1}^{t}\alpha_i,\qquad 0<\alpha_i<1.
$$

长序列里直接维护 $$\Gamma_t$$ 很容易下溢；如果后面的 chunkwise 公式还要用 $$1/\Gamma_t$$ 做 rescale，又会把一个很小的数翻成一个很大的数，带来上溢风险。log-decay 的处理方式是先令

$$
g_i=\log\alpha_i\le 0,
$$

于是连乘变成求和：

$$
\log\Gamma_t=\sum_{i=1}^{t}g_i.
$$

更常用的其实是两个位置之间的相对 decay：

$$
\prod_{r=i}^{j}\alpha_r
=
\exp\left(\sum_{r=i}^{j}g_r\right)
=
\exp(G_j-G_{i-1}).
$$

这样就不用先算两个可能已经 underflow 的累计乘积，再做除法；中间过程只是在处理普通的 log-sum 和差分。需要注意的是，log-space 不是万能的。它避免的是中间连乘太早下溢，最后如果仍然要计算很大的 $$\exp(-G)$$，还是可能超出低精度范围。

这也解释了后面 Kimi K3 的改动为什么不矛盾。Kimi K3 改的是 $$\alpha_t$$ 这条 decay 分支：从 unbounded negative-Softplus 换成 lower-bounded scaled sigmoid；$$\beta_t$$ 和输出 gate 仍然有各自的职责。

### 6.2 从 negative-Softplus 到 lower-bounded decay

接着是遗忘门的参数化。Gated DeltaNet / Kimi Linear 一类做法通常用 negative-Softplus 生成 log-decay：

$$
g_t=-e^A\operatorname{Softplus}(z_t),\qquad \alpha_t=\exp(g_t).
$$

这个写法能保证 $$\alpha_t\in(0,1)$$，但 $$g_t$$ 没有下界，某个 channel 可以在一步里被压到非常接近 0。Kimi K3 的 lower-bounded decay 改的是这里：

$$
g_t=g_{\min}\operatorname{Sigmoid}(e^Az_t),\qquad g_{\min}=-5.
$$

于是 $$g_t\in(-5,0)$$，$$\alpha_t\in(e^{-5},1)$$。从递推语义上说，它限制了“最强遗忘”的幅度；从 kernel 角度说，它把 chunk 内累计 decay 的动态范围钉住了。KDA 的 chunkwise 公式里要用累计衰减 $$\Gamma$$ 以及 $$1/\Gamma$$ 对 key 做 rescale，如果 $$\Gamma$$ 可以任意小，低精度下就容易溢出。lower-bounded decay 让 16-token tile 内的 reciprocal rescaling 仍在 BF16 能处理的范围里，后面 FlashKDA 才能把更多 tile 交给 dense Tensor Core matmul。

训练 kernel 是另一层差异。单看递推公式，Gated DeltaNet 和 KDA 都可以 token by token 地跑；大模型预训练则必须切成 chunk，让 chunk 之间传 recurrent state、chunk 内尽量并行。KDA 的 chunkwise 形态把 inter-chunk state 读取和 intra-chunk causal tile 分开，FlashKDA 的重点也在这里：让这套递推在 GPU 上变成足够规则的矩阵乘，而不是再增加一个门控。

<a id="56-lightning-attention把-causal-linear-attention-跑起来"></a>

### 6.3 Lightning Attention：把 causal linear attention 跑起来

Lightning Attention 容易被看成另一种 attention 公式，更准确的定位却是 linear attention 的一条 **GPU-friendly 计算路线**。

问题出在 causal linear attention 的训练侧。按定义，每个位置要读“之前所有 token 累积出来的状态”，朴素实现通常会写成 prefix-sum / cumsum：

$$
S_t = S_{t-1} + \phi(k_t)v_t^\top,
\qquad
o_t = \phi(q_t)^\top S_t.
$$

这个写法复杂度是线性的，但在 GPU 上不一定快：cumsum 沿序列维有依赖，访存和同步开销会把理论优势吃掉。Lightning Attention 的核心做法是把序列按固定 block size $$C$$ 切开：

- **block 内**：长度不大，显式算局部的 causal $$QK^\top V$$ 交互；这里仍然不是 softmax attention；
- **block 间**：更远的历史不再逐 token 读，而是压进 linear state，再让当前 block 的 query 去读这个 state；
- **实现上**：forward/backward 都做 tiling，让计算更贴近 FlashAttention 那种 IO-aware kernel。

为什么 block 内可以直接算 $$QK^\top V$$？把第 $$b$$ 个 block 的起点记为 $$l$$。对 block 内某个位置 $$t$$，它的状态可以拆成：

$$
S_t
=
S_{l-1}
+
\sum_{i=l}^{t} k_i v_i^\top .
$$

于是输出也自然拆成两部分：

$$
o_t =
\underbrace{S_{l-1}^\top q_t}_{\text{block 前历史}}
+
\underbrace{\sum_{i=l}^{t} v_i(k_i^\top q_t)}_{\text{block 内历史}}.
$$

第一项只依赖 block 开始前累计好的 state。第二项只涉及当前 block 内的 $$Q_b,K_b,V_b$$，把 block 内所有 token 一起写出来，就是一个带 causal mask 的小矩阵乘：

$$
O_{\text{intra}}
=
\operatorname{tril}(Q_bK_b^\top)V_b.
$$

这里的 $$\operatorname{tril}$$ 只保留下三角，保证 block 内第 $$t$$ 个 token 不能读未来 token。它看起来像普通 attention 的 $$QK^\top V$$，但没有 softmax；它只是把 linear attention 在同一个小 block 内的 prefix recurrence 改写成矩阵乘。

可以粗略写成：

$$
\begin{aligned}
S_b &= \lambda^C S_{b-1} + \operatorname{BlockUpdate}(K_b,V_b,\lambda), \\
O_b &= O_{\text{intra}} + O_{\text{inter}}.
\end{aligned}
$$

这里的 block 指 kernel tiling 的计算块，不是模型语义里的“句子/文档块”。实现通常会根据显存、SRAM、head dim 和吞吐选择 $$C$$。比如一个长度为 $$L$$ 的序列会被切成 $$T=\lceil L/C\rceil$$ 个连续 token block；第 $$b$$ 个 block 内的 token 彼此做局部 causal linear attention，之前所有 block 的信息则通过 $$S_{b-1}$$ 进入。

这个 $$C$$ 的量级也要放准。Lightning Attention-2 论文把它写成可调的 block size；官方 Triton 实现里主 block 取 `BLOCK = 64`，backward 里还会用 `CBLOCK = 32` 的子块。工程上常见的是 32/64/128 token 这类 tile 大小。这里的 block 更接近 GPU tile：它不承担 sparse attention 的检索语义，也不对应数据 pipeline 里的文档片段，主要服务于 SRAM、寄存器和 Tensor Core 吞吐。

伪代码可以写得更贴近实现一些：

```python
def lightning_attention(Q, K, V, block_size, decay):
    KV_state = zeros([d, d])
    outputs = []
    for Qb, Kb, Vb in chunk(Q, K, V, block_size):
        # 当前 block 内部：一个小的 masked QK^T V，没有 softmax
        O_intra = local_causal_linear(Qb, Kb, Vb, decay)

        # 当前 block 之前的所有历史：只读累计 state
        O_inter = read_state(Qb, KV_state, decay)

        outputs.append(O_intra + O_inter)

        # 当前 block 写入 state，供后续 block 使用
        KV_state = update_state(KV_state, Kb, Vb, decay)

    return concat(outputs)
```

![Gated DeltaNet、KDA 与 Lightning Attention 的实现侧差异](/assets/img/blog/linear-attention/linear_variants_kda_lightning_gpt_image2.png)

*这张图只画实现侧差异：Gated DeltaNet/KDA 主要改变 recurrent state 的更新规则；Lightning Attention 更偏 kernel schedule，把 block 内局部计算和 block 间 state 读取拆开。*

Lightning Attention 主要解决 causal linear attention 如何跑快；Gated DeltaNet、KDA 主要改进状态的写入与遗忘。MiniMax、Ling 这类模型最终要同时处理两件事：状态规则必须有效，blockwise kernel 也必须在训练和推理中兑现吞吐。


<a id="8-为什么它仍然不能完全替代-full-attention"></a>

## 7. 固定状态的容量与时间尺度

linear attention 最大的问题也很清楚：固定状态一定有容量瓶颈。

full attention 像一张历史 token 表。只要 KV 还在，当前 query 理论上可以重新访问任意位置。linear attention 更像一份不断更新的摘要。摘要可以很强，但摘要终究不是原文。

这会影响几类能力：

| 能力 | Full attention | Linear attention 的压力 |
|---|---|---|
| 精确 needle retrieval | 可以直接打到远处 token | 远处细节可能已被状态压缩 |
| 多跳证据组合 | 每层可重新访问证据 | 依赖状态是否保留中间证据 |
| 复制、引用、代码精确定位 | token-level KV 很有优势 | 状态可能混叠相似片段 |
| 长程位置关系 | RoPE/位置机制较成熟 | 位置感知要和状态更新重新配合 |
| 可解释性 | attention score 可直接检查 | 状态读写不容易解释成选中了哪个 token |

这些能力差异解释了近期开源模型为什么普遍采用 hybrid，而没有删掉所有 full attention 层：

```text
多数层：linear / recurrent / state-based token mixing
少数全局层：full attention / MLA / sparse attention，提供远程内容读取
可选局部层或分支：SWA，保留近期位置的精细交互
```

Kimi Linear 选择 KDA + MLA，Gated DeltaNet 论文也讨论和 SWA/Mamba2 的 hybrid。这个方向比“全盘替换”更可信。

<a id="81-spectralshift训练窗口变长状态能记住多远"></a>

### 7.1 SpectralShift：训练窗口变长，状态能记住多远

容量之外还有时间尺度问题：状态虽然能递推到任意位置，早期信息的影响却可能已经衰减到几乎不可读。[SpectralShift](https://arxiv.org/abs/2609.14320)（2026-09）专门研究 Gated DeltaNet 的长上下文继续预训练。

按 §5.3 的公式，单头状态可写成 $$S_t=A_tS_{t-1}+B_t$$，其中：

$$
A_t=\alpha_t(I-\beta_tk_tk_t^\top),\qquad B_t=\beta_tk_tv_t^\top.
$$

固定输入产生的读写信号后，第 $$i$$ 步写入到第 $$t$$ 步的贡献会经过 $$A_tA_{t-1}\cdots A_{i+1}B_i$$。因此，有效记忆长度取决于这串状态转移怎样保留、衰减不同方向的信息，不能只看 state 的尺寸。作为标量直觉，$$0.999^{10000}\approx4.5\times10^{-5}$$：单步 retention 看似很高，跨万步后仍可能很弱；真实 GDN 还叠加了 key-directed correction，不能用这个标量例子替代完整分析。

SpectralShift 关注两件事：长依赖需要足够多的慢衰减模式，状态清理和上下文切换又需要保留快衰减模式。它调整 $$\alpha$$ 投影的初始化，而非一律把遗忘关掉。设参考投影矩阵为 $$W_\alpha^{\mathrm{ref}}$$，元素均值为 $$\mu$$，$$J$$ 为同形状的全 1 矩阵，重参数化写成：

$$
W_\alpha'=\mu J+s_1(W_\alpha^{\mathrm{ref}}-\mu J),\qquad 0<s_1\le1.
$$

这会缩放投影中偏离全局均值的部分。进入目标长度训练后，再单独缩放这一分支的学习率：$$\eta_\alpha'=s_2\eta_\alpha$$。$$s_1$$ 调整初始衰减谱，$$s_2$$ 控制继续训练对它的改动；其他模块不因此统一降低学习率。慢模式是否增加、快模式是否保留，依赖论文分析中的输入轨迹与扰动条件，需要用实际模型验证。

论文在 1.5B MoE 基线上把 8K 窗口扩到 32K、64K、128K，并比较分阶段延长和直接延长。对预训练实验的启发是：除了调全局 attention 的 RoPE 和长度 curriculum，还应单独检查 recurrent 层的 decay 分布与远程检索曲线。§6.1 的 log-space 处理改善数值计算，§6.2 的 lower-bounded decay 限制单步遗忘幅度；SpectralShift 处理的是训练后保留了哪些时间尺度，三者不能互相替代。

<a id="9-系统侧难点在哪里"></a>

## 8. 训练与推理接口

linear attention 不是把 `attention.py` 里的 softmax 换掉就结束。训练和推理都会多出新的 contract。

<a id="91-训练侧不能真的逐-token-for-loop"></a>

### 8.1 训练侧：不能真的逐 token for-loop

训练沿 §6 的 chunkwise 路径组织：先并行计算块内读写信号，再组合块间状态，反向按同一依赖传梯度。分布式切序列时还要传 state/transition；recompute 则需恢复块起点状态，不能把每块都当成独立序列。kernel fusion 与低精度范围决定吞吐，跨设备组合和显存生命周期见第四篇。

<a id="92-推理侧kv-cache-少了state-cache-来了"></a>

### 8.2 推理侧：KV cache 少了，state cache 来了

decode 要保存 recurrent state，而非增长的 KV list。一个具体例子是 speculative decoding：从已确认前缀状态试跑四个 draft，若只接受前两个，第四步状态不能直接留下。系统需恢复前缀 checkpoint，再 replay 两个已接受位置，或采用其他等价的中间状态保存方案；短卷积状态也要同步恢复。

prefix cache 同理：命中某段历史时，要拿到该边界的 recurrent state，而非只拿 token pages。hybrid 模型必须让 state、局部卷积历史与全局 KV 对应同一前缀；K3 的 projected-input replay 见 §9.2。

<a id="93-megatron-训练侧看-module-spec-和并行边界"></a>

### 8.3 Megatron 训练侧：看 module spec 和并行边界

Megatron 的 `experimental_attention_variant='gated_delta_net'` 是查找模块的入口，`linear_attention_freq` 决定层分布。需要沿 module spec 核对三件事：投影 heads 与短卷积怎样切分，CP 如何传边界状态，distributed checkpoint 如何保存权重。实验入口不等于某个模型报告中的完整训练 recipe。

<a id="10-开源-llm-的-linear-attention-recipe"></a>

## 9. 开源 LLM 的 Linear Attention Recipe

下面比较近期开源模型怎样组合状态层与周期性全局读取，以及怎样迁移已有 checkpoint。

![Linear Attention 演进与开源模型 recipe](/assets/img/blog/linear-attention/linear_attention_evolution_recipe_gpt_image2_v2.png)

*图：左侧按相关机制归类，不表示严格的时间顺序或继承关系；右侧比较 hybrid recipe。MiniMax-01 保留周期性 softmax 层，Ling 的配比以 Ling/Ring 2.6 报告为准。*

公开资料较完整的几条路线，恰好覆盖了这套 recipe 的不同环节：

| 模型线 | linear attention 组件 | full attention 组件 | 大致配比 / 形态 | 主要看点 |
|---|---|---|---|---|
| MiniMax-01 | Lightning Attention | softmax attention | 7 Lightning + 1 softmax | 将 hybrid attention 与 MoE 扩展到长上下文、大模型训练 |
| Kimi Linear / Kimi K3 | KDA | MLA / Gated MLA | 3 KDA + 1 MLA/Gated MLA | 从 gated delta rule 往 KDA 演进，后面又把 kernel、state cache、speculative decoding 一起做进去 |
| Qwen3-Next / Qwen3.5 | Gated DeltaNet | Gated Attention | 默认 3 linear + 1 full | 更像把 Gated DeltaNet 放进标准 Transformer block，并用固定间隔保留 full attention |
| Qwen3.8-Flash-Next | Gated DeltaNet | Qwen Sparse Attention (QSA) | 3 GDN + 1 QSA | 多数层继续走 recurrent state，全局层改成 micro-block indexer 驱动的 sparse attention |
| Ling-2.5/2.6/3.0 | Lightning Attention | MLA | Ling/Ring 2.6 报告里是 7 linear + 1 MLA | 将已有 GQA checkpoint 平滑迁移成 hybrid linear architecture |

<a id="101-minimaxlightning-attention-的大规模试验"></a>

### 9.1 MiniMax：Lightning Attention 的大规模试验

MiniMax-01 每七个 Lightning Attention block 后接一个 softmax attention block，将这套 hybrid 结构与 MoE 一起扩展到 foundation model 规模。

这里的 Lightning Attention 指 §6.3 的 blockwise kernel 路线：block 内用局部 causal $$QK^\top V$$ 计算交互，block 间历史由 recurrent state 承接。Ling/Ring 2.6 后来继续采用这一路线，并把问题推进到 MLA、checkpoint 迁移和 context parallel。MiniMax-01 提供了规模化证据，Ling 则补上已有大模型如何切换架构。

<a id="102-kimi从-kda-到-k3linear-state-也要服务化"></a>

### 9.2 Kimi：从 KDA 到 K3，linear state 也要服务化

Kimi K3 技术报告把这条线讲得更具体。它每个 block 仍然采用 **3 KDA + 1 Gated MLA** 的 hybrid pattern，并在 backbone 末尾额外放一个 Gated MLA，确保最后一层做全局 attention。KDA 负责高频、位置敏感、recency-aware 的序列混合；Gated MLA 负责周期性的全局内容交互。

这里讨论的是 Kimi Linear / K3；Kimi K2/K2.5 的语言主干采用 MLA，不属于这条 KDA 路线。

KDA 的 channel-wise retention 和 delta correction 已在 §5.4 推导。模型 recipe 的重点是这些状态层如何与全局层、缓存和解码流程组合。

K3 随后把 KDA 的数值范围和系统接口继续往前推：

| 改动 | 直接影响 |
|---|---|
| lower-bounded decay 与 causal tile | §6.2 已推导数值范围；K3 将它用于 16-token tile，使更多 diagonal/off-diagonal 计算采用 dense Tensor Core 路径 |
| full-rank output gate | KDA 输出门从 low-rank parameterization 改为 input-dependent full-rank projection，近似写作 $$y_t=W_o[\operatorname{Sigmoid}(W_gx_t)\odot\operatorname{RMSNorm}(\tilde{o}_t)]$$，其中 $$W_o,W_g$$ 是 learned projections，$$\tilde{o}_t$$ 是 KDA 读 state 后的中间输出；Gated MLA 也使用类似的 channel-wise gate |
| NoPE 层间分工 | KDA 通过 recurrent gating 和 decay 提供位置敏感与近因偏置，MLA 层专注 unrestricted global content interaction |
| FlashKDA 与 state cache | 训练和 prefill 使用 chunkwise FlashKDA，跨设备长上下文使用 KDA Context Parallelism；推理时 KDA recurrent state 与 MLA KV cache 进入统一 paged pool，prefix cache 命中边界同时保存或恢复两类状态 |
| speculative decoding replay | KDA state 每步原地更新，draft 被拒绝时无法直接回滚。K3 缓存体积更小的 projected inputs，验证后只把 accepted tokens 在片上 replay 成正确状态 |

这些配置需要联合测量：状态容量、数值范围、cache 命中与 replay 开销都会影响最终吞吐。

<a id="103-qwengated-deltanet--周期性全局层"></a>

### 9.3 Qwen：Gated DeltaNet + 周期性全局层

Qwen3-Next / Qwen3.5 这条线和 Kimi 很像，都是“多数层走 recurrent linear state，少数层保留 full attention”。但它选的 linear 模块更接近 Gated DeltaNet，而不是 KDA。

公开的 Qwen3-Next 配置里，`layer_types` 默认由 `full_attention_interval=4` 生成：第 4、8、12... 层是 `full_attention`，其余是 `linear_attention`，形成 3:1 pattern。配置还单独列出 `linear_key_head_dim`、`linear_value_head_dim`、`linear_num_key_heads`、`linear_num_value_heads`、`linear_conv_kernel_dim` 等字段，linear attention 因而成为模型结构中的一等模块，而非一次 kernel 替换。

Qwen3.8-Flash-Next 将周期性全局层换成 QSA，形成 **3 GDN + 1 QSA**：多数层更新固定状态，少数层先选 micro-block，再做 sparse attention。QSA 在 256K continued pretraining 阶段引入，推理评估覆盖到 1M；选择与索引复用见第三篇。

因此与 Kimi K3 的差异有两个明确位置：全局层从 Gated MLA 换成 QSA；状态层的 decay/beta 仍是 per-head scalar，而非 KDA 的 per-key-channel retention。报告中的 FlashQLA 是实现这套 GDN 的 kernel 路径，不改变前面区分的状态语义。

<a id="104-ling从-gqa-checkpoint-迁移到-lightning-attention--mla"></a>

### 9.4 Ling：从 GQA checkpoint 迁移到 Lightning Attention + MLA

Ling 这条线的重点不太一样。Ling/Ring 2.6 技术报告里说得很直接：Ling-2.0 已经有大量训练投入，如果重新从零训练一个 trillion-parameter 模型代价太高，所以他们做的是 **retrofit**：把已有 GQA-based checkpoint 迁移到 Lightning Attention + MLA 的 hybrid architecture。

报告先比较 $$M=2,4,8,16$$ 的 scaling law，选定 $$M=8$$：每 8 层放 7 层 Lightning Attention 和 1 层 MLA，比例比 Kimi/Qwen 的 3:1 更激进。转换也不是逐层替换后直接续训。linear attention 层要从 GQA 扩成更接近 MHA 的参数形态并加入 gating，保留的 full attention 层则从 GQA 转为 MLA，用低秩 latent 压缩 KV cache。

完整流程依次经过 Lightning Attention Conversion、Linear Warmup、MLA Conversion、MLA Warmup，之后才进入大规模 continue pre-training；QK Norm removal、Partial RoPE adaptation 与 TransMLA conversion 都在这些阶段内处理。Ling 这条线由此补上了前两个案例没有详细回答的问题：一个已经投入大量训练的 GQA checkpoint，怎样在不从零训练的前提下迁到 hybrid 架构。Ling 3.0-Flash 的公开说明更偏模型规模和上下文能力；attention 转换的细节，目前仍以 Ling/Ring 2.6 report 更完整。

<a id="105-从案例里抽出的约束"></a>

### 9.5 从案例里抽出的约束

这些案例共同保留了全局读取，但密度和实现不同：Kimi/Qwen 约每四层一个全局层，MiniMax-01 与 Ling/Ring 2.6 约每八层一个。比较时应固定状态预算与训练条件，再分别检查全局层配比、状态更新规则、checkpoint 迁移和状态恢复，不能只按“用了哪种 linear attention”归因。

<a id="11-小结改变的是历史表示"></a>

## 10. 小结：改变的是历史表示

linear attention 的回潮并不意味着 softmax attention 失效。变化来自更长的上下文与生成轨迹，以及随之放大的 serving 成本。softmax 保留可随机访问的 token-level KV，linear attention 则把历史预先折叠进 fixed-shape recurrent state；前者擅长按 query 精确寻址，后者用有限容量换取不随序列增长的层状态。

从 feature map 的乘法换序，到 gating、delta rule 和 KDA，推导始终围绕同一个矛盾：状态必须足够小，写入和遗忘又要尽量少损失未来可能需要的信息。近期开源模型给出的工程答案是 hybrid——多数层走状态，少数层保留全局 attention，并让 kernel、并行方式与 cache 系统一起适配。

[第三篇]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %}) 会接着处理剩下的问题：recurrent state、精确 KV、latent KV 和 sparse selection 为什么需要共存，以及 serving runtime 怎样理解这些不同的记忆层。

## 延伸阅读

linear attention 这条线很容易被论文名带着跑。更稳的读法，是把它拆成公式、状态更新、训练 kernel 和模型 recipe 四层。

- 公式入口可以从科学空间的两篇文章和 *Transformers are RNNs* 开始。这里先把 softmax attention 的矩阵乘法换序看明白：为什么历史可以从一串 KV 变成一个可递推的状态。
- 早期方法建议接着读 Performer、RetNet、RWKV 和 Lightning Attention。它们各自处理 feature map、retention、RNN 化和 chunkwise parallelism，能帮人看清“线性复杂度”背后到底换来了什么约束。
- Gated DeltaNet 与 KDA 适合放在状态更新这一层读。有限状态的问题不只是容量小，还包括什么时候写、写多少、旧信息怎么遗忘；gating、delta rule 和 decay 都是在处理这件事。
- 接着读 KDN 与 SpectralShift：前者增加 uncertainty state 来决定写入强度，后者研究上下文延长时的衰减谱。它们分别补充“怎样写”和“能保留多久”，都有比大规模模型 recipe 更受控的实验。
- 读 MiniMax、Kimi、Qwen 和 Ling 这些模型报告时，先比较 hybrid ratio 和迁移路径。哪些层保留 full/MLA attention，哪些层换成 state layer，往往比模块名字更能解释最终系统行为。
- 真要落地到 serving 和训练，还要接第三篇和第四篇。state cache、prefix reuse、speculative decoding replay、linear attention CP 这些接口问题，决定 linear attention 能不能从公式优势变成系统收益。

## 参考资料

### 科学空间 / 苏剑林

- [线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)
- [线性注意力简史：从模仿、创新到反哺](https://spaces.ac.cn/archives/11033)
- [A Brief History of Linear Attention: From Imitation and Innovation to Feeding Back](https://www.tylerromero.com/translations/scientific-spaces/a-brief-history-of-linear-attention/)：上面一篇的英文翻译，原站偶尔访问不稳定时可以用来对照。
- [科学空间 attention 标签](https://spaces.ac.cn/tag/attention/)：可以顺着这里找 Transformer 升级之路、RoPE、线性 attention 相关系列。

### 经典 linear attention / recurrent attention

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)
- [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- [Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658)
- [Various Lengths, Constant Speed: Efficient Language Modeling with Lightning Attention](https://proceedings.mlr.press/v235/qin24c.html)
- [OpenNLPLab/lightning-attention](https://github.com/OpenNLPLab/lightning-attention)

### 近期 gated / hybrid linear attention

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [Kalman Delta Networks: Uncertainty-aware Associative Memory](https://arxiv.org/abs/2609.07816)
- [SpectralShift: Effective Context Window Extension of Gated DeltaNet via Spectral Reparameterization](https://arxiv.org/abs/2609.14320)
- [SpectralShift code](https://github.com/RUCAIBox/GDN-SpectralShift)
- [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)
- [MoonshotAI/Kimi-K2.5](https://github.com/MoonshotAI/Kimi-K2.5)
- [Qwen3-Next: Towards Ultimate Training Efficiency](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)
- [Qwen3.5](https://qwen.ai/blog?id=qwen3.5)
- [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)
- [Qwen3NextConfig in Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/configuration_qwen3_next.py)
- [Ling and Ring 2.6 Technical Report](https://arxiv.org/abs/2606.15079)
- [inclusionAI/Ling-V2.5](https://github.com/inclusionAI/Ling-V2.5)
- [Ant Ling model docs](https://developer.ant-ling.com/en/docs/models/ling/)
