---
layout: post
title: "Attention Residual 与 Hyper-Connections：从 HC、iHC、mHC、oHC 到 xHC"
date: 2026-07-27
last_modified_at: 2026-10-27
tags: [Pretraining, Transformer, Residual, MoE]
description: "梳理多 residual stream、显式深度检索与矩阵记忆，解释 HC/iHC/mHC/oHC/xHC、Full/Block/Delta AttnRes，以及 MSAR、低秩路由和 LDM 的机制、公式、shape 与实验结果。"
---

最近，Attention Residuals 及相关的残差连接改造受到越来越多关注，也已经进入多款开放权重模型的架构。[Kimi K3](https://github.com/MoonshotAI/Kimi-K3) 采用 Attention Residuals；在多残差流这条路线上，[DeepSeek V4](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 使用 mHC，[混元 Hy4-preview](https://github.com/Tencent-Hunyuan/Hy4-preview) 使用 iHC。这些设计分别从跨层聚合和多 stream 读写入手，改进层与层之间的信息传递。

标准 Transformer 把每个 sublayer 的输出加回同一条 residual stream。这个设计好训、好并行，但也把跨层信息压进了一个不断累加的 hidden state：后面的层能读到总和，却不能直接选择某一层的贡献。Attention Residuals 把这一步固定累加改成可学习的深度聚合，让当前层能够选择需要的历史信息。

> **多 stream 方法增加可读写的 residual 槽位；AttnRes 方法增加对历史 source 的显式访问。**

## 0. 两条路线与矩阵记忆

![Residual 连接的方法脉络与关联：多 residual streams、Attention Residuals 与 Depth Memory](/assets/img/blog/attention-residual/method-map.svg)

*方法关系总览：实线表示设计分支或变体，灰色虚线表示机制关联，橙色虚线表示组件借用；按机制组织，不代表严格时间线。*

[查看 SVG 原图](/assets/img/blog/attention-residual/method-map.svg)

| 路线 | 状态如何变化 | 代表方法 |
|---|---|---|
| 多 residual streams | 从 $$[C]$$ 扩展到 $$[N,C]$$，每层读出一个输入，再混合、写回 | HC、iHC、mHC、oHC、xHC |
| Depth-source attention | 缓存历史 branch outputs、block sums 或 deltas，沿 source 维聚合 | Full / Block AttnRes、Delta AttnRes / Delta Block、LR-AttnRes、MSAR |
| 矩阵关联记忆 | 将历史 key–value 关联递推写入固定大小的矩阵，通过 query 读取 | LDM |

多 stream 内部还要区分 residual transition：iHC 固定成 identity；mHC 使用双随机矩阵；oHC 使用正交矩阵；xHC 在 mHC 的基础上丰富写回分量，并稀疏更新 stream。

近期工作进一步细分了 depth attention：Low-Rank AttnRes 缩小打分所用的 key，MSAR 为每个 source 构造多个上下文视图。Depth Memory 则提出另一种存储方式：LDM 将历史 key–value 关联写入固定大小的矩阵，后续层通过 query 读取。它保留跨层检索能力，但不再逐项缓存历史 source。

## 1. Residual Stream 缺少什么

标准 PreNorm Transformer 用一条 residual stream 累计 Attention 和 MLP 的输出。加法提供了直接的传播路径，也让工程实现只需维护一个 hidden state；不同历史贡献则被压进同一个总和，后续层无法沿独立的历史索引逐项读取。

模型仍然可以通过 branch function 学习输出什么。这里欠缺的是 residual path 自身的显式读写机制：当前层选择哪些历史贡献，新输出写入哪个槽位，以及旧信息如何在槽位之间传播。

很深的 PreNorm 网络还可能出现 residual magnitude 随深度增长、后续 branch 相对贡献减弱的现象。改造 residual path 因此要同时回答表达与稳定性问题：增加历史访问能力后，原来的直通路径如何保留？

## 2. 方法分类地图

### 2.1 按 residual state 形态分类

| 方法族 | 代表方法 | Residual state | 每层读什么 | 每层写什么 | 核心收益 | 主要代价 |
|---|---|---|---|---|---|---|
| 标准 residual | PreNorm Transformer | $$h \in \mathbb{R}^C$$ | 当前累计 state | 当前 layer output | 稳定、简单 | 深度路由固定 |
| 多 stream residual | HC | $$X \in \mathbb{R}^{N \times C}$$ | 多 stream 动态聚合 | 多 stream 动态写回 | 扩展 residual memory | 自由 mixing 带来稳定性风险 |
| Identity multi-stream residual | iHC | $$X \in \mathbb{R}^{N \times C}$$ | 多 stream 动态聚合 | $$H^{res}=I$$ + 动态写回 | 保留精确 identity transition，成本低 | 没有显式 cross-stream mixing |
| 约束多 stream residual | mHC | $$X \in \mathbb{R}^{N \times C}$$ | 多 stream 动态聚合 | doubly-stochastic stream mixing + write-back | 稳定性更好 | memory traffic 和 kernel 复杂 |
| 正交多 stream residual | oHC | $$X \in \mathbb{R}^{N \times C}$$ | 多 stream 动态聚合 | $$H^{res}\in SO(N)$$ + 动态写回 | 线性 mixing 整体保范数 | 验证规模与训练配置仍有限 |
| 稀疏多 stream residual | xHC | $$X \in \mathbb{R}^{N \times C}$$ | dense read 全部 stream | sparse write $$k$$ 个 active stream | 大 $$N$$ 更有性价比 | router、TopK、Flash 变体更复杂 |
| 深度 attention residual | Full AttnRes | 历史 depth sources | embedding + 历史 layer outputs | weighted replacement / aggregation | depth-wise softmax routing | 存储/通信随层数增长 |
| 分块深度 attention | Block AttnRes | block-level sources | 历史 block sum + 当前 partial block | block-level aggregation | 降低通信和缓存成本 | block 内信息被压缩 |
| Delta 深度 attention | Delta AttnRes / Delta Block | layer/block deltas | 历史增量 | additive delta routing | 避免 cumulative source 冗余 | 仍需维护 source list |
| 低秩深度路由 | LR-AttnRes | 全宽 depth sources | 低维 key 打分、全宽 value 聚合 | branch outputs / block sums | 降低路由打分成本 | value 缓存与聚合仍为全宽 |
| 多视图深度 attention | MSAR | 单流 depth sources，按需展开 | 多个全宽上下文视图，各自沿 depth 检索 | 单流 branch outputs / block sums | 同一历史的多种上下文读法 | 卷积展开、多路检索与融合 |
| 矩阵关联记忆 | LDM | residual + memory matrix + inverse covariance | 动态 query 读取矩阵 | RLS 更新 key–value 关联 | 状态大小不随深度增长 | 矩阵状态、读写及高效反向实现 |

### 2.2 按“路由对象”分类

HC family 在 stream 维路由：HC 自由混合，iHC 固定 $$H^{res}=I$$，mHC 约束为双随机混合，oHC 约束为正交混合，xHC 则读取全部 streams、只更新选中的部分。

AttnRes 在 depth source 维路由：Full AttnRes 聚合 embedding 与历史 branch outputs，Block AttnRes 聚合已完成的 block sums 与当前 partial sum，Delta 系列则选择 layer / block deltas。LR-AttnRes 保留这些全宽 values，只缩小打分空间；MSAR 对每种 source 视图分别执行 depth softmax。LDM 的 query 则读取压缩后的关联记忆，没有一组可直接解释为“第几个历史 source”的 softmax 权重。

- **HC/iHC/mHC/oHC/xHC 更像是在宽度维度上增加 residual memory slots。**
- **AttnRes/Delta AttnRes 更像是在深度维度上增加可学习 source selection。**
- 两者都包在 Attention / MLP 等 sublayer 的 residual path 上。跨 token 的混合主要由 Attention 承担；常规 MLP / MoE 逐 token 计算，xHC 的 temporal convolution 则额外引入局部 token mixing。

一个容易混淆的点是：HC / iHC / mHC / oHC / xHC 里的 stream 不是“第几层的 hidden”。它们不会维护一个 $$[h_0,h_1,\ldots,h_l]$$ 的历史层列表，也不会让当前层显式选择“读第 5 层还是第 17 层”。这些方法是在每个 token 的 residual state 里多开 $$N$$ 个并行槽位；不同层的信息是经过多次 read、mix、write 之后，隐式沉淀到这些槽位里的。换句话说，HC family 的 memory bank 在 stream 维，AttnRes family 的 memory bank 才在 depth/source 维。

### 2.3 Shape 与伪代码速查

为突出 residual 机制，公式和伪代码省略 batch 与 sequence 维；完整张量通常还带有 $$B,T$$ 两维，例如 $$h\in\mathbb{R}^{B\times T\times C}$$。$$F$$ 表示 Attention 或 MLP / MoE sublayer。

| 符号 | 简化 shape | 含义 |
|---|---:|---|
| $$C$$ | scalar | hidden size / residual stream width |
| $$L$$ | scalar | 网络的 sublayer 总数；一个 Attention + MLP 层含两个 sublayers |
| $$N$$ | scalar | HC/iHC/mHC/oHC/xHC 的 residual stream 数，是人工设定的 expansion rate |
| $$k$$ | scalar | xHC 每个 sublayer 更新的 active stream 数，通常 $$k \ll N$$ |
| $$K_r$$ | scalar | xHC temporal augmentation 后的 write-back component 数 |
| $$G$$ | scalar | 本文统一按 sublayer 计的 block size；一个 Attention + MLP 层含两个 sublayers |
| $$N_{\mathrm{blk}}$$ | scalar | Block AttnRes / Delta Block 的 block 数，避免和 stream expansion $$N$$ 混用 |
| $$h_l$$ | $$[C]$$ | 标准 Transformer 的单条 residual state |
| $$X_l$$ | $$[N,C]$$ | HC/iHC/mHC/oHC/xHC 的多 residual stream state |
| $$H_l^{pre}$$ | $$[1,N]$$ | 从 $$N$$ 条 stream 读出 sublayer input 的 read mapping |
| $$H_l^{res}$$ | $$[N,N]$$ 或 $$[k,k]$$ | residual streams 之间的 mixing matrix |
| $$H_l^{post}$$ | $$[N,1]$$ 或 $$[k,K_r]$$ | 把 sublayer output 写回 stream 的 write mapping |
| $$h_l^{in}$$ | $$[C]$$ | 读出后喂给 Attention / MLP 的输入 |
| $$h_l^{out}$$ | $$[C]$$ | Attention / MLP 产生的新信息 |
| $$S_l$$ | $$[M,C]$$ | depth sources；Kimi 原始定义为 branch outputs / block sums，cumulative-state 版本另行注明 |
| $$\Delta_l$$ | $$[M,C]$$ | Delta AttnRes 的 depth delta sources |
| $$\alpha_l$$ | $$[M]$$ | depth-wise softmax routing weights |
| $$r$$ | scalar | LR-AttnRes 的 key / query 宽度，$$r<C$$ |
| $$m$$ | scalar | MSAR 的 source 视图数，主配置为 4；xHC 小节中的同名变量另指 fixed stream 数 |
| $$N_g,D_k,C_v$$ | scalars | LDM 的分组数、key 宽度与组内 value 宽度，$$C_v=C/N_g$$ |
| $$\mathcal{M}_l$$ | $$[N_g,C_v,D_k]$$ | LDM 的矩阵记忆；单组公式省略分组维 |
| $$P_l$$ | $$[N_g,D_k,D_k]$$ | LDM 的 inverse covariance state |

$$h_l^{in}$$ 是 residual routing 之后读出来、准备喂给当前 sublayer 的表示；$$h_l^{out}$$ 是 Attention / MLP / MoE 根据这个输入算出的新信息。标准 residual 里这两者通常藏在 `norm(h)` 和 `F(norm(h))` 里；HC/iHC/mHC/oHC/xHC 把“读出输入”和“写回新信息”显式拆开了。

深度以 sublayer 为单位。`F` 表示 Attention 或 MLP/MoE 本体，`norm()` 显式放在调用前；mapping 使用的 RMSNorm 是另一组操作。`rmsnorm(S)` 总是沿最后的 hidden 维归一化，softmax 沿 source 维执行。xHC 的 temporal augmentation 例外，需要读同一序列内当前 token 及其之前的输出，不能仅凭一个 $$[C]$$ 向量计算。

标准 PreNorm residual 的单 sublayer 最简单：

```python
# h: [C]
h_in = norm(h)       # [C]
h_out = F(h_in)       # [C]
h_next = h + h_out    # [C]
```

HC family 的共同形式是 read、sublayer、mix、write。以下伪代码采用 **mHC 的参数化**，原始 HC 的 read/write gates 与之不同。

```python
# X: [N, C]
# W_pre: [NC, N], W_post: [NC, N], W_res: [NC, N^2]

z = rmsnorm(vec(X))                          # [NC]

pre_logits = alpha_pre * (z @ W_pre) + b_pre # [N]
post_logits = alpha_post * (z @ W_post) + b_post
res_logits = alpha_res * mat(z @ W_res) + b_res
# z @ W_res: [N^2], mat(...): [N, N]

H_pre = sigmoid(pre_logits)                  # [N]
H_post = 2 * sigmoid(post_logits)            # [N]
H_res = sinkhorn(exp(res_logits))            # [N, N], mHC only

h_in = sum_i(H_pre[i] * X[i])                # [C]
h_out = F(norm(h_in))                       # [C]

X_mixed = stack([
    sum_i(H_res[j, i] * X[i])
    for j in range(N)
])                                           # [N, C]
write = stack([
    H_post[j] * h_out
    for j in range(N)
])                                           # [N, C]
X_next = X_mixed + write                     # [N, C]
```

iHC 固定 residual transition 为 identity。下例沿用 mHC 风格的 read/write gates；不同 iHC 模型的 gate 与 norm 配置可能有所不同。

```python
# X: [N, C]
z = rmsnorm(vec(X))                          # [NC]

H_pre = sigmoid(alpha_pre * (z @ W_pre) + b_pre)       # [N]
H_post = 2 * sigmoid(alpha_post * (z @ W_post) + b_post) # [N]

h_in = sum_i(H_pre[i] * X[i])                # [C]
h_out = F(norm(h_in))                       # [C]

# identity transition: no stream mixing, no Sinkhorn
X_next = stack([
    X[j] + H_post[j] * h_out
    for j in range(N)
])                                           # [N, C]
```

这样写以后，旧 stream 不会被 $$H^{res}$$ 反复乘来乘去；stream 之间的差异主要来自每层不同的 read/write 系数和不同历史写入，而不是显式的 stream mixing。它比 mHC 少掉 per-token $$N\times N$$ mixing 和 Sinkhorn projection，也更容易保住 identity path；代价是旧 state 之间没有直接的线性 mixing；经由共享 sublayer 的间接交互仍然存在。

xHC 保持 dense read，但只对 $$k$$ 条 active streams 做 residual mixing 和 write-back。这样 $$N$$ 主要控制 residual memory capacity，$$k$$ 主要控制每层更新成本。

```python
# X: [N, C], k << N

z_all = rmsnorm(vec(X))                       # [NC]
H_pre = sigmoid(alpha_pre * (z_all @ W_pre) + b_pre)  # [N]
h_in = sum_i(H_pre[i] * X[i])                 # [C]
h_out = F(norm(h_in))                       # [C]

# Attention side does not use temporal augmentation.
# MLP / MoE side produces richer write-back components.
# sequence_outputs[t] is h_out; other entries are this sublayer's outputs at other tokens.
if is_mlp_or_moe:
    out_aug = temporal_aug(sequence_outputs)   # [T, K_r, C], causal conv
    if use_gram_schmidt:                     # paper uses GS with AdamW
        out_aug = modified_gs(out_aug)       # orthogonalize along K_r, per token
    out_aug = out_aug[t]                     # [K_r, C] for this token
else:
    out_aug = h_out[None, :]                  # [1, C], K_r = 1

z_router = layernorm(vec(X))                 # [NC], router uses LayerNorm
score = sigmoid(z_router @ W_router)         # [N]
routed_local = topk(score[not_fixed], k - m)  # [k - m], local positions
routed_idx = map_to_stream_idx(routed_local)  # [k - m], global stream indices
active_idx = concat(fixed_idx, routed_idx)    # [k]
p = concat(ones(m), score[routed_idx])        # [k], fixed streams have weight 1

X_act = X[active_idx]                         # [k, C]
z_act = rmsnorm(vec(X_act))                   # [kC]

H_res = sinkhorn(exp(alpha_res * mat(z_act @ W_res) + b_res))
# z_act @ W_res: [k^2], H_res: [k, k]

H_post = 2 * sigmoid(alpha_post * mat(z_act @ W_post) + b_post)
# z_act @ W_post: [k * K_r], H_post: [k, K_r]
# for Attention side, K_r = 1 and H_post is [k, 1]

X_mixed = stack([
    sum_i(H_res[j, i] * X_act[i])
    for j in range(k)
])                                            # [k, C]
write = stack([
    p[j] * sum_r(H_post[j, r] * out_aug[r])
    for j in range(k)
])                                            # [k, C]

X_next = X.copy()
X_next[active_idx] = X_mixed + write          # inactive streams unchanged
```

在模型最后，HC family 还需要把 multi-stream hidden reduce 回普通 head 接受的单条 hidden。mHC / oHC / xHC 在语言模型里通常是在 final RMSNorm / unembedding 前沿 stream 维求和；Chimera 这类 diffusion backbone 的 iHC 则写成最终把 $$M$$ 条 stream 平均回单条表示。两者在 final RMSNorm 前只差一个固定尺度，但实际等价性仍受 norm epsilon、后续层和 checkpoint recipe 影响；复现时应沿用原实现。输出头接收的仍是 $$C$$ 维表示：

```python
# X_final: [N, C]
h_final = sum_i(X_final[i])                   # [C]
logits = lm_head(final_norm(h_final))
```

Full AttnRes 不扩展 residual stream，而是维护历史 depth sources。注意这里的 `S` 是沿层深缓存的 source representations，不是 HC family 的 stream slots；它的长度随层数增长。

```python
# S: [M, C], visible depth sources before layer l
# S[0] is usually the embedding/base state; later entries follow the paper's source definition
# w_l: [C], learned pseudo-query for this layer/sublayer

K = rmsnorm(S)                       # [M, C], keys
V = S                                # [M, C], values

score = K @ w_l                     # [M]
alpha = softmax(score, dim="depth")  # [M], sources compete along depth

h_route = sum_i(alpha[i] * V[i])     # [C], replacement-style routed state
h_out = F(norm(h_route))             # [C], current residual-branch source

S_next = append(S, h_out)            # [M + 1, C], visible to later layers
# next layer will recompute its input by depth-attending over S_next
```

Block AttnRes 把细粒度 branch outputs 累计成 block summaries。**每个 sublayer 前仍然执行 depth attention**，可见 sources 是已完成的 blocks，加上当前 block 非空的 partial sum。只有“把 partial sum 封口成一个新 source”发生在 block boundary；不能把它理解为只在边界做 attention。source 数约为 $$N_{\mathrm{blk}}\approx L/G$$。

```python
# S_blk: [M, C], completed block summaries; include embedding/base as the first source
# partial_blk: None or [C], current block partial sum

if partial_blk is None:
    S_visible = S_blk                         # [M, C], first sublayer in a block
else:
    S_visible = concat(S_blk, [partial_blk])  # [M + 1, C]
q = learned_query[l]                          # [C]
alpha = softmax(rmsnorm(S_visible) @ q, dim=0)  # [M] or [M + 1]
h_route = alpha @ S_visible                   # [C], replacement-style input

h_out = F(norm(h_route))             # [C]
partial_blk = h_out if partial_blk is None else partial_blk + h_out

if is_block_end(l, G):
    S_blk = append(S_blk, partial_blk)
    partial_blk = None
```

Delta AttnRes 针对 cumulative-source AttnRes 的冗余问题，把 source 换成 per-sublayer deltas，并且采用 additive routing：routing 结果作为额外信息加回当前 residual stream，而不直接替换它。

```python
# h: [C], D: [M, C] previous sublayer deltas
if len(D) == 0:
    routed_delta = 0                 # [C], or use an explicit null source
else:
    q = learned_query[l]             # [C]
    alpha = softmax(rmsnorm(D) @ q, dim=0)  # [M]
    routed_delta = alpha @ D         # [C]
h_in = h + routed_delta              # [C], additive routing
h_out = F(norm(h_in))                # [C]
h_next = h + h_out                   # [C], keep original residual update
D_next = append(D, h_out)            # [M + 1, C]
```

Delta Block 将一个 block 内的 branch outputs 累计成 delta source。下面按论文 Appendix C 的示意，保留已完成 blocks 和当前非空 partial delta；主 residual state `h` 持续累加，不在边界清零。

```python
# h: [C], D_blk: completed block deltas, partial_delta: None or [C]
visible = D_blk if partial_delta is None else concat(D_blk, [partial_delta])
if len(visible) == 0:
    routed_delta = zeros_like(h)
else:
    q = learned_query[l]                              # [C]
    alpha = softmax(rmsnorm(visible) @ q, dim=0)       # [M] or [M + 1]
    routed_delta = alpha @ visible                    # [C]

h_in = h + routed_delta                              # [C]
h_out = F(norm(h_in))                                # [C]
h_next = h + h_out                                   # [C], base residual persists
partial_delta = h_out if partial_delta is None else partial_delta + h_out

if is_block_end(l, G):
    D_blk = append(D_blk, partial_delta)
    partial_delta = None                             # only this accumulator resets
```

这段伪代码按论文 Appendix C 表达 source 与 residual 的关系，当前非空 partial delta 参与路由，省略了张量打包与 checkpoint 初始化。若实现只路由已完成的 blocks，或额外加入 base / Null Source，应分别注明 source 集合，因为这些选择会改变 routing 的输入。

## 3. Baseline：标准 Residual 的能力边界

如果令每个 sublayer 的增量为：

$$
v_i = F_i(\mathrm{Norm}(h_i)),
$$

那么第 $$l$$ 个 sublayer 之前的状态是：

$$
h_l = h_0 + \sum_{i=0}^{l-1} v_i.
$$

这里所有历史增量的权重都是 1。单独的固定加法算子没有以下显式能力：

1. **选择性读取**：第 $$l$$ 层不能说“我更想看第 5 层、第 17 层，而不是所有历史平均混合后的结果”。
2. **结构化写回**：每层只能把新增量写回同一条 stream。
3. **历史分量竞争**：不同层的贡献不会通过 softmax / gate 竞争概率质量。
4. **多槽位记忆**：所有信息被压进同一个 $$C$$ 维 hidden state。

对应到方法设计，多 stream 路线把一个 hidden state 变成多个 hidden state；depth attention 路线则让每层从历史 source 中动态选择。

### 3.1 标准 Residual 为什么稳定

标准 residual 稳在一条近似恒等的直通路径。以 PreNorm sublayer 为例：

$$
h_{l+1}=h_l + F_l(\mathrm{Norm}(h_l)).
$$

如果初始化早期 $$F_l$$ 的输出较小，或者某个 sublayer 暂时没有学到有效变换，那么这一层至少近似满足：

$$
h_{l+1}\approx h_l.
$$

这就是 identity path。浅层 hidden 不需要经过一个新矩阵才能传到深层，而是默认被保留下来。反向传播时也类似，梯度可以沿着这条加法路径绕过复杂的 attention / MLP Jacobian。这条路径降低了深层优化的难度，但完整 Jacobian 仍是 $$I+J_{F\circ\mathrm{Norm}}$$；有 identity 项不等于整个网络自动不爆炸、不衰减。

stable propagation 说的是另一面：信息沿着很多层传播时，尺度不能系统性爆炸或衰减。标准 residual 展开后更像：

$$
h_L = h_0 + \sum_{i=0}^{L-1} F_i(\mathrm{Norm}(h_i)).
$$

旧信息 $$h_0$$ 不会每层都被乘一个新的线性变换。新增信息会累加进来，但旧 residual state 的主路径仍然保持直接传递，这让深层网络在初始化和训练早期有一个比较稳的动力学。

### 3.2 自由 $$H^{res}$$ 为什么危险

HC 把一条 residual stream 扩成多条以后，旧信息的传播形式变成：

$$
X_{l+1}=H_l^{res}X_l + H_l^{post}\mathrm{out}_l.
$$

如果先忽略每层新产生的 $$\mathrm{out}_l$$，只考虑已有 residual state 怎么往后传，就得到：

$$
X_L \approx H_{L-1}^{res}H_{L-2}^{res}\cdots H_0^{res}X_0.
$$

旧信息的传播形式随之改变：标准 residual 的旧信息是直接加法保留；HC 里旧信息每过一层都要被 $$H_l^{res}$$ 重新混合一次。若 $$H_l^{res}$$ 是完全自由矩阵，它当然表达力更大，但也带来三个隐患。

第一，identity path 变弱。除非每层 $$H_l^{res}$$ 都接近 identity，否则浅层信息不再是“原样传下去”，而是每层都被改写一次。残差通路会越来越像一个 recurrent linear system。

第二，尺度不可控。自由矩阵的某些 singular value 可能大于 1，也可能小于 1。几十层、上百层连乘后，某些方向会被持续放大，某些方向会被压小，训练早期的小偏差也可能被深度放大。

第三，stream 之间可能相互抵消。自由矩阵允许负权重，例如 $$x'_1=x_1-x_2$$。这种操作有表达力，但在大规模预训练里也更容易破坏 residual memory 的保守性，尤其是在初始化和 loss spike 附近。

mHC 的动机正是在这里：保留 stream mixing，同时把 $$H_l^{res}$$ 限制到一个更像 residual 的稳定区域。doubly stochastic matrix 的行和、列和都为 1，且元素非负。它允许保守的 stream mixing，限制线性直通项的扩张；它仍可能压缩 stream 差异，也可能混合符号相反的激活而发生抵消。更重要的是，doubly stochastic matrices 的乘积仍然是 doubly stochastic，所以这个稳定约束沿深度组合后仍然成立。

mHC 希望借助这个约束，在多 stream 版本里重新找回标准 residual 最重要的性质，让旧信息拥有一条尺度可控、近似保守的传播路径。

## 4. HC / iHC / mHC / oHC：多 Residual Stream 路线

### 4.1 Hyper-Connections: 从单 stream 到多 stream

Hyper-Connections 的基本想法是把 residual stream 从 $$C$$ 维扩成 $$N \times C$$：

$$
X_l = [x_{l,1}, x_{l,2}, \ldots, x_{l,N}]^\top \in \mathbb{R}^{N \times C}.
$$

这些 stream 从哪里来？在网络入口，原来的 token embedding / hidden vector $$h_0\in\mathbb{R}^C$$ 会被 lift 成一个 hyper hidden matrix。最朴素的写法就是复制 $$N$$ 份：

$$
X_0 = [h_0,h_0,\ldots,h_0]^\top \in \mathbb{R}^{N\times C}.
$$

$$N$$ 是独立的结构超参，与 attention head 数、Transformer 层数或 MoE expert 数没有固定的推导关系。HC / mHC 常把它叫 expansion rate；mHC 大规模实验里常用 $$N=4$$，xHC 为了继续扩大 residual memory capacity，把总 stream 数推到 $$N=16$$，但每层只更新 $$k=4$$ 条 active streams。

每层不再直接对单个 $$h_l$$ 做 sublayer，而是先从多 stream 中读出一个输入：

$$
\mathrm{input}_l = H_l^{pre} X_l.
$$

经过 layer function：

$$
\mathrm{out}_l = F_l(\mathrm{Norm}(\mathrm{input}_l)).
$$

再写回多 stream：

$$
X_{l+1} = H_l^{res} X_l + H_l^{post} \mathrm{out}_l.
$$

其中：

- $$H_l^{pre}$$ 控制从多个 stream 读出什么。
- $$H_l^{res}$$ 控制旧 residual streams 之间如何混合。
- $$H_l^{post}$$ 控制新 layer output 如何写回多个 stream。

刚初始化时，多条 stream 可以是同一个 $$h_0$$ 的拷贝；训练开始后，$$H_l^{pre}$$、$$H_l^{res}$$、$$H_l^{post}$$ 会让它们逐层分化。某条 stream 可能更偏向保留浅层信息，另一条可能吸收更多近层 FFN 输出，但这是一种递归混合产生的隐式深度记忆，不是显式按层索引的历史缓存。

到模型输出处，multi-stream state 还要还原成普通语言模型头能处理的单 hidden。mHC / oHC / xHC 论文采用的做法很直接：在 final RMSNorm 和 unembedding 之前，把 $$N$$ 条 stream 沿 stream 维求和，得到 $$C$$ 维表示。多 stream 是模型内部的 residual memory 形态，LM head 的输入宽度仍是 $$C$$。

![mHC 论文 Figure 1：标准 residual、HC 和 mHC 的 residual stream 结构](/assets/img/blog/attention-residual/mhc_figure1_residual_paradigms.png)

*图 1：mHC 论文 Figure 1 裁图，对比标准 residual、HC 和 mHC 的 residual stream 结构。*

但原始 HC 的风险也很明显：

- $$H_l^{res}$$ 是自由矩阵时，可能破坏 residual identity mapping。
- 多层自由矩阵连乘后，signal norm 可能放大或衰减。
- 多 stream state 会引入额外 memory traffic。
- 如果没有强约束，训练大模型时稳定性会比标准 residual 差。

### 4.2 mHC: 用 Birkhoff Polytope 恢复稳定性

[mHC](https://arxiv.org/abs/2512.24880v1) 的核心修正，是把 $$H_l^{res}$$ 从自由矩阵改成 doubly stochastic matrix。

约束为：

$$
\begin{aligned}
H_l^{res}\mathbf{1} &= \mathbf{1},\\
\mathbf{1}^{\top}H_l^{res} &= \mathbf{1}^{\top},\\
H_l^{res} &\ge 0.
\end{aligned}
$$

也就是说，doubly stochastic 同时满足 row-stochastic 和 column-stochastic。行和为 1，表示每条新的 stream 都是旧 streams 的非负加权平均，不会凭空把尺度放大成 $$2x_1+3x_2$$。列和为 1，表示某条旧 stream 的总贡献不会被复制到很多输出 stream 里反复放大。非负性排除了通过负系数执行 $$x_1-x_2$$ 这类相减，但激活本身仍可异号，不能据此断言混合后没有抵消。

这个集合是 Birkhoff polytope，即 permutation matrices 的凸包。可以把它理解成介于“只交换 stream 位置”和“在 stream 之间做平滑平均”之间的一类矩阵。放在 $$H_l^{res}$$ 上，它不再允许任意线性变换，而是把 stream mixing 收进“重新分配已有信息”的范围里。

双随机约束的准确性质是 **non-expansive（非扩张）**，而非对所有方向保范数：

$$
\|H^{res}\|_2=1,\qquad \|H^{res}X\|_F\leq\|X\|_F.
$$

例如 $$H=\frac{1}{N}\mathbf{1}\mathbf{1}^{\top}$$ 会把所有 streams 变成均值，完全抹去 stream 差异。它不会放大整体范数，却远不是 identity。双随机矩阵的乘积仍然双随机，因此固定 mixer 的跨层线性传播保持这一性质。

还需要区分 mixer 与完整网络：$$H^{res}$$ 本身依赖输入，完整 Jacobian 包含 mapping 对输入的导数，以及 sublayer / write-back 的导数。上述范数约束针对给定 $$H^{res}$$ 的线性直通项，不能直接当成整个网络梯度有界的证明。有限轮 Sinkhorn 也只近似满足约束，需要监控行、列和误差。

### 4.3 iHC: 固定 $$H^{res}=I$$

iHC，也就是 identity Hyper-Connections，是 mHC 公式里的一个极简特例：

$$
H_l^{res}=I_N.
$$

每层仍然有输入相关的 $$H_l^{pre}$$ 和 $$H_l^{post}$$，所以 sublayer 的输入来自多条 stream 的动态聚合，并不固定绑定某一条 stream；新产生的 $$\mathrm{out}_l$$ 也可以按不同系数写回多条 stream。但旧 stream 自身不再经过一个动态矩阵混合：

$$
X_{l+1}=X_l+H_l^{post}\mathrm{out}_l.
$$

iHC 保留多 stream read/write，省去旧 state 之间的直接 mixing：

- 旧信息沿 stream 维有一条精确 identity transition，不会被自由矩阵放大，也不会被 doubly stochastic mixer 收缩。
- 不需要生成 $$N^2$$ 个 residual mixing logits，也不需要 Sinkhorn iterations。
- 仍然能通过 $$H^{pre}$$ 和 $$H^{post}$$ 让不同 stream 承担不同写入历史。

[Chimera](https://arxiv.org/abs/2607.28611) 在视觉 diffusion backbone 中给出了 iHC 的 read/write 定义；[Hy4-preview 官方说明](https://github.com/Tencent-Hunyuan/Hy4-preview) 也明确使用 iHC，配置为 4 条 residual streams。两者说明这一简化有实际使用场景，但不能把不同 backbone 的效果当作语言模型中 iHC 相对 mHC 的直接消融。Chimera 的输出端采用 stream average。

iHC 的限制也很清楚：stream 之间没有显式交换。某条 stream 想影响另一条 stream，只能绕一圈：先被后续 sublayer 的 read 聚合读出来，经过 Attention / MLP 产生新 output，再通过 write 写进其他 stream。也就是说，iHC 很适合作为低成本 multi-stream baseline，但它没有 mHC / oHC 那种直接的 stream mixing 通道。

### 4.4 oHC：用正交矩阵做 stream mixing

[oHC 论文](https://arxiv.org/abs/2609.02672) 从 mHC 的收缩性质出发：双随机约束固定了均值方向，却可能不断衰减 stream 之间的差异；iHC 避开收缩，同时也放弃直接 mixing。

oHC 将 residual matrix 约束到特殊正交群：

$$
H_l^{res}\in SO(N),\qquad (H_l^{res})^\top H_l^{res}=I,\qquad \det(H_l^{res})=1.
$$

所有 singular values 都等于 1，因此对给定 mixer：

$$
\|H_l^{res}X\|_F=\|X\|_F.
$$

但这不表示任意两条 stream 的距离保持不变。左乘 $$H$$ 是在 stream 索引空间中旋转，不是在每条 stream 的 feature 空间施加同一个旋转。令 $$J=\frac{1}{N}\mathbf{1}\mathbf{1}^{\top}$$，stream 的 centered difference energy 是 $$E_\perp(X)=\|(I-J)X\|_F^2$$；一般的正交矩阵不与 $$J$$ 交换，所以未必有 $$E_\perp(HX)=E_\perp(X)$$。

论文进一步分析了均值与差异子空间之间的交换：mHC / iHC 不会由 residual transport 把均值转成差异，oHC 则允许这条通道。它的动机应理解为**保留整体等距传播，同时允许更丰富的 stream 交互**，而不是每一步严格保持所有 stream 间距。

在常用的 $$N=4$$ 设置下，oHC 用一对单位四元数参数化 $$SO(4)$$，不需要 Sinkhorn 迭代。论文主实验是 3.9B total / 0.4B active 的 latent-MoE，训练 73B tokens。这个结果适合支持 iHC/mHC/oHC 的受控比较，尚不能替代更大规模长训的验证。

![oHC 论文 Figure 2：mHC、iHC 和 oHC 对 stream 的影响](/assets/img/blog/attention-residual/ohc_figure2_constraints.png)

*图 2：oHC 论文 Figure 2 的概念示意。mHC 的 mixing 可能压缩差异，iHC 保留直通项，oHC 引入整体等距的 stream mixing，并允许均值与差异子空间交换。*

### 4.5 mHC 的参数化

mHC 通常先从 flattened multi-stream state 生成三个 mapping：

$$
\bar{x}_l = \mathrm{vec}(X_l) \in \mathbb{R}^{1 \times NC}.
$$

动态映射：

$$
\begin{aligned}
\tilde{H}_l^{pre} &= \alpha^{pre} \cdot \mathrm{RMSNorm}(\bar{x}_l) W^{pre} + b^{pre},\\
\tilde{H}_l^{post} &= \alpha^{post} \cdot \mathrm{RMSNorm}(\bar{x}_l) W^{post} + b^{post},\\
\tilde{H}_l^{res} &= \alpha^{res} \cdot \mathrm{mat}(\mathrm{RMSNorm}(\bar{x}_l) W^{res}) + b^{res}.
\end{aligned}
$$

把关键 shape 展开后是：

| 张量 | Shape | 说明 |
|---|---:|---|
| $$X_l$$ | $$[N,C]$$ | 当前 token 的 multi-stream state |
| $$\bar{x}_l=\mathrm{vec}(X_l)$$ | $$[1,NC]$$ | 把 stream 维和 hidden 维展平 |
| $$W^{pre}$$ | $$[NC,N]$$ | 生成 read mapping logits |
| $$W^{post}$$ | $$[NC,N]$$ | 生成 write mapping logits；实现里常先存成 $$[1,N]$$ |
| $$W^{res}$$ | $$[NC,N^2]$$ | 生成 residual mixing logits |
| $$\bar{x}'_l W^{pre}$$ | $$[1,N]$$ | 对应 $$\tilde{H}_l^{pre}$$ |
| $$\bar{x}'_l W^{post}$$ | $$[1,N]$$ | 对应 $$\tilde{H}_l^{post}$$，写回时可转成 $$[N,1]$$ |
| $$\bar{x}'_l W^{res}$$ | $$[1,N^2]$$ | 还不是矩阵，只是一串 $$N^2$$ 个 logits |
| $$\mathrm{mat}(\bar{x}'_l W^{res})$$ | $$[N,N]$$ | 把 $$N^2$$ 个 logits reshape 成 stream mixing matrix |
| $$b^{pre},b^{post}$$ | $$[1,N]$$ | 静态 read / write bias |
| $$b^{res}$$ | $$[N,N]$$ | 静态 residual mixing bias |

`mat()` 只做 reshape：把线性投影得到的 $$N^2$$ 个 logits 排成 $$N\times N$$ 矩阵，不引入参数。随后取 $$\exp$$ 得到正矩阵，再通过 Sinkhorn-Knopp 交替做行、列归一化，近似满足双随机约束。

这里的 $$\alpha^{pre}$$、$$\alpha^{post}$$、$$\alpha^{res}$$ 要和原始 HC 连接矩阵里的 $$\alpha_{i,j}$$ 区分开。它们是动态映射的 learnable gating scalars，论文实验里初始化为 0.01，随后和其他参数一起通过反向传播更新。它们控制 input-dependent term 的初始幅度：训练刚开始时，mapping 主要由 bias / static part 决定，接近稳定初始化；训练推进后，动态项再逐渐打开，让 $$H^{pre}$$、$$H^{post}$$、$$H^{res}$$ 变成输入相关的路由。

$$H_l^{post}$$ 前面的系数 2 也和初始化尺度有关。$$\sigma(\cdot)$$ 的输出范围是 $$(0,1)$$，如果直接用 sigmoid，零 logit 会给出 0.5，相对单位写回会缩小新贡献；而 $$2\sigma(\cdot)$$ 的范围是 $$(0,2)$$，零 logit 对应 1。这样 post-mapping 可以围绕标准 residual 的写回尺度波动：小于 1 表示衰减，接近 1 表示按原尺度写回，大于 1 表示轻微放大。pre-mapping 负责读出输入，各系数被约束在 $$(0,1)$$，但总读出尺度还取决于 $$N$$ 和 stream 内容；post-mapping 负责把新产生的 layer output 写回 residual streams，需要保留“写回一个完整 residual update”的自由度。

再做约束：

$$
\begin{aligned}
H_l^{pre} &= \sigma(\tilde{H}_l^{pre}),\\
H_l^{post} &= 2\sigma(\tilde{H}_l^{post}),\\
H_l^{res} &= \mathrm{SK}_{T}(\exp(\tilde{H}_l^{res})).
\end{aligned}
$$

这里 $$H_l^{pre}$$ 和 $$H_l^{post}$$ 的系数非负，但激活仍可为负；非负系数不排除激活之间的抵消。$$\mathrm{SK}_T$$ 表示对正矩阵执行 $$T$$ 轮交替行、列归一化，近似落入双随机集合。$$H^{pre}$$ 的各项在 $$(0,1)$$ 内，却不要求总和为 1，所以它不是概率分布，也不保证读出范数不增。

mHC 论文里的常用设置：

| 超参 | 典型值 | 说明 |
|---|---:|---|
| stream expansion $$N$$ | 4 | mHC 大规模实验主设置 |
| gating init $$\alpha$$ | 0.01 | 让动态项初期较弱 |
| Sinkhorn iterations | 20 | 近似 doubly stochastic |

### 4.6 mHC 的工程代价

mHC 的主要压力不在 FLOPs，而在 residual stream 相关的 memory access 和 kernel launch。

额外操作包括：

- 对 $$NC$$ 维 flattened state 做 RMSNorm / projection。
- 生成 $$H^{pre}$$、$$H^{post}$$、$$H^{res}$$。
- 对 $$H^{res}$$ 做 Sinkhorn iterations。
- 应用多 stream mixing。
- backward 时保存或重算这些中间量。

因此 mHC 的系统优化重点是：

- fused mapping generation。
- fused Sinkhorn kernel。
- fused residual mixing + write-back。
- activation recompute。
- pipeline stage boundary 上 overlap communication / recompute。

DeepSeek 报告其内部大规模实现中，$$N=4$$ 的 mHC 训练额外时间开销约为 6.7%。这是特定系统优化后的测量，迁移到另一套 kernel 与并行配置时需要重新 profile。

### 4.7 三种约束的差别

iHC 省去 residual mixer，保留动态 read/write；mHC 允许非负的 stream 重分配，同时限制扩张；oHC 保留整体范数，并允许均值与差异子空间交换。三者改变的是旧 state 的传播方式，每个 sublayer 仍只提供一个 write-back signal。

xHC 论文进一步观察到，mHC 的 $$N$$ 从 4 扩到 8/16 时，质量收益趋于饱和，mapping 生成成本却快速上升。xHC 针对的是这组 large-$$N$$ 的信息与成本瓶颈，与 oHC 讨论的约束几何有所区别。

## 5. xHC：扩展 stream 数量的收益与成本

xHC 的全称是 **Expanded Hyper-Connections**。这里的 xHC 指 [2026-07-16 arXiv v1 技术报告](https://arxiv.org/abs/2607.14530v1)，作者单位包含 Dots Studio, Xiaohongshu Inc.。它继承 mHC 的多 stream residual 设计，但认为 mHC 在扩大 $$N$$ 时遇到两个瓶颈：

1. **Information bottleneck**：stream 数变多了，但每个 sublayer 仍然只提供一个 write-back output。
2. **Cost bottleneck**：dense $$H^{res}\in\mathbb{R}^{N\times N}$$ 的生成和混合成本随 $$N$$ 快速增长。

xHC 的核心设计就是对这两个瓶颈分别下手：

| 瓶颈 | xHC 的对应设计 |
|---|---|
| Information bottleneck | Temporal Feature Augmentation |
| Cost bottleneck | Sparse Residual-Stream Architecture |

![xHC 论文 Figure 3：标准 residual、mHC 和 xHC 对比](/assets/img/blog/attention-residual/xhc_figure3_overview.png)

*图 3：xHC 论文 Figure 3 裁图，对比标准 residual、mHC 和 xHC。xHC 的关键变化是 dense read all streams，但只 sparse write $$k=4$$ 个 active streams。*

### 5.1 为什么 mHC 扩大 N 会饱和

在 mHC 中，虽然 state 有 $$N$$ 个 stream，但每个 sublayer 输出仍是一个 $$\mathrm{out}_l \in \mathbb{R}^C$$。写回时，多个 stream 只是用不同系数组合这个单一 output：

$$
\Delta X_i \propto H_{i}^{post}\mathrm{out}_l.
$$

当 $$N$$ 很小时，例如 $$N=4$$，这已经足够产生不同历史路径。

当 $$N$$ 扩到 16，问题变成：

- stream 槽位变多。
- 每层新信息来源仍然只有一个方向。
- 多个 stream 之间更容易形成冗余组合。

新增写回在 stream 维上的秩受限，这就是论文讨论的信息瓶颈。

另一方面，mHC 的 dynamic residual mapping 需要生成 $$N^2$$ 个系数，而且输入是 $$NC$$ 维。粗略看，相关 projection 成本会随 $$N$$ 快速上升。xHC 论文把这个总结为 large-$$N$$ expansion 的 cost bottleneck。

论文中的关键观察是：

| 方法 | 从 $$N=4$$ 到 $$N=16$$ 的收益 | 额外训练 FLOPs |
|---|---:|---:|
| mHC | loss 降低约 0.006 | +32% |
| xHC | loss 降低约 0.012 | +4% |

这组 2.5B MoE 扫描表明，xHC 在扩展到 $$N=16$$ 时仍有 loss 收益，同时新增 FLOPs 明显少于 mHC。

### 5.2 Temporal Feature Augmentation：给 write-back 更多方向

xHC 不直接为每个 stream 单独跑一个 sublayer，因为那会把 attention/MLP 主计算乘上 $$N$$。它选择一个便宜得多的方案：对 layer output 做轻量 causal depthwise convolution，生成多个局部上下文增强的 write-back components。

设 sublayer output 带 sequence 维度为：

$$
\mathrm{out}\in\mathbb{R}^{S\times C}.
$$

xHC 生成：

$$
\mathrm{out}^{aug} =
[
\mathrm{out};
\mathrm{DWConv}_{\kappa_1}(\mathrm{out});
\ldots;
\mathrm{DWConv}_{\kappa_r}(\mathrm{out})
]
\in \mathbb{R}^{S\times K_r\times C}.
$$

主设置中：

| 参数 | 值 |
|---|---:|
| conv branch 数 $$r$$ | 3 |
| kernel sizes | $$\{4,8,12\}$$ |
| write-back components $$K_r$$ | 4 |
| conv 类型 | causal depthwise 1D conv |

可以这样理解：attention / MLP 的输出本身是一个 write-back 方向；邻近 token 的 causal depthwise conv 输出提供不同局部上下文尺度；多个 component 让不同 active streams 有机会写入不同信息，而不是都围绕同一个 output 做缩放。

xHC 只在 **MLP / MoE FFN 后**做 temporal augmentation。原因是 attention 本身已经做 token mixing；论文经验上，在 attention 后加 temporal augmentation 会更容易训练不稳。

### 5.3 Gram-Schmidt Orthogonalization：避免增强分量全都指向同一方向

depthwise conv 是逐 channel 的局部滤波，生成的分量可能和原始 $$\mathrm{out}$$ 高度相关。如果直接把这些 component 交给 $$H^{post}$$，可能只是重复放大同一个方向。

xHC 因此对每个 token 的 $$K_r$$ 个 component 做 modified Gram–Schmidt。以 $$v_1=\mathrm{out}$$ 为起点，第 $$j$$ 个卷积分量 $$g_j$$ 依次减去已有方向：

$$
\begin{aligned}
u^{(0)}&=g_j,\\
u^{(i)}&=u^{(i-1)}-\frac{\langle u^{(i-1)},v_i\rangle}{\langle v_i,v_i\rangle+\varepsilon}v_i,\quad i=1,\ldots,j,\\
v_{j+1}&=u^{(j)}.
\end{aligned}
$$

这一步降低写回分量的方向冗余；它不是单位化，不能单独保证输出范数或完整训练梯度有界。实现还要处理近零向量，并确认归约只沿 channel 维进行，不引入未来 token 信息。

在论文的 Muon 实验中，作者移除了 GS，并只对 backbone 的 Attention / MLP 投影矩阵使用 Muon；xHC 的 routing / mapping 参数仍用 AdamW。这是该实验配置中的经验结果。Muon 约束的是优化器更新，GS 处理的是前向激活，两者不在同一空间，不能据此认为 Muon 普遍替代了激活正交化。

### 5.4 Sparse Residual-Stream Architecture：dense read + sparse write

xHC 不再每层更新所有 $$N$$ 个 streams。它保留 dense read：每层仍然读取所有 $$N$$ 个 streams；同时改成 sparse write：每层只更新 $$k$$ 个 active streams。

主设置：

| 参数 | 值 | 说明 |
|---|---:|---|
| $$N$$ | 16 | residual stream 总数 |
| $$k$$ | 4 | 每个 sublayer active streams 数 |
| fixed streams | 2 | 始终 active 的稳定通路 |
| routed streams | 2 | 由 router TopK 选择 |
| router activation | sigmoid | 减轻 softmax 的分数竞争，仍需检查 stream 使用率 |

router 看完整 multi-stream state：

$$
s = \sigma(\mathrm{LN}(\mathrm{flatten}(X_l))W^r)\in\mathbb{R}^N.
$$

active set 由 fixed streams 和 TopK routed streams 组成：

$$
I = \mathrm{FixedIdx}\cup \mathrm{TopK}_{i\notin \mathrm{FixedIdx}}(s_i, k-m).
$$

#### Dense read

xHC 保留 dense read：

$$
\mathrm{input}_l = \sum_{i=1}^{N} H_{l,i}^{pre}x_{l,i}.
$$

如果 read 也 sparse，可能出现信息暂时不可见：某个 stream 在上一层被更新，但下一层没被选中读取，那么这条信息路径有效长度会变短。

Dense read 保证：

- 所有 stream 的信息都能被当前 layer 看到。
- 即使只更新 $$k$$ 个 stream，未更新 stream 仍参与后续计算。
- sparse write 不会把 residual memory 切成互不通信的碎片。

#### Sparse write

xHC 只对 active streams 生成和应用 $$H^{res}$$ / $$H^{post}$$：

$$
H_l^{res} = \mathrm{SK}(f^{res}(X_{active}))\in\mathbb{R}^{k\times k},
$$

$$
H_l^{post} = f^{post}(X_{active})\in\mathbb{R}^{k\times K_r}.
$$

写回：

$$
\Delta X_{active,j}
= p_j\sum_{r=1}^{K_r}H_{l,j,r}^{post}\mathrm{out}^{aug}_r.
$$

active streams 更新为：

$$
X_{active}^{new}
= H_l^{res}X_{active}+\Delta X_{active}.
$$

inactive streams 保持不变。

稀疏更新主要节省 residual-mapping 的生成成本：从 $$O(N^3C)$$ 降到 $$O(k^3C)$$。其中 $$N$$ 控制 residual 槽位数量，$$k$$ 控制活跃 mixing 的规模。完整 xHC 仍需读取全部 streams、生成 router / pre-mapping，并存储 $$NC$$ state，因此总成本仍然依赖 $$N$$。

### 5.5 xHC 的主结果

xHC 论文在 18B 和 28B MoE 上比较 vanilla、mHC 和 xHC。mHC 使用 $$N=4$$，xHC 使用 $$N=16,k=4$$，训练 FLOPs 近似可比。

| Scale | Vanilla Avg | mHC Avg | xHC Avg | xHC vs mHC |
|---|---:|---:|---:|---:|
| 18B MoE | 40.6 | 44.8 | 48.8 | +4.0 |
| 28B MoE | 47.8 | 50.5 | 53.6 | +3.1 |

代表性 benchmark 提升：

| Scale | 相对 mHC 的代表性提升 |
|---|---|
| 18B | ARC-Challenge +5.9, BBH +5.8, C3 +5.6, HumanEval +6.1 |
| 28B | CommonsenseQA +5.7, HumanEval +4.3, C3 +3.8, MMLU +3.7 |

这些结果表明，xHC 的收益没有停在 loss 上，也能转化到 reasoning、knowledge、code 和中文任务。

### 5.6 xHC 消融：两个部件都必要

10B MoE 消融结果可以概括为：

| 方法 | $$N$$ | Temp Aug | Sparse | Dense Read | $$k$$ | Val Loss |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | - | - | - | - | - | 2.029 |
| mHC | 4 | - | - | - | - | 2.004 |
| mHC | 16 | - | - | - | - | 1.998 |
| mHC + Temp Aug | 16 | yes | no | - | - | 1.984 |
| xHC | 16 | yes | yes | yes | 4 | 1.983 |

这组消融将信息与成本两个因素分开了：

- 从 mHC $$N=16$$ 到 mHC + Temp Aug，loss 从 1.998 到 1.984，支持了该配置下写回信息不足的解释。
- 再加入 sparse architecture，loss 基本保持 1.983，但 FLOPs overhead 从约 20.1% 降到约 3.3%，说明 sparse write 主要解决成本问题。
- 去掉 Dense Read 或 fixed streams 都会退化，说明 sparse update 需要稳定的信息通路。
- $$k=2$$ 不够，$$k=8$$ 略好但更贵，$$k=4$$ 是主推荐折中。
- 该消融中 sigmoid router 优于 softmax；论文将其与减轻 winner-take-all 联系起来。这不保证 stream 使用均衡，仍需记录选择频率和写回强度。

### 5.7 xHC-Flash：降低 residual 访存开销

xHC 默认设置虽然 FLOPs overhead 不高，但 residual-stream operations 的 HBM I/O 不低。

论文给出的 per-token per-sublayer memory traffic 近似为：

| 方法 | 设置 | Total I/O |
|---|---|---:|
| mHC | $$N=4$$ | 34C |
| mHC | $$N=16$$ | 130C |
| xHC | $$N=16,k=4$$ | 73.5C |
| xHC-Flash | $$N=16,k=4$$ | 51C |
| xHC-Flash-4sub | $$N=16,k=4$$ | 40C |

xHC-Flash 通过跨连续 sublayers 共享 full-state read / routing，并增量维护可复用结果来摊销 I/O。表中的 $$C$$ 是标量元素访问量的估计单位，并非字节数或实测时间；dtype、读写次数、recompute 和 kernel 实现仍会影响速度。xHC-Flash 是改变共享与更新安排的模型变体，不能把它简单视为与 full xHC 逐点等价的 kernel fusion；等价性只适用于其自身设计内部可精确复用的部分。

这也解释了为什么较低的 FLOPs overhead 不保证较短的训练时间：residual I/O、kernel launch、重算和跨 stage 通信仍可能抵消计算效率上的收益。xHC-Flash 的作用，是在论文配置中进一步降低这部分访存成本。

## 6. Attention Residuals：把 Residual Sum 改成 Depth-wise Softmax Attention

和 HC family 不同，Attention Residuals 不把 hidden state 扩成 $$N$$ 条 stream，而是缓存历史 source representations，让每层沿 depth 做 attention。

![Attention Residuals 论文 Figure 1：标准 residual、Full AttnRes 和 Block AttnRes](/assets/img/blog/attention-residual/attnres_figure1_overview.png)


*图 4：Attention Residuals 论文 Figure 1 裁图，对比标准 residual、Full AttnRes 和 Block AttnRes。*

### 6.1 Full AttnRes

Full AttnRes 可以先按“depth 维度上的普通 attention”来理解。标准 residual 会把历史信息固定累加起来；Full AttnRes 则显式维护一个 source list：

$$
S_l=[s_0,s_1,\ldots,s_{l-1}],\qquad s_i\in\mathbb{R}^{C}.
$$

$$s_0$$ 通常是 token embedding / base hidden。Kimi Full AttnRes 的其余 sources 是历史 residual branch outputs；Block AttnRes 将它们累计为已完成的 block sums 与当前 partial sum。Delta 论文分析的 cumulative-source baseline 则对累计 hidden states 做 replacement routing。这三种 source 定义分别对应不同的更新过程。

第 $$l$$ 层不再直接使用上一层 hidden，而是对所有历史 source 做 softmax attention：

$$
h_l = \sum_{i<l}\alpha_{i\rightarrow l}v_i,
$$

其中 value $$v_i$$ 就是对应 source，key 由 source 归一化得到：

$$
k_i=\mathrm{RMSNorm}(s_i),\qquad v_i=s_i.
$$

query 使用每层一个 learned pseudo-query，而不依赖当前 hidden 的额外投影：

$$
q_l=w_l,\qquad w_l\in\mathbb{R}^{C}.
$$

于是 attention weight 是：

$$
\alpha_{i\rightarrow l}
=
\mathrm{softmax}_i
\left(
w_l^\top k_i
\right).
$$

RMSNorm 用来减弱 source 尺度对 logits 的影响。没有它，较大范数可能放大与 query 的点积；影响的符号还取决于方向，并非大范数一定获得更高权重。learned pseudo-query 也有一个工程上的好处：$$w_l$$ 和当前 forward activation 解耦，系统可以提前知道这一层会发起什么 depth query；如果改成 input-dependent query，表达力可能更强，但解码和流水线里的 source 访问会更难调度。

`h_route` 是本层聚合得到的输入，`h_out` 是当前 sublayer 写给后续层的新 source。下一层用自己的 pseudo-query 重新聚合更新后的列表，因此没有一条固定执行 `h_route + h_out` 的 hidden-state 递推。Block AttnRes 则将细粒度 branch outputs 累计成 block summary。

成本也要按 depth 来估。对单个 token，Full AttnRes 在第 $$l$$ 层要扫 $$l$$ 个 source，总算术量约为 $$O(L^2C)$$，缓存或跨 stage 搬运的 source 规模是 $$O(LC)$$。因为 $$L$$ 通常远小于 sequence length，算术本身不一定是主矛盾；真正麻烦的是 activation recompute 和 pipeline parallel：本来可以丢掉、之后重算的中间 hidden，现在可能要作为后续层的 depth source 保留或跨 stage 传递。

Full AttnRes 提供了完整机制和细粒度参照。Block AttnRes 把 source 从逐层 output 压到 block summaries，以更粗的粒度换取可控的通信和缓存规模。

### 6.2 Block AttnRes

Full AttnRes 的 source 数随深度增长。Block AttnRes 将相邻 branch outputs 求和成为 block source，同时在每个 sublayer 前，对 embedding、已完成 block sums 和当前非空 partial sum 执行 attention。当前 partial sum 只是本 block 的贡献，不包含之前 blocks 的累计 hidden。

| 变体 | 可见 depth sources | source 存储量，单 token |
|---|---|---|
| Full AttnRes | embedding + 之前的 branch outputs | $$O(LC)$$ |
| Block AttnRes | embedding + 完成的 blocks + 非空 partial sum | $$O(N_{\mathrm{blk}}C)$$ |

$$L$$ 按 sublayer 计，$$N_{\mathrm{blk}}\approx L/G$$。每个 sublayer 都要扫描可见 blocks，直接实现的总 routing 算术量约为 $$O(LN_{\mathrm{blk}}C)$$；batching、缓存和 online softmax 可以改善常数与访存。

当 $$G=1$$ 时，每个 branch output 都成为 source，恢复 Full 的粒度。只保留一个大 block 则接近单流累加的存储形式；由于 embedding 和 partial sum 仍以可学习权重竞争，不能无条件认定它与标准 residual 的固定加法完全等价。

Full / Block AttnRes 在输出端仍需对 source list 做 depth aggregation，生成 $$[C]$$ 表示，再送入 final norm / unembedding。最后一个 branch output 或 partial sum 只包含局部贡献；输出 query / norm 的配置由具体实现决定。

### 6.3 Kimi AttnRes 的实验信号

[Kimi Attention Residuals 报告](https://arxiv.org/abs/2603.15031)中，Block AttnRes 被集成到 Kimi Linear 48B total / 3B active 架构，并预训练 1.4T tokens。

报告中的主要结论：

- scaling law 上 Full 和 Block AttnRes 都优于 baseline。
- Block AttnRes 接近 Full AttnRes，在最大设置上差距很小。
- Block AttnRes 能缓解 PreNorm dilution，使 hidden output magnitude 更有界。
- learnable softmax depth routing 让 gradient norm 在不同层之间更均匀。
- 推理 latency overhead 在典型负载下小于 2%。

下面取 Kimi 论文 Table 2 最大的一组 scaling 配置：528M activated parameters（不含 embedding）、119B tokens、8192 context。它与前述 48B total / 3B active 的主实验是不同配置。

| 方法 | Val Loss |
|---|---:|
| Baseline | 1.719 |
| Block AttnRes | 1.693 |
| Full AttnRes | 1.692 |
| mHC-lite | 1.694 |

在 Kimi 这套训练 recipe 下，depth-wise attention residual 已经有很强竞争力。

### 6.4 AttnRes 的潜在问题

在 Delta 论文的分析口径里，replacement-style AttnRes 的 source 是 cumulative hidden states 或 block-level cumulative sums。相邻累计 states 共享历史前缀，可能产生较高相关性：

$$
h_{i+1} = h_i + v_i.
$$

Delta 论文的实验中，累计 source 的相关性随深度增加，routing 对比度随之下降。

Kimi Block AttnRes 同样压缩了 block 内部的细粒度贡献，但它保存的是各 block 的贡献之和；Full AttnRes 则直接保存 branch outputs。两者与累计 hidden-state sources 的内容不同。

### 6.5 Low-Rank AttnRes：用低维 key 选择全宽 value

[Low-Rank Attention Residuals 的 9 月修订版](https://arxiv.org/abs/2607.09694v2)将路由打分与 value 传递分开：历史 source 仍是 $$C$$ 维，但 key 和 learned pseudo-query 只需 $$r$$ 维。主方法直接取 source 的最后 $$r$$ 个坐标，不增加 key projection：

$$
k_i=\operatorname{tail}_r(s_i)\in\mathbb{R}^{r},\qquad
\alpha_{i\rightarrow l}=\operatorname{softmax}_i\bigl(q_l^\top\operatorname{RMSNorm}(k_i)\bigr),\qquad
h_l=\sum_i\alpha_{i\rightarrow l}s_i\in\mathbb{R}^{C}.
$$

这里先截取，再在 $$r$$ 维内做 RMSNorm。Block 版本也先累计全宽 partial sum，再切出它的 key。每次读取的点积打分从 $$O(MC)$$ 降到 $$O(Mr)$$；全宽 value 的缓存与加权求和仍然保留。

论文用 $$N_{\mathrm{blk}}=8,r=C/4$$ 的配置训练 20B tokens，得到：

| 规模 | AttnRes Val Loss | LR-AttnRes Val Loss | AttnRes 平均准确率 | LR-AttnRes 平均准确率 |
|---|---:|---:|---:|---:|
| 1B | 2.800 | 2.780 | 41.87 | 42.27 |
| 4B | 2.738 | 2.706 | 42.97 | 43.54 |

在相同 FAST-AttnRes kernel 下，仅将路由宽度从 $$C$$ 缩到 $$C/4$$，完整训练 step 的吞吐提升为 0.59%–1.20%。收益幅度与 value 路径仍然全宽相符，不能将打分维度缩为四分之一理解成整个模块快四倍。

论文还给出了一个 checkpoint 转换结果。将全部 read sites 的静态 query 堆成矩阵 $$Q$$，并吸收 routing RMSNorm 的 affine gains 后，只要 $$r\geq\operatorname{rank}(Q)$$，且 backbone 支持一致的正交坐标重参数化，就能把 query 子空间旋转到保留坐标中，维持相同的路由与模型输出。**这项等价转换要求先按全宽 RMS 归一化，再保留坐标**，与上面从头训练时的 sliced-key normalization 不同。作者在六个 0.5B–4B checkpoints 上完成了无重训转换；对 Kimi K3 的应用仍是结构分析，尚未实际转换或测量推理加速。

### 6.6 MSAR：为每个历史 source 提供多个上下文视图

[Multi-Stream Attention Residuals（MSAR）](https://arxiv.org/abs/2610.13630)于 10 月提出，将 AttnRes 的一次 depth read 扩展成多路检索。它的 stream 是从历史 source 派生的全宽视图：原表示保留一份，再用 kernel size 为 4、8、12 的 causal depthwise convolution 构造三份局部上下文表示。对 source $$s_i$$，恢复 token 下标后：

$$
s_{i,t}^{(0)}=s_{i,t},\qquad
s_{i,t}^{(a)}=\sum_{\delta=0}^{\kappa_a-1}c_{i,a,\delta}\odot s_{i,t-\delta},\quad a=1,\ldots,m-1.
$$

卷积只访问当前及之前的 token，序列开头补零。每个视图都保留 $$C$$ 维；它与按通道切分多头的做法不同，也不维护 HC 那样跨层递推的 $$[m,C]$$ residual state。

每个目标 sublayer 为每个视图学习一个 pseudo-query。softmax 仍沿历史 source 维执行，各视图独立选择深度信息：

$$
\alpha_{i\rightarrow l}^{(a)}
=\operatorname{softmax}_i\left((q_l^{(a)})^\top\operatorname{RMSNorm}(s_i^{(a)})\right),\qquad
z_l^{(a)}=\sum_i\alpha_{i\rightarrow l}^{(a)}s_i^{(a)}.
$$

检索结果组成 $$Z_l\in\mathbb{R}^{m\times C}$$，再用 mHC 风格的 sigmoid pre-mapping 融合：

$$
g_l=\sigma\!\left(\eta_l\operatorname{RMSNorm}(\operatorname{vec}(Z_l))W_l+b_l\right)\in\mathbb{R}^{m},\qquad
h_l^{in}=\sum_{a=0}^{m-1}g_{l,a}z_l^{(a)},\quad W_l\in\mathbb{R}^{mC\times m}.
$$

这一步借用了动态 read gate，没有引入 mHC 的 residual mixing 或 Sinkhorn。gate 也不要求和为 1。

下面的伪代码保留 token 维，展示一次 source 读取；`expand` 按 source / view 使用对应卷积，embedding 与非空 partial source 也遵循同样的展开规则。

```python
# S: [M, T, C], single-stream sources
# q_l: [m, C], learned pseudo-query per view
# W_pre: [m*C, m]
V = expand(S)                               # [M, T, m, C], identity + causal conv
K = rmsnorm(V)                              # normalize C
score = einsum("mtac,ac->mta", K, q_l)      # [M, T, m]
alpha = softmax(score, dim=0)                # depth sources, independently per view
Z = einsum("mta,mtac->tac", alpha, V)       # [T, m, C]
gate = sigmoid(eta * (rmsnorm(flatten_last2(Z)) @ W_pre) + bias)
# gate: [T, m]; eta initialized to 0.01
h_in = einsum("ta,tac->tc", gate, Z)        # [T, C]
h_out = F(norm(h_in))                        # [T, C], accumulated into source bank
```

主实验使用 Block MSAR。它先将 branch outputs 累计成**单流 block sum**，读取时才展开为多视图；可见 source 包括 embedding、完成的 block sums 与非空 partial sum。因此，与相同分块和 pipeline schedule 的 Block AttnRes 相比，持久 source 存储和通信仍为 $$O(N_{\mathrm{blk}}C)$$，无需长期保存 $$m$$ 份展开结果。上面的伪代码显式构造 `V` 便于说明；优化实现可将展开与 depth attention 融合，但卷积、检索读出和融合仍有计算与访存开销。

论文在 18B / 28B MoE（约 1.7B / 2.7B activated parameters）上报告了 12 个 benchmark 的平均分：

| 规模 | Vanilla | mHC | AttnRes | MSAR | MSAR 相对 AttnRes |
|---|---:|---:|---:|---:|---:|
| 18B | 37.5 | 41.7 | 41.8 | 45.7 | +3.9 |
| 28B | 44.6 | 47.4 | 49.4 | 52.4 | +3.0 |

在该文的 10B 消融中，AttnRes 的 Pile loss 为 1.867，直接组合 AttnRes + mHC 为 1.861，MSAR 为 1.847；去掉卷积后回到 1.862，用均值代替 pre-mapping 则为 1.851。这组结果支持了“上下文视图 + 动态融合”的组合，而不只是增加几组 query。

28B 配置相对 Vanilla 增加约 0.27% activated parameters、0.48% 估算训练 FLOPs；后者不包含 activation recompute 与 optimizer update，也不是实测运行时间。论文未直接比较 xHC，且主实验附录的部分 token budget / batch 字段未列出，现有数据不足以建立跨论文的统一排名。

## 7. Delta Attention Residuals：从 Cumulative States 改成 Deltas

[Delta Attention Residuals](https://arxiv.org/abs/2605.18855v1) 从 cumulative-source baseline 的冗余现象出发，做了两项改动：用 deltas 作为路由 source，并把 replacement routing 改为 additive routing。

其动机是，相邻累计 states 共享大量历史，而 delta 直接表示一个 sublayer 新增的贡献。论文的组合配置同时采用 delta sources 与 additive routing，在相应实验中报告了更尖锐的路由和更低的验证 PPL。

![Delta Attention Residuals 论文 Figure 2：标准 residual、Full AttnRes 和 Delta AttnRes 对比](/assets/img/blog/attention-residual/delta_figure2_comparison.png)


*图 5：Delta Attention Residuals 论文 Figure 2 裁图，按该文的 cumulative-source baseline 对比标准 residual、AttnRes 与 Delta AttnRes。Delta 版本保留 residual stream，并把 depth routing 的 source 从 cumulative states 改成 per-sublayer deltas。*

### 7.1 Routing Collapse

Delta 论文在 Qwen3-0.6B 等实验中观察到，深层 cumulative-source 路由的最大权重降至约 0.2，而 Delta Block 的路由通常更尖锐。0.2 本身并不能证明“接近均匀”：均匀分布的最大权重是 $$1/M$$，还要知道可见 source 数 $$M$$。判断 collapse 应同时检查归一化熵、top-1 与 top-2 间隔、source cosine 和层间变化。

作者将选择性下降与累计 source 的高相似度联系起来，并用每个 sublayer 的增量构造另一组 sources：

$$
v_i = h_{i+1} - h_i.
$$

不同 sublayer 的 delta 往往对应不同功能和子空间，因此更容易保持 routing sharpness。

### 7.2 Additive Routing

Delta Attention Residuals 不用 replacement routing，而用 additive routing：

$$
\hat{h}_l
=
\tilde{h}_l
+
\sum_i \alpha_{i\rightarrow l}v_i.
$$

其中：

$$
\alpha_{i\rightarrow l}
=
\mathrm{softmax}_i
\left(
w_l^\top \mathrm{RMSNorm}(v_i)
\right).
$$

和 AttnRes 对比，差别集中在这里：

| 维度 | Cumulative-source AttnRes | Delta AttnRes |
|---|---|---|
| Source | cumulative states | deltas |
| Routing | replacement / weighted state | additive perturbation |
| Residual stream | 可能被替换 | 默认保留 |
| Init 行为 | uniform average 可能偏离原模型 | zero query 仍产生非零平均；严格无扰动需额外设计 |
| Block boundary | 封口并重置当前 partial sum，历史 source 保留 | 主 residual 不 reset，partial-delta 累加器仍需封口 / 重置 |

additive routing 保留了原 residual state，但这与初始化时“完全不改模型”是两回事。如果 $$w_l=0$$，则每个 source 的权重是 $$1/M$$：

$$
r_l=\frac{1}{M}\sum_{i=1}^{M}v_i.
$$

一般而言 $$r_l\neq0$$。在 source 列表里加入一个零值 Null Source，也只是把平均的分母改成 $$M+1$$，不会自动清除其余 source 的贡献。Delta 论文的 fine-tuning 实验使用了 Null Source，报告了更平滑的起始 loss；公开代码也包含该设计，但仅凭“query 与 null 均为零”不能推出严格 identity。

在 checkpoint conversion 中，可额外引入零初始化的输出门控，使初始 forward 与旧模型一致：

$$
h_l^{in}=h_l+g_l\,r_l,\qquad g_l\big|_{init}=0.
$$

另一种额外设计是用 mask / gate 让初始化时只选择零 source。转换前后的 logits 与 loss 可用于核对初始扰动；这两种设计均独立于论文的基本 additive-routing 公式。

### 7.3 Delta Block

按本文约定，$$L$$ 是 sublayer 数，细粒度 Delta AttnRes 最多保存 $$L$$ 个 branch deltas。Block 版本将完成的 deltas 聚合为约 $$L/G$$ 个 sources，再计入可见的 partial delta。

Delta Block 把连续层的 delta 聚合成 block-level delta：

$$
\Delta_b
=
h_{(b+1)G} - h_{bG}
=
\sum_{i\in \mathrm{block}\ b}v_i.
$$

当前 block 未结束时，也可以把已累积的 partial delta 加入可见列表（见第 2.3 节）。然后做：

$$
\hat{h}_l
=
\tilde{h}_l
+
\sum_b \alpha_{b\rightarrow l}\Delta_b.
$$

这样 source 数从 sublayer 数降到 block 数，同时保持“看的是变化量，不是累计状态”。

### 7.4 Delta AttnRes 的实验信号

Delta 论文在 220M、533M、1044M 和 Qwen3-8B-like 7.57B 上报告了短训实验。以下 PPL、吞吐和显存均对应论文自身的实现与硬件；比较时应保留这些训练条件。

自定义 1044M 配置中：

| 方法 | Val PPL | Tok/s | Mem |
|---|---:|---:|---:|
| Baseline | 29.70 | 108k | 22.5GB |
| AttnRes | 31.76 | 71k | 31.6GB |
| Full AttnRes | 33.36 | 49k | 52.0GB |
| Delta Block | 29.19 | 86k | 28.4GB |
| Delta AttnRes | 29.13 | 34k | 77.7GB |

7.57B 配置中：

| 方法 | Val PPL | Tok/s | Mem |
|---|---:|---:|---:|
| Baseline | 17.43 | 21.4k | 41.6GB |
| AttnRes | 18.58 | 12.5k | 44.0GB |
| Delta Block | 16.00 | 14.0k | 42.7GB |

Delta Block 在 Qwen3-0.6B scale 上还观察到 routing sharpness 明显更高：平均 max weight 约 0.62，而 AttnRes 约 0.35；深层 AttnRes 会降到约 0.2，Delta Block 可以保持约 0.6 到 1.0 的范围。

### 7.5 Kimi 与 Delta 的结果如何对应

Kimi 的报告包含 Kimi Linear 48B total / 3B active 的长训验证；Delta 则在 Qwen-like / custom 短训配置中，观察到其复现的 cumulative-source AttnRes 出现退化。两者的架构、训练预算与 source 定义不同。

Delta 作者将自己的实现称为 faithful reimplementation，其中用于分析的 baseline 采用 cumulative sources；Kimi Full 的 values 是 branch outputs，Block 的 values 是各 block 的贡献之和。

Delta 的结果支持进一步检验 source 冗余和 additive routing 的作用。要分离两项改动的贡献，需要在同一配置中分别比较 cumulative / delta sources 与 replacement / additive routing；这些结果本身不构成 Kimi 原始 Full / Block 方法普遍失效的证据。

## 8. Depth Memory：把历史写进固定大小的矩阵

[Depth Memory](https://arxiv.org/abs/2610.18809)于 10 月提出一个统一的 read / write 框架：标准 residual 将输出加到单向量，HC 将输出写进多个槽位，AttnRes 将输出追加到显式 source 列表。它进一步给出 Linear Depth Memory（LDM），用矩阵关联记忆保存 key–value 关系，后续层通过 query 检索。

### 8.1 从 source 列表到关联矩阵

省略分组与门控时，LDM 维护原 residual $$h_l\in\mathbb{R}^{C}$$ 和矩阵 $$\mathcal{M}_l\in\mathbb{R}^{C\times D_k}$$。读取与 sublayer 更新为：

$$
h_l^{in}=h_l+\mathcal{M}_lq_l,\qquad
v_l=F_l(\operatorname{Norm}(h_l^{in})),\qquad
h_{l+1}=h_l+v_l.
$$

最简单的写入是 $$\mathcal{M}_{l+1}=\mathcal{M}_l+v_lk_l^\top$$。此时读取结果为 $$\mathcal{M}_lq_l=\sum_{i<l}v_i(k_i^\top q_l)$$，即没有 softmax 归一化的线性 depth attention。历史关联压进矩阵后，状态大小不再随 source 数增长，代价是无法按历史索引逐项恢复或选择 value。

LDM 的主方法进一步把写入视为回归：寻找能从历史 key 重建对应 value 的矩阵。它用 recursive least squares（RLS）递推求解，并引入 inverse covariance $$P_l\in\mathbb{R}^{D_k\times D_k}$$。以下公式取 observation weight 为 1，保留正的 forgetting factor $$\lambda_l$$：

$$
\kappa_l=\frac{P_lk_l}{\lambda_l+k_l^\top P_lk_l},\qquad
u_l=v_l-\mathcal{M}_lk_l,
$$

$$
\mathcal{M}_{l+1}=\mathcal{M}_l+u_l\kappa_l^\top,\qquad
P_{l+1}=\frac{P_l-\kappa_lk_l^\top P_l}{\lambda_l}.
$$

$$u_l$$ 是当前 value 相对记忆预测的误差，$$P_l$$ 调整不同 key 方向上的更新强度。原始 outer-product 累加、delta-rule 近似和 RLS 因而对应不同的写入规则，不能仅凭“都是线性 attention”将它们视为同一种方法。完整版本还对 key / value 乘以 observation weight 的平方根，并使用读门控和可学习 gain。

### 8.2 分组状态与动态寻址

实际实现将 hidden 分成 $$N_g$$ 组，每组 value 宽度 $$C_v=C/N_g$$。每个 token 的状态为：

$$
h_l:[C],\qquad
\mathcal{M}_l:[N_g,C_v,D_k],\qquad
P_l:[N_g,D_k,D_k].
$$

主配置使用 16 组、$$D_k=8$$。query 由读取前的 residual 生成，key 由更新后的 residual 生成，均通过小的 shared latent representation 并做 L2 normalization；这与 AttnRes 每个 read site 的静态 pseudo-query 不同。各组独立读取，拼接回 $$C$$ 维后加到原 residual。

以下是 observation weight 为 1 的分组递推，省略 read gain / gate；`make_query` 和 `make_key` 代表完整的动态地址生成网络：

```python
# h: [C], memory: [N_g, C_v, D_k], P: [N_g, D_k, D_k]
# before writing the embedding: memory = 0, P[g] = ridge_scale * I
q = make_query(h)                           # [N_g, D_k], L2-normalized
read = stack([memory[g] @ q[g] for g in range(N_g)])  # [N_g, C_v]
h_in = h + flatten(read)                    # [C]
v = F(norm(h_in))                           # [C], full Attention / MLP
h_next = h + v
key = make_key(h_next)                      # [N_g, D_k], L2-normalized
v_groups = reshape(v, (N_g, C_v))
memory_next, P_next = memory.copy(), P.copy()
for g in range(N_g):
    p_key = P[g] @ key[g]                   # [D_k]
    rls_gain = p_key / (forget[g] + key[g] @ p_key)
    error = v_groups[g] - memory[g] @ key[g] # [C_v]
    memory_next[g] = memory[g] + outer(error, rls_gain)
    P_next[g] = (P[g] - outer(rls_gain, key[g] @ P[g])) / forget[g]
```

分组只用于关联记忆，Attention / MLP 仍在完整 hidden 上执行。embedding 也会写入初始 memory，LM head 前再做一次读取。

固定深度状态不等于较低的实际峰值显存。$$D_k=8$$ 时，memory matrix 本身就有 $$8C$$ 个元素，此外还有 residual 和 $$N_gD_k^2$$ 个 inverse covariance 元素。论文通过反向时重建旧 state 的 *unwinding*、窗口化写入和 CuTe DSL kernel 减少中间状态保存与访存，但相应训练设置下的峰值显存仍高于 Block AttnRes：例如 1.7B 配置为 94.0 GiB 对 80.3 GiB。

### 8.3 预训练结果与性能测试

论文在 400M Llama-like / 50B tokens 和 1.5B Qwen-like / 100B tokens 上进行预训练，为各方法分别搜索 learning rate。下表取主配置 $$D_k=8$$：

| 规模 / 训练量 | 方法 | Val PPL | 平均 zero-shot 准确率 |
|---|---|---:|---:|
| 400M / 50B | PreLN | 12.73 | 49.05 |
| 400M / 50B | Full AttnRes | 12.35 | 49.98 |
| 400M / 50B | Block AttnRes | 12.12 | 50.61 |
| 400M / 50B | mHC | 12.23 | 49.55 |
| 400M / 50B | LDM | 12.08 | 50.46 |
| 1.5B / 100B | PreLN | 9.48 | 58.41 |
| 1.5B / 100B | Block AttnRes | 9.26 | 59.42 |
| 1.5B / 100B | mHC | 9.25 | 59.68 |
| 1.5B / 100B | LDM | 9.15 | 59.71 |

LDM 在两组主实验中取得最低验证 PPL，准确率收益则更小：400M 的平均准确率仍由 Block AttnRes 领先，1.5B 的 LDM 与 mHC 接近。论文还在 B200 上测量了 1.7B–14B 模型的吞吐，LDM 在所有测试设置下快于 Full AttnRes，并与 Block AttnRes 有竞争力；这些较大模型的性能测试不等于已经验证了 14B 规模的训练质量。

## 9. 横向比较：槽位、显式 source 与矩阵记忆

### 9.1 方法机制对比

| 方法 | 改造对象 | Source / Memory | Weight 类型 | 直通项或聚合方式 | 大规模主要风险 |
|---|---|---|---|---|---|
| 标准 residual | 固定加法 | 单 hidden state | 固定 1 | identity 加新贡献 | 深度路由固定 |
| HC | 多 stream | $$N$$ streams | dynamic dense | 无 identity 保证 | 自由 mixing 的稳定性风险 |
| iHC | 多 stream | $$N$$ streams | dynamic read/write | identity transition | 没有直接 stream mixing |
| mHC | 多 stream | $$N$$ streams | dynamic 双随机 mixing | 给定 mixer 非扩张 | I/O、Sinkhorn 和差异收缩 |
| oHC | 多 stream | $$N$$ streams | dynamic 正交 mixing | 给定 mixer 整体等距 | 更大规模与不同 recipe 待验证 |
| xHC | 多 stream | $$N$$ streams + active set | dynamic sparse constrained | active mixer 非扩张，inactive 原样保留 | router / kernel 复杂 |
| Full AttnRes | depth attention | embedding + branch outputs | token-dependent softmax | 以聚合结果构造输入 | $$O(LC)$$ source 保存 / 通信 |
| Block AttnRes | depth attention | block sums / partial sum | token-dependent softmax | 以聚合结果构造输入 | source 粒度与缓存成本 |
| Delta Block | depth attention | block deltas / partial delta | token-dependent softmax | residual + 路由增量 | source 历史与初始化 |
| LR-AttnRes | depth attention | 全宽 branch outputs / block sums | 低维 key 上的 softmax | 全宽 value 聚合 | value 缓存与访存仍在 |
| MSAR | 多视图 depth attention | 单流 sources，按需展开 | 每个视图独立 depth softmax + sigmoid 融合 | 融合检索结果作为输入 | 卷积、多路检索与 kernel 实现 |
| LDM | 矩阵关联记忆 | residual + memory + inverse covariance | 动态 query，无 source softmax | residual + memory read | 矩阵状态大小、RLS 与反向实现 |

表中 identity / non-expansive / isometry 描述的是直通项或给定 mixer 的性质，不是完整网络的梯度保证。

### 9.2 状态与额外计算

| 方法 | Residual / source 状态，单 token | 主要额外计算 |
|---|---|---|
| iHC | $$N\times C$$ streams | 动态 read/write，无 residual mixer |
| mHC | $$N\times C$$ streams | read/write、$$N^2$$ 个 mixing logits、Sinkhorn 与 mixing |
| oHC | $$N\times C$$ streams | read/write、正交矩阵构造与 mixing；$$N=4$$ 时使用一对单位四元数 |
| xHC | $$N\times C$$ streams | dense read、router、active mixing；MLP / MoE 侧增加 causal conv 与可选 GS |
| Block AttnRes | 完成的 block sums 与非空 partial sum | 每个 sublayer 的 depth softmax 与加权聚合 |
| Delta Block | 主 residual、完成的 block deltas 与非空 partial delta | depth softmax、加权聚合与 additive routing |
| LR-AttnRes | 全宽 source bank，key 可取其低维 view | 低维打分与全宽 value 聚合 |
| Block MSAR | 单流 block sources；多视图读取结果为临时状态 | causal depthwise conv、各视图 depth attention、动态融合 |
| LDM | $$C+CD_k+N_gD_k^2$$ 个状态元素，与深度无关 | 动态 query / key、矩阵读取、RLS 写入；反向可 unwinding |

### 9.3 实验结果的比较范围

现有结果还不足以给这些方法排一个统一的质量顺序。xHC 的主要证据来自 18B / 28B MoE，Kimi 的主要验证包含 48B total / 3B active 长训，oHC 是 3.9B total / 0.4B active 的受控比较，Delta 则包含 Qwen-like 短训与 checkpoint fine-tuning。数据、训练预算、优化器和 kernel 都不同。

MSAR 的 18B / 28B MoE 比较支持其在该文配置下优于 AttnRes 与 mHC；LDM 的主质量结果到 1.5B；LR-AttnRes 的主要补充是低维路由与有条件的等价转换。估算 FLOPs、实测吞吐与拟合的等 loss 计算收益也应分别比较。


## 10. 结论

HC family 增加 residual 槽位，再通过 read、write 与 mixing 组织信息。iHC 保留 identity transition；mHC 用双随机约束限制线性直通项的扩张；oHC 保留整体范数，并允许均值与差异子空间交换；xHC 则丰富写回信息、稀疏更新 streams，使较大的 $$N$$ 在论文配置下继续带来收益。

Full / Block AttnRes 显式访问历史 branch outputs 或 block sums。Delta 同时改变 source 与 routing，采用 delta sources 和 additive 聚合；LR-AttnRes 保留全宽 value、缩小打分空间；MSAR 为同一组 sources 增加上下文视图和独立的深度检索。LDM 则将历史关联写入固定大小的矩阵，用动态 query 读取。

这些方法改变了 residual 信息的存储、选择或写入方式，成本取决于具体状态与访问过程。固定大小的状态仍可能占用更多显存，较少的打分 FLOPs 也只带来有限的端到端加速；checkpoint 转换还需满足初始化或坐标重参数化的条件。

## 参考

- [Hyper-Connections](https://arxiv.org/abs/2409.19606)
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers](https://arxiv.org/abs/2607.28611)
- [oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions](https://arxiv.org/abs/2609.02672)
- [Tencent Hunyuan Hy4-preview](https://github.com/Tencent-Hunyuan/Hy4-preview)
- [Kimi K3 官方模型说明](https://github.com/MoonshotAI/Kimi-K3)
- [DeepSeek V4 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [Delta Attention Residuals](https://arxiv.org/abs/2605.18855)
- [xHC: Expanded Hyper-Connections](https://arxiv.org/abs/2607.14530v1)
- [xHC 作者代码仓库](https://github.com/aHapBean/xHC)
- [Delta AttnRes 作者代码仓库](https://github.com/wdlctc/delta-attention-residuals-code)
- [Low-Rank Attention Residuals（v2，2026-09-22）](https://arxiv.org/abs/2607.09694v2)
- [Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630)
- [Depth Memory](https://arxiv.org/abs/2610.18809)
