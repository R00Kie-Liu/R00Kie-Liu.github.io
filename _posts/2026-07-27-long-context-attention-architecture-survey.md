---
layout: post
title: "长话长说（一）：长上下文 Attention 架构总览"
date: 2026-07-27
last_modified_at: 2026-10-21
tags: [LLM, Attention, Long Context, Pretraining]
description: "梳理长上下文 LLM 中 attention 与 KV cache 架构的演进，包括 GQA、MLA、Kimi Linear/KDA、SWA、DSA/QSA、MiniMax MSA 与 CSA/HCA。"
---

## 写在前面

> 本文是长上下文 attention 架构系列的第一篇：先把 GQA、MLA、SWA、linear attention 和 sparse attention 放到同一张图里。第二篇单独讲 [linear attention]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})，第三篇继续讨论 [长上下文 attention 设计趋势]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %})。

上下文从 32K、128K 推到 1M 时，attention 架构究竟减少了哪些计算和存储？

只谈 RoPE 外推或模型标称长度，很容易把几类成本混在一起。训练和 prefill 主要受全量 token 交互的二次计算约束；decode 不再重算完整的 $$L\times L$$ 矩阵，却要在每一步反复读取不断增长的 KV cache；并发起来以后，cache 容量和布局又会限制 batch size。于是，同样被称作“高效 attention”的方法，动的可能根本不是同一个变量。

GQA/MQA 减少 KV head，MLA 压缩每个 token 的 KV 表示；SWA 固定缩小可见范围，DSA/QSA/MSA 按内容挑选远程 token、micro-block 或 block，CSA/HCA 则先把远程记忆压短再读；Kimi Linear/KDA 更彻底一些，让多数层用递推状态承接历史，只周期性保留全局 attention。后文都按同一组问题来比较：历史以什么形式保存，每个 query 实际读取什么，以及被省下的是训练计算、decode 带宽还是 cache 容量。

![长上下文注意力架构：实现区别与优劣势](/assets/img/blog/long-context-attention/long_context_attention_methods_gpt_image2_v2.png)

*图 1：长上下文 attention 的几条主线。先区分“压 KV 表示”“减少可见位置”和“把历史改成状态”这几种不同问题。*

<a id="series-map"></a>

| 篇目 | 主要问题 | 完整展开的内容 |
|---|---|---|
| 本篇 | 历史怎么存、每次怎么读 | 基础架构、统一伪代码与成本坐标 |
| [第二篇：Linear Attention]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %}) | 固定状态如何读写和遗忘 | GDN/KDA、数值范围、chunkwise 计算与模型 recipe |
| [第三篇：设计趋势]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %}) | 最近的瓶颈转移到哪里 | 选择成本、粒度、表示与 runtime 访问计划 |
| [第四篇：训练 infra]({% post_url 2026-08-18-long-context-large-scale-llm-training-optimization %}) | 结构如何在集群上训练 | kernel、CP、显存生命周期和动态调度 |
| [第五篇：内部记忆]({% post_url 2026-09-04-long-context-internal-memory-survey %}) | 窗口外的信息如何持续使用 | 参数知识、动态状态、更新与删除 |
| [第六篇：跨层 KV 复用]({% post_url 2026-10-13-cross-layer-kv-reuse %}) | 多层如何共享历史及构建过程 | CLA、YOCO/KITE、HySparse、索引复用与 VFold |

## 1. 基线：标准 MHA 的长上下文瓶颈

比较架构前，先区分训练、prefill 和 decode。它们处理的是同一套注意力关系，计算次数与需要保留的状态却不同。

| 符号 | 含义 |
|---|---|
| $$B,L,N$$ | batch size、训练或 prefill 长度、模型层数 |
| $$S,T$$ | decode 当前可见历史长度、连续生成的 token 数 |
| $$H_q,H_{kv},d_h$$ | query heads、KV heads、每个 head 的维度 |
| $$d=H_qd_h$$ | attention hidden size |
| $$s$$ | 每个缓存元素的字节数；BF16/FP16 为 2 |


<a id="11-训练-forward主要瓶颈是-l2-attention-compute"></a>
<a id="13-训练-backward通常比-forward-更贵并且仍受-l2-约束"></a>
<a id="14-推理-prefill像训练-forward但不需要-backward"></a>

### 1.1 同一段历史，在不同阶段花在哪里

单层标准 MHA 的 Q/K/V/O projection 随 $$BLd^2$$ 增长，attention core 则要计算 $$QK^\top$$ 和 $$PV$$。按一个 multiply-add 为 1 MAC、2 FLOPs 计算：

$$
\mathrm{MACs}_{\mathrm{attn}}\approx2BL^2d,\qquad
\mathrm{FLOPs}_{\mathrm{attn}}\approx4BL^2d.
$$

这是完整矩阵的近似；causal attention 的下三角会减少约一半的常数，但仍保留 $$L^2$$ 项。

| 阶段 | 需要做什么 | 主要压力 |
|---|---|---|
| 训练 forward | 为所有受监督位置计算输出 | projection/MLP 的线性项与 attention 的二次项 |
| 训练 backward | 求 $$dQ,dK,dV$$、softmax 与权重梯度，必要时重算前向中间量 | 同样的二次交互，常数通常更大 |
| prefill | 处理输入，建立后续生成所需的 KV；通常只需要末位置 logits | 长输入的 attention 计算和缓存构建 |
| decode | 为每个新位置计算 Q/K/V，读取已有 KV 并追加新条目 | 历史读取带宽、缓存容量，以及模型权重和 FFN/MoE 计算 |

短上下文时，projection/MLP 可能占更大比例；上下文拉长后，full attention 的二次项会迅速增加。具体瓶颈还取决于 batch、模型宽度、稀疏模式和实现，不能仅按最大上下文长度判断。

![标准 MHA 的长上下文复杂度来源](/assets/img/blog/long-context-attention/mha_long_context_complexity_gpt_image2_refined.png)

*图 2：训练和 prefill 处理整段 token 交互；decode 每步只有一个新 query，但会反复读取历史。*


<a id="12-训练-activation-memoryflashattention-省的是显式-attention-matrix不是-l2-计算"></a>
<a id="15-推理-decode主要瓶颈从-compute-转向-kv-cache-读取带宽"></a>

### 1.2 FlashAttention 省中间矩阵，KV 压缩省持久历史

朴素实现的单层 attention matrix 有 $$BH_qL^2$$ 个元素。以 $$B=1,H_q=64,L=128K$$ 为例：

$$
64\times131072^2\approx1.1\times10^{12}\ \text{elements},
\qquad \text{BF16 storage}\approx2.2\ \text{TB/layer}.
$$

FlashAttention 通过 tiling、online softmax 和 backward recomputation，避免把这张完整矩阵写入 HBM。attention activation 的存储因而更接近 $$O(BLd)$$，但数学上的二次交互仍在。具体 kernel、反向与并行实现见[第四篇]({% post_url 2026-08-18-long-context-large-scale-llm-training-optimization %})。

推理的 KV cache 是另一类对象。它在 prefill 后继续保留，供 decode 使用。忽略量化元数据和 padding，单个请求的总容量约为：

$$
\mathrm{Bytes}_{\mathrm{KV,total}}=2NLH_{kv}d_hs.
$$

decode 在长度为 $$S$$ 的历史上，单层 attention core 约为 $$O(Sd)$$，理想情况下需要读取约 $$2SH_{kv}d_hs$$ 字节的 KV。连续生成 $$T$$ 个 token，其 attention 工作量累积为：

$$
\sum_{t=1}^{T}O((L+t)d)
=O\left(TLd+\frac{T^2d}{2}\right).
$$

所以，避免物化 attention matrix、减少历史 KV 容量、减少每步读取位置，是三种可以组合的优化。长输出会累积 decode 成本；频繁工具调用还会持续增加新的 prefill，缓存命中后的成本分析见[第六篇]({% post_url 2026-10-13-cross-layer-kv-reuse %})。

<a id="16-一个-kv-cache-量级例子"></a>

### 1.3 一个 KV cache 量级例子

假设一个模型有：

- $$N=80$$ 层。
- $$H_q=64$$。
- $$d_h=128$$，所以 $$d=8192$$。
- BF16 KV cache，$$s=2$$ bytes。

标准 MHA 中 $$H_{kv}=64$$。单个 request 的 all-layer KV cache 大小为：

$$
\mathrm{Bytes}_{\mathrm{KV,total}}=2NLH_{kv}d_hs.
$$

| 上下文长度 | MHA: $$H_{kv}=64$$ | GQA: $$H_{kv}=8$$ | MQA: $$H_{kv}=1$$ |
|---:|---:|---:|---:|
| 32K | 80 GiB | 10 GiB | 1.25 GiB |
| 128K | 320 GiB | 40 GiB | 5 GiB |
| 1M | 2.5 TiB | 320 GiB | 40 GiB |

这里的 GiB/TiB 按 $$2^{30}$$ / $$2^{40}$$ bytes 的量级口径近似。表中只算 **一个 request 的 KV cache**，还没有算模型权重、runtime workspace、fragmentation、batching、paged cache metadata、speculative decoding 的 draft/target cache 等额外开销。它解释了为什么：

- MHA 在超长上下文 decode 上几乎不可扩展。
- GQA/MQA 是现代 LLM 的基础设施。
- MLA 继续压缩每个 token 的 KV 表示。
- Kimi Linear/KDA、SWA、DSA/QSA/MSA、CSA/HCA 进一步减少“每层是否需要读完整长 KV”或“读到的历史表示是否必须是原始 token 级 KV”。

把训练和推理合在一起看，标准 MHA 在长上下文下会暴露出四类瓶颈：

1. **prefill 延迟**：输入长文档、代码仓库、工具轨迹时，$$L^2$$ 代价很重。
2. **decode 内存带宽**：每步生成都要读很长的 KV cache。
3. **KV cache 容量**：batch size、并发请求和长上下文长度相互挤压。
4. **cache 管理**：不同层、不同请求、不同压缩状态的 KV 需要不同布局和 eviction 策略。

换句话说，长上下文 attention 不能只看“模型是否声明支持 1M”，更要看：

> **每个新 token 到底要读多少历史信息，这些历史信息以什么形式存储，哪些信息被精确看见，哪些信息被压缩或跳过。**

<a id="2-总体演进图"></a>

## 2. 这些路线可以组合

开头的路线图按历史表示区分方法，后面的伪代码再比较执行路径。它们不是一条依次替代的链：GQA 可以配合 block sparse，MLA 可以配合 token selection，recurrent 层也可以和 sparse global 层混合。跨层共享则沿网络深度减少重复表示，见第六篇。

因此，下表每行描述的是一个改动维度；实际模型可能同时落在多行。

## 3. 快速对比表

| 路线 | 核心改动 | KV cache 变化 | 每个 query 看见什么 | 训练/迁移难度 | 代表模型/报告 |
|---|---|---:|---|---|---|
| MHA | 每个 Q head 独立 K/V | 最大 | 全历史 token | 最标准 | 早期 Transformer LLM |
| MQA | 所有 Q heads 共享一组 K/V | 大幅减少 | 全历史 token | 转换后需训练恢复质量 | PaLM 类路线 |
| GQA | 多个 Q heads 共享一组 K/V | 按 `Hq/Hkv` 减少 | 全历史 token | 低，可从 MHA uptrain | Llama、Mistral、Qwen、MiMo 等 |
| MLA | 将 K/V 联合低秩压缩到 latent vector | 比 GQA 更激进 | 全历史 latent KV | 中，需要配套 RoPE 解耦 | DeepSeek-V2/V3、GLM MLA 系列 |
| Kimi Linear / KDA | 多数层用 gated linear attention，周期性插入 MLA | 大幅减少长序列 KV/cache 压力 | 线性状态 + 周期性全局层 | 中到高，需要专门 kernel 与 hybrid ratio | Kimi Linear、Kimi K3/KDA 路线 |
| SWA | 每层只看最近窗口 | 局部层按 window 限制 | 局部历史 | 低，结构规则 | Mistral、Gemma、MiMo、Step |
| Hybrid SWA/global | 多数 SWA 层，少数全局层 | 多数层省 KV/compute | 局部 + 间歇全局传播 | 低到中 | MiMo-V2-Flash、Step 3.5 Flash |
| DSA / QSA / LSA | indexer 为 query 动态选重要历史，并继续优化 indexer 成本 | 依实现而定，主要省 attention compute | 内容相关 top-k token / micro-block + sink/window 固定预算 | 高，需要 indexer warmup/adaptation | GLM-5、Qwen3.8-Flash-Next、LongCat/LSA 路线 |
| CSA/HCA | 先沿序列压缩 KV，再在压缩记忆上 attention | 显著减少 | 压缩 KV + sparse selected KV + local tail | 高，需要 cache/layout/kernel 协同 | DeepSeek-V4 |
| CSA2 + CED | 跨层复用压缩 KV 与索引，并让 decoder 的全局 KV 来自 encoder 输出 | 进一步减少全局与持久化 cache | 当前层 query 读取共享 KV，局部 SWA 仍逐层计算 | 高，需要共享策略、QAT 与 replay 配合 | DeepSeek-V4.1-Flash |
| MiniMax MSA | GQA 上增加 block selector，选择 KV blocks | 主要省 attention compute，可配合 GQA 省 KV | 每个 GQA group 的 top-k blocks | 高，需要 KL alignment 和 warmup | MiniMax-M3 / MSA report |

### 3.1 用伪代码比较实现路径

上面的表容易记成一串名词。换成伪代码会更直观：这些方法大多不是在同一个位置做优化，有的改 KV cache 的 shape，有的改 mask，有的改“历史”本身。

![长上下文 Attention 架构实现路径对比](/assets/img/blog/long-context-attention/attention_architecture_pseudocode_map_gpt_image2.png)

*图 3：实现路径示意图。CSA/HCA 是两种层配置，分别读取压缩条目与近期精确 KV；KDA 的各层维护独立 state，MLA 层另有 KV cache。MLA 一列展示投影吸收后的 decode 路径。*

下面的代码用于比较依赖关系，省略位置编码、归一化和输出投影。前三种 attention 统一使用 $$[B,H,L,d_h]$$ 布局，代码中的 `Dh` 对应 $$d_h$$；矩阵乘法的最后两维始终是序列与通道，softmax 沿历史位置维执行。

~~~python
def full_attention(x):
    Q, K, V = project_qkv(x)             # [B, H, L, Dh]
    score = Q @ K.transpose(-1, -2) / sqrt(Dh)  # [B, H, L, L]
    return softmax(causal_mask(score), dim=-1) @ V

def gqa_attention(x, group_size):
    Q = project_q(x)                     # [B, Hq, L, Dh]
    K, V = project_grouped_kv(x)          # [B, Hkv, L, Dh]
    # 这里只为表达配对而展开；实际 GQA kernel 不必物化重复 KV
    K, V = repeat_kv_heads(K, V, group_size)
    score = Q @ K.transpose(-1, -2) / sqrt(Dh)
    return softmax(causal_mask(score), dim=-1) @ V

def sliding_window_attention(x, window):
    Q, K, V = project_qkv(x)             # [B, H, L, Dh]
    score = Q @ K.transpose(-1, -2) / sqrt(Dh)
    score = causal_and_window_mask(score, window)
    return softmax(score, dim=-1) @ V
~~~

MLA 改的是持久缓存：保存 latent 和位置 key。概念上可以展开 K/V 做 attention；高效 decode 还可把 up-projection 吸收到 query 和输出投影一侧，避免为全部历史恢复完整 K/V。

~~~python
def mla_step(x_t, cache):
    c_kv_t, k_rope_t = down_project_to_latent(x_t)
    cache.append(c_kv_t, k_rope_t)
    return latent_attention(x_t, cache)  # 直接读 latent 的等价执行路径
~~~

DSA/QSA/MSA 都增加选择过程，但选择单位不同。下面按单个 query 写，候选必须满足因果约束；批量实现需要为每个 query/head group 分别组织索引。

~~~python
def sparse_selected_step(x_t, history):
    q_t = project_q(x_t)
    candidates = causal_candidates(history)
    ids = selector(x_t, candidates).topk(k)
    K, V = gather_main_kv(history, ids)
    return attention(q_t, K, V)
~~~

CSA 和 HCA 改变远程历史本身。DeepSeek-V4 将两类层交替使用，每层都带近期精确窗口；不是在同一层同时拼入两种远程 bank。

~~~python
def compressed_attention_step(x_t, history, layer_type):
    local_kv = recent_exact_window(history)
    if layer_type == "CSA":
        remote = compressed_entries(history, moderate_ratio)
        ids = indexer(x_t, remote).topk(k)
        memory = concat(local_kv, remote[ids])
    elif layer_type == "HCA":
        remote = compressed_entries(history, high_ratio)
        memory = concat(local_kv, remote)  # 不做远程 top-k
    return attention(project_q(x_t), memory)
~~~

KDA/linear attention 将多数层的历史改成 recurrent state。以 3:1 hybrid 为例，每个 KDA 层维护独立状态，全局层维护自己的 KV cache：

~~~python
def kda_hybrid_step(x_t, states, global_cache):
    for i in range(3):
        x_t, states[i] = kda_layer[i].step(x_t, states[i])
    x_t = gated_mla_layer.step(x_t, global_cache)
    return x_t, states
~~~

下面分别展开这些路线的缓存表示、读取方式与训练条件。

## 4. GQA：从“每头一份 KV”到“分组共享 KV”

### 4.1 冗余来自 KV head

标准 MHA 中，`Hq = Hkv`，每个 query head 都有独立的 K/V cache。decode 阶段的内存读取量随 `Hkv` 线性增加。MQA 把所有 query heads 共享到一组 K/V，极大减少 cache，但 GQA 论文的转换实验观察到了质量下降。

GQA 是 MHA 和 MQA 之间的折中：

- MHA：`Hkv = Hq`
- MQA：`Hkv = 1`
- GQA：`1 < Hkv < Hq`

如果 `Hq = 64`、`Hkv = 8`，KV cache 大约减少 `8x`，同时保留多个 KV groups；转换后的质量仍需通过继续训练验证。

### 4.2 共享到什么程度

| 形式 | K/V 组织方式 | 直觉 |
|---|---|---|
| MHA | 每个 query head 都有独立 K/V | 表达最完整，但 KV cache 最大 |
| GQA | 多个 query heads 共享一个 K/V group | 在质量和 cache 之间折中 |
| MQA | 所有 query heads 共享单组 K/V | cache 最省，共享后的质量需验证 |

GQA 不改变每个 query 的可见 token 范围，仍然是 full causal attention。它只减少 K/V head 的冗余，所以它的主要收益在 **decode KV cache 和内存带宽**，而不是从根上消除 `L^2` prefill。

### 4.3 训练方式

GQA 论文提出可以把已有 MHA checkpoint 转成 MQA/GQA checkpoint：对 K/V projection heads 做 mean pooling，再用少量继续训练恢复质量。论文中展示了用原始预训练 compute 的约 5% 进行 uptraining，GQA 能接近 MHA 质量，同时接近 MQA 的速度。

### 4.4 省下 cache，但没有省掉全局计算

GQA 与标准 attention kernel 兼容，主要减少 decode 的 KV 容量和读取量；每个 query 仍看全历史，prefill 的二次交互不变。group 数是静态超参，可以再叠加 sparse selection；MLA 则换了另一种 KV 表示。

## 5. MLA：从“共享 KV head”到“压缩 KV 表示”

### 5.1 从共享 head 到缓存 latent

DeepSeek-V2 的 MLA 将每个 token 的 K/V 联合压成一个 latent。关键不只是“存小向量、用时还原”：高效 decode 可以把内容 key 的上投影移到当前 query 一侧，把 value 的上投影移到汇聚之后，从而直接读取 latent。

```text
写缓存：hidden_t -> latent c_t + 独立位置 key
读缓存：当前内容 query 先变换 -> 与历史 latent 打分
        -> 按 attention 权重汇聚 latent -> 输出侧变换
```

这与第三篇 [GQLA 的双路径等式]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %}#gqla-dual-path)使用相同的乘法换序原理；GQLA 进一步约束组内共享投影，以支持展开的 GQA 路径。DeepSeek-V2 报告的 93.3% KV cache reduction 属于其具体配置，不能当成 MLA 对任意 GQA 的固定倍率。

### 5.2 和 GQA 的差异

GQA 减少独立 KV heads 数；MLA 减少每个 token 缓存的表示维度。两者都保留逐 token 历史，没有自动减少可访问位置数。DSA/QSA/MSA 则在读取时选择位置或块，不能把“少存几个维度”和“少读几个位置”混为一谈。

### 5.3 RoPE 解耦为什么重要

上述换序要求历史内容 key 形如“固定上投影 × latent”。若在展开后的每个历史 key 上直接施加随位置变化的 RoPE，变换就不再是对所有历史通用的一份固定矩阵，无法直接按原方式吸收到当前 query。

MLA 因而把内容与位置分支拆开：内容点积走 latent 路径，位置点积使用独立的 RoPE query/key，再组合分数。cache 保存 latent 与位置 key，而不是整套展开后的每头 K/V。

### 5.4 压缩的是表示，不是序列

MLA 保留全历史访问，减少单 token 缓存，但条目数仍随长度增长。实现需要配套内容／位置分支、投影吸收和 decode kernel；KV 量化、分页与稀疏读取可以继续作用在这套表示上。


<a id="64-和-kimi-k3-的关系"></a>

## 6. Kimi Linear / KDA：从 full attention 到线性状态 + 周期性全局层

### 6.1 历史不再总是一张 KV 表

GQA 和 MLA 仍然保留“对历史 token 做 attention”的基本范式，只是减少 KV head 或 KV 表示维度。Kimi Linear 的思路更进一步：让大部分 token mixing 层不再使用 full attention，而是使用 Kimi Delta Attention (KDA) 这类 gated linear attention，把长程历史压进可递推更新的状态中；再周期性插入 MLA/full attention 层，避免纯线性 attention 丢失全局信息流。

linear attention 这条线和 GQA/MLA/SWA/DSA 的思路差异比较大，[《长话长说（二）》]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})专门讲它从 kernelized attention 到 gated/delta/state/hybrid 的来龙去脉。

Kimi Linear 论文的核心配置是：

- 采用 **3:1 KDA-to-MLA hybrid ratio**，即 3 层 KDA 后接 1 层 MLA。
- KDA 是改进的 gated delta-rule linear attention，使用更细粒度的 gating 和专门 kernel。
- MLA 层使用 NoPE，KDA 成为主要位置感知 operator。
- 报告称在 1M context 下，Kimi Linear 相比 MLA baseline 可减少最高 75% KV cache，并获得最高约 6x decoding throughput。

### 6.2 技术直觉

```text
普通 full attention / MLA:
  每层都显式读取历史 KV cache

Kimi Linear:
  KDA layer: hidden_t + recurrent state -> output_t, state_{t+1}
  KDA layer: 继续更新状态
  KDA layer: 继续更新状态
  MLA layer: 周期性全局 attention，校准长程信息流
```

按历史表示划分，Kimi Linear 是一套 hybrid memory hierarchy：

- KDA 层负责高频、低成本、可递推的长序列 token mixing。
- MLA 层负责周期性全局读取，补足线性状态对精确检索和复杂组合的不足。
- 相比 SWA，它用状态携带窗口之外的历史。
- 相比 DSA/QSA/MSA，它不显式从历史 token、micro-block 或 block 中做 top-k 选择。
- 相比 CSA/HCA，它不先构造 compressed KV entries 再 sparse read，而是把大部分层变成 recurrent/linear update。

### 6.3 和 SWA、DSA/QSA/MSA 的差异

| 问题 | Kimi Linear / KDA | SWA | DSA / MSA |
|---|---|---|---|
| 长历史表示 | 递推状态 + 周期性 MLA | 局部窗口 + 跨层传播 | selector 选 token/block |
| 是否显式读取远程 KV | 只在 MLA 层显式读取 | global 层才显式读取 | sparse selected positions |
| 稀疏依据 | 不是 sparse selection | 位置距离 | 内容重要性 |
| 训练重点 | KDA 表达力、hybrid ratio、kernel | window/global 比例 | indexer warmup / KL 对齐 |
| 系统收益 | decode throughput 和 KV/cache 压力 | 简单稳定、局部层便宜 | 长文检索更直接 |

### 6.4 从 Kimi Linear 到 Kimi K3

Kimi K3 延续 3 KDA + 1 Gated MLA，并在末尾额外放置全局层。它继续调整 decay 参数化、输出门与 kernel，使 recurrent state 能配合 prefix cache 和 speculative decoding。这里需要记住的架构事实是：KDA 状态固定，MLA 历史仍随长度增长，两种状态都必须正确保存和恢复。

per-head 与 per-channel retention、lower-bounded decay 的数值动机，以及 state replay 的完整解释见[第二篇]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})。

### 6.5 状态化的收益与边界

多数层的固定状态减少了长历史 cache 与扫描压力，周期性全局层仍保留精确内容读取。代价是状态容量有限：更新规则、decay、hybrid 比例与状态恢复都影响质量，不能把它视为 full attention 的等价替换。完整机制和系统接口由第二篇展开。

## 7. SWA：从全历史 attention 到局部窗口传播

### 7.1 用固定窗口截断单层视野

Sliding Window Attention (SWA) 直接限制每个 token 只能访问最近 `W` 个 token：

```text
token t attends to [t-W, ..., t-1, t]
```

这样 attention compute 从 full attention 的 `O(L^2)` 变为约 `O(L * W)`，decode 时多数层也只需保留局部窗口 KV。

Mistral 7B 报告中使用 GQA + SWA，并说明 stacked Transformer layers 可以让信息跨层逐步传播到窗口之外。MiMo-V2-Flash 则采用 5:1 SWA/global hybrid attention，128-token sliding window，并扩展到 256K context。

### 7.2 局部窗口为什么不是“完全看不见远处”

如果每层窗口为 `W`，单层只有局部可见性；但多层堆叠后，信息可以逐层扩散：

```text
layer 1: token t sees t-W ... t
layer 2: token t can indirectly receive info from t-2W ... t
layer n: receptive field grows with depth
```

不过这种扩散是间接的，和 full attention 直接检索远处 token 不一样。因此现代长上下文模型通常不会只用纯 SWA，而是采用 hybrid 结构：

- 多数层用 SWA，建模局部语法、代码邻近依赖、短程推理。
- 少数层用 global/full attention，负责长程信息汇聚。
- 或者在 SWA 中加入 attention sink、特殊 global token、局部/全局交替层。

### 7.3 MiMo-V2-Flash 的经验

MiMo-V2-Flash 是 SWA 路线很好的近期案例：

- 重复的 hybrid block 按 5:1 SWA/global 排布，首层另改为 global attention。
- sliding window 为 128。
- 总计 39 个 SWA layers 和 9 个 global attention layers。
- 在报告的训练与评测设置下，带 attention sink 的 128 window 在长文任务上优于 512 window；一般任务上的结果则较接近。

作者推测，较小窗口的局部归纳偏置、正则化作用和更明确的局部／全局分工可能解释这一现象。它支持在该配置下选择 128，而不能推出所有 hybrid 模型都应使用更小窗口。

### 7.4 窗口便宜，远程读取依赖其他层

SWA 的规则模式便于执行，也无需额外 selector；代价是单层不能直接检索远处 token。窗口、sink 和全局层比例要一起确定，远程检索仍需全局读取或跨层传播补足。


<a id="86-longcatlsadsa-的-indexer-也要继续优化"></a>

## 8. DSA：从固定稀疏模式到内容自适应选择

### 8.1 固定窗口漏掉了什么

SWA 的缺点是稀疏模式固定：远处 token 是否重要，与它的位置关系绑定，而不是与内容绑定。DSA 的核心思想是让模型根据内容选择重要历史 token。

在 GLM-5 报告中，DSA 指 DeepSeek Sparse Attention。报告描述它用 dynamic, fine-grained selection mechanism 替代传统 dense `O(L^2)` attention，并强调不同于 sliding window 这类固定 pattern，DSA 会根据内容决定哪些 token 重要。

简化理解：

```text
query token q_t
  -> lightweight indexer scores historical tokens
  -> select top-k important tokens
  -> core attention attends selected tokens
```

### 8.2 DSA 和 SWA 的本质差别

| 问题 | SWA | DSA |
|---|---|---|
| 稀疏依据 | 距离近就看，距离远就不看 | 内容重要就看 |
| pattern | 固定 | 动态 |
| 远程检索 | 依赖 global 层或跨层传播 | 可以直接选远程 token |
| 训练难度 | 低 | 高 |
| 额外模块 | 无或很少 | indexer / top-k selector |
| 稳定性问题 | window/global 比例 | selector 对齐、top-k 确定性、RL 分布漂移 |

### 8.3 为什么 DSA 常需要 warmup/adaptation

DSA 的 selector 决定了核心 attention 的可见历史。如果 selector 一开始乱选，主模型会直接丢信息。因此 GLM-5 报告采用两阶段策略：

1. **dense warmup / indexer warmup**：冻结或基本保持 base model，训练 indexer，让它学会接近 full/dense attention 的重要性分布。
2. **sparse adaptation**：启用 sparse attention，让模型在新 attention pattern 下继续训练。

GLM-5 报告还讨论了 DSA 在 RL 阶段的问题：DSA 多了一个 indexer，它的 top-k 选择会影响 rollout 和训练一致性，因此需要关注 top-k determinism，并默认在 RL 中冻结 indexer 参数以提升稳定性。

### 8.4 计算省下来以后，selector 成了新风险

内容选择保留了直接检索远程证据的路径，但 selector 漏选后，主 attention 无法补回。除了选择质量，还要计入 scoring、top-k、离散 KV 读取以及 rollout 的索引确定性；完整模块的加速不能只按保留 token 比例估算。

### 8.5 QSA：把周期性全局层也稀疏化

Qwen3.8-Flash-Next 提供了另一种组合方式：多数层仍然是 Gated DeltaNet (GDN)，每 4 层里的全局层不再用普通 full attention，而是换成 Qwen Sparse Attention (QSA)。这就把两条路线接在了一起：GDN 负责把大部分层的历史压成 recurrent state，QSA 负责让少数全局层在超长上下文里少读一些远程 token。

QSA 和 DSA 的目标相近，都是让 query 根据内容选择历史；差异在 selector 的粒度。QSA 先用轻量 indexer 把历史压成 micro-block 表示，再在 micro-block 粒度上估计重要性，选出的上下文才进入 core sparse attention。这样 indexer 不必逐 token 扫完整历史，也比纯 token top-k 更容易接近规整的 sparse kernel。

```text
query token / block
  -> compressed lightweight indexer scores micro-blocks
  -> select relevant context blocks
  -> sparse core attention reads selected context
```

技术报告在 256K sequence 的 continued pretraining 阶段引入 QSA，替换 backbone 和 MTP module 的 full-attention 层。MTP 还会跨 draft steps 复用 QSA indices；这种索引如何成为 runtime 的访问计划，见[第三篇]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %}#qsa-index-reuse)。

### 8.6 LongCat/LSA：selector 也有自己的成本

主 attention 稀疏以后，逐 query 扫描历史的 indexer 仍可能在训练/prefill 中带来二次 scoring，离散 top-k 地址也可能使读取受带宽限制。LongCat Sparse Attention（LSA）针对这些成本调整 streaming-aware、跨层与分层索引。其机制和近期 selector 研究放在[第三篇]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %})，本篇只强调：main attention 的 sparsity 不等于完整模块的加速比例。


<a id="9-csahca在压缩记忆上做-sparse-attention"></a>

## 9. CSA/HCA：在压缩记忆上做选择或密集读取

### 9.1 先缩短记忆，再决定读哪里

DSA 仍然要在很长历史上做重要性选择。context 到 1M token 后，问题不只是“选不选得准”，还有“被选对象本身是不是太多、太大”。DeepSeek-V4 的 CSA/HCA 往前多走了一步：先把历史 KV 沿序列维压成更少的 entries，再在这些压缩记忆上做 attention。

DeepSeek-V4 报告使用交替的 CSA/HCA 层，并采用 shared-KV MQA：每个远程 entry 同时充当 key 和 value。这与 DeepSeek-V2/V3 的标准 MLA 缓存结构不同，不能仅把它理解为在原 MLA latent 上再加一层压缩。两类层分别是：

- **CSA: Compressed Sparse Attention**
- **HCA: Heavily Compressed Attention**

报告称，在 1M context 下，DeepSeek-V4-Pro 相比 DeepSeek-V3.2 只需要 27% single-token inference FLOPs 和 10% KV cache；DeepSeek-V4-Flash 进一步降低到 10% FLOPs 和 7% KV cache。

### 9.2 CSA：压缩 + 稀疏选择

CSA 从 hidden states 同时投影出候选 KV 与逐通道压缩 logits。logits 加上 learned 组内位置偏置，沿 token 维做 softmax，再逐通道加权求和，形成一个 entry；不是简单平均，也不是由当前读取 query 临时压缩。

[原文 §2.3.1](https://arxiv.org/html/2606.19348#S2.SS3.SSS1)使用两路投影，将当前 $$m$$ 个位置的一路与前一组 $$m$$ 个位置的另一路合并，即每隔 $$m$$ 个 token 形成一个覆盖最多 $$2m$$ 个位置的重叠 entry。分组完成、且全部来源对当前 query 已因果可见后，entry 才能参与远程读取；未成组和近期细节由精确窗口承接。lightning indexer 在这些压缩条目上选 top-k，再与近期 KV 一起送入 shared-KV MQA。

```text
original hidden states / KV
  -> compress every m tokens into compressed KV entries       # 先缩短历史长度
  -> indexer scores compressed indexer keys                  # selector 也看压缩后的历史
  -> top-k selected compressed KV entries                    # 远程只读少量压缩记忆
  -> concat with sliding-window exact KV                     # 近期信息仍保留细粒度
  -> shared-key-value MQA
```

![DeepSeek-V4 CSA architecture](/assets/img/blog/long-context-attention/deepseek_v4_csa_fig3.png)

*图源：DeepSeek-V4 Technical Report, Figure 3。这里最关键的是两条路径：主 attention 读 selected compressed KV + sliding window KV；indexer 自己也用 compressed indexer keys，而不是对完整历史逐 token 扫一遍。*

所以 CSA 里有三种“记忆”同时存在：最近窗口的精确 KV、压缩后的远程 KV、以及给 indexer 用的压缩 key。这样做的好处是 selector 面对的序列更短，主 attention 的远程读取也更短；代价是 cache layout 变复杂，而且压缩 entry 能不能代表组内关键 token，会直接影响 long-context recall。

### 9.3 HCA：更激进的远程压缩

HCA 和 CSA 最大的差异有两点。

第一，HCA 使用更大的组 $$m'\gg m$$，同样由候选 KV、压缩 logits 和位置偏置生成逐通道加权代表，但不做 CSA 的跨组重叠。每个完整组产生一个 entry，形成更粗的远程摘要。

第二，HCA 不再使用 CSA 那套 sparse top-k 选择。它把 heavily compressed entries 和 sliding-window exact KV 拼接起来做 shared-key-value MQA。也就是说，CSA 是“压缩后再选择”，HCA 更接近“重压缩后作为长期背景读入”。

```text
CSA:
  every m tokens -> compressed entries -> top-k selector -> attention

HCA:
  every m' tokens -> heavily compressed entries -> attention
  where m' >> m and no sparse top-k selector is used

both:
  recent tokens -> sliding-window exact KV
```

![DeepSeek-V4 HCA architecture](/assets/img/blog/long-context-attention/deepseek_v4_hca_fig4.png)

*图源：DeepSeek-V4 Technical Report, Figure 4。和 CSA 对比时，可以注意 HCA 没有 lightning indexer / top-k selector 这条支路，压缩率也更大。*

把它们放在一起，DeepSeek-V4 的 KV cache 就不再是一张同质的 token-level KV 表，而是一个 heterogeneous cache：

```text
recent exact states
  + CSA compressed KV
  + CSA indexer KV
  + HCA heavily compressed KV
```

这也解释了为什么 CSA/HCA 不只是 attention trick。它会影响 compression buffer、indexer KV、recent exact tail、cache block 分配、eviction、context parallelism 和 serving runtime。只改 attention 公式，不改 cache layout，很难兑现它的收益。

### 9.4 CSA/HCA 和 DSA 的关系

DSA 的核心是“从原始历史里选重要 token”。CSA/HCA 的核心是“先把历史压缩成更少的 entries，再选择或读取这些 compressed entries”。两者都在减少远程读取，但动刀位置不同。

| 路线 | 远程记忆形态 | 是否有 selector | 局部精确信息 | 直觉 |
|---|---|---|---|---|
| DSA | 原始 token / token-level entries | 有，选 token | 常与窗口或 sink 组合 | 少看原始历史 |
| MSA | 原始 KV blocks | 有，选 block | 取决于实现 | 少看更规整的 blocks |
| CSA | 每 $$m$$ 个 token 压成 compressed KV entry | 有，选 compressed entries | sliding window exact KV | 先压短，再精挑 |
| HCA | 每 $$m'$$ 个 token 压成 heavily compressed entry，$$m'\gg m$$ | 通常无 top-k selector | sliding window exact KV | 用粗摘要保留极长历史痕迹 |

如果用一句话分开它们：CSA 还在做“远程检索”，只是检索对象变成 compressed entries；HCA 更像“长期摘要层”，牺牲细粒度检索，换更低的 KV cache 和计算成本。

### 9.5 同时压 compute 和 cache 的代价

CSA/HCA 的吸引力在于同时压 compute 和 cache，尤其适合 1M context 这类极端长度。它的风险也很直接：压缩会丢信息，selector 可能选错，cache layout 比 GQA/MLA 复杂得多，训练与推理都要配套实现。它不像 GQA 那样能作为轻量结构替换接到旧模型上，更接近一次 architecture、kernel、cache manager 和 serving framework 的共同改造。

2026 年 9 月的 [DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) 又增加了跨层共享这一维度：CSA2 分别控制是否复用全局 KV、是否重新计算 top-k 索引；Causal Encoder-Decoder（CED）则让上半部 decoder 的全局 KV 从 encoder 输出投影得到，减少长 prompt 的 prefill 工作。报告中的 890 bytes/token 指常驻 HBM 的全局 KV footprint，不能当作包含 SWA、临时 buffer 的总显存；主 KV 使用 FP4，SWA KV 仍使用 FP8。共享模式和近似 SWA replay 的细节放在[第六篇 §4]({% post_url 2026-10-13-cross-layer-kv-reuse %}#csa2-ced)，这里把它记作“序列压缩之后，再减少跨层重复存储与计算”。

## 10. MiniMax Sparse Attention：GQA 上的 block-level content selection

### 10.1 为什么选择 KV blocks

MiniMax Sparse Attention (MSA) 的目标和 DSA 类似：不要让每个 query 看全部历史。但它选择的是 **KV blocks**，而不是单个 token。这样能让 sparse attention 更规整，更适合 GPU kernel。

MSA 建立在 GQA 之上，新增两条分支：

- **Index Branch**：轻量 query/key projection，为每个 GQA group 给历史 KV blocks 打分。
- **Main Branch**：只对被选中的 blocks 做精确 block-sparse softmax attention。

### 10.2 核心流程

| 步骤 | 分支/模块 | 作用 |
|---|---|---|
| 1 | hidden states | 同时送入 Index Branch 和 Main Branch |
| 2 | Index Branch | 为每个 query / GQA group 估计历史 KV blocks 的重要性 |
| 3 | top-k blocks + forced local block | 选中远程重要 blocks，并强制保留局部 block |
| 4 | Main Branch | 只在 selected blocks 上做精确 block-sparse softmax attention |
| 5 | KL alignment | 用 Main Branch 的注意力分布作为 teacher，训练 Index Branch 的选择分布 |

KL alignment 的 teacher 来自 Main Branch，而不是给 top-k 索引直接反传；Index Branch 输入使用 stop-gradient，避免这项辅助目标改动 backbone。训练先 warmup indexer，再启用 sparse training / sparse CPT。

### 10.3 MSA 和 DSA 的差异

| 问题 | DSA | MiniMax MSA |
|---|---|---|
| 选择粒度 | 更偏 token / fine-grained selection | block-level selection |
| 底座 | 常与 MLA/DSA indexer 关联 | 明确 built upon GQA |
| selector 训练 | warmup/adaptation | KL alignment + stop-gradient + indexer warmup |
| kernel 形态 | token top-k 和 sparse attention | block sparse，更利于规整 GPU 执行 |
| local 保底 | 可组合 SWA/local branch | 强制包含 local block |

### 10.4 性能口径

MSA 技术报告称，在 1M context 下，相比 GQA baseline，MSA 在相同 head 配置下降低 per-token attention compute 28.4x，并在 H800 上取得 14.2x prefill、7.6x decoding wall-clock speedup。

这里别读过头：这份 PDF 更准确地说是 MSA attention/infra 技术报告，不是完整 MiniMax-M3 训练报告。它能支撑 MSA 架构、indexer 训练和 kernel 效率分析，但不应过度推断生产 M3 的完整数据配比或 RL pipeline。

### 10.5 block 规整性与选择精度

GQA 减少 KV head 冗余，MSA 再减少读取块数。整块 gather 更规整，但块内仍需精确计算；漏选关键 block 会丢证据，因此需要与 indexer 训练、局部保底和专用 kernel 一起评估。

## 11. 收束：分清改的是哪一层

实现选择先从实际瓶颈开始：训练/prefill 的 $$L^2$$ 计算、decode 的 KV 读取、cache 容量，或 selector 与 runtime 的控制开销。前面的比较表说明各路线改了什么，接下来还要核对训练与推理接口。

落到实现上，训练侧先检查三件事：attention variant 能不能 forward/backward，能不能和 TP/SP/CP、activation recompute 一起工作，selector 或 compression branch 需不需要 warmup。Megatron 这类训练框架里，GQA、MLA、local attention 已经比较接近内建路径；gated delta net、DSA 等实验入口可以帮助定位模块边界，但不能直接等同于 Kimi KDA、LongCat LSA 或 DeepSeek-V4 CSA/HCA 的完整 recipe。

推理侧的接口又不同。GQA/MQA 改 KV cache shape；MLA 缓存 latent KV；SWA 可以只保留 rolling local KV；KDA 要维护 state cache；DSA/QSA/MSA 要保存 selected token/micro-block/block metadata；CSA/HCA 还要同时管理 local exact tail、compressed KV 和 indexer KV。训练侧用 PyTorch reference 或 FlexAttention 把 sparse pattern 训通，只是第一步；serving 里还要回到 paged cache、prefix/radix cache、chunked prefill、continuous batching 和 CUDA graph 这些约束。

[第二篇]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %}) 沿着“KV list → recurrent state”单独推导 linear attention；[第三篇]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %}) 继续梳理近期长文 attention 设计趋势；[第四篇]({% post_url 2026-08-18-long-context-large-scale-llm-training-optimization %}) 再把这些结构放回训练 infra。

## 延伸阅读

这篇主要提供坐标系。继续读时，不必按论文时间顺序排，我更建议顺着“到底压缩了什么”往下走。

- 如果先关心 KV cache，读 GQA/MQA 和 MLA。GQA/MQA 说明 KV head 为什么可以共享；MLA 再往前一步，把每个 token 的 KV 表示压成 latent。读 DeepSeek-V2/V3 时，重点可以放在 latent cache、up-projection 和 RoPE 解耦上，不要停在“省 KV”这个结论。
- 如果关心 linear attention，先接着读[第二篇]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})。读 Kimi Linear/KDA 时，关键放在多数层如何把历史写进 recurrent state，以及少数全局层为什么仍要保留。
- 如果关心 sparse attention，把 DSA/QSA/MSA 和 LongCat/LSA 分开看。DSA/QSA/MSA 回答“远处 token、micro-block 或 block 怎么选”；LongCat/LSA 更关心 selector 自身的代价、跨层复用和 streaming 场景下的稳定性。
- 如果关心 1M context 之后的 memory 形态，CSA/HCA、LCA 这类 compressed memory 工作更值得连起来读。它们先承认完整 token memory 太贵，再问压缩后的远程记忆还能不能被精读。
- 如果要落到训练系统，继续读[第四篇]({% post_url 2026-08-18-long-context-large-scale-llm-training-optimization %})。架构上说“少存、少读、少算”之后，还要经过 kernel、CP/SP、packing 和 cache layout，才知道省下来的成本有没有被别处吃回去。

## 参考资料

### GQA / MQA

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

### MLA

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

### Kimi Linear / KDA

- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [Kimi K3 Tech Blog: Open Frontier Intelligence](https://www.kimi.com/blog/kimi-k3)
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)

### SWA / Hybrid Local-Global

- [Mistral 7B](https://arxiv.org/abs/2310.06825)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [MiMo-V2-Flash Technical Report](https://arxiv.org/abs/2601.02780)

### DSA / GLM

- [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)
- [GLM-5 repository](https://github.com/zai-org/GLM-5)
- [GLM-5.2 official blog](https://z.ai/blog/glm-5.2)
- [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)
- [LongCat Sparse Attention: Taming the Lightning via Streaming-aware Hierarchical Cross-Layer Indexing](https://arxiv.org/abs/2608.01662)
- [Introducing LongCat-2.0](https://tech.meituan.com/2026/06/30/LongCat2.0.html)

### CSA/HCA

- [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969)
- [DeepSeek-V4-Pro Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

### MiniMax Sparse Attention

- [MiniMax Sparse Attention Technical Report](https://arxiv.org/abs/2606.13392)
- [MiniMax-M3 official blog](https://www.minimax.io/blog/minimax-m3)
- [MiniMax-M3 Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M3)
