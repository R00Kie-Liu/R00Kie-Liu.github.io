---
layout: post
title: "长话长说（六）：KV 跨层复用，从 CLA、YOCO 到 KITE 与 VFold"
date: 2026-10-13
last_modified_at: 2026-10-27
tags: [LLM, Attention, KV Cache, Long Context, Pretraining, Inference]
description: "从 agentic 场景的 prefill 开销解释 YOCO-style 架构为何重新受到关注，再区分缓存、生成来源、索引与聚合状态共享，梳理 CLA、KITE/SST、CSA2/CED、HySparse2、GSM、LatentIndex 和 VFold。"
---

## 写在前面

> 这是长上下文 attention 架构系列的第六篇，从[第三篇的研究趋势]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %})中拆出跨层复用，单独讨论。GQA、MLA 和序列压缩的基础见[第一篇]({% post_url 2026-07-27-long-context-attention-architecture-survey %})。 完整目录见[系列导航]({% post_url 2026-07-27-long-context-attention-architecture-survey %}#series-map)。

GQA 减少每层的 KV heads，MLA 把每个 token 的 KV 存小，稀疏 attention 减少每次读取的位置。这些方法仍可能保留一个默认前提：每过一层，都要生成、保存和选择一遍历史。

这个前提并非必须。后面的层可以带着新的 query，读取前面某层生成的 KV。但沿深度复用也有代价：高层原本能写入自己的历史表示，现在可能只能重新解释一份较浅的表示。省下多少存储，与丢掉多少表达自由度，是同一个设计的两面。

2024 年的 CLA、YOCO 已经把这条路线讲得很明确。到 2026 年，KITE、CED、HySparse2 又在不同设计中采用了 YOCO-style 的分工：一部分网络构建历史，另一部分网络读取历史。为什么这时值得重新讨论它？一个重要原因，是多轮 agent 把持续处理新输入的成本推到了更显眼的位置。

本文初稿依据截至 2026-10-13 的公开材料，本次补入 GSM 的 2026-10-15 修订版。先讨论工作负载动机，再展开 KITE/SST、CSA2/CED、HySparse2、GSM、局部历史消融，以及 LatentIndex、VFold。跨层复用包含几种不同机制，并非每一种都能缩短 prefill 路径。

## 为什么 YOCO-style 架构又受到关注

### 多轮 agent 会反复带来需要 prefill 的新输入

一次 agent 交互通常是“生成行动 → 执行工具 → 返回观测 → 再次调用模型”。模型可能只生成一条搜索请求或 shell 命令，工具却返回几千个 tokens 的网页、代码或日志。下一轮决定怎么做之前，模型必须处理这些新内容；任务持续得越久，这样的输入阶段就会重复越多次。

[HySparse2 的引言](https://arxiv.org/html/2609.26368)直接以短动作、长观测作为设计动机；[KITE](https://arxiv.org/abs/2609.27294)也强调，prompt caching 之后仍可能留下大量 uncached input。这里的变化是某些工作负载中 **prefill 在总推理成本里的占比上升**，不一定是单个输入 token 本身变贵，更不能仅凭输入输出 token 比就断言 GPU 成本由哪一侧主导。

### Prefix cache 能复用旧状态，不能免掉新观测的计算

如果旧前缀命中了可复用的完整状态，模型只需要处理新增后缀，不必每轮把整个对话从头计算一遍。以一个示意请求为例：已有 64K tokens 的历史状态可用，工具新返回 4K tokens，那么本轮新增 prefill 是这 4K，而非全部 68K。

但这 4K 仍有两部分工作：在 full-attention 层，新位置的 query 要读取已有的长历史，还要处理新后缀内部的因果关系；在主干中，新位置通常仍需逐层经过投影、归一化、FFN/MoE 等计算。稀疏 attention 可以减少读取的位置，却不自动省去这些逐层计算。

如果旧状态已被淘汰、请求被路由到无法取得兼容缓存的 worker，或者改写了前缀，则还可能需要补算未命中的部分。[DeepSeek-V4.1-Flash §2.2](https://arxiv.org/html/2609.19969)明确指出，频繁工具调用在 KV cache miss 时会带来显著的 prefill 开销。因此，讨论 agent 的输入成本时，应区分正常新增后缀和额外的缓存未命中，不能默认每次工具返回都重算全部历史。


<a id="yoco-缩短的是把输入变成可复用历史的路径"></a>

### 输入成本为什么会推动历史构建与读取分工

当大量新观测反复进入模型时，一个直接问题是：每个输入位置是否都必须经过全部层，才能留下后续生成需要的历史？YOCO 将历史构建与上层读取分开，KITE 则进一步让新增容量集中在读取一侧。§3 会展开缓存齐备后哪些计算不再需要，以及哪些生成位置仍必须运行完整网络。

较小的 KV 还能减少会话状态的存储和传输，但能否提高命中率取决于缓存预算与调度策略。缓存容量、缓存构建深度和每步读取量是不同收益，不能用一个压缩倍率概括。

近期几项工作采用这类分工，与输入开销有关，但长思考的 reasoning tokens、长代码输出仍可能让 decode 占主导。决定收益的是缓存复用后剩余的 prefill 占比，以及新增读取模块的 decode 成本和质量表现。KITE 的量化比较见 §3.3，CED 与 HySparse2 的局部状态处理见 §4–5。

<a id="sparse-linear-sharing"></a>

### Sparse 与 linear attention：哪些收益可以叠加

近期采用跨层共享的模型常同时使用 sparse attention，容易让人以为两者必须配套。其实，sparse attention 决定当前 query 读取哪些历史位置，跨层 KV 共享决定历史表示由哪些层生成、保存多少份。很多动态稀疏设计仍保留完整的候选 KV，只减少每次选中的位置，因此仍有跨层压缩的空间；dense attention 同样可以共享 KV。只有索引复用这一类优化，才以存在相应的稀疏选择过程为前提。

纯 linear attention 的情况不同。以 [Gated DeltaNet、KDA]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %}) 为例，推理时每层保存固定形状的 recurrent state，不保留随上下文增长的逐 token KV 列表。因此，它们不需要靠跨层 KV 共享来解决同样的缓存增长问题。各层状态本身仍占空间，但跨层 state sharing 需要重新设计状态的写入和读取，不能把不同层的状态直接合并后当作等价优化。

**状态大小固定，不代表构建状态的计算可以跳过。** 沿用前面的例子：工具返回 4K 个新 tokens，常规多层 linear 模型仍需处理各位置的投影、状态更新与 FFN/MoE。上层的状态更新依赖这些位置的上层输入，只计算末位置会遗漏其余新内容。Chunkwise 算法可以并行组织这段计算，却不会自动消除依赖。YOCO 能省去大量位置的上半部计算，是因为上半部读取已经备齐的共享历史，不需要再用这些位置更新自己的私有历史状态。

[原始 YOCO](https://arxiv.org/abs/2405.05254) 本身就给出了两条路线互补的例子：下半部可以使用 gated retention，以固定大小的递推状态处理输入；其输出再构建一份逐 token 的全局 KV，供上半部的 global cross-attention 共享。因此，整个 YOCO 并非固定状态的纯 linear 模型，但它也不要求上半部采用 sparse attention。

对于 linear + dense/sparse 的混合模型，跨层 KV 共享可以作用于仍保存逐 token 历史的 attention 层，linear 层继续维护自己的状态。不过，若被跳过的上层仍有需要更新的私有 recurrent state，仅共享其他层的 KV 并不足以让整段 prefill 提前结束。是否能省掉这段计算，最终仍要检查每一类历史状态的生成依赖。

<a id="1-五种共享不能用一个压缩倍率概括"></a>

## 1. 六种“共享”，不能用一个压缩倍率概括

设模型有 $$N$$ 层，batch size 为 $$B$$，历史长度为 $$T$$，每层有 $$H_{kv}$$ 个 KV heads，head dimension 为 $$d_h$$，每个元素占 $$b$$ 字节。忽略量化 scale、padding 和临时 buffer，标准 KV 存储约为

$$
M_{\mathrm{KV}}=2NBTH_{kv}d_hb.
$$

假设每 $$g$$ 层只保留一份同规格的 KV，缓存中的层数因子便从 $$N$$ 降到约 $$N/g$$。例如，32 层、8 个 KV heads、head dimension 128、128K tokens、BF16、batch 1，KV 约为 16 GiB；相邻两层共享后约为 8 GiB。这里的 128K 按 $$128\times1024$$ 计算，数字不含模型权重和其他状态。

共享可以作用于 main KV、生成 KV 的 hidden-state 来源、最终 top-k indices、indexer latent／候选池、对齐后的 value，或 GSM 中已聚合远程信息的有界状态。前五类主要复用历史表示或选择依据，GSM 进一步限制 decoder 直接读取的历史范围。能否减少 prefill，还取决于缓存的生成依赖。

共享 hidden-state 来源时，两个层都从同一个 $$H$$ 生成 KV，但若投影不同，$$HW^K_1$$ 和 $$HW^K_2$$ 仍是两份张量。**共享输入来源，不等于共享缓存本身。**

另一个区别是存储与读取。CLA 可以让两层指向同一块缓存，但两层 query 不同，通常仍要各读一遍缓存、各做一次 $$QK^\top$$ 和 $$PV$$。除非实现额外做了数据驻留或执行融合，KV 容量减半并不意味着 attention 带宽需求或延迟减半。原始 CLA 论文就明确区分了这两种收益。

<a id="sharing-execution-comparison"></a>

### 1.1 从共享对象到实际执行收益

下面按本文后续案例比较。表中的“提前结束”均指大量 prompt 位置的缓存构建；用于生成 logits 的末位置仍需相应上层计算。

| 方法 | 共享或提前生成什么 | 剩余依赖与 prefill 条件 |
|---|---|---|
| CLA | 组内 main KV | 各层仍用自己的 query；后续 provider 依赖前层输出，通常不能提前结束 |
| YOCO | 单一 global bank | 下半部维护自身状态；上半部不写私有历史，缓存可提前备齐 |
| KITE/SST | Prefiller 逐层 KV | bulk input 只跑 Prefiller，末位置再过 Decoder |
| CSA2/CED | KV/indices，提前生成 decoder global KV | 全局部分可提前构建，私有 SWA 仍需 replay |
| HySparse2 | bridging 来源与组内 bank/selection | cross-decoder 无独立 SWA bank，缓存可由下半部构建 |
| GSM | 聚合历史后的近期共享 KV 状态 | encoder 保留历史检索；decoder 只读固定窗口，后段 encoder 可按局部依赖裁剪 suffix |
| IndexCache / CLSA | 最终 indices；CLSA 还共享 main KV | 各层重算权重；是否提前结束仍取决于层私有历史，索引复用本身不足以保证 |
| LatentIndex | indexer latent／候选池 | 本层重打分、独立选 top-k，main KV 与前向依赖仍在 |
| VFold | 对齐后合并的历史 V | 各层保留 key 与 attention，仍计算新位置的 KV，不能提前结束 |

由此再读各节，就能区分容量压缩、减少选择与缩短前向依赖。下半部生成的 KV 供多少层使用，并不单独决定是否能提前结束；被跳过的层还剩不剩私有历史状态，才是执行条件。

## 2. CLA：后面的层改为读取前面的 KV

[Cross-Layer Attention（CLA）](https://arxiv.org/abs/2405.12981) 是最直接的起点。每组的第一层生成 K/V，后续层跳过自己的 K/V projection，但保留 Q projection、attention 输出投影、残差和 FFN。CLA2 表示相邻两层共享，CLA3 表示每三层共享；它们还可以与 GQA/MQA 叠加。

设第 $$\ell$$ 层读取第 $$a(\ell)$$ 层的缓存，其中 $$a(\ell)\leq\ell$$。忽略 batch、多头和位置编码，计算变成

$$
Q_\ell=X_\ell W^Q_\ell,\qquad
K_a=X_aW^K_a,\qquad V_a=X_aW^V_a,
$$

$$
O_\ell=\operatorname{softmax}
\left(\frac{Q_\ell K_a^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}\right)V_aW^O_\ell.
$$

$$X_\ell$$ 是本层 attention 的输入，$$M_{\mathrm{causal}}$$ 是因果 mask。共享没有绕过该层计算，只是改变了它读取的内容。

下面是单步 decode 的示意，$$D$$ 为模型 hidden dimension，$$H_q$$ 为 query heads 数。为突出依赖，省略 attention gate、QK Norm 等模型特有组件；缓存实现应追加新条目，而不是每步复制整个 prefix。

~~~python
# x: [B, D]，当前 token 的 hidden state
# provider[l]：第 l 层使用哪一层生成的 KV，且 provider[l] <= l
# cache[a].K/V: [B, T, Hkv, Dh]
for l in range(num_layers):
    h = attn_norm[l](x)                      # [B, D]
    q = q_proj[l](h).reshape(B, Hq, Dh)
    q = apply_q_position(q, position)

    a = provider[l]
    if a == l:
        k = k_proj[l](h).reshape(B, Hkv, Dh)
        v = v_proj[l](h).reshape(B, Hkv, Dh)
        k = apply_k_position(k, position)
        cache[a].append(k, v)                # 每个 provider 每步只追加一次

    z = gqa_decode(q, cache[a].K, cache[a].V) # [B, Hq, Dh]
    x = x + out_proj[l](z.reshape(B, Hq * Dh))
    x = x + ffn[l](ffn_norm[l](x))
~~~

为什么共享后不一定损失很大？因为本层仍能用新的 query 从同一组 value 中形成不同组合，随后还有本层的输出投影和 FFN。它失去的是“用本层 hidden state 重新写一份历史”的能力，而非整个层的表达能力。

这也解释了为什么不能把普通 checkpoint 的 KV 直接绑在一起就当作等价优化。原有 $$W^Q_\ell$$ 学会的是与本层 key 空间配合，突然换成另一个层的 KV，匹配关系可能失效。CLA 的主要证据来自从头训练的 1B、3B 模型：两层共享可在接近原 MQA 精度的同时进一步降低缓存。它没有证明任意共享倍数、任意 checkpoint 都无损。

[MLKV](https://arxiv.org/abs/2406.09297) 把跨 head、跨 layer 的 KV sharing 放到一起探索；[系统性研究](https://arxiv.org/abs/2410.14442)还比较了让低层 query 读取高层 KV 的设计。后一类可能保留更深的历史表示，却会引入额外训练和 prefill 依赖。[LCKV](https://arxiv.org/abs/2405.10637) 属于需要进一步研究这一依赖的路线，不能直接套用上面“provider 已经算完”的单向伪代码。

## 3. YOCO 与 KITE：把历史写入和读取能力分开扩展

### 3.1 YOCO 的单一全局 bank

[YOCO](https://arxiv.org/abs/2405.05254) 把复用范围从相邻层扩到上半个网络。下半部 self-decoder 处理输入，输出一份用于构建全局 KV 的表示；上半部 cross-decoder 反复读取这份 KV。两部分都保持因果性，并不是双向编码器读取完整未来输入。

![YOCO 的 self-decoder 与 cross-decoder](/assets/img/blog/cross-layer-kv-reuse/yoco-fig2-architecture.png)

*图源：[YOCO，Figure 2](https://arxiv.org/html/2405.05254v2)。下半部负责生成共享的全局 KV，上半部各层保留自己的 query。*

令下半部输出为 $$H_{\mathrm{mid}}\in\mathbb R^{T\times D}$$，共享缓存为

$$
K_*=\operatorname{Norm}(H_{\mathrm{mid}})W^K,\qquad
V_*=\operatorname{Norm}(H_{\mathrm{mid}})W^V.
$$

上半部第 $$\ell$$ 层用自己的 $$Q_\ell$$ 读取 $$K_*,V_*$$，但不向一份层私有的全局缓存写入历史。原论文的 self-decoder 使用 gated retention 或 SWA 等有界状态的算子，因此长序列下主要增长的是这一个全局 bank，下半部还有逐层的 recurrent state 或局部窗口，不能把它们算成零。

这个结构为什么能省 prefill？

标准 decoder 要为每个 prompt token 算到每一层，才能留下各层 KV。YOCO 的全局缓存到网络中部就已齐备，上半部旧 token 的输出又不会用于生成另一份历史缓存，因此那些输出无需全部计算。要获得第一个生成 token 的 logits，仍需让 prompt 最后一个位置经过上半部；后续生成位置也仍经过整个网络。

~~~text
标准 decoder：
所有 prompt tokens -> 逐层计算，并留下每层 KV -> 最后位置的 logits

YOCO：
所有 prompt tokens -> self-decoder -> 全局 KV
最后一个位置的中间表示 + 全局 KV -> cross-decoder -> logits
~~~

这里“跳过”来自计算依赖消失，不是预测某些层“不重要”后做 early exit。它保证的是同一个 YOCO 模型在相应执行条件下的结果，不是 YOCO 与原始 Transformer checkpoint 等价。

这条思路也进入了实际模型。[Gemma 3n 官方开发者说明](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)介绍了把中间层的 local/global KV 直接供上层共享，以改善 prefill。它说明跨层复用不限于超长文服务器模型；端侧多模态输入同样会受缓存构建成本约束。不同实现是否保留私有局部状态，仍需逐个核对。

<a id="kite-sst"></a>

### 3.2 KITE / SST：扩容读 KV 的塔，不扩容写 KV 的路径

StepFun 的 [KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](https://arxiv.org/abs/2609.27294)（2026-09-23）把这个结构用于模型扩容：先训练较小的模型，再从其中间 checkpoint 扩出只读取 KV 的新模块，随后联合训练。KITE 是扩容原则，论文中的具体架构叫 **Step Scale Transformer（SST）**。

这里的 invariant 指结构与接口：新增 Decoder 不产生未来位置需要的 KV，也不能引入必须保存旧 Decoder 状态的跨 token 依赖。它不表示 Prefiller 权重冻结，更不表示 KV 数值在继续训练中保持不变。

![KITE 的两阶段训练与 SST 双塔结构](/assets/img/blog/cross-layer-kv-reuse/kite-fig2-sst-training.png)

*图源：[KITE，Figure 2](https://arxiv.org/pdf/2609.27294)。左侧先训练 Prefiller，右侧从源权重初始化新增 Decoder，再联合训练；绿色连线传递对应层 KV，橙色连线构成 token-local entry bridge。*

SST 的实验配置有两个 18 层的塔，hidden width 均为 2,304。先完整执行 Prefiller，再执行 Decoder；Decoder 第 $$\ell$$ 层用自己的 query，读取 Prefiller 第 $$\ell$$ 层的 KV：

$$
q^D_{\ell,t}=Q^D_\ell(h^D_{\ell-1,t}),\qquad
a^D_{\ell,t}=
\operatorname{Attn}\left(
q^D_{\ell,t},
K^P_\ell[\mathcal I_\ell(t)],
V^P_\ell[\mathcal I_\ell(t)]
\right).
$$

上标 $$P,D$$ 分别表示两个塔，$$t$$ 是 token 位置，$$Q^D_\ell$$ 包含 query 投影与归一化，$$\mathcal I_\ell(t)$$ 是该层原有 causal full/sliding mask 允许读取的位置。两塔沿用源模型的 position IDs，残差、FFN/MoE 等操作只在当前 token 内进行。

这与 YOCO 的区别在于：YOCO 的上半部多层读取同一份 global bank，SST 则保留 Prefiller 的逐层 KV，让新增层与对应层配对。Decoder 可以访问多个深度的历史表示，却不需要自己再保存一套 KV。源模型的 attention pattern 以三个 sliding 层接一个 full 层为主，因此 SST 共享的不只是全局 KV，也包括相应层的局部 KV。

Decoder 的入口也不是仅复制 token embedding。令 $$e_t,h^P_{L,t}\in\mathbb R^{2304}$$ 分别为 embedding 和 Prefiller 顶层输出，论文采用

$$
h^D_{0,t}=
\operatorname{RMS}_{\epsilon}(e_t)
+\operatorname{RMS}_{\epsilon}(h^P_{L,t}),
\qquad
\operatorname{RMS}_{\epsilon}(v)=
\frac{v}{\sqrt{\operatorname{mean}(v^2)+\epsilon}}.
$$

这里 $$L=18$$，$$\epsilon=10^{-5}$$，两个 RMS 操作都没有可训练 affine 参数。Decoder 因而同时获得原始 token 信息和 Prefiller 完整处理后的表示；联合训练的梯度通过 entry bridge 和共享 KV 两条路径回到 Prefiller。

推理阶段可以用下面的流程理解。它是执行依赖示意，不是官方实现；$$T_\ell$$ 表示该层实际保留的全局或局部历史长度。

~~~python
# e: [B, T, 2304]
# kv[l].K/V: [B, T_l, Hkv_l, Dh_l]
# rms_no_affine 使用论文的 epsilon，不带可训练缩放
e = embedding(prompt_ids)
kv, hP_last = prefiller.prefill(e, return_last_hidden=True)
h = rms_no_affine(e[:, -1]) + rms_no_affine(hP_last)  # [B, 2304]

# 只为 prompt 的最后一个位置计算 Decoder
for l in range(18):
    h = decoder[l](h, kv=kv[l], position=T - 1)      # 不写新的 KV
logits = lm_head(shared_output_norm(rms_no_affine(h)))

# 后续每个新 token 仍要经过两个塔
# Prefiller 先更新各层 KV，再由 Decoder 逐层读取
~~~

给定 Prefiller 的输出，Decoder 的一个位置不依赖其他 Decoder 位置的历史，所以推理时可以省掉旧位置的 Decoder 计算。训练时却要为每个受监督位置预测 next token，新增塔不能因此只算最后一个位置。这里节省的是 prompt-wide cache construction，不能推成训练和 decode 都只跑半个模型。

### 3.3 KITE 的收益还来自分阶段训练，不能只按最终参数量比较

KITE 先训练约 33.8B 的源 MoE，再复制源层初始化独立 Decoder，扩到约 67B；embedding、输出归一化和 LM head 继续共享，随后两塔联合续训。转换后 loss 先升后恢复，因此并非保持原函数的无损扩容。具体 token 数、optimizer state 继承与 schedule 见论文，比较成本则必须包含源训练阶段。

比较累计训练开销时，论文把两个阶段都计入：

$$
C_{\mathrm{train}}\approx
3\left(n_s f_s+n_c f_c\right).
$$

$$n_s,n_c$$ 是源模型与 continuation 阶段的训练 token 数，$$f_s,f_c$$ 是相应结构的每 token forward FLOPs，系数 3 近似计入 forward 与 backward。这样比较的是累计理论 FLOPs，而非相同训练 tokens，或只计算扩容后的训练费用。

按累计理论 FLOPs 比较，SST 在论文的训练 loss、下游任务和 held-out ARXIV NLL 上优于 47B、63B 对照。推理却有不同取舍：

| 模型 | bulk-prefill 活跃 body 参数 | decode 活跃 body 参数 |
|---|---:|---:|
| 47B classic | 约 1.477B | 约 1.477B |
| 63B classic | 约 2.016B | 约 2.016B |
| 67B SST | 约 1.120B，只运行 Prefiller 主体 | 约 2.155B，两个塔都运行 |

表中是论文用于分析的活跃主体参数统计，不包含所有端到端成本。SST 的 decode 主体比两个对照都大，优势取决于能否用更多的 prefill 节省抵消它。

论文在 prefill:decode **成本权重**为 75:25 时，估计 SST 比 47B、63B 对照分别省 6.7%、31.6%。这里不是输入输出 token 比，也不是实测 GPU 成本或 serving 加速；用于参考这些权重的 OpenRouter 账单估计同样不等于 GPU 工作量。在该 proxy 下，prefill 成本占比需要超过约 65.5%，SST 才比 47B 对照更便宜。

另外，continuation 使用 4,096 序列长度；论文的任务评测并非百万上下文或完整 agent 流程评测。KITE 对这篇综述的价值，是把 KV 复用与训练扩容连起来，而不是已经证明在所有长上下文 workload 下都更快。

<a id="csa2-ced"></a>

## 4. CSA2 与 CED：缓存、选择、生成来源分别决定

[DeepSeek-V4.1-Flash](https://arxiv.org/abs/2609.19969) 同时调整这三个对象，容易因为名称相邻而被当成一项优化。实际可以分开理解：CSA2 规定哪些层生成或复用缓存与索引，CED 改变 decoder 全局缓存的生成来源。

### 4.1 CSA2 的三种模式

每层静态配置为 Full、Reindex 或 Reuse。这里的 **Full 是完整执行 CSA2 路径，不是 dense full attention**。

| 模式 | main KV 与 indexer K | top-k indices | 本层始终保留的计算 |
|---|---|---|---|
| Full | 生成新的 main KV，并由它投影 indexer K | 运行 indexer | main Q、SWA KV 和 attention 输出 |
| Reindex | 复用前一个可用 Full 层的缓存 | 用本层 indexer Q 重新选择 | main Q、SWA KV 和 attention 输出 |
| Reuse | 复用同一份缓存 | 沿用针对该缓存最近生成的索引 | main Q、SWA KV 和 attention 输出 |

例如，四层依次配置成 Full → Reuse → Reindex → Reuse。第一层建立 bank 并选择位置，第二层读取同一组位置；第三层不新建 bank，但可以改选，第四层跟随第三层的选择。缓存表示只存一份，选择可以更新两次，各层的 attention query 始终不同。

Reindex 提供的自由度很具体：即使无法重写历史表示，当前层也不必读与前层完全相同的历史位置。共享表示与共享决策因此可以使用不同的频率。

CSA2 的 decoder 还采用分层索引。第一个 Full 层扫描完整可见 main KV，同时选出候选 blocks；后续 Reindex 只在这个较大的候选池里细选。例如，2,048 个 block、每个 8 个位置，形成 16,384 个候选，各层再从中选自己的 Top-512。候选池不等于最终读取集合，第一个 indexer 也仍要扫描全历史。

### 4.2 CED 让高层 KV 提前可用，但局部状态仍有依赖

CED 把下半部作为 causal encoder。上半部 decoder 的全局缓存不再从各层输入生成，而从 encoder 最后输出 $$H_{\mathrm{mid}}$$ 经层相关投影生成：

$$
C_\ell=H_{\mathrm{mid}}W_\ell^{KV},\qquad
Z_\ell=H_{\mathrm{mid}}W_\ell^Z.
$$

$$C_\ell$$ 是用于压缩的 KV entries，$$Z_\ell$$ 是对应的压缩权重；随后按 CSA2 的方式形成 main KV。与 YOCO 的单一共享 bank 不同，CED 允许不同 Full 层从同一来源生成不同 bank，再供各自的 Reindex/Reuse 层使用。

因此，“Full 层生成自己的 KV”在 CED decoder 中指它有自己的投影和 bank，不是指它仍使用自己的 decoder hidden state。

全局缓存可以提前构建，局部 SWA 却仍来自各 decoder 层的输入。为了准备开始生成时的局部缓存，上半部不能完全不运行。窗口在深度上逐层传播，精确重建依赖的 prompt 尾部可能远大于一个窗口。

V4.1 使用 bounded replay 控制成本：只重放最近一个窗口，并把 SWA 的可见范围截断到重放区间。decoder 在 post-training 中模拟同样的重放以适应它；encoder 命中全局 prefix cache、却缺少局部状态时，也有相应的 bounded replay 路径。

这个操作得到的是近似局部状态，不是数学上的完整恢复。它使长时间持久化存储可以只保存 global KV；仍在使用的局部状态可以短时保留，缺失时再重建。不能把“不进长期缓存”理解为“运行时不存在局部状态”。

报告中的 890 bytes/token 指全局 KV footprint，而非总显存。主 KV 采用 FP4，SWA KV 保留 FP8，量化、跨层复用和持久化策略共同影响最终数字，不能把整个收益归因于 KV sharing。

<a id="hysparse-evolution"></a>

## 5. 从 HySparse 到 HySparse2：共享 KV 之后，怎样缩短 prefill

[HySparse](https://arxiv.org/abs/2602.03560)（2026-02-03，下面称第一版）的起点是：既然少数 full-attention 层已经生成 KV 并计算了全局注意力，后面的 sparse 层能否同时复用这些 KV 和选择结果？[HySparse2](https://arxiv.org/abs/2609.26368)（2026-09-22）保留这条思路，再处理第一版尚未解决的两件事：多轮长文检索的选择粒度，以及 prefill 仍需贯穿全部主干的依赖。

### 5.1 HySparse：一个 full-attention 层服务多个 sparse 层

第一版的主干由重复的 hybrid blocks 组成，每组是一个 full-attention 层接若干 sparse 层。这里的 hybrid block 指一组网络层，下面的 64-token block 则指序列分块，二者不要混淆。

组内 full-attention 层照常生成自己的 K/V、执行全局 attention，同时为每个 query 位置输出选中的历史 blocks。后续 sparse 层保留自己的 query，在同一批选中位置上重新计算 attention 权重，但读取的 K/V 来自组内 full 层。共享的是候选位置和历史表示，不是直接复用 full 层的 attention 输出；这些选择也会随 query 位置变化，并非整段序列只选一次。

第一版默认把历史按连续 64 个 tokens 分块，用块内最大的 attention 权重作为重要性分数，再选择 16 个 blocks，对应 1,024 个 tokens 的全局预算。GQA 下，同一 KV group 内各 query heads 的分数还会取最大值，使组内 heads 共用索引。[论文 §3.2 与 Algorithm 1](https://arxiv.org/html/2602.03560v1#S3.SS2)说明了实现：修改 FlashAttention，在分块计算时顺带输出 block 级分数，无需把完整 attention 矩阵写回显存。

论文把这种选择称为 *oracle token selection*，是因为它直接使用 full 层的真实 attention scores，而非训练额外的 proxy indexer。这里的 oracle 只对应提供选择的 full 层；它并不保证同一集合也是每个后续 sparse 层的最优选择。

每个 sparse 层还保留一条独立的 128-token SWA 分支。稀疏分支读取共享的全局 KV，SWA 分支则用本层输入生成自己的局部 KV，两条分支共享本层 query，输出分别经 sigmoid gate 后相加。第一版因此已经减少了逐层全局 KV 和重复索引，但局部 KV 仍然是层私有的。

**这还不是 YOCO-style 的 prefill 提前结束。** 各组 full 层仍从自己的输入生成全局 KV，要先得到前面各层的输出；sparse 层也要建立自己的 SWA 状态。即使大多数层的全局读取变稀疏了，第一版仍需逐层处理 prompt，不能只在前半部把整网缓存一次备齐。

### 5.2 HySparse2：从组内复用到两级共享

第二版将主干分成两部分：self-decoder 混合 full attention 与 SWA，cross-decoder 混合 full attention 与 sparse attention。第一版的组内 KV Reuse 保留在 cross-decoder 中；新增的外层 KV Bridging，则把它所需全局缓存的生成来源提前到 self-decoder。

![HySparse2 的两级 KV 共享](/assets/img/blog/cross-layer-kv-reuse/hysparse2-fig2-architecture.svg)

*图源：[HySparse2，Figure 2](https://arxiv.org/html/2609.26368v1)。左侧是跨 self-/cross-decoder 的 KV Bridging，右侧是一个 hybrid block 内的 KV Reuse。*

外层的 KV Bridging 只连接 self-decoder 和 cross-decoder 中的 full-attention 层。若下半部层 $$i$$ 为上半部层 $$j$$ 提供来源，则

$$
K^{\mathrm{cross}}_j=H^{\mathrm{self}}_iW^K_j,\qquad
V^{\mathrm{cross}}_j=H^{\mathrm{self}}_iW^V_j,\qquad
Q^{\mathrm{cross}}_j=H^{\mathrm{cross}}_jW^Q_j.
$$

这里省略归一化。$$H^{\mathrm{self}}_i$$ 和 $$H^{\mathrm{cross}}_j$$ 都是相应层的输入；一个 self-decoder 来源可以供多个 cross-decoder 层使用，但它们仍有自己的 K/V projection，未必生成相同缓存。

内层的 KV Reuse 则发生在 hybrid block 内：full-attention 层拥有 bank，并从其 attention scores 中得到 token-level selection；后续 sparse 层共享这份 bank 和 selection。它没有另外训练一个轻量 indexer，而是把少数 full-attention 层的打分同时用于后续稀疏读取。

选择粒度也从 block 变为 token。同样是 1,024 个全局位置，第一版选择 16 个块，块与块不必相邻；第二版可以把预算分配给分散的 1,024 个 tokens。若一条关键证据只占块内几个位置，整块入选会同时带入许多邻居；多轮工具观测中的证据可能分散在不同轮次，细粒度选择因此更有价值。HySparse2 §4.3 在相同主干、全局预算和局部窗口下比较两种粒度，观察到检索与图推理收益。这个消融支持选择粒度的作用，不能用来单独解释整个模型的 prefill 加速。

与第一版一样，HySparse2 中的 full-attention 确实执行全局密集读取，与 CSA2 的 Full mode 不同；保留这部分计算是它的设计代价。

### 5.3 去掉私有 SWA 分支，是 prefill 能跳过上半部的关键

HySparse 原先在 sparse 层旁保留独立 SWA 分支。HySparse2 改为强制把近期窗口的位置纳入 sparse selection，且远程与近期条目都从共享的 full-attention bank 读取。论文配置选择 128 个近期 token，再加窗口外的 1,024 个 global tokens。

两种实现表面上都“保留最近 128 个 token”，实际内容不同：

| 实现 | 最近位置的 KV 从哪里来 | 是否需要本层 prompt 历史 |
|---|---|---|
| 独立 SWA 分支 | 本层 hidden state 经本层投影 | 需要，本层还要保存和重建局部 bank |
| 强制近期位置入选 | 共享 full-attention bank 中的对应条目 | 不需要额外的本层局部 bank |

第二种保留的是近期位置的可访问性，放弃了“近期位置在本层深度重新表示”的自由度。它不是把原 SWA 分支原样实现得更快，而是改变了模型结构。

这样，cross-decoder 所需的全部 KV 都能由 self-decoder 的 hidden states 构建，prefill 的缓存构建阶段可以在下半部结束，不需要 CSA2/CED 那样的 decoder SWA replay。为生成第一个 token 而计算最后位置的 cross-decoder 输出，仍不可省略。

第一版的 7B 消融发现，去掉独立 SWA 或让它共享全局 KV 会损失质量；第二版则在新主干上重新训练，并保证近期位置入选。其 §4.4 的 Forced SWA 与独立 Gated SWA 各有胜负：前者 RULER-v2 更好，GSM8K、MRCR-v2 与 LongPPL 更差。因此它是用部分质量取舍换取私有状态消失和 prefill 提前结束，不能当作原 SWA 的无损加速。

### 5.4 如何理解它的实验收益

论文主实验是 49 层的 80B-A3B MoE：约 500B tokens、32K context 的预训练，随后约 100B tokens 的轻量 post-training，并扩展到 256K。比较对象共享相应阶段的数据和训练日程，但 attention 配置不是单变量实验：HySparse2 使用 MQA，HySparse 和 Hybrid SWA 使用 GQA，全局层数量、局部处理和选择粒度也有区别。

在论文的 1M-token 成本计算中，HySparse2 相对 HySparse 的 prefill FLOPs 降低 2.92×，FP8 KV 为 2.69 GB，对照为 6.72 GB。这是指定结构下的 FLOPs 与缓存统计，不是实测 2.92× 端到端延迟，也不能单独归因于两级共享。

长文质量评测延伸到 256K；1M 的成本曲线不等于已完成同长度的质量验证。对于预训练 recipe，更有参考价值的是它把“独立局部表示”和“prefill 完全提前结束”之间的取舍做成了可比较结构。

<a id="gsm-shared-global-state"></a>

### 5.5 GSM：把远程读取结果写进共享状态

[GSM: Efficient Language Modeling with Shared Global State](https://arxiv.org/abs/2609.33465v3)（初版 2026-09-27，下面依据 v3）进一步追问：共享了 KV 和 top-k 以后，是否还需要每个 decoder 层重新汇聚远程历史？它将这项工作集中到 encoder，decoder 只读取已经包含远程信息的一段近期表示。

![GSM 的多阶段历史聚合与共享状态](/assets/img/blog/cross-layer-kv-reuse/gsm-fig1-architecture.png)

*图源：[GSM，Figure 1](https://arxiv.org/html/2609.33465v3#S1.F1)。上方 Refine 分支逐阶段累积读取结果，编码结束后才与主干输出融合，再构建一份供所有 decoder 层共享的近期 KV。*

encoder 主干沿用稀疏检索和 KV-sharing group，在若干组末另设聚合分支。令 $$h_t^{(s)}$$ 为位置 $$t$$ 在阶段 $$s$$ 的主干表示，$$r_t^{(s-1)}$$ 为此前累积的读取结果。第一阶段只从当前表示生成 query，后续阶段从 $$[h_t^{(s)};r_t^{(s-1)}]$$ 的拼接生成新 query，再累加

$$
r_t^{(s)}=r_t^{(s-1)}+
O_s\operatorname{Attn}\left(q_t^{(s)};
\mathcal W_t^{(s)},B^{(s)}[\mathcal I_t^{(s)}]\right),
\qquad r_t^{(0)}=0.
$$

$$\mathcal W_t^{(s)}$$ 是该阶段局部 KV，$$B^{(s)}$$ 是历史 bank，$$\mathcal I_t^{(s)}$$ 是对应 encoder 层已经选出的 top-k，$$O_s$$ 是输出投影。聚合分支复用这三者，不增加独立 indexer；自己的 query 改变候选条目的权重，**不会重新选择 top-k**。累积量也不在阶段之间写回 encoder 主干，只在编码结束时融合：

$$
z_t=e_t+r_t^{(S)},\qquad m_t=\operatorname{KV}_{\mathrm{sh}}(z_t,p_t),
$$

其中 $$e_t$$ 是最终 encoder 表示，$$p_t$$ 是位置元数据，shared KV mapping 包含归一化、投影和位置编码。每个位置生成一个条目，并未把若干 tokens 沿序列维压成一个条目。decoder 在位置 $$t$$ 只读取最近 $$M$$ 个有效位置：

$$
\mathcal M_t=[m_{\max(1,t-M+1)},\ldots,m_t],\qquad M=W+K.
$$

$$W$$ 和 $$K$$ 分别是 encoder 的局部窗口与稀疏条目预算；实际窗口还受文档边界和 mask 约束。30B-A3B 配置为 $$W=128,K=512,M=640$$。这些是近期位置的 KV，其表示已经吸收远程信息，不是从原始历史中再选出的 640 个位置，也不是 640 个可学习 memory tokens。

同一位置的全部 decoder 层共享 $$\mathcal M_t$$，各层仍根据前层输出生成自己的 query，保留输出投影、attention sink 和 FFN/MoE；状态在 decoder 内保持不变。与 YOCO、HySparse2 相比，decoder 失去了当步按新 query 回查远程历史的自由，换来有界的读取与共享缓存。若有 $$L_D$$ 个 decoder 层，忽略逐 token 投影和 FFN，其 attention 成本为 $$O(L_DMd)$$，共享 KV 为 $$O(Md_{\mathrm{KV}})$$；$$d$$ 和 $$d_{\mathrm{KV}}$$ 分别概括 attention 宽度与条目存储宽度。

**固定的是 decoder 的状态，不是整网历史存储。** encoder 仍保存历史表示，以便后续位置再次检索。这也将它与纯 recurrent linear attention 区分开：生成下一个 token 时，encoder 可以利用新输入回查尚未写进状态的信息；当前 token 的 decoder 则始终受已有共享状态约束。

### 5.6 GSM 的 prefill 裁剪与实验范围

只需预测下一个 token 时，decoder 仍只执行 prompt 最后一个位置。与 YOCO 的区别是，它所需共享状态只有最后 $$m=\min(T,M)$$ 个位置；encoder 在完成历史 bank 构建以后，还可能按后续局部依赖缩短计算区间。

设 encoder 有 $$L_E$$ 层，最后一个历史 KV 来源层为 $$\ell_*$$。论文 §3.2 给出的依赖裁剪是：先在该来源层完成全序列 KV 构建，再让各层 $$\ell\geq\ell_*$$ 只对最后

$$
R_\ell=\min\left(T,\ m+(L_E-\ell)(W-1)\right)
$$

个位置执行 attention 读取和后续块内计算；聚合分支同步裁剪。额外的 suffix 覆盖后续 SWA 对前层输入的需求，范围随深度逐渐缩至 $$m$$。这与 §8.1 的精确 suffix 分析相同，都要沿依赖倒推，不能把所有后段 encoder 层一律裁到 $$M$$。

论文以同深度、hidden width、数据和训练设置的 DeepSeek-V4.1-Flash 式主干为基线，从头训练，而非直接压缩已发布模型。1.5B dense、8B-A1.5B 和 30B-A3B 的十项下游平均分接近，后两档略高；各配置只有单次训练，未评估跨随机种子的波动。

30B-A3B 在 31B input tokens 的主训练后适配到 64K，再测 4K–256K 的效率。256K、H800/H100、batch 1 下，prefill 吞吐约为基线 3.37×，cached decode 约 1.17–1.22×；request cache 从 0.491 降至 0.295 GiB，约减少 40%。这里统计 encoder、局部窗口和共享状态的唯一底层分配，而非只算 decoder。总 GPU 峰值仍约 72.5–72.8 GiB；32 卡训练基准虽达到 2.63× 吞吐，峰值分配却从 50.61 增至 56.75 GiB/GPU。附录 A.3 的推理流程描述为完整 prompt encoder 加末位置 decoder，上述速度数字没有单独量化依赖裁剪的增益。

近 1M 的结果则来自额外 YaRN 扩展和约 5B input tokens 的 RULER-derived 任务适配：64 个多 key 样例全对，64 个多跳样例答对 59 个。它支持“encoder 保留历史检索、decoder 只读 640 个状态条目”能处理这些任务，不等于对任意百万 token 任务的无损保证。0.5B 的聚合阶段消融也显示，三阶段改为单阶段后，平均 RULER 从 94.33% 降到 83.33%，而增加到十阶段没有继续提升；状态形成前的检索深度有实际作用。

## 6. IndexCache 到 LatentIndex：共享选择，还是共享选择依据

这一组工作与 KV 共享相邻，却不能混为一谈。main KV 是最终供 attention 读取的内容，indexer cache 是用来决定读哪里的辅助表示，两者可能完全独立。

### 6.1 IndexCache 和 CLSA 减少重复选择

[IndexCache](https://arxiv.org/abs/2603.12201) 把层划分成运行 indexer 的 Full 层和复用最近 Full 层 top-k 的 Shared 层。各层 main attention 仍可以保留自己的 KV；省去的是重复选择，不是 main KV 存储。

它提供两条适配路径。training-free 版本在校准集上按语言模型 loss 搜索保留哪些 indexer；training-aware 版本用被服务的多个层的 attention 分布共同监督保留下来的 indexer。共享选择会影响多个层，监督目标因此也不宜只代表 provider 自己。

[CLSA / You Only Index Once](https://arxiv.org/abs/2606.06467) 则建立在 YOCO-style KV-sharing 结构上，既共享 cross-decoder 的 KV，也复用一次 token-level top-k routing。它节省了两种开销，同时要求同一组选择能服务更多不同深度的 query。

这种做法的局限不在于“后面层没有 query”：各层依然会重新计算 attention 权重。局限在于，如果某个位置没有进入共享的 top-k，后面层再合适的 query 也读不到它。

### 6.2 LatentIndex 保留各层不同的 top-k

[LatentIndex](https://arxiv.org/abs/2610.04635)（2026-10-03）改为共享连续的 indexer latent，而不直接要求各层选中同一组 token。

![DSA、IndexCache 与 LatentIndex 的共享对象对比](/assets/img/blog/cross-layer-kv-reuse/latentindex-fig1-sharing.png)

*图源：[LatentIndex，Figure 1](https://arxiv.org/html/2610.04635v1)。IndexCache 传递离散选择，LatentIndex 传递共享 latent，再由各层独立打分。图中为简洁只画了一组 query，实际 query 是层相关的。*

为说明关键计算，先省略 RoPE，把向量统一写成列向量。组内首层是 anchor，其 hidden state 为 $$x^{(a)}_t\in\mathbb R^D$$，共享投影为 $$P\in\mathbb R^{D\times d_C}$$。每个历史位置的 latent 为：

$$
c_t=P^\top x^{(a)}_t\in\mathbb R^{d_C}.
$$

第 $$\ell$$ 层的有效 indexer key 由自己的 decoder 定义：

$$
\widehat k^{(\ell)}_t=D_\ell c_t+b_\ell,
\qquad D_\ell\in\mathbb R^{d_I\times d_C}.
$$

$$d_I$$ 是原 indexer key 维度，$$d_C$$ 是共享 latent 维度，$$b_\ell\in\mathbb R^{d_I}$$ 是层相关 bias。$$D_\ell$$ 是一个投影矩阵，这里的 decoder 不是 YOCO 的上半个 Transformer。

如果先恢复所有历史 key，额外计算会很贵。但打分只需要点积，可以把矩阵移到 query 一侧：

$$
\left(q^{(\ell)}_{s,h}\right)^\top\widehat k^{(\ell)}_t
=
\underbrace{\left(D_\ell^\top q^{(\ell)}_{s,h}\right)^\top}_{\text{变换后的 query}}
c_t
+\left(q^{(\ell)}_{s,h}\right)^\top b_\ell.
$$

$$s$$ 是当前 query 位置，$$t\leq s$$ 是历史位置，$$h$$ 是 indexer head，$$q^{(\ell)}_{s,h}\in\mathbb R^{d_I}$$。每个 query head 只变换一次，随后直接与共享 cache 打分，无需逐条重建历史 key。DSA-style indexer 再经 ReLU 和 head 加权得到位置分数：

$$
u^{(\ell)}_{s,t}
=\sum_h w^{(\ell)}_{s,h}
\operatorname{ReLU}
\left[
\left(D_\ell^\top q^{(\ell)}_{s,h}\right)^\top c_t
+\left(q^{(\ell)}_{s,h}\right)^\top b_\ell
\right].
$$

$$w^{(\ell)}_{s,h}$$ 是各 indexer head 的权重。bias 项虽然对所有历史位置相同，却位于 ReLU 之前，不能因为“top-k 对统一平移不敏感”就删去。各层对自己的 $$u^{(\ell)}$$ 取 top-k，最终读取的仍是本层 main KV。

完整方法把 latent 分为 RoPE 与 NoPE 两部分；RoPE decoder 还必须与位置旋转相容。上面的简式只说明投影吸收，不表示任意跨频率线性映射都能直接用于 RoPE。

### 6.3 共享 latent 省存储，缩小候选池才减少全历史扫描

若一个组有 $$m$$ 层，原来保存 $$m$$ 份 $$d_I$$ 维 indexer keys，改后保存一份 $$d_C$$ 维 latent，payload 比例约为

$$
\frac{d_C}{m d_I}.
$$

但若每层仍独立扫描完整 prefix，索引计算不会随缓存份数一起消失。LatentIndex 的 hierarchical selection（HS）让 anchor 先从全历史提出较大的候选集合，后续各层只在其中重打分。若候选预算为 $$C\geq k$$，一组每个 query 的打分位置数从

$$
mT\quad\longrightarrow\quad T+(m-1)\min(C,T).
$$

这和直接共用最终 top-k 不同：候选是搜索边界，最终选择仍可变化。代价是 anchor 候选池漏掉的证据，后续层无法补回。

论文在 DeepSeek-V3.2 的四层共享配置中报告 logical indexer-cache storage 减少 61.1%，不是总 KV 减少 61.1%。HS 在 8K–128K 下的 2.30–2.72× 是 decode indexer 加速，不是整个模型 decode 加速。

其 training-free 路线通过离线低秩回归拟合共享投影和各层 decoder，不更新原主干权重；这仍需要校准数据和新增固定参数。training-aware 路线还探索了让组内所有 routing 输入都来自 anchor，从而在组入口计算多层路由。后者进一步改变输入来源，不能与“每层都用自己当前 hidden state 打分”的实现混写。

<a id="vfold"></a>

## 7. VFold：对齐坐标、折叠变换，再合并 value 缓存

[VFold](https://arxiv.org/abs/2610.12338)（2026-10-08）面向已有预训练模型：把相邻两层对**同一个历史 token** 的 value 对齐，再平均成一份缓存。两层的 key 继续分别保存，query 和 attention 权重也各自计算。它合并的是层维度的冗余，不是把相邻 tokens 合成一个 token。

这里有三个步骤：用校准数据找到坐标对齐映射；把映射及其逆变换吸收到模型权重；推理时将较旧的 value 缓存合并。前两步准备可共享的表示，第三步才真正节省缓存。

### 7.1 坐标对齐：两层的信息相近，维度含义却未必对应

两层使用不同的 value projection，即使包含相近的信息，输出通道也不必一一对应。下面是一个二维示意，假设第二层的两根坐标轴恰好与第一层相反排列：

~~~text
同一个历史 token：
第 10 层 value                  [1.0, 2.0]
第 11 层 value                  [2.2, 0.8]
第 11 层交换坐标后              [0.8, 2.2]

直接平均原始表示                [1.6, 1.4]
在共同坐标中平均                [0.9, 2.1]
~~~

交换坐标后，才能比较对应通道的差异。实际映射可以是更一般的可逆线性变换。VFold 在校准文本上收集相邻层对相同位置的 value，用 CCA（典型相关分析）拟合对齐映射。这是离线校准，不需要为每个请求重新拟合，也不要求额外微调模型。

下面先用单 head、行向量记法。保持第 $$\ell$$ 层坐标不变，将下一层映射到它的坐标系：

$$
\widetilde V_{\ell+1}=V_{\ell+1}R.
$$

$$R$$ 是校准得到的可逆映射。对齐只让两份表示更接近，不能保证它们完全相同。多头模型还要匹配相邻层的 KV heads：论文用 Hungarian algorithm 选择配对，再做 head 内的线性变换。不同 heads 的 attention 权重不同，不能随意跨 head 混合通道；对应的 head permutation 也要同步应用到相关投影。

### 7.2 折叠进权重：内部换坐标，输出投影负责抵消

直接替换 value 会改变 attention 输出，因此还需要补偿。对待变换的层，设 $$X$$ 是 attention 输入，$$V=XW_V$$，$$A$$ 是 softmax 后的 attention 权重，$$W_O$$ 是输出投影，原输出为 $$O=AVW_O$$。如果 value 右乘 $$R$$，同时让输出投影左乘 $$R^{-1}$$，就有

$$
AVW_O=(AVR)(R^{-1}W_O).
$$

中间表示换了坐标，输出回到原来的表示空间。这个等式也说明，推理时无须在 attention 两侧额外执行一对矩阵乘法。可以离线改写权重：

$$
W'_V=W_VR,\qquad W'_O=R^{-1}W_O.
$$

之后照常计算 $$V'=XW'_V$$，得到的已经是对齐坐标中的 value，输出也直接使用修改后的 $$W'_O$$。这就是把变换 *fold* 进权重：校准后改写一次投影，而非每次读取缓存再恢复原坐标。GQA 下，同一个 KV head 服务的所有 query heads，其对应的输出投影片段都需要做相应补偿。

**到这一步，在精确算术下原层的函数保持不变，缓存大小也没有减少。** 实际浮点执行会有数值误差，但尚未进行有损的缓存平均。

### 7.3 缓存合并：两层各用自己的注意力，读取同一份近似 value

完成对齐后，两层在同一历史位置的 value 才进行平均。对一段可以合并的历史，写成矩阵就是

$$
V_{\mathrm{merge}}=\frac{V_\ell+\widetilde V_{\ell+1}}{2}.
$$

相邻两层原来保存 $$K_\ell,V_\ell,K_{\ell+1},\widetilde V_{\ell+1}$$，现在只需保存 $$K_\ell,K_{\ell+1},V_{\mathrm{merge}}$$。读取这段压缩历史时，两层的计算分别近似为

$$
\widehat O_\ell=A_\ell V_{\mathrm{merge}}W_{O,\ell},\qquad
\widehat O_{\ell+1}=A_{\ell+1}V_{\mathrm{merge}}W'_{O,\ell+1}.
$$

$$A_\ell$$ 和 $$A_{\ell+1}$$ 由各层自己的 query、key 计算，可以关注不同历史位置。第二层使用已经折叠逆变换的输出投影，所以两层能从同一份 value 读回各自表示空间的内容。上式只展示可合并历史部分；受保护位置仍读取各层自己的 value。

二维例子中的共享结果 $$[0.9,2.1]$$，与第一层的 $$[1.0,2.0]$$ 和对齐后的第二层 $$[0.8,2.2]$$ 都有差异。**误差从平均替代这一步开始。** 对齐让差异更小，却无法恢复平均时丢掉的层间差异；后续 hidden state、query 和 key 也可能随之变化。

论文还区分当前计算与历史存储：新 value 先以未平均的形式参与本层 attention。实验保留最前面的 4 个 sink tokens 和最近 128 个 tokens，普通条目离开近期窗口后才合并。因此，当前 value 不会在首次使用前就被平均，但当前输出仍可能因读取了已压缩的旧历史而产生误差。

### 7.4 省下的是 value 存储，整层计算仍然保留

为什么只动 V？V 的变换可以在输出投影中消去；K 决定 softmax 前的分数，还涉及 query 匹配、位置编码和归一化，不能未经条件检查就套同样的变换。只合并 V 也让层私有的 key 与路由得以保留。

缓存统计要按两侧分别算：若 K/V 原本等大，将 V 减半、K 不变，总 KV 只减少约 25%，不是 50%；受保护条目还会使实际比例略有变化。

它与 YOCO 的执行依赖也不同。VFold 中，两层仍各自计算新位置的投影、attention 和 FFN，再把可合并的旧 value 存成一份；没有把高层缓存的生成提前，也不能因此跳过上层 prefill。这里直接得到的是缓存容量收益。

论文在 Llama-3.1-8B-Instruct、Mistral-Small-3.1-24B-Instruct 和 Qwen3-8B 上评测 LongBench 与 16K RULER。性能接近 full cache，但不等于运行一定更快：其 A100-80G、Llama-3.1-8B、8K prompt、生成 256 tokens 的表格里，batch 1 的 TPOT 从 21.0 ms 增至 27.8 ms，最大可容纳 batch 从 33 增至 38。这里得到的是容量与质量的折中，当前实现仍有缓存维护开销。

## 8. 共享多少历史，保留多少层私有信息

上述设计留下一个实际问题：共享全局 KV 以后，本层究竟还需要什么？

“每层 query 不同”可以保留读取差异，却不能补回根本不存在的层私有历史。另一方面，为每层保留局部历史，会增加存储和缓存构建依赖。下面两类工作分别提供质量证据和适配办法。

### 8.1 局部历史消融：窗口里存的内容确实可能有价值

[Shared Global KV with Layer-Specific Local History](https://arxiv.org/abs/2609.28006)（2026-09-23）区分了两个变量：局部 bank 是否包含历史，以及局部 KV 是否一定来自本层自己的 hidden state。

它的主要设置是 126M、16 层、2K context：下半部生成共享 global KV，上半部每层可以保留 128-position local history，也可以只保留当前 token 的 local KV。后一个对照仍能读取完整 global prefix，因此比较的不是“能读历史”和“完全不能读历史”。

在八个随机种子的同源测试中，保留局部历史比 current-only 对照的 held-out perplexity 约低 1.4%。容量、条目数和训练计算控制支持历史内容本身的作用；不过外部分布的书籍实验仍有不确定性，下游任务结果也不一致。相对 GQA、相邻层 KV sharing，完整设计还使用了更大的缓存，长请求延迟更高。

这项工作不能推出“独立 SWA 一定比 HySparse2 好”。模型尺度、训练数据、路由和融合方式都不同。它提供的是一个应当保留的消融：在相同 global bank 下，区分近期位置可访问、近期层私有表示和单纯增加局部条目数的效果。

论文还推导了精确的逐层 suffix 执行。上层只需最后位置的输出，但为构建其局部 KV，需要前层一段输入；再往前，所需区间继续扩大。以 8 个上层、包含当前位置的 128-token 窗口为例，一个充分的统一 suffix 可长达 1,017 tokens，而非 128。逐层裁剪可以少做一部分计算，保留完整 cache 的精确性；bounded replay 则通过进一步截断接受近似。二者优化的条件不同。

### 8.2 共享策略需要训练适配，也会改变并行切分

[Stochastic KV Routing](https://arxiv.org/abs/2604.22782) 在训练时让层随机选择自己的 KV 或某个前层的 KV，推理时再固定共享方案。训练中的随机性作用在 KV 来源，而不是跳过整个层；目标是让 query 对不同来源更稳健，使同一模型能适应不同缓存预算。

它适合回答“部署时才知道共享多少层怎么办”，但不意味着任意共享配置都无损。应同时检查 full-cache 精度、不同共享倍数，以及转换后的长文检索。

共享组也会约束 pipeline 的层切分：producer 与 consumer 跨 stage 时，需要传输 KV activations，并在反向汇聚 consumer 的梯度。KV-Pipe 的负载调节与训练通信细节见[第四篇 §8.4]({% post_url 2026-08-18-long-context-large-scale-llm-training-optimization %}#kv-sharing-pipeline)。

对预训练实验，建议至少固定三组比较条件：相同总参数或明确补齐省去的 projection 参数；相同训练 tokens 并另报 FLOPs；相同 decode batch、输入/输出长度和 cache precision。否则结构能力、训练预算与“缓存小了所以能开更大 batch”的收益很容易混在一起。

## 9. 回到开头：深度还在，历史写入可以更稀疏

跨层 KV 复用并不要求模型变浅。它保留多层 query、输出投影和 FFN，把历史表示的写入频率降下来。随后还有几种独立选择：是否保留本层局部历史，是否允许各层重新索引，以及是否把缓存生成提前到网络中部。

多轮 agent 让这一区分更有实际意义：即使复用已有 prefix，新的工具观测仍要进入模型。YOCO-style 架构将处理这些输入所需的缓存构建深度，与生成时使用的完整模型深度分开；收益是否足够大，需要按真实请求的缓存命中、剩余 prefill 和 decode 成本评估。

CLA 适合建立最小机制理解；YOCO 解释为什么生成来源会改变 prefill，KITE 则把这种分工用于训练中的模型扩容；CSA2/CED 和 HySparse2 展示局部状态带来的取舍；GSM 把共享对象推进到已聚合历史的有界状态；LatentIndex 说明共享表示仍可保留独立选择；VFold 则提醒我们，K 与 V 也不必采用同一种压缩。

读新论文时，最有用的检查顺序是：标出物理保存的 bank，再标出每个 bank 的生成来源、读取层和选择集合。最后核对省掉的是容量、投影、索引还是整段前向依赖。这样才能判断它是否适合自己的模型，而不是只比较一个总压缩倍率。

## 延伸阅读

- 从实现入手：先读 CLA §2 和 YOCO §2，对照本文伪代码，分别画出“谁写 cache”和“谁读 cache”。MLKV 与 *A Systematic Study of Cross-Layer KV Sharing* 可补充不同共享分组和来源深度。
- 研究训练扩容：KITE §2–3 与 Appendix D 讲 SST 的逐层 KV 配对、entry bridge、权重/optimizer state 继承；§4.3 和 Appendix A 说明 prefill/decode 成本权重。尤其要区分结构不变、权重冻结和函数保持。
- 研究 prefill：先读 HySparse §3 的组内共享，再对照 HySparse2 §3–4 的 KV Bridging、token 选择和局部分支消融，以及 DeepSeek-V4.1-Flash §2.2、§3.2.2。重点是层私有状态是否存在、需要精确 suffix 还是近似 replay，以及第一个生成位置怎样执行。
- 研究共享状态：GSM §3 说明如何复用 encoder 的候选并累积读取结果，§4 与 Appendix A 给出状态窗口、检索深度和效率口径。区分固定的 decoder 状态与仍随历史增长的 encoder cache。
- 研究 sparse routing：按 IndexCache → CLSA → CSA2 Reindex → LatentIndex 阅读。区分 main KV、indexer keys、候选池和最终 indices，尤其注意各篇速度数字的统计对象。
- 迁移现有 checkpoint：VFold 提供 value-only 校准压缩，Stochastic KV Routing 提供训练适配，二者改变模型的方式不同。较早的 [KVSharer](https://arxiv.org/abs/2410.18517) 也值得对照：原始 KV 相似度本身不足以决定可共享性，应直接测任务 loss 和检索质量。
- 做质量消融：局部历史论文将 history content、input source、entry count 和 fusion 分开控制，可用于设计自己的实验，而非直接照搬其小模型结论。

## 参考资料

- [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981)，2024-05。
- [You Only Cache Once: Decoder-Decoder Architectures for Language Models](https://arxiv.org/abs/2405.05254)，2024-05。
- [KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](https://arxiv.org/abs/2609.27294)，StepFun，2026-09。
- [Layer-Condensed KV Cache for Efficient Inference of Large Language Models](https://arxiv.org/abs/2405.10637)，2024-05。
- [MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding](https://arxiv.org/abs/2406.09297)，2024-06。
- [A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference](https://arxiv.org/abs/2410.14442)，2024-10。
- [KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing](https://arxiv.org/abs/2410.18517)，2024-10。
- [Introducing Gemma 3n: The developer guide](https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/)，官方工程说明。
- [HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing](https://arxiv.org/abs/2602.03560)，2026-02。
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)，2026-03。
- [Stochastic KV Routing: Enabling Adaptive Depth-Wise Cache Sharing](https://arxiv.org/abs/2604.22782)，2026-04。
- [You Only Index Once: Cross-Layer Sparse Attention with Shared Routing](https://arxiv.org/abs/2606.06467)，2026-06。
- [KV-Pipe: On the Relation Between KV Sharing and Pipeline Parallel Efficiency in LLMs](https://arxiv.org/abs/2608.15943)，2026-08；正文同时参考其修订版。
- [DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression](https://arxiv.org/abs/2609.19969)，2026-09。
- [HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing](https://arxiv.org/abs/2609.26368)，2026-09。
- [Shared Global KV with Layer-Specific Local History](https://arxiv.org/abs/2609.28006)，2026-09。
- [GSM: Efficient Language Modeling with Shared Global State](https://arxiv.org/abs/2609.33465v3)，2026-09；本文依据 2026-10-15 的 v3。
- [LatentIndex: Cross-Layer Sharing with Layer-Specific Selection for Sparse Attention](https://arxiv.org/abs/2610.04635)，2026-10。
- [VFold: Symmetry-Aware Cross-Layer Value Cache Compression](https://arxiv.org/abs/2610.12338)，2026-10。
