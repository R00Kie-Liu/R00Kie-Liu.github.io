---
layout: post
title: "长话长说（三）：Attention 设计趋势，选择、压缩与访问调度"
date: 2026-08-04
last_modified_at: 2026-10-21
tags: [LLM, Attention, Long Context, Pretraining, Inference]
description: "梳理 2026 年前后长上下文 attention 机制的研究趋势：selector/indexer 降本、选择粒度重排、linear state 与 sparse retrieval 合流、表示感知的压缩与执行，以及 runtime 访问调度。"
---

## 写在前面

> 这是长上下文 attention 架构系列的第三篇。第一篇看 [attention 与 KV cache 总览]({% post_url 2026-07-27-long-context-attention-architecture-survey %})，第二篇单独拆 [linear attention]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})；这一篇讨论最近工作的方向。 完整目录见[系列导航]({% post_url 2026-07-27-long-context-attention-architecture-survey %}#series-map)。

第一、二篇已经解释了 KV 压缩、稀疏选择和递推状态。继续读近期论文，更值得追问的是：这些部件进入模型后，原来的瓶颈消失了吗，还是转移到了选择、压缩和访问调度上？

下面沿五个设计问题展开：选择过程本身花多少钱，选择单位怎样影响证据保留，多种历史通道如何分工，压缩与执行怎样适配缓存表示，以及 runtime 能否利用模型给出的访问计划。每个方向都用机制或实验支撑，基础定义留给前两篇。

## 1. 先拆四个互相挤压的成本

长上下文 attention 的优化对象可以先压成四个坐标。

第一是 **选择计算**：历史 scoring、top-k 和候选管理花多少时间。第二是 **cache 容量**：每个历史 token 要留下多少东西。第三是 **读取量与访存形态**：当前 query 读多少历史，地址是连续 block 还是离散 token。第四是 **runtime 可调度性**：访问计划能否提前暴露给 cache manager、prefetcher 和 serving backend。

这四个坐标比“某个方法是不是 sparse”更有用。DSA、QSA、MSA、LCA、SparDA 这些工作看似都在省 attention，实际动刀位置不同：有的省 main attention，有的省 indexer，有的压 cache，有的让访问计划更容易被系统利用。

后面几节按“选择成本 → 选择单位 → 记忆分工 → cache 语义 → serving 访问计划”往下走。Kimi K3、Ling/Ring 2.6 这类 hybrid block 设计仍然重要，但在这里只作为背景，不再展开配比。

## 2. 趋势一：selector/indexer 成为新的主战场

早期 sparse attention 的说法通常是：full attention 要看全部历史，sparse attention 只看 top-k，于是计算量下降。这个说法没错，但它漏掉了一件事：top-k 本身从哪里来？

如果 selector 仍要对完整 prefix 扫一遍，attention core 省下来的计算会转移到 indexer。上下文到 128K、512K、1M 后，这个转移很容易变成新的主瓶颈。2026 年的一批工作，核心都不再是“能不能稀疏”，而是“稀疏决策本身能不能便宜、稳定、硬件友好”。

| 工作 | 它主要改哪一段 | 直觉 |
|---|---|---|
| HISA | DSA 的 token-level indexer | 先粗筛 block，再在候选 block 内做 token top-k |
| LongCat/LSA | DSA/Lightning Indexer 的训练与访存成本 | streaming-aware indexing、cross-layer indexing、hierarchical indexing 分别处理训练、跨层复用和推理访问 |
| MISA | DSA indexer heads | 把多个 indexer heads 看成 MoE，只激活少数 head 做重 scoring |
| DoSA | DSA indexer 的完整打分与选择 | 先证明哪些 token 不可能进入 top-k，再跳过它们的完整多头打分 |
| Tactic | 固定 top-k budget | 用累计 attention mass 控制近似质量，token 数随 query 改变 |

HISA 是最容易理解的一条。DeepSeek Sparse Attention 一类 token-level sparse 方法会为每个 query 给历史 token 打分，再把 top-k token 交给 Sparse MLA。粒度很细，但 indexer 要扫 prefix。

HISA 把搜索路径改成两级：

1. block-level coarse filtering：用 pooled block representation 先排除大部分无关区间；
2. token-level refinement：只在候选 block 里运行原来的 token indexer，输出仍是 token 级 top-k。

![HISA two-stage indexer](/assets/img/blog/long-context-architecture-watchlist/hisa_indexer_fig1.png)

*图源：HISA, Figure 1。左侧是 DSA indexer 对历史 token 逐个打分，右侧是 HISA 先选 block，再在候选 block 里选 token。*

LongCat/LSA 从 streaming-aware、跨层和分层索引处理同一成本；这里保留两个判断：main attention 变稀疏，indexer 仍可能有 $$O(L^2)$$ scoring；top-k 地址离散，还可能把执行瓶颈推向访存。具体索引优化见延伸阅读。

[MISA](https://arxiv.org/html/2605.07363#S4.SS1)先让所有 indexer heads 只对少量 pooled block keys 打分，按每个 head 在这些块上的加权贡献绝对值均值，选出活跃 heads；随后仅用它们扫描原始 token keys，再取 token top-k。HISA 沿 head 轴归约、选 blocks，MISA 沿 block 轴归约、选 heads。这里的 router 是便宜的统计计算，不需另外训练一个 FFN-MoE 路由器。

<a id="dosa-certified-pruning"></a>

### 2.1 DoSA：不算完整分数，也能证明某个 token 选不上

HISA 与 MISA 都缩小了搜索开销，但它们保留 token-level 输出，不代表保留原始 DSA 的同一组 top-k。HISA 的 block 粗筛可能漏掉关键 token；MISA 即使在初筛后用全部 heads 精排，也无法救回已被排除的赢家。[DoSA（Dots Sparse Attention）](https://arxiv.org/abs/2610.13990)追问的是：能否少做完整打分，又不改变原 indexer 的选择规则？

DSA indexer 的分数为 $$s_i=\sum_h w_h[q_h^\top k_i]_+$$，其中 $$q_h$$ 是当前 query 的第 $$h$$ 个 indexer head，$$k_i$$ 是历史 token 的 key，$$w_h$$ 是 head 权重，$$[x]_+=\max(x,0)$$。少算 head 时，关键是给未算的贡献留出一个不会低估的余量。

用一个自造的二维例子：只有两个 head，权重均为 1，令 $$q_1=(2,0)$$、$$q_2=(1,0.1)$$。只计算 head 1，将另一个拆成已知方向与残差：

$$
q_2=0.5q_1+d,\qquad d=(0,0.1).
$$

由 ReLU 的次可加性和柯西不等式，未计算的 head 满足：

$$
[q_2^\top k_i]_+
\le 0.5[q_1^\top k_i]_++\|d\|\|k_i\|
=0.5[q_1^\top k_i]_++0.1\|k_i\|.
$$

所以只做一次 head 点积，就能得到两头总分的上界：

$$
s_i\le U_i=1.5[q_1^\top k_i]_++0.1\|k_i\|.
$$

假设要选 top-2，已经完整打分的两个种子 token 得分为 10、8，全体 token 的第二名就至少有 8 分。用 $$\tau=8$$ 筛查其他 key：

| 待检查的 key | 已算的 $$q_1^\top k_i$$ | 上界 $$U_i$$ | 决定 |
|---|---:|---:|---|
| $$(1,0)$$ | 2 | 3.1 | 排除：最高也不到 8，无需算 head 2 |
| $$(3,0)$$ | 6 | 9.3 | 保留并完整打分，实际得分为 9，进入 top-2 |

一般地，从可见历史取至少含 $$K$$ 个不同 token 的种子集 $$\mathcal S$$，以其第 $$K$$ 大完整分数为 $$\tau$$；记全体可见 token 的第 $$K$$ 大分数为 $$T_K$$，则：

$$
\tau=\operatorname{KthLargest}_{i\in\mathcal S}s_i\le T_K,
\qquad s_i\le U_i<\tau\le T_K.
$$

这条不等式就是不漏选的依据：只剪掉 $$U_i<\tau$$ 的 token，等于阈值也保留。上界偏松只会多留候选、多算一些；幸存者仍按原始完整分数排名。

上面的拆分对应论文 §2.3 的非负投影与残差范数界；实际算法把多个未计算的正贡献折叠到少量 retained heads 上，并可联合残差进一步收紧上界。负权重 heads 若直接忽略，也能得到安全但偏松的上界；DoSA 用聚合方向估计至少应扣除多少分，再收紧它。投影系数和残差界按 query 准备后沿 key 维复用，key 范数也可复用，省下的是大量 head × key 点积。

![DoSA certified screening and refinement pipeline](/assets/img/blog/long-context-architecture-watchlist/dosa_execution_fig2.png)

*图源：[DoSA, Figure 2](https://arxiv.org/pdf/2610.13990)。按 ①→④ 阅读：少量 heads 筛查 → 计算上界并压紧候选 → 幸存者完整多头打分 → 原始分数归约与稀疏写回。图中省略了种子初始化和最终 top-k；灰色、紫色分别表示省去的计算和写回。*

这张图还说明，剪枝比例不能直接换算成 GPU 加速比。幸存者变成离散地址后，仍要组织 key 读取和 Tensor Core 计算；论文把候选压紧、共享 key 加载、完整打分后的阈值过滤接在一起，才同时减少算术、HBM 写回与 top-k 输入。

| 方法 | 少算了什么 | 相对原始 DSA top-k 的保证 |
|---|---|---|
| HISA | 未入选 block 内的 token 打分 | 粗筛可能漏选，保留的是 token-level 接口 |
| MISA | 初筛阶段的大部分 indexer heads | 部分头打分属于近似，后续精排不能补回漏选 |
| DoSA | 已被上界排除的 token 的完整多头打分 | 上界保守、候选不丢失且并列规则一致时，可保持原始选择 |

这里的“无损”有实现条件。FP8 路径需要覆盖数值误差的保守界；候选 buffer 溢出不能直接截断，必须精确合并或恢复全域计算。它也只相对于原始 DSA scorer，而非与 dense softmax attention 等价。DoSA 仍扫描全部 key，不直接压缩主 KV cache，也没有把最坏情况下的二次 prefill 复杂度消掉。

prefill 与 decode 的实现还要分开：前者利用 query 分组执行上面的筛查与完整打分；后者每步 query 很少，论文保留完整 FP8 scoring，用近期选中的 key 在当前 query 下重新打分来建立阈值，主要减少选择与写回。实验中约 1M 上下文、TP8、请求间清空 prefix cache 时，相对 SGLang 的 TTFT 加速在 GLM-5.2、DeepSeek-V3.2 上分别为 1.86×、1.94×；这不是生成阶段吞吐提升，也不是长上下文质量提升的证据。

### 2.2 跨层复用与自适应预算

索引还可以沿网络深度复用。IndexCache、CLSA、CSA2 和 LatentIndex 分别复用最终选择、共享缓存或连续索引表示，保留的层间自由度不同。这条路线的机制与取舍见[第六篇：KV 跨层复用]({% post_url 2026-10-13-cross-layer-kv-reuse %})。

Tactic 不固定每步读 $$k$$ 个 token，而以保留的 attention mass 为目标。若 $$a_i$$ 是完整 attention 在位置 $$i$$ 的归一化权重，目标集合 $$I$$ 满足：

$$
p(I)=\sum_{i\in I} a_i,\qquad p(I)\ge \tau.
$$

[实现](https://arxiv.org/html/2502.12216#S4)没有先计算全部 $$a_i$$：prefill 对 keys 做聚类；decode 用 query 与聚类中心的点积给簇排序，再采样部分 token，拟合排序后的指数分数分布，以估计达到质量阈值 $$\tau$$ 所需的读取量；头部异常高分项单独计算。最后对选中 tokens 运行实际 attention。分布尖时少读、平时多读，但估计误差会影响保留质量，这不是 DoSA 式的精确排除保证。

这组工作给第三篇带来的趋势判断很明确：长上下文 attention 的研究重心，正在从 main attention kernel 旁移到 selector。谁来选、多久选一次、跨几层复用、选出来的地址是否连续，都会影响最终吞吐。

## 3. 趋势二：选什么单位，正在变成架构问题

稀疏 attention 不是一个单点方案。真正难的是粒度：选 token 精确但难跑快，选 block 规整但可能漏细节，选 micro-block 介于两者之间；如果底层是 MLA，甚至还要问是在 latent space 里选，还是还原成 full K/V 后再选。

可以先用一张表抓住差异：

| 路线 | 选择单位 | 优点 | 代价 |
|---|---|---|---|
| DSA / HISA / MISA | MLA/MLA-style cache 上的 token，或候选 block 内 token | 证据粒度细，适合精确检索 | indexer 成本高，KV 地址不连续 |
| NSA | compressed block + selected token + sliding window | 把全局概览、细粒度证据和局部连续性放在一起 | pattern 复杂，需要训练时原生支持 |
| CSA / HCA | compressed entries / heavily compressed entries | 先把远程历史压短，再选择或读取 | 压缩损失和 heterogeneous cache layout |
| MiniMax MSA | GQA group 下的 KV block | 访存更规整，block-sparse kernel 更容易做快 | block recall 决定质量上限 |
| Qwen3.8-Flash-Next / QSA | micro-block | 比 token-level 更规整，比粗 block 更细 | 需要轻量 indexer 和 sparse core attention 配合 |
| LCA | MLA latent group | 同时减少 cache footprint 和 attention 向量数 | pooling 可能抹掉组内少数关键 token |

表中的基础机制见第一篇，这里关注选择单位造成的后果。整块选择有利于连续访存，却会为少数关键 tokens 同时支付邻近位置的预算；逐 token 选择能把预算分散给不同轮次的证据，又会增加 top-k 和离散 gather 的成本。Micro-block 缩小了这两个极端之间的跨度，压缩 entry 则进一步改变了证据本身的表示。

[HySparse2 的粒度消融](https://arxiv.org/abs/2609.26368)在相同主干、全局 token 预算和局部窗口下，比较 64-token block 与 token selection，观察到检索与图推理收益。这支持细粒度在特定任务上的价值；硬件吞吐是否更好，还要单独测量。其完整架构与两版演进见[第六篇]({% post_url 2026-10-13-cross-layer-kv-reuse %}#hysparse-evolution)。

因此，比较粒度时应同时固定读取预算，并分别报告证据召回和访问成本。只给 sparsity ratio，无法分辨改进来自更准的选择、更好的表示，还是更规整的执行。


<a id="7-位置编码和-cache-语义也开始分层"></a>

## 4. 趋势三：linear state 和 sparse retrieval 合到同一个 recipe

这一节的问题是：将 recurrent state 与 sparse retrieval 放在同一个模型中，是否真的形成了不同的信息通道？Qwen 的组合设计提供了实例，状态干预实验则提供另一类证据。

Qwen3.8-Flash-Next 是这里最合适的主例。它的 token mixing 是 **3 个 Gated DeltaNet 层 + 1 个 QSA 层**：GDN 把多数层的历史压进 fixed-size recurrent state，避免每层都维护和扫描长 KV；QSA 则接管周期性全局层，用 micro-block indexer 先筛远程上下文，再做 sparse core attention。

这个组合比普通 recurrent/full hybrid 更进一步：全局层本身也不再是 dense attention。换句话说，Qwen3.8-Flash-Next 把两个问题拆给两个模块：GDN 负责“多数层不要随 $$L$$ 增长”，QSA 负责“少数层仍能按内容找远程证据，但主 attention 只读取选中上下文”。QSA indices 跨 draft steps 的复用，则把这一分工进一步延伸到 runtime，见 §6.1。

MiniCPM-SALA 可以作为另一个对照。SALA 是 Sparse Attention and Linear Attention：25% sparse attention，75% linear attention。linear 层承担大部分低成本长程混合，sparse 层保留少数远程证据的精确读取。它的 HyPE 还把位置编码按层拆开：linear attention 层应用 RoPE，让 recurrent memory 保持位置敏感；sparse attention 层不加 RoPE，尽量减少远距离信息受位置信号衰减的影响。

Kimi K3 的另一种分工是 KDA 提供位置敏感与近因偏置，MLA 层使用 NoPE；SALA 则让 linear 层使用 RoPE、sparse 层不加 RoPE。它们说明位置处理需要配合信息通道，而不是给出一条统一的 NoPE 或 RoPE 结论。

SALA 还强调迁移路径：用 HALO 做 Transformer-to-hybrid conversion，再稳定训练和长上下文继续训练。这个点对预训练工程很现实。attention 结构不是单独替换一个 PyTorch module 就结束；它会连到 checkpoint conversion、warmup、位置编码和后续长上下文数据阶段。

这种分工也开始有直接的干预证据。[What Attention Recalls and Recurrence Controls in Hybrid Language Models](https://arxiv.org/abs/2609.04434)（2026-09）在 Qwen3.5、Falcon-H1 上做两种实验：split-prefill 只保留预填充得到的 KV 或 recurrent state；state-swap 则把一段上下文的 KV 与另一段上下文的 recurrent state 配对。在论文测试的任务里，精确检索主要沿 KV 通道保留，输出语言和 persona 等控制信息更多沿 recurrent state 保留；交换状态后，答案内容与表达方式也呈现相应分离。

这给 hybrid recipe 增加了一种检查方法：在同一个 checkpoint 内隔离两种历史表示，观察任务具体依赖哪一条路径。它不能证明所有 recurrent 模型都缺乏精确检索能力，也没有直接测试这里的 Qwen3.8-Flash-Next；但比仅凭层数配比推测模块职责更接近因果验证。


<a id="5-趋势四被选对象怎么存必须贴着-mla-cache-语义"></a>
<a id="5-趋势四压缩方法开始围绕实际缓存表示设计"></a>

## 5. 趋势四：压缩与执行开始围绕实际缓存表示设计

序列压缩不能只描述“多少 token 变成一个向量”，还要交代被压缩的对象及其位置语义；同一个 latent 表示怎样执行，也会改变缓存读取与计算成本。这里有四种不同关系：

| 路线 | 实际缓存与操作 | 比较重点 |
|---|---|---|
| MLA + DSA/HISA | 保留 token-level latent KV，改变读取集合或搜索路径 | 不恢复全部历史 K/V，怎样高效选择 |
| DeepSeek-V4 CSA/HCA | 序列压缩后在 shared-KV MQA 上读取，CSA 另做 top-k | 压缩率、选择损失与近期精确窗口 |
| MLA + LCA | 对 semantic latent 做 pooling，对位置 key 选 anchor | 内容与位置是否需要不同压缩规则 |
| GQLA | 同一 joint latent 支持紧凑吸收与展开 GQA 两条路径 | 缓存字节数、计算量和 TP 分片怎样共同影响速度 |

CSA/HCA 属于从 DeepSeek 系列继续探索压缩历史的路线，但其 shared-KV MQA 不能当作标准 MLA latent cache 的简单叠加。基础结构见第一篇；这里最值得进一步展开的是 LCA 怎样利用 MLA 已有的内容／位置拆分。

### 5.1 LCA：在 MLA latent space 里压上下文

Latent-Condensed Attention (LCA) 直接在 MLA 的缓存空间里压缩：推理时保存的是 latent $$C^{KV}$$ 和位置 key $$K^R$$，condensation 也作用在这两个对象上。

MLA 里的信息有两类：

| 组件 | 作用 | LCA 怎样处理 |
|---|---|---|
| semantic latent $$C^{KV}$$ | 表示 token 的内容语义 | 加权合成近似代表 |
| positional key $$K^R$$ | 承载 RoPE 位置关系 | 选一个真实位置作 anchor，避免混合位置信号 |

[LCA §4.2–4.3](https://arxiv.org/html/2604.12452#S4.SS2)把远处历史按连续的 $$g$$ 个 token 分组，近期窗口保留原始 latent 和位置 key。权重来自一个具体的打分过程：取最近 $$g$$ 个 query 的平均向量，给组内各 key 做缩放点积，再在组内 softmax，得到 $$\alpha_i$$。它反映近期 query 关注哪些内容，不是固定的平均权重。

![Latent-Condensed Attention overview](/assets/img/blog/long-context-architecture-watchlist/lca_overview_fig1.png)

*图源：Latent-Condensed Attention, Figure 1。semantic latent 使用 pooling，positional key 使用 anchor selection。*

例如一组有四个 token，得到的权重为 $$(0.1,0.2,0.6,0.1)$$：内容 latent 按这个比例混合，位置 key 则直接取第三个 token 的。对组 $$G$$，即：

$$
C^{KV}_{\mathrm{rep}} = \sum_{i \in G} \alpha_i C^{KV}_i,
\qquad
K^R_{\mathrm{rep}} = K^R_{j^\ast},\quad j^\ast=\arg\max_{i \in G}\alpha_i.
$$

后续 attention 读取的是“远处的代表条目 + 近期原始条目”。decode 时，每积累 $$g$$ 个新 token，就用最近 $$g$$ 个 query 的平均向量，为缓冲区最早的 $$g$$ 个 token 生成一个代表并追加到历史缓存；已压缩的旧组不会每步重新展开、重算。这样减少了条目数，但单个代表无法保留组内所有细节，group size 和近期窗口大小仍影响质量。

LCA 说明表示感知的压缩与选择粒度是两项独立设计：HISA 改搜索路径，LCA 改被搜索的对象。BLASST 则在 kernel 内决定是否继续计算，放在 §6.4 对照。

<a id="gqla-dual-path"></a>

### 5.2 GQLA：同一份权重，缓存 latent 还是展开的 K/V？

LCA 减少历史代表向量的数量，[GQLA（Group-Query Latent Attention）](https://arxiv.org/abs/2605.15250)则不在这里合并 token。它改的是每个 token 的 KV 参数化：保留一个 joint latent，让同组 query heads 共享 K/V 上投影，从而让同一套权重既能走紧凑的 MQA-absorb 路径，也能走可分片的 GQA 路径。这里依据的是 2026-10-14 的 v4。

![GQLA equivalent GQA and MQA-absorb execution paths](/assets/img/blog/long-context-architecture-watchlist/gqla_dual_paths_fig2.png)

*图源：[GQLA, Figure 2](https://arxiv.org/pdf/2605.15250v4)。左侧先展开每组 K/V，调用 GQA kernel；右侧把上投影吸收到 query/output 计算中，直接读取共享 latent。两条路径使用同一套 GQLA 权重，差别在中间表示和实际缓存布局。*

设 token 的共享 latent 为 $$c_s\in\mathbb R^r$$，第 $$j$$ 组的内容 key/value 上投影为 $$W_j^K,W_j^V\in\mathbb R^{d\times r}$$。当前 query head 属于组 $$j$$，内容 query 为 $$q\in\mathbb R^d$$。忽略单独保留的 RoPE 分支，内容打分与 value 汇聚可以写成：

$$
q^\top(W_j^Kc_s)=((W_j^K)^\top q)^\top c_s,
\qquad
\sum_s a_s W_j^V c_s=W_j^V\!\left(\sum_s a_s c_s\right).
$$

其中 $$s$$ 是历史位置，$$a_s$$ 是当前 head 在该位置的 softmax 权重。左边先展开历史 K/V；右边将 key 上投影移到当前 query，将 value 上投影移到加权汇聚之后。保持位置分支和 softmax 缩放不变时，两条路径在精确算术下得到相同输出。

把等式对应到一次 decode，执行顺序是：

```text
GQA：   历史 latent 先展开为各组 K/V 并缓存
        -> 当前 query 读取所属组的 K/V -> 汇聚输出
absorb：历史只缓存 latent 和位置 key
        -> 变换当前 query -> 对 latent 做 attention -> 变换汇聚结果
```

| 执行路径 | 历史里存什么 | 主要取舍 |
|---|---|---|
| MQA-absorb | 共享 latent 与位置 key | 单份缓存紧凑，但 attention 在 latent 维计算；论文采用的 head-wise TP 下，latent 跨 rank 复制 |
| GQA | 每组展开的 K/V | 单卡完整缓存可能更大，但组可以沿 TP 分片，也能复用已有 GQA kernel |

这里的 rank 指一个 TP 进程，论文部署中对应一张 GPU。GQA 的组可以分给不同 GPU，各卡只存自己负责的组；上述 absorbed 路径则每卡保留共享 latent。因此，展开未必增加单卡显存：论文配置在 TP=1 时 GQA 的完整缓存更大，到 TP=8，每 token 每 rank 占 640 bytes，反而小于 absorbed 路径的 1152 bytes。这些数已计入实际 kernel 的位置 key 布局。

硬件与 TP 会一起改变路径偏好：论文 kernel 实验在 TP=1 时，H800 偏向 MQA-absorb、H20-3e 偏向 GQA；TP=8 时 GQA 在两者上都更快。8 张 H20-3e 上，同一转换后模型在相同 batch、8K–64K 输入、512-token 输出且不开 MTP 时，GQA 吞吐提高 22.42%–34.97%；另一个按各自容量调 batch 的实验达到 50.07%，包含额外的容量收益。

最后要区分两件事：固定 GQLA 模型的双路径等价，不代表任意 MLA/GQA checkpoint 都能无损转成 GQLA。论文的 TransGQLA 转换含近似，部分模型需要 CPT/SFT 恢复；路径也在部署配置时选定，并未验证在线逐步切换缓存。它提供的是一个更具体的趋势：架构可以主动暴露多种等价执行形式，让硬件、TP 和负载决定使用哪一种。

<a id="csa2-ced"></a>

沿深度复用还会改变缓存的生成来源与读取层。CSA2/CED 如何分开共享 KV 与索引，以及局部状态为何需要 replay，在[第六篇 §4]({% post_url 2026-10-13-cross-layer-kv-reuse %}#csa2-ced)展开。


<a id="6-趋势五attention-pattern-要变成推理-runtime-的访问计划"></a>

## 6. 趋势五：选择结果与按需计算进入 runtime

<a id="qsa-index-reuse"></a>

### 6.1 QSA：一次选择可以服务多个 draft steps

Qwen3.8-Flash-Next 报告提到，MTP module 会在 speculative decoding steps 之间复用 QSA indices。普通 QSA 先由 indexer 选出 top-k micro-block ids，主 attention 再读取对应 KV；MTP 则将这组远程索引用于后续 draft steps，减少重复 scoring、top-k 和地址构造。

一种理解是：相邻 draft steps 共享很长的 prefix，变化主要在很短的 tentative suffix，因此远程选择有复用空间。这是对设计动机的解释，报告未据此证明任意步长下索引都稳定。复用改变的是 draft 路径，可能影响接受率和延迟；最终输出仍走相应的验证流程。

例如用一组远程 micro-block ids 服务多个 draft steps，复用的是“去哪里读”的地址；各步仍用自己的 query 重新计算 attention 权重。这不要求各步的权重或新增 KV 相同。

### 6.2 从访问计划到预取与分层缓存

长上下文 decode 每生成一个 token 都要读历史。Sparse attention 把读取范围缩小后，系统仍要做三件事：算出要读哪些块，确认这些块在 HBM 还是 CPU，把下一层需要的块尽量提前搬上来。做不到这三件事，理论 sparsity 很容易被 PCIe transfer、page table 和 metadata 开销吃掉。

[SparDA](https://arxiv.org/html/2606.04511#S4.SS1)在 Q/K/V 之外增加一个 **Forecast** projection。第 $$l$$ 层产生的 $$F_l$$ 与下一层已有历史 key 的压缩摘要打分，选出下一层要读的 blocks；这些小摘要留在 GPU，完整历史 KV 可以放在 CPU。因此，不必先搬完整 KV 才能决定搬哪些块。

一次 decode 的先后关系是：

```text
Layer l:
  F_l + 下一层 GPU 上的历史 key 摘要 -> 下一层 block ids
  发起所选历史 KV 的 CPU->GPU 预取
  Q_l 执行本层 attention，与预取重叠
Layer l+1:
  等待所需 KV 就绪，用 Q_{l+1} 对这些 KV 计算 attention
```

初始与近期窗口另行保留，远程选择不依赖下一层尚未生成的当前 token key。预取只是尽量隐藏传输：若搬运比本层计算更久，下一层仍需等待。prefill 时 KV 已在 GPU，主要收益则来自更轻的 Forecast selector。

![SparDA architecture](/assets/img/blog/long-context-architecture-watchlist/sparda_architecture_fig2.png)

*图源：SparDA, Figure 2。右侧的 decode 路径最关键：Forecast 先预测下一层要读的 KV blocks，runtime 才有机会把 prefetch 藏到当前层计算里。*

[SPIN](https://arxiv.org/html/2604.26837#S4)接管的是选完之后的取数。partition 是算法选出的逻辑单位，可以是一块、一簇或一个 token；page 是分配和搬运内存的物理单位，一个 partition 可以对应多个 pages。SPIN 用映射表把两者接起来。

用一个示意过程理解：selector 选中 partition 2、7；查表发现 2 已在 GPU，7 还在 CPU，就复用 2，只为 7 腾出 GPU pages 并搬运对应 KV。随后把两者的 GPU page 地址交给 attention。下一步若仍选中 7，就能直接命中缓存。不同算法可以保留各自的选择逻辑，共用这套查询、替换和搬运流程。

![SPIN memory manager](/assets/img/blog/long-context-architecture-watchlist/spin_memory_manager_fig5.png)

*图源：SPIN, Figure 5。沿 Retrieve 路径阅读：逻辑 partition ids → 查询驻留状态与 page 映射 → 为缺失项安排 GPU 空间并搬运 → 返回 attention 所需的 GPU pages。*

相关工作各补一个环节：AsyncTLS 结合 block 粗筛与 token 精选，利用相邻步骤的重叠访问减少新增块传输；Vortex 的 vFlow/vTensor 提供 Python 前端与 page-centric tensor 抽象，方便把新 pattern 接入 serving backend。具体机制留在延伸阅读；这里的主线是“提前选地址 → 查缓存 → 只搬缺失项 → 按就绪地址计算”。

<a id="61-on-demand-attention先决定这一步是否值得全局读取"></a>

### 6.3 On-Demand Attention：先决定这一步是否值得全局读取

[On-Demand Attention（ODA）](https://arxiv.org/abs/2609.20734)（2026-09）把选择提到每个 decoding step：先用初始 sink tokens 和近期窗口运行 Local，再由 recall head 判断是否值得调用 Full。

~~~text
保存本步开始前的历史状态
  -> 运行 Local，得到候选 hidden state
  -> recall head 估计追加全局计算的收益
  -> 若触发 recall，从相同历史状态重算本步 Full
  -> 只提交被选分支的当前状态，继续生成
~~~

训练时，冻结主干，在同一历史上配对运行 Local/Full，用参考 next token 的 NLL 差减去成本惩罚，监督 recall head。部署时没有参考答案，head 只接收已有 hidden states 和当前 token embedding。历史 KV 始终保留，因此连续 Local steps 之后仍可回读远处；但缓存中的历史来自实际选中的分支，不等于一路运行 Full 得到的历史轨迹。

ODA 保留历史 KV，省去部分步骤的全局读取；触发 Full 时仍需支付 Local 尝试的成本。报告中接近 2× 的 decode 吞吐来自 A100、Qwen3-1.7B、约 128K 输入和预设 recall schedule，不含 prefill，也不能直接当作学习策略的端到端加速。


<a id="52-hisa-和-blasstmla-之后indexer-与-kernel-都要跟上"></a>

### 6.4 BLASST：在 kernel 内判断哪些块可以少算

[BLASST](https://arxiv.org/html/2512.12087#S3.SS1)在 FlashAttention 扫描 KV blocks 的过程中，判断哪些块后续可以少算。假设此前最大的 attention logit 是 10，当前块最大只有 0，那么当前块每一项相对已见最大项的指数值至多为 $$e^{0-10}\approx4.5\times10^{-5}$$。这为忽略低贡献块提供了依据；它不是整个块的总误差保证。

kernel 用当前块的最大 logit $$\tilde m_j$$ 与扫描至今的最大值 $$m_j$$ 比较，阈值为 $$0<\lambda<1$$：

$$
\tilde m_j - m_j < \log \lambda.
$$

例如 $$\lambda=10^{-4}$$ 时，上面的块满足条件。此时 $$QK^\top$$ 已经算过，才能知道块内最大值；省去的是后面的 softmax 指数与归约、V block 读取及 $$PV$$ 计算。

与 ODA 的整步 Local/Full 选择不同，它在一个 kernel 内跳过块的后续处理；与 DoSA 的 top-k 排除证明也不同，这里忽略的是小而非严格为零的贡献，阈值引入近似。它可以服务 full/MLA attention，收益取决于分数分布与块顺序，不压缩历史 latent。

<a id="8-小结长上下文-attention-正在从算子变成协议"></a>

## 7. 小结：长上下文 attention 正在从算子变成协议

把前面的线索放在一起，长上下文 attention 的研究重心已经不只是替换 softmax：

| 层级 | 代表工作 | 解决什么 | 新问题 |
|---|---|---|---|
| 历史通道分工 | Qwen GDN+QSA、SALA、state-swap 实验 | 检查状态与稀疏读取各承担什么 | 层配比不能替代功能证据 |
| sparse selector | DSA、HISA、MISA、DoSA、Tactic、LongCat/LSA | 降低重要历史的选择与读取成本 | 近似筛选与精确剪枝的保证不同，排序与离散访存仍有开销 |
| block/micro-block sparse | NSA、MiniMax MSA、Qwen QSA | 更接近硬件友好的稀疏计算 | block recall 与粒度选择 |
| 表示感知的压缩与执行 | LCA、CSA/HCA、GQLA | 让压缩和执行适配 latent 或 shared-KV 的具体结构 | 内容丢失、位置语义与 TP 后的实际缓存布局需要分别检查 |
| runtime 访问与按需计算 | QSA index reuse、SparDA、SPIN、AsyncTLS、Vortex、ODA、BLASST | 安排预取、缓存位置与全局读取时机 | 控制开销、状态提交和真实质量收益都要核对 |

评估这些设计时，除了 attention FLOPs，还应追问 selector 自己花了多少时间、共享或重建状态是否改变质量，以及稀疏访问能否转成真实吞吐。模型决定历史的表示和读取方式，runtime 承担它们的存储、传输和状态提交；两边的假设必须一致。

## 延伸阅读

这篇更像一张趋势地图。后面继续追，可以按下面几条线读。

- **基础结构**：先读第一篇里的 GQA、MLA、SWA、DSA/QSA/MSA、CSA/HCA，再读第二篇里的 Gated DeltaNet、KDA、Lightning Attention。第三篇默认这些概念已经在读者脑子里。
- **selector/indexer 降本**：HISA、LongCat/LSA、MISA、DoSA 和 Tactic 可以放在一起读。DoSA §2 与 Appendix C 尤其适合对照“近似筛掉候选”和“证明候选选不上”，并检查数值误差、候选容量与恢复路径。
- **选择粒度**：NSA、MiniMax MSA、Qwen QSA、MoBA/Optimizing MoBA 都值得对照。重点不在 sparse ratio 本身，而在 token、block、micro-block 对精度、kernel 和访存的不同约束。
- **缓存表示、压缩与执行**：DSA/HISA 改 Sparse MLA 的 token 选择路径；CSA/HCA 读取 shared-KV MQA 的序列压缩条目；LCA 直接压 MLA latent；GQLA §3–§5 对照同一权重的双路径与 TP 分片，注意路径等价不等于 checkpoint 转换无损。BLASST 属于 §6.4 的 kernel 内按需计算。
- **runtime-aware serving memory**：SparDA、SPIN、AsyncTLS、Vortex 更接近推理系统，但在这里读它们，是为了理解 attention pattern 怎样变成 prefetch、page table、metadata 和 serving backend 的输入。
- 按需全局读取：ODA 的 §2 与 Appendix C 讲 recall 监督、Local/Full 状态提交和历史轨迹的限制，适合与始终执行全局检索的 sparse attention 对照。
- **hybrid 功能干预**：*What Attention Recalls and Recurrence Controls* 提供 split-prefill / state-swap 方法，适合用来检查自己训练的模型是否真的形成了预期分工。

## 参考资料

- [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- [Ling and Ring 2.6 Technical Report](https://arxiv.org/abs/2606.15079)
- [MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling](https://arxiv.org/abs/2602.11761)
- [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089)
- [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- [Qwen3.8-Flash-Next Technical Report](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)
- [LongCat Sparse Attention: Taming the Lightning via Streaming-aware Hierarchical Cross-Layer Indexing](https://arxiv.org/abs/2608.01662)
- [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2606.19348)
- [What Attention Recalls and Recurrence Controls in Hybrid Language Models](https://arxiv.org/abs/2609.04434)
- [HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing](https://arxiv.org/abs/2609.26368)
- [On-Demand Attention: Language Models Know When to Recall](https://arxiv.org/abs/2609.20734)
- [MISA: Mixture of Indexer Sparse Attention for Long-Context LLM Inference](https://arxiv.org/abs/2605.07363)
- [DoSA: Dots Sparse Attention](https://arxiv.org/abs/2610.13990)
- [GQLA: Group-Query Latent Attention for Hardware-Adaptive Decoding（v4）](https://arxiv.org/abs/2605.15250v4)
- [MiniMax Sparse Attention](https://arxiv.org/abs/2606.13392)
- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)
- [Optimizing Mixture of Block Attention](https://arxiv.org/abs/2511.11571)
- [Latent-Condensed Transformer for Efficient Long Context Modeling](https://arxiv.org/abs/2604.12452)
- [HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention](https://arxiv.org/abs/2603.28458)
- [Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs](https://arxiv.org/abs/2502.12216)
- [SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511)
- [Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving](https://arxiv.org/abs/2604.26837)
- [AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention](https://arxiv.org/abs/2604.07815)
- [BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding](https://arxiv.org/abs/2512.12087)
- [Vortex: Efficient and Programmable Sparse Attention Serving for AI Agents](https://arxiv.org/abs/2606.06453)
