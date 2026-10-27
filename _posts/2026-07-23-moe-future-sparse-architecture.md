---
layout: post
title: "MoE我们喜欢你（八）：稀疏计算与参数复用的新方向"
series: moe-we-like-you
series_order: 8
last_modified_at: 2026-10-26
date: 2026-07-23
tags: [MoE, Architecture, Inference]
description: "从 2025-2026 年的研究型架构工作出发，讨论 MoE 之后的稀疏计算轴：depth、attention、memory、跨层复用、专家执行和推理部署。"
---


## 0. 导读

前几篇把 MoE 作为 FFN 的条件计算层：每个 token 选一小组专家，以有限 active compute 使用更大的参数池。这一篇扩大讨论范围，看看深度、attention、memory 和跨层参数是否也能按需组织。

这些工作处在不同层级。MoBA 改 attention 的访问范围，Engram 增加 lookup memory，跨层共享改变参数归属，serving 调度则可能完全不改模型。下面按它们改变的资源与执行方式比较，避免把所有方法都当作同一种 MoE 变体。

## 1. 从计算对象到参数和部署组织

传统 MoE 的稀疏轴很明确：每个 token 从一组 FFN experts 里选 top-k。这个设计主要解决“总参数容量”和“每 token 计算量”之间的矛盾。

可以从下面几个问题进入：

| 资源或执行问题 | 选择 / 复用的对象 | 代表工作 |
|---|---|---|
| depth / iteration | 每个 token 是否需要同样深的计算，深层信息如何保留 | Mixture-of-Depths、MoDA、LoopMoE |
| layer / expert sharing | 每层是否必须维护独立 experts，routing path 能否跨层传递 | MoEUT、ReXMoE、MoRE、GMoE、UniPool、Expert Tying |
| attention context | 每个 query 是否需要看完整历史上下文 | MoBA |
| attention head / scheme | 每个 token 是否需要所有 heads，或同一种 attention 形式 | MoH、MoAS、Attention Sink as native MoE |
| memory lookup | 静态知识是否应该靠计算重建 | PEER、STEM、Engram / Conditional Memory |
| expert routing / execution | 选中 expert 后是否一定要全量执行，experts 是否必须同宽同构 | SoftMoE、MoDSE、HMoE、MoHGE、MoSE、DAG-MoE |
| deployment modularity | 是否必须加载完整 MoE，能不能按域抽取、更新或转成 dense | EMO、BAR、MoE-to-Dense |
| serving-time reuse | batch、speculative decoding、parallel block 内能否共享 expert working set | XShare、DES、EcoSpec |

表格用于定位下文，不表示这些方法可以互相替换。尤其要区分减少执行量、共享唯一参数和改变参数存储位置：三者节省的成本不同。

## 2. 深度：跳过、读取历史，还是复用计算

[Mixture-of-Depths][mod] 是这条线的起点。它让每层只处理一部分 token，其它 token 通过 residual 直通。这里不扩参数容量，真正被改掉的是 Transformer 里“每个 token 每层都付同样计算”的默认假设。

2026 年的 [MoDA][moda] 往前走了一步：把 depth stream 也做成 attention 可读的 KV。当前层的 query 可以同时 attend 当前序列 KV 和之前层留下来的 depth KV。换句话说，过去层的中间表示不用全挤进 residual hidden state，也可以像上下文一样被检索。

![MoDA depth attention](/assets/img/blog/moe-we-like-you/08-moda-depth-attention.png)

*图 1：MoDA 把 depth stream 也做成可检索的 KV。图源：Mixture-of-Depths Attention, Fig. 3。*

这里处理的是现代 Transformer 的老问题：深度增加后，信息不一定按比例变好，反而可能被 residual stream 稀释。MoDA 给模型加了一条跨层读历史的通道。正文实验里，MoDA 在 700M 和 1.5B 语言模型上都做了 400B token 训练；默认变体复用 attention-side depth KV，再给 FFN-side depth KV 加投影，报告了比 OLMo2 baseline 更低的 PPL 和更高的下游平均分。作者还做了硬件实现，说明 fused depth/sequence attention 不能只停在结构图上。

[LoopMoE][loopmoe] 走的是另一条 depth 路线：少堆唯一层，把一个 shared loop body 执行多轮。直觉上，它把“有效深度”从物理层数里拿出来，变成迭代计算。循环本身还不够，LoopMoE 还配了 IterAdaLN，让同一组共享权重在不同 iteration、不同 token 上表现不同；同时调整 attention 与 FFN 的 active 参数比例，避免循环 MoE 变成 FFN-heavy 或 attention-heavy 的畸形结构。

这里有三种不同动作：Mixture-of-Depths 选择哪些 token 继续计算，MoDA 读取历史层表示，LoopMoE 复用 block 计算多轮。后两者不自动意味着计算更少，应分别比较质量、唯一参数量和执行成本。

## 3. 参数复用：专家是否必须归单层所有

标准 MoE 为每个 MoE layer 保存一套专家。增加层数也会增加唯一专家参数；跨层复用试图把这两个预算分开。重点比较两种路径：开放邻层的候选专家，或将相邻层乃至全模型的专家组织成共享池。共享也有两种预算口径：减少唯一权重，或保留唯一专家预算、扩大每层的候选范围。

[ReXMoE][rexmoe] 保留每层 experts，但允许第 `l` 层 router 访问相邻 `r` 层的 experts。候选池从 `N` 变成 `rN`，不必为了扩大候选数而缩小 expert width。它与 fine-grained 的区别在于：前者借用已经存在的参数，后者把参数切成更小的专家。

ReXMoE 的消融显示，单纯扩大候选池的提升有限，配合 Progressive Scaling Routing 才获得更明显收益。PSR 在训练初期使用较小候选池，再逐步放开跨层复用。reuse group 变大还可能增加欠利用专家和负载偏斜，vLLM 实验中的 prefill 也会变慢。因此候选池大小同时是模型与系统变量。

[MoRE（Mixture of Reused Experts）][more] 将相邻若干层组成 reuse block，共享 expert FFN 权重，保留各层独立的 attention 与 router。默认 reuse factor `R=2`：原来两层各有 `N` 个 experts，现在两层都从同一个 `2N` 专家池中选择 top-k，唯一专家总数仍为 `2N`。它扩展了每层可访问的参数范围，也允许相邻层连续调用同一个 expert；在这一预算口径下，专家权重占用没有减半。

共享专家如何知道自己正在处理哪一层？MoRE 在输入上加可学习的 depth embedding，令 $x'_\ell=x_\ell+d_\ell$，再将它同时传给该层 router 与共享 experts。每层选择不同专家的自由度由独立 router 保留，depth embedding 则给共享计算提供层号信息。router 的输出维度扩大与新增 depth embedding 仍需计入总参数。

以论文的 `N=32, k=4` 为例，每层候选组合从 $\binom{32}{4}=35{,}960$ 增至 $\binom{64}{4}=635{,}376$；每次仍执行 4 个 experts。增加的是路由组合，expert matmul 预算基本不变，但更宽的 router 仍有额外开销。论文在 114M–1.15B 规模、匹配参数与计算预算的实验中报告更低 PPL 和更好的下游表现。[官方代码][more-code]通过权重绑定实现共享。相邻层共用一块权重是否带来 cache 复用，则还取决于执行布局与驻留容量，不能从参数共享直接推出延迟收益。

[UniPool][unipool] 让所有层共享 global expert pool，每层只保留独立 router。论文先将若干生产模型的深层路由随机化，观察到质量下降较小，作为深层专家冗余的诊断；这个结果仍受被测模型、层范围和任务集合限制。

共享池需要重新定义负载均衡。某层不使用一个 expert，不代表这个 expert 在全模型中没有工作。UniPool 在 pool 级别统计使用量，并用 NormRouter 处理层间 logit scale 差异。在 182M–978M 模型、30B Pile tokens 的实验中，reduced-pool 版本以常规 MoE 41.6%–66.7% 的 expert 参数预算达到相近或更好的结果；这些结果对应上述模型、数据量和参数预算。

还有几种共享粒度可作为对照：

| 方法 | 共享什么 | 还保留什么差异 |
|---|---|---|
| [MoEUT][moeut] | shared-layer Transformer 中的计算模块 | layer grouping、归一化和各次执行的状态 |
| [GMoE][gmoe] | 跨层 Global Experts，并传播 routing 状态 | 每层 Local Expert 与 routing head |
| [Expert Tying][expert-tying] | 连续若干层的 expert FFN 权重 | 每层 attention、router 和 norm |

![GMoE global experts](/assets/img/blog/moe-we-like-you/08-gmoe-global-experts.png)
*图 2：GMoE 共享 Global Experts，保留每层 Local Expert，并在相邻层传播 routing logits。图源：GMoE, Fig. 1。*

这些方法改变专家共享的范围。评估时需要同时记录唯一参数、pool-level 使用量、rank 负载与 prefill/decode 耗时，确认少存的权重没有换来更重的候选检索或调度。

## 4. Attention：选择上下文、头与计算形式

MoE 通常绕开 attention，只替换 FFN。但长上下文时代，attention 本身也越来越像需要 routing。

[MoBA][moba] 把历史 KV 按 block 切开，每个 query 只选择 top-k 个相关 block 做 attention。它像是把 MoE 的 top-k gating 从 expert 维度搬到了 context-block 维度。和固定 sliding window / attention sink 不同，MoBA 的 block 选择是 query-dependent；同时当前 block 强制保留，保证 causal 语义和局部上下文。

![MoBA block attention](/assets/img/blog/moe-we-like-you/08-moba-block-attention.png)


*图 3：MoBA 将 MoE 式 top-k routing 从 expert 维度搬到 context block 维度。图源：MoBA, Fig. 1。*

MoBA 把“长上下文注意力”从静态稀疏模式改成动态检索模式。论文在多种上下文长度下讨论了 scaling、hybrid full attention 和效率；一个实用结论是，MoBA 可以先用 sparse prefill 降成本，再保留少量 full attention 层或阶段，缓解 SFT 里 prompt token 不参与 loss 带来的梯度问题。

[MoH][moh] 则把 attention heads 当成 experts。它让每个 token 只激活一部分 routed heads，同时保留一些 shared heads 承担公共能力。大量工作都观察到 attention head 有冗余，但传统 pruning 是静态删 head；MoH 改成 token-level head routing。正文里还展示了一个有用信号：预训练好的 LLaMA3-8B 可以继续调成 MoH-LLaMA3-8B，不一定要从零训练。

[MoAS][moas] 更偏概念验证：它把 MHA、GQA、MQA 当成三种 attention schemes，让 router 在三者之间加权。实验规模很小，不能当强证据，但它把 attention routing 的对象从 head 推到了 KV sharing 形态。对推理来说，这个维度重要，因为 MHA/GQA/MQA 的 KV cache 成本差异很大。

另一条 2026 线索是 [Attention Sink Forges Native MoE in Attention Layers][attention-sink]。它把 Vanilla Attention、Sink Attention、Gated Attention 放在一起分析，认为 attention sink 或 gating factor 会在 head 维度上形成隐式 MoE；一些 heads 长期贡献生成，另一些 heads 接近 inactive，类似 MoE 里的 expert collapse。论文提出 sink-aware load balance loss 来缓解 head collapse。由此可以增加一个诊断维度：attention 里也可能出现 MoE 式的竞争和塌缩。

## 5. Memory：用检索补充参数计算

[PEER][peer] 是这条线的早期坐标。它把 experts 细到 single-neuron MLP，再用 product-key retrieval 从百万级专家里检索。传统 MoE 是“几百个较大的专家”；PEER 更像“一个可检索的参数库”。核心问题也随之改变：如果专家数从几百扩大到百万，router 不能再全量打分，必须用检索结构。

[STEM][stem] 把这条线推进到 FFN 内部。它不外挂 memory module，而是把 gated FFN 的 up-projection 换成 layer-local embedding table：token id 查出一条向量，再和 dense gate projection 相乘，down-projection 保持 dense。这样没有 runtime router，也没有专家负载均衡；embedding table 可以放在 CPU 并异步 prefetch。代价是 token-indexed lookup 缺少上下文自适应，所以论文只替换 up-projection，保留 gate/down 来做上下文混合。实验主要在 350M/1B，适合在这个规模内比较 lookup 与 dense projection 的取舍。

2026 年的 [Engram / Conditional Memory][engram] 把这个问题说得更直接：MoE 是 conditional computation，而语言里还有很多 local、static、formulaic pattern，更适合 conditional memory。Engram 用 hashed N-gram lookup 检索静态 embedding，再用 hidden state 做 gating 和融合。这里新增的是一个和 MoE 并列的稀疏轴：一部分 inactive parameter budget 给 routed experts，一部分给 memory slots。

![Engram conditional memory](/assets/img/blog/moe-we-like-you/08-engram-conditional-memory.png)


*图 4：Engram 用 hashed N-gram lookup 提供 conditional memory，与 Attention/MoE 计算路径并列。图源：Conditional Memory, Fig. 1。*

Engram 进一步比较 routed experts 与 memory slots 如何分配参数预算。正文里有一个 U-shaped allocation：在固定 total params 和 active params 下，纯 MoE 不是最优，把约 20%-25% sparse budget 分给 Engram 反而更好。大规模实验里，Engram-27B 是从 MoE-27B 里减少 routed experts，再把参数 realloc 到 5.7B memory module；作者报告它在知识、推理、代码数学和长上下文上都优于 iso-parameter / iso-FLOPs MoE baseline。

这条线的系统含义也不同于 MoE。MoE 的专家选择依赖 hidden state，推理时很难提前知道要读哪些 expert；STEM 和 Engram 的索引更静态，分别来自 token id 和 N-gram，可以预取，可以放到 host memory，甚至做多级 cache。它们把一部分“参数容量”从 GPU HBM 里解耦出来。

## 6. Expert 执行：粒度、宽度与组合

标准 MoE 有一个隐藏假设：router 选中 top-k experts 后，每个 expert 都全量执行，最后按 router weights 加权求和。2025-2026 的一些工作开始动这个假设。

这里引用的 2026 年 [SoftMoE][softmoe] 调整 router 本身。它用可微的 soft top-k relaxation 替代硬 top-k，并在全模型范围内学习每层平均激活多少 expert。固定 `k` 在这里变成可学习的跨层 budget；论文观察到后层往往会分到更多 active experts。评估时除了平均激活量，还需检查从训练松弛到实际推理执行的差异。

[MoDSE][modse] / [HMoE][hmoe] 动的是另一个默认假设：所有 experts 必须同宽同构。普通 MoE 里，每个 expert 的 FFN hidden size 一样，router 只决定“去哪一个 expert”；异构 MoE 则让不同 experts 有不同参数量，希望简单 token 多走小 expert，难 token 再付更大的 expert cost。MoDSE 用不同大小的 expert pairs 做初步验证，HMoE 更系统地比较 expert size distribution，并提出 P-Penalty loss，避免 router 过度偏向大 expert。这里改的是 expert pool 内部的容量分配：从均匀切块，变成一组不同尺寸的工具。

这个方向马上会碰到系统问题：token 数均衡不等于计算均衡。一个大 expert 收到一个 token，和一个小 expert 收到一个 token，GEMM cost、显存占用和 rank tail 都不同。[MoHGE][mohge] 的处理方式是把 experts 按 size 分组，先选 group、再在 group 内选 expert；同时用 All-size Group-decoupling Allocation 把不同 size 的 experts 成组分到 GPU 上，再用 intra-group auxiliary loss 维持组内均衡。论文报告 MoHGE 在接近普通 MoE 质量的同时减少约 20% total parameters，并保持更平衡的 GPU utilization。对异构专家，应以 per-rank compute/time 统计检查均衡，而非仅比较 token 数。

[MoSE][mose] 关注的是 expert 内部宽度。它认为标准 MoE 只决定“选哪个 expert”，但一旦选中就完整执行，这会让 accuracy-compute trade-off 很离散。MoSE 把每个 expert 做成 slimmable FFN：中间维度按 nested width 切片，同一个 expert 可以用 25%、50%、75%、100% 等不同宽度执行。这样一来，router 不只选择 expert，还能按 token / budget 决定每个 active expert 用多宽。

![MoSE slimmable expert](/assets/img/blog/moe-we-like-you/08-mose-slimmable-expert.png)


*图 5：MoSE 不只选择 expert，也调整被选中 expert 的执行宽度。图源：MoSE, Fig. 1。*

这和端侧/serving 直接相关。对于 easy token，选中 expert 但只用窄宽度可能足够；对于 hard token，再给更多宽度。正文里 MoSE 做了 from-scratch 训练，也做了对已有 DeepSeek MoE checkpoint 的 continual-pretraining adaptation；还用轻量 test-time training 学 router confidence 到 width 的映射。它给 MoE 加的是一个连续 compute knob。

[DAG-MoE][dag-moe] 动的是另一个假设：专家输出不一定只能加权求和。标准 weighted sum 是 permutation-invariant 的，top-k experts 之间没有顺序和交互。DAG-MoE 把选中的 experts 组织成一个 DAG，用轻量 DAG learning module 学它们之间的结构聚合。它的目标是在不增加 router-side complexity 的情况下，扩大 expert combination space，并让一个 MoE layer 内部近似多步组合计算。

![DAG-MoE structural aggregation](/assets/img/blog/moe-we-like-you/08-dag-moe-structural-aggregation.png)


*图 6：DAG-MoE 将标准 MoE 的加权求和改成结构化聚合。图源：DAG-MoE, Fig. 1。*

DAG learning module 也增加实现开销。若比较结构化聚合与 weighted sum，需要在相近参数和计算预算下，分别报告质量与执行时间。

这一组工作都在追问 top-k 之后还能不能继续调：router budget、expert size、expert width、expert 之间的组合结构，都可以变成设计空间。

## 7. 扩散式语言生成：分配去噪过程中的计算

[LLaDA-MoE v2][llada-moe-v2] 把 MoE 放进 diffusion language model。这个工作有意思的地方不在 30B total / 3B active 的规模本身，而在 diffusion LM 的训练/推理语义：masked denoising 会并行预测多个位置，和 AR LLM 的左到右 next-token prediction 不同。这样一来，batch size、learning rate、model-data allocation、activation ratio、expert granularity、shared-expert ratio，都不能直接套 AR MoE 的经验。

正文里，LLaDA-MoE v2 做了 MoE dLLM scaling law：作者发现高 compute 下 batch size 和 learning rate 的缩放趋势需要 dLLM-specific calibration；architecture 上采用 30B total / 3B active，128 routed experts、top-8、一个 shared expert，并训练了 23.5T tokens。这组实验将 MoE 容量扩展与 masked denoising 的训练预算联系起来。

[REFLEX][reflex] 则看 MoE dLLM 的推理分配问题。扩散式生成每个 denoising forward 会反复访问多个 token state，不同 token 距离 commitment 的状态不同，默认固定 top-k 可能把同样 expert budget 浪费在不同 refinement demand 的 token 上。REFLEX 不改 router，而是在推理时按 refinement role 和 confidence progress 重新分配 expert count。论文报告在 LLaDA-MoE 和 LLaDA2.0-mini 上平均减少约 15% selected expert-token pairs，同时多数 benchmark 质量不降或提升。

![REFLEX refinement-aware allocation](/assets/img/blog/moe-we-like-you/08-reflex-refinement-aware-allocation.png)


*图 7：REFLEX 在扩散式语言模型推理中按 refinement state 重新分配 expert budget。图源：REFLEX, Fig. 1。*

到这里，MoE inference 的问题已经不只在 top-k kernel。生成范式从 AR 变成 iterative refinement 后，router budget 的组织单位也会改变。

## 8. 部署：减少驻留、更新与搬运

还有一组工作不直接改变 Transformer block，但会改变 MoE 怎么被训练、部署和压缩。

[EMO][emo] 问的是：标准 MoE 的 expert 能不能按任务/领域拆出来单独用？EMO 作者将常规 MoE 子集部署时的质量损失与专家分工联系起来：一个文档的 token 可能访问分散的、承担低层语言功能的专家，领域子集因此不容易直接抽取。EMO 在预训练时让同一文档的 token 先共享一个 document-level expert pool，再在池内做 routing；这样专家子集更容易按语义/领域聚在一起。正文里 14B total / 1B active 的 EMO 在完整模型质量上接近普通 MoE，同时在只保留 25% 或 12.5% experts 时，domain subset 的掉点远小于普通 MoE。

![EMO document-level expert pool](/assets/img/blog/moe-we-like-you/08-emo-document-level-expert-pool.png)


*图 8：EMO 把同一文档的 tokens 约束在共享 expert pool 内路由，让 experts 更容易形成领域级模块。图源：EMO, Fig. 2。*

[BAR][bar] 把 MoE 当成 post-training 的组合机制，而不是预训练里的稀疏容量层。它先为 math、code、tool use、safety 等领域独立训练 domain experts，每个 expert 可以走自己的 mid-training / SFT / RL 流水线，最后合成一个 MoE 并只训练轻量 router。这个方向的价值在于工程组织：新增或升级一个能力时，不必把所有领域重新混在一起跑完整后训练。

![BAR modular post-training](/assets/img/blog/moe-we-like-you/08-bar-modular-post-training.png)


*图 9：BAR 将不同领域的后训练流水线拆成独立 experts，再合并成单一 MoE，并支持单 expert 升级。图源：BAR, Fig. 1。*

[MoE-to-Dense][moe-to-dense] 则走相反方向：如果部署端根本放不下完整 expert pool，那就把已经训练好的 MoE 转成 dense。它先给 experts 打分、选择和分组，再把专家权重拼成 dense FFN，最后用 MoE teacher 做 distillation。这个方向不替代 MoE 训练，但可能替代一部分 MoE serving：训练时用 MoE 拿容量，交付时导出更小的 dense 学生模型。

Serving 侧还有更系统的 expert working set 复用。[XShare][xshare] 把 batch 内 token 的专家选择看成 batch-level 优化，目标是减少一个 batch 激活的 expert union；[DES][des] 把 diffusion LLM 的 parallel block 看成 sequence-level coreset selection，减少并行生成时的 expert explosion；[EcoSpec][ecospec] 则在 speculative decoding 选 draft token 时加入 marginal expert cost，让验证集合尽量复用已有 experts。这些工作不改模型权重，但会影响 MoE 在真实服务中的收益，因为 decode 阶段很多时候卡在 expert weight movement，而非 FLOPs。

![XShare batch-level expert pruning](/assets/img/blog/moe-we-like-you/08-xshare-batch-level-expert-pruning.png)


*图 10：XShare 先按 batch 聚合 expert utility，再限制每个 token 在共享专家池内选择 top-k，减少 batch 级 expert union。图源：XShare, Fig. 2。*

## 9. 这些方向与 MoE 怎样组合

前面的工作分别改变执行深度、attention 访问、参数复用、memory lookup 或部署方式，因此“替代 MoE”没有统一的比较对象。若问题是固定 active compute 下扩大 FFN 参数容量，需要比较专家或 lookup 结构；若问题是长上下文 attention 成本，MoE 本身就不直接提供答案。

更有用的组合实验是先固定预算，再改变一种资源组织方式。例如减少部分 routed experts，将参数预算分给 memory；或共享相邻层专家，观察节省的唯一权重是否抵得上 routing 和负载控制成本。这样才能区分结构互补、功能替换和纯部署优化。

## 10. 小结：比较节省的是哪一种资源

这些方法改变了不同的约束。MoBA 减少 attention 读取的上下文；跨层共享可用于压缩唯一专家参数，也可像 MoRE 一样在固定专家预算下扩大候选池，同时把负载统计从单层扩展到共享池；STEM / Engram 的 token 或 N-gram 索引可以提前确定，使部分参数能够预取。相同的“稀疏”标签，背后对应不同的存储、访问与计算方式。

组合它们时，先明确要减少哪种成本，再统一模型、数据与执行预算。例如，shared pool 的价值不仅是少存权重，还包括是否能控制跨层叠加的流量；lookup memory 的价值也取决于索引提前可知之后，预取和融合能否跟上计算。具体机制与各论文的实验范围，决定了哪些组合值得继续验证。

## 系列导航

上一篇：[第七篇：RL 训推不一致与路由对齐问题]({% post_url 2026-07-31-moe-rl-train-serving-mismatch %})

下一篇：[第九篇：DeepEP v1/v2、MoonEP 与 EP 通信优化]({% post_url 2026-07-23-moe-ep-communication-optimization %})

## 延伸阅读

[Lilian Weng 的大模型训练笔记][lilian-train-large]适合补分布式训练和 MoE 的历史背景，尤其是把 expert parallelism 放回 data / tensor / pipeline parallel 的大图里看。

实现 attention 稀疏化时，可以结合 [MoBA GitHub][moba-github]。README 里把 block sparse attention、query-to-block routing 和代码入口放在一起，比只读论文 abstract 更容易定位实现边界。

## 参考资料

- [MoRE: Mixture of Reused Experts][more]；[官方代码][more-code]
- [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)
- [MoEUT: Mixture-of-Experts Universal Transformers](https://arxiv.org/abs/2405.16039)
- [Mixture of A Million Experts](https://arxiv.org/abs/2407.04153)
- [MoH: Multi-Head Attention as Mixture-of-Head Attention](https://arxiv.org/abs/2410.11842)
- [Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models](https://arxiv.org/abs/2411.04996)
- [MoBA: Mixture of Block Attention for Long-Context LLMs](https://arxiv.org/abs/2502.13189)
- [HMoE: Heterogeneous Mixture of Experts for Language Modeling](https://arxiv.org/abs/2408.10681)
- [Mixture of Diverse Size Experts](https://arxiv.org/abs/2409.12210)
- [ReXMoE: Reusing Experts with Minimal Overhead in Mixture-of-Experts](https://arxiv.org/abs/2510.17483)
- [Mixture of Attention Schemes: Learning to Route Between MHA, GQA, and MQA](https://arxiv.org/abs/2512.20650)
- [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)
- [STEM: Scaling Transformers with Embedding Modules](https://arxiv.org/abs/2601.10639)
- [Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs](https://arxiv.org/abs/2602.00879)
- [Attention Sink Forges Native MoE in Attention Layers](https://arxiv.org/abs/2602.01203)
- [MoSE: Mixture of Slimmable Experts for Efficient and Adaptive Language Models](https://arxiv.org/abs/2602.06154)
- [XShare: Collaborative in-Batch Expert Sharing for Faster MoE Inference](https://arxiv.org/abs/2602.07265)
- [Mixture-of-Depths Attention](https://arxiv.org/abs/2603.15619)
- [Train Separately, Merge Together: Modular Post-Training with Mixture-of-Experts](https://arxiv.org/abs/2604.18473)
- [MoHGE: Mixture of Heterogeneous Grouped Experts for Scalable and Efficient Language Modeling](https://arxiv.org/abs/2604.23108)
- [EMO: Pretraining Mixture of Experts for Emergent Modularity](https://arxiv.org/abs/2605.06663)
- [UniPool: A Globally Shared Expert Pool for Mixture-of-Experts](https://arxiv.org/abs/2605.06665)
- [Pruning and Distilling Mixture-of-Experts into Dense Language Models](https://arxiv.org/abs/2605.28207)
- [DAG-MoE: From Simple Mixture to Structural Aggregation in Mixture-of-Experts](https://arxiv.org/abs/2606.01062)
- [LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling](https://arxiv.org/abs/2606.04438)
- [Tying the Loop: Tied Expert Layers in Mixture-of-Experts Language Models](https://arxiv.org/abs/2606.16825)
- [SoftMoE: Soft Differentiable Routing for Mixture-of-Experts in LLMs](https://arxiv.org/abs/2606.17952)
- [GMoE: Global Mixture of Experts with Logit Propagation](https://aclanthology.org/2026.acl-long.2065/)
- [Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts](https://arxiv.org/abs/2607.12696)
- [REFLEX: Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models](https://arxiv.org/abs/2608.01784)
- [LLaDA MoE v2: Scaling Mixture-of-Experts Diffusion Language Models](https://arxiv.org/abs/2608.03457)

[mod]: https://arxiv.org/abs/2404.02258
[lilian-train-large]: https://lilianweng.github.io/posts/2021-09-25-train-large/
[moba-github]: https://github.com/MoonshotAI/MoBA
[moeut]: https://arxiv.org/abs/2405.16039
[peer]: https://arxiv.org/abs/2407.04153
[moh]: https://arxiv.org/abs/2410.11842
[hmoe]: https://arxiv.org/abs/2408.10681
[modse]: https://arxiv.org/abs/2409.12210
[moba]: https://arxiv.org/abs/2502.13189
[rexmoe]: https://arxiv.org/abs/2510.17483
[moas]: https://arxiv.org/abs/2512.20650
[engram]: https://arxiv.org/abs/2601.07372
[stem]: https://arxiv.org/abs/2601.10639
[des]: https://arxiv.org/abs/2602.00879
[attention-sink]: https://arxiv.org/abs/2602.01203
[mose]: https://arxiv.org/abs/2602.06154
[xshare]: https://arxiv.org/abs/2602.07265
[moda]: https://arxiv.org/abs/2603.15619
[bar]: https://arxiv.org/abs/2604.18473
[mohge]: https://arxiv.org/abs/2604.23108
[emo]: https://arxiv.org/abs/2605.06663
[unipool]: https://arxiv.org/abs/2605.06665
[moe-to-dense]: https://arxiv.org/abs/2605.28207
[dag-moe]: https://arxiv.org/abs/2606.01062
[loopmoe]: https://arxiv.org/abs/2606.04438
[expert-tying]: https://arxiv.org/abs/2606.16825
[softmoe]: https://arxiv.org/abs/2606.17952
[gmoe]: https://aclanthology.org/2026.acl-long.2065/
[ecospec]: https://arxiv.org/abs/2607.12696
[reflex]: https://arxiv.org/abs/2608.01784
[llada-moe-v2]: https://arxiv.org/abs/2608.03457

[more]: https://arxiv.org/abs/2609.18176
[more-code]: https://github.com/EricQiu6/MoRE
