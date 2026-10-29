---
layout: post
title: "长话长说（五）：从参数知识到内部记忆"
date: 2026-09-04
last_modified_at: 2026-10-28
tags: [LLM, Long Context, Memory, Attention, Agent]
description: "从 FFN 参数知识出发，比较 latent pool、在线 fast weights 与跨交互状态；结合 ProTTT、Metis/MemT-1.0、REMORY，讨论记忆怎样写入、稳定保留并服务后续任务。"
---

## 写在前面

> 这是长上下文 attention 架构系列的第五篇。前面几篇分别看了 [attention 与 KV cache 总览]({% post_url 2026-07-27-long-context-attention-architecture-survey %})、[linear attention]({% post_url 2026-07-29-linear-attention-from-kv-cache-to-recurrent-state %})、[近期长上下文 attention 趋势]({% post_url 2026-08-04-long-context-architecture-watchlist-2026 %})，以及 [超长上下文训练 infra]({% post_url 2026-08-18-long-context-large-scale-llm-training-optimization %})。 完整目录见[系列导航]({% post_url 2026-07-27-long-context-attention-architecture-survey %}#series-map)。

设一个助手先收到“项目使用 Python 3.11”，几轮之后又收到“部署环境已升级到 3.12”。两条消息都在窗口里时，它可以重新阅读并判断哪条有效；原文被压缩或淘汰后，答案取决于先前留下了什么，以及更新是否覆盖了旧值。再要求它忘记一条信息，问题又从“能不能找回”变成了“状态里是否还留有可恢复的痕迹”。

这几种行为不能用一个 context length 概括。长窗口保存更多原文；参数知识提供训练中形成的关联；latent memory 和 fast weights 则尝试让交互在模型内部留下可持续使用的表示。它们的容量、更新对象和重置边界各不相同。

本文从 FFN 的参数记忆解释出发，逐步讨论压缩表示、可检索的 latent pool、在线状态更新，以及跨交互的 memory procedure。近期几篇工作把其中的取舍变得更具体：ProTTT 说明外层目标会影响在线更新学到什么；MemT-1.0 分析擦除算子怎样同时影响稳定性与保留；REMORY 则让连续记忆补充已经存在的文字摘要。Memorilla 与选择性写入的受控实验作为对照，用于区分读出预算、总容量和漏写风险。

讨论依据截至 2026-10-27 的公开材料。下文的 Metis 指 **Metis: Memory Foundation Model**；后续 MemT-1.0 的机制单独说明，不混用两代模型的公式。贯穿全文的问题是：历史经过了哪些变换，后续任务还能从中得到什么？

![LLM Memory 相关工作的发展脉络与关键接口](/assets/img/blog/long-context-internal-memory/memory_research_threads_overview.png)

*图：Memory 研究的并行脉络。各列按保存对象和读写接口分组，图中仅标出部分代表工作；固定读出预算、固定状态容量与跨交互持久化是不同约束，具体机制和实验范围见正文。*

## 1. 先从 LLM 参数里的知识和记忆说起

模型能回答一个事实，可能因为它从参数中恢复了关联，也可能因为证据就在当前输入中。这两条路径都能产生正确答案，却对应不同的写入与更新过程。

### 1.1 LM 能不能直接当知识库

LAMA / *Language Models as Knowledge Bases?* 用 cloze query 测试模型能否补出事实；*How Much Knowledge Can You Pack Into the Parameters of a Language Model?* 继续用 closed-book QA 检查参数里的知识。二者建立了一个经验前提：不给外部证据，模型也能答出部分关系和事实。但这些知识通常随 checkpoint 固定，更新、定位和追溯来源需要另设方法。

### 1.2 FFN 像 key-value memory

Geva 等人的 *Transformer Feed-Forward Layers Are Key-Value Memories* 把问题推进到架构层。采用列向量记法，设输入 $$x\in\mathbb R^d$$，FFN 中间宽度为 $$m$$，两层计算为：

$$
\operatorname{FFN}(x)=W_2f(W_1x+b_1)+b_2,\qquad
W_1\in\mathbb R^{m\times d},\quad W_2\in\mathbb R^{d\times m}.
$$

$$f$$ 是逐元素激活函数，$$b_1\in\mathbb R^m$$、$$b_2\in\mathbb R^d$$ 是偏置，无偏置实现可置零。把 $$W_1$$ 第 $$i$$ 行记作 $$k_i^\top$$，$$W_2$$ 第 $$i$$ 列记作 $$v_i$$，展开矩阵乘法：

$$
a_i=f(k_i^\top x+b_{1,i}),\qquad
\operatorname{FFN}(x)=\sum_{i=1}^m a_iv_i+b_2.
$$

输入 $$x$$ 相当于 query，$$k_i$$ 检测某种输入模式，激活 $$a_i$$ 决定对应 $$v_i$$ 对输出的贡献。例如 $$a=(0,1.2,0.3)$$ 时，忽略偏置，输出就是 $$1.2v_2+0.3v_3$$。内积、激活和求和都是显式计算；训练学到的是 keys、values 和偏置，普通推理时这些参数固定。

![Transformer FFN 作为 key-value memory 的原论文示意图](/assets/img/blog/long-context-internal-memory/ffn_key_value_memories_fig1_pipeline.png)

*图源：Geva et al., Transformer Feed-Forward Layers Are Key-Value Memories, Figure 1。第一层匹配 keys，第二层组合 values，形成写回 residual stream 的输出。*

这里的 value 是影响后续预测的向量方向，不能直接当成一段文本或一条完整事实；同一事实也可能依赖多个神经元和多层计算。系数 $$a_i$$ 不经过 softmax，不要求和为 1，是否允许负值取决于激活函数。SwiGLU 同样可展开为 down-projection 列向量的加权和，只是系数来自两个输入投影分支的逐元素乘积。后面 §6.4 的 TTCD 就进一步让这部分 value 参数在读取上下文时在线变化。

Product Key Memory 和 Memory Layers at Scale 显式扩大 key-value 参数容量，用稀疏访问控制计算开销。[MoME](https://arxiv.org/abs/2609.15126) 则为同一个 token 配置多个 slots，由当前 hidden state 路由，以区分不同语义。MoME 的受控实验主要在 sub-billion 规模。这些表项在普通推理时仍是静态参数；随输入变化的是激活或路由，而非部署后新写入的事实状态。

### 1.3 从定位知识到编辑知识

Knowledge Neurons 用 attribution 定位与事实表达有关的神经元；ROME 通过 causal tracing 定位关联，再用 rank-one update 编辑；MEMIT 扩展到批量事实，MEND 则训练一个从梯度生成更新的 editor。它们回答“怎样改参数里的关联”。后文的动态 memory 另有一个要求：不断到来的交互应能更新一份状态，并明确该状态属于哪个用户、序列或会话。

### 1.4 参数记忆之外：非参数 memory 为什么一直存在

kNN-LM 检索近邻 token 分布，REALM 和 RAG 将检索接入语言建模，RETRO 用更大的文本库，Memorizing Transformer 则读取过去的 key-value pairs。外部寻址让新增、删除和保留原始证据更直接，但它也引入检索、存储和上下文组装。后面的内部状态可以补充这条路径，是否值得补充要由相同任务与预算下的结果判断。

参数知识、外部记录和动态状态因此应分别追踪：谁写入，什么能被修改，后续通过什么接口读取。FFN 的展开提供了共同的计算直觉；它没有消除这些接口差异。

<a id="2-把-knowledge-和-memory-分开后再给-internal-memory-一个工作定义"></a>

## 2. 定义接口，也定义状态的生命周期

这里采用一个接口定义：**internal memory 是由历史输入形成、能在后续计算中继续使用的内部状态或连续表示。** 保存它的物理位置不是分类依据；关键是后续模型怎样寻址、读取和更新。

先把外部记忆作为对照，按表示形态列出几类接口。它们并不互斥：memory tokens 可以组成长期 pool，fast weights 也可以只是单次序列的临时状态。

| 形态 | 历史怎么保存 | 后续怎么读取 | 代表工作 | 关键问题 |
|---|---|---|---|---|
| textual / external memory | 文本、向量库、数据库、工具日志 | 检索后拼回 prompt | RAG、MemoryBank、MemGPT、Mem0 | 可解释、可控，但流程在模型外部 |
| compressed context / memory tokens | summary vectors、memory slots、beacon activations | 作为 soft prompt 或压缩 KV 参与 attention | RMT、AutoCompressor、ICAE、Activation Beacon | 解决长文压缩，但通常还是单次上下文管理 |
| latent memory pool | per-layer hidden states、latent prefix、可更新 memory pool | 每层 attention 读 latent memory | MemoryLLM、M+、NextMem、MemDefrag | 压缩密度高，但有位置错位、干扰和容量问题 |
| recurrent / fast-weight memory | 固定形状 state 或动态参数 | query 读 state，state 随输入更新 | Titans、LiveMem、TTCD、ProTTT、Metis/MemT-1.0 | 容量与更新成本受配置限制，需学会保留和防串扰 |

表示形态之外，还要单独记录生命周期。一个张量放在 forward 内部，不自动意味着它跨会话持久存在。

| 接口或对象 | 变化的是什么 | 持续范围与重置边界 |
|---|---|---|
| 普通 FFN、静态 memory layer | 离线训练得到的参数；推理时通常固定 | 随 checkpoint 保存，更新需另行训练或编辑 |
| KV、linear/recurrent state | 当前输入产生的 activation/state | 通常按请求或序列管理；跨请求复用依赖系统显式保存与恢复 |
| memory tokens / latent pool | 连续向量或逐层 prefix | 可以服务一篇文档，也可以跨多轮保存，取决于具体协议 |
| TTT / fast weights | 序列或 episode 内更新的参数副本 | 更新对象常可视为状态；需明确是否在新序列开始时重置 |
| Metis-style native state | 固定控制器读写的动态 memory | 面向跨交互延续；状态归属、持久化和删除仍需配套管理 |

后面比较方法时，同时追问表示、寻址、更新规则与持续范围。Recurrent memory 不必是长期记忆，外部保存的 latent 也不因落在模型文件之外就失去内部表示的性质。

把写入、读取与回答分开：

```text
write:   M_{t+1} = U(M_t, x_t)
read:    r_t = R(M_t, x_t)
answer:  y_t ~ p(y | x_t, r_t)
```

读写规则还需要两个条件：训练时什么结果会奖励一次写入，以及状态在什么时候被提交、保存或重置。后文的 ProTTT 改变前者，Metis/MemT-1.0 明确后者。仅从张量形状无法推出一份记忆能否跨会话持续存在。

<a id="3-为什么长上下文不等于记忆"></a>

## 3. 原文离开窗口后，模型还能使用什么

普通 KV cache 可以按请求保留，recurrent state 也可以按请求保存；二者都需要系统恢复协议。它们的差别首先在表示：逐 token KV 留下较多直接可读的历史条目，固定形状 state 则把历史混合进有限表示。一个 1M token 窗口仍会到顶，固定状态则从一开始就要决定如何分配容量。

prefix cache 能免去命中前缀的重复计算，但新增工具观测仍需 prefill；历史读取和状态存储也继续产生开销。外部检索可以只送入部分证据，但召回和使用是两步，保留过的信息未必能在当前问题下被取出。

回到开头的版本更新：保存两条原文、生成“当前用 3.12”的文字摘要、把关联写进 fast weights，都可能支持正确回答。但更新其他项目时是否串扰、原文移除后是否仍可回答、撤销事实后是否还会被恢复，需要分开测试。后文按这些机制比较方法，而不把更长的窗口或更大的状态直接当作更可靠的记忆。

<a id="4-第一条脉络memory-tokens先把长文压成短状态"></a>

## 4. 压缩历史：把原文变成可读取的连续表示

先从最直接的接口开始：把一段历史编码成少量连续向量，后续计算读取这些向量。压缩预算决定能留多少信息，训练目标决定哪些信息值得留。

### 4.1 RMT：把 memory token 从上一段传到下一段

Recurrent Memory Transformer (RMT) 的想法很朴素：把长序列切成 segments，在每个 segment 里放一些特殊 memory tokens。处理完当前 segment 后，把这些 memory tokens 的输出作为下一段的输入。

```text
segment 1 + memory_0 -> Transformer -> memory_1
segment 2 + memory_1 -> Transformer -> memory_2
segment 3 + memory_2 -> Transformer -> memory_3
```

RMT 在段间传递 memory tokens，后续 scaling 工作展示了百万 token 量级的处理长度。传递长度与有效容量仍不同：有限向量经过反复更新后，未必保留未来问题需要的远处细节。

### 4.2 AutoCompressor 与 ICAE：summary vector / memory slot

AutoCompressor 把长文分段处理，每段输出若干 summary vectors，再把这些向量作为后续段落的 soft prompt。它把“摘要”从自然语言搬到连续空间里：模型学习一组能帮助后续 language modeling 的向量，而不生成可读摘要文本。

ICAE 更接近 autoencoder。它用一个 LoRA-adapted LLM encoder 把上下文压成 memory slots，再让原 LLM decoder 在这些 slots 条件下回答问题或恢复文本。论文里常用的说法是 4× context compression：比如 512 token 原文压成 128 个 memory slots。

summary vectors 和 memory slots 都让历史以连续表示进入后续 attention。它们主要验证给定文本被压缩后的可用性；持续交互中的修订、遗忘与状态恢复，需要额外协议和监督。

### 4.3 Activation Beacon：不只压 token，而是压每层 activation

Activation Beacon 往前走了一步。它插入 beacon tokens，让原始 token 的信息在 self-attention 中被压到 beacon tokens 的 activations 里；编码完一个 chunk 后，丢弃原始 token 的 KV，只保留 beacon activations。后续 chunk 和生成阶段都可以读这些压缩后的 activation。

它保留的是每层 K/V activations，后续 chunk 和生成阶段都能读取，压缩结果可逐块累积。因此要分别统计压缩后的条目数、每个条目的逐层开销，以及历史继续增加时总存储是否仍增长。“用更少的位置表示历史”与“整份状态大小固定”是两种约束。

<a id="5-第二条脉络latent-memory-pool把记忆存成-per-layer-hidden-states"></a>

## 5. 存储与读出：容量固定在哪一侧

连续表示还可以组成池：哪些历史留在池中，哪些条目在本次 query 下被读出，可以采用不同策略。

### 5.1 MemoryLLM：固定大小 memory pool 与 self-update

MemoryLLM 把模型参数分成两部分：

$$
\phi: \text{静态 backbone}, \qquad \theta: \text{动态 memory pool}.
$$

其中 $$\theta$$ 是每层的一组 memory tokens / hidden vectors。生成时，当前 token 可以 attend 到这些 memory tokens；遇到新知识时，模型通过 self-update 把新信息写进 memory pool，同时保持 pool 大小固定。

它的写入过程大概可以理解成：

```text
old memory pool + new text -> Transformer layers -> new memory tokens
drop part of old memory -> append new memory tokens -> updated memory pool
```

MemoryLLM 用近百万次 update 检查 operational integrity，关注持续吸收新知识后的运行完整性；这不等于最早写入的事实都能保持同样的可恢复率。

代价也很明显。memory pool 很大，论文主实验里是 1B 量级的 memory 参数；读写仍要处理 per-layer latent prefix；旧知识通过 drop/replace 逐渐淡出，细粒度可控性有限。

### 5.2 M+：latent memory 也需要检索

[M+](https://arxiv.org/html/2502.00592#S3.SS2)把 MemoryLLM 更新时原本丢弃的 memory tokens 存进更大的长期池，并记录年龄。生成时，每层由共同训练的 retriever 选出少量长期 tokens，按年龄排序，与该层短期池拼接，再供当前 hidden states 通过 cross-attention 读取。长期池到上限后淘汰最老条目。

因此，“能存多少”和“本轮读多少”被拆开：大池负责保存，retriever 控制读取预算。论文报告的知识保持范围从 20K 以下推进到 160K 以上，不能理解成每步都把全部长期池送进 attention。

### 5.3 NextMem：先把事实压成可重构 latent memory

NextMem 把 agent 观察到的事实编码成 latent memory，再训练模型从 latent 自回归重构原文。训练先对齐重构能力，再逐步用 latent 替换原文，让后续计算适应用压缩表示读取事实。它与 AutoCompressor / ICAE 的主要差异在任务对象：面向后续 agent 决策需要的事实，而非只压缩当前问答的文档。重构目标提高了对精确细节的要求，但不等于任意事实都能无损恢复。

### 5.4 MemDefrag：latent memory 的问题不只在容量，还在碎片化

MemDefrag 处理已有 latent fragments 的组织，而非重训一个记忆模型。fragment 形成时的位置编码与后来拼接的位置可能错位；全部平铺成前缀，也会混入当前 query 不需要的内容。它利用中间层对各 fragment 的 attention density 排序、重排与过滤，容量超限时再按 informativeness 分配遗忘比例。这里改变的是保留和送入后续计算的 fragments，不是重新编码一份完整历史。

<a id="memorilla-query-dependent-readout"></a>

### 5.5 Memorilla：等到问题出现，再决定怎样压缩读出

[Memorilla](https://arxiv.org/abs/2610.21375v1)（2026-10）将长期保存和当前读出进一步分开。冻结 encoder 为每个文档块生成一次 embedding，组成可增量更新的库；query 到来后，先用 query embedding 调整一组 slots，再让这些 slots 连续 attend 整个库，输出冻结 LLM 输入空间中的 soft tokens。主问答配置使用 16 个 tokens，读取过程没有离散 top-k 截断。

这把部分选择推迟到需求已知时：同一份历史可以为不同 query 生成不同连续前缀，写入时不必把所有潜在用途一次压进相同的小状态。但 16 只是进入 LLM 的读出位置数，文档 embedding 库与全库读取仍随历史增长。它与固定总状态的 native memory 有不同的容量约束。

query-aware、无 query 和 hard top-k 等对照支持查询条件的作用。主实验使用冻结 Qwen3-Embedding-4B、Qwen3-8B 与约 194M 可训练 memory 参数，且包含评测领域的监督训练，两个个性化任务另行微调。它展示了可训练读出接口的价值，不代表对任意新任务免训练。TextWorld 实验使用 8-token 配置，应与主问答的 16-token 结果分开。

<a id="6-第三条脉络recurrent-statefast-weights-与-test-time-training"></a>

## 6. 在线写入：更新规则与学习目标

第二篇推导了 linear attention、Gated DeltaNet 和 KDA。这里关注这些更新怎样形成一张可读写的关联表，以及训练为何会让它保留某类信息。

沿用第二篇的 key-dim × value-dim 记法，令 $$S_t\in\mathbb R^{d_k\times d_v}$$，一个先衰减再做 delta correction 的更新为：

$$
\bar S_t=\lambda_t S_{t-1},\qquad
S_t=\bar S_t+\beta_tk_t(v_t-\bar S_t^\top k_t)^\top.
$$

这里 $$k_t\in\mathbb R^{d_k}$$ 是写入 key，$$v_t\in\mathbb R^{d_v}$$ 是目标 value，$$\lambda_t$$ 控制保留程度，$$\beta_t$$ 控制写入强度。$$\bar S_t^\top k_t$$ 是衰减后的旧状态沿当前 key 读出的 value；误差由明确的减法得到，再写回状态。

固定形状限制了存储增长，也迫使不同历史共享有限通道。delta correction 规定怎样修正一条关联，但需要保留哪些关联，还受投影、门控和外层训练目标影响。

### 6.1 Titans：让 memory module 在 test time 学会记忆

[Titans §3.1](https://arxiv.org/html/2501.00663#S3.SS1)让一个 neural memory 学习 key 到 value 的关联。输入投影出 $$k_t,v_t$$，memory 网络的动态权重记作 $$\Theta$$，写入目标为：

$$
\ell_t(\Theta)=\|\mathcal M_\Theta(k_t)-v_t\|_2^2,
\qquad g_t=\nabla_\Theta\ell_t(\Theta_{t-1}).
$$

若旧 memory 已能预测这个 value，误差与梯度通常较小；否则梯度提供修正方向。论文再用动量保留近期更新方向，用遗忘门衰减旧权重。这里在线改变的是 memory 网络权重，K/V/Q 投影及更新规则由外层训练学习；读取时把 query 输入 memory 网络得到 readout。这个梯度目标才是 surprise 的具体含义，不是直接判断文本“重要不重要”。

读出接入主干有三种方式：

| 方式 | memory 怎么接进模型 | 直觉 |
|---|---|---|
| MAC, Memory as Context | memory readout 作为上下文供 attention 使用 | 像把内部记忆翻译成可读上下文 |
| MAG, Memory as Gate | sliding-window attention 做短期记忆，neural memory 作为 gated branch | 短期精读 + 长期淡化记忆 |
| MAL, Memory as Layer | 用 memory layer 顺序替换/补充部分序列层 | 更像 recurrent layer |

三种方式共享在线写入思想，区别在 readout 作为上下文、门控分支还是串行层参与后续计算。

### 6.2 δ-mem：小状态也能通过 attention correction 起作用

δ-mem 更轻量。它在 frozen full-attention backbone 旁边维护一个很小的 online state of associative memory，例如论文主打的 $$8 \times 8$$ state。读的时候，当前输入 query 这个 state 得到 $$r_t$$；随后 $$r_t$$ 变成 query-side 和 output-side 的低秩修正：

```text
old state S_{t-1}
  -> read r_t
  -> Δq_t, Δo_t
  -> steer attention and output
  -> gated delta write -> S_t
```

记忆通过 attention correction 影响冻结主干。小状态能提供有用历史信号，但其容量证据不支持把 8×8 配置当成复杂长期记忆的充分预算。

### 6.3 LiveMem：把 context turnover 变成训练条件

LiveMem 抓住了一个很实际的问题：模型有 recurrent state，不代表它真的用这个 state。只要关键证据还在 attention window 里，模型完全可能绕过 memory branch，继续靠原始 KV 回答。

所以 LiveMem 把问题定义成 **state continuity under context turnover**。主 attention path 只保留有界工作窗口；旧 chunk 被 FIFO 弹出，系统 prompt 作为 attention sink 保留；GDN2 side branch 则持续更新固定形状 memory state。当证据离开当前 KV window 后，模型还要靠这个 memory state 回答。

它的训练也围绕这个约束设计：用动态 attention mask 模拟 context turnover，让历史在训练时就从主 attention 中消失，迫使模型学习把必要信息写进 side branch。

这里通过移除原文通道，让任务结果依赖 memory，而非仅增加一个可被忽略的分支。这个干预也成为后面比较训练目标的基准条件。

### 6.4 TTCD：用长短窗口的表示差决定写入什么

[Test-Time Context Distillation（TTCD）](https://arxiv.org/abs/2608.01672)（2026-08）把重点放在内层学习目标上。单纯重构已经读到的输入，并不保证有限状态保留的是后续预测需要的信息。TTCD 在同一位置使用较长窗口 $$w_T$$ 的 teacher 路径和较短窗口 $$w_S$$ 的 student 路径，令 $$w_T>w_S$$，用两者的表示差指导在线写入。两条路径都只使用已到达的因果上下文，teacher 也不等于能读取未来答案。

论文重点研究 In-Place TTCD：把 SwiGLU 的 down-projection $$W\in\mathbb R^{d\times m}$$ 作为 fast weights。设两条路径在 down-projection 前的激活为 $$z_t^T,z_t^S\in\mathbb R^m$$，输出为 $$y_t^T=Wz_t^T$$、$$y_t^S=Wz_t^S$$。对第 $$i$$ 个 chunk $$C_i$$，先用同一份 chunk 起点权重计算两条路径，再更新：

$$
\mathcal L_i^{\mathrm{CD}}
=\frac{1}{2|C_i|}\sum_{t\in C_i}
\left\|y_t^S-\operatorname{stopgrad}(y_t^T)\right\|_2^2,
$$

$$
W_i=W_{i-1}
-\frac{\eta_i}{|C_i|}
\sum_{t\in C_i}(y_t^S-y_t^T)(z_t^S)^\top.
$$

$$\eta_i$$ 是在线更新步长。teacher 输出停止梯度；表示差决定向 value 参数写入什么，student 激活决定以后什么输入会读到这次更新。每一项都是 rank-one correction，但整块更新是多项之和。

下面只展示单层 fast-weight 写入。投影、短卷积和完整训练图省略；代码用行向量 batch，与上面的列向量公式等价：

~~~python
# W_down: [d, m]，离线训练得到的初始 down-projection
# W_fast 是当前序列的状态，不直接覆盖共享模型 checkpoint
W_fast = W_down.clone()

for chunk in causal_chunks:
    z_short, z_long = dual_window_features(chunk)
    # z_short, z_long: [n, m]，n 是当前 chunk 的 token 数
    y_short = z_short @ W_fast.T               # [n, d]
    y_target = stop_gradient(z_long @ W_fast.T)
    error = y_short - y_target                 # [n, d]
    grad_W = error.T @ z_short / len(chunk)    # [d, m]
    emit_student_output(y_short)
    W_fast = W_fast - eta(chunk) * grad_W      # 供后续 chunk 使用
~~~

这与 §1.2 的 FFN 记忆解释正好接上：普通 FFN 读取固定 value 参数，IP-TTCD 在读完一块上下文后修正这些参数，使后续短窗口计算能利用已经离开窗口的信息。它首先提供的是 sequence-specific memory，跨会话的保存、删除与生命周期仍需额外设计。

teacher–student 路径在 test-time 上下文处理期间也需要计算，不能只把它当作离线蒸馏后消失的教师。论文从头预训练到 760M、继续预训练到 8B；双窗口 attention 的融合 kernel 降低了 prefill 额外开销，但训练仍比其标准 Transformer 基线慢约 1.1–1.2×。因此要同时评估保留的信息、在线写入成本和后续读取收益。

### 6.5 MoNe：冻结主干，分段写入外挂 neural memory

[MoNe](https://arxiv.org/abs/2608.17616)（2026-08）选择外挂模块而非直接更新主干 MLP。读取长上下文时，每层用自己的 activation 生成 key/value，通过局部 associative loss 更新该层 fast-weight memory；回答时，从 query 读取 memory，生成供冻结主干 attention 使用的 memory tokens。

它把工作分成两个阶段：先逐段消费历史并写入，再用固定大小的记忆回答。论文的关联目标是增大 memory 输出与目标 value 的内积，不能直接套成 Titans 或 GDN 的平方误差更新。梯度只在相应 memory layer 内计算，避免每次写入都从最终 LM loss 反传整个主干。

“冻结主干”也不代表完全免训练：LoRA adapters、学习率/动量投影和输出缩放等元参数需要离线训练，部署后才固定这些规则、只更新 fast weights。论文主要用 Qwen2.5-0.5B 和 RULER 受控检索任务验证，尚不足以推出真实长期对话或大规模模型上的同等收益。

它与 Metis 的比较点在更新接口。MoNe 用局部梯度把分段上下文写入 neural memory；Metis 更强调由训练得到的 native procedure 管理跨交互状态。MoNe 对固定查询长度的读取成本不随历史长度增长，但读入历史的预处理仍为 $$O(N)$$，其中 $$N$$ 是历史 token 数。

<a id="prottt-semantic-outer-objective"></a>

### 6.6 ProTTT：内层预测历史，外层监督语义

TTCD 在在线更新时改变目标，MoNe 用局部关联目标写入；[ProTTT](https://arxiv.org/abs/2609.32564v1)（2026-09）则保留内层 next-token prediction，改变初始化怎样被训练。它给冻结模型的层间加低秩 adapter，以这些参数作为用户 memory。用户历史按时间切块，从共享初始化开始顺序适配：

$$
\psi_{u,0}=\psi_0,\qquad
\psi_{u,j}=\psi_{u,j-1}-\eta\nabla_\psi
\mathcal L_{\mathrm{NTP}}(C_{u,j};\psi_{u,j-1}).
$$

这里用 $$\psi$$ 记 adapter 参数，以免与前文的 backbone 和 latent pool 记号混淆；$$u$$ 是用户，$$C_{u,j}$$ 是第 $$j$$ 个历史 chunk。形成用户状态后，外层要求模型据此生成训练用户的语义 profile，并将损失反传穿过内层更新：

$$
\mathcal L_{\mathrm{outer}}(\psi_0)
=\mathbb E_u\left[-\frac{1}{|P_u|}\log p_{\theta,\psi_{u,J}}
(P_u\mid\mathrm{profile\ instruction})\right].
$$

$$P_u$$ 是训练期的 profile 目标，损失按其 token 数归一化，$$\theta$$ 为冻结主干。部署时只在新用户的历史上做内层更新，随后固定该用户的 adapter 回答；新增历史可以从现有状态继续适配，不需要提供 profile 标签或重算全部旧历史。

同样消费一段文本，普通 NTP 可能优先拟合局部措辞；ProTTT 用适配后的语义任务塑造初始化，使这些更新更有机会提取可用于个性化的知识。论文保持 TTT 过程不变、替换 profile 预测、响应选择和响应生成等外层目标，观察到不同的收益方向。因此，**写入规则相同，外层目标不同，形成的记忆也可能不同。**

主实验为 Qwen3-0.6B，另有 Llama-3.2-1B-Instruct。它优于 full-history ICL 和所比较的参数方法，但 PersonaMem Type 6 上 BM25 检索更好。CMV 的约 10× 单查询延迟改善发生在记忆构建后，另有一次用户历史处理成本；这些结果主要支持小模型个性化，尚未验证一般长期事实管理。

一个相邻训练思路是 [Delayed Supervision](https://arxiv.org/abs/2609.32312v1)：在事实之后插入无关事件，再问仍有效或已修订的问题。LaCT/DeltaNet 构造至少间隔 4,096 tokens 和 16 次状态变更；问答在独立分支上计分后丢弃，答案不进入继续运行的事件流。它将“经过干扰后仍可使用”变成监督条件。现有对照未隔离 delay 本身相对普通 QA、答案格式训练及额外计算的贡献，适合支持训练目标的讨论，而非证明延迟监督普遍更优。

<a id="7-metis把记忆写成模型原生过程"></a>

## 7. 从 Metis 到 MemT-1.0：让跨交互状态可更新

TTT 可以将序列写入动态参数副本；Metis 则明确把跨交互状态与读写控制器作为模型接口。其后续 MemT-1.0 继续修改更新规则与容量配置，两者应分开比较。

论文把 native memory 拆成两件事：

1. **Native Memory State**：模型在多次 inference 之间保持一个持续演化的内部状态；
2. **Native Memory Procedure**：remember、forget、update、reflect 这些操作由模型 forward computation 自主完成，而不是由外部规则触发。

用一个简化式子写，就是：

$$
(Y_t, M_{t+1}) = f_\Phi(X_t, M_t),
$$

其中 $$\Phi$$ 是静态模型参数，$$M_t$$ 是当前 memory state。交互时，$$\Phi$$ 不更新，$$M_t$$ 会随着输入和输出变成 $$M_{t+1}$$。

这和普通长上下文模型的区别在于：历史不再必须作为 $$C_t$$ 拼到 prompt 里。它也和 test-time training 不完全一样：TTT 通常用自监督 loss 在当前序列里适配参数；Metis 更强调跨 step 的 persistent memory state，以及明确的 memory procedures。

### 7.1 Metis block：local memory + hyper memory

Metis 将 memory 分支接到 attention block；公开的 Qwen3.5 hybrid 实现接在 full-attention 层，linear 层保留原路径。每个 Metis block 有两部分：

| 组件 | 是什么 | 作用 |
|---|---|---|
| local memory block | 动态的 dense memory network $$M_t^{(l)} \in \mathbb{R}^{d_k \times d_v}$$，以及 query-key normalization vector $$S_t^{(l)} \in \mathbb{R}^{d_k}$$ | 保存当前 step 之前积累的 memory state |
| hyper memory block | mid-training 后固定的参数，如 importance scorer、memory key/value/query projections | 从当前激活里选择和投影信息，决定怎么更新 local memory |

local memory 随交互变化；hyper memory 是离线训练得到、在线固定的读写控制器。

### 7.2 写入：先选重要激活，再更新 memory state

一次交互结束后，Metis 对该步中间激活做归一化，用 learned vector 打分并 softmax，保留累计概率达到 $$\rho$$ 的高分位置（同时满足最少选择数）。例如概率为 $$(0.6,0.3,0.1)$$、$$\rho=0.8$$ 时，前两个位置已足够；$$\rho$$ 是概率质量门槛，不是固定保留 80% tokens。训练用 straight-through estimator 向 scorer 传梯度。

简化成流程就是：

```text
本步激活 -> 打分并选位置 -> 投影为 memory K/V
         -> 对旧 memory 做 gated-delta 修正 -> 提交本步新状态
```

论文 §3.3 先解释线性累加版，最终模型采用 GDU。按[公开 GDU 实现](https://huggingface.co/IAAR-Shanghai/Metis-27B/blob/main/metis_hyper_memory.py)，省略层标和 batch，将 $$n$$ 个所选位置的 keys（已归一化缩放）及 values 记作 $$K\in\mathbb R^{n\times d_k}$$、$$V\in\mathbb R^{n\times d_v}$$，更新可整理成：

$$
\bar M=\alpha M_t,\qquad
E=V-K\bar M,\qquad
M_{t+1}=\bar M+K^\top\operatorname{Diag}(\beta)E.
$$

$$\alpha$$ 是所选激活产生的 sigmoid 遗忘门的加权平均；向量 $$\beta$$ 合并选择权重、sigmoid 写入门与更新比例。$$K\bar M$$ 读出旧 memory 对这些 keys 已保存的内容，$$E$$ 才是待修正的差值。normalization state $$S$$ 使用相同擦写结构，但目标 value 换成 1。实现将所选 tokens 对同一旧状态的修正批量汇总，是并行近似，不是逐 token 递推的精确展开。第二篇解释了 delta 核心；这里新增的是跨交互的选择、提交和持久状态。

### 7.3 读取：memory attention 作为原 attention 的并联分支

读 memory 时，Metis 不把 $$M_t$$ 解码成文本再拼回 prompt，而是让当前 hidden state 生成 memory query：

$$
\tilde Q_t^{(l)} = \operatorname{L2Norm}(\tilde H_t^{(l)}\tilde W_Q^{(l)}).
$$

这里 L2Norm 逐 head 归一化 query。随后做“读矩阵／归一化质量”的逐行相除，按[公开 normalized-memory 实现](https://huggingface.co/IAAR-Shanghai/Metis-4B/blob/main/metis_local_memory.py)，写成：

$$
\tilde A_t^{(l)}
= \operatorname{diag}(\tilde Q_t^{(l)} S_t^{(l)}+\mathbf 1)^{-1}
   \tilde Q_t^{(l)}M_t^{(l)}.
$$

$$\mathbf 1$$ 是稳定分母的全 1 向量，$$S_t$$ 记录与写入配套的归一化状态。读出经 Norm 对齐尺度后，与当前上下文 attention 混合：

$$
A_t^{(l)}
= \gamma \cdot A_{\text{orig}}^{(l)}
+ (1-\gamma)\cdot \operatorname{Norm}(\tilde A_t^{(l)}).
$$

$$\gamma\in[0,1]$$ 平衡两分支。当前 query 读取先前提交的 memory，当前上下文仍由原 attention 处理；本步提交的新状态供后续交互使用。

### 7.4 训练：从压缩到 memory operation

重构要求状态保留来源信息，操作监督则要求同一份状态能被记住、修订、遗忘或组合。Metis 的 memory-specific mid-training 将目标分成三类：

| 目标 | 训练什么 | 直觉 |
|---|---|---|
| memory reconstruction | 根据早先输入，在后续 query 中重构内容 | 测 memory state 的信息容量上限 |
| memory operation | remember、forget、update、reflect | 测模型是否按指令改变 memory state |
| regularization | multi-entity binding、selective forgetting、memory pollution | 防止相似事实串扰，避免不该用 memory 时乱用 |

这也是 Metis 和普通 context compressor 的分界。context compressor 主要问“压完还能不能回答”；Metis 还问“模型能不能根据语义命令改变自己的 memory state”。

### 7.5 Metis 的结果应该怎么读

Metis 的实验设置里有一个重要对照：no-context。也就是说，后续 query 不再拿到原始历史，只能依赖模型内部 memory state。标准 Qwen3.5 backbone 在这种设置下几乎不能恢复历史事实；Metis 能明显恢复一部分 memory operation 和 memory-based QA 能力。

但论文也写得比较谨慎。Metis 仍然不是外部记忆的完整替代：

- fixed-size latent state 容量有限，长时间累积后会衰减；
- 语义相近的事实可能在 latent space 里混淆；
- forgetting 通常比 remembering / update 更难泛化；
- memory state 的可解释性、可控删除、权限隔离都还远不如文本数据库。

这些实验支持模型学习内部状态的读写程序；容量、删除与权限隔离仍是部署前需要另外验证的接口。

<a id="memt-stable-updates"></a>

### 7.6 MemT-1.0：稳定擦除，减少对强遗忘的依赖

[MemT-1.0](https://arxiv.org/abs/2610.26825v2)（2026-10，以下依据 v2）延续 local/hyper memory 分工，但不再沿用 Metis 的位置选择与批量旧状态修正。它在一次 turn 的写入阶段按 token 更新多通道状态，采用 Stable GDN-2，移除原有 query-key normalization vector，并以归一化、输出 gate 等组件组织读取。生成读取先前提交的快照，写入形成后续交互使用的新状态。

其中最值得展开的是 Appendix C 的擦除几何。采用列向量、归一化 key，省略层和通道索引，原 GDN-2 的一个擦除因子可写为

$$
A=I-k(\beta\odot k)^\top.
$$

向量 gate 作用在各 key channel 上时，$$\beta\odot k$$ 未必与 $$k$$ 共线。此时用一个方向测量、沿另一个方向修正，得到非对称算子：某些输入方向的范数可以增加。特征值没有越过单位圆，并不足以保证单步不增益；简单加重衰减会同时削弱长期保留。

MemT 将 gate 的平方根移到两侧，令

$$
u=\sqrt\beta\odot k,\qquad A_{\mathrm{stable}}=I-uu^\top.
$$

在 $$\|k\|_2=1$$、各 gate 分量位于 $$(0,1)$$ 的条件下，有 $$\|u\|_2^2\leq1$$。该算子沿 $$u$$ 的分量缩放为 $$1-\|u\|_2^2$$，正交分量不变，因此擦除部分在欧氏范数下非扩张。完整更新还含逐通道衰减和新 value 写入：

$$
M_i=(I-u_i u_i^\top)D_iM_{i-1}
+u_i(\omega_i\odot v_i)^\top.
$$

这里 $$D_i$$ 来自 log-decay，$$\omega_i$$ 是 write gate。这个推导约束的是旧状态的变换，不是任意输入下完整带写入系统的无限期有界保证。

![MemT-1.0 对 GDN-2 擦除几何的分析](/assets/img/blog/long-context-internal-memory/memt-stable-gdn2-erase-geometry-fig12.svg)

*图源：[MemT-1.0，Figure 12](https://arxiv.org/html/2610.26825v2#A3.F12)。图中关闭新写入、取衰减为恒等，仅比较同一输入下的单步擦除；不能将其读成整个长期记忆系统的容量实验。*

### 7.7 扩容还需要区分读写地址、状态和参数

MemT 的 Mixture-of-Storage（MoS）将一个 block 扩成多个 memory experts。读取与写入使用不同 router 投影，但共享 expert 地址；读取权重混合专家 readout，写入权重调节对应专家的 erase/write 强度。它采用 **dense routing**，全部专家参与计算，与只激活少数专家的 sparse MoE 不同。

这样既能让读取需求与保存需求采用不同权重，也增加了动态状态和可训练 hyper memory 参数。论文在同一训练阶段比较 2/4 专家，较大配置通常更好；由于两种容量同时改变，不能只归因于多存了几份矩阵。报告的 128K 下约 66× 是 Nano 持久状态相对 full-context KV 的特定存储对照，也不等于整个服务进程显存降低同样倍数。强文本记忆系统总体仍领先，尤其在需要详细证据的长历史任务上。

Metis 到 MemT 的变化说明，持续写入不只需要选内容，还要让更新几何、状态规模和读写训练能够配合。native 接口提供了测试这些变量的位置，可靠的删除与来源追溯仍需另行验证。

<a id="8-和-metis-接近的近期工作"></a>

<a id="8-按更新接口比较-metis-与相邻工作"></a>

## 8. REMORY：让 latent memory 补充文本摘要

前面的状态主要试图独立承载一部分历史。[REMORY](https://arxiv.org/abs/2610.11287v1)（2026-10）保留已有摘要，问一个更具体的问题：如果摘要已经表达了任务主线，连续向量还应学什么？

在第 $$i$$ 次 compaction，令 $$H_i$$ 包含此前摘要、已有 residual memory 和此后的交互，新摘要为 $$S_i^{\mathrm{text}}$$。memory network 根据两者生成有界 soft tokens：

$$
M_i^{\mathrm{res}}=C_\varphi(H_i;S_i^{\mathrm{text}}),\qquad
|M_i^{\mathrm{res}}|\leq m_{\mathrm{res}}.
$$

下一次调用给冻结 actor 输入摘要与这些 tokens 的拼接，后续 compaction 重复同一过程。这里另加上标以区别前文的矩阵 state。源文本仍可通过普通 archive 工具检索；continuous memory 参与当前计算，但不承担保存每条原始记录的全部职责。

![REMORY 的摘要条件残差记忆与训练](/assets/img/blog/long-context-internal-memory/remory-residual-compensation-fig3.svg)

*图源：[REMORY，Figure 3](https://arxiv.org/html/2610.11287v1#S2.F3)。摘要同时作为后续输入和 memory 构建的条件；右侧曲线是两阶段训练的优化诊断，不是独立任务准确率。*

### 8.1 给定摘要之后，优化剩余的预测差异

训练先让 soft memory 支持重构，再转向 continuation：teacher 读取完整历史，student 读取固定摘要加 residual，二者预测同一后续回答。只更新 memory network，通过冻结 actor 的梯度优化 student-to-teacher reverse KL：

$$
\mathcal L_{\mathrm{res}}
=\frac{\sum_j w_jD_{\mathrm{KL}}(q_j\|p_j)}{\sum_j w_j}.
$$

$$p_j$$ 是完整历史条件下的 teacher 分布，$$q_j$$ 是摘要加 memory 条件下的 student 分布，$$w_j$$ 是位置权重。memory encoder 不接收未来回答。由于摘要已经可用，剩余预测差异决定 soft tokens 还需要提供什么；训练并不要求它们各自对应一条遗漏事实，也不保证其内容与摘要没有重叠。

这是一个条件化的记忆任务：不用先把所有历史压成同一份万能摘要，而是利用已有文字表示，学习能改善后续计算的补充。它与 ProTTT 的共同点是，状态的内容取决于后续任务怎样监督，而非只取决于压缩率。

### 8.2 收益落在哪里，比总分接近更有信息

SummHay 的同一组 92 个查询给出四种对照：

| 输入表示 | Coverage | Citation F1 | Joint score |
|---|---:|---:|---:|
| 完整原文 | 72.46 | 56.60 | 43.41 |
| 文字摘要 | 67.55 | 56.98 | 39.84 |
| 摘要 + 等额位置预算的额外文字 | 67.52 | 56.94 | 39.78 |
| 摘要 + REMORY | 67.95 | 61.03 | 43.39 |

joint score 接近完整原文，内部构成却不同：信息覆盖仍较低，来源归因更好。相对文字摘要，coverage 只增加约 0.40 点，citation F1 增加约 4.04 点；等预算的文字补充没有相同收益。这支持连续补充有助于将已恢复的信息与证据关联，不能据此声称所有历史细节都被无损保存。

最终 residual 最多 4,096 slots，source encoding 和中间表示可以更大；两个 actor 对应的可训练 memory network 约为 1.94B 和 1.502B 参数。agent 实验允许普通 archive 检索，报告的 token-priced generation costs 也不含 source encoding、memory network 和工具总成本。与其他接口比较时，需要把写入、回答、恢复原文分别计费。

### 8.3 邻近接口：复用计算与 episode 内参数适配

FlashMem 直接从已有 last hidden state 和 frozen KV cache 合成 latent memory，并以 attention entropy 触发 consolidation，关注已有计算能否减少再次编码。

TMEM 则把 memory 写入参数这件事做得更直接。它在 agent rollout 里维护 fast LoRA weights $$\Delta_t$$。当上下文预算到顶时，agent 把当前会话蒸馏成 QA-style supervision，然后做轻量 online SFT 更新 $$\Delta_t$$；后续动作从 $$\pi_{\theta_0+\Delta_t}$$ 采样。这个方向很像 test-time training 和 agent memory 的结合。它和 Metis 的主要区别是：Metis 推理时 learned weights 冻结，动态变化的是 native memory state；TMEM 让 LoRA fast weights 在 episode 内真的更新。

RAG-Tuned-LLM 与 RING 将检索式行为或外部知识内化到参数路径中，分别涉及训练数据、Knowledge Expert 和 RL 检索。它们有助于比较知识注入，但不能仅凭推理时不检索原文，就认定模型维护了持续更新的跨交互状态。

<a id="9-这条线真正发生了什么变化"></a>

## 9. 用同一条交互轨迹检查记忆生命周期

比较这些方法时，可以在同一份历史上设置连续的干预，而不只测一次“读入后问答”：

| 测试阶段 | 控制条件 | 要检查什么 |
|---|---|---|
| 写入后立即查询 | 原文仍在窗口 / 原文已移除分别测试 | 答案来自原文通道还是 memory |
| 持续加入无关历史 | 固定总状态预算，增加干扰和相似实体 | 保持率、串扰与有效容量 |
| 修正旧事实 | 新旧事实实体相同、值冲突 | 是否采用新值，是否污染其他实体 |
| 要求遗忘并继续交互 | 多种改写、延迟查询和重新保存状态 | 行为上不再读出，还是存在可恢复残留 |
| 保存、恢复与重置 | 分开检查同序列、跨轮、跨会话 | 声称的持续记忆是否有实际状态支持 |

这些是评估设计建议，并非每篇论文都完成了全部干预。ProTTT 的用户信息移除对照、REMORY 的等位置文字补充、Metis 的 no-context，分别排查“学到任务先验”“只是多给了输入预算”“仍在读取原文”等解释。对照的作用不同，应放在各自主张旁边。

还要区分三个预算：持久保存多少、每个 query 读取多少、训练和写入花多少。Memorilla 固定读出位置而保留增长的库，MemT 固定每条轨迹的配置状态，REMORY 在文字摘要之外增加连续输入；不能用同一个压缩倍率替代这些差别。

<a id="10-还有哪些问题没有解决"></a>

## 10. 容量、遗忘与成本：还有哪些边界

更新规则和读出接口已有多种实现，长期可靠性仍取决于以下具体条件。

### 10.1 容量：fixed-size state 到底能记多少

只要 memory state 固定，容量就有限。线性 state、GDN state、Metis 的 $$M_t$$、δ-mem 的小矩阵，都无法保证任意历史事实可恢复。它们更像压缩过的工作记忆，而不是数据库。

增加状态应同时测试保持率、相似实体串扰和事实修订，而不只测立即问答。选择性写入还会引入另一种损失：未来有用的信息可能在进入状态之前就被跳过。

[Write What Matters](https://arxiv.org/abs/2610.34180v1)（2026-10）把“是否写该 token”和“更新几个 slots”分开。在十个种子、两个合成任务中，固定四个 slots 时，信息密集的 Feature Recall 偏向 always-write（0.472 对 0.337），稀疏 Passkey 则偏向 learned selection（0.222 对 0.048）；所测 adaptive allocation 也没有胜过固定分配。它提供的是受控反例：过滤干扰和避免漏写要根据信息密度权衡，不能推成自然语言中的普适路由规则。

[Proteus](https://arxiv.org/abs/2608.16844)（2026-08）进一步研究有效容量的开放时机：序列早期只启用一部分记忆容量，随着上下文增长逐步开放更多容量，促使早期信息先压缩，并为后续写入留下空间。它调整固定预算内的使用顺序，总状态仍有容量上限。论文在多种 neural memory 架构上报告收益，适合与“直接增加 state size”做对照实验。

### 10.2 遗忘：删除 latent memory 比写入难

文本记忆可以删一条记录，latent memory 不能这么简单。一个事实可能被分散写进许多通道、层和 update 轨迹里。所谓 forget，往往只是让后续读出时不再暴露它，而不一定真的从状态中抹掉。

这对隐私、安全和可控个性化都很关键。用户说“忘掉 X”时，系统要能给出可验证的保证；而 latent state 的删除目前很难审计。

### 10.3 可解释性：memory state 不是人能读的笔记

文本式外部记忆便于检查原文、删除记录和追溯来源。内部表示可以减少重复拼接或限制读取预算，但写入、读取和恢复的延迟取决于具体接口；其内容也更难直接检查。

MemDefrag 用中间层 attention density 找 target fragment，是一个有趣方向；Metis 用 memory operation 数据约束行为，也是一个方向。但如果 memory 要承担个性化、合规和长期 agent 状态，光有行为评估还不够。

### 10.4 训练：模型会不会绕过 memory

LiveMem 用 context turnover 移除捷径；ProTTT 用适配后的语义任务训练初始化；Metis 用操作监督要求状态可修订。三者都让训练条件贴近预期用途，但约束不同。评估要在原文离开工作窗口后继续查询，并将记忆模块置空或替换，检查回答是否真的依赖该状态。

<a id="15-上下文更丰富参数一定学到更多吗"></a>

### 10.5 上下文更丰富，参数一定学到更多吗

[Information Abundance Paradox](https://arxiv.org/abs/2608.12218)（2026-08）研究了一个与预训练直接相关的问题：当答案所需信息已经能从训练上下文读到，模型是否还会同样积极地把它内化进参数？

论文固定 token 预算、模型配置与更新次数，改变预训练窗口；在其长文数据设置中，下游表现随窗口增长到中间最优点后出现下降。配套的 SFT、梯度分配与模块干预实验支持一种解释：丰富的相关上下文提供了更容易的预测路径，训练压力会更多转向上下文读取，对参数知识的依赖相应改变。

这是有条件的实证：预训练最大为 750M，窗口从 512 到 32K，比较并非 FLOP-matched，不能推成“长上下文必然损害知识”。它提示评估至少拆开 closed-book、正常证据和缺失/误导证据三种条件。长窗口下 loss 更低，并不自动说明参数记忆更强；前面的 LiveMem 和 TTCD 也都需要通过 mask 或写入目标，让模型真正使用预期的记忆路径。

<a id="105-和外部记忆的关系不是二选一"></a>

### 10.6 和外部记忆的关系：不是二选一

原始记录支持审计和精确追溯；文字摘要提供可读的任务状态；latent readout 或 recurrent state 参与模型内部计算。REMORY 给出了文字加连续补充的具体例子，Memorilla 则保留可增长的库、生成查询相关的短前缀。如何组合取决于后续问题是否可预期、是否需要恢复原文，以及更新和删除的责任边界。

```text
external memory: 可审计的文本/结构化记录
latent memory:   压缩后的可检索内部表示
state memory:    随交互连续演化的模型内部状态
```

连续状态可以减少重复输入，但写入、读取和恢复的成本仍要逐项测量。对于开头的项目版本，模型既要在回答时采用新值，也应能说明来源；若只有 latent 关联而没有外部记录，后一项不会自动成立。

<a id="11-小结从-context-management-到-state-management"></a>

## 11. 小结：记忆要对后续行为负责

FFN 的 key-value 展开解释了固定参数怎样提供关联；memory tokens 和 latent pool 让历史成为可保存的连续表示；fast weights 与 native procedure 则把更新接进后续计算。它们共有读写接口，但写入对象与持续范围并不相同。

这些工作进一步拆开了几个变量：ProTTT 让外层语义目标塑造在线适配，MemT 分析旧状态的擦除几何并扩展读写通道，REMORY 根据已有摘要学习补充，Memorilla 等到 query 出现才压缩当前读出。状态更大、上下文更短或更新更稳定，各自只解决一部分问题。

回到项目版本的例子，可靠记忆要在原文离开窗口后采用新值，在其他信息持续写入后仍可使用，并能按要求修订或遗忘。训练目标需要奖励这些行为，测试需要隔离原文、任务先验和额外输入预算，系统则要明确保存、恢复和重置的边界。记忆的价值最终体现在后续任务中仍能使用哪些信息，以及为此付出了多少代价。

## 延伸阅读

- 从静态关联开始：LAMA、closed-book QA、FFN key-value memory，再读 ROME/MEMIT，区分参数知识的形成与编辑。
- 比较连续表示：RMT、ICAE、Activation Beacon 说明不同的压缩接口；MemoryLLM/M+ 比较固定池和长期检索；Memorilla 区分保存预算与查询条件读出。
- 比较在线写入：Titans 与第二篇的 delta 推导将 $$S_t$$ 解释为关联表；TTCD 比较长短窗口目标，MoNe 比较局部梯度接口，ProTTT 比较外层语义监督；[E²-TTT](https://arxiv.org/abs/2608.21308) 讨论 chunk 起点梯度近似下的并行时序结构。
- 比较跨交互状态：先读 Metis 的 local/hyper memory 与提交边界，再读 MemT-1.0 §2 和 Appendix C 的稳定更新；MoS 的扩容要连同参数、状态和计算一起比较。
- 比较文字与 latent 的互补：REMORY §2 的摘要条件训练和 Table 1 的归因对照，比只比较压缩率更能说明保留了什么。
- 设计容量实验：Proteus 调整固定预算内的激活顺序；Write What Matters 分开 token 选择与 slot 分配。受控任务用于定位机制，长期交互仍需另测。

## 参考资料

### Memory tokens 与 context compression

- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)
- [Scaling Transformer to 1M tokens and beyond with RMT](https://arxiv.org/abs/2304.11062)
- [Adapting Language Models to Compress Contexts](https://arxiv.org/abs/2305.14788)
- [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945)
- [Long Context Compression with Activation Beacon](https://arxiv.org/abs/2401.03462)

### Long-term / latent memory

- [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
- [Augmenting Language Models with Long-Term Memory](https://arxiv.org/abs/2306.07174)
- [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/abs/2402.04624)
- [M+: Extending MemoryLLM with Scalable Long-Term Memory](https://arxiv.org/abs/2502.00592)
- [NextMem: Towards Latent Factual Memory for LLM-based Agents](https://arxiv.org/abs/2603.15634)
- [MemDefrag: Latent Memory Defragmentation for Large Language Models](https://arxiv.org/abs/2607.05969)
- [Memorilla: Latent Semantic Memory for LLMs](https://arxiv.org/abs/2610.21375v1)，2026-10。
- [REMORY: Learning Residual Memory for Context Compaction](https://arxiv.org/abs/2610.11287v1)，2026-10。

### Recurrent / fast-weight / native memory

- [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- [δ-mem: Efficient Online Memory for Large Language Models](https://arxiv.org/abs/2605.12357)
- [Gated DeltaNet-2: Decoupling erase and write in linear attention](https://arxiv.org/abs/2605.22791)
- [FlashMem: Distilling Intrinsic Latent Memory via Computation Reuse](https://arxiv.org/abs/2601.05505)
- [LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference](https://arxiv.org/abs/2608.02515)
- [Metis: Memory Foundation Model](https://arxiv.org/abs/2607.26760)
- [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825v2)，本文依据 2026-10-20 的 v2。
- [MemTensor/Metis](https://github.com/MemTensor/Metis)
- [Learning What to Remember: Test-Time Training via Context Distillation](https://arxiv.org/abs/2608.01672)
- [MoNe: Modular Neural Memory for Efficient Long Context Inference](https://arxiv.org/abs/2608.17616)
- [ProTTT: Learning to Learn Semantic User Memory with Test-Time Training](https://arxiv.org/abs/2609.32564v1)，2026-09。
- [Delayed Supervision for Test-Time Language Models](https://arxiv.org/abs/2609.32312v1)，2026-09。
- [Write What Matters: Disentangling Token Selection from Slot Allocation in Recurrent Memory](https://arxiv.org/abs/2610.34180v1)，2026-10。
- [Rethinking Expressivity and Efficiency in Test-Time Training](https://arxiv.org/abs/2608.21308)
- [Proteus: Incremental Memory Activation for Long-Context Sequence Modeling](https://arxiv.org/abs/2608.16844)

### 参数知识、架构记忆与模型编辑

- [Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066)
- [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [Large Memory Layers with Product Keys](https://arxiv.org/abs/1907.05242)
- [MoME: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup](https://arxiv.org/abs/2609.15126)
- [Information Abundance Paradox: Long-Context Training Undermines Parametric Knowledge](https://arxiv.org/abs/2608.12218)
- [Memory Layers at Scale](https://arxiv.org/abs/2412.09764)
- [Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696)
- [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)
- [Fast Model Editing at Scale](https://arxiv.org/abs/2110.11309)
- [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)

### Agent memory 与参数化检索

- [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841)
- [MemGen: Weaving Generative Latent Memory for Self-Evolving Agents](https://arxiv.org/abs/2509.24704)
- [Tuning LLMs by RAG Principles: Towards LLM-native Memory](https://arxiv.org/abs/2503.16071)
- [Scaling Self-Evolving Agents via Parametric Memory](https://arxiv.org/abs/2606.04536)
- [RING: Retrieval-Internalized Generation for Continual Large-Scale Knowledge Injection](https://arxiv.org/abs/2608.01630)
- [Are We Ready For An Agent-Native Memory System?](https://arxiv.org/abs/2606.24775)
