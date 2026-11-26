---
layout: post
title: "计算的回路（三）：Latent Reasoning 的连续状态与隐式计算"
date: 2026-10-19
last_modified_at: 2026-11-26
permalink: /blog/loop-latent-reasoning/
visibility: unlisted
sitemap: false
series: 计算的回路
series_description: 循环模型、迭代生成、隐式推理与推理系统
series_order: 3
lang: zh-CN
tags: [Latent Reasoning, Loop Transformer, Continuous Thought, Chain of Thought, Test-Time Compute]
description: "不写出文字思维链，模型如何保存和使用中间结果？从连续反馈、循环工作区到状态干预与跨轮读出，讨论 Latent Reasoning 与 Loop Transformer 的联系。"
excerpt: "不写出文字思维链，模型如何保存和使用中间结果？从连续反馈、循环工作区到状态干预与跨轮读出，讨论 Latent Reasoning 与 Loop Transformer 的联系。"
---

> 计算的回路 · [一 · 循环架构]({% post_url 2026-10-19-loop-transformer-architecture %}) / [二 · 去噪与生成]({% post_url 2026-10-19-loop-denoising-flows %}) / **三 · 隐式推理** / [四 · 推理系统]({% post_url 2026-11-26-loop-transformer-serving %})

一个模型答对题目之前，某个中间状态里可能已经能读出答案。这个结果还留下一个问题：如果改掉那部分状态，下一轮计算和最终答案会跟着变吗？这决定了我们测到的是可读出的相关信息，还是模型实际用到的中间结果。

状态可以沿序列传递，也可以在固定位置反复更新。连续反馈、循环工作区和读出网络的实验，分别抓住了这条链上的不同环节。

## 1. 隐式过程与额外计算

显式 Chain-of-Thought（CoT）把中间过程写成离散 token：

$$
q\longrightarrow t_1,t_2,\ldots,t_m\longrightarrow a,
$$

其中 $$q$$ 是问题，$$t_i$$ 是中间文本，$$a$$ 是答案。模型每生成一个 token，都把它加入后续上下文。过程是否在界面上展示，不改变它仍然沿离散 token 进行计算这一事实。

本文关注三种额外计算的载体。一个连续向量可以反馈给下一个位置，多组 latent slots 可以组成固定工作区，整段表示也能交给共享层反复更新。

另一些 implicit CoT 工作把训练中的显式过程逐步内化，推理时直接回答。这类方法减少了显式过程，却不一定保留额外的 latent rollout，也未必增加测试时计算。<a class="paper-citation" id="cite-2-1" href="#ref-2" title="Training Large Language Models to Reason in a Continuous Latent Space；§5.2 的 iCoT 对照" aria-label="参考文献 2，§5.2 的 iCoT 对照">[2，§5.2 的 iCoT 对照]</a>

即使增加了位置或循环，模型也可能忽略它们。新增状态对答案的作用，需要通过计算量对照和状态干预来检验。

隐藏文本、连续反馈和增加深度，改变的是不同部分：

| 设置 | 直接变化 | 仍需检验的性质 |
|---|---|---|
| 不显示中间文本 | 用户看不到过程 | 底层是否仍在生成离散 CoT |
| 用连续向量替代中间 token | 反馈表示发生了变化 | 是否保留了推理所需的信息与候选分支 |
| 增加 latent positions 或 loop 次数 | 增加了可用状态或计算 | 模型是否利用了额外预算，求解是否改善 |

<a class="paper-link" href="https://arxiv.org/abs/2310.02226v3" title="Think before you speak: Training Language Models With Pause Tokens">Pause Tokens</a><a class="paper-citation" id="cite-1-1" href="#ref-1" title="Think before you speak: Training Language Models With Pause Tokens" aria-label="参考文献 1">[1]</a> 在输入与答案之间放入可学习的占位符，增加可操作的位置。输入符号仍是离散的 pause token，经过网络后却可以携带题目相关的连续表示。论文中的收益依赖训练阶段也包含这种延迟，仅在测试时插入占位符没有同样的保证。

这些占位符没有普通词项的语义，仍然可以承担中间计算。由于它们预先已知，可以一起插入输入，并不需要像连续 thought 那样逐步生成下一位置的输入，两者的串行开销不同。

## 2. 连续 thought 的反馈路径

### 2.1 Coconut：不经过词表，直接反馈隐藏状态

在普通自回归解码中，设当前位置的顶层状态是 $$h_t$$，词嵌入表为 $$E$$，则下一位置的输入来自

$$
p_t=\operatorname{softmax}(W_{\mathrm{out}}h_t),\qquad
x_{t+1}\sim p_t,\qquad
e_{t+1}=E[x_{t+1}].
$$

Coconut<a class="paper-citation" id="cite-2-2" href="#ref-2" title="Training Large Language Models to Reason in a Continuous Latent Space" aria-label="参考文献 2">[2]</a> 在 latent mode 中跳过离散解码，直接令

$$
e_{t+1}=h_t.
$$

这里的 $$h_t$$ 已经过模型的最终归一化。下一次调用继续使用同一骨干网络，得到新的 hidden state，再反馈到后面。完成规定的 latent 阶段后，模型切回语言模式生成答案。

这条路径直接传递上一轮计算得到的状态，不要求它对应某个 token。中间表示因此可以超出单个词项，但容量仍受维度、数值精度和训练约束限制。

Coconut 从有显式 CoT 的样本开始训练，逐阶段把前面的推理步骤替换成连续 thought，再预测剩余文本和最终答案。latent 位置本身没有对应的 token 交叉熵目标，梯度通过连续反馈链传播。<a class="paper-citation" id="cite-2-3" href="#ref-2" title="Training Large Language Models to Reason in a Continuous Latent Space；§3" aria-label="参考文献 2，§3">[2，§3]</a>

损失只要求连续状态帮助预测后续内容，不要求逐字恢复被删除的 CoT。模型因此可以学到不同于原始文本的表示，其求解效果则取决于任务。

在论文的 GPT-2 设置下，GSM8k 上 Coconut 为 34.1 ± 1.5%，低于 CoT 的 42.9 ± 0.2%；在需要搜索的 ProsQA 上，Coconut 为 97.0 ± 0.3%，高于 CoT 的 77.5 ± 1.9%。<a class="paper-citation" id="cite-2-4" href="#ref-2" title="Training Large Language Models to Reason in a Continuous Latent Space；表 1" aria-label="参考文献 2，表 1">[2，表 1]</a>

在逻辑图任务中，作者观察到连续状态保留多个候选后继，并将这种行为解释为类似广度优先搜索。模型可能借此推迟分支选择，但这项观察限于特定任务，还不足以确定一般连续状态是否执行 BFS，或能够保留多少条独立轨迹。

### 2.2 Soft Thinking：反馈词向量的概率混合

<a class="paper-link" href="https://arxiv.org/abs/2505.15778v1" title="Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space">Soft Thinking</a><a class="paper-citation" id="cite-3-1" href="#ref-3" title="Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space" aria-label="参考文献 3">[3]</a> 保留词表预测，但不立即抽取一个 token。令 $$E\in\mathbb R^{V\times d}$$，下一位置使用

$$
e_{t+1}=E^\top p_t
=\sum_{v=1}^{V}p_t(v)E[v].
$$

这种输入处在词嵌入的凸包内。它与 Coconut 直接反馈 $$h_t$$ 不同：前者通过词表概率和 embedding 建立接口，后者直接连接输出隐藏空间与输入空间。Soft Thinking 是推理时方法，不要求按 Coconut 的课程重新训练；论文也明确指出，混合 embedding 会带来分布外输入，长时间反馈可能失稳，并引入基于低熵连续步数的 Cold Stop 机制。

混合 embedding 未必保留全部词表分布。当 $$V>d+1$$ 时，映射 $$p\mapsto E^\top p$$ 在整个概率单纯形上不可能是一一对应的：增广矩阵 $$[E^\top;\mathbf1^\top]$$ 存在非零零空间方向，某些不同分布会得到相同的混合 embedding。虽然所有词项的概率都参与了计算，送回模型的 $$d$$ 维向量未必保留它们的全部信息。

后续的非线性计算通常也不满足

$$
F\!\left(\sum_v p(v)E[v]\right)
=\sum_v p(v)F(E[v]).
$$

一次处理平均 embedding，通常不等价于分别运行所有候选、再加权合并结果。它能延缓离散选择，却没有自动实现多分支的并行搜索。

### 2.3 自回归模型原本也有连续记忆

标准 Transformer 生成 token 后，之前位置的 keys 和 values 仍留在 KV cache 中，供后续 attention 访问。离散反馈符号虽然最多携带 $$\log_2 V$$ 比特，模型跨步保留的信息却不止这个符号，还包括这条连续记忆通路。

连续反馈让深层状态直接回到后续输入端，重新获得一轮计算。<a class="paper-link" href="https://arxiv.org/abs/2608.08888v1" title="Full-bandwidth transformer">Full-bandwidth Transformer</a><a class="paper-citation" id="cite-11-1" href="#ref-11" title="Full-bandwidth transformer" aria-label="参考文献 11">[11]</a> 把前一步顶层状态与当前 token embedding 通过门控融合，送入下一步，同时保留标准 KV cache 和离散 token 生成。

这条 latent feedback 跨越 token 步骤，既不要求先执行一段纯 latent thought，也不同于同一 token 上的深度循环。

<a class="paper-link" href="https://arxiv.org/abs/2608.17981" title="Recirculation">Recirculation</a><a class="paper-citation" id="cite-22-1" href="#ref-22" title="Recirculation" aria-label="参考文献 22">[22]</a> 则把较深层的一部分 activation 投射到后续 input step 的浅层，与该层状态混合。它冻结现成模型的主权重，选择回流的来源层、目标层和混合系数；adaptive 版本另用轻量预测器调节强度。这种回流同时跨越 depth 和 step，使前后步骤的状态可以在相近层级继续更新。它与 Full-bandwidth 的接口不同，但都不能只按同一 token 内的循环次数描述。

这类跨步反馈有顺序成本：原本可以并行预填充的输入，需要按 step 传递状态。Recirculation 论文报告生成阶段额外延迟较小，因为生成本身已经串行。它所处理的长期状态跟踪，与单次 token 预测内部增加深度，是两个相关而不同的问题。

### 2.4 CODI：对齐显式与连续路径的状态

如果只给最终答案损失，中间状态可以采用很多不同的编码方式。答案正确并不能说明其中一段向量对应“列方程”，另一段对应“消元”；模型也可能学会忽略新增的 latent 步骤。

Coconut 用逐步移除显式 CoT 的课程缓解这个问题。<a class="paper-link" href="https://arxiv.org/abs/2502.21074v3" title="CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation">CODI</a><a class="paper-citation" id="cite-4-1" href="#ref-4" title="CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation" aria-label="参考文献 4">[4]</a> 采用另一种监督：同一个模型同时执行显式 CoT 的 teacher task 和连续 thought 的 student task，在答案前的指定位置对齐两条路径的隐藏状态。

用 $$u_T^{(\ell)}$$ 和 $$u_S^{(\ell)}$$ 表示两条路径在该位置、第 $$\ell$$ 层的状态，可以把蒸馏项写成

$$
\mathcal L_{\mathrm{KD}}
=\frac1M\sum_{\ell=1}^{M}
\left\|u_S^{(\ell)}-\operatorname{sg}(u_T^{(\ell)})\right\|_1.
$$

实现中还用 teacher 激活的标准差校准各层尺度。总目标结合 student 的答案损失、teacher 的显式 CoT 与答案损失，以及上述蒸馏项。两个分支共享模型；teacher 在蒸馏项中 stop-gradient，仍通过自己的语言建模损失学习。<a class="paper-citation" id="cite-4-2" href="#ref-4" title="CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation；§3" aria-label="参考文献 4，§3">[4，§3]</a>

CODI 对齐的是指定读出位置跨层的状态，不是要求每个连续 thought 与每个文本 token 一一对应。它在连续反馈前还加入 MLP 和归一化，处理 hidden space 与输入空间之间的接口。论文还从训练 CoT 中移除直接给出最终结果的末步，减少 teacher 通过复制答案产生蒸馏目标的捷径。

论文在 GPT-2 尺度的 GSM8k 设置中报告了接近显式 CoT 的准确率和 3.1 倍 CoT 压缩。推理仍需要逐步生成连续状态，每步仍要调用骨干，所以还需另计总 FLOPs 和端到端时延。训练也包含 teacher 分支和蒸馏损失。<a class="paper-citation" id="cite-4-3" href="#ref-4" title="CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation；§4" aria-label="参考文献 4，§4">[4，§4]</a>

只监督答案时，中间表示的自由度较大。Coconut 的课程让它逐步接替文本过程，CODI 则约束最终供解答使用的状态，使其接近显式路径。两者虽然在推理时减少了文本输出，训练仍然使用显式 CoT。

这里蒸馏的是显式与连续路径之间的状态。不同循环深度之间也可以蒸馏预测，但那不要求 latent 对应文本步骤；LoopOPD 的训练机制见[第一篇 §5.3]({% post_url 2026-10-19-loop-transformer-architecture %}#loopopd)。

## 3. 序列位置与共享深度

### 3.1 沿序列追加状态，与沿深度更新状态

Coconut 式反馈可以抽象为

$$
(h_k,\mathcal C_{k+1})
=\operatorname{LM}_\theta(e_k,\mathcal C_k),
\qquad e_{k+1}=P(h_k),
$$

其中 $$\mathcal C_k$$ 包含前面的 KV，$$P$$ 可以是恒等映射，也可以是投影。每步通常增加一个 latent 位置，骨干网络的普通层栈仍然执行。共享发生在序列步之间。每步内部仍执行原有的独立层。

标准 recurrent-depth Transformer 则把某组状态反复送入同一个计算块：

$$
H^{(r+1)}=F_\theta(H^{(r)},E(q)),
\qquad H^{(r)}\in\mathbb R^{N\times d}.
$$

$$r$$ 是循环深度，$$N$$ 是当前工作区的位置数。输入注入、归一化和状态初始化取决于具体架构；循环也可能只作用于部分位置。与逐步追加 latent token 相比，这种方法可以在位置数不变时增加计算，但会重复访问同一组参数和状态。

两者可以沿两个轴理解：

| 计算轴 | 增加预算时改变什么 | 典型机制 |
|---|---|---|
| 序列位置 | 新增可存储、可被 attention 访问的中间位置 | 显式 CoT、Coconut 式 latent 序列 |
| 循环深度 | 在既有位置上继续更新表示 | Recurrent-depth Transformer、LoopLM |
| 位置与深度同时使用 | 提供多个 latent slots，并反复更新它们 | Looped padded Transformer、LOTUS、LLoCoT |

沿序列展开时，中间结果保存在新位置；沿深度循环时，已有表示可以被继续修订。两者还可以组合，在每轮更新一组 latent positions。

<a class="paper-link" href="https://arxiv.org/abs/2610.24766v1" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction">ECHO</a><a class="paper-citation" id="cite-23-1" href="#ref-23" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction；§2.3、§3.1" aria-label="参考文献 23，§2.3、§3.1">[23，§2.3、§3.1]</a> 提供了另一种对照：它复用骨干末层，沿未来 token 的预测范围递归。隐状态在各步之间反馈，下一步还接收前一个离散 token 的 embedding，每一步预测一个更远的位置。相同参数持续使用，可以对应逐步推进的输出目标；它与固定 latent 工作区中的反复修订有不同的状态、位置和监督关系。共享末层与训练深度采样的细节见[第一篇 §5.4]({% post_url 2026-10-19-loop-transformer-architecture %}#echo-causal-recursion)。

### 3.2 共享规则与模拟 CoT 的条件

许多算法反复使用相同的局部操作，例如传播约束、更新候选集合、逐步改进一个解。共享计算块为这类重复操作提供了结构上的偏置：训练可以把“该做什么更新”存进参数，把“现在算到哪里”存进状态。

这并不要求每一轮做完全相同的事。即使参数不变，输入状态不同、可访问信息不同，网络产生的更新也不同。状态还可以编码阶段、局部进度或待解决的变量。

共享参数只约束了更新规则，没有规定模型必须推进求解。训练仍可能得到重复预测、遗忘中间量或稳定于错误答案的循环。

<a class="paper-link" href="https://arxiv.org/abs/2502.17416v1" title="Reasoning with Latent Thoughts: On the Power of Looped Transformers">Reasoning with Latent Thoughts</a><a class="paper-citation" id="cite-5-1" href="#ref-5" title="Reasoning with Latent Thoughts: On the Power of Looped Transformers" aria-label="参考文献 5">[5]</a> 从理论和实验两方面研究了这条联系。其 Theorem 5.4 构造了一个 looped Transformer：针对固定输入长度和 $$m$$ 步 CoT，在输入后追加 $$m$$ 个 dummy positions，并增加所需的内部维度、注意力头及常数层数开销，用 $$m$$ 次循环模拟原模型的 CoT 计算。

构造用工作区、计数和更新控制来指定每轮更新的位置、传递前面的结果，并模拟 token 编解码。这些额外机制是用 $$m$$ 次循环模拟 $$m$$ 个 CoT token 的条件，直接重复原模型并不具备它们。

因此，循环计算在表达能力上足以表示这类顺序过程。普通预训练能否学到这一构造、实现后的成本是否更低，定理没有回答。

对于可并行的子问题，多位置工作区可以在一轮中更新多个中间量；具体并行度仍受问题依赖和 attention 结构限制。

### 3.3 没有过程标签，也可能学出递归计算

<a class="paper-link" href="https://arxiv.org/abs/2502.05171v2" title="Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach">Scaling up Test-Time Compute with Latent Reasoning</a><a class="paper-citation" id="cite-6-1" href="#ref-6" title="Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" aria-label="参考文献 6">[6]</a> 的 recurrent-depth 模型在语言建模训练后，部分推理任务会随测试时循环深度增加而改善。它没有使用逐步 CoT 标签，说明这种监督不是学出可用递归计算的必要条件；各轮是否对应可读的逻辑步骤，仍未得到验证。

该模型从随机状态出发，却没有采用扩散污染路径和局部去噪目标。<a class="paper-link" href="https://arxiv.org/abs/2510.25741v5" title="Scaling Latent Reasoning via Looped Language Models">Ouro</a><a class="paper-citation" id="cite-7-1" href="#ref-7" title="Scaling Latent Reasoning via Looped Language Models" aria-label="参考文献 7">[7]</a> 又将共享深度用于更大规模的语言预训练；架构与预算比较见[第一篇]({% post_url 2026-10-19-loop-transformer-architecture %})。

内部循环也可以与显式 CoT 同时使用：每次 token 预测内部多做几轮，再输出推理文本。因此，latent reasoning 不必排斥文本过程，内部深度与外部文本长度可以分别分配。

## 4. 循环工作区与过程监督

### 4.1 LOTUS：固定位置、反复修订、文本目标

LOTUS<a class="paper-citation" id="cite-8-1" href="#ref-8" title="Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" aria-label="参考文献 8">[8]</a> 把多位置 latent 工作区、共享循环和显式 CoT 监督结合起来。

LOTUS 在问题后放置 $$K$$ 个 latent blocks，每个 block 有 $$c$$ 个位置，总工作区为 $$Kc$$ 个 latent positions。它先计算问题前缀的缓存 $$\mathcal C_q$$，然后在固定前缀条件下反复更新 latent 工作区。

初始化与循环更新分别为

$$
H^{(0)}=f_\theta(E_{\mathrm{lat}}\mid\mathcal C_q),
$$

$$
H^{(r)}=f_\theta(E_{\mathrm{lat}}+H^{(r-1)}\mid\mathcal C_q),
\qquad r=1,\ldots,R.
$$

其中 $$E_{\mathrm{lat}}$$ 是 latent 输入表示，$$H^{(r)}\in\mathbb R^{Kc\times d}$$。与 Coconut 每轮追加一个位置不同，LOTUS 在已有的一组位置上继续计算。问题前缀不必反复计算，latent 部分则由共享模型共同更新。

多个 latent positions 在同次调用中计算，但仍有顺序依赖。LOTUS 以 causal LM 为骨干，位置间的信息传播受因果 attention 约束。

设第 $$i$$ 个推理步骤 token 化后为 $$T_{i,1},\ldots,T_{i,c}$$，通过补齐或截断与 block 宽度对齐。LOTUS 在最终循环状态上计算

$$
\mathcal L_{\mathrm{step}}
=\frac1{N_{\mathrm{valid}}}
\sum_{(i,j)\in\mathcal M}
\operatorname{CE}\!\left(W_{\mathrm{out}}H^{(R)}_{i,j},T_{i,j}\right),
$$

其中 $$\mathcal M$$ 排除 padding 位置。再结合基于最终 latent 状态生成答案的损失：

$$
\mathcal L=\mathcal L_{\mathrm{answer}}
+\lambda\mathcal L_{\mathrm{step}}.
$$

Gold CoT token 只作为监督标签，不进入 latent 工作区的输入。测试时，模型直接使用 post-loop 状态生成答案，无须先将 latent 解码成推理文本。

LOTUS 主方法在最后一轮之后，对各 latent 位置并行施加过程交叉熵，并沿全部循环反传。它不同于第二篇 <a class="paper-link" href="https://arxiv.org/abs/2609.11801v1" title="Thinking with Looped Flows">Looped Flows</a> 的逐步局部监督。论文另有辅助解码器版本，采用不同的监督读取方式。<a class="paper-citation" id="cite-8-2" href="#ref-8" title="Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers；§3" aria-label="参考文献 8，§3">[8，§3]</a>

Block 与推理步骤对齐，多轮循环共同修订这些 blocks，最终状态才用于预测相应文本。因此，一个 CoT step 对应的是工作区中的位置分工，不是某一次独立循环。

### 4.2 LOTUS 的准确率与分阶段延迟

在论文的 Llama-3.2-3B-Instruct、GSM8k-Aug 紧凑数学表达式训练设置下，主配置采用 $$K=6$$、$$c=25$$、$$R=6$$。GSM8K 测试结果如下：

| 方法 | GSM8K 准确率 |
|---|---:|
| 显式 CoT | 71.5% |
| CODI | 60.8% |
| LOTUS | 70.0 ± 0.9% |
| LOTUS + CODI | 70.6 ± 0.2% |

LOTUS 的结果使用三个随机种子。它的准确率接近同设置下的显式 CoT，加入 CODI 蒸馏后进一步缩小了差距。<a class="paper-citation" id="cite-8-3" href="#ref-8" title="Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers；表 1" aria-label="参考文献 8，表 1">[8，表 1]</a>

延迟实验用单张 H100 NVL，batch size 为 1，采用 greedy 解码。在紧凑数学表达式设置中，LOTUS 的思考阶段约快 2.5 倍，计入预填充与答案生成后约快 2 倍。<a class="paper-citation" id="cite-8-4" href="#ref-8" title="Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers；表 2" aria-label="参考文献 8，表 2">[8，表 2]</a>

用较长的自然语言 CoT 训练时，LOTUS 的思考阶段约快 7 倍，准确率接近显式 CoT。<a class="paper-citation" id="cite-8-5" href="#ref-8" title="Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers；表 3" aria-label="参考文献 8，表 3">[8，表 3]</a>

消融显示，工作区架构和过程监督都影响了结果。目前的对照集中在这套数学任务。

### 4.3 LLoCoT：用连续目标训练固定工作区

From Chain-of-Thought to Loops<a class="paper-citation" id="cite-16-1" href="#ref-16" title="From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers" aria-label="参考文献 16">[16]</a> 提出的 LLoCoT 从 $$K$$ 个可学习的初始 slots 出发。这些初值不依赖当前问题；进入共享 Transformer 后，工作区在问题 $$q$$ 条件下更新：

$$
\widetilde H^{(r)}=T_\theta(H^{(r-1)};q),\qquad
(\mu^{(r)},\sigma^{(r)})=G_\theta(\widetilde H^{(r)}).
$$

Gaussian head 为每个 slot 预测 latent thought 的均值与尺度。除最后一轮外，将均值映射回隐藏空间，反馈给下一轮：

$$
H^{(r)}=\widetilde H^{(r)}+P_\theta(\mu^{(r)}).
$$

各 slots 在一次前向中同时更新，但循环之间仍然串行。主配置使用 causal attention；“非自回归 latent reasoning”在这里指不必逐 slot 采样并等待下一次调用，不意味着 attention 必须双向，也不意味着答案生成非自回归。

其连续目标先由冻结 VAE 编码显式 CoT，再经可训练 target-space flow 变换。最终 Gaussian 分布以似然目标学习这些表示，非末轮另有辅助监督；答案训练使用目标 latent 作为条件。测试时才使用模型生成的 latent，因此仍存在目标 latent 与生成 latent 的分布差异。VAE 与 target-space flow 都不在推理时运行，后者也不是 FLaRe 那种需要多步积分的推理过程。<a class="paper-citation" id="cite-16-2" href="#ref-16" title="From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers；§3、附录 A" aria-label="参考文献 16，§3、附录 A">[16，§3、附录 A]</a>

与 LOTUS 相比，监督从逐位置的文本 token 转为连续表示的概率分布。主配置使用 Qwen3-1.7B、64 个 slots 和两轮循环，在 HumanEval、HumanEval+、MBPP、MBPP+ 上的平均分为 63.6，base 为 57.3。no-feedback 消融增加到更多循环后会退化，不能由固定工作区直接推出深度外推能力。

论文在单张 H100 80GB、batch=1、128 个固定代码 prompt 上，分别测量了各阶段的开销：

| 指标 | Reasoning SFT | LLoCoT |
|---|---:|---:|
| 首答案 token 延迟 | 3.17 s | 0.09 s |
| 串行推理/latent 阶段 | 2.47 s | 0.06 s |
| 端到端延迟 | 15.16 s | 11.47 s |
| 解析 FLOPs | $$1.7\times10^{12}$$ | $$2.8\times10^{12}$$ |
| 峰值已分配显存 | 3.3 GiB | 6.3 GiB |

LLoCoT 的首答案 token 约快 36 倍，端到端约快 1.3 倍，FLOPs 和峰值显存却更高。它用更大的并行工作区换掉了串行等待。测量排除了预热、tokenization、采样和网络时间，两种模型的答案长度也可能不同。<a class="paper-citation" id="cite-16-3" href="#ref-16" title="From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers；§4.3、表 4" aria-label="参考文献 16，§4.3、表 4">[16，§4.3、表 4]</a>

### 4.4 FLaRe：生成的 latent 是否被答案使用

FLaRe<a class="paper-citation" id="cite-17-1" href="#ref-17" title="What Matters for Latent Reasoning with Flow Matching" aria-label="参考文献 17">[17]</a> 先将 CoT 压缩到 VAE 空间，再用问题条件下的 flow 模型生成 latent thought。记标准化后的 CoT latent 为 $$z^*$$，则训练路径与速度目标为

$$
z_t=(1-t)\epsilon+tz^*,\qquad
\mathcal L_{\mathrm{FM}}
=\mathbb E\|v_\theta(z_t,t,q)-(z^*-\epsilon)\|_2^2.
$$

FLaRe 的同一模型先生成 latent，再以它为条件自回归回答。其默认设置使用 8 个 512 维 slots 和 20 步 Euler 积分。循环发生在 latent 的生成过程中，不要求把骨干内部不同层也改为共享参数；因此它与 recurrent-depth Transformer 相邻，但不是同一种架构。

训练配方包括符号化 CoT、latent 扰动、答案读出设计，以及用模型自身生成并验证的 thoughts 继续训练。VAE decoder 用于还原 CoT、构造后续训练目标和分析，实际答案路径不必先生成这段文本。<a class="paper-citation" id="cite-17-2" href="#ref-17" title="What Matters for Latent Reasoning with Flow Matching；§2–3" aria-label="参考文献 17，§2–3">[17，§2–3]</a>

论文区分两种正确率：direct accuracy 直接测模型依据 latent 给出的答案；decoded accuracy 先用冻结 VAE decoder 还原 CoT，再读出答案。decoder 可能补全或纠正不完美 latent，后者高于前者时，不能说直接答案路径已经拥有同样能力。

增加采样步数的结果也体现了这个差异。在论文的预算扫描中，两个训练阶段的 decoded accuracy 分别增加 18.2、24.2 个百分点，direct accuracy 最多增加 1.8 个百分点。更强的 decoder 从 latent 中读出了更多答案，原有答案头的收益则小得多。<a class="paper-citation" id="cite-17-3" href="#ref-17" title="What Matters for Latent Reasoning with Flow Matching；§4.4" aria-label="参考文献 17，§4.4">[17，§4.4]</a>

为检查 latent 是否参与计算，作者只修改题目中的一个数字，构造答案不同的 twin question，再把 twin 的 thought 注入原问题，观察答案是否跟随被注入过程；另有清零 thought 与仅保留题目的对照。相比任意噪声破坏，这种替换更接近可解释的中间量干预。结果支持相应算术任务中的 latent 依赖性，但 twin 自身可能答错，原始替换比例与“两题都先答对”的条件比例必须分开。

FLaRe 分别检验了文本还原、答案对 latent 的依赖和预算变化。这些证据限于所测算术模型与训练设置；还原出的 CoT 也未必记录了完整的内部计算。

### 4.5 LoopVL：视觉位置也参与循环更新

<a class="paper-link" href="https://arxiv.org/abs/2609.38426v1" title="LoopVL: Recurrent Visual Intelligence">LoopVL: Recurrent Visual Intelligence</a><a class="paper-citation" id="cite-15-1" href="#ref-15" title="LoopVL: Recurrent Visual Intelligence" aria-label="参考文献 15">[15]</a> 将视觉特征投影到语言空间，与指令一起放入循环骨干。Penguin 视觉编码器使用预训练权重并被冻结；约 1B 的语言骨干则自行进行文本预训练，再经过多模态训练与后训练。

其骨干有两组不共享彼此参数的模块 $$L,H$$，每组包含 16 层 Transformer；各自的多次调用共享参数。默认 H2L3 日程为

$$
[L\rightarrow L\rightarrow L\rightarrow H]\times2.
$$

模型维护相互耦合的两组状态：$$L$$ 在当前 $$H$$ 条件下反复更新，$$H$$ 再整合 $$L$$ 的结果，供下一周期使用。32 个存储层因此执行了 128 次层调用。视觉位置保留图像网格坐标，在这些调用中持续更新；每个 model cycle 开始时，还会通过门控重新注入初始投影视觉表示。<a class="paper-citation" id="cite-15-2" href="#ref-15" title="LoopVL: Recurrent Visual Intelligence；§2.1" aria-label="参考文献 15，§2.1">[15，§2.1]</a>

循环更新的是已有视觉特征在当前问题条件下的表示，无须反复运行视觉编码器。论文将后续周期中的视觉更新丢弃或冻结，并额外训练一个后续周期只读视觉状态的模型；正常更新版本在四项对照评测上均更好。重新训练的只读对照减少了仅由测试时突变引起掉分的解释，支持视觉状态更新具有任务价值，但还没有确定哪一次区域转移完成了哪个推理步骤。<a class="paper-citation" id="cite-15-3" href="#ref-15" title="LoopVL: Recurrent Visual Intelligence；§2.4、图 3" aria-label="参考文献 15，§2.4、图 3">[15，§2.4、图 3]</a>

在同为 0.14T token 的对照中，LoopVL 与非循环 1B 基线的骨干训练计算量分别约为 $$2.47\times10^{21}$$ 和 $$0.92\times10^{21}$$ FLOPs。两个 4B 非循环基线约为 $$2.89\times10^{21}$$、$$2.99\times10^{21}$$ FLOPs，LoopVL 在部分任务上超过它们，在另一些任务上略低。同参数优势包含更多执行深度的作用，不是同计算量比较。<a class="paper-citation" id="cite-15-4" href="#ref-15" title="LoopVL: Recurrent Visual Intelligence；表 1" aria-label="参考文献 15，表 1">[15，表 1]</a>

固定 checkpoint 后，推理日程的外推并不顺利。H2L3 的 MMStar 为 63.47，增加到 H3L3、H4L3 后分别为 58.20、24.00；即使总执行深度相同，H4L1 也只有 24.67。循环次数、$$L/H$$ 调用次序与训练日程共同决定效果。<a class="paper-citation" id="cite-15-5" href="#ref-15" title="LoopVL: Recurrent Visual Intelligence；表 4" aria-label="参考文献 15，表 4">[15，表 4]</a>

作者将周期之间明显的视觉注意力重分配称为 Visual Aha Moments。注意力熵、集中程度和区域热图描述的是读取模式，不能单独确定模型是否产生了新的推理步骤。LoopVL 的状态更新对照与 Shared Weights 的路由替换回答的是不同问题：前者检验更新是否有用，后者检验特定注意力模式能否选择特定转移。

## 5. 状态中的信息怎样进入后续计算

### 5.1 从 probe 到状态干预

把 latent 向量投影回文本，可以检查它与概念或步骤的关联；训练 probe 可以检测答案质量信息。不过，信息能被外部读出，不代表模型的后续计算也在使用它。

要检验因果作用，可以移除或打乱某些状态，或把一条轨迹中的特定中间量替换为另一条轨迹的值，观察后续预测是否按预期变化。对照还须排除信息泄漏和分布外破坏：任意扰乱都可能使模型掉分，并不特指某种推理机制。

<a class="paper-link" href="https://arxiv.org/abs/2607.18553v3" title="Operational Proto-Introspection in Looped Language Models: Process-Quality Taps, Executable Branching, and the Readout-Control Boundary">Operational Proto-Introspection</a><a class="paper-citation" id="cite-10-1" href="#ref-10" title="Operational Proto-Introspection in Looped Language Models: Process-Quality Taps, Executable Branching, and the Readout-Control Boundary" aria-label="参考文献 10">[10]</a> 在 Ouro-RLTT 等模型的轨迹上读取过程质量，回答前的 probe 不接触答案区域或 gold value，并进一步尝试用读出分数选择答案或改变生成过程。

状态分数在部分选择性回答和候选选择实验中带来了收益；方向 steering、等计算量的 loop 分配等生成控制实验却没有检测到相应改善。读出已经能影响某些外部决策，尚未稳定地改善内部生成。非 looped 对照也能复现部分同类读出，可读信号并非全部由 recurrence 产生。

与上述生成控制的阴性结果不同，Shared Weights 在受控任务上训练专门的状态映射，并进一步检验 attention 路由。两者的训练条件、干预目标和任务范围不同。

### 5.2 Shared Weights：状态能否选择下一轮操作

Shared Weights<a class="paper-citation" id="cite-13-1" href="#ref-13" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does" aria-label="参考文献 13">[13]</a> 用图行走检验共享块的计算分工。在模型未经干预的轨迹里，有的循环前进一个节点，有的前进多个节点，还有的停留在已经到达的目标。一个 loop 并不天然对应一个算法步骤。

作者冻结共享 Transformer 块 $$F_\theta$$，在循环边界训练一个逐 token 的映射 $$J$$：

$$
H^{(r+1)}=F_\theta(J(H^{(r)})).
$$

图模型中的 $$J$$ 由对角项、rank-48 低秩项和偏置组成。它本身不在 token 之间汇聚信息，因此图关系的读取与组合仍须由冻结的 $$F_\theta$$ 完成。对同一个进入状态，分别训练的 $$J_{\mathrm{one}}$$ 和 $$J_{\mathrm{two}}$$ 让下一轮选择前进一步或两步；在 D8L6 的一个骨干上，未见图中的目标选择准确率分别达到 99.6% 和 98.2%。<a class="paper-citation" id="cite-13-2" href="#ref-13" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does；§4" aria-label="参考文献 13，§4">[13，§4]</a>

这些映射也能短程组合。连续应用两次一步映射，两步目标的准确率为 93.6%；但第三次应用后降到 56.3%。

可选的操作还受到骨干训练的影响。作者训练了五对初始化、架构、输入和更新预算匹配的图模型，一组只监督最终答案，另一组增加逐步节点监督。随后用相同映射族分别训练一步、两步控制器：逐步监督组的一步准确率为 100%，高于仅终点监督组的 66.5%；两步准确率却是 13.2% 对 50.5%，方向反了。<a class="paper-citation" id="cite-13-3" href="#ref-13" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does；§6、图 7" aria-label="参考文献 13，§6、图 7">[13，§6、图 7]</a>

在这些任务上，过程监督使一步推进更容易被调用，却没有同时改善跨两步的计算。结果支持 $$J$$ 选择骨干已经学到的操作，可调用的操作依赖骨干的训练。状态如何通过 attention 改变操作，还需要进一步的内部干预，见 §5.3。

### 5.3 替换路由，保留被读取的内容

状态映射改变答案后，仍有两种可能：它改变了信息读取的方式，或者直接把有用答案信息写进了状态。Shared Weights<a class="paper-citation" id="cite-13-4" href="#ref-13" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does" aria-label="参考文献 13">[13]</a> 通过区分 attention pattern 与 value 来检验前一种机制。

设两次运行的注意力模式为 $$A_1,A_2$$，values 为 $$V_1,V_2$$。接收方原来的 head 输出是 $$A_2V_2$$，两种替换分别为

$$
\text{pattern patch}:\quad A_2V_2\longrightarrow A_1V_2,
\qquad
\text{output patch}:\quad A_2V_2\longrightarrow A_1V_1.
$$

前者只借用从哪些位置读取信息的模式，内容仍来自接收方。例如来源图中有 $$C\to D$$，接收图同位置有 $$C\to E$$，替换 pattern 后预期读出 $$E$$，替换完整输出则可能带来 $$D$$。作者在位置对齐的图任务和语言任务中观察到了这种区别；在一步与两步控制之间交换 pattern，也能改变被选择的目标。<a class="paper-citation" id="cite-13-5" href="#ref-13" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does；§5" aria-label="参考文献 13，§5">[13，§5]</a>

语言实验使用 Ouro-2.6B：先在 1–4 步信件转交任务上微调，再冻结骨干，在 1–8 步请求上训练状态映射。八步任务的 256 个测试样本中，完整 steering 答对 251 个，仅移植选定 heads 的 steering pattern 答对 248 个，未干预模型答对 0 个。只替换 values、使用无关图的 pattern 或错误循环位置的 pattern 等对照没有恢复效果。<a class="paper-citation" id="cite-13-6" href="#ref-13" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does；§5.2、附录 D" aria-label="参考文献 13，§5.2、附录 D">[13，§5.2、附录 D]</a>

这为进入状态通过注意力路由改变后续计算提供了因果证据。但八步请求已经进入映射的训练范围，结果不是未经训练的深度外推；受控信件任务也不同于开放式数学和代码推理。结合 §5.2 的复用退化，当前证据支持有限任务中的操作选择，尚不支持一个通用、可长程组合的 latent 程序接口。

### 5.4 Jacobian Lens：跨循环的读写窗口

Jacobian Lens<a class="paper-citation" id="cite-21-1" href="#ref-21" title="Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?" aria-label="参考文献 21">[21]</a> 用 virtual unrolling 将面向逐层 Transformer 的 Jacobian lens 扩展到 Ouro-2.6B、Huginn-0125，并以 Qwen3.6-27B 为非循环参照。三者都出现可读出、可因果写入和消融的中层 workspace，但信息跨循环保存的方式不同。

Ouro 在每个 loop 重新构造与 token 位置对齐的工作区内容，线性 transport 不能直接把它带过循环边界，因此写入和消融需要覆盖后续每一轮才有效。Huginn 的原始 latent 传递能跨 16 次 recurrence 保留内容，但读写干预只在约两轮宽的滑动窗口内有效。

论文还区分了“把已有内容说出来”和“把新注入内容说出来”：前者不要求显式深监督，后者在这组模型里只出现在有逐轮监督的 Ouro 上。这与 Shared Weights 的监督对照相呼应，但测量对象不同：一个检验注入内容能否进入输出，一个检验哪些操作容易通过状态映射调用。

Jacobian lens 只观察会影响当前目标的内容，对已保存但暂时不驱动 next token 的信息存在结构性盲区；估计出的容量也受读出能力限制。这些实验揭示了工作区的读写接口，不能给出隐状态中全部信息的清单。

## 6. 读出与状态动态
{: #readout-dynamics }

### 6.1 读出损失看不见的变量

直观上，给每一轮都增加答案或过程损失，似乎就能约束整条 latent trajectory。<a class="paper-link" href="https://arxiv.org/abs/2606.24898v1" title="Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models">Dense Supervision Is Not Enough</a><a class="paper-citation" id="cite-9-1" href="#ref-9" title="Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models" aria-label="参考文献 9">[9]</a> 给出了一个反例：读出可以对某些状态变化不敏感，而这些变化仍然影响下一次循环。

令读出为

$$
p(h)=\operatorname{softmax}(W\operatorname{RMSNorm}(h)).
$$

忽略归一化中的 $$\varepsilon$$，对 $$\alpha>0$$ 有

$$
\operatorname{RMSNorm}(\alpha h)=\operatorname{RMSNorm}(h),
\qquad \mathcal L(\alpha h)=\mathcal L(h).
$$

沿缩放方向求导，得到

$$
\langle\nabla_h\mathcal L,h\rangle=0.
$$

立即读出的交叉熵看不见径向尺度变化，但 pre-norm 残差循环

$$
h_{r+1}=h_r+B_\theta(\operatorname{Norm}(h_r))
$$

仍然把原始 $$h_r$$ 的尺度向后传递。于是，每轮都能预测得不错，状态范数却可能持续漂移。论文在没有 inter-loop normalization 的 44M、129M 模型中检验了这一机制，并对比了尺度可见的读出、显式范数惩罚和循环内归一化等干预。

实际归一化含 $$\varepsilon$$，尺度不变性只近似成立；后续损失还可能通过递归路径传来间接梯度。这个局部机制允许尺度失控，却不意味着所有使用归一化的循环都会发散。

同一个隐状态既供读出头预测，又是下一轮的输入。读出损失约束不到的变化，仍可能影响后续迭代的稳定性。

### 6.2 LoopCD：利用早晚循环的预测差异

<a class="paper-link" href="https://arxiv.org/abs/2610.02185v1" title="Decoding Looped Transformers Better for (Almost) Free">Apple 的 LoopCD</a><a class="paper-citation" id="cite-14-1" href="#ref-14" title="Decoding Looped Transformers Better for (Almost) Free" aria-label="参考文献 14">[14]</a> 不训练状态控制器，而是在解码时利用已经计算出的循环轨迹。设 $$z_R$$ 是末轮 logits，$$z_b$$ 是较早的参考循环经过同一输出网络得到的 logits，则

$$
z'=z_R+\omega(z_R-z_b),\qquad \omega\geq0.
$$

相比参考轮，经过更多计算才获得支持的 token 会被进一步增强，反之则被抑制。它借用了 contrastive decoding 与 diffusion autoguidance 的思路，弱参考来自同一前缀下较少计算的预测，不需要另一套模型或重新训练。<a class="paper-citation" id="cite-14-2" href="#ref-14" title="Decoding Looped Transformers Better for (Almost) Free；§3" aria-label="参考文献 14，§3">[14，§3]</a>

LoopCD-Logits 需要为参考状态额外执行一次输出网络；若骨干包含 post-loop coda，这部分也要重复。LoopCD-Hidden 则先组合状态

$$
h'=h_R+\omega(h_R-h_b),
$$

再执行一次 coda 和词表读出，只增加轻量向量运算。由于输出网络含非线性，两种形式并不等价。参考状态也不必总是第一轮：带高斯初始化的 Huginn，其 Hidden 版本选择经过数轮更新后的状态，以避开初始噪声。<a class="paper-citation" id="cite-14-3" href="#ref-14" title="Decoding Looped Transformers Better for (Almost) Free；附录 B.4、E.1" aria-label="参考文献 14，附录 B.4、E.1">[14，附录 B.4、E.1]</a>

完整深度下，Logits 版本将 Ouro-2.6B-Thinking 的 AIME 2024 pass@1 从 61.88% 提高到 73.33%；Hidden 版本将 Huginn 的 HumanEval 从 22.56% 提高到 31.71%。这说明循环间的差异能影响最终 token 选择，不要求先把状态翻译成可读的 CoT。<a class="paper-citation" id="cite-14-4" href="#ref-14" title="Decoding Looped Transformers Better for (Almost) Free；表 1、3" aria-label="参考文献 14，表 1、3">[14，表 1、3]</a>

论文还在七项选择题平均分上测试了减半循环深度，选定配置可匹配或超过完整深度的无引导基线，解析前向 FLOPs 减少 22.5%–48.2%。这组成本按 512-token prefill 计算，不是长答案生成的实测延迟；引导强度与参考轮也经过筛选，不能假定任意模型、任务和参数都有收益。<a class="paper-citation" id="cite-14-5" href="#ref-14" title="Decoding Looped Transformers Better for (Almost) Free；§4.3、附录 B.4、D" aria-label="参考文献 14，§4.3、附录 B.4、D">[14，§4.3、附录 B.4、D]</a>

LoopCD 证明的是轨迹可以用于改进读出，并未证明 $$h_R-h_b$$ 对应某个逻辑步骤。它也没有训练更好的内部更新规则。状态映射改变冻结块下一轮的输入，LoopCD 改变输出读出；两种干预都能改善特定任务，却不能互相替代机制证据。

另有 POSTECH 的 <a class="paper-link" href="https://arxiv.org/abs/2609.24196v1" title="LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models">LoopCD（POSTECH）</a><a class="paper-citation" id="cite-20-1" href="#ref-20" title="LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models" aria-label="参考文献 20">[20]</a> 同样使用 LoopCD 这个名称，但采用末轮与浅轮对数概率之差及 plausibility constraint。它分析对不稳定 hard tokens 的较强影响，并不显式运行 hard-token 检测。两篇论文需要按作者或编号区分，上面的 Hidden 版本和实验数字属于 Apple 工作。

### 6.3 ImR：平均读出不要求状态停下来

末轮读出 $$g(h_T)$$ 会随停止时刻变化。<a class="paper-link" href="https://arxiv.org/abs/2610.10996v1" title="Invariant-Measure Reasoners: Stable Representations for Latent Reasoning">ImR</a><a class="paper-citation" id="cite-18-1" href="#ref-18" title="Invariant-Measure Reasoners: Stable Representations for Latent Reasoning" aria-label="参考文献 18">[18]</a> 改为对轨迹上的 logits 求平均：

$$
\bar z_T=\frac1{T-B}\sum_{t=B+1}^{T}g(h_t),\qquad T>B,
$$

其中 $$B$$ 为可选的 burn-in 长度。平均的是 logits，不是先平均状态再读出；当 $$g$$ 非线性时，两种操作不等价。

取 $$B=0$$，用轨迹的经验测度表示这个平均：

$$
\widehat\mu_T=\frac1T\sum_{t=1}^T\delta_{h_t},\qquad
\bar z_T=\frac1T\sum_{t=1}^T g(h_t)=\int g\,\mathrm d\widehat\mu_T.
$$

其中 $$\delta_{h_t}$$ 是集中在状态 $$h_t$$ 上的单位点质量。轨迹不必停在一点，仍可能逐渐形成稳定的访问分布。

若轨迹的经验测度弱收敛，且读出满足连续有界等条件，平均 logits 可以收敛。极限还有唯一最大类别时，最终标签才稳定。例如状态在两个点间周期性交替，每一步都不接近固定点，两点读出的时间平均仍可稳定。任意有界轨迹却未必满足所需的测度收敛条件。

ImR 在 maze、Sudoku 中测试了读出头微调与从头训练。它区分了状态收敛和预测稳定，没有证明稳定答案正确，也尚未给出通用 LLM 的结论。与 §6.1 的 readout blind spot 一起，它说明状态动态和立即读出之间既可能存在有害的不可见变化，也可能存在不影响稳定预测的持续运动。

## 7. 验证条件与计算成本

### 7.1 工作区、深度与文本长度的预算

沿着 LOTUS 的设计，可以用 $$N=Kc$$ 个 latent positions 保存中间状态，用共享更新块执行 $$R$$ 轮计算，再按题目选择两种预算。$$N$$ 控制工作区大小，$$R$$ 控制修订深度；简单题和复杂题未必需要同样的组合。

增加 $$N$$ 会扩大状态容量和 attention 成本，增加 $$R$$ 会延长顺序依赖，两者都可能使训练更难。测试时单独改变某项预算，还可能让模型偏离训练分布。

在总 FLOPs、延迟或显存分别受限的条件下，可以比较“多位置、浅循环”与“少位置、深循环”。仅匹配参数量时，更大的 latent 工作区往往也消耗更多计算，表示方式与计算预算的影响会混在一起。

连续 thought 通常仍占序列位置、产生 KV，并需要依次调用骨干；固定工作区也要为多个位置执行 attention。LOTUS 与 LLoCoT 的分阶段测量已经显示，减少串行等待、减少总 FLOPs 和减少峰值显存并不是同一个结果。

逐 token 的循环预算还受训练范围和历史缓存约束。TaH2、ELR 与 Ragged KV 分别处理深度分配、多预算训练和早退后的历史访问，具体机制见[第一篇的深度预算]({% post_url 2026-10-19-loop-transformer-architecture %}#depth-budget)和[第四篇的早退缓存]({% post_url 2026-11-26-loop-transformer-serving %}#adaptive-cache)。这些方法不直接优化 latent 工作区大小，也不能替代位置数与循环深度的联合对照。

### 7.2 过程目标与模型自己的 rollout

第二篇讨论的 Looped Flows<a class="paper-citation" id="cite-12-1" href="#ref-12" title="Thinking with Looped Flows" aria-label="参考文献 12">[12]</a> 用有时间关联的局部去噪目标训练持久状态，无须让中间位置对应文本步骤。它的监督来自污染后的答案，CODI、LOTUS 的过程监督则来自显式 CoT 的表示或 token。

FLaRe 已在 CoT latent 训练后使用自身生成并验证的 thoughts 继续学习，提供了“过程表示与模型 rollout 结合”的实例。但将同样训练直接应用到 LOTUS 式工作区，或只在部分循环保留可读目标，仍是不同方案，不能借用 FLaRe 的结果替代验证。

模型自己的 rollout 尤其适合检验这类训练：从真实答案的污染状态恢复，与纠正自己生成的高置信度错误，面对的是不同分布。前一种任务学得好，后一种仍可能失败。

### 7.3 可监控性与显式检查点

连续状态不直接输出文本，容易引出一个推测：更多内部循环会让显式 CoT 更难监控。<a class="paper-link" href="https://arxiv.org/abs/2610.02741v1" title="On the Chain-of-Thought Monitorability of Looped Language Models">CoT Monitorability</a><a class="paper-citation" id="cite-19-1" href="#ref-19" title="On the Chain-of-Thought Monitorability of Looped Language Models" aria-label="参考文献 19">[19]</a> 对此作了直接测试。

它使用 MonitorBench 的八类任务，检查模型是否在 CoT 中暴露并使用影响答案的提示因素，并加入要求隐藏因素或规避监控的压力测试。同模型增加循环时，部分 Logic/Science/Engineering Cue Answer 任务的可监控性下降；但跨模型比较未发现 LoopLM 系统性地比匹配参数或深度的非循环模型更难监控。

前一结论不能外推到所有任务，后一结论也不是两类架构等价的证明：同模型变深主要在 Ouro 上检验，跨模型比较仍有训练数据和架构的混杂因素。显式 CoT 的长度、可读性和对决策因素的忠实暴露需要分别测量。

可监控性与安全性也不是同一指标：模型可以不暴露决策过程却拒绝危险请求，也可以给出可读过程却仍违反安全要求。SafeBridge 的跨深度安全实验见[第一篇 §5.5]({% post_url 2026-10-19-loop-transformer-architecture %}#safebridge)；若部署允许切换循环预算，默认轮数的安全结果不足以覆盖其他执行路径。

<a class="paper-link" href="https://thinkingmachines.ai/news/inkling-2-mini/" title="Introducing Inkling 2 Mini">Inkling 2 Mini 的可变深度实验</a><a class="paper-citation" id="cite-24-1" href="#ref-24" title="Introducing Inkling 2 Mini" aria-label="参考文献 24">[24]</a> 把对照扩大到了另一套模型。发布版固定运行三轮，配套的 [TML-d24-varloop-research](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) 可以在同一组权重上改轮数。该研究 checkpoint 从一轮增到三轮时，无 CoT 的任务能力时间跨度由 16 秒增至 37 秒。作者还报告，所测串行推理任务在十轮以内继续改善，公开 CoT 变短但仍可读。时间跨度是任务难度的衡量，计算延迟另计。

可读性与 MonitorBench 测到的决策依据暴露程度是两种观测。若模型在增加内部计算后输出一段更短的过程，需要再检查它是否保留了真正决定答案的因素。可变轮数 checkpoint 让这项检查能固定权重，分别改变轮数和是否允许写出 CoT。

模型也可以在内部完成一段修订，再输出可验证的方程、程序状态或工具调用。读者或验证器可以在这些检查点给出反馈。

这种设计会增加输出开销，同时允许外部检查中间对象，可用于需要中间验证的证明或代码任务。简单封闭式问答可能只需最终答案。若检查点来自 latent 摘要，其内容是否忠实于实际计算仍须单独验证；流畅的解释未必是完整的计算记录。

### 7.4 状态、深度与监控的对照

| 要测的量 | 对照设计 |
|---|---|
| 状态对后续计算的影响 | 替换一个有明确含义的中间量，比较预测是否按预期改变。同时做匹配的扰动对照，排除一般性的状态破坏。 |
| 增加一轮的收益与成本 | 逐轮记下改对和改错的答案，再与同 FLOPs 下增加位置、延长 CoT 的方案比较。另报完成时间和峰值显存。 |
| 公开 CoT 中可辨认的决策依据 | 固定权重和任务，改变循环次数，逐项检查影响答案的提示因素在公开 CoT 中是否仍可识别。 |

Inkling 的可变轮数 checkpoint 已经显示，多跑几轮可以让公开 CoT 变短，同时保持可读。固定这组权重，接下来可以逐项检查：思维链缩短后，真正决定答案的线索还剩多少？

> 计算的回路 · [一 · 循环架构]({% post_url 2026-10-19-loop-transformer-architecture %}) / [二 · 去噪与生成]({% post_url 2026-10-19-loop-denoising-flows %}) / **三 · 隐式推理** / [四 · 推理系统]({% post_url 2026-11-26-loop-transformer-serving %})

## 参考文献

1. <span id="ref-1" class="reference-anchor"></span>[Think before you speak: Training Language Models With Pause Tokens](https://arxiv.org/abs/2310.02226v3). arXiv:2310.02226v3，2024-04-21 更新。占位符、训练期延迟与额外计算位置。 <a class="reference-backlink" href="#cite-1-1" aria-label="返回文献 1 首次引用处" title="返回正文">↩</a>
2. <span id="ref-2" class="reference-anchor"></span>[Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769v4). arXiv:2412.06769v4，2026-08-23 更新。Coconut 机制与课程见 §3；逻辑搜索分析见 §4；主结果见表 1。 <a class="reference-backlink" href="#cite-2-1" aria-label="返回文献 2 首次引用处" title="返回正文">↩</a>
3. <span id="ref-3" class="reference-anchor"></span>[Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space](https://arxiv.org/abs/2505.15778v1). arXiv:2505.15778v1，2025-05-21。概率混合 embedding 与 Cold Stop 见 §3。 <a class="reference-backlink" href="#cite-3-1" aria-label="返回文献 3 首次引用处" title="返回正文">↩</a>
4. <span id="ref-4" class="reference-anchor"></span>[CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074v3). arXiv:2502.21074v3，2025-09-23 更新。teacher/student、跨层指定位置蒸馏见 §3，实验见 §4。 <a class="reference-backlink" href="#cite-4-1" aria-label="返回文献 4 首次引用处" title="返回正文">↩</a>
5. <span id="ref-5" class="reference-anchor"></span>[Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416v1). arXiv:2502.17416v1，2025-02-24。CoT 模拟的条件与构造见 Theorem 5.4、附录 B.3。 <a class="reference-backlink" href="#cite-5-1" aria-label="返回文献 5 首次引用处" title="返回正文">↩</a>
6. <span id="ref-6" class="reference-anchor"></span>[Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171v2). arXiv:2502.05171v2，2025-02-17 更新。递归架构与训练见 §3，3.5B 模型与评测见后续实验章节。 <a class="reference-backlink" href="#cite-6-1" aria-label="返回文献 6 首次引用处" title="返回正文">↩</a>
7. <span id="ref-7" class="reference-anchor"></span>[Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741v5). arXiv:2510.25741v5，2026-07-01 更新。Ouro 的预训练与深度分配。 <a class="reference-backlink" href="#cite-7-1" aria-label="返回文献 7 首次引用处" title="返回正文">↩</a>
8. <span id="ref-8" class="reference-anchor"></span>[Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779v2). arXiv:2606.31779v2，2026-07-13 更新。LOTUS 机制见 §3，精度与分阶段延迟见表 1–3，硬件与解码条件见 §4.2 脚注。 <a class="reference-backlink" href="#cite-8-1" aria-label="返回文献 8 首次引用处" title="返回正文">↩</a>
9. <span id="ref-9" class="reference-anchor"></span>[Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models](https://arxiv.org/abs/2606.24898v1). arXiv:2606.24898v1，2026-06-12。读出尺度不变性与递归尺度动态见 §2。 <a class="reference-backlink" href="#cite-9-1" aria-label="返回文献 9 首次引用处" title="返回正文">↩</a>
10. <span id="ref-10" class="reference-anchor"></span>[Operational Proto-Introspection in Looped Language Models: Process-Quality Taps, Executable Branching, and the Readout-Control Boundary](https://arxiv.org/abs/2607.18553v3). arXiv:2607.18553v3，2026-07-28 更新。回答前读出、决策用途与生成控制的证据边界。 <a class="reference-backlink" href="#cite-10-1" aria-label="返回文献 10 首次引用处" title="返回正文">↩</a>
11. <span id="ref-11" class="reference-anchor"></span>[Full-bandwidth transformer](https://arxiv.org/abs/2608.08888v1). arXiv:2608.08888v1，2026-08-09。token 间的顶层隐状态反馈，以及它与既有 KV 记忆的区别。 <a class="reference-backlink" href="#cite-11-1" aria-label="返回文献 11 首次引用处" title="返回正文">↩</a>
12. <span id="ref-12" class="reference-anchor"></span>[Thinking with Looped Flows](https://arxiv.org/abs/2609.11801v1). arXiv:2609.11801v1，2026-09-10。局部去噪监督与持久递归状态；详细分析见本系列第二篇。 <a class="reference-backlink" href="#cite-12-1" aria-label="返回文献 12 首次引用处" title="返回正文">↩</a>
13. <span id="ref-13" class="reference-anchor"></span>[Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does](https://arxiv.org/abs/2609.39892v1). arXiv:2609.39892v1，2026-09-30。图任务控制与复用见 §4；attention patching 与 Ouro 信件任务见 §5、附录 D；监督对照见 §6。 <a class="reference-backlink" href="#cite-13-1" aria-label="返回文献 13 首次引用处" title="返回正文">↩</a>
14. <span id="ref-14" class="reference-anchor"></span>[Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/abs/2610.02185v1). arXiv:2610.02185v1，2026-10-01。LoopCD 两种引导空间见 §3；任务结果见表 1–3；参考轮、超参数与解析 FLOPs 见附录 B.4、D、E。 <a class="reference-backlink" href="#cite-14-1" aria-label="返回文献 14 首次引用处" title="返回正文">↩</a>
15. <span id="ref-15" class="reference-anchor"></span>[LoopVL: Recurrent Visual Intelligence](https://arxiv.org/abs/2609.38426v1). arXiv:2609.38426v1，2026-09-29。架构与视觉状态干预见 §2；计算预算见表 1；训练日程与推理日程对照见表 3–4；注意力诊断见 §5.4–5.7。 <a class="reference-backlink" href="#cite-15-1" aria-label="返回文献 15 首次引用处" title="返回正文">↩</a>
16. <span id="ref-16" class="reference-anchor"></span>[From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers](https://arxiv.org/abs/2610.11472v1). arXiv:2610.11472v1，2026-10-08。LLoCoT 架构与目标见 §3、附录 A，效率口径见 §4.3、表 4。 <a class="reference-backlink" href="#cite-16-1" aria-label="返回文献 16 首次引用处" title="返回正文">↩</a>
17. <span id="ref-17" class="reference-anchor"></span>[What Matters for Latent Reasoning with Flow Matching](https://arxiv.org/abs/2610.06666v1). arXiv:2610.06666v1，2026-10-05。FLaRe 训练见 §2–3，latent 用途、读出与计算预算干预见 §4、附录 E。 <a class="reference-backlink" href="#cite-17-1" aria-label="返回文献 17 首次引用处" title="返回正文">↩</a>
18. <span id="ref-18" class="reference-anchor"></span>[Invariant-Measure Reasoners: Stable Representations for Latent Reasoning](https://arxiv.org/abs/2610.10996v1). arXiv:2610.10996v1，2026-10-07。经验测度与平均读出的条件见 §3。 <a class="reference-backlink" href="#cite-18-1" aria-label="返回文献 18 首次引用处" title="返回正文">↩</a>
19. <span id="ref-19" class="reference-anchor"></span>[On the Chain-of-Thought Monitorability of Looped Language Models](https://arxiv.org/abs/2610.02741v1). arXiv:2610.02741v1，2026-10-02。MonitorBench 子任务、压力测试和匹配模型比较。 <a class="reference-backlink" href="#cite-19-1" aria-label="返回文献 19 首次引用处" title="返回正文">↩</a>
20. <span id="ref-20" class="reference-anchor"></span>[LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models](https://arxiv.org/abs/2609.24196v1). arXiv:2609.24196v1，2026-09-21。POSTECH LoopCD；与文献 14 分别引用。 <a class="reference-backlink" href="#cite-20-1" aria-label="返回文献 20 首次引用处" title="返回正文">↩</a>
21. <span id="ref-21" class="reference-anchor"></span>Wenlong Wang and Fergal Reid. [Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?](https://arxiv.org/abs/2609.01924). arXiv:2609.01924，2026。工作区读写、跨循环保存与读出边界。 <a class="reference-backlink" href="#cite-21-1" aria-label="返回文献 21 首次引用处" title="返回正文">↩</a>
22. <span id="ref-22" class="reference-anchor"></span>Michael C. Mozer et al. [Recirculation](https://arxiv.org/abs/2608.17981). arXiv:2608.17981，2026。跨 input step 的深层状态回流及其预填充成本。 <a class="reference-backlink" href="#cite-22-1" aria-label="返回文献 22 首次引用处" title="返回正文">↩</a>
23. <span id="ref-23" class="reference-anchor"></span>Yuxuan Zhou et al. [ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction](https://arxiv.org/abs/2610.24766v1). arXiv:2610.24766v1，2026-10-17。预测范围递归与同位置深度循环的区别见 §2.3、§3.1。 <a class="reference-backlink" href="#cite-23-1" aria-label="返回文献 23 首次引用处" title="返回正文">↩</a>
24. <span id="ref-24" class="reference-anchor"></span>[Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/)。参见 Monitorability of looped transformers，区分固定三轮发布版与可变深度实验。 <a class="reference-backlink" href="#cite-24-1" aria-label="返回文献 24 首次引用处" title="返回正文">↩</a>
{: .references }
