---
layout: post
title: "计算的回路（一）：Loop Transformer 的架构、训练与扩展规律"
date: 2026-10-19
last_modified_at: 2026-11-26
permalink: /blog/loop-transformer-architecture/
visibility: unlisted
sitemap: false
series: 计算的回路
series_description: 循环模型、迭代生成、隐式推理与推理系统
series_order: 1
lang: zh-CN
tags: [Loop Transformer, Latent Reasoning, Recurrent Depth, Scaling]
description: "增加独立层和重复执行共享层，怎样改变模型深度、训练预算和推理开销？"
excerpt: "增加独立层和重复执行共享层，怎样改变模型深度、训练预算和推理开销？"
---

> 计算的回路 · **一 · 循环架构** / [二 · 去噪与生成]({% post_url 2026-10-19-loop-denoising-flows %}) / [三 · 隐式推理]({% post_url 2026-10-19-loop-latent-reasoning %}) / [四 · 推理系统]({% post_url 2026-11-26-loop-transformer-serving %})

要让 Transformer 在预测一个 token 前多做几步计算，最直接的办法是增加网络层数。在宽度和层结构相同的情况下，每加一层，隐状态就多经过一次变换，模型也多出一层独立权重。沿这个方向扩展，执行深度会和参数量一起增加。

也可以让已有的几层多运行一遍。第一遍的输出送回同一组层，第二遍从更新后的隐状态接着算。同一个 token 因此能在模型内部经历更多层变换，模型只需保存一份共享层的权重。这就是 Loop Transformer 的基本做法。

多执行一轮仍要付出计算开销。固定参数量时，可以问这些额外计算能把任务表现提高多少。固定训练或推理算力时，则要比较同一份预算用在共享层还是独立层上更划算。

## 1. 循环复用了哪一段网络
{: #1-共享的是参数增加的是执行深度 }

### 1.1 基本定义

普通 decoder-only Transformer 可写为：

$$
h_0 = \mathrm{Emb}(x),\quad
h_L = F_L \circ \cdots \circ F_2 \circ F_1(h_0),\quad
y = \mathrm{LMHead}(h_L).
$$

其中每层 `F_i` 有独立参数。Looped Transformer 则把某个共享函数或共享层栈 `F` 反复应用：

$$
h_0 = \mathrm{Emb}(x),\quad
h_t = F(h_{t-1}),\ t=1,\ldots,R,\quad
y = \mathrm{LMHead}(h_R).
$$

如果 `F` 是 `k` 层 Transformer block，那么 `(k ⊗ R)` 表示 `k` 个物理层循环 `R` 次，有效深度约为 `kR`，但存储参数只对应 `k` 层。

![Loop Transformer 方案演进总览]({{ '/assets/img/blog/loop-series/loop_transformer_evolution_overview.png' | relative_url }})

*图 1：参数共享、循环深度、训练稳定性与系统预算之间的关系。*

### 1.2 循环发生在哪里

深度循环可以重复整栈、某段中间层，也可以让每个物理层局部重复。循环单元不同，参数访问、状态传递和并行执行方式都会改变。

| 结构 | 重复计算的对象 | 主要区别 |
|---|---|---|
| Model-loop | 整个共享层栈 | 同一组状态反复经过完整 core |
| Layer-loop | 每个物理层局部重复 | 改变 MoE 路由和执行局部性，<a class="paper-link" href="https://arxiv.org/abs/2607.16051" title="Loop the Loopies!">Loopie</a> 采用 |
| Middle/selective loop | 中间层或选定层段 | 前置、后置网络只执行一次 |
| Mixer-only loop | 状态混合算子 | 可以不重复 dense FFN，<a class="paper-link" href="https://arxiv.org/abs/2608.18230" title="Allocating Recurrent Compute in Looped Language Models">MixerLoop</a> 采用 |
| Token-level recursion | 仅继续计算的 tokens | 路由与缓存必须支持不同退出深度 |
| Parallel loop | 带位置偏移的多轮执行 | 减少同 token 依赖，但改变跨轮位置关系 |
| Token-step feedback | 前一生成步的深层状态 | 沿序列反馈，不等于同 token 的深度循环 |
| Horizon recursion | 用共享末层递归预测更远的 token | <a class="paper-link" href="https://arxiv.org/abs/2610.24766v1" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction">ECHO</a><a class="paper-citation" id="cite-81-1" href="#ref-81" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction" aria-label="参考文献 81">[81]</a> 沿预测范围推进，每步的目标位置改变 |

Universal Transformer 和 ALBERT 式共享是相关背景，但参数共享本身不保证模型学会迭代求解。<a class="paper-link" href="https://arxiv.org/abs/2608.08888" title="Full-bandwidth Transformer">Full-bandwidth Transformer</a>、<a class="paper-link" href="https://arxiv.org/abs/2608.17981" title="Recirculation">Recirculation</a> 沿 token 或 depth-step 回流状态，也不能与纯 recurrent depth 合并计算有效层数。它们的反馈机制见第三篇。

边界连接也是架构的一部分。<a class="paper-link" href="https://arxiv.org/abs/2604.21254" title="Hyperloop Transformers">Hyperloop</a><a class="paper-citation" id="cite-1-1" href="#ref-1" title="Hyperloop Transformers" aria-label="参考文献 1">[1]</a> 将 residual stream 扩为多个 parallel lanes，在循环边界读写和混合；其 looped 实验中，每轮边界使用 hyper-connections 比每层都加更有效。循环边界和普通非共享层间边界不必采用同样的信息通路。

### 1.3 Ouro 与 MoR：固定深度和 token 级递归

<a class="paper-link" href="https://arxiv.org/abs/2510.25741" title="Scaling Latent Reasoning via Looped Language Models">Ouro</a><a class="paper-citation" id="cite-2-1" href="#ref-2" title="Scaling Latent Reasoning via Looped Language Models" aria-label="参考文献 2">[2]</a> 提供 1.4B、2.6B 参数的 LoopLM 基座，以共享层栈增加每个 token 的有效计算深度。它强调循环对知识操作、组合与多步推理的作用，后续 <a class="paper-link" href="https://arxiv.org/abs/2602.10520" title="Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models">RLTT</a>、<a class="paper-link" href="https://arxiv.org/abs/2603.19714" title="LoopRPT: Reinforcement Pre-Training for Looped Language Models">LoopRPT</a>、<a class="paper-link" href="https://arxiv.org/abs/2605.07721" title="Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models">MELT</a>、<a class="paper-link" href="https://arxiv.org/abs/2605.16343" title="LoopQ: Quantization for Recursive Transformers">LoopQ</a> 等工作都在这一模型族上实验。这些后续结果并非彼此独立的模型族验证。

![Ouro / LoopLM 架构示意]({{ '/assets/img/blog/loop-series/ouro_looplm_architecture.png' | relative_url }})

*图 2：输入处理、共享递归层栈和输出读出构成基本的 LoopLM。*

<a class="paper-link" href="https://arxiv.org/abs/2507.10524" title="Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation">Mixture-of-Recursions（MoR）</a><a class="paper-citation" id="cite-3-1" href="#ref-3" title="Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation" aria-label="参考文献 3">[3]</a> 则让 router 在 token 粒度决定深度。expert-choice 每轮选 top-k tokens，负载容易控制，但训练要防止 causality leakage；token-choice 由各 token 选择递归深度，另需处理负载均衡。路由同时决定训练路径、推理计算和 KV 布局。

MoR 的实验覆盖 135M—1.7B。逐 token 分配深度胜过部分固定轮数的对照，但在 1.7B 上，普通 Transformer 仍可能略强。早退后，深轮怎样读取历史 KV，见[第四篇]({% post_url 2026-11-26-loop-transformer-serving %}#adaptive-cache)。

<a class="paper-link" href="https://arxiv.org/abs/2610.23036" title="Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design">Towards Looped Models Done Right I</a><a class="paper-citation" id="cite-87-1" href="#ref-87" title="Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design" aria-label="参考文献 87">[87]</a> 用同一组参数规模、执行深度和训练 token 拆开比较循环位置、输入注入和状态组织。在 730M、336B tokens 的对照中，完整的 Huginn 式设计在十项评测都优于所比较的 Ouro 式版本。受控消融把主要收益指向首尾各执行一次的结构和输入注入，随机初始化与双状态组织的结果较不一致。MoE 实验也观察到拓扑收益。两种结构使用论文统一的训练配方。

### 1.4 近期模型怎样安排循环

中间层复用已经用到了从数十亿到数千亿参数的模型。下表横向比较每个 token 经过的层数。能调节轮数的 checkpoint，还可以用同一组权重测出增加一轮的边际收益。

| 模型 | 权重与每 token 的层执行 | 训练或部署选择 |
|---|---|---|
| <a class="paper-link" href="https://www.zhipuai.cn/zh/research/164" title="GLM-5.4">GLM-5.4</a><a class="paper-citation" id="cite-83-1" href="#ref-83" title="GLM-5.4" aria-label="参考文献 83">[83]</a> | 750B 级基座，中间 28 层执行两轮。官方已开放 Coding Plan，公告把权重列为后续发布计划。 | 从已完成预训练的 GLM-5 改造。逐 token 门控控制候选状态的写入幅度，第二轮复用首轮的稀疏历史和检索索引，并另存本轮滑窗 KV。 |
| <a class="paper-link" href="https://huggingface.co/sensenova/SenseNova-Loop-9B" title="SenseNova-Loop-9B">SenseNova-Loop-9B</a><a class="paper-citation" id="cite-84-1" href="#ref-84" title="SenseNova-Loop-9B" aria-label="参考文献 84">[84]</a> | 约 9B 开放权重，Apache 2.0。前后各 4 层执行一次，中间 24 层执行两次。32 个物理层合计执行 56 次。 | 两路 mHC、线性与全注意力交替，支持文本和图像输入。模型卡另列约 14.6B 推理等效参数。 |
| <a class="paper-link" href="https://thinkingmachines.ai/news/inkling-2-mini/" title="Inkling 2 Mini">Inkling 2 Mini</a><a class="paper-citation" id="cite-85-1" href="#ref-85" title="Inkling 2 Mini" aria-label="参考文献 85">[85]</a> | 26.4B 总参数、3.4B 活跃参数。24 层中的中间 12 层执行三轮，每 token 共执行 48 层，权重激活量约 5.4B。Apache 2.0。 | 发布版固定三轮。另有同权重可改轮数的研究 checkpoint，适合评测增加深度的边际收益。 |
| <a class="paper-link" href="https://arxiv.org/abs/2611.03998" title="Dory1.0-4B: A Looped Normalized-GPT">NVIDIA Dory1.0-4B</a><a class="paper-citation" id="cite-86-1" href="#ref-86" title="Dory1.0-4B: A Looped Normalized-GPT" aria-label="参考文献 86">[86]</a> | 4B 开放权重，预训练 6T tokens，推理时可改轮数。 | 在 nGPT 骨干上训练弹性深度，历史 token 的循环 KV 只保留末轮。 |

## 2. 循环能替代多少独立参数

### 2.1 参数匹配与计算匹配

$$k$$ 个物理层循环 $$R$$ 次，有效深度约为 $$kR$$，但若训练时同样展开，循环部分的计算也近似增加 $$R$$ 倍。同参数对照能衡量存储效率，却不能单独说明训练效率。

<a class="paper-link" href="https://www.preprints.org/manuscript/202610.1404" title="Looped Language Models: A Survey and Meta-Analysis of Recurrent-Depth Transformers">Azizi 等人的 survey</a><a class="paper-citation" id="cite-82-1" href="#ref-82" title="Looped Language Models: A Survey and Meta-Analysis of Recurrent-Depth Transformers" aria-label="参考文献 82">[82]</a> 把参数匹配、训练 FLOPs 匹配和推理 FLOPs 匹配分开统计。同参数对照能显示多轮计算带来的收益。要和加深、加宽或增加数据比较，还需要控制训练预算。部署时再单独测达到同等任务质量所需的计算和时延。

训练时，共享参数能省下权重、梯度和优化器状态的存储。每次共享层调用仍要执行前向和反向，完整展开的中间激活也不会因权重共享而自动合并。循环模型可能因此容纳更大的 microbatch，却仍执行接近原有有效深度的计算；是否缩短训练时间，取决于节省的显存能否改善吞吐、通信和硬件利用率。Loopie 的训练收益正包含了这部分配置变化。

### 2.2 一次 recurrence 的等效参数价值

<a class="paper-link" href="https://arxiv.org/abs/2604.21106" title="How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models">How Much Is One Recurrence Worth?</a><a class="paper-citation" id="cite-4-1" href="#ref-4" title="How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models" aria-label="参考文献 4">[4]</a> 提出 iso-depth scaling law。它固定有效深度为 20 层，在 `r = 1, 2, 4, 8` 的循环次数之间比较，估计重复计算相当于多少独立参数容量。

它提出 recurrence-equivalence exponent `φ`：

$$
N_{\mathrm{eff}} = N_{\mathrm{once}} + r^\phi N_{\mathrm{rec}}.
$$

其中 `N_once` 与 `N_rec` 分别表示非循环部分和共享循环部分的参数量。`φ=1` 时，循环部分的等效容量随执行次数线性增加；`φ=0` 时，增加循环不再提高这一拟合容量。

在论文的基线训练配置下，`φ` 约为 0.46；引入 hyperconnections 后，提升到约 0.65。两种配置中，等效容量都随循环次数增加，但增长慢于独立参数的线性增加。这个指数依赖训练配置与信息流设计，不是循环架构的固定常数。

### 2.3 Loopie：MoE 配置与训练吞吐

Loopie<a class="paper-citation" id="cite-5-1" href="#ref-5" title="Loop the Loopies!" aria-label="参考文献 5">[5]</a> 比较局部重复与整栈循环，并把训练吞吐纳入模型配置。

传统 LoopLM 多采用 model-loop：`Layer1 -> Layer2 -> ... -> LayerL -> Layer1 -> Layer2 -> ... -> LayerL`。

Loopie 采用 layer-loop：`Layer1 -> Layer1 -> Layer2 -> Layer2 -> ... -> LayerL -> LayerL`。

![Loopie 中 model-loop 与 layer-loop 的对比]({{ '/assets/img/blog/loop-series/loopie_layer_vs_model_loop.png' | relative_url }})



*图 3：Loopie 把循环粒度从整栈 model-loop 改成局部 layer-loop，以更好匹配 MoE routing、训练吞吐和执行 locality。*

相比跨全模型循环，layer-loop 在同一层内重复计算，执行更局部，流水线也更简单。在作者的 MoE 实验中，它还有助于减少循环中 expert routing 的不稳定；训练初期可能落后于 model-loop，但到约 1.2T token 后超过了后者。

Loopie 同时调整存储宽度、存储层数与执行深度，并将显存和吞吐变化用于模型配置：

1. 从 Qwen3-like MoE reference 出发。
2. 将 stored layers 减半。
3. 每个 stored layer 运行两次。
4. 利用 activation memory 降低带来的 microbatch headroom，将 per-device microbatch 翻倍。
5. 把实际训练吞吐收益再投入到模型容量中。

因此，这组比较并非严格匹配理论 FLOPs，而是在相同硬件、相同 token budget、近似相同 optimizer-step wall-clock time 下比较模型质量。吞吐与 microbatch 的变化也是训练收益的一部分。

### 2.4 SMELT：同时匹配 FLOPs、参数与 KV

<a class="paper-link" href="https://arxiv.org/abs/2609.01343" title="SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers">SMELT</a> 把每 token FLOPs、非嵌入参数量和 KV 缓存都对齐到非循环基线，再比较共享层能带来多少收益。

多跑一轮会增加 FLOPs。SMELT 收窄隐藏层来抵消计算，再增加专家数补回参数量，并调整注意力头维度和 GQA 配比对齐 KV。选定的配置让中间一半的层跑两轮，循环残差按 1/2 缩放。

在这些对照里，中间层循环优于整栈循环。匹配 FLOPs 后，继续加深会迫使模型变窄，两轮优于三轮或四轮。作者把选定架构扩到最高 54B 非嵌入参数，并分别拟合循环模型与基线的训练扩展曲线。<a class="paper-citation" id="cite-6-1" href="#ref-6" title="SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers" aria-label="参考文献 6">[6]</a>

按拟合的最优训练曲线，SMELT 达到同等损失约省 7%–18% 训练 FLOPs。下游提升在代码上最大，也随样本长度和上下文示例数量增加而扩大。

第二轮改写 value 和输出的幅度比 query、key 更明显，注意力也更少集中在段首。这些观测可以帮助解释长样本上的收益。

循环范围和轮数主要在 200M 规模上选定。论文按算术 FLOPs 匹配训练预算，实际用时还取决于吞吐。

<details markdown="1">
<summary>SMELT 的配对评测</summary>

| 指标 | 循环模型的结果 |
|---|---:|
| 等损失所需训练 FLOPs 降幅，按拟合曲线 | 6.8%–18.0% |
| DCLM Completion 胜出的配对 | 96/96 |
| DCLM Core 胜出的配对 | 83/96 |
| MMLU 胜出的配对，基线明显高于随机 | 29/30 |

</details>

### 2.5 稀疏度与循环收益的饱和

<a class="paper-link" href="https://arxiv.org/abs/2605.09165" title="Sparse Layers are Critical to Scaling Looped Language Models">Sparse Layers</a><a class="paper-citation" id="cite-7-1" href="#ref-7" title="Sparse Layers are Critical to Scaling Looped Language Models" aria-label="参考文献 7">[7]</a> 在 isoFLOP 比较中发现，dense looping 容易损失容量，而不同循环激活不同专家，可以部分补偿共享参数的限制。这不保证任意 MoE 循环都优于 dense 模型；循环范围和每次激活量仍影响最优配置。

早先的等效参数映射常按循环次数线性或幂律增长。<a class="paper-link" href="https://arxiv.org/abs/2609.40316v1" title="Scaling Laws for Looped Mixture of Experts">Looped MoE Scaling Laws</a><a class="paper-citation" id="cite-8-1" href="#ref-8" title="Scaling Laws for Looped Mixture of Experts" aria-label="参考文献 8">[8]</a> 改用有界形式，并让曲线依赖稀疏度：

$$
N_{\mathrm{eff}}(R,m)=N_{\mathrm{act}}+
\kappa_1(m)N_{\mathrm{loop}}\left(1-e^{-(R-1)/\kappa_2(m)}\right),
\qquad \kappa_j(m)=\kappa_jm^{-\theta}.
$$

这里 $$m=N_{\mathrm{act}}/N_{\mathrm{total}}$$，$$N_{\mathrm{loop}}$$ 是循环部分的激活参数规模。固定 $$m$$ 时，多一轮带来的等效参数增量逐渐减小；更稀疏的配置提高拟合上限，也延长收益尚未饱和的区间。$$N_{\mathrm{eff}}$$ 描述 loss 的拟合容量，不等于决定实际 FLOPs 的展开参数量。

在等训练算力对照中，较小的 LoopMoE 跑四到五轮，BBH 和 GSM8K 的平均表现接近较大的 MoE。它的十四项总平均仍落后，推理计算约为大模型的 1.5–1.8 倍。<a class="paper-citation" id="cite-8-2" href="#ref-8" title="Scaling Laws for Looped Mixture of Experts；§3–4、表 2" aria-label="参考文献 8，§3–4、表 2">[8，§3–4、表 2]</a>

这条曲线拟合的是特定模型族的容量饱和。论文估计部署内存时只统计权重，KV 和其他运行时状态需要另计。

### 2.6 增长日程与 boundary operator

<a class="paper-link" href="https://arxiv.org/abs/2609.19107" title="How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents">Model Growth</a><a class="paper-citation" id="cite-9-1" href="#ref-9" title="How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents" aria-label="参考文献 9">[9]</a> 把 loop 放进 model growth 的框架：训练到中途再增加 core 的执行次数。新增 pass 可以继续共享同一组权重，也可以复制后解除共享；两者都增加 executed depth，但只有后者增加存储参数。论文还单独测试 `Norm(h) + αe` 形式的 boundary operator，用它分离“递归增长”和“跨 block 重新注入早期表示”的贡献。

只遍历数据一遍时，作者拟合的曲线显示，中途加深改善了训练损失的扩展指数。解除共享的增长方案收益更大，单独加边界连接也有用。数据固定、训练多遍时，随着计算预算增加，达到最低损失的配置倾向于用更多轮。

<details markdown="1">
<summary>Model Growth 的比较口径</summary>

| 对照 | 论文报告的结果 |
|---|---|
| 同训练栈，预算约 10^20 FLOPs | 解除共享、继续共享、只加边界连接的等损失计算倍率分别约 1.55×、1.36×、1.25×。 |
| 固定 100M 个语料 token，训练 10 遍 | 调整轮数达到经过 weight decay 调参的非循环基线最优损失，所需计算约为基线的 1/2.2。 |
| 7.4B growth model，拟合区间外 | CORE 与 GPT-3 13B 相当，约用对方 1/20 的预训练计算。同训练栈的等损失对照在 $$1.2\times10^{21}$$ FLOPs 处约有 1.8× 的计算优势。 |

这里的扩展指数来自训练损失曲线 $$L(C)=L_\infty+aC^{-\alpha}$$ 中的 $$\alpha$$。[TaH2](https://arxiv.org/abs/2609.35748v1) 测的是推理准确率随 $$\log_2 C_{\mathrm{decode}}$$ 变化的斜率。

</details>

### 2.7 LOOM 与 Foil：更多循环的专家结构

<a class="paper-link" href="https://arxiv.org/abs/2610.01153v1" title="Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts">LOOM</a><a class="paper-citation" id="cite-10-1" href="#ref-10" title="Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts" aria-label="参考文献 10">[10]</a> 结合 residual scaling、每轮输入重注入、独立的 per-loop router 和传递历史输出的 Looping Residual，让 100M—1.7B MoE 在更多循环下稳定训练。它的近似等 FLOPs 实验与固定专家配置的加深实验需要分开。

LOOM 的近似等 FLOPs 实验在 700M 模型上把一轮增至五轮，perplexity 下降，平均准确率小幅上升。多跑的轮次主要靠每次少激活专家来补偿，top-k 从 26 降到 4。固定专家配置的实验能从更深循环获得更多改善，同时需要更多计算。<a class="paper-citation" id="cite-10-2" href="#ref-10" title="Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts；表 2–4" aria-label="参考文献 10，表 2–4">[10，表 2–4]</a>

<a class="paper-link" href="https://arxiv.org/abs/2609.35751v1" title="How to Loop MoE: Flatten the Experts, Untie the Attention">Foil</a><a class="paper-citation" id="cite-11-1" href="#ref-11" title="How to Loop MoE: Flatten the Experts, Untie the Attention" aria-label="参考文献 11">[11]</a> 研究共享范围。把专家数、专家层数、循环次数记为 $$(E,D,L)$$，变换

$$
(E,D,L)\longrightarrow(aE,D/a,aL)
$$

保持专家参数量和主要层调用次数，却让每次路由面对更大的专家池。Foil 共享专家与 router，attention 在各轮保留独立参数；这与把整块全部共享的设计不同。

Foil 在约 550M 参数的实验里把验证损失降低了约 0.01 nat。主计算量接近，但更大的专家池会增加 router 打分开销。<a class="paper-citation" id="cite-11-2" href="#ref-11" title="How to Loop MoE: Flatten the Experts, Untie the Attention；主实验、附录 A" aria-label="参考文献 11，主实验、附录 A">[11，主实验、附录 A]</a>

LOOM 分开各轮 router，Foil 分开各轮 attention。两者都在共享参数的同时，给不同轮次保留选择计算的余地。

## 3. 共享更新怎样稳定训练

反复访问同一组参数，会改变残差、梯度和激活尺度的相关性。训练稳定、测试时深度外推与答案正确是不同要求，不能由其中一个推出另外两个。

### 3.1 Parcae：受约束的递归动态

<a class="paper-link" href="https://arxiv.org/abs/2604.12946" title="Parcae: Scaling Laws for Stable Looped Language Models">Parcae</a><a class="paper-citation" id="cite-12-1" href="#ref-12" title="Parcae: Scaling Laws for Stable Looped Language Models" aria-label="参考文献 12">[12]</a> 将循环模型视为离散动力系统，通过约束注入参数控制递归动态的谱性质，并按序列采样深度，降低可变深度带来的训练方差。

在 RDM 和 Transformer 设置中，这套设计比此前的循环模型更稳定。论文还拟合了训练循环数与数据量共同扩展的 scaling law，并测量测试时增加循环的饱和曲线。

### 3.2 Fully Looped Transformer：缩短跨轮信息路径

<a class="paper-link" href="https://arxiv.org/abs/2605.18797" title="Simply Stabilizing the Loop via Fully Looped Transformer">Fully Looped Transformer</a><a class="paper-citation" id="cite-13-1" href="#ref-13" title="Simply Stabilizing the Loop via Fully Looped Transformer" aria-label="参考文献 13">[13]</a> 将普通循环架构的训练困难归因于残差状态爆炸、梯度振荡，以及跨轮信息传递路径过长：前一轮的信息只能从当前轮第一层进入，深层需要经过多次变换才能访问它。

它提出 Fully Looped Architecture，让上一轮 loop 的输出可被当前轮每一层访问；并用 Attention Injection 而非简单 residual addition 注入前一轮状态，以减少无界残差流。

在高循环次数的训练中，普通 LT 更容易崩溃，Fully Looped Architecture 与 Attention Injection 的组合则更稳定。增加推理轮数在训练覆盖的范围内可能提升性能，超过该范围后仍不稳定。

![Fully Looped Transformer 与 Attention Injection]({{ '/assets/img/blog/loop-series/fully_looped_transformer_attention_injection.png' | relative_url }})


*图 4：Fully Looped Transformer 通过让当前轮各层访问上一轮状态，并用 Attention Injection 控制注入方式，缓解简单 residual 注入带来的不稳定。*

### 3.3 DeepLoop：共享访问改变 residual scaling

<a class="paper-link" href="https://arxiv.org/abs/2607.13491" title="DeepLoop: Depth Scaling for Looped Transformers">DeepLoop</a><a class="paper-citation" id="cite-14-1" href="#ref-14" title="DeepLoop: Depth Scaling for Looped Transformers" aria-label="参考文献 14">[14]</a> 分析了残差缩放中的相关性问题：普通深层 Transformer 的 residual scaling 假设不同层参数相对独立，但 looped Transformer 中同一 residual branch 会被多次访问，误差具有相关性。

论文提出 tied-depth effect，并推导出 loop-aware exponent。其结论是：

- 对 decorrelated visits，普通 DeepNorm 风格 exponent 仍可解释。
- 对 aligned visits，即同一物理层在固定深度反复访问，保守 exponent 需要调整到 `p=1/2`。
- 在 GPT-2 small/medium backbone 与 ARC-AGI 风格任务上，DeepLoop 在较大 loop count 下优于 baseline。

因此，共享参数下的初始化与残差缩放需要考虑多次访问之间的相关性，不能直接套用独立层的规则。

### 3.4 残差流里需要保留多少历史

<a class="paper-link" href="https://arxiv.org/abs/2611.05234" title="ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals">ARLoop</a><a class="paper-citation" id="cite-88-1" href="#ref-88" title="ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals" aria-label="参考文献 88">[88]</a> 让每层沿执行深度选择历史残差。在作者的受控实验里，短窗口在较大循环深度下达到或超过全历史 attention。公开的 0.6B 和 1.2B checkpoint 把这个设计带到了实际的预训练规模。<a class="paper-link" href="https://arxiv.org/abs/2611.07226" title="Loopback Transformer: Towards a Unified View of Residual Looped Transformers">Loopback Transformer</a><a class="paper-citation" id="cite-89-1" href="#ref-89" title="Loopback Transformer: Towards a Unified View of Residual Looped Transformers" aria-label="参考文献 89">[89]</a> 则拆开了两种写入对象。若把完整块输出写入记忆，后续加权也会重复计入块输入已经携带的信息。它把更新与累计状态分开，在匹配训练 token 和执行深度的 ClimbMix 对照中改善了验证损失。

只要模型能稳定更新几轮，往往已经足以处理部署预算。要把轮数继续推到训练范围外，单靠中间轮的交叉熵监督还不够。[Dense Supervision Is Not Enough](https://arxiv.org/abs/2606.24898) 给出了一个具体盲点，归一化读出看不见的一些尺度变化仍会影响下轮更新。[STARS](https://arxiv.org/abs/2605.26733) 用 Jacobian 谱约束改善了加法和数学任务中的外推，[动力学相位实验](https://arxiv.org/abs/2608.26556) 则在小型系统里发现换一个相位就会失去原有的步数规律。[Power Method 的构造](https://arxiv.org/abs/2606.00605) 给出了能学习一种具体迭代算法的条件。[第三篇讨论读出和状态的区别]({% post_url 2026-10-19-loop-latent-reasoning %}#readout-dynamics)。

## 4. 固定预算、弹性深度与逐 token 退出
{: #depth-budget }

固定循环数让每个 token 承担相同的额外深度。Early exit 则允许一部分位置较早结束计算，将更多轮次留给其他位置。它需要同时解决两个问题：浅轮状态能否完成预测，以及当前状态是否值得继续更新。前者依赖训练，后者由深度路由或停止规则决定。

有些模型接受外部指定的循环预算，有些在进入循环前分配深度，还有些每轮之后重新决定是否继续。这些设计的决策信息与执行开销不同：

| 深度控制方式 | 决策时机与信号 | 代表工作 |
|---|---|---|
| 给定预算 | 用户指定轮数，模型适配该预算下的更新 | <a class="paper-link" href="https://arxiv.org/abs/2602.11451" title="LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation">LoopFormer</a>、<a class="paper-link" href="https://arxiv.org/abs/2610.12582v1" title="One Model to Loop Them All: Elastic Latent Recurrence for Language Models">ELR</a> |
| 预分配深度 | 从初始状态预测循环次数或深度分布 | MoR token-choice、<a class="paper-link" href="https://arxiv.org/abs/2609.15160" title="T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning With Dynamic Routing">T-LoopFormer</a> |
| 逐轮预测停止概率 | 当前状态送入 gate，累计停止概率达到阈值后退出 | Ouro |
| 预测已经稳定 | 检查相邻轮预测的 KL、熵、概率间隔等 | <a class="paper-link" href="https://arxiv.org/abs/2502.05171v2" title="Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach">Huginn</a>、轨迹读出诊断工作 |
| 预测继续计算的收益 | 用额外深度是否降低目标损失来监督决策器 | Ouro gate 后训练、<a class="paper-link" href="https://arxiv.org/abs/2609.03379" title="RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory">RecurTrace</a>、TaH2 |

预测继续计算收益的训练信号，可以与逐轮 gate 配合使用。给定预算的弹性训练则为多个深度提供可用状态，本身不决定每个 token 的退出时刻。

### 4.1 LoopFormer：给定预算下的轨迹压缩

LoopFormer<a class="paper-citation" id="cite-22-1" href="#ref-22" title="LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation" aria-label="参考文献 22">[22]</a> 借用了 diffusion/shortcut 的思路，把循环过程看成一条从 `t=0` 到 `t=1` 的 trajectory。

几个设计细节：

- 每个 loop 条件化于 `(t_{i-1}, Δ_i)`，即当前 normalized time 和 step size。
- 用 sine-cosine embedding + MLP 产生 modulation 信号，调节 RMSNorm 和 residual gate。
- 训练时同时使用完整 trajectory 和 shortcut trajectory。
- 用 consistency loss 让短 trajectory 对齐完整 trajectory。

它不直接让模型决定何时停止，而是让用户给定 budget `M`，模型能在 `M <= L` 的任意预算下工作。

![LoopFormer 的 elastic-depth 训练流程]({{ '/assets/img/blog/loop-series/loopformer_elastic_depth.png' | relative_url }})

*图 5：LoopFormer 把 loop 过程写成可压缩的 trajectory，用 shortcut consistency 训练同一模型适配不同推理预算。图源 [Jeddi 等，LoopFormer 图 1](https://arxiv.org/html/2602.11451v1#S3.F1)，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。*

### 4.2 ELR：兼顾少轮与多轮

ELR<a class="paper-citation" id="cite-23-1" href="#ref-23" title="One Model to Loop Them All: Elastic Latent Recurrence for Language Models" aria-label="参考文献 23">[23]</a> 处理 one-loop tax：为多循环训练模型之后，单循环能力可能下降。它用 token/channel 级步长控制共享块的更新：

$$
s_i=s_{i-1}+\beta_i\odot(r_i-s_{i-1}),
$$

其中 $$r_i$$ 为当前共享块提出的状态，$$\beta_i$$ 为可学习调节量。模型同时接收循环编号和总预算，并在同一次 optimizer update 内混合单轮与多轮训练梯度，避免一个预算的学习长期压过另一个。

ELR 的控制实验匹配了参数和训练数据。4B 模型只跑一轮时，十项平均分距非循环基线约半个百分点。跑四轮后，平均分又提高约 7 个百分点，首 token 和生成延迟都超过基线的两倍。

把混合深度训练放到后四分之一，仍会多用约 16% 训练 FLOPs。训练至 7T tokens 的 Koi-4B 来自另一组实验，缺少匹配的非循环对照。其 Artificial Analysis 数字由作者自行测得。

<details markdown="1">
<summary>ELR 的控制实验设置</summary>

参数和数据匹配的实验覆盖 4B dense hybrid（1T tokens）和 30B-A3B hybrid MoE（300B tokens）。

| 4B 模型对照 | 报告值 |
|---|---:|
| 十项平均分，单轮 ELR / 非循环 | 65.13 / 65.65 |
| 四轮相对单轮的平均分增益 | 7.25 个百分点 |
| 四轮相对非循环的首 token 延迟 / 解码延迟 | 2.33× / 2.06× |

</details>

ELR 学的是在给定预算下可用的状态，并不直接决定每个 token 何时退出。它与 TaH2 的逐 token 预算分配、LoopFormer 的 shortcut consistency 回答的是不同部分的问题。

### 4.3 停止概率、预测稳定性与中间监督

Ouro<a class="paper-citation" id="cite-2-2" href="#ref-2" title="Scaling Latent Reasoning via Looped Language Models" aria-label="参考文献 2">[2]</a> 在每轮末尾用轻量 gate 预测当前 token 的条件停止概率。若第 $$r$$ 轮的概率为 $$\lambda_r$$，退出深度分布为

$$
\pi(r)=\lambda_r\prod_{j<r}(1-\lambda_j),\qquad r<R,
\quad
\pi(R)=\prod_{j<R}(1-\lambda_j).
$$

末轮接收全部剩余概率。推理时累计退出概率超过阈值便停止；提高阈值通常会增加执行深度。训练阶段的中间预测损失让多个深度都能输出，深度先验则影响 gate 的分配。Ouro 还在冻结骨干后，用已经测得的逐轮损失改善训练 gate，使它识别收益是否趋于停滞。

Huginn<a class="paper-citation" id="cite-24-1" href="#ref-24" title="Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" aria-label="参考文献 24">[24]</a> 采用更直接的预测稳定性规则：相邻循环输出分布的 KL 小于阈值时停止。这类规则不需要另训一个 gate，但计算信号本身也有成本，尤其是每轮都经过完整词表读出时。预测不再变化也可能对应稳定的错误答案，因此阈值需要结合实际任务质量校准。

<a class="paper-link" href="https://arxiv.org/abs/2607.20519" title="Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts">Adaptive Depth</a><a class="paper-citation" id="cite-25-1" href="#ref-25" title="Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts" aria-label="参考文献 25">[25]</a> 将 learned halting gate 的作用拆成两部分：

1. 训练时决定不同 depth 的 loss 权重，从而塑造 recurrent trajectory。
2. 推理时作为 readout 决定何时退出。

这两个角色纠缠后，很难判断 gate 失败是因为 readout 不好，还是 trajectory 本身没有形成可早退的中间状态。

论文用 fixed-prior depth supervision 和 post-hoc readouts 做解耦实验，发现：

- 简单的 confidence、margin、predictive KL 等 post-hoc readout 在良好 trajectory 上可匹配甚至超过 learned gate。
- adaptive depth 更应理解为 trajectory-readout problem，而不是 gate-learning problem。

这一结果把中间状态训练与退出信号的选择分开了：训练先形成多轮可用的预测，再比较哪些信号能以更少计算保持质量。

<a class="paper-link" href="https://arxiv.org/abs/2606.18206" title="Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers">Fixed-Point Reasoners</a><a class="paper-citation" id="cite-26-1" href="#ref-26" title="Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers" aria-label="参考文献 26">[26]</a> 用 pre-norm、residual scaling 和 iteration-wise input mixing 改善状态传播，再用 fixed-point residual 判断是否停止。<a class="paper-link" href="https://arxiv.org/abs/2605.12466" title="Solve the Loop: Attractor Models for Language and Reasoning">Solve the Loop</a><a class="paper-citation" id="cite-27-1" href="#ref-27" title="Solve the Loop: Attractor Models for Language and Reasoning" aria-label="参考文献 27">[27]</a> 则用 attractor module 定义隐式状态，并通过 implicit differentiation 训练；其内存与优化方式不同于普通展开训练的 LoopLM。

固定点接近程度只是特定动力学下的信号。收敛点还必须随输入有意义地变化，稳定的错误答案并不合格。相关输入依赖分析见 <a class="paper-link" href="https://arxiv.org/abs/2604.15259" title="Stability and Generalization in Looped Transformers">Stability and Generalization</a><a class="paper-citation" id="cite-28-1" href="#ref-28" title="Stability and Generalization in Looped Transformers" aria-label="参考文献 28">[28]</a>。这些固定点方法主要在结构化求解任务上验证，不能直接推广成通用 LM 的停机规则。

### 4.4 RecurTrace 与 T-LoopFormer：分配样本和 token 的计算

RecurTrace<a class="paper-citation" id="cite-29-1" href="#ref-29" title="RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory" aria-label="参考文献 29">[29]</a> 先处理“循环只看上一状态”的信息瓶颈。它让 looped layer 沿 loop-time 轴注意自己此前几轮的状态，再用 loss-improvement oracle 训练 halting head。MathQA 上，它以平均 `2.0` 次 loop 达到 `56.9%`，在 matched compute 下比最佳固定深度高 `2.2` 个百分点；从 0.6B 到 8B 的四个规模上，相对同预算微调基线的生成准确率增益由 `0.6` 增至 `3.4` 个百分点。它仍按整个样本选择深度，memory 只读同位置的三轮窗口，实验也限于冻结 Qwen3 backbone 和 direct-answer targets。

T-LoopFormer<a class="paper-citation" id="cite-30-1" href="#ref-30" title="T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning With Dynamic Routing" aria-label="参考文献 30">[30]</a> 把粒度推进到 token。router 从每个 token 的初始 hidden state 一次性预测深度分布，再按执行到当前轮的累计概率构造逐渐缩小的活跃集合；它不在每轮结束后重新读取新状态来决策。gather-scatter 将活跃 token 压紧后执行共享块，再写回原位置。recursion-wise KV cache 只保存执行到对应深度的位置，attention 也只读取这些有效条目。论文在 NanoGPT 风格模型、100B FineWeb-Edu tokens 和 `24×` executed-depth 预算上报告了 perplexity、十项 zero-shot task 与端到端延迟结果。与 RecurTrace 相比，它更接近 MoR 的 token-choice routing，也更直接处理真实解码调度；但作者仍把更大规模预训练和更广泛 latent-reasoning 任务留作后续。

RecurTrace 保留跨轮历史，T-LoopFormer 则把计算分配细化到 token，并相应调整 KV 布局。

### 4.5 TaH2：循环深度与 CoT 长度的同预算比较

TaH2<a class="paper-citation" id="cite-31-1" href="#ref-31" title="Improving Test-Time Scaling with Adaptive Looped Transformers" aria-label="参考文献 31">[31]</a> 从同一 Qwen3-Base checkpoint 出发，用相同后训练数据比较单次前向模型、全栈循环、部分层循环与自适应循环。论文中的 Ouro/Huginn 对照是在这一初始化上复现相应结构，不是直接拿原始 Ouro/Huginn 预训练 checkpoint 比较。

当输出 token 上限增加时，部分循环模型的准确率随解码 FLOPs 增长得更快，却在重叠预算区间内仍低于非循环基线。固定深度让每个 token 都多算，许多 token 的预测损失几乎不变，甚至变差。曲线斜率更陡并不自动代表同预算准确率更高。

TaH2 联合训练骨干、输入注入 updater 和逐 token 决策器，以额外一轮对真实目标 token 的预测损失改善作为监督依据：

$$
\delta_t^{(r)}=\ell_t^{(r)}-\ell_t^{(r+1)},\qquad
\ell_t^{(r)}=-\log q_t^{(r)}(y_t^*).
$$

训练从正收益 token 中按累计收益覆盖率选取继续标签，并用收益相关权重监督决策器。已经停止的 token 额外执行一次无梯度 lookahead，获取下一轮损失；推理时不需要这次调用或真实目标。训练路径也遵循当前决策器，而非先在固定完整深度训练、再临时加早退。输出使用各已执行深度预测的停止概率加权混合。<a class="paper-citation" id="cite-31-2" href="#ref-31" title="Improving Test-Time Scaling with Adaptive Looped Transformers；§4" aria-label="参考文献 31，§4">[31，§4]</a>

TaH2 同时改变了历史状态的访问方式：各轮分别保留 KV，第 $$m$$ 轮、位置 $$t$$ 的 query 可以读取位置 $$s\le t$$、深度 $$j\le m$$ 的已执行状态。这种 extended duo-causal attention 让深轮能够访问浅轮历史。训练中的 lookahead 状态只用于获取监督，不作为其他 token 可以读取的历史。因此，它的计算收益包含了深度决策、状态更新和 attention 连接的共同作用。<a class="paper-citation" id="cite-31-3" href="#ref-31" title="Improving Test-Time Scaling with Adaptive Looped Transformers；附录 A.2" aria-label="参考文献 31，附录 A.2">[31，附录 A.2]</a>

在 Qwen3-1.7B 的 AIME24–26 对照中，TaH2 在相同解码预算下取得更高准确率。论文的 4B 和 8B 实验也报告了部分任务收益。<a class="paper-citation" id="cite-31-4" href="#ref-31" title="Improving Test-Time Scaling with Adaptive Looped Transformers；§5.2–5.3" aria-label="参考文献 31，§5.2–5.3">[31，§5.2–5.3]</a>

实测服务时，双轮上限配置相较标准版的一次回答仍慢约三成，准确率与时延的折中仍有改善。训练计算也高于普通 SFT，论文里的对照限于这一后训练设置。<a class="paper-citation" id="cite-31-5" href="#ref-31" title="Improving Test-Time Scaling with Adaptive Looped Transformers；表 2、图 5、§6" aria-label="参考文献 31，表 2、图 5、§6">[31，表 2、图 5、§6]</a>

<details markdown="1">
<summary>TaH2 的推理预算和延迟对照</summary>

| Qwen3-1.7B 设置 | 报告值 |
|---|---|
| 输出截断在 4K–16K，每翻倍解码 FLOPs 的准确率增量 | TaH2 2.74 个百分点，标准版 1.79 个百分点。论文的 53% 指斜率差。 |
| 输出上限扩至 32K，取基线峰值处的 191.7 TFLOPs/回答 | TaH2 15.4%，标准版 12.0%。 |
| 扩展 Mini-SGLang 中的双轮配置，相对标准版 | 每 token 解码 FLOPs 增约 22%，端到端延迟增加 30%–34%。 |

</details>

Dory 将可变深度训练用于 4B 模型。作者在 6T tokens 的预训练后做 SFT 和 RLVR，还用难度课程给较难的 prompt 分配较深的循环。几个 agentic benchmark 上，四轮配置的输出比单轮短，论文报告的 token 数比值为 1.3–2.5。若要知道任务完成得是否更快，还得乘上每个 token 的计算开销，再测稀疏注意力、缓存和并发的实际影响。后半部分在[推理系统篇]({% post_url 2026-11-26-loop-transformer-serving %})展开。

## 5. 预训练转换、跨轮学习与安全对齐

共享结构本身没有规定各轮应该如何分工。普通语言建模、跨轮奖励、蒸馏和过程目标会塑造不同的中间状态；评估时不能把训练数据与额外监督的贡献全部归给循环。

### 5.1 从预训练模型引入循环

<a class="paper-link" href="https://arxiv.org/abs/2605.11011" title="LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models">LoopUS</a><a class="paper-citation" id="cite-32-1" href="#ref-32" title="LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models" aria-label="参考文献 32">[32]</a> 将预训练模型划分为 encoder、reasoning、decoder，只循环中间部分，并加入 selective decay gate、random deep supervision 和 confidence head，分别处理状态漂移、训练展开与自适应计算。


![LoopUS 的 upcycling 流程]({{ '/assets/img/blog/loop-series/loopus_upcycling_pipeline.png' | relative_url }})


*图 6：LoopUS 在预训练骨干中选出循环部分，而非要求整套模型从头训练。*

<a class="paper-link" href="https://arxiv.org/abs/2605.23872" title="Training-Free Looped Transformers">Training-Free Looped Transformers</a><a class="paper-citation" id="cite-33-1" href="#ref-33" title="Training-Free Looped Transformers" aria-label="参考文献 33">[33]</a> 则在冻结模型中插入数值积分式 wrapper，比较中间窗口整体迭代与逐层迭代。它不更新参数，适用边界也不同；MoE 的路由变化可能使整块重复不稳定。推理时包装已有层与训练一个 LoopLM，不能按同一训练预算解释。

CoT 的连续目标和去噪目标是另外两类训练信号。前者怎样约束工作区见第三篇，后者怎样连接局部监督与长递归见第二篇。

### 5.2 跨轮奖励：RLTT 与 LoopRPT

RLTT<a class="paper-citation" id="cite-34-1" href="#ref-34" title="Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models" aria-label="参考文献 34">[34]</a> 不只对最终循环的 next-token 分布分配奖励，还使用跨轮加权的 log-probability：

$$
\sum_r\omega_r\log P_\theta^{(r)}(y_j\mid x,y_{<j}).
$$

权重可来自退出概率、递进权重或均匀权重。若 forward 已经产生各轮 logits，额外计算有限，但仍需要保存跨轮信息。它奖励的是 latent trajectory 对输出策略的贡献，不要求每轮对应一个文本推理步骤。

LoopRPT<a class="paper-citation" id="cite-35-1" href="#ref-35" title="LoopRPT: Reinforcement Pre-Training for Looped Language Models" aria-label="参考文献 35">[35]</a> 把这种思路用于 reinforcement pre-training，选择教师熵较高的 hard tokens，结合 EMA teacher、逐步准确率增益、时间惩罚与带噪 latent rollout，共同训练退出策略和骨干。Ouro-1.4B/2.6B 的结果包含这套训练机制的共同作用，不能视为只增加循环的消融。

### 5.3 LoopOPD：在浅轮生成的轨迹上蒸馏
{: #loopopd }

<a class="paper-link" href="https://arxiv.org/abs/2610.10623v1" title="Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models">LoopOPD</a><a class="paper-citation" id="cite-36-1" href="#ref-36" title="Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models" aria-label="参考文献 36">[36]</a> 用同一 LoopLM 的较浅循环作 student、末轮作 teacher。student 先生成 rollout，teacher 在这些相同前缀上给出分布，再最小化

$$
\mathbb E_{y\sim p_{\mathrm{student}}}
\sum_t\operatorname{KL}\!\left(
p_{\mathrm{student}}(\cdot\mid y_{<t},x)
\,\middle\|\,
\operatorname{sg}[p_{\mathrm{teacher}}(\cdot\mid y_{<t},x)]
\right).
$$

LoopOPD 固定 teacher；D-LoopOPD 随共享模型的更新，用当前末轮更新教师目标。虽然目标侧停止梯度，student 的参数更新仍会影响后续末轮计算，形成跨轮的双向影响。它不同于只在固定 teacher 数据上模仿，也不同于推理时修改 logits 的 LoopCD。

Ouro-2.6B-Thinking 上，论文报告 AIME 2024 末轮 Avg@16 从 62.08 到 71.67。这里是十六次采样结果的平均准确率，不是“至少一次成功”的 pass@16。student 与 teacher 的深度间隔也重要，太弱的浅轮或过于接近末轮的中间轮未必适合蒸馏。局部改善分析有充分条件，不能推出模型可以无限自我提升。

### 5.4 ECHO 的训练目标沿序列推进
{: #echo-causal-recursion }

ECHO 复用骨干最后一层。每次把上一步的隐状态和前一个 token 的 embedding 融合起来，再预测更远的位置。共享末层因此同时承担普通 next-token 预测和多步预测。它在训练时按 step 采样展开长度，抽到的前缀逐步计算并对各步损失取平均。[MTP 篇]({% post_url 2026-10-23-mtp-pretraining-self-drafting %}#54-echo未来监督是否应该直接约束主干最后一层)介绍了训练深度和实验对照。[第四篇]({% post_url 2026-11-26-loop-transformer-serving %}#draft-types)接着讨论这些未来预测怎样生成草稿。

<a id="dds-loss-weights"></a>

<details markdown="1">
<summary>DDS 的执行频率和梯度权重</summary>

Dynamic Depth Sampling（DDS）减少实际执行的训练展开。每个 optimizer step 采样一次深度，同一步的各 microbatch 使用相同深度，再训练从第一步到该深度的整个前缀。论文采用

$$
D\sim\pi,\qquad
\pi(1,2,3)=(0.6,0.3,0.1),\qquad
\mathbb E[D]=1.5.
$$

各深度被执行的频率是 $$f_k=P(D\ge k)$$，即 $$(1,0.4,0.1)$$。实现对已执行的 $$D$$ 项损失取平均，因此相应的期望损失系数为

$$
\omega_k=\mathbb E\!\left[\frac{\mathbf 1\{D\ge k\}}{D}\right]
=\sum_{d=k}^{K}\frac{\pi(d)}{d}.
$$

执行频率与归一化后的梯度权重需要区分。DDS 也与推理 early exit 不同：这里采样的是训练展开长度，深轮在未被采到时确实不执行；它没有根据单个 token 的当前状态作停机决策。<a class="paper-citation" id="cite-81-3" href="#ref-81" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction；§3.2、附录 A.1" aria-label="参考文献 81，§3.2、附录 A.1">[81，§3.2、附录 A.1]</a>

</details>

### 5.5 安全对齐必须覆盖允许的执行深度
{: #safebridge }

<a class="paper-link" href="https://arxiv.org/abs/2610.10625v1" title="Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models">SafeBridge</a><a class="paper-citation" id="cite-37-1" href="#ref-37" title="Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models" aria-label="参考文献 37">[37]</a> 在 Ouro-1.4B/2.6B 上发现，同一 prompt 的安全表现随循环深度变化，而且更深不一定更安全。方法结合深度相关调制、选择性状态传递、多深度安全监督和一致性约束，将对齐目标覆盖到多个执行深度。

作者在 192 个攻击配置上统计平均攻击成功率。加上跨轮安全监督后，1.4B 从 DPO 对照的 21.06% 降至 4.56%，2.6B 从 44.93% 降至 9.85%。论文还检查了实用性和过度拒答。它指出的是动态深度部署中的具体风险：默认深度安全，不能推出所有允许深度都安全。

若部署允许动态深度，需要检查所有允许预算及其切换，而非只评测默认轮数。安全拒答与过程可监控性是不同性质：<a class="paper-link" href="https://arxiv.org/abs/2610.02741v1" title="On the Chain-of-Thought Monitorability of Looped Language Models">CoT Monitorability</a><a class="paper-citation" id="cite-38-1" href="#ref-38" title="On the Chain-of-Thought Monitorability of Looped Language Models" aria-label="参考文献 38">[38]</a> 的任务构造和证据边界放在第三篇；较低攻击成功率不能替代对解释忠实性的验证。

## 6. 预算相同以后，循环的收益还剩多少
{: #training-compute }

<a class="paper-link" href="https://arxiv.org/abs/2610.00673" title="Closing the Loop: Practical Training Recipes for Looped Language Models">Closing the Loop</a><a class="paper-citation" id="cite-90-1" href="#ref-90" title="Closing the Loop: Practical Training Recipes for Looped Language Models" aria-label="参考文献 90">[90]</a> 从头训练了一组可比较的模型。同数据、同 token 预算和同参数下，1.4B LoopLM 的十二项评测都胜过对应的非循环模型。把推理计算对齐到 3.9B 非循环模型后，数学和阅读理解更接近，知识和常识上的差距仍在。这两组实验放在一起，才能估计共享层补回了多少独立参数的作用。

在已训练模型上加 loop 时，保留基座能力是另一项约束。<a class="paper-link" href="https://arxiv.org/abs/2610.42030" title="There Is No Free Loop: When Does Looping a Trained Model Pay for Its Inference Compute?">There Is No Free Loop</a><a class="paper-citation" id="cite-91-1" href="#ref-91" title="There Is No Free Loop: When Does Looping a Trained Model Pay for Its Inference Compute?" aria-label="参考文献 91">[91]</a> 在 Qwen3 / Qwen3.5 的 0.8B–9B 范围同时比较同参数的继续训练基线和同推理 FLOPs 基线。论文中只有 0.8B 实验胜过两组对照，4B 和 9B 在所测任务上未胜过同参数的非循环版本。GLM-5.4 的技术公告则给出了大规模继续训练的实例。它公布的 30B 对照约多用 10% 训练计算，报告了长链任务上的改善。两组结果使用的基座、数据和训练规模不同，正好说明同一标签下还要区分怎样改造基座。

<a id="kv-sharing减少缓存之后计算还剩多少"></a>

模型选定以后，多跑一轮会增加每个 token 的计算，也可能让答案变短。实际延迟和吞吐还受历史读取与批处理开销影响。[第四篇的跨轮 KV]({% post_url 2026-11-26-loop-transformer-serving %}#cross-loop-kv)从历史读取开始算这部分成本。

## 附录 · 文献索引

<details markdown="1">
<summary>展开早期工作与专题索引</summary>

下表汇集 Loop 相关文献。去噪与生成方法在第二篇展开，连续表示、工作区与读出机制在第三篇展开；本篇详细讨论架构、预算和系统成本。

| 时间 | 工作 | 类型 | 一句话定位 |
|---|---|---|---|
| 2026-01 | <a class="paper-link" href="https://arxiv.org/abs/2601.10242" title="Loop as a Bridge: Can Looped Transformers Truly Link Representation Space and Natural Language Outputs?">Loop as a Bridge</a> | 核心/机制 | 研究 looped Transformer 中 representation readout 与文本自验证之间的差距 |
| 2026-02 | RLTT | 核心/训练 | 对 latent thought trajectory 做 RL credit assignment |
| 2026-02 | LoopFormer | 核心/自适应 | trajectory conditioning + shortcut consistency，实现 elastic depth |
| 2026-03 | LoopRPT | 核心/训练 | 对 LoopLM 做 reinforcement pre-training，奖励 latent steps |
| 2026-04 | Loop, Think, & Generalize | 核心/泛化 | 研究 recurrent-depth Transformer 的组合泛化和深度外推 |
| 2026-04 | <a class="paper-link" href="https://arxiv.org/abs/2604.11791" title="A Mechanistic Analysis of Looped Reasoning Language Models">Mechanistic Analysis</a> | 核心/机制 | 分析 looped reasoning LM 的阶段、attention 与循环动态 |
| 2026-04 | Parcae | 核心/稳定与 scaling | 动力系统约束下的稳定 looped LM 与 scaling laws |
| 2026-04 | Stability and Generalization | 核心/理论 | 分析 recall、outer normalization 与 fixed-point 泛化 |
| 2026-04 | How Much Is One Recurrence Worth? | 核心/scaling law | 用 recurrence-equivalence exponent 衡量一次循环价值 |
| 2026-04 | <a class="paper-link" href="https://arxiv.org/abs/2604.21215" title="The Recurrent Transformer: Greater Effective Depth and Efficient Decoding">Recurrent Transformer</a> | 相邻/效率 | 时间方向 recurrent attention，强调有效深度和 decoding efficiency |
| 2026-04 | Hyperloop Transformers | 核心/架构 | loop 边界引入 hyper-connections 改善信息流 |
| 2026-05 | MELT | 扩展/内存 | 固定大小 latent state 更新 KV cache，降低 loop memory 增长 |
| 2026-05 | Sparse Layers are Critical | 核心/MoE | 显示 sparse/MoE 对 looped LM scaling 的影响 |
| 2026-05 | LoopUS | 核心/upcycling | 把预训练 LLM 转成 looped latent refinement model |
| 2026-05 | Solve the Loop / Attractor Models | 相邻/fixed-point | 用 attractor/fixed-point refinement 建模语言与推理 |
| 2026-05 | LoopQ | 扩展/量化 | 处理 LoopLM 量化中的跨 loop 分布漂移和误差累积 |
| 2026-05 | Fully Looped Transformer | 核心/稳定 | 用 fully looped architecture + attention injection 稳定高循环训练 |
| 2026-05 | LT2 | 扩展/效率 | linear/sparse attention + looped Transformer |
| 2026-05 | Training-Free Looped Transformers | 核心/部署 | 不重训，在冻结模型中插入 loop wrapper |
| 2026-05 | <a class="paper-link" href="https://arxiv.org/abs/2605.26106" title="Looped Diffusion Language Models">LoopMDM</a> | 扩展/diffusion | 把 selective looping 用到 masked diffusion LM |
| 2026-05 | STARS | 核心/稳定 | Jacobian spectral radius regularization 稳定 latent trajectory |
| 2026-05 | Power Method Theory | 理论 | 证明带 LayerNorm 的 looped Transformer 可学习 power method |
| 2026-06 | Fixed-Point Reasoners | 核心/自适应 | pre-norm + residual scaling + fixed-point halting |
| 2026-06 | LoopCoder-v2 | 核心/代码模型实践 | 7B PLT coder；系统研究 loop count 的非单调收益 |
| 2026-06 | Dense Supervision Is Not Enough | 核心/监督 | 提出 readout blind spot，说明 dense CE 不能约束所有递归变量 |
| 2026-06 | <a class="paper-link" href="https://arxiv.org/abs/2606.31779" title="Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers">LOTUS</a> | 核心/latent CoT | 用 CoT-step supervision 对齐 looped latent blocks |
| 2026-07 | Looped State-Space LMs | 扩展/SSM | 把 loop 推广到 Mamba/Hybrid Mamba-Transformer |
| 2026-07 | DeepLoop | 核心/残差缩放 | loop-aware residual scaling rule |
| 2026-07 | Loop the Loopies! | 核心/MoE scaling | layer-loop MoE 与 compute-matched Loopie recipe |
| 2026-07 | <a class="paper-link" href="https://arxiv.org/abs/2607.17843" title="Möbius Learning: Cyclic Depth Folding in Transformers">Möbius Learning</a> | 相邻/分布式 | cyclic depth folding 与 Möbius parallelism |
| 2026-07 | <a class="paper-link" href="https://arxiv.org/abs/2607.18553" title="Operational Proto-Introspection in Looped Language Models">Operational Proto-Introspection</a> | 核心/机制 | 读出 looped hidden trajectory 的过程质量，但控制收益有限 |
| 2026-07 | LatentMT | 扩展/应用 | 用 LoopLM latent reasoning 做机器翻译 |
| 2026-07 | Adaptive Depth | 核心/自适应诊断 | 把 adaptive depth 拆成 trajectory 与 readout 问题 |
| 2026-07 | Nanbeige4.2-3B | 核心/模型实践 | 3B agentic model 中采用 Looped Transformer |
| 2026-08 | Full-bandwidth Transformer | 相邻/latent feedback | 将 top-layer hidden state 反馈到下一 decoding step，扩宽自回归反馈通道 |
| 2026-08 | Compositional Tool Calling | 核心/应用 | 检验 recurrent depth 对多调用依赖、组合与自适应计算的作用 |
| 2026-08 | Recirculation | 相邻/推理时 recurrence | 冻结主模型，把深层状态回流到后续 step 的浅层，用于 belief-state tracking |
| 2026-08 | Think Shallow, Solve Deep | 核心/稳定 | 用 finite-time dynamical regime 诊断 test-time depth 是否安全 |
| 2026-08 | Allocating Recurrent Compute | 扩展/架构 | MixerLoop 只重复 Gated DeltaNet mixer，不重复 dense FFN |
| 2026-08 | Dynamical Phase Selection | 核心/理论 | 初始化可选择不同动力学相位，并改变 test-time compute 的 scaling law |
| 2026-09 | SMELT | 核心/MoE scaling law | 在 FLOPs、参数和 KV cache 同时匹配下，给出 middle-50% loop-twice recipe |
| 2026-09 | <a class="paper-link" href="https://arxiv.org/abs/2609.01924" title="Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?">Jacobian Lens</a> | 核心/机制 | 比较 Ouro、Huginn 与普通 Transformer 的 workspace 读写接口 |
| 2026-09 | RecurTrace | 核心/自适应 | 用 loop-time memory 和 oracle-distilled halting 分配样本级递归深度 |
| 2026-09 | <a class="paper-link" href="https://arxiv.org/abs/2609.11801" title="Thinking with Looped Flows">Looped Flows</a> | 扩展/训练 | 用局部去噪目标训练可跨更多迭代传递计算的 recurrent state |
| 2026-09 | T-LoopFormer | 核心/自适应 | token-choice routing、gather-scatter 与 recursion-wise KV cache |
| 2026-09 | Right Direction, Wrong Step | 核心/机制 | 将有益方向与过大的 recurrent update step 区分开 |
| 2026-09 | LoopSpec | 核心/部署 | 用早期 recurrent state 做流水化 self-speculative decoding |
| 2026-09 | Model Growth and Scaling Exponents | 核心/scaling law | 研究 loop growth、boundary operator 与 scaling exponent 的关系 |
| 2026-09 | TaH2 | 核心/自适应 | 用在线循环收益监督深度选择，比较 CoT 长度增长时的同预算准确率 |
| 2026-09 | LoopVL | 扩展/多模态 | 耦合 L/H 模块反复更新视觉与语言状态，检验视觉更新及日程外推 |
| 2026-09 | <a class="paper-link" href="https://arxiv.org/abs/2609.39892v1" title="Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does">Shared Weights, Selected Computations</a> | 核心/机制 | 状态映射与 attention patching 检验共享块如何选择不同转移 |
| 2026-09 | <a class="paper-link" href="https://arxiv.org/abs/2609.40305v1" title="Looped Diffusion Transformer">Looped-DiT</a> | 扩展/图像生成 | 深监督与注意力调节支持内部循环，比较循环深度与去噪步数 |
| 2026-09 | LoopCD（POSTECH） | 核心/解码 | 用末轮与浅轮对数概率做对比，分析对不稳定 token 的影响 |
| 2026-09 | Foil | 核心/MoE | 扩大每层专家池、减少专家层数，循环共享专家但不共享 attention |
| 2026-09 | Scaling Laws for Looped MoE | 核心/scaling law | 拟合稀疏度条件下有界的循环等效参数收益 |
| 2026-10 | LoopCD（Apple） | 核心/解码 | 比较 logits 与 hidden 两种跨轮引导，主动改变预测分布 |
| 2026-10 | LOOM | 核心/MoE 训练 | 残差缩放、输入重注入与分轮路由支持更多次循环 |
| 2026-10 | LPT / H-LPT | 核心/记忆 | 第一轮写全局 KV，后续轮读取共享记忆与短局部窗口 |
| 2026-10 | CoT Monitorability | 核心/监控 | 比较循环深度与架构对显式推理可监控性的影响 |
| 2026-10 | <a class="paper-link" href="https://arxiv.org/abs/2610.04198v1" title="ALoDLM: Adaptively Looped Diffusion Language Models">ALoDLM</a> | 扩展/diffusion LM | 内循环结合离散 token 提交、latent 反馈和学习式停机 |
| 2026-10 | <a class="paper-link" href="https://arxiv.org/abs/2610.05538v1" title="LiFT: Loop Flow Transformers">LiFT</a> | 扩展/图像生成 | 在内部计算深度上监督从初始速度预测到 flow target 的修正路径 |
| 2026-10 | <a class="paper-link" href="https://arxiv.org/abs/2610.06666v1" title="What Matters for Latent Reasoning with Flow Matching">FLaRe</a> | 相邻/latent reasoning | 生成 CoT latent 并评测其用途、读出与修订能力 |
| 2026-10 | HLA | 核心/KV 压缩 | 近窗口保留精确 KV，较远历史使用可直接查询的 latent |
| 2026-10 | Ragged KV | 核心/动态计算 | 读取历史 token 已有的合适深度 KV，避免早退后补算 |
| 2026-10 | ResidualQuant | 核心/KV 量化 | 以末轮 KV 为锚点，低比特存储其他轮的残差 |
| 2026-10 | LoopOPD / D-LoopOPD | 核心/训练 | 在浅轮生成的轨迹上执行跨轮 on-policy 蒸馏 |
| 2026-10 | SafeBridge | 核心/安全 | 通过状态传递和联合监督对齐多个循环深度的安全表现 |
| 2026-10 | <a class="paper-link" href="https://arxiv.org/abs/2610.09876v1" title="Think Before You Paint: Recursive Latent Reasoning for Diffusion Models">PaTh</a> | 扩展/视觉推理 | 递归 Thinker 在每个去噪步控制冻结的 Painter |
| 2026-10 | ImR | 核心/稳定读出 | 对 latent 轨迹的 logits 求平均，不要求单点收敛 |
| 2026-10 | LLoCoT | 核心/latent reasoning | 固定 latent 工作区并行更新，以连续 CoT 目标监督 |
| 2026-10 | InfiLoop | 核心/稳定性 | 用带年龄衰减的流式状态聚合支持长迭代 |
| 2026-10 | ELR / Koi | 核心/弹性深度 | 混合深度训练与自适应更新，减少单循环能力损失 |
| 2026-10 | RAMP | 核心/混合精度 | 早轮 FP8、后轮 BF16，降低 recurrent prefill 成本 |
| 2026-10 | ECHO | 相邻/因果递归与 MTP | 复用末层预测未来 token，采样训练展开长度，并用于自草稿生成 |

</details>

> 计算的回路 · **一 · 循环架构** / [二 · 去噪与生成]({% post_url 2026-10-19-loop-denoising-flows %}) / [三 · 隐式推理]({% post_url 2026-10-19-loop-latent-reasoning %}) / [四 · 推理系统]({% post_url 2026-11-26-loop-transformer-serving %})

## 参考文献

1. <span id="ref-1" class="reference-anchor"></span>Abbas Zeitoun, Lucas Torroba-Hennigen, Yoon Kim. [Hyperloop Transformers](https://arxiv.org/abs/2604.21254), 2026. <a class="reference-backlink" href="#cite-1-1" aria-label="返回文献 1 首次引用处" title="返回正文">↩</a>
2. <span id="ref-2" class="reference-anchor"></span>Rui-Jie Zhu et al. [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741), 2025/2026 updated. <a class="reference-backlink" href="#cite-2-1" aria-label="返回文献 2 首次引用处" title="返回正文">↩</a>
3. <span id="ref-3" class="reference-anchor"></span>Sangmin Bae et al. [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524), 2025. <a class="reference-backlink" href="#cite-3-1" aria-label="返回文献 3 首次引用处" title="返回正文">↩</a>
4. <span id="ref-4" class="reference-anchor"></span>Kristian Schwethelm et al. [How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped Language Models](https://arxiv.org/abs/2604.21106), 2026. <a class="reference-backlink" href="#cite-4-1" aria-label="返回文献 4 首次引用处" title="返回正文">↩</a>
5. <span id="ref-5" class="reference-anchor"></span>Zitian Gao et al. [Loop the Loopies!](https://arxiv.org/abs/2607.16051), 2026. <a class="reference-backlink" href="#cite-5-1" aria-label="返回文献 5 首次引用处" title="返回正文">↩</a>
6. <span id="ref-6" class="reference-anchor"></span>Shaowen Wang et al. [SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers](https://arxiv.org/abs/2609.01343), 2026. <a class="reference-backlink" href="#cite-6-1" aria-label="返回文献 6 首次引用处" title="返回正文">↩</a>
7. <span id="ref-7" class="reference-anchor"></span>Ryan Lee et al. [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165), 2026. <a class="reference-backlink" href="#cite-7-1" aria-label="返回文献 7 首次引用处" title="返回正文">↩</a>
8. <span id="ref-8" class="reference-anchor"></span>[Scaling Laws for Looped Mixture of Experts](https://arxiv.org/abs/2609.40316v1)，2026-09-30，v1。 <a class="reference-backlink" href="#cite-8-1" aria-label="返回文献 8 首次引用处" title="返回正文">↩</a>
9. <span id="ref-9" class="reference-anchor"></span>Zixi Chen et al. [How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents](https://arxiv.org/abs/2609.19107), 2026。v2，2026-09-17 更新。 <a class="reference-backlink" href="#cite-9-1" aria-label="返回文献 9 首次引用处" title="返回正文">↩</a>
10. <span id="ref-10" class="reference-anchor"></span>[Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts](https://arxiv.org/abs/2610.01153v1)，2026-10-01，v1。 <a class="reference-backlink" href="#cite-10-1" aria-label="返回文献 10 首次引用处" title="返回正文">↩</a>
11. <span id="ref-11" class="reference-anchor"></span>[How to Loop MoE: Flatten the Experts, Untie the Attention](https://arxiv.org/abs/2609.35751v1)，2026-09-28，v1。 <a class="reference-backlink" href="#cite-11-1" aria-label="返回文献 11 首次引用处" title="返回正文">↩</a>
12. <span id="ref-12" class="reference-anchor"></span>Hayden Prairie et al. [Parcae: Scaling Laws for Stable Looped Language Models](https://arxiv.org/abs/2604.12946), 2026. <a class="reference-backlink" href="#cite-12-1" aria-label="返回文献 12 首次引用处" title="返回正文">↩</a>
13. <span id="ref-13" class="reference-anchor"></span>Rao Fu et al. [Simply Stabilizing the Loop via Fully Looped Transformer](https://arxiv.org/abs/2605.18797), 2026. <a class="reference-backlink" href="#cite-13-1" aria-label="返回文献 13 首次引用处" title="返回正文">↩</a>
14. <span id="ref-14" class="reference-anchor"></span>Shuzhen Li et al. [DeepLoop: Depth Scaling for Looped Transformers](https://arxiv.org/abs/2607.13491), 2026. <a class="reference-backlink" href="#cite-14-1" aria-label="返回文献 14 首次引用处" title="返回正文">↩</a>
15. <span id="ref-15" class="reference-anchor"></span>Xiao-Wen Yang et al. [Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](https://arxiv.org/abs/2605.26733), 2026.
16. <span id="ref-16" class="reference-anchor"></span>Rituraj Sharma and Tu Vu. [Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models](https://arxiv.org/abs/2606.24898), 2026.
17. <span id="ref-17" class="reference-anchor"></span>Ivan Viakhirev et al. [Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth](https://arxiv.org/abs/2608.18222), 2026.
18. <span id="ref-18" class="reference-anchor"></span>Gunn Kim. [Dynamical Phase Selection Controls Compute Scaling in Looped Transformers](https://arxiv.org/abs/2608.26556), 2026.
19. <span id="ref-19" class="reference-anchor"></span>Zhihao Guo et al. [Right Direction, Wrong Step: Geometric Analysis of Finite-Step Failure in Looped Transformers](https://arxiv.org/abs/2609.16665), 2026.
20. <span id="ref-20" class="reference-anchor"></span>[Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals](https://arxiv.org/abs/2610.11570v1)，2026-10-08，v1。
21. <span id="ref-21" class="reference-anchor"></span>[Invariant-Measure Reasoners: Stable Representations for Latent Reasoning](https://arxiv.org/abs/2610.10996v1)，2026-10-07，v1。
22. <span id="ref-22" class="reference-anchor"></span>Ahmadreza Jeddi, Marco Ciccone, Babak Taati. [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451), ICLR 2026. <a class="reference-backlink" href="#cite-22-1" aria-label="返回文献 22 首次引用处" title="返回正文">↩</a>
23. <span id="ref-23" class="reference-anchor"></span>[One Model to Loop Them All: Elastic Latent Recurrence for Language Models](https://arxiv.org/abs/2610.12582v1)，2026-10-08，v1。 <a class="reference-backlink" href="#cite-23-1" aria-label="返回文献 23 首次引用处" title="返回正文">↩</a>
24. <span id="ref-24" class="reference-anchor"></span>Jonas Geiping et al. [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171v2). 2025。 <a class="reference-backlink" href="#cite-24-1" aria-label="返回文献 24 首次引用处" title="返回正文">↩</a>
25. <span id="ref-25" class="reference-anchor"></span>Andrei Cristian Popescu et al. [Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and Trajectory Readouts](https://arxiv.org/abs/2607.20519), 2026. <a class="reference-backlink" href="#cite-25-1" aria-label="返回文献 25 首次引用处" title="返回正文">↩</a>
26. <span id="ref-26" class="reference-anchor"></span>Sajad Movahedi et al. [Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers](https://arxiv.org/abs/2606.18206), 2026. <a class="reference-backlink" href="#cite-26-1" aria-label="返回文献 26 首次引用处" title="返回正文">↩</a>
27. <span id="ref-27" class="reference-anchor"></span>Jacob Fein-Ashley and Paria Rashidinejad. [Solve the Loop: Attractor Models for Language and Reasoning](https://arxiv.org/abs/2605.12466), 2026. <a class="reference-backlink" href="#cite-27-1" aria-label="返回文献 27 首次引用处" title="返回正文">↩</a>
28. <span id="ref-28" class="reference-anchor"></span>Asher Labovich. [Stability and Generalization in Looped Transformers](https://arxiv.org/abs/2604.15259), 2026. <a class="reference-backlink" href="#cite-28-1" aria-label="返回文献 28 首次引用处" title="返回正文">↩</a>
29. <span id="ref-29" class="reference-anchor"></span>Yuxiang Wang et al. [RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory](https://arxiv.org/abs/2609.03379), 2026. <a class="reference-backlink" href="#cite-29-1" aria-label="返回文献 29 首次引用处" title="返回正文">↩</a>
30. <span id="ref-30" class="reference-anchor"></span>Mingqian Yu et al. [T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning With Dynamic Routing](https://arxiv.org/abs/2609.15160), 2026. <a class="reference-backlink" href="#cite-30-1" aria-label="返回文献 30 首次引用处" title="返回正文">↩</a>
31. <span id="ref-31" class="reference-anchor"></span>Yichen You et al. [Improving Test-Time Scaling with Adaptive Looped Transformers](https://arxiv.org/abs/2609.35748v1), 2026-09-28。TaH2；深度监督见 §4，计算与延迟比较见 §5.2、表 2。 <a class="reference-backlink" href="#cite-31-1" aria-label="返回文献 31 首次引用处" title="返回正文">↩</a>
32. <span id="ref-32" class="reference-anchor"></span>Taekyhun Park and Yongjae Lee. [LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models](https://arxiv.org/abs/2605.11011), 2026. <a class="reference-backlink" href="#cite-32-1" aria-label="返回文献 32 首次引用处" title="返回正文">↩</a>
33. <span id="ref-33" class="reference-anchor"></span>Lizhang Chen et al. [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872), 2026. <a class="reference-backlink" href="#cite-33-1" aria-label="返回文献 33 首次引用处" title="返回正文">↩</a>
34. <span id="ref-34" class="reference-anchor"></span>Jonathan Williams and Esin Tureci. [Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought Trajectories Improves Reasoning in Looped Language Models](https://arxiv.org/abs/2602.10520), 2026. <a class="reference-backlink" href="#cite-34-1" aria-label="返回文献 34 首次引用处" title="返回正文">↩</a>
35. <span id="ref-35" class="reference-anchor"></span>Guo Tang et al. [LoopRPT: Reinforcement Pre-Training for Looped Language Models](https://arxiv.org/abs/2603.19714), 2026. <a class="reference-backlink" href="#cite-35-1" aria-label="返回文献 35 首次引用处" title="返回正文">↩</a>
36. <span id="ref-36" class="reference-anchor"></span>[Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models](https://arxiv.org/abs/2610.10623v1)，2026-10-07，v1。 <a class="reference-backlink" href="#cite-36-1" aria-label="返回文献 36 首次引用处" title="返回正文">↩</a>
37. <span id="ref-37" class="reference-anchor"></span>[Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models](https://arxiv.org/abs/2610.10625v1)，2026-10-07，v1。 <a class="reference-backlink" href="#cite-37-1" aria-label="返回文献 37 首次引用处" title="返回正文">↩</a>
38. <span id="ref-38" class="reference-anchor"></span>[On the Chain-of-Thought Monitorability of Looped Language Models](https://arxiv.org/abs/2610.02741v1)，2026-10-02，v1。 <a class="reference-backlink" href="#cite-38-1" aria-label="返回文献 38 首次引用处" title="返回正文">↩</a>
39. <span id="ref-39" class="reference-anchor"></span>Victor Conchello Vendrell et al. [Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models](https://arxiv.org/abs/2605.07721), 2026.
40. <span id="ref-40" class="reference-anchor"></span>[Enabling Dynamic Computation in Looped LMs](https://arxiv.org/abs/2610.09013v1)，2026-10-06，v1。
41. <span id="ref-41" class="reference-anchor"></span>[The Surprising Effectiveness of Shared Memory in Looped Transformers](https://arxiv.org/abs/2610.02383v1)，2026-10-01，v1。
42. <span id="ref-42" class="reference-anchor"></span>[Hybrid Latent Attention for Looped Language Models](https://arxiv.org/abs/2610.07940v1)，2026-10-06，v1。
43. <span id="ref-43" class="reference-anchor"></span>[ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals](https://arxiv.org/abs/2610.10381v1)，2026-10-07，v1。
44. <span id="ref-44" class="reference-anchor"></span>Rui Fang et al. [LoopQ: Quantization for Recursive Transformers](https://arxiv.org/abs/2605.16343), 2026.
45. <span id="ref-45" class="reference-anchor"></span>[RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models](https://arxiv.org/abs/2610.13041v1)，2026-10-09，v1。
46. <span id="ref-46" class="reference-anchor"></span>Jian Yang et al. [LoopCoder-v2: Only Loop Once for Efficient Test-Time Computation Scaling](https://arxiv.org/abs/2606.18023), 2026.
47. <span id="ref-47" class="reference-anchor"></span>SangLyul Cho et al. [LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers](https://arxiv.org/abs/2609.17184), 2026.
48. <span id="ref-48" class="reference-anchor"></span>Chunyuan Deng et al. [LT2: Linear-Time Looped Transformers](https://arxiv.org/abs/2605.20670), 2026.
49. <span id="ref-49" class="reference-anchor"></span>Ruhai Lin et al. [Allocating Recurrent Compute in Looped Language Models](https://arxiv.org/abs/2608.18230), 2026.
50. <span id="ref-50" class="reference-anchor"></span>Harsh Kohli et al. [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822), 2026.
51. <span id="ref-51" class="reference-anchor"></span>Lyumin Wu, Chenyang Zhang, Yuan Cao. [Looped Transformers with Layer Normalization Provably Learn the Power Method](https://arxiv.org/abs/2606.00605), 2026.
52. <span id="ref-52" class="reference-anchor"></span>Nanbeige LLM Lab. [Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model](https://arxiv.org/abs/2607.22083), 2026.
53. <span id="ref-53" class="reference-anchor"></span>Wei-Rui Chen et al. [LatentMT: Machine Translation with Latent Reasoning](https://arxiv.org/abs/2607.18618), 2026.
54. <span id="ref-54" class="reference-anchor"></span>Andrei Cristian Popescu et al. [Looped Language Models Improve Compositional Tool Calling](https://arxiv.org/abs/2608.18171), 2026.
55. <span id="ref-55" class="reference-anchor"></span>Zhenxuan Yu et al. [Looped State-Space Language Models with Adaptive Exit-State Selection](https://arxiv.org/abs/2607.10110), 2026.
56. <span id="ref-56" class="reference-anchor"></span>Zhe Qian et al. [LoopVL: Recurrent Visual Intelligence](https://arxiv.org/abs/2609.38426v1), 2026-09-29。架构与视觉干预见 §2，预算与日程比较见表 1、3–4。
57. <span id="ref-57" class="reference-anchor"></span>Nikunj Saunshi et al. [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416), ICLR 2025.
58. <span id="ref-58" class="reference-anchor"></span>Bohong Wu et al. [Parallel Loop Transformer for Efficient Test-Time Computation Scaling](https://arxiv.org/abs/2510.24824), 2025.
59. <span id="ref-59" class="reference-anchor"></span>Guanxu Chen et al. [Loop as a Bridge: Can Looped Transformers Truly Link Representation Space and Natural Language Outputs?](https://arxiv.org/abs/2601.10242), 2026.
60. <span id="ref-60" class="reference-anchor"></span>Hugh Blayney et al. [A Mechanistic Analysis of Looped Reasoning Language Models](https://arxiv.org/abs/2604.11791), 2026.
61. <span id="ref-61" class="reference-anchor"></span>Costin-Andrei Oncescu et al. [The Recurrent Transformer: Greater Effective Depth and Efficient Decoding](https://arxiv.org/abs/2604.21215), 2026.
62. <span id="ref-62" class="reference-anchor"></span>Sanghyun Lee et al. [Looped Diffusion Language Models](https://arxiv.org/abs/2605.26106), 2026.
63. <span id="ref-63" class="reference-anchor"></span>Ying Fan, Anej Svete, Kangwook Lee. [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779), 2026.
64. <span id="ref-64" class="reference-anchor"></span>Tongtian Zhu. [Möbius Learning: Cyclic Depth Folding in Transformers](https://arxiv.org/abs/2607.17843), 2026.
65. <span id="ref-65" class="reference-anchor"></span>Jan Kirin. [Operational Proto-Introspection in Looped Language Models](https://arxiv.org/abs/2607.18553), 2026.
66. <span id="ref-66" class="reference-anchor"></span>Xi Wang et al. [Full-bandwidth Transformer](https://arxiv.org/abs/2608.08888), 2026.
67. <span id="ref-67" class="reference-anchor"></span>Michael C. Mozer et al. [Recirculation](https://arxiv.org/abs/2608.17981), 2026.
68. <span id="ref-68" class="reference-anchor"></span>Zhaoliang Chen and Jie Fu. [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs](https://arxiv.org/abs/2609.01117), 2026.
69. <span id="ref-69" class="reference-anchor"></span>Wenlong Wang and Fergal Reid. [Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?](https://arxiv.org/abs/2609.01924), 2026.
70. <span id="ref-70" class="reference-anchor"></span>Ayhan Suleymanzade et al. [Thinking with Looped Flows](https://arxiv.org/abs/2609.11801), 2026.
71. <span id="ref-71" class="reference-anchor"></span>Zhiying Lu. [LoopVAE: Recurrent Depth Across Scales for Visual Tokenization](https://arxiv.org/abs/2609.11516), 2026.
72. <span id="ref-72" class="reference-anchor"></span>Jiaju Wu, Yi Hu, Muhan Zhang. [Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does](https://arxiv.org/abs/2609.39892v1), 2026-09-30。状态控制、路由替换及监督对照见 §4–6。
73. <span id="ref-73" class="reference-anchor"></span>Yong Xien Chng et al. [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305v1), 2026-09-30。内部循环与深监督见 §2，预算及组件消融见表 2、4、图 6。
74. <span id="ref-74" class="reference-anchor"></span>Weihao Liu et al. [Decoding Looped Transformers Better for (Almost) Free](https://arxiv.org/abs/2610.02185v1), 2026-10-01。LoopCD 见 §3；性能见表 1–3；解析计算量见附录 D。
75. <span id="ref-75" class="reference-anchor"></span>[LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models](https://arxiv.org/abs/2609.24196v1)，2026-09-21，v1。
76. <span id="ref-76" class="reference-anchor"></span>[ALoDLM: Adaptively Looped Diffusion Language Models](https://arxiv.org/abs/2610.04198v1)，2026-10-03，v1。
77. <span id="ref-77" class="reference-anchor"></span>[LiFT: Loop Flow Transformers](https://arxiv.org/abs/2610.05538v1)，2026-10-04，v1。
78. <span id="ref-78" class="reference-anchor"></span>[What Matters for Latent Reasoning with Flow Matching](https://arxiv.org/abs/2610.06666v1)，2026-10-05，v1。
79. <span id="ref-79" class="reference-anchor"></span>[Think Before You Paint: Recursive Latent Reasoning for Diffusion Models](https://arxiv.org/abs/2610.09876v1)，2026-10-07，v1。
80. <span id="ref-80" class="reference-anchor"></span>[From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers](https://arxiv.org/abs/2610.11472v1)，2026-10-08，v1。
81. <span id="ref-81" class="reference-anchor"></span>Yuxuan Zhou et al. [ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction](https://arxiv.org/abs/2610.24766v1). arXiv:2610.24766v1，2026-10-17。共享末层与 DDS 见 §3，草稿生成见 §4，训练与损失归一化见附录 A.1；[官方实验记录](https://github.com/ernie-research/ECHO)。 <a class="reference-backlink" href="#cite-81-1" aria-label="返回文献 81 首次引用处" title="返回正文">↩</a>
82. <span id="ref-82" class="reference-anchor"></span>[Looped Language Models: A Survey and Meta-Analysis of Recurrent-Depth Transformers](https://www.preprints.org/manuscript/202610.1404)。 <a class="reference-backlink" href="#cite-82-1" aria-label="返回文献 82 首次引用处" title="返回正文">↩</a>
83. <span id="ref-83" class="reference-anchor"></span>[GLM-5.4 技术公告](https://www.zhipuai.cn/zh/research/164)。 <a class="reference-backlink" href="#cite-83-1" aria-label="返回文献 83 首次引用处" title="返回正文">↩</a>
84. <span id="ref-84" class="reference-anchor"></span>[SenseNova-Loop-9B 官方模型卡](https://huggingface.co/sensenova/SenseNova-Loop-9B)。 <a class="reference-backlink" href="#cite-84-1" aria-label="返回文献 84 首次引用处" title="返回正文">↩</a>
85. <span id="ref-85" class="reference-anchor"></span>[Introducing Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/)。 <a class="reference-backlink" href="#cite-85-1" aria-label="返回文献 85 首次引用处" title="返回正文">↩</a>
86. <span id="ref-86" class="reference-anchor"></span>[Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/abs/2611.03998)。 <a class="reference-backlink" href="#cite-86-1" aria-label="返回文献 86 首次引用处" title="返回正文">↩</a>
87. <span id="ref-87" class="reference-anchor"></span>[Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design](https://arxiv.org/abs/2610.23036)。 <a class="reference-backlink" href="#cite-87-1" aria-label="返回文献 87 首次引用处" title="返回正文">↩</a>
88. <span id="ref-88" class="reference-anchor"></span>[ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals](https://arxiv.org/abs/2611.05234)。 <a class="reference-backlink" href="#cite-88-1" aria-label="返回文献 88 首次引用处" title="返回正文">↩</a>
89. <span id="ref-89" class="reference-anchor"></span>[Loopback Transformer: Towards a Unified View of Residual Looped Transformers](https://arxiv.org/abs/2611.07226)。 <a class="reference-backlink" href="#cite-89-1" aria-label="返回文献 89 首次引用处" title="返回正文">↩</a>
90. <span id="ref-90" class="reference-anchor"></span>[Closing the Loop: Practical Training Recipes for Looped Language Models](https://arxiv.org/abs/2610.00673)。 <a class="reference-backlink" href="#cite-90-1" aria-label="返回文献 90 首次引用处" title="返回正文">↩</a>
91. <span id="ref-91" class="reference-anchor"></span>[There Is No Free Loop: When Does Looping a Trained Model Pay for Its Inference Compute?](https://arxiv.org/abs/2610.42030)。 <a class="reference-backlink" href="#cite-91-1" aria-label="返回文献 91 首次引用处" title="返回正文">↩</a>
{: .references }
