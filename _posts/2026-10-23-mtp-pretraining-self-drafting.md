---
layout: post
title: "LLM 里的 MTP：从多步预训练到投机解码"
date: 2026-10-23
tags: [Pretraining, MTP, Speculative Decoding, LLM]
description: "沿多步预训练与投机草拟两条脉络，梳理 DeepSeek-V3、LightMTP、NextLat、SPS，以及 2026 年开放模型在参数共享、后训练和部署上的取舍。"
---

## 0. 写在前面

现在的模型报告经常提到 MTP，但同样三个字母，指向的东西差别很大：有的从预训练开始加未来 token 的 loss，有的给现成模型补一个 drafter；有的保留因果链逐步预测，有的一次并行算一整个 block。

之前写 [DFlash 到 dLLM Self-Spec]({% post_url 2026-06-29-dflash-to-dllm-self-spec-part-1-serving-speedup %}) 时，主要关心 draft、verify 和 serving runtime。加入预训练这条线以后，需要回答两个问题：**多步监督能不能让主干学得更好，以及怎样便宜地提出一段能被主干接受的草稿。**

这两个问题有交集，也各自发展出了不少工作。相似的辅助头可以服务于不同目标，多 token 输出也可以来自不同的推理过程。下面按它们解决的问题组织这段脉络。

资料截至 **2026 年 10 月 23 日**，图文与表述于 **2026 年 10 月 29 日**修订。

## 1. 两个起点：多步预测用于解码，还是用于预训练

多位置预测用于加速生成，早于今天这批 LLM。2018 年的 [Blockwise Parallel Decoding](https://arxiv.org/abs/1811.03115) 已经研究并行提出多个时间步、再保留验证通过的最长前缀。到 2022–2023 年，[speculative decoding](https://arxiv.org/abs/2211.17192) 给出了用廉价草稿与接受／拒绝校正保持目标采样分布的机制。这里先区分两个层次：怎样产生候选，以及怎样验证候选。

### 1.1 Medusa：给现成模型增加未来预测头

2024 年 1 月的 [Medusa](https://arxiv.org/abs/2401.10774) 在主干上附加多个 decoding heads，提出不同未来位置的候选，再用树状结构批量验证。它提供冻结主干和联合微调两类方案，主要评价解码加速；从预训练开始加入未来监督，是接下来另一条线的问题。

### 1.2 Meta MTP：把未来预测放进预训练目标

普通 next-token prediction（NTP）在位置 $t$ 上，根据前缀 $x_{\leq t}$ 预测 $x_{t+1}$。早期 MTP 最直接的扩展，是让同一个主干表示 $h_t$ 接多个预测头：

$$
\mathcal L_{\mathrm{MTP}}
=-\sum_t\sum_{k=1}^{n}\lambda_k
\log q_k(x_{t+k}\mid h_t).
$$

第一个头预测下一 token，第二个头预测下下个 token，以此类推。这里的 $n$ 包括普通 next-token 位置。

[Meta 的 Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737) 是这条路线的重要起点：共享 trunk、独立未来预测头，从预训练阶段引入多位置监督。原论文主方案的每个 head 包含额外的 Transformer 层，附录另有线性头变体。它报告了代码等生成任务上的收益，也展示了将预测头用于推理加速的可能性。

![Meta MTP 的共享 trunk 与多位置预测头](https://arxiv.org/html/2404.19737v1/img/main_fig_col.png)

*图 1：Meta MTP 的总览原图（论文 Figure 1）。上半部分是共享 trunk 与独立未来预测头，下半部分是论文的 MBPP 实验。结构图说明多个 offset 如何得到监督；图中的能力结果对应原论文实验配置。来源：[Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737)。*

多出来的目标会约束 $h_t$：这个表示既要支持当前预测，也要对后面的序列有用。独立头的分布乘积可以定义一个联合分布，但没有显式建模未来 token 之间的条件依赖。

举个例子，前缀是 `def`。未来可能是不同函数名、括号、参数。几个独立头分别猜后续位置，很可能各自给出了合理答案，拼在一起却不对应同一条 continuation。问题在于，第二个头没有显式看到第一个头实际选了什么。

两条线在这里已经分开了。Medusa 主要问额外预测头能否减少解码轮数，Meta MTP 还问未来监督能否改善 backbone。后面的因果预测、参数共享和递归训练，都会同时影响这两个目标，但影响未必同向。

## 2. EAGLE 与 DeepSeek-V3：两种训练制度下的因果预测

并行多头的一个限制，是后续预测没有显式依赖前面实际选出的 token。2024 年 1 月的原始 [EAGLE](https://arxiv.org/abs/2401.15077) 从草拟一侧处理这个问题：在 feature 层面自回归，并加入向前错开一步的 token 序列，减少未来 feature 的不确定性。

到了 2024 年底，DeepSeek-V3 将因果多步预测作为联合预训练目标。二者的输入结构和递归思路有联系，训练制度不同：专用 drafter 围绕既有目标模型学习，联合 MTP 则让未来 loss 参与主干优化。下面用 V3 的结构把这件事写具体。

[DeepSeek-V3](https://arxiv.org/html/2412.19437v2) 的 MTP 保留了因果链。每个额外预测模块接收前一步表示和移位后的 token embedding，再预测更后一个位置。

为了避免不同论文的深度编号混在一起，下面用 $h_t^{(0)}$ 表示主干输出，它负责预测 $x_{t+1}$；$k\geq1$ 表示额外的 MTP 深度。省略 attention mask 等实现细节，可以写成：

$$
\begin{aligned}
u_t^{(k)} &= M_k\left[
\operatorname{RMSNorm}(h_t^{(k-1)});\,
\operatorname{RMSNorm}(e(x_{t+k}))
\right],\\
h_t^{(k)} &= T_k(u^{(k)})_t,\\
q_t^{(k)} &= \operatorname{softmax}(W_{\mathrm{out}}h_t^{(k)}).
\end{aligned}
$$

$q_t^{(k)}$ 的标签是 $x_{t+k+1}$。也就是说，第一个额外模块看到 $x_{t+1}$，预测 $x_{t+2}$。embedding 和 output head 与主干共享，融合投影和 Transformer block 承担额外的预测计算。[结构见报告 §2.2](https://arxiv.org/html/2412.19437v2#S2.SS2)。

训练时这里用 ground-truth token 做 teacher forcing；推理时，递归输入的 token 来自 draft。前一个 token 会影响后一个预测，未来依赖接上了，训推之间的差异也跟着进来了。

### 2.1 MTP loss 本来就能更新主干

只要没有 stop-gradient，辅助模块的 loss 就可以沿着 $h_t^{(0)}$ 反传进 backbone，同时训练主干与额外参数。

真正需要问的是：**新增预测任务，有多少由辅助模块吸收，又给主干留下了怎样的约束？**

对应的计算图很简单：

```text
backbone ── h ── auxiliary block ── output head ── future-token loss
```

loss 可以同时更新 backbone 和 auxiliary block。如果 auxiliary block 有独立参数，它就有空间专门适配未来位置；如果这个 block 与主干层绑在一起，多步目标会共同约束同一组变换参数。两种结构都有梯度回到主干，区别在参数自由度和优化耦合方式。

这说明两种结构的优化约束不同；哪种更有利于 downstream，需要对照实验判断。

### 2.2 训一层，递归用多次，问题在哪里

第一次使用辅助模块时，输入 hidden state 来自主干；第二次使用时，输入变成了模块自己产生的 hidden state。这两个表示分布未必一样。再加上 teacher forcing 换成 sampled token，误差会继续影响后续预测。

因此，递归 draft 至少有两个训练问题：模块是否见过自身递归产生的表示，以及它是否适应实际 draft token 的分布。增加训练展开深度主要处理前一部分；是否使用模型生成轨迹、如何构造训练输入，还会影响后一部分。

## 3. 联合 MTP 的实际取舍：训练深度、参数共享与部署适配

DeepSeek-V3 之后，一个实际问题浮出来：为了训练主干值得加几层，和为了推理加速值得用几层，答案可能不同。

### 3.1 MiMo：先训练一层，再补训推理预测层

2025 年的 [MiMo-7B](https://arxiv.org/html/2505.07608#S2.SS2) 报告，初步实验中增加预训练 MTP 层没有带来更多收益，因此预训练只用一层。之后它复制这层以构造额外预测层，冻结主干和原来的第一层，专门微调新增层以加速推理。

在 MiMo 的实验里，增加预测层的预训练收益与草拟收益并不同步：预训练没有继续受益，专门适配推理时却用得上。这个观察的范围是该报告的实验设置。

### 3.2 GLM-5：共享辅助参数，多步展开训练

[GLM-5 的做法](https://arxiv.org/html/2602.15763) 很直接：训练展开 3 个 MTP 层，但共享参数。用上面的记号，就是把各深度独立的 $T_k$ 改成共享的 $T_{\mathrm{aux}}$。

这样，同一模块在训练里会接触不同深度的输入，部署时也不需要为每个深度保存一份预测模块。报告在私有 prompt 集、相同 4 个 speculative steps 下，对比了 GLM-5 与 DeepSeek-V3.2，接受长度分别为 2.76 和 2.55。这是跨模型比较，接受长度的差距也包含主干和其他训练差异；参数共享的单独作用还需要同模型消融。[报告 §2.1](https://arxiv.org/html/2602.15763#S2.SS1)

参数共享也有边界。它省掉多份权重和 optimizer state，不会让多步 forward/backward 免费。不同递归深度的 activation、attention 状态如何保留，仍取决于实现；共享权重也不自动意味着可以共享 KV cache。

从这里往后，“MTP 有几层”就不太够用了。至少要把物理参数份数、训练展开深度和推理 draft 长度分别记下来。

### 3.3 IQuest-Q1：联合训练以后，转换成递归 drafter

2026 年 10 月的 [IQuest-Q1](https://arxiv.org/html/2610.30949v1) 则给出了单模块递归的部署实例。这个 320B / 15B active 的 MoE 从预训练开始使用两层 MTP；完成训练流程以后，再用专门阶段将第一层改造成递归 drafter，丢掉第二层。推理时同一模块复用七个 drafting steps，并把 attention 改成较小窗口的 SWA。[报告 §2.1 与附录](https://arxiv.org/html/2610.30949v1#S2.SS1)

它把两个阶段的需求分开了：预训练阶段用多步监督帮助 backbone 学习，部署前再调整模块，让它适应递归输入和长上下文成本。七步是草拟长度，不是七个 token 都会被接受。

因此，联合预训练、辅助层共享、主干层复用和后续 drafter adaptation，应当分别记录。“支持 MTP”不足以说明这套设计。

和 MiMo 对照，两者都在联合预训练之后适配预测模块。MiMo 复制并训练额外预测层，IQuest 则保留一个模块递归复用，再用小窗口 attention 控制每一步的上下文开销。

## 4. 专用 drafter 的演进：递归、并行与半自回归

另一条线继续围绕现成 target model 优化 drafter。这里也会出现多 token prediction，但不应把它们全部算作改善预训练能力的方法。

### 4.1 EAGLE-3：让递归草拟摆脱 feature 回归约束

[EAGLE-3](https://arxiv.org/abs/2503.01840) 在 2025 年调整了 EAGLE 的训练方式：放弃对目标 hidden feature 的直接回归约束，改为融合主干多层特征、直接预测 token，并通过 training-time test 让训练接触递归使用的状态。它不等于联合预训练 MTP，但讨论“模块第二次看到的输入是谁产生的”时，这篇值得一起读。

### 4.2 DFlash 与 DSpark：串行草拟成本和无效验证

当逐 token drafter 的串行调用开始限制加速，2026 年的 [DFlash](https://arxiv.org/abs/2602.06036) 改用轻量 block diffusion drafter，在 target context features 的条件下并行生成草稿。这样减少了草拟侧的串行 forward，但块内依赖和后缀接受率仍然影响收益。

![DFlash 的 target 特征注入与并行草拟](/assets/img/blog/mtp-survey/dflash-system.png)

*图 2：DFlash 推理设计原图，对应论文 Figure 2。target 的多层 hidden features 被融合并投影为 draft 各层的 K/V，供 drafter 并行预测。候选随后交给 target 验证，这一步未在原图展开。图中的特征注入与“回归 target 最终层 KV”是不同的操作。来源：[DFlash 论文](https://arxiv.org/abs/2602.06036)及[作者公开原图](https://raw.githubusercontent.com/jianc99/jianc99.github.io/master/images/dflash_system.png)。*

[DSpark](https://arxiv.org/abs/2607.05147) 在并行 backbone 后加入轻量 sequential module，同时用 confidence 和 engine 吞吐曲线调度验证长度。它处理的是两个具体问题：块内 token 怎样保持依赖，以及高并发下哪些后缀值得送进 target。这里和预训练 MTP 的共同点是需要高质量未来预测，评价重点则落在真实服务效率。

### 4.3 DeepSeek-V4.1-Flash：把专用 drafter 纳入后训练流程

V4.1-Flash 的报告明确说，backbone 预训练阶段省去了 MTP 模块。之后引入的 DSpark 用 3 个 Transformer blocks、128-token SWA，一次并行算出 5 个草稿位置的 base logits，再由轻量 Markov head 补位置间依赖。confidence head 估计接受概率，scheduler 结合 engine 吞吐曲线决定验证长度。它先冻结 backbone 单训 drafter；后训练期间继续跟随 policy 更新，但草稿 loss 不反传主干。[技术报告](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS3)

具体机制在之前的 [DSpark 那篇]({% post_url 2026-06-29-dflash-to-dllm-self-spec-part-3-dspark %}) 里展开过。这个训练流程明确把草稿目标与主干目标分开，同时在后训练阶段继续维护二者的对齐。

主干已经训好，或者 policy 在 RL 中持续变化时，单独维护 drafter 可以直接围绕当前 policy 和服务请求优化。联合预训练的优势在于更广的数据覆盖，两者最终要在实际请求分布上比较质量与成本。

## 5. 近期改进：各自补哪一块

沿着这两条线，近期还有几类尝试，分别修改参数约束、监督形式、补训成本和调度方式。

### 5.1 NextLat：给定 token，学习状态怎样转移

[NextLat](https://arxiv.org/html/2511.05963v4) 把问题写成条件状态转移：给定当前 hidden state 和下一个 token，预测读入这个 token 以后的 hidden state，再递归展开。真实未来表示作为 stop-gradient target，训练同时使用 latent regression 和经过冻结 readout 的分布 KL。

这个条件很关键。$h_t\to h_{t+1}$ 和 $(h_t,x_{t+1})\to h_{t+1}$，看起来只多一个输入，任务本质已经变了。前者必须处理多种未来，后者是在已知 token 路径上学习状态更新。

论文从理想条件下的 NTP 一致性与状态转移一致性出发，讨论 predictive belief state。这个理论解释提供了一个方向：模型可以通过下一 token 的观测更新内部状态。有限模型能否逼近这些条件、是否带来普遍 NTP 增益，仍要由实验回答，不能把定理直接当作预训练保证。

### 5.2 LightMTP：未来监督能不能更轻

[LightMTP](https://arxiv.org/html/2610.06031v1) 用两层 GELU MLP，从当前位置的最终 hidden state 预测同一模型未来位置的中间层表示。target 来自同一次 causal forward，并做 stop-gradient；监督采用归一化后的 cosine loss。S 版本预测一个未来 offset，C 版本一次输出多个 offset 的拼接表示，推理时可以移除预测器。

省略各项归一化，可以把任务写成：

$$
f_k(h_t^{(L)})\approx\operatorname{sg}(h_{t+k}^{(\ell)}).
$$

它和 NextLat 的区别在于，预测器没有得到中间真实 token。这里学的是对未来表示的预判，不是给定 token 后的状态更新。它也没有直接要求每个未来位置都经过大词表 softmax。

LightMTP 的主要吸引力是监督成本。论文 1.38B 设置中，NTP 与默认 LightMTP-C 的 FineWeb-Edu byte-level PPL 都是 1.683，CORE 分别为 40.43 和 40.30；这组结果接近保持 NTP 表现，收益随 horizon 和任务变化。实验采用随参数量变化的 token 预算，并非严格等 token 或等 FLOPs 比较。[实验与消融](https://arxiv.org/html/2610.06031v1#S5)

读这类 latent loss，我会追问两件事。第一，未知未来有多种可能，单个预测向量是否只是学到了平均方向？第二，cosine 更接近，究竟保留了哪些对 token 决策有用的信息？这需要 readout、drafter 或任务探针检验，再测量接受率与真实加速。

### 5.3 SPS：存状态和做预测，是否应该用同一份表示

[The State-Prediction Separation Hypothesis](https://arxiv.org/abs/2607.01218) 提了一个更靠近架构的问题：同一条计算流既要产生当前预测，又要留下供后文读取的 KV，这两个职责是否应该分开？SPS 在真实 token 后插入 `<predict>` 位置，让输入流保存持久状态，预测流承担 NTP，预测流 KV 只在短窗口保留。

![SPS 及其控制组的 attention mask](/assets/img/blog/mtp-survey/sps-attention-masks.png)

*图 3：SPS 作者公开的 attention-mask 原图。蓝色位置保存持久 key，橙色位置只在窗口内保留；黑色三角标记计算预测 loss 的位置。比较 SPS、Delayed State 和 2× Memory，可以区分“多做一步计算”“多存一份状态”与“分开状态和预测”的作用。来源：[SPS 论文](https://arxiv.org/abs/2607.01218)及[官方图文件](https://github.com/lil-lab/sps/blob/main/figures/attention_masks.png)。*

这里分开的是激活和可见性，权重仍然共享。state stream 也仍收到当前 NTP loss 的梯度，因为预测流需要读取它；不能把 separation 理解成 stop-gradient。

[Almost Free SPS](https://arxiv.org/abs/2609.03807) 再让预测流完全不写 K/V，只用 query 读取输入流状态。它还通过优化 kernel、共享 FFN 和在训练后段才启用分离来减少成本。预测流仍有额外 FLOPs；“几乎免费”的推理表现依赖测试中的硬件与负载，不能直接推广到所有 batch。[方法和成本分析](https://arxiv.org/html/2609.03807)

这条线给 MTP 的启发是：增加未来任务以前，先想清楚监督在要求哪份表示承担什么职责。让历史状态保留更多可供未来读取的信息，与让当前预测表示直接拟合多个未来标签，可能产生不同的优化结果。

SPS 的 NTP 结果对应保留预测流的完整模型；若要将其用于 dLLM 草拟，还需要测量 drafter 的学习成本和接受率。与它相邻的 [Decode-Branch](https://arxiv.org/abs/2608.12385) 也使用只读分支，但最终输出包含主路与分支的概率混合，同样要区分完整模型与主路单独的表现。

![SPS 不同计算流收到的未来与当前梯度之比](/assets/img/blog/mtp-survey/sps-gradient-analysis.png)

*图 4：SPS 论文 Figure 5(a) 的梯度分析，取自作者公开图文件。横轴是未来 loss 的 offset，纵轴是该 loss 相对当前 loss 的梯度范数比；实线和虚线区分输入流与预测流。它展示未来监督在两条计算流中的分配，语言建模和解码收益则由各自的实验衡量。来源：[论文分析](https://arxiv.org/html/2607.01218#S5)及[官方原图](https://github.com/lil-lab/sps/blob/main/figures/gradient_params_fp.png)。*

更早的 [YOCO](https://arxiv.org/abs/2405.05254) 已经研究“先形成全局 KV，再由 cross-decoder 复用”。它与 SPS 的问题设置并不相同：共享缓存解决的计算和存储问题，不自动等同于消除 state/prediction 的任务冲突。沿着状态如何产生、谁能读取以及 loss 怎样回传，可以比较这些设计。

### 5.4 ECHO：未来监督是否应该直接约束主干最后一层

[ECHO](https://arxiv.org/html/2610.24766v1) 研究另一种参数约束：递归使用的 Transformer block，就是 backbone 的最后一层。除了轻量融合层，未来预测使用的参数也参与普通 next-token 生成。

两种共享的区别可以写成：

```text
GLM-5：backbone ── shared auxiliary block ── 再用同一个 auxiliary block
ECHO ：backbone 最后一层 ── 再用这层 ── 再用这层
```

ECHO 还做了 Dynamic Depth Sampling。每个 optimizer step 抽一个深度 $D$，只展开到这个深度。例如最大深度为 3，抽样概率为 $(0.6,0.3,0.1)$，则平均额外展开深度为 1.5；各深度实际被训练的频率是 $P(D\geq k)$，即 $(1,0.4,0.1)$。

作者在 10B-A1B MoE、600B 训练 token 的受控设置下，报告了相对辅助 MTP 的能力收益，以及超出训练深度继续 draft 的结果。平均深度 1.5 的配置接近完整深度 3 的 benchmark 均值，MTP 增量计算约减半。这里省的是额外展开开销，总预训练成本没有减半；能否推广到更大规模，还需要验证。[ECHO §3–6](https://arxiv.org/html/2610.24766v1#S3)

这个设计对应的实验问题是：给定一份额外训练预算，是增加辅助模块的表达能力更划算，还是进一步约束主干现有参数更划算？做对照时，最好同时保留 matched-token 和 matched-FLOPs 两组比较，否则容易把多花计算和结构收益混在一起。

### 5.5 补训预测头：已有模型需要付出多少成本

Match the Distribution, Not the Compute 关心现成模型补训 MTP 的成本。作者冻结 Qwen3-8B，用约 2.5B 个目标生成 token 训练链式预测头，并报告有竞争力的预期加速。文中也研究放宽验证与动态启用预测头。读结果时要把严格验证和允许分布偏移的部分分开，预期加速也需要真实 engine 的 wall-clock 结果来检验。[论文，2026-10-01](https://arxiv.org/abs/2610.00888)

### 5.6 AMTP：在线选择草拟长度

AMTP 把草拟长度变成在线决策。它从验证反馈中估计草稿前缀能活多长，再选择预期 tokens/s 更好的 horizon，不需要重训模型，也保留原本的严格 verifier。作者相对固定窗口报告了最高约 10% 的吞吐收益。这里优化的是给定 drafter 的使用方式。[Adaptive Multi-Token Prediction，2026-10-20](https://arxiv.org/abs/2610.28619)

### 5.7 无训练探测与无验证生成

还有两条边界路线。[ESP](https://arxiv.org/abs/2603.17942) 用 embedding-space probing 在冻结模型上构造未来候选，再批量验证；[MTP via Self-Distillation](https://arxiv.org/abs/2602.06019) 则通过在线蒸馏改造模型，不依赖额外 verifier，接受一定质量变化。它们都涉及多 token 生成，但训练约束和质量保证不同，适合单列比较。

## 6. 2026 年开放模型里的设计选择

前面是方法与训练流程，下面回到模型发布，把能核实的策略放在同一张表里。这里的开放模型包含开放权重发布，具体许可证仍以对应仓库为准。

| 模型 | 已披露的预测策略 | 读这份设计时要注意什么 |
|---|---|---|
| GLM-5 / GLM-5.2 | GLM-5 多深度共享辅助参数；5.2 官方称改进 MTP，接受长度最高增加 20% | 5.2 公告未披露完整架构改动。[报告](https://arxiv.org/html/2602.15763)、[公告](https://z.ai/blog/glm-5.2) |
| Qwen3.8-Flash-Next | 模型卡写明 1 层 MTP、multi-step training，并单列约 4B MTP 参数 | 单层可以多步训练；卡片不足以确定展开步数和完整 loss recipe。[模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| Nemotron 3 Ultra | 2 个共享权重的 MTP 层；SFT 保留 MTP objective；另有冻结主干、on-policy 补训 head 的 MTP Boosting | 除了预训练，还用不同递归步的 hidden 混合模拟推理输入，继续适配 drafter。[技术报告 §3.4](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) |
| Gemma 4 系列 | 发布配套 MTP drafters，使用 target activations 并共享 KV cache | 重点落在部署组件和上下文计算复用；不能直接套用联合预训练 MTP 的能力结论。[官方说明](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/) |
| DeepSeek-V4.1-Flash | backbone 预训练不加 MTP；预训练后引入 DSpark，后训练时持续维护 drafter，对主干截断草稿梯度 | 联合预训练 MTP 并非所有开放模型的必选项。[报告 §2.4.3](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS3) |
| IQuest-Q1 | 两层联合预训练，再将第一层改为七步递归、小窗口 drafter | 训练结构与最终部署结构不同。[报告](https://arxiv.org/html/2610.30949v1) |
| Kimi-K3 | 公开配置中 `num_nextn_predict_layers: 0` | 只能确认这份配置没有声明原生 next-n 层，不能反推内部训练和服务策略。[配置](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json) |

表中按有明确披露的版本记录。后续版本若只有性能公告，其结构细节仍以新报告或配置为准。

因果递归 MTP、并行 diffusion-style draft、半自回归 head 和后装 drafter 仍然并存。多份报告共同强调的是，在 SFT/RL 或部署前维护 draft 与最终 policy 的一致性。

## 7. 接受长度怎样变成实际加速

MTP 的部署指标不能停在“第几个头准确率多高”。投机解码提交的是连续前缀，前面拒绝以后，后面猜得再好也通常用不上。

设 $A_k$ 表示前 $k$ 个 draft token 全部被接受的事件。在每轮额外提交一个目标侧 correction/bonus token 的常见设置下，忽略 EOS 等边界：

$$
\mathbb E[N_{\mathrm{commit}}]
=1+\sum_{k=1}^{K}P(A_k).
$$

如果 $a_i=P(\text{第 }i\text{ 个接受}\mid A_{i-1})$，则 $P(A_k)=\prod_{i=1}^{k}a_i$。这里用的是条件概率链式分解，不需要假设各位置独立。前缀接受和 bonus token 的机制可参见 [speculative decoding 原始工作](https://arxiv.org/abs/2211.17192)。

做一个假设例子：前四步的条件接受率是 0.9、0.8、0.6、0.5，对应 prefix survival 为 0.9、0.72、0.432、0.216，期望提交量为 3.268。再加第五步，即使它的条件接受率有 0.5，也只增加 0.108 个期望 token。为了这 0.108，系统还要承担额外 draft、verify 和状态维护成本。

所以应该把收益写成：

$$
\mathrm{tokens/s}\approx
\frac{\mathbb E[N_{\mathrm{commit}}]}
{T_{\mathrm{draft}}+T_{\mathrm{verify}}+T_{\mathrm{runtime}}}.
$$

这只是便于拆问题的近似，真实 serving 还有 batching 和流水重叠。它至少提醒我们：增加深度会改分子，也会改分母。权重共享能省参数，SWA 能压上下文开销，block draft 能减串行调用，动态 verify 能少花无效 target compute；它们改善的是不同部分。

### 7.1 接受率还取决于怎样采样

对同一已接受前缀，若草稿 token 从 $q$ 采样，采用标准严格接受规则，则平均单步接受概率为：

$$
\sum_xq(x)\min\left(1,\frac{p(x)}{q(x)}\right)
=\sum_x\min(p(x),q(x))
=1-\operatorname{TV}(p,q).
$$

这解释了为什么只拟合正确 token 的 top-1 还不够：严格 sampling 的接受率依赖两个分布的重合程度。temperature、top-p 等配置改变了有效分布，也会改变这个重合程度；公式要在实际使用的分布和对应校正规则下理解。[接受规则与校正分布见原论文](https://arxiv.org/abs/2211.17192)。

数学、代码、自由文本和 RL rollout 的分布不同，greedy 下的接受长度不能直接搬到非零温度的 rollout 上。对 reasoning model 做完 SFT/RL 以后，除了问主干能力涨了多少，还应该再问一次：旧 MTP 现在与新 policy 对得上吗？

## 8. 如果要做一轮实验

我会先定清楚额外 compute 的用途。如果主要目标是 backbone 能力，先做 NTP、独立辅助层、共享辅助层三组，再加 ECHO 式最后层共享；如果主要目标是给现成 checkpoint 加速，就从冻结主干、单独训练 drafter 开始。两条实验都能做，但不必一开始就混成同一张收益表。

| 要回答的问题 | 至少要保留的比较 |
|---|---|
| 多步监督有没有改善主干 | 相同 token budget 与相同 FLOPs budget；NTP loss 加下游能力 |
| 共享参数是否值得 | 独立辅助层、共享辅助层、主干层共享；固定数据与其他 recipe |
| 递归使用是否稳定 | 每个位置的条件接受率、prefix survival，区分训练深度内外 |
| 加速有没有兑现 | 同硬件、同 engine，记录 draft / verify / runtime 延迟及真实吞吐 |
| 后训练是否破坏对齐 | base、SFT、RL checkpoint 分阶段测；实际采样设置与请求类型 |

实现时还有几个不太显眼的地方。shifted labels 要在 packed document 边界截断，不能把下一篇文档开头当未来标签；assistant-only SFT 的 loss mask 要与移位后的预测位置对齐；共享层的 optimizer state 和 checkpoint 转换要保持一致，避免训练里 tied、导出后却分成两份。递归调用的 position、mask 和 KV 生命周期也要逐步核对。

做这轮实验时，loss 权重、展开深度和共享范围要一起记录。先跑稳基线，再一次只改一个关键变量，才能分辨收益来自哪里。

## 9. 把架构与训练流程放在一起

沿这两条线看下来，MTP 的几个基本选择可以独立讨论：未来位置是否显式依赖前面的 token，辅助模块是否跨深度共享，未来 loss 是否更新主干，以及训练结束后是否需要重新适配 drafter。

![MTP 架构设计与训练推理范式总结](/assets/img/blog/mtp-survey/mtp-design-paradigms.png)

*图 5：按本文内容生成的架构与训练／推理范式总览。上半部分比较独立多头、因果递归、参数共享、latent 监督、状态／预测分工与 block draft；下半部分整理三种常见训练与部署流程。图中方框省略了 norm、投影与 attention mask 等细节；具体 token 索引以正文公式为准。SPS 属于表征分工相关工作，NextLat 的转移以给定 token 为条件；它们与多步 token 监督的联系分别见第 5 节。总览为图像生成示意；图 1–4 来自论文或作者公开仓库。*

这些设计可以组合。共享参数没有规定未来标签怎样给，block draft 也没有规定是否从预训练开始联合优化。判断一套方案是否值得采用，仍需要分别测主干能力、后训练适配成本，以及给定服务负载下的实际生成速度。

## 参考资料

- [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [MiMo: Unlocking the Reasoning Potential of Language Model](https://arxiv.org/abs/2505.07608)
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
- [ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction](https://arxiv.org/abs/2610.24766)
- [IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems](https://arxiv.org/abs/2610.30949)
- [DeepSeek-V4.1-Flash Technical Report](https://arxiv.org/abs/2609.19969)
- [Qwen3.8-Flash-Next Model Card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Nemotron 3 Ultra Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
- [Gemma 4 MTP Drafters](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
- [LightMTP](https://arxiv.org/abs/2610.06031)
- [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/abs/2511.05963)
- [The State-Prediction Separation Hypothesis](https://arxiv.org/abs/2607.01218)
- [Almost Free State Prediction Separation](https://arxiv.org/abs/2609.03807)
- [Decode-Branch Transformers](https://arxiv.org/abs/2608.12385)
- [You Only Cache Once](https://arxiv.org/abs/2405.05254)
- [Match the Distribution, Not the Compute](https://arxiv.org/abs/2610.00888)
- [Adaptive Multi-Token Prediction](https://arxiv.org/abs/2610.28619)
- [Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](https://arxiv.org/abs/2603.17942)
- [Multi-Token Prediction via Self-Distillation](https://arxiv.org/abs/2602.06019)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
