---
layout: post
title: "连续 dLLM 导读（选读 A）：连续与离散的混合模型"
date: 2026-11-19
last_modified_at: 2026-11-19
visibility: unlisted
sitemap: false
series: continuous-dllm
series_order: 9
tags: [dLLM, Diffusion, Flow Matching]
math: true
description: "逐项推导 CANDI、CADD、CCDD 的连续与离散联合状态、训练目标和采样，分析共享 latent 怎样保留跨位置依赖。"
permalink: /2026/11/19/continuous-dllm-09-hybrid-models.html
---

混合模型把 token 或 mask 的显式状态与连续信息通道结合起来，与你已有的离散 dLLM 工作直接相连。本篇围绕三个问题展开：连续通道保存什么，怎样协调多个位置，以及为此增加多少计算。

第 3 篇之后可先读直觉；准备复现时，建议补完第 5、6 篇。

## mask 状态和连续草稿保存的信息不同

一个 mask 只标明该位置尚未确定。连续辅助状态则可以在相邻去噪步骤之间保存不同程度、不同方向的候选信息。它可能是词表向量的有噪表示，也可能是由网络形成的 latent。

离散模型的隐藏特征也能承载丰富语义。这里关注的是采样器在两轮之间显式保留什么，以及这些状态怎样继续更新。

## 共享潜变量为什么能连接条件独立的输出头

考虑一个一般性的潜变量模型。先抽取同一个 z，再让各位置根据 z 和当前状态 x 生成：

$$
p(w\mid x)
=\int p(z\mid x)\prod_{\ell=1}^{L}p(w^\ell\mid z,x)\,\mathrm dz.
$$

给定 z 时各位置可以独立，积分掉共享 z 后，最终 token 却可以相关。比如令 z 的符号决定 AA 或 BB，就恢复了第 1 篇的教学分布。

多个位置必须使用协调的潜变量信息。若各位置分别独立选择分支，再独立生成，共同选择产生的关联就会消失。

## 手算共享 latent 产生的相关性

令 z 以相同概率取 −1 或 +1。给定 z=+1 时，两个位置各自以 0.9 概率输出 A；给定 z=−1 时，各自以 0.1 概率输出 A。两个位置在给定 z 后独立。

把 z 积分掉，每个位置的 A 概率仍是一半；但两个位置都取 A 的概率为：

$$
P(AA)=\frac12(0.9)^2+\frac12(0.1)^2=0.41.
$$

类似地，BB 为 0.41，AB 和 BA 各为 0.09。若直接独立使用无条件边缘，四项却会各为 0.25。

| 组合 | 共享 latent 后的联合概率 | 独立使用无条件边缘 |
| --- | --- | --- |
| AA | 0.41 | 0.25 |
| AB | 0.09 | 0.25 |
| BA | 0.09 | 0.25 |
| BB | 0.41 | 0.25 |

把条件概率从 0.9、0.1 推到 1、0，就恢复严格的 AA/BB 分布。这个变化展示了共享 latent 如何逐渐加强位置间的关联。

关键是两个位置使用同一次 z 抽样。实际系统中，应区分“共享一个随机状态”与“各位置独立抽各自状态”，并检查后续网络是否又把它们联合处理。

## 用协方差把共享 latent 的作用写出来

把两个位置是否为 A 分别记成 0/1 指示变量 U、V。前面的例子有 E[U]=E[V]=0.5，而 E[UV]=P(AA)=0.41，所以：

$$
\mathrm{Cov}(U,V)=0.41-0.5\times0.5=0.16.
$$

一般的全协方差公式是：

$$
\mathrm{Cov}(U,V)
=\mathbb E[\mathrm{Cov}(U,V\mid Z)]
+\mathrm{Cov}(\mathbb E[U\mid Z],\mathbb E[V\mid Z]).
$$

若给定 Z 后两个位置独立，第一项为零。第二项仍可以非零：同一个 Z 会让两个位置的条件均值一起升高或降低。这就是“给定 latent 独立，边缘上仍相关”的完整统计解释。

反过来，把 latent 先平均成一个固定草稿，再独立采样 token，可能抹掉这类共同变化。保留一份共享随机选择，与只保留它的平均值，是不同的生成机制。

## CANDI 为什么把两种“变坏”分开建模

高维 one-hot 加 Gaussian 时，有两个不同量。第一个是某个错误坐标超过正确坐标的比例；第二个是 argmax 已经不再是正确 token 的概率。它们都叫“破坏身份”会造成混淆。

取 Y=eᵧ+σε，给定一个错误类别 k：

$$
r(\sigma)
=\Pr(Y_k>Y_y)
=\Phi\!\left(-\frac1{\sqrt2\sigma}\right).
$$

r 不依赖 V。它衡量的是一个错误竞争者的排名。整个 token 的错误率则为：

$$
\rho_V(\sigma)
=1-\int\phi(z)\Phi(z+1/\sigma)^{V-1}\,\mathrm dz.
$$

它依赖 V，因为只需一个竞争者超过正确坐标，argmax 就变了。因此一个噪声日程可能让绝大多数错误坐标仍排在正确坐标下面，同时在大词表上已经改变不少 token 的身份。训练的分类难度和连续坐标的逐步变化，会集中到不同时间区域。

[CANDI v3](https://arxiv.org/abs/2510.22510v3) 明确区分这两类变化。为对应论文，本节暂用 corruption time τ：τ=0 是干净，τ=1 是强噪声，与主线时间 t=1−τ 相反。

### 训练样本怎样构造

先抽一个保留标记 Mᵢ，1 表示该位置保持为已知干净 token：

$$
M_\tau^i\sim\operatorname{Bernoulli}(a(\tau)),
\qquad a(\tau)=1-\tau.
$$

另构造 Gaussian 状态 Zτ=C+σ(τ)ε，再形成实际输入：

$$
X_\tau^i
=M_\tau^i C^i+(1-M_\tau^i)Z_\tau^i.
$$

这样，a 控制多少位置已经可靠，σ 控制未确定位置还保留多少连续证据。网络在同一输入中既能利用干净 anchor，也能处理未决定位置的软信息。它学习原 token 的后验，主要使用 masked-diffusion 风格的加权 CE。

论文还令 rank degradation r*(τ) 在一个有限范围内线性变化，由上一条 Gaussian 公式反解：

$$
\sigma(\tau)
=-\frac1{\sqrt2\,\Phi^{-1}(r^*(\tau))},
\qquad 0<r^*(\tau)<\tfrac12.
$$

于是离散保留率与连续 rank 变化可以分别标定。它并不是让 argmax 错误率也被迫等于同一条曲线，三个量不能混用。

### 反向一步同时决定什么

从 τ 走向更干净的 s<τ。已揭示的位置保持原样；未揭示位置以：

$$
p_{\mathrm{reveal}}
=\frac{a(s)-a(\tau)}{1-a(\tau)}
=\frac{\tau-s}{\tau}
$$

的概率变成干净 token。例如 τ=0.8 到 s=0.6，每个未决位置有 0.25 概率被提交。提交时按模型后验选择 token；仍未决的位置继续按连续 score 做更新。两类动作使用同一次网络得到的后验，但对状态产生不同作用。

~~~text
q = model(整段混合状态, corruption time)
对已经揭示的位置：保持 token 与干净向量
对仍未揭示的位置：
    以 p_reveal 决定本轮是否提交
    若提交：从 q 选 token，设为它的干净表示
    若不提交：由 q 的均值形成 score，继续连续更新
~~~

### 为什么它还能避免每步维护完整 V 维运算

Transformer 实际读取的是输入投影 Y=W X。线性投影可以搬进连续更新，但还会出现期望表示 Wq。CANDI 的近似采样器从 q 抽 token，查表得到 W eₖ，用一次或多次 Monte Carlo lookup 近似 Wq，再在投影空间更新缓存状态。

这省去了部分大矩阵乘法和持续存储完整 noisy one-hot 的开销；仍需预测词表 logits。一次 lookup 的条件期望正确，不表示有限步随机近似与完整 ODE 的样本分布完全一样，论文将它作为有实验支持的近似算法。

另外，初始 V 维 Gaussian 投影后具有 W Wᵀ 协方差。随手换成 d 维标准 Gaussian，通常又会改变先验；如果要采用这种替换，需要重新定义并训练相应路径，而不能把它当作纯粹的内存优化。

## CADD 怎样让 mask 位置保留连续语义

[CADD](https://arxiv.org/abs/2510.01329) 也保留离散 mask，但它的连续变量是词向量通道。仍用数据到噪声的时间记号，干净词向量为 E(wᵢ)。在时刻 τ 的边缘分布中：

$$
Z_\tau^i=
\begin{cases}
E(w_i),&X_\tau^i\ne\mathrm{MASK},\\
\sqrt{\bar\gamma_\tau}E(w_i)
+\sqrt{1-\bar\gamma_\tau}\,\varepsilon^i,
&X_\tau^i=\mathrm{MASK}.
\end{cases}
$$

所以离散位置变成 MASK 后，并没有把所有词特定信息同时抹掉；连续通道仍保留逐渐衰减的线索。其前向链的设计保证可以直接从这条边缘分布采样训练输入，不必重放“这个位置具体在哪一轮开始 mask”。

实现上，可见位置已经由 discrete embedding 提供词向量，额外连续分支可以置零；mask 位置则把 noisy continuous embedding 与离散 MASK 的表示融合，再送入共享 Transformer。模型输出每位置的 token 后验。

默认简化训练主要是 masked token CE：

$$
\mathcal L_{\mathrm{CADD}}
=\mathbb E\left[
-\sum_{i:X_\tau^i=\mathrm{MASK}}
\log q_\theta(w_i\mid X_\tau,Z_\tau,\tau)
\right].
$$

连续干净估计由同一后验加权词表得到：

$$
\widehat Z_0^i=\sum_kq_\theta^i(k)E_k.
$$

完整联合 ELBO 还包含连续 KL，对应一个带权向量 MSE；论文主实验选择较简单的 CE 训练，并不是声称这条 CE 在有限模型上逐项等于完整 ELBO。

### 一次采样更新如何使用两种状态

网络先联合读取当前 mask 序列与连续状态。若某位置被选中揭示，抽一个 token，把它设为已知上下文，其连续状态也设成对应干净 embedding；若仍保留 mask，就用 Gaussian reverse kernel 更新它的连续状态，干净均值来自上一式。已揭示位置保持不动。

假设某 mask 位置的 q 给 A 0.75、B 0.25，而 E_A=(1,0)、E_B=(0,1)。如果它还不揭示，连续更新可以朝均值 (0.75,0.25) 移动，保留两种可能；如果它被揭示为 A，则状态变成干净 E_A，后续位置可以把 A 当作已提交条件。

这个例子把“保留候选信息”与“宣布一个离散决定”分开了。CADD 不必先生成一整份最终 latent，再运行另一套 token decoder；两种状态在同一采样过程中配合推进。

## CCDD 为什么使用真正的双头联合去噪

[CCDD v2](https://arxiv.org/abs/2510.03206v2) 的连续目标可以来自预训练语言编码器的上下文状态，因而包含超出单个词表向量的语义信息。它对 clean token w 与 clean latent Z₀ 分别加离散噪声、Gaussian 噪声：

$$
q_\tau(X_\tau,Z_\tau\mid w,Z_0)
=q_\tau^{\mathrm{disc}}(X_\tau\mid w)
q_\tau^{\mathrm{cont}}(Z_\tau\mid Z_0).
$$

这只是“给定完整 clean pair 后，两个前向噪声独立”。反向网络同时读取 Xτ、Zτ，用两个 head 输出 token logits 和连续噪声估计：

$$
f_\theta(X_\tau,Z_\tau,\tau)
\longrightarrow
\bigl(\ell_\theta,\widehat\varepsilon_\theta\bigr).
$$

其损失明确包含两条监督：

$$
\mathcal L
=\lambda_{\mathrm{disc}}(\tau)\mathcal L_{\mathrm{masked\ CE}}
+\lambda_{\mathrm{cont}}(\tau)
\|\widehat\varepsilon_\theta-\varepsilon\|^2.
$$

token head 学会利用连续语义去决定词；continuous head 又利用已显露的 token 和整段离散上下文修正 latent。这与“只用分类后验加权 embedding 就得到连续预测”的 CADD 不同。

### 采样时两个 head 怎样接到原有算法

若采用 VP 参数化，Zτ=ατZ₀+στ ε；以下在 ατ>0 的有限噪声网格上计算。网络预测噪声后，先换成 clean latent：

$$
\widehat Z_0
=\frac{Z_\tau-\sigma_\tau\widehat\varepsilon_\theta}
{\alpha_\tau}.
$$

从较噪的 τ 到较干净的 s，一种 DDIM 更新为：

$$
Z_s=\alpha_s\widehat Z_0
+\sigma_s\widehat\varepsilon_\theta.
$$

token head 则接离散反向 kernel：对于 absorbing mask，决定哪些位置继续 mask，哪些位置按预测后验揭示。两种更新都以旧的联合状态为条件，下一轮再读取更新后的两种状态。

~~~text
初始化：X 全为 mask，Z 为 Gaussian
每轮：
    logits, eps_hat = shared_network(X,Z,time)
    按离散反向 kernel 更新 X
    由 eps_hat 计算 clean latent，再用 DDIM／DDPM 更新 Z
返回最终 token 通道的结果
~~~

每步反向 kernel 可以采用因子化实现，两个 head 却不是互不交流的模型。动态耦合来自共享条件和反复更新；这和“输出头是否分开”属于不同问题。

### 实验中的提升来自哪一份额外信息

CCDD 表 2 使用 Qwen3-Embedding-0.6B 的上下文表示，并以相同 Qwen tokenizer、33B 训练 token 重训基线。在所列 92.1M 可训练模型配置中，MDLM 的验证 PPL 上界为 39.17，CCDD-MDiT 为 29.22。

这个结果体现了联合接口的有效性，也继承了预训练表示提供的知识。更大的 MMDiT 配置还增加了参数，不能把所有行的改善都归因于“连续通道”一个变量。对于自己的消融，应固定表示来源、可训练参数和 tokenization，再比较是否保留连续演化。

## 三种混合模型放在一起应怎样区分

| 方法 | 连续信息保存在哪里 | 核心监督 | 一轮中新发生的动作 |
| --- | --- | --- | --- |
| CANDI | 未决位置的 noisy one-hot／其投影 | 加权 token CE | 部分位置提交，其余位置连续更新 |
| CADD | mask 位置配套的词向量 latent | 主要为 masked CE | 用类别后验决定揭示，并形成连续 clean 均值 |
| CCDD | 与 token 通道并行的上下文 latent | masked CE＋连续 MSE | 两个 head 联合读取、分别更新两条通道 |

三者都不等于“给离散 dLLM 多接一个任意 float 输入”。需要明确这个输入来自哪条前向分布、推理时如何演化，以及训练 loss 怎样约束它。

## 共享 latent 的联合能力，什么时候会在实现中丢掉

若 Z 是共享随机变量，混合分布

$$
P(w_1,\ldots,w_L)
=\int P(z)\prod_iP(w_i\mid z)\,\mathrm dz
$$

可以表达相关性。采样时先抽一份 z，再让各位置在这份共同条件下选择，保留了混合变量。若先对所有 z 平均每个位置的概率，再按位置独立抽样，得到的却是：

$$
\prod_i\int P(z)P(w_i\mid z)\,\mathrm dz,
$$

一般不是前一个联合分布。对于只有 AA、BB 的例子，前者可以恰好正确，后者又会产生 AB、BA。

同理，“论文使用若干 latent 样本平均 logits／概率”并不意味着任何平均位置、任何随后采样方式都保留同一联合 law。连续状态要通过实际的共享轨迹、后续上下文或保留的混合变量发挥作用。混合模型也不会在纯噪声的第一次调用上自动消除所有少步误差。

还应区分轨迹表达能力与最终文本分布：连续中间态能保存更细的信息，并不直接证明离散模型无法表达同一个终点分布。离散 AR、带足够步骤的离散扩散本来就可以表达丰富的联合文本。混合方法的研究价值，需要落实到有限参数、有限训练与有限采样预算下的信息保留和优化。

## 怎样设计一个有解释力的起步实验

先固定离散基线的数据、tokenizer 和主干，再引入一种连续状态。在相同训练预算和采样成本下，至少比较原基线、增加容量但不携带连续状态的控制、完整混合模型，以及打乱跨步连续状态的版本。

打乱实验可以帮助判断收益是否真的依赖那条信息通道，但要留意它可能制造分布外输入。把它与去掉该通道后重新训练的消融配合，结论会更稳妥。

观察少步准确率、位置间约束冲突、生成多样性和时间成本。若增益只出现在额外增加前向次数之后，应把它解释为系统层面的取舍，而不是单独归因于状态连续。

## 建议的复现顺序

先确保纯离散基线在你的数据上正常，再只加一个最小连续提示接口。接着观察提示是否随采样时间逐步改善，并用真实编码 latent 与模型生成 latent 作对照。

若真实编码 latent 有效、生成 latent 无效，可以先检查提示生成与接口匹配，并结合输入分布差异和预算控制做验证。这比同时更换 token 去噪器与 latent 生成器，更容易定位问题。

最后再尝试联合训练、独立噪声日程或少步蒸馏。先稳定一个可解释接口，能减少把多个失败原因叠在一起的风险。

## 方法归纳与出处

[CANDI](https://arxiv.org/abs/2510.22510)、[CADD](https://arxiv.org/abs/2510.01329) 与 [CCDD](https://arxiv.org/abs/2510.03206) 提供三类混合实现。逐项复现时，再核对其联合反向转换、状态融合与推理近似。

**自测：** 条件于共享 z 的逐位置独立，与直接条件于 x 的逐位置独立，为什么会给出不同的联合分布？你的系统里，连续信息是只在一次前向内部存在，还是被保留并更新到了下一轮？

## 自测参考答案

给定共享 z 的条件独立，允许 z 在边缘化以后给所有位置带来共同变化。直接给定 x 后独立采样，则没有这层混合所产生的相关性。上面的 0.41 对 0.25 就是具体差异。

判断自己的模型是否使用持续的连续信息，应检查下一次调用能否读到上一次更新的 latent，以及改变或移除这份 latent 是否影响采样。仅在一次 Transformer 前向内部存在的 hidden state，是另一种接口。

## 阅读导航

[返回总目录]({% post_url 2026-11-19-continuous-dllm-index %}) · [上一篇 第8篇 从论文结果走向研究判断]({% post_url 2026-11-19-continuous-dllm-08-research-evaluation %}) · [下一篇 选读B Simplex 与球面上的流]({% post_url 2026-11-19-continuous-dllm-10-simplex-sphere %})
