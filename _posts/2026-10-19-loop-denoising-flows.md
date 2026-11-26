---
layout: post
title: "计算的回路（二）：去噪训练与迭代生成"
date: 2026-10-19
last_modified_at: 2026-11-26
permalink: /blog/loop-denoising-flows/
visibility: unlisted
sitemap: false
series: 计算的回路
series_description: 循环模型、迭代生成、隐式推理与推理系统
series_order: 2
lang: zh-CN
tags: [Loop Transformer, Recurrent Reasoning, Denoising, Flow Matching, Diffusion]
description: "去噪目标如何训练带记忆的递归网络？内部循环又怎样改善扩散生成？从 Looped Flows 的推导到 Looped-DiT、LiFT 与自适应生成，分析两种结合方式。"
excerpt: "去噪目标如何训练带记忆的递归网络？内部循环又怎样改善扩散生成？从 Looped Flows 的推导到 Looped-DiT、LiFT 与自适应生成，分析两种结合方式。"
---

> 计算的回路 · [一 · 循环架构]({% post_url 2026-10-19-loop-transformer-architecture %}) / **二 · 去噪与生成** / [三 · 隐式推理]({% post_url 2026-10-19-loop-latent-reasoning %}) / [四 · 推理系统]({% post_url 2026-11-26-loop-transformer-serving %})

训练一个会反复修正答案的模型，难处在于前几轮的状态写得好不好，往往要等到最后一轮才知道。把这个信号传回早期状态，得保存一路上的计算。循环越长，反传的代价越大。

[Thinking with Looped Flows](https://arxiv.org/abs/2609.11801v1) 给每一轮分配一个可以单独训练的任务。模型从不同噪声水平的输入恢复答案，并把中间状态留给下一轮。它能不能学会逐步修正，取决于下一轮有没有用到留下来的信息。

## 1. 截断反传之后，怎样训练持久状态

把一个递归求解器简写为

$$
h_{k+1}=F_\theta(h_k,c),\qquad
\widehat Y_k=G_\theta(h_{k+1}),
$$

其中 $$c$$ 是题目，$$h_k$$ 是跨轮保留的状态，$$\widehat Y_k$$ 是第 $$k$$ 轮的答案预测。若损失只加在末轮，较早的状态要通过一串 Jacobian 接收反馈。

$$
\frac{\partial h_K}{\partial h_j}
=J_{K-1}J_{K-2}\cdots J_j,
\qquad
J_r=\frac{\partial F_\theta(h_r,c)}{\partial h_r}.
$$

长展开增加激活存储，也可能带来梯度消失、爆炸或方向失真。截断反向传播能减轻这些问题，却让远期损失无法直接约束早期更新：某一步保存的信息是否有利于后续求解，不再有完整的梯度反馈。

一种常用办法是逐轮监督，并把传入状态从计算图中分离：

$$
h_{k+1}=F_\theta(\operatorname{sg}(h_k),c),
\qquad
\mathcal L_k=\operatorname{CE}(G_\theta(h_{k+1}),Y).
$$

$$\operatorname{sg}$$ 表示 stop-gradient：前向数值不变，反向不再穿过它。状态仍向后传递，每一步的损失仍更新共享参数 $$\theta$$，但后一步损失不再通过前一步的激活约束其状态写入。

逐轮监督没有规定各轮怎样分工。每轮都接收同一题目、预测同一答案，局部最优可能只是重复上一轮的策略。如果把训练拆成一组彼此关联的子问题，局部更新就有机会承担不同的计算。

去噪可以构造这组子问题：从不同污染程度的答案中恢复干净答案，无须人工标注推理步骤。接近干净答案时，模型学习修正小扰动；信息很少时，它必须更多地依靠题目本身。各个噪声水平共享同一个终点，但输入条件不同。

噪声水平只改变了学习条件，各轮是否因此形成有效分工，仍取决于训练结果。

## 2. 从答案去噪到概率流

### 2.1 离散答案的连续插值

设答案由 $$L$$ 个离散位置组成，每个位置有 $$V$$ 种取值。用 one-hot 矩阵表示干净答案：

$$
Y\in\{0,1\}^{L\times V}.
$$

以数独为例，每个待填格子对应一个位置，每个数字对应一个类别；给定线索由题目条件 $$c$$ 保留。采样与答案独立的高斯噪声

$$
\epsilon\sim\mathcal N(0,\sigma^2 I),
$$

然后构造线性插值

$$
I_t=(1-t)\epsilon+tY,\qquad 0\leq t\leq1.
$$

这里 $$t=0$$ 对应噪声，$$t=1$$ 对应答案。$$I_t$$ 是连续值矩阵，通常不在概率单纯形上，也不是一道已经填好部分数字的数独。网络根据这个矩阵和题目 $$c$$ 预测每个位置的类别概率。

理想去噪器定义为

$$
D_t(x;c)=\mathbb E[Y\mid I_t=x,c].
$$

对于 one-hot 目标，逐位置交叉熵的总体最优预测正是各位置的条件类别概率，也就是上式的条件均值。因而可以用普通分类监督学习这个去噪器，无须为每个训练样本先生成完整的求解轨迹。

### 2.2 为什么预测终点就足以得到速度

沿一条给定 $$(\epsilon,Y)$$ 的插值路径，有

$$
\frac{\mathrm d I_t}{\mathrm dt}=Y-\epsilon.
$$

由 $$I_t=(1-t)\epsilon+tY$$ 消去 $$\epsilon$$，得到

$$
Y-\epsilon=\frac{Y-I_t}{1-t}.
$$

在 $$I_t=x$$ 的条件下取期望，便得到概率流的速度场：

$$
v_t(x;c)
=\mathbb E[Y-\epsilon\mid I_t=x,c]
=\frac{D_t(x;c)-x}{1-t},\qquad t<1.
$$

于是，学习“从当前噪声状态预测干净答案”，也就学习了把当前状态推向数据分布的速度。理想条件下，对

$$
\frac{\mathrm dX_t}{\mathrm dt}=v_t(X_t;c),
\qquad X_0\sim\mathcal N(0,\sigma^2I)
$$

积分，可以沿插值所定义的边缘分布，从噪声走向答案分布。这一联系解释了标准 flow matching 为何可以逐时间点训练：每个局部监督目标都在估计同一条概率路径上的速度场。<a class="paper-citation" id="cite-1-1" href="#ref-1" title="Thinking with Looped Flows；§2" aria-label="参考文献 1，§2">[1，§2]</a>

训练样本的 $$I_t$$ 沿已知噪声与真实答案之间的直线变化；推理时的 $$X_t$$ 由学到的速度场生成，未必沿着某个训练样本的直线移动。局部预测误差和数值积分误差还会使它偏离理想概率路径。

### 2.3 条件均值不是一份合法答案

如果一道问题有两个等可能的答案 $$(0,1)$$ 和 $$(1,0)$$，那么在无法区分它们时，两个位置取值为 $$1$$ 的条件概率都是 $$0.5$$。这个均值不等于其中任何一个答案；独立解码还可能得到不合法的 $$(0,0)$$ 或 $$(1,1)$$。

去噪器的中间输出因此不必满足全部离散约束。它是概率预测，整个向量及题目共同参与后续更新；不同初始噪声可以把轨迹引向不同答案。理想概率流处理的是联合分布的演化，而不是把某次边缘均值直接四舍五入就完成联合求解。

实际网络是否学好了这些关系、有限步积分能否到达有效解，需要用完整答案的正确率、约束冲突和解覆盖率衡量。逐位置交叉熵低，并不保证整道题正确。

## 3. Looped Flows：带记忆的去噪器

### 3.1 持久状态与局部目标

标准去噪器每次读取 $$X_t$$ 和题目；前面的计算只能通过当前 $$X_t$$ 影响后面。<a class="paper-link" href="https://arxiv.org/abs/2609.11801v1" title="Thinking with Looped Flows">Looped Flows</a> 额外保留一个 hidden state，让中间计算不必全部挤进答案概率矩阵。<a class="paper-citation" id="cite-1-2" href="#ref-1" title="Thinking with Looped Flows；§3" aria-label="参考文献 1，§3">[1，§3]</a>

为简化记号，用 $$h_i$$ 统称持久状态；论文实际基于 TRM 使用两个内部状态。训练的主要形式是

$$
(\widehat Y_i,h_{i+1})
=D_\theta(I_{t_i},\operatorname{sg}(h_i);c,t_i),
$$

$$
\mathcal L_{\mathrm{LF}}
=\mathbb E\left[\sum_{i=0}^{K-1}
\operatorname{CE}(\widehat Y_i,Y)\right].
$$

时间点满足 $$t_0<t_1<\cdots<t_{K-1}$$。一次训练展开共享同一组题目 $$c$$、答案 $$Y$$ 和噪声 $$\epsilon$$，因此输入依次是同一条直线上的不同插值；噪声系数随时间减小。状态 $$h_i$$ 沿这些步骤向前传递，但梯度不跨去噪步骤传播。

每步的标签始终是最终干净答案 $$Y$$，不是下一个插值 $$I_{t_{i+1}}$$，也不是一段显式推理过程。所谓局部目标，指的是局部计算图上的终点预测损失。

上一轮留下的状态，会在同一道题、同一份噪声、稍高信噪比的下一轮输入下继续使用。作者把这称为时间对齐的去噪目标。状态复用因此进入了训练过程，但各轮没有独立的步骤标签，也未必形成可解释的推理分工。

### 3.2 前向展开与梯度范围

论文的训练最多展开 16 个去噪步骤，每一步都有局部损失。一次去噪器调用包含一整个 TRM supervision step：共享网络反复更新两个状态，执行三轮内部 cycle，前两轮不记录梯度，最后一轮反传。因此，一次去噪本身就包含多次共享网络调用。<a class="paper-citation" id="cite-1-3" href="#ref-1" title="Thinking with Looped Flows；附录 A、E" aria-label="参考文献 1，附录 A、E">[1，附录 A、E]</a>

一次去噪器内部的共享层调用数、前向经过的去噪步骤数、梯度穿过的步骤数，分别决定了不同部分的开销。省下长程 BPTT，主要减少计算图存储和反向计算，前面的前向计算仍然要执行。

训练还使用 ACT 正确率预测头，在预测已经足够好时提前结束该训练展开。这个头也用于多轨迹推理后的候选选择；论文的推理算法并不靠它提前停止概率流积分。

### 3.3 推理同时更新答案状态和隐藏状态

推理时没有真实答案，不能继续构造 $$I_t$$。从噪声 $$X_0$$ 出发，在网格

$$
0=t_0<t_1<\cdots<t_N=1
$$

上执行

$$
(\widehat Y_i,h_{i+1})=D_\theta(X_i,h_i;c,t_i),
$$

$$
X_{i+1}
=X_i+\frac{t_{i+1}-t_i}{1-t_i}(\widehat Y_i-X_i).
$$

这是最简单的 Euler 更新。令

$$
\alpha_i=\frac{t_{i+1}-t_i}{1-t_i},
$$

则 $$0<\alpha_i\leq1$$，更新可以写成

$$
X_{i+1}=(1-\alpha_i)X_i+\alpha_i\widehat Y_i.
$$

更新比例由时间间隔和剩余时间共同决定；在等间距网格上，它随推进而增大，最后一步 $$\alpha_{N-1}=1$$。只在 $$t_i<1$$ 处调用去噪器和速度公式，不需要在 $$t=1$$ 计算那个分母为零的表达式。

更细的时间网格会同时带来更多积分步和更多 hidden-state 更新。因此，增加步数后的收益不能全归于数值积分误差变小；模型也获得了更多递归计算机会。反过来，单纯增加步数也没有单调改善准确率的数学保证。

论文还使用带 noise-backtracking 的随机积分：把当前状态适度退回噪声更大的时刻，再去噪向前推进。它在实验中通常改善准确率或解覆盖率。即便不额外注入随机性，确定性 ODE 也可以因初始噪声不同而产生多样输出；“确定性积分”不等于“每道题只有一个输出”。<a class="paper-citation" id="cite-1-4" href="#ref-1" title="Thinking with Looped Flows；§3.2、附录 C" aria-label="参考文献 1，§3.2、附录 C">[1，§3.2、附录 C]</a>

### 3.4 加入记忆后的概率流

§2 的速度恒等式针对理想条件均值 $$D_t(x;c)$$。现在网络额外读取 $$h_i$$，而 $$h_i$$ 依赖前面的输入和预测。整个系统的状态已经是 $$(X_i,h_i)$$，不再只是 $$X_i$$。

对于这个有记忆、存在预测误差的系统，原边缘分布的保持性和到正确答案的全局收敛性都尚未得到证明。Looped Flows 借用了标准 flow 的局部监督结构，长程递归的有效性则主要由实验支持。

### 3.5 求解精度与组件消融

Looped Flows 大体沿用 TRM 的骨干，只增加噪声投影和时间嵌入。数独用约 5M 参数的 MLP-Mixer，Maze 和 ARC 等任务用约 7M 参数的非因果 Transformer。<a class="paper-citation" id="cite-1-5" href="#ref-1" title="Thinking with Looped Flows；附录 A" aria-label="参考文献 1，附录 A">[1，附录 A]</a>

论文主表的单轨迹结果如下：

| 测试集与指标 | TRM | Looped Flows |
|---|---:|---:|
| Sudoku-Extreme，整题正确率 | 87.4% | 97.9 ± 0.4% |
| Maze-Hard，整题正确率 | 85.3% | 86.7 ± 1.1% |
| ARC-AGI-1，pass@2 | 44.6% | 58.8 ± 1.8% |
| ARC-AGI-2，pass@2 | 7.8% | 12.2 ± 1.9% |

Looped Flows 报告三个随机种子的结果；TRM 等基线包含已发表数字，并非全部按相同训练预算重训。“单轨迹”指没有额外使用五轨迹 best-Q 集成，ARC 的评测仍采用 pass@2，并未在此报告 pass@1。<a class="paper-citation" id="cite-1-6" href="#ref-1" title="Thinking with Looped Flows；表 1" aria-label="参考文献 1，表 1">[1，表 1]</a>

提升主要出现在数独和 ARC，Maze-Hard 与 FPRM 的差距很小。数独经五轨迹集成后超过 99%，这个结果要付出额外的采样和筛选开销。

去掉时间条件、插值输入或噪声关联后，ARC 上的结果如下：

| Looped Flows 训练变体 | ARC-AGI-1 pass@2 | ARC-AGI-2 pass@2 |
|---|---:|---:|
| 完整模型 | 58.8 ± 1.8% | 12.2 ± 1.9% |
| 去掉时间条件 | 56.4% | 9.9% |
| 去掉插值输入 | 51.5% | 9.9% |
| 同时去掉时间条件与插值输入 | 43.6% | 5.0% |
| 噪声水平不再递减 | 51.6% | 9.9% |
| 步骤之间不共享噪声 | 56.4% | 10.8% |

在 ARC-AGI-1 上，只去掉插值输入比只去掉时间标签掉分更多。两者一起拿掉时，结果接近 TRM。表里的差值也可能包含两项设置之间的相互作用。<a class="paper-citation" id="cite-1-7" href="#ref-1" title="Thinking with Looped Flows；表 3" aria-label="参考文献 1，表 3">[1，表 3]</a>

主实验还在 Sudoku 和 ARC-AGI-2 上使用了 pseudotarget 正则化，将部分后续输入的插值终点由真实答案换成上一轮分离梯度后的连续预测：

$$
\widetilde I_{t_i}
=(1-t_i)\epsilon+t_i\operatorname{sg}(\widehat Y_{i-1}),\qquad i\geq1.
$$

第一步仍使用真实答案构造插值，交叉熵标签也始终是真实答案 $$Y$$。后续输入因此减少了对真实答案的直接暴露，模型也能接触自己不完美的预测。主表结果包含这项正则化的作用。<a class="paper-citation" id="cite-1-8" href="#ref-1" title="Thinking with Looped Flows；附录 D、E" aria-label="参考文献 1，附录 D、E">[1，附录 D、E]</a>

### 3.6 共享噪声也可能提供代数捷径

共享噪声会让两个时间点的输入包含可精确消去的共同项。取同一条真实答案插值上的两个时间点 $$s<t$$：

$$
I_s=(1-s)\epsilon+sY,\qquad
I_t=(1-t)\epsilon+tY.
$$

把第一式乘以 $$(1-t)$$、第二式乘以 $$(1-s)$$，相减即可消掉噪声：

$$
Y=\frac{(1-s)I_t-(1-t)I_s}{t-s}.
$$

如果 hidden state 保存了前一次输入及其时间，模型就可能绕过题目，直接用两次观测恢复标签。即使局部去噪损失很低，这条捷径也可能是原因。<a class="paper-citation" id="cite-1-9" href="#ref-1" title="Thinking with Looped Flows；附录 B" aria-label="参考文献 1，附录 B">[1，附录 B]</a>

但推理输入不含真实答案。采用确定性 Euler 采样，令第一步预测为 $$P=\widehat Y_0$$。从 $$t_0=0$$ 出发，第一次更新得到

$$
X_1=(1-t_1)\epsilon+t_1P.
$$

它形式上像训练插值，终点却是模型自己的预测 $$P$$，不是真实答案 $$Y$$。继续使用同样的代数消噪规则，恢复出的仍然是 $$P$$；以后每一步也会沿着 $$\epsilon$$ 到 $$P$$ 的直线前进。只依赖这条捷径时，增加迭代不能修正第一步的错误。

论文观察到增加步数仍可提高求解率，与纯粹重复第一步预测的行为不符。不过，网络仍可能同时利用共享噪声相关性和依赖题目的计算，现有结果没有排除这种混合机制。

作者推测，局部优化会使这条捷径更难学：stop-gradient 阻止后一步损失直接训练前一步去保存消噪所需的信息，而每步自身的损失更倾向于保留有利于当前预测的特征。这一解释尚未得到证明；共享参数和状态复用仍可能间接促成捷径。

要分辨这两种机制，可以让模型处理自己生成的错误状态，检查后续更新能否纠错，再改变跨步噪声关联，观察深度收益是否保留。单独的训练损失无法作出这种区分。

## 4. 训练轨迹与推理轨迹

### 4.1 FRM：用模型自己的轨迹构造 carry

<a class="paper-link" href="https://arxiv.org/abs/2606.29150v3" title="Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners">Flow Reasoning Models</a><a class="paper-citation" id="cite-2-1" href="#ref-2" title="Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners" aria-label="参考文献 2">[2]</a> 同样从高斯噪声到 one-hot 答案的线性路径出发，但主要传递的是上一轮干净答案预测，即 self-conditioning carry。它不等同于 Looped Flows 的通用 hidden memory。

固定当前噪声状态 $$x_t$$ 和时间 $$t$$，可以把递归预测写成

$$
s_t^{(k+1)}=D_{\theta,t}(x_t;c,s_t^{(k)}).
$$

这里 $$k$$ 表示反复修正预测的深度，$$t$$ 表示概率路径上的时间。在固定时间处修订预测，与推进概率流，是两种不同的更新。

常规 self-conditioning 训练往往先在真实答案构造的 $$I_t$$ 上预测一次，把结果 detach，再作为第二次预测的条件。推理时，carry 却是经过多轮反馈后形成的状态。一次预测产生的误差，与多轮递归积累出来的误差不一定相同。后者可能已经非常自信，甚至形成一个稳定但错误的固定点。

FRM 的 Fixed-Point Forcing（FPF）针对这个差异构造训练条件：采样监督时间 $$t$$，再采样一个更早的时间 $$t_{\mathrm{start}}$$，以真实答案构造的 $$I_{t_{\mathrm{start}}}$$ 为起点，运行模型自己的 self-conditioned 积分过程直到 $$t$$，把得到的最终预测 detach 后送入受监督的去噪器。<a class="paper-citation" id="cite-2-2" href="#ref-2" title="Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners；§3、图 4" aria-label="参考文献 2，§3、图 4">[2，§3、图 4]</a>

用于计算损失的去噪器调用仍接收标准插值

$$
I_t=(1-t)\epsilon+tY,
$$

标签仍是 $$Y$$。FPF 替换的是 self-conditioning carry，因此主要缓解 carry 一侧的分布偏移；受监督的噪声输入与推理轨迹之间仍有差异。

固定点 $$F(s^*)=s^*$$ 仍然可能对应错误答案。FRM 的训练针对后期迭代产生的错误，但并不保证所有固定点都对应正确答案。深度外推失败除了可能源于更新不稳定，也可能是模型没有学过如何处理深层循环产生的状态。

### 4.2 Timestep-free reasoning：训练课程与推理日程分开

<a class="paper-link" href="https://arxiv.org/abs/2609.01449v1" title="Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning">Timestep-free Iterative Reasoning</a><a class="paper-citation" id="cite-3-1" href="#ref-3" title="Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning" aria-label="参考文献 3">[3]</a> 做了一个更直接的干预：训练后不再逐渐降低噪声，而是在每一步持续输入最大强度的噪声。模型仍能求解数独，效果甚至优于它自己的退火推理。

持续加噪只作用于待预测的可见通道。题目线索和跨步 hidden state 都被保留，模型仍有求解所需的条件和记忆。

该方法训练时采用

$$
a(t)=\sin\frac{\pi t}{2},\qquad
b(t)=\cos\frac{\pi t}{2},
$$

并在非线索位置构造 $$x_k=a(t_k)Y+b(t_k)\epsilon_k$$。不同步骤使用新采样的独立噪声 $$\epsilon_k$$，hidden state 则持续向前传播。与 Looped Flows 的共享噪声不同，这里不存在同一份噪声可被两次观测精确抵消的前提。

训练展开长度在 20–160 步之间采样，用 4 个去噪步骤的截断 BPTT 优化，在非线索位置施加均方误差。梯度可以在四步窗口内传播，窗口之间才 detach。网络没有显式时间条件，数独版本内部循环共享 block 八次，并采用行、列和宫对应的注意力连接；迷宫版本使用局部卷积通信。<a class="paper-citation" id="cite-3-2" href="#ref-3" title="Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning；§2、附录 G、H" aria-label="参考文献 3，§2、附录 G、H">[3，§2、附录 G、H]</a>

推理时，在非线索位置用上一轮预测构造

$$
x_{k+1}=a(t_{k+1})\widehat Y_k+b(t_{k+1})\epsilon_{k+1}.
$$

若始终令 $$t=0$$，则 $$a(0)=0$$、$$b(0)=1$$，这个通道每步都变回全新的高斯噪声。上一轮可见预测不再传入下一轮，跨步计算主要由未被加噪的 hidden state 承担。对同一个模型，论文报告的 Sudoku-Extreme 结果是：

| 推理时的噪声设置 | 整题正确率 |
|---|---:|
| 按日程逐步减小噪声 | 99.56% |
| 每一步保持最大噪声，保留线索和 hidden state | 99.90% |
| 不再加噪，直接反馈预测 | 49.39% |

退火和保持最大噪声，在数独推理中都能得到接近满分的结果。训练时把噪声水平固定，正确率却降至约 83%，完全去掉噪声的消融报告为 0%。逐步降低噪声有助于训练，学成后的求解器可以在持续的最大噪声下工作。<a class="paper-citation" id="cite-3-3" href="#ref-3" title="Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning；§3、补充表 2" aria-label="参考文献 3，§3、补充表 2">[3，§3、补充表 2]</a>

换到 Maze-Unique，最大噪声的平均分略低于退火。这个实验用的模型和数据也与 Looped Flows 的 Maze-Hard 不同。

Looped Flows 还没有做同样的推理干预，先前的去噪日程是否也可以去掉，仍需在它自己的模型上验证。

推理拉长到 10,000 步时，绝大多数充分确定的数独轨迹在观察窗口内保持同一输出。包含欠确定问题的训练更容易得到会离开稳定答案的轨迹。线索数量和掩码形状也随设置改变，解是否唯一只是可能的原因之一。<a class="paper-citation" id="cite-3-4" href="#ref-3" title="Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning；§3、附录 F" aria-label="参考文献 3，§3、附录 F">[3，§3、附录 F]</a>

## 5. 什么时候该多跑一轮去噪器

外层推进会换一个噪声时间或遮罩状态。内层循环可以在同一个状态上继续更新特征，也可以边更新边提交 token。两种调度都要把共享块的调用计入预算。

### 5.1 内部循环与外部步数不是同一预算

以共享中间块为例，外部执行 $$N$$ 步，第 $$n$$ 步内部循环 $$R_n$$ 次，计算量近似为

$$
C\approx\sum_{n=1}^{N}
\left(C_{\mathrm{head}}+R_nC_{\mathrm{mid}}+C_{\mathrm{tail}}\right).
$$

增加 $$N$$ 会更频繁地推进外部生成状态，增加 $$R_n$$ 则在当前状态下改进预测。对于连续 flow，外部积分更细可以减小离散化误差，内部加深可能改善速度估计；对于 masked diffusion，还要区分每一步究竟提交了多少 tokens。

相同的调用次数未必有相同 FLOPs、延迟或状态容量，比较预算时还要计入单次调用的规模和执行方式。

### 5.2 同一输入多算一轮 · LoopMDM

<a class="paper-link" href="https://arxiv.org/abs/2605.26106v1">LoopMDM</a><a class="paper-citation" id="cite-4-1" href="#ref-4" title="Looped Diffusion Language Models" aria-label="参考文献 4">[4]</a> 在 masked diffusion 去噪器内部循环一小段共享 Transformer 层。模型分成 head、mid-block 和 tail，首尾各执行一次，中间部分执行 $$R$$ 次：

$$
H^{(0)}=\operatorname{head}(x_t),\qquad
H^{(r+1)}=\operatorname{mid}_\theta(H^{(r)}),
$$

$$
p_\theta(Y\mid x_t)=\operatorname{tail}(H^{(R)}).
$$

有效深度为 $$L_{\mathrm{head}}+R L_{\mathrm{mid}}+L_{\mathrm{tail}}$$。训练时随机采样循环次数，只在末轮输出上计算标准 masked diffusion NELBO。

外层采样器仍执行 masked diffusion 的去噪步骤。特征只在单次去噪调用内部循环，架构没有增加 Looped Flows 那样跨去噪调用保留的 latent memory。

论文主要预训练实验使用 170M 参数、12 层模型，选择两个靠前的中间层循环，最大训练循环数为 12。在其数据与配置下，达到相同测试 NLL 所需训练 FLOPs 最多降至基线的约 $$1/3.3$$。收益来自这套选择性循环配置，能否推广到其他 diffusion LM 仍需验证。<a class="paper-citation" id="cite-4-2" href="#ref-4" title="Looped Diffusion Language Models；§3、§4" aria-label="参考文献 4，§3、§4">[4，§3、§4]</a>

论文还观察到被 mask 位置之间的注意力交互增强。这可以支持“内部迭代改善位置间信息交换”的解释，但输出的概率校准、并行采样的联合一致性，仍需要独立验证。

### 5.3 换成均匀污染后，内部循环还有用吗

上面是遮罩去噪。<a class="paper-link" href="https://arxiv.org/abs/2610.30066" title="Looped Uniform Diffusion Language Models">Looped Uniform DLM</a><a class="paper-citation" id="cite-11-1" href="#ref-11" title="Looped Uniform Diffusion Language Models" aria-label="参考文献 11">[11]</a> 研究的是均匀扩散，当前序列中普通词表 token 也可能是错的。作者保持训练目标和采样器，循环少数前部 Transformer 层。与达到相近验证 NLL 的非循环 UDLM 相比，论文在 OpenWebText 和 FineWeb-Edu 报告的最高训练 FLOPs 节省分别是 2.8× 和 3.3×，参照模型的训练长度约 0.5T tokens。

作者还用等 NFE 比较生成 perplexity。每次去噪内部的执行深度有变化，所以 NFE 适合描述外层调用频率。要回答推理计算量是否下降，仍需使用 §5.1 的总量。

### 5.4 图像生成的同终点监督 · Looped-DiT

<a class="paper-link" href="https://arxiv.org/abs/2609.40305v1">Looped-DiT</a><a class="paper-citation" id="cite-6-1" href="#ref-6" title="Looped Diffusion Transformer" aria-label="参考文献 6">[6]</a> 将内部循环用于文本到图像生成。它以像素空间的 MiniT2I 为骨干，将 17 个 MMDiT blocks 分成前置 6 层、共享中间 5 层和后置 6 层，训练时中间部分执行四次。一次去噪调用因此有 32 层的执行深度，存储的仍是 17 层参数。图像与文本 hidden states 一起在中间块中更新；采样器随后才推进到下一个时间点。<a class="paper-citation" id="cite-6-2" href="#ref-6" title="Looped Diffusion Transformer；§2–3" aria-label="参考文献 6，§2–3">[6，§2–3]</a>

与 LoopMDM 相比，监督方式有一个明显区别：Looped-DiT 将每个内部循环的状态都送入共享的后置网络，预测同一张干净图像。此处用 $$Y$$ 表示连续像素图像，而非前面结构化求解器的 one-hot 答案。在固定的 $$x_t=tY+(1-t)\epsilon$$、时间 $$t$$ 和文本条件下，训练目标为

$$
\mathcal L_{\mathrm{DS}}
=\mathbb E\!\left[
\sum_{r=1}^{R}w_r
\frac{\|\widehat Y_t^{(r)}-Y\|_2^2}
{d_x\max\{1-t,\tau\}^2}
\right],\qquad \tau>0.
$$

$$d_x$$ 是图像元素数，分母对应以干净图像预测表示的 flow-matching 权重，并在终点附近截断。四轮的 Final + Mean 配置使用 $$w=(1/3,1/3,1/3,1)$$。这里各轮共享同一噪声输入和时间；Looped Flows 的局部目标则分布在不同时间点，并在调用之间传递状态。逐轮损失也不等于截断反传：Looped-DiT 沿内部循环反传，没有采用 Looped Flows 的跨去噪步 stop-gradient 方案。中间读出只在训练时执行，不增加推理调用，但会增加训练计算。<a class="paper-citation" id="cite-6-3" href="#ref-6" title="Looped Diffusion Transformer；§2.1、附录 A.1" aria-label="参考文献 6，§2.1、附录 A.1">[6，§2.1、附录 A.1]</a>

模型还调节共享块中的注意力更新。论文比较了逐 token、逐 head 的门控，以及 Exclusive Self Attention（XSA）。后者移除 attention 输出沿当前 token 自身 value 方向的分量，避免反复累加这部分更新。它与逐轮监督共同改善循环效果；不能把完整方法的增益全部归给参数共享。

在 B/32 的四项评测平均分上，对照如下：

| 配置 | 平均分 |
|---|---:|
| 非循环 MiniT2I | 55.2 |
| 仅增加循环 | 56.2 |
| 循环 + Final + Mean 监督 | 57.5 |
| 循环 + XSA | 58.5 |
| 循环 + Final + Mean 监督 + XSA | 59.1 |
| 加深的非共享模型，匹配训练与推理计算量并使用深监督 | 58.1 |

表末两行用同样的训练和推理 FLOPs 比较。260M 的循环模型略胜 473M 的加深模型，前几行消融也显示，循环本身只贡献了完整方法的一部分收益。<a class="paper-citation" id="cite-6-4" href="#ref-6" title="Looped Diffusion Transformer；表 2、4" aria-label="参考文献 6，表 2、4">[6，表 2、4]</a>

论文还固定每张图像的推理 FLOPs，将四轮循环与单轮预测后增加去噪步数比较。单轮对照包括同一循环 checkpoint 只执行一轮，以及单独训练的非循环模型。在所测试的预算区间内，四轮内部计算的结果更好。<a class="paper-citation" id="cite-6-5" href="#ref-6" title="Looped Diffusion Transformer；图 6" aria-label="参考文献 6，图 6">[6，图 6]</a>

这个实验区分了两种误差来源。更多去噪步可以减小时间离散化误差，却不能保证单次网络预测足够准确；内部循环可以继续处理当前噪声状态下的约束，但不会自行推进外部时间。两种预算不能只按“调用了多少步”视为等价。

Looped-DiT 也展示了更深循环修正文字、数量与组合关系的例子。这些现象与内部迭代修订相符，尚未确定对应什么算法。其跨模型主表中，260M 模型超过 1.7B InternVL-U，推理计算量约为后者的 $$1/4.9$$；该结果来自各自的训练与官方推理配置，不能替代上述同骨干消融和预算匹配实验。<a class="paper-citation" id="cite-6-6" href="#ref-6" title="Looped Diffusion Transformer；表 1、图 4" aria-label="参考文献 6，表 1、图 4">[6，表 1、图 4]</a>

### 5.5 沿内部深度改变监督 · LiFT

Looped-DiT 的各轮都拟合同一终点。这样的深监督要求中间预测尽早准确，却没有规定相邻循环应承担多少修正。<a class="paper-link" href="https://arxiv.org/abs/2610.05538v1">LiFT</a><a class="paper-citation" id="cite-7-1" href="#ref-7" title="LiFT: Loop Flow Transformers" aria-label="参考文献 7">[7]</a> 另外引入计算深度 $$s\in[0,1]$$，为不同深度定义不同的回归目标。

固定外部噪声输入 $$x_t$$、生成时间 $$t$$ 和类别条件。prelude 产生初始状态 $$h_0$$，共享读出得到初始速度估计 $$u_0$$。用其停止梯度版本作锚点，再连到该训练样本的 flow-matching 速度目标 $$u^*$$：

$$
b=\operatorname{sg}(u_0),\qquad
\bar u_s=(1-s)b+s u^*.
$$

这里 $$s$$ 不表示图像从噪声变干净的程度；在整个内部循环中，外部 $$x_t,t$$ 都固定。它只标记一次速度预测的修正进度。若 $$u^*=x_1-x_0$$，外部 flow 的目标是学习数据传输速度，内部回归路径则连接“当前速度估计”与“训练速度目标”。

设第 $$k$$ 轮预测为 $$u_k$$，训练深度网格满足 $$s_K=1$$，则

$$
\mathcal L_{\mathrm{LiFT}}
=\mathbb E\left[\frac1{Kd}\sum_{k=1}^{K}
\left\|u_k-\bigl((1-s_k)b+s_ku^*\bigr)\right\|_2^2\right].
$$

$$d$$ 为预测维度。末轮恢复普通 flow-matching 目标，中间轮只承担对应比例的修正。锚点在目标侧 detach，预测侧的梯度仍沿内部循环回传，不是 Looped Flows 的跨步截断训练。LiFT 还用小权重的 $$\|u_0-u^*\|^2/d$$ 监督 prelude，避免初始预测漂移导致中间目标失去意义。<a class="paper-citation" id="cite-7-2" href="#ref-7" title="LiFT: Loop Flow Transformers；§3" aria-label="参考文献 7，§3">[7，§3]</a>

推理用 $$s_k=k/K_{\mathrm{inf}}$$ 选择内部网格。增加循环数会细化这张网格，但终点仍为 $$s=1$$；最后一次速度预测交给外部采样器。每个外部积分步骤都会重新初始化 prelude 状态，没有保存 Looped Flows 式的跨去噪步 hidden memory。

ImageNet-256 类条件、无 CFG 的实验中，XL/2 R12 训练两轮，推理增加到十六轮后 FID 从 17.07 降至 9.31。但同训练循环数下，它不普遍优于相同执行深度的 dense 模型，B/2 配置也没有全面获胜。这一训练目标支持部分模型的深度外推，收益随模型配置而变化。<a class="paper-citation" id="cite-7-3" href="#ref-7" title="LiFT: Loop Flow Transformers；主实验与深度消融" aria-label="参考文献 7，主实验与深度消融">[7，主实验与深度消融]</a>

内部回归目标是一条直线，也不代表实际隐状态沿直线运动，更不等于最终速度误差必然随循环数单调下降。LiFT 约束的是各深度的读出；更新是否能在更细网格上保持有效，仍由模型和训练决定。

### 5.6 循环中提交 token · ALoDLM

<a class="paper-link" href="https://arxiv.org/abs/2610.04198v1">ALoDLM</a><a class="paper-citation" id="cite-8-1" href="#ref-8" title="ALoDLM: Adaptively Looped Diffusion Language Models" aria-label="参考文献 8">[8]</a> 在一次外部去噪步骤内，既修订 latent，也逐步确定离散 token。设 $$U$$ 为未提交位置，第 $$s$$ 次内部预测的分布为 $$p_i^{(s)}$$，低熵位置组成

$$
B^{(s)}=\{i\in U:H(p_i^{(s)})\leq\tau\}.
$$

这些位置采样并提交 token。后续循环输入中，已提交位置使用 token embedding，其余位置继续传递共享 core 的输出：

$$
h_i^{(s)}=
\begin{cases}
\operatorname{Emb}(\widehat y_i),&i\text{ 已提交},\\
\widetilde h_i^{(s)},&i\text{ 未提交}.
\end{cases}
$$

因此，后续循环面对的离散上下文也会改变；它不同于固定同一个 $$x_t$$、只提高速度预测精度的 LiFT。

另一个 ExitGate 预测“此前未停止时，本轮停止”的条件概率 $$\lambda_i^{(s)}$$，累计停机概率为

$$
a_i^{(s)}=a_i^{(s-1)}+
\bigl(1-a_i^{(s-1)}\bigr)\lambda_i^{(s)},\qquad a_i^{(0)}=0.
$$

移除新提交位置后，若 $$U$$ 为空，或其平均累计停机概率达到阈值 $$q$$，内循环停止；否则继续，直到最大深度。预测熵控制 token 提交，ExitGate 控制是否继续细化，不能用一个“置信度早退”概括二者。也不能因为 gate 有逐 token 输出，就说执行时每个位置都独立停止计算。<a class="paper-citation" id="cite-8-2" href="#ref-8" title="ALoDLM: Adaptively Looped Diffusion Language Models；§3.1" aria-label="参考文献 8，§3.1">[8，§3.1]</a>

训练把条件 NELBO 与退出日程联合建模，处理前面的提交和退出如何影响后续预测。模型由 Qwen3-1.7B/8B 在 WeDLM 的 streaming block-diffusion 框架下转换，直接对 5B-token 语料 SFT，最大循环数为四，没有额外 CPT 阶段。1.7B 循环全 28 层，8B 循环中间 16 层。<a class="paper-citation" id="cite-8-3" href="#ref-8" title="ALoDLM: Adaptively Looped Diffusion Language Models；§3.2–4.1" aria-label="参考文献 8，§3.2–4.1">[8，§3.2–4.1]</a>

十一项 benchmark 的主质量表平均分为 65.5、80.3，但这张表约束各方法每个生成步骤只提交一个 token。GSM8K 上相对 vLLM Qwen3-8B 的约 2.7 倍吞吐，来自另一套熵控制的并行解码配置，质量处于相近水平。不能把两套协议拼成“十一项任务均在同质量下快 2.7 倍”；不同模型的训练数据与转换过程也没有完全匹配。

其缓存访问同样依赖实际执行深度：优先读取前缀 token 同深度 KV，缺失时使用已有的最深状态。这能避免为每个前缀补齐全部循环，但改变了深层 attention 所见的历史表示，质量需要在该缓存规则下评测。

### 5.7 提交前先更新未确定位置 · dLoop

<a class="paper-link" href="https://arxiv.org/abs/2611.26275" title="dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement">dLoop</a><a class="paper-citation" id="cite-12-1" href="#ref-12" title="dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement" aria-label="参考文献 12">[12]</a> 从预训练扩散语言模型出发，把中间层改成共享块。整个内循环面对同一个部分遮罩状态。未提交位置在多轮中交换信息，然后再依据较深层的预测决定本次提交哪些 token。这个顺序与 ALoDLM 的循环内提交不同。

论文在 GSM8K、MATH、MBPP 和 HumanEval 上同时报告了准确率与计入循环开销的 tokens per forward。dLoop 在四项任务中都比 Fast-dLLM 的准确率与调整后的 TPF 高，d3LLM 的 TPF 更高。作者还观察到，额外深度对本来就收敛慢、预测不稳定的轨迹帮助更大。对这类方法，单次能提交多少位置和完成一个答案需要多少次共享块调用应一起报告。

## 6. 同样重复网络，状态与监督却不同

| 工作 | 循环主要传递什么 | 受监督输入的来源 | 梯度如何处理 | 推理时的迭代 |
|---|---|---|---|---|
| Looped Flows | 跨去噪步的隐藏状态 | 有序插值，基本形式共享噪声；部分任务加入 pseudotargets | 去噪步骤间 stop-gradient，调用内部仍有反传 | 概率流与隐藏状态一起更新 |
| FRM | 上一次干净答案预测 | 标准插值；FPF 用模型 rollout 构造 carry | 构造 carry 的 rollout 分离梯度 | self-conditioning 与 flow 积分 |
| Timestep-free reasoning | 未被加噪的隐藏状态 | 递减噪声课程，各步独立噪声 | 四步截断 BPTT | 可退火，也可持续输入最大噪声 |
| LoopMDM | 单次去噪内部的特征 | masked diffusion 的污染样本 | 在最终内部循环输出上监督 | 外层去噪、内层共享层循环 |
| Looped Uniform DLM | 同一步均匀去噪的内部状态 | 普通词表 token 的均匀污染 | 保留原扩散训练目标 | 外部 sampler 保持不变 |
| Looped-DiT | 单次去噪内部的图文特征 | 同一噪声图像、时间与文本条件 | 内部循环反传，各轮共享后置网络做终点监督 | 内部循环深度与外部去噪步数分别设置 |
| LiFT | 单次速度预测内部的隐藏状态 | 固定外部输入，内部目标随计算深度变化 | 内部循环反传，初始预测在目标侧 detach | 细化内部深度网格，末轮速度交给外部积分 |
| ALoDLM | 未确定位置的 latent 与已提交 token | masked 输入及退出/提交日程 | 联合学习条件 NELBO 与退出日程 | 预测熵控制提交，ExitGate 控制继续细化 |
| dLoop | 部分遮罩状态下未提交位置的特征 | 预训练扩散模型的去噪输入 | 共享中间块进行内部 refinement | 等内循环结束后再提交 |
| [PaTh](https://arxiv.org/abs/2610.09876v1) | Thinker 的递归状态 | 当前去噪状态及任务条件 | 重建监督，Painter 参数冻结 | Thinker 内循环产生 Painter 的条件 |
| [FReDA](https://arxiv.org/abs/2606.08357v2) | token 草稿及置信度、提交状态 | 模型自己生成的草稿 | 前序草稿生成无梯度，只监督末次调用 | 自修正或多候选选择 |

例如，PaTh 的一次递归只更新条件，LoopMDM 的一次去噪还会执行整个骨干。ALoDLM 一旦提交了 token，后续轮次的可见上下文也会随之变化。表中的调用次数需要和各自更新的对象一起读。

LoopCD 的预测对比只改变输出读出，不推进生成时间，也不构造去噪训练目标。因此它与 diffusion guidance 的联系不能代替上述训练机制，相关读出公式与实验见[第三篇]({% post_url 2026-10-19-loop-latent-reasoning %})。

## 7. 表示成本与训练—推理分布

数独的每个待填格子只有九种数字，答案长度固定，约束和验解规则明确。语言输出的词表更大、长度可变，还允许多种有效答案，表示成本和监督目标都会随之变化。

例如，连续 one-hot 流的可见状态有 $$L\times V$$ 个标量。取 $$L=4096$$、$$V=128000$$，单个这样的张量以 BF16 存储就约为 1.05 GB，其他缓冲和激活还会继续占用显存。改用离散 mask、低维连续表示或分块生成可以改变这部分成本，但也需要相应调整去噪目标和采样机制。

缓存也受影响。自回归推理通常不改写已经生成的前缀，因而可以复用其 KV；如果一个 loop 不断修订整段候选答案，相关位置的表示和注意力关系就可能改变。缓存能否继续复用，取决于具体的因果分块、状态设计或近似方案。

<a class="paper-link" href="https://arxiv.org/abs/2610.06666v1" title="What Matters for Latent Reasoning with Flow Matching">FLaRe</a><a class="paper-citation" id="cite-10-1" href="#ref-10" title="What Matters for Latent Reasoning with Flow Matching" aria-label="参考文献 10">[10]</a> 将 CoT 压缩为低维 latent，再用条件 flow 生成 thought，说明不必维护完整的 one-hot 状态。其总成本还包括 VAE 训练、多步采样和答案读出。它的工作区与“latent 是否被答案使用”的实验在第三篇展开。

多候选采样还要计入其他轨迹与评分成本。内部循环、外部生成步骤、每步提交数和候选数共同决定预算，单个“步数”无法概括。

状态记忆、噪声关联和积分步数可能分别影响深度收益。Looped-DiT 已对内部循环与外部去噪做了同 FLOPs 比较，但没有同时检验跨步记忆。这些因素需要在各自的架构中单独对照：

| 待检验因素 | 实验设计 |
|---|---|
| 跨步记忆与单步去噪能力 | 固定总计算量，对照有无跨步状态；另做测试时清零、打乱状态的干预 |
| 共享噪声与代数捷径 | 对照共享与独立噪声，并在由模型生成的错误轨迹上检查后续修正能力 |
| 深层状态的分布偏移 | 固定受监督输入，比较一次预测 carry 与多步 rollout carry，记录深度增加后的正确率和置信度 |
| 积分精度与递归深度 | 在相同网络调用预算下，分别增加 flow 步数与固定时间下的 refinement 次数 |
| 终点监督与分深度目标 | 保持骨干、训练与推理预算，比较各轮同一终点目标和 LiFT 式渐进目标 |
| 离散提交与内部停机 | 分别控制 ALoDLM 的熵阈值与停机阈值，比较相同质量下的提交数、循环数和延迟 |
| 训练日程与推理日程 | 固定训练模型，在保留条件和 memory 的前提下比较退火、固定噪声与纯预测反馈 |

稳定性还要与正确性分别衡量。代码、数学或约束任务可以通过测试与验解识别稳定的错误答案；开放式文本则需要质量与多样性评价。唯一解求解器希望最终停在同一个答案上，生成模型可能需要保留多个有效答案，未必适合把所有轨迹压向同一个吸引点。

## 8. 再给去噪器一次调用，它会改变什么

在 Looped Flows 里，要追踪的是上一轮留下的状态有没有帮助恢复当前答案。共享噪声可能让模型学到一条更短的恢复路径，这也是为什么换噪声、换 carry 或检查错误轨迹比测最终准确率更能说明机制。在 dLoop 和 ALoDLM 里，要追踪的是准备提交的位置。相同计算下究竟该继续交换信息，还是先提交一批 token 再更新上下文，取决于那一批预测有多稳定。

> 计算的回路 · [一 · 循环架构]({% post_url 2026-10-19-loop-transformer-architecture %}) / **二 · 去噪与生成** / [三 · 隐式推理]({% post_url 2026-10-19-loop-latent-reasoning %}) / [四 · 推理系统]({% post_url 2026-11-26-loop-transformer-serving %})

## 参考文献

1. <span id="ref-1" class="reference-anchor"></span>Ayhan Suleymanzade et al. [Thinking with Looped Flows](https://arxiv.org/abs/2609.11801v1). arXiv:2609.11801v1，2026-09-10。局部去噪与递归训练见 §2–3；主结果与消融见表 1、3；架构、共享噪声捷径、随机采样与 pseudotargets 见附录 A–E。 <a class="reference-backlink" href="#cite-1-1" aria-label="返回文献 1 首次引用处" title="返回正文">↩</a>
2. <span id="ref-2" class="reference-anchor"></span>Alec Helbling et al. [Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners](https://arxiv.org/abs/2606.29150v3). arXiv:2606.29150v3，2026-09-01 更新。self-conditioning、状态分布偏移与 Fixed-Point Forcing 见 §2–3、图 4。 <a class="reference-backlink" href="#cite-2-1" aria-label="返回文献 2 首次引用处" title="返回正文">↩</a>
3. <span id="ref-3" class="reference-anchor"></span>Mariia Drozdova et al. [Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning](https://arxiv.org/abs/2609.01449v1). arXiv:2609.01449v1，2026-09-01。训练和推理设置见 §2–3；课程、稳定性、BPTT 与迷宫实验见补充材料。 <a class="reference-backlink" href="#cite-3-1" aria-label="返回文献 3 首次引用处" title="返回正文">↩</a>
4. <span id="ref-4" class="reference-anchor"></span>Sanghyun Lee et al. [Looped Diffusion Language Models](https://arxiv.org/abs/2605.26106v1). arXiv:2605.26106v1，2026-05-25。选择性内部循环、随机循环次数与预训练结果见 §3–4。 <a class="reference-backlink" href="#cite-4-1" aria-label="返回文献 4 首次引用处" title="返回正文">↩</a>
5. <span id="ref-5" class="reference-anchor"></span>Haotian Sun et al. [Forward-Free Diffusion Language Models with BPTT-Free Looped Refinement](https://arxiv.org/abs/2606.08357v2). arXiv:2606.08357v2，2026-08-27 更新。模型草稿分布、refinement 目标、训练与采样见 §3，语言模型实验见 §4。
6. <span id="ref-6" class="reference-anchor"></span>Yong Xien Chng et al. [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305v1). arXiv:2609.40305v1，2026-09-30。内部循环、深监督与注意力调节见 §2；计算量匹配见表 2；循环与去噪步数比较见图 6；组件消融见表 4。 <a class="reference-backlink" href="#cite-6-1" aria-label="返回文献 6 首次引用处" title="返回正文">↩</a>
7. <span id="ref-7" class="reference-anchor"></span>[LiFT: Loop Flow Transformers](https://arxiv.org/abs/2610.05538v1). arXiv:2610.05538v1，2026-10-04。内部计算坐标、轨迹目标与 prelude 损失见 §3；FID 与深度/预算对照见实验及附录。 <a class="reference-backlink" href="#cite-7-1" aria-label="返回文献 7 首次引用处" title="返回正文">↩</a>
8. <span id="ref-8" class="reference-anchor"></span>[ALoDLM: Adaptively Looped Diffusion Language Models](https://arxiv.org/abs/2610.04198v1). arXiv:2610.04198v1，2026-10-03。提交与停机见 §3.1，训练目标见 §3.2，质量与并行解码协议见 §4、附录 D。 <a class="reference-backlink" href="#cite-8-1" aria-label="返回文献 8 首次引用处" title="返回正文">↩</a>
9. <span id="ref-9" class="reference-anchor"></span>[Think Before You Paint: Recursive Latent Reasoning for Diffusion Models](https://arxiv.org/abs/2610.09876v1). arXiv:2610.09876v1，2026-10-07。Thinker/Painter 分工见 §3，任务参数量见表 5。
10. <span id="ref-10" class="reference-anchor"></span>[What Matters for Latent Reasoning with Flow Matching](https://arxiv.org/abs/2610.06666v1). arXiv:2610.06666v1，2026-10-05。FLaRe 的 latent 构造与 flow 训练见 §2–3，直接/解码读出及干预见 §4。 <a class="reference-backlink" href="#cite-10-1" aria-label="返回文献 10 首次引用处" title="返回正文">↩</a>
11. <span id="ref-11" class="reference-anchor"></span>[Looped Uniform Diffusion Language Models](https://arxiv.org/abs/2610.30066)。均匀污染、等验证 NLL 与 NFE 对照。 <a class="reference-backlink" href="#cite-11-1" aria-label="返回文献 11 首次引用处" title="返回正文">↩</a>
12. <span id="ref-12" class="reference-anchor"></span>[dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement](https://arxiv.org/abs/2611.26275)。提交前的 latent refinement，以及计入循环计算的 TPF。 <a class="reference-backlink" href="#cite-12-1" aria-label="返回文献 12 首次引用处" title="返回正文">↩</a>
{: .references }
