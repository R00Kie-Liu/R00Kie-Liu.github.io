---
layout: post
title: "连续 dLLM 导读（五）：连续模型怎样训练好"
date: 2026-11-19
last_modified_at: 2026-11-19
visibility: unlisted
sitemap: false
series: continuous-dllm
series_order: 5
tags: [dLLM, Diffusion, Flow Matching]
math: true
description: "具体构造可辨认率、条件熵与方差日程，解释 CE／MSE、时间调制、self-conditioning、NELBO 与完整训练更新。"
permalink: /2026/11/19/continuous-dllm-05-training.html
---

表示和动力学确定以后，训练还要决定：预测什么、把预算分给哪些噪声水平，以及是否保留上一次预测。本篇分别解释这些选择，让每项改动的作用都能落到具体训练环节上。

建议先读第 4 篇。重点是把目标、噪声路径、训练权重和辅助输入分开分析。

## 预测对象和损失函数怎样对应

| 目标 | 网络学习的量 | 常见训练形式 |
| --- | --- | --- |
| 干净表示 | 条件均值 Dₜ | 向量 MSE 或 token CE 后映射 |
| 噪声 | 条件噪声均值 | 噪声 MSE |
| 瞬时速度 | bₜ | Flow Matching 回归或去噪重参数化 |
| Score | 对数密度对状态的梯度 | Denoising score matching |
| 似然 | 一个完整概率模型给文本的概率 | 变分界及相应加权项 |

有些扩散模型把 v prediction 定义为 αε−σC 一类特定组合，它未必等于当前时间坐标下的 dx/dt。遇到 v 时，应先检查定义，再判断它是否表示瞬时速度。

CE 和 MSE 在理想条件下有时能提供同一个均值所需的信息，但有限模型、噪声权重和 embedding 联合学习，会让它们的实际效果不同。选择损失还必须与表示的可训练参数一起分析。

## 换一个预测目标也会换训练权重

在线性插值下，把同一个干净预测 Dθ 换成速度预测 bθ=(Dθ−x)/(1−t)。训练配对的真实速度是 u=C−ε=(C−x)/(1−t)，所以每个样本都有：

$$
\|b_\theta-u\|^2
=\frac{\|D_\theta-C\|^2}{(1-t)^2}.
$$

这意味着，未加权的速度 MSE 与未加权的干净预测 MSE，并不是同一个训练目标。前者在这个参数化下，等于额外用 1/(1−t)² 加权后者。接近终点时，这个权重会很大。

这条等式比较的是两种 MSE 参数化。更换预测目标时，还应检查输入输出缩放、损失权重和时间采样，否则无法区分收益来自预测对象，还是来自隐含的重加权。

## 用目标函数看清训练任务

先固定表示和噪声路径。在 token CE 的设置中，一个一般目标为：

$$
\mathcal L(\theta)
=\int_0^1 \lambda(t)\,
\mathbb E_{w,\varepsilon}\left[
-\frac1L\sum_{\ell=1}^L
\log q_\theta(w^\ell\mid X_t,t)
\right]\mathrm dt.
$$

λ(t) 指定哪些噪声水平在目标中更重要，训练时抽取 t 的分布决定怎样估计这个积分；二者可以有意配合，也可以通过重要性补偿分开。

固定时间与观测后，CE 等于真实后验熵加上预测与真实后验的 KL。高 CE 既可能来自观测信息不足，也可能来自网络误差；因此分配训练预算时，要区分不可消除的不确定性与尚未学好的部分。

把模型预测熵、真实后验熵和 CE 混用也会误判。前者只描述网络自己有多不确定；一个过度自信但经常答错的网络，可以有很低的预测熵和很高的 CE。

## CE、概率 MSE 和 embedding MSE 到底差在哪里

给定带噪状态，真实类别后验记作 p，模型输出记作 q。CE 的条件风险分解为 H(p)+KL(p||q)，最优 q=p。若 one-hot 概率上使用平方损失，最优解也仍是 p，但优化信号并不相同。

例如两类问题中，正确类别的预测只有 0.01。CE 为 −log(0.01)≈4.605，对正确类别 logit 的梯度是 −0.99。若使用半平方概率损失 ½∑(q−eᵧ)²，损失约 0.9801，对同一 logit 的梯度约 −0.0196。两者的统计最优解相同，对非常确信但错误的预测却给出很不一样的修正。

改到 embedding MSE，几何又进入目标。设 E 的每行为一个词向量：

$$
\|E^\top q-E^\top e_y\|^2
=(q-e_y)^\top EE^\top(q-e_y).
$$

EEᵀ 是由词表几何决定的误差度量。把概率错分给一个距离很近的词，可能只付出很小的 embedding MSE；CE 则仍然直接惩罚正确词的概率。若 E 也在训练，这个误差度量本身还会变化。

这解释了两件看似矛盾的事：分类后验可以为连续采样提供正确均值；likelihood 目标下学习出的几何，却未必应该再用额外 CE 强行拉开。[RePlaid](https://arxiv.org/abs/2605.18530v3) 的结果提醒我们，最终要比较完整概率模型，而不是把“CE 最优解正确”理解成“任何 CE 组合都优于任何 MSE”。

## 为什么语言的噪声 schedule 需要单独设计

对图片而言有用的噪声区间，不一定适合 token 身份恢复。词表大小、向量间距和范数会影响 token 在何时变得可识别。第 4 篇的距离与噪声比，解释了这种依赖。

固定表示尺度后，常用下面的系数比标记信噪水平，称为 SNR。其绝对含义仍与表示的归一化约定有关：

$$
\mathrm{SNR}(t)=\frac{\alpha_t^2}{\sigma_t^2}.
$$

log SNR 与 log NSR 的符号相反：后者是噪声与信号的比。统一时间方向时，也要一起核对这个符号，避免把更干净与更嘈杂读反。

## 三种调整不能混成一件事

第一种是改变路径 αₜ、σₜ，也就是每个时间对应什么分布。第二种是改变训练中抽取 t 的频率或损失权重。第三种是保持模型不变，只改变推理时有限步的落点。这三者可以有关联，但作用不同。

比如，若原目标是积分 λ(t)ℓ(t)，改为从 q(t) 采样，为保持同一个积分目标，需要对应的重要性权重：

$$
\int_0^1\lambda(t)\,\mathbb E[\ell(t)]\,\mathrm dt
=\mathbb E_{t\sim q}\left[\frac{\lambda(t)}{q(t)}\ell(t)\right].
$$

改变抽样分布而不补偿，会改变训练权重；加入相应补偿，则可以保持原积分目标，只改变估计方式。[CDCD §3.3](https://arxiv.org/abs/2211.15089) 的 time warping 可在这个框架下理解。

[LangFlow §4](https://arxiv.org/abs/2604.11748) 根据去噪过程中信息变化的区间分配训练与采样预算。核心问题是 CE 随噪声水平在哪里变化最快，以及把计算移到这些区域后，是否改善了生成。

## 时间重参数化会怎样同时影响训练和采样

令 τ=F(t) 单调递增，并保持两端对应。若在 τ 上均匀抽样，诱导到 t 上的密度为 F′(t)。若沿 τ 写 ODE，则速度也需要链式法则：

$$
\frac{\mathrm dx}{\mathrm d\tau}
=b_t(x)\,\frac{\mathrm dt}{\mathrm d\tau}.
$$

在 τ 上均匀取点，会在原来的 t 轴上得到非均匀网格。这样可以展开狭窄的 token 识别过渡区，让时间条件和有限步求解都更充分地利用它。

重参数化必须同时处理抽样密度、损失权重和速度的链式法则。遗漏相应变换，就可能改变训练目标或生成动力学。

### 用信息曲线理解 schedule 的目标

把平均剩余不确定性记为 H(t)。沿噪声到数据方向，它总体上下降。若希望各步消除近似相同的信息，可以把更多节点放在 −H′(t) 较大的区间，也就是不确定性下降最快的地方。

关键在于变化率。纯噪声端的 H 可能很高，但时间稍变时几乎没有新信息；值得增加节点的，是信息快速改变的过渡带。

![相同预算下的时间节点安排](/assets/img/blog/continuous-dllm/time-schedule.png)

图 1 相同节点预算下，均匀时间网格可能只有少数节点落在信息快速变化的区间；重新安排时间，可以在该区间分配更多节点。两行各有 8 个节点。过渡带位置仅作示意，实际应由表示、路径和统计量确定；这张图解释节点分配的直觉，不代表某篇方法的测量结果。

CDCD 的 time warping、FLM 的 token 可辨认性日程、LangFlow 的信息分配原则，都在重新安排去噪进度。它们采用的统计量和优化方式不同，合适的节点仍要结合表示与任务验证。

## 三种时间日程怎样具体构造

时间分配至少有三种可以落地的准则：根据 one-hot 的可辨认概率，根据训练中观察到的条件不确定性，以及根据 likelihood 估计的方差。它们有时得到相似的时间网格，但优化理由不同。

### FLM 从一个可计算的恢复概率构造时间变换

考虑一个词的 one-hot 观测 X=t eᵧ+(1−t)σε。直接对 X 取 argmax，正确类别获胜的条件是：

$$
\varepsilon_y+\frac{t}{(1-t)\sigma}
>\max_{k\ne y}\varepsilon_k.
$$

给定 εᵧ=z，其余 V−1 个独立 Gaussian 都必须小于 z+κ，其中 κ=t/((1−t)σ)。因此恢复概率可以写成一维积分：

$$
P_{\mathrm{rec}}(t)
=\int_{-\infty}^{\infty}
\phi(z)\,\Phi(z+\kappa)^{V-1}\,\mathrm dz.
$$

φ 是标准高斯密度，Φ 是其累积分布函数。这个公式说明了 V 为什么重要：即使每个错误坐标单独超过正确坐标的概率很小，V−1 个竞争者的最大值也可能很大。

用 Gaussian 最大值约为 √(2 log V) 做粗略估计，V=50,000、σ=1 时，信号与竞争噪声相当的位置大约在 t≈4.65/(1+4.65)≈0.82。它只是过渡位置的量级估计，不是精确的 50% 恢复时刻。精确曲线由上面的一维积分计算。

[FLM v3 §4](https://arxiv.org/abs/2602.16813v3) 将这条恢复曲线标准化为进度 F(t)，在 F 上均匀抽样和放采样节点，让每一步带来接近的恢复进展。它保留原来的线性插值；改变的是训练和推理把多少预算放在各个物理噪声水平。

不要记成“语言一定在 t≈0.82 才开始去噪”。向量范数、噪声尺度和插值系数都会移动这个位置。例如 LangFlow 在其 embedding 与 VP 参数化下，观察到主要 CE 变化位于原始时间的高噪声小区间。日程必须跟着表示标定，不能只复制一个 t 轴上的区间。

### CDCD 与 LangFlow 从 CE 拟合信息变化

记每位置平均剩余不确定性为：

$$
H(t)=\frac1L\sum_i H(W_i\mid X_t,\text{context}).
$$

纯噪声端的值由 token 先验与给定上下文决定，未必等于 log V。若希望累计信息进展均匀，可以定义：

$$
F(t)=\frac{H(0)-H(t)}{H(0)-H(1)},
\qquad
F'(t)\propto-H'(t),
\qquad
t=F^{-1}(u),\quad u\sim U[0,1].
$$

在最优分类器上，平均 CE 就是条件熵；实际 CE 还加上模型后验误差。CDCD 因此用观测 CE 拟合单调曲线，再归一化、反演。一个具体的实现过程是：把时间分箱，累计各箱的 CE；用单调分段线性函数拟合它；将归一化的信息进展当 CDF；抽均匀 u，通过逆 CDF 得到训练时间；推理时同样取信息进度的等距分位点。

[LangFlow v4](https://arxiv.org/abs/2604.11748v4) 在 logNSR 坐标 γ=log(σ²/α²) 上做这件事。VP 路径中，σ²=sigmoid(γ)、α²=sigmoid(−γ)，噪声到数据对应 γ 从大到小。用 γ 条件化网络，可以把原始 t 轴上很窄的高噪声区间展开；它再拟合不确定性曲线的变化率，用于训练抽样和推理节点。

观察到的 CE 很高，不代表一定要多放节点。高噪声端可能整段都难，但彼此之间几乎没有信息差别；变化率大的区域，才是上述准则重点分配预算的地方。训练初期 CE 还混有大量模型误差，拟合曲线应有平滑、慢更新和适当的探索范围，避免一次噪声估计就把某个区间永久丢掉。

### Plaid 与 RePlaid 为何从方差出发也得到相似曲线

如果固定目标是积分 J=∫ℓ(t)dt，改成从 p(t) 抽样时，应估计 ℓ(t)/p(t)。这样改变抽样频率不会改变 J，但会改变估计方差。最小化二阶矩得到的理想密度满足：

$$
p^*(t)\propto
\sqrt{\mathbb E[\ell(t,\text{sample})^2]}.
$$

若目标是降低梯度方差，对应的量则是梯度范数的二阶矩。它与“只采平均 loss 最大的位置”不同，也与“直接改变无权重 CE 的训练分布”不同。

[RePlaid §5](https://arxiv.org/abs/2605.18530v3) 延续 likelihood 训练，在保持期望目标的同时学习抽样安排。其理论分析把加权 MSE 与 Gaussian 信道的 I–MMSE 关系联系起来，解释为何近似均匀的损失贡献会对应接近线性的信息变化。实际结果中，学出的日程也呈现近似线性的 token CE，但它并非先把“CE 必须线性”硬编码成训练目标。

因此，看到两个方法都使用非均匀节点，仍然需要知道它们是否做了重要性补偿：没有补偿时是在重新分配优化重点，有补偿时可能是在用更低方差估计同一个目标。

## self-conditioning 让网络带着上一次判断继续工作

普通去噪器只读取当前状态和时间。Self-conditioning 额外输入一次旧预测：

$$
\widehat C_i=D_\theta(x_{t_i},t_i,z_i),\qquad z_{i+1}=\widehat C_i.
$$

直觉上，它给网络保留了一份“上次认为答案是什么”的草稿。连续状态本身仍然更新，草稿只作为辅助条件。

![Self-conditioning 的两条信息通道](/assets/img/blog/continuous-dllm/self-conditioning.png)

图 2 Self-conditioning 的推理信息流。青色通道是由采样器更新的连续状态，琥珀色通道是传给下一轮的干净状态预测草稿。两处网络框表示同一组权重在不同时间的调用。草稿与连续状态承担不同作用；训练时如何生成无梯度草稿，见下文伪代码。

一种常见训练做法，是随机为部分样本增加辅助预测：先做一次无梯度前向，再把结果作为草稿输入有梯度的预测分支。推理时，第一轮从空草稿开始，后续轮次使用上一步预测。

[Analog Bits](https://arxiv.org/abs/2208.04202) 使用了这种预测反馈技巧，LangFlow 则研究了它在语言模型中的作用。评估时既要统计额外前向成本，也要观察错误草稿是否导致重复或过早确定。

## self-conditioning 的训练接口长什么样

下面是一种教学用的训练结构。detach 把第一份预测当作常量输入，使第二次前向的梯度不再穿过它。具体启用概率与输入形式可以随实现变化。

```text
z = 全零辅助输入
以一定概率执行
  在不记录梯度的条件下
    z = network(noisy_state, time, zeros)
  z = detach(z)

prediction = network(noisy_state, time, z)
loss = 对 prediction 计算训练目标
只对有梯度的预测分支反向传播
```

若训练时的草稿总来自同一时间的一次预测，采样时却来自上一时间步，模型面对的条件分布会有差异。步数变少或步长很不均匀时，这个差异值得专门检查。

另一个常见混淆是把 self-conditioning 等同于缓存。它存的是预测信息，不一定让下一轮少算注意力；如果为了构造草稿额外跑一次网络，计算成本还会增加。

## 去噪 Transformer 的输入和注意力怎样设计

对 embedding 状态，输入通常是 B×L×d。投影到 hidden width dmodel，经过多层双向 Transformer，再输出每位置的词表概率或干净向量。双向注意力让当前未知位置互相交换证据；它并不等于训练时看到真实答案，答案只通过当前噪声水平允许的信息进入输入。

时间必须进入网络。否则相同向量在高噪声端可能只是随机扰动，在低噪声端却可能是很可靠的 token 证据，网络无法判断应信任多少。一种常见实现是 adaptive normalization：

$$
\widetilde h
=a_\theta(t)\odot\operatorname{Norm}(h)+c_\theta(t),
\qquad
h_{\mathrm{next}}=h+g_\theta(t)\odot F_\theta(\widetilde h).
$$

a、c 调整特征尺度与偏置，g 控制残差更新。RoPE 负责位置关系，时间条件负责噪声水平，二者不是同一个输入。CDCD 使用时间相关的 shift、scale，现代 DiT 类模型广泛使用自适应归一化与残差门控。

Self-conditioning 需要再输入旧预测 z。ELF 把 X 与 z 沿特征维拼接，再投回 d；LangFlow 使用带额外投影的残差融合，并将新投影初始化为零，使模型开始时仍处理原来的 noisy input。两者都是把旧预测变成网络条件，不是直接把旧预测当作当前积分状态。

ELF 还采用另一种时间接口：在序列前加入表示时间、guidance 强度和 denoise/decode 模式的 control tokens。这样可以共享大部分网络权重。不同架构的时间接口会影响参数量、表示方式与缓存条件，不能只按“都是 Transformer”认为计算完全相同。

### 条件生成需要明确的 source、target 边界

[DiffuSeq](https://arxiv.org/abs/2210.08933) 把源文本和带噪目标放在一起建模；[SeqDiffuSeq](https://arxiv.org/abs/2212.10325) 使用独立 source encoder，把它的输出提供给各次去噪。后一种方式可避免每一步重新编码源文本。块生成还可以让前缀使用因果注意力，当前目标块内部使用双向注意力。

注意力矩阵的允许范围可以写成：

| Query 所在位置 | 可读取的 Key |
| --- | --- |
| 已完成前缀 | 自己与更早的前缀 |
| 当前带噪块 | 全部已完成前缀和整个当前块 |
| 后续尚未生成块 | 当前调用中不作为答案输入 |

训练时要与推理时的可见性一致。若当前块的真实 clean embedding 被意外保留在未标记的辅助通道中，CE 会很好看，但模型只是复制答案；若训练总见完整噪声序列，推理却突然把任意位置替换成可靠上下文，也会产生接口偏移。

## 每个 token 有自己的时间以后，训练样本是什么

同步训练给全句一个 t。异步训练给每个位置 τᵢ：

$$
X^i=(1-\tau_i)\varepsilon^i+\tau_i C^i,
\qquad \boldsymbol\tau\in[0,1]^L.
$$

τᵢ=1 表示该位置已干净；较小值表示还不确定。网络不仅要读 X，还要知道各位置的 τ，才能区分“一个可靠的条件 token”和“恰好靠近这个 token 的噪声向量”。

这让训练任务接近离散 dLLM 熟悉的部分完成输入，但未完成部分仍保留连续证据。随机 clean prefix、随机 infilling、按位置推进、根据预测改变进度，是不同的训练分布；仅有 tokenwise time 的参数接口，不表示网络已学过它们全部。

提交机制也需要训练支持。MLFM 在已知位置保留干净向量，只对未决定部分去噪；Lafa 在块内学习不同位置的进度，并让已完成前缀的时间保持固定。这样才能在推理时把某些预测变成条件，而非向同步模型输入它从未见过的混合状态。

## ELBO 先理解它为什么出现

连续潜变量模型通常不能像 AR 一样，直接把一串 next token 概率相乘得到文本似然。一个办法是定义包含潜变量的完整概率模型，再优化数据对数似然的下界。

$$
\log p_\theta(w)\geq \mathrm{ELBO}(w),
\qquad
-\log p_\theta(w)\leq-\mathrm{ELBO}(w).
$$

取负号后，ELBO 对应一个 NLL 上界。它由完整概率模型的重建、先验、中间噪声路径与权重共同决定。[Plaid](https://arxiv.org/abs/2305.18619) 采用了这类似然训练路线。

下面先推导下界的方向，再说明它与真实似然相差什么。若模型还使用 self-conditioning，似然评估也要明确草稿条件如何构造，保证评价的是所定义的概率模型。

## 用 Jensen 不等式看懂 ELBO 的方向

用 z 统称潜变量，q(z|w) 是你选的辅助分布。把数据似然改写为 q 下的期望，再利用 log 是凹函数，就得到：

$$
\begin{aligned}
\log p_\theta(w)
&=\log\mathbb E_{q(z\mid w)}
\left[\frac{p_\theta(w,z)}{q(z\mid w)}\right]\\
&\geq
\mathbb E_{q(z\mid w)}
\left[\log p_\theta(w,z)-\log q(z\mid w)\right].
\end{aligned}
$$

右侧就是 ELBO。扩散模型中的 z 可以表示一串有噪状态；展开联合概率后，重建、先验和中间转移项就把去噪训练连接到完整概率模型。

## ELBO 与真实似然之间的差距

在刚才的 Jensen 下界中，代入真实后验 pθ(z|w)，可进一步整理出恒等式：

$$
\log p_\theta(w)
=\mathrm{ELBO}(w)
+\mathrm{KL}\!\left(q(z\mid w)\|p_\theta(z\mid w)\right).
$$

KL 非负，说明下界与真实 log p(w) 之间还有一段距离。负 ELBO 改善，可能来自模型概率提高，也可能来自辅助后验更接近真实后验。比较不同方法时，必须明确这段差距由什么决定。

在一个有噪状态链的模型中，常见负 ELBO 会分成三类：干净端的文本重建项、中间转移分布之间的 KL、噪声端与所选 prior 的匹配项。高斯条件下，有些 KL 可以化成带时间权重的 MSE；这就是似然目标与去噪回归之间的一条连接。

若随意丢掉这些时间权重，新的目标可能更适合某种样本质量，却不再是同一条严格似然界。看到 loss 名称相似，要检查它在概率模型里到底对应哪一项。

## 连续 NELBO 中的每一项训练谁

Jensen 不等式解释了下界方向，具体训练还需要知道各项。对有限噪声端点、指定高斯路径与 token decoder，一类连续变分扩散界可写为：

$$
-\log p_\theta(w)
\leq
\mathcal L_{\mathrm{decode}}
+\mathcal L_{\mathrm{prior}}
+\frac12\int_0^1
\operatorname{SNR}'(t)
\mathbb E\|D_\theta(X_t,t)-C(w)\|^2\,\mathrm dt.
$$

这里仍取噪声到数据方向，SNR=α²/σ² 随 t 增加。各项的职责是：

- decode 项要求最少噪声的连续状态能恢复 token，约束文字读出和可学习表示。
- prior 项比较最强噪声端的真实前向分布与推理初始化分布，处理端点没有完全达到标准 Gaussian 的误差。
- 积分项要求网络在所有噪声层级恢复干净表示，权重由 SNR 的变化决定。

训练一次只需抽一个 t、一个噪声 ε，构造 Xₜ，再估计相应积分项，不需要展开整段生成轨迹。最少噪声端的重建与最强噪声端的 prior，则要按所定义模型一并计算。

这也是 Plaid、RePlaid 与纯 denoising CE 的区别。前者用一个完整的离散文字 likelihood bound 共同约束表示、去噪器与 decoder；后者主要学正确的后验，若要报告文字 likelihood，还需要额外定义生成与读出概率并建立相应的评估界。普通 CE 训练 loss 不能直接指数化后称为测试文本的 perplexity。

有限端点很重要。如果 σ₁=0、α₀=0 等极限被截成数值上的小常数，界中的解码与 prior 也必须跟着该模型计算。只截积分，却仍沿用理想端点的常数，会得到不再对应同一模型的数字。

## 一份能看清梯度与张量的训练更新

下面给一个 embedding-CE 模型的参考更新。它包含 clean prefix、可学习但归一化的词表、部分样本的 self-conditioning，以及只对目标区域计算的 CE。它用于说明完整数据流，不是声称所有论文采用同一配方。

~~~python
# tokens: [B,L]；net(x,tau,draft) -> [B,L,V] logits
# prefix_length: [B]，每项在 0..L-1，保证目标区域非空
# optimizer 同时管理 net 和 embedding 的可训练参数
# embedding.weight: [V,d]；sample_time 已实现所选时间分布
B, L = tokens.shape
E = torch.nn.functional.normalize(embedding.weight, dim=-1)
C = E[tokens]                                      # [B,L,d]

t = sample_time(B).to(C.device)                    # [B]
tau = t[:, None].expand(B, L).clone()
positions = torch.arange(L, device=C.device)[None]
active = positions >= prefix_length[:, None]       # [B,L]
tau = torch.where(active, tau, torch.ones_like(tau))

eps = torch.randn_like(C)
x = tau[..., None] * C + (1-tau[..., None]) * eps
draft = torch.zeros_like(x)
use_sc = torch.rand(B, device=C.device) < 0.5
with torch.no_grad():
    if use_sc.any():
        q0 = net(x[use_sc], tau[use_sc],
                 torch.zeros_like(x[use_sc])).softmax(-1)
        draft[use_sc] = q0 @ E                    # 不传过这次预测

logits = net(x, tau, draft)
per_token = torch.nn.functional.cross_entropy(
    logits.transpose(1, 2), tokens, reduction="none")
loss = (per_token * active).sum() / active.sum()
optimizer.zero_grad()
loss.backward()
optimizer.step()
~~~

这段更新里，真实答案进入输入的途径是已经声明的加噪路径和 clean prefix；草稿来自模型自己能读到的同一观测。第二次预测的梯度仍会通过 x 回到可学习 E，但不会穿过第一次无梯度预测。若要固定 E，关闭它的梯度即可；若换成 likelihood 训练，就必须加入上一节的完整项和时间权重，而非只给这段 CE 换个名称。

每份样本的 self-conditioning 概率是 1/2，且辅助前向只计算被选中的样本，平均训练代价因而约为额外半次前向，反向仍只通过主分支。普通推理重用上一步已经得到的草稿，不自动多花一次调用；在同一状态额外做几轮 fixed-point refinement，才会增加 NFE。

运行这类更新前，还要固定三个直接影响结果的选择：时间分布是否改变目标权重，prefix／infilling 样本怎样构造，以及不同长度目标如何归一化。它们决定了网络实际训练的条件任务，不能留给推理阶段临时补上。

## 代表方法各自改变了什么

| 方法 | 在本篇框架中的主要选择 |
| --- | --- |
| CDCD | 分类预测构造 score 联合词表 embedding 与 time warping |
| Plaid | 从完整概率模型出发进行似然界训练 并优化相关表示和日程 |
| LangFlow | 连接 embedding 连续模型与 flow 训练 改善信息分配和 self-conditioning |
| One-hot FLM | 固定身份编码 用 CE 预测均值 并重新分配 token 识别进度 |

把每种方法拆到这些选择上，就能看清其改进落在哪里。完整效果仍由表示、主干、目标、日程和采样器共同决定。

## 方法归纳与出处

[CDCD](https://arxiv.org/abs/2211.15089)、[Plaid](https://arxiv.org/abs/2305.18619)、[LangFlow](https://arxiv.org/abs/2604.11748) 与 [FLM](https://arxiv.org/abs/2602.16813) 提供了相关实现。逐项复现发布模型时，再核对其损失系数、时间拟合器与训练配置。

**建议的小实验：** 固定表示和网络，只画逐噪声区间的 CE、token 恢复率和梯度尺度，再分别改变时间采样和 self-conditioning。先找训练预算落在哪里，比同时引入五个技巧更容易解释结果。

## 自测参考答案

保持积分目标不变时，非均匀抽 t 需要 λ(t)/q(t) 的补偿；不补偿通常是在改变训练权重。Self-conditioning 则改变了网络的条件输入，和这两种时间选择并非同一个改动。

判断训练是否变好，可以先固定主干、表示和预算，分开画不同噪声区间的预测误差。再看整体似然和采样结果，避免只因一个总体 loss 下降就判定机制正确。

## 阅读导航

[返回总目录]({% post_url 2026-11-19-continuous-dllm-index %}) · [上一篇 第4篇 文本应该放在哪个连续空间]({% post_url 2026-11-19-continuous-dllm-04-representations %}) · [下一篇 第6篇 连续模型怎样采样和控制]({% post_url 2026-11-19-continuous-dllm-06-sampling-guidance %})
