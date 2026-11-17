---
layout: post
title: "MoE我们喜欢你（十）：视觉生成与原生多模态中的专家设计"
series: moe-we-like-you
series_order: 10
last_modified_at: 2026-11-17
date: 2026-07-23
tags: [MoE, Diffusion, Video, WorldModel]
description: "从噪声阶段、视觉 token 与文本/视觉处理路径出发，比较图像和视频 MoE 的专家选择、计算分配、语义引导与跨模态共享。"
---

<a id="0-视觉生成中的专家选择粒度"></a>
<a id="1-去噪迭代时空-token-与采样开销"></a>
<a id="0-图像和视频生成为什么需要不同的专家设计"></a>

## 0. DiT 怎样使用 timestep

Dense DiT 已经会根据噪声阶段调整特征处理。讨论时间专家之前，可以先从它的条件注入方式出发：timestep 经过怎样的编码，改变 block 中的哪些计算，以及哪些参数仍由所有阶段共享。

### 0.1 timestep 的编码与输入

在当前噪声阶段，DiT 接收带噪图像或 latent $x_t$，将它切成 patch 并映射为视觉 token。模型还接收 $t$，用来标识当前的噪声状态。视觉位置编码描述 token 在图像或视频中的位置，扩散 timestep 描述去噪过程中的状态；同一个噪声阶段会同时处理许多视频帧。

固定单调噪声日程下，timestep 与噪声强度相互对应。采样循环中的第几次迭代则随 solver 和步数配置变化，不能直接代替模型使用的 $t$ 或 noise level $\sigma$。

以[原始 DiT][dit-paper]的类别条件模型为例，$t$ 先经过正弦位置编码和 MLP，再与类别 embedding 相加：

$$
\begin{aligned}
e_t&=\mathrm{MLP}(\mathrm{SinCos}(t)),\\
c_t&=e_t+\mathrm{Embed}(y).
\end{aligned}
$$

$y$ 是类别标签，$c_t$ 是送入各 block 的条件向量。[官方实现][dit-code]中，视觉 token 加上空间位置编码后进入主干，$c_t$ 则交给各 block 的条件调制网络。文本条件 DiT 还可以通过 cross-attention、联合序列等方式融合文本；这里先用原始实现说明 timestep 的作用。

[![DiT 的时间条件输入与 adaLN-Zero block](/assets/img/blog/moe-visual-generation/dit-adaln-zero-conditioning.png)](/assets/img/blog/moe-visual-generation/dit-adaln-zero-conditioning.png)

*图 1：视觉 latent 经 patchify 进入主干，timestep 与类别条件进入调制网络。adaLN-Zero 分别调整 attention、FFN 的归一化特征与残差输出。*

### 0.2 adaLN-Zero：调制特征与残差分支

DiT 的 adaLN-Zero 从 $c_t$ 生成 scale、shift 和残差缩放参数。设 $h$ 为 block 内的 token 序列，$F$ 为一条 attention 或 FFN 分支，省略层号后可写成：

$$
\begin{aligned}
u&=(1+\gamma(c_t))\odot\mathrm{LN}(h)\\
 &\quad+\beta(c_t),\\
h'&=h+\alpha(c_t)\odot F(u).
\end{aligned}
$$

$\gamma$ 与 $\beta$ 调整归一化后的特征，$\alpha$ 控制分支输出加入残差的幅度。每个 block 为 attention 和 FFN 分别生成一套参数，共六个通道向量；同一样本内，这些向量广播到各视觉位置。不同位置仍有各自的输入特征，并通过 attention 交换信息。

“Zero”来自调制输出层的零初始化：训练开始时 $\alpha=0$，各残差 block 从恒等映射开始。训练后，这些值可以随条件变化。这里的 gate 是对分支输出的连续缩放，没有进行专家打分或 top-k 选择。

因此，即使 attention 和 FFN 的权重始终共享，DiT 也能随 timestep 改变特征与残差贡献。后文所谓“时间专家”，需要在这个已有能力上解释新增的作用。

### 0.3 有了时间调制，为什么还需要专家

adaLN 随时间调制特征，但同一套 attention、FFN 权重仍要拟合整条去噪轨迹。阶段专家为不同噪声分布增加专用参数，让一部分权重只接收对应区间的训练更新。逐 token MoE 则进一步让同一轮中的主体、背景和运动边界选择不同变换。若目标是加速，还需要决定各阶段应执行多少计算，仅改变固定 top-k 的专家 ID 并不会减少调用次数。

时间可以通过主干调制进入 router 读取的 hidden state，也可以直接决定专家选择，或进入被选专家内部的条件变换。这些设计分别改变路由输入、执行路径和专家函数，后面会逐一讨论。

文本条件与视觉 token 也不对称：文本描述希望生成什么，带噪视觉状态则要不断被更新。原生多模态模型进一步把文本预测、视觉理解和图像生成放进同一个主干，参数共享也会影响不同目标之间的训练干扰。

因此，设计 MoE 时有三件事需要分别确定：在哪个层级设置专家，用什么信号选择专家，以及每个位置获得多少专家计算。

| 设计问题 | 可以使用的信号 | 改变什么 | 对应目的 |
|---|---|---|---|
| 哪些参数处理当前噪声阶段 | timestep、噪声强度 | 整套 denoiser、block 路径或 LoRA | 扩大阶段容量，调整共享程度 |
| 各阶段运行多少主干计算 | 时间区间、步骤敏感度 | 子网络深宽、大小模型调用 | 降低完整采样成本 |
| 每个视觉位置使用哪些 FFN | 当前 hidden state、文本条件、空间与时间上下文 | token–expert 分派 | 在局部内容之间分配容量 |
| 每个位置分到多少计算 | expert-choice 竞争、容量预测 | 专家调用数量 | 将计算转向需求更高的位置与采样阶段 |
| 专家应怎样形成分工 | 语义原型、专家分组、模态或任务约束 | 路由训练与候选范围 | 减少无效重复或任务干扰 |
| 文本、视觉共享哪些参数 | 模态类型、理解/生成路径 | FFN、router 或更完整的 Transformer 参数 | 保留各自处理能力，同时交换语义信息 |

下面先从阶段专家讨论参数共享，再进入逐位置路由、计算量分配，以及文本、视觉与任务之间的专家分工。

<a id="2-模型级选择按-timestep-换-denoiser"></a>

<a id="1-按噪声阶段切换整套-denoiser"></a>

## 1. 噪声阶段怎样选择参数与执行路径

### 1.1 Wan2.2：按噪声区间选择模型

模型级 denoising expert 将一条采样轨迹拆成几个区间，为各区间保存独立的主干参数。每个 denoiser 仍接收 timestep 条件，但只需专门拟合自己负责的噪声分布。

以预测噪声的参数化为例，一个 denoiser 可以写成：

$$
\hat{\epsilon}=D_{\theta}(x_t,t,c)
$$

所有 timestep 共用同一套参数 $\theta$。timestep expert 做的是先把 $t$ 或 noise level $\sigma$ 切成几个连续区间 $S_1,\cdots,S_m$，再定义一个固定映射：

$$
\begin{aligned}
g(t)&=i,\quad t\in S_i,\\
\hat{\epsilon}&=D_{\theta_i}(x_t,t,c).
\end{aligned}
$$

训练时采样到哪个 $t$，就更新对应的 $D_{\theta_{g(t)}}$；推理时 scheduler 走到哪个 $t$，也只调用对应的 expert。这里没有 token-level top-k，也没有 per-patch all-to-all。每次 denoiser 评估只执行当前区间的一个 expert；一个采样 step 是否包含 CFG 等额外评估，由 pipeline 决定。总 checkpoint/参数量变成多份，显存压力取决于专家是否同时常驻 GPU，或在 high/low 阶段之间 offload。

[eDiff-I][ediffi]先训练覆盖全噪声分布的基线，再从该基线初始化不同噪声区间的专家。[ERNIE-ViLG 2.0][ernie-vilg]将 1000 个训练 timestep 分成 10 段，每段使用一个 U-Net expert，text encoder 共享。推理时沿轨迹切换 denoiser，各个区间的专家共同完成一张图，而每次模型评估只执行其中一套。

[Wan2.2][wan22]的 A14B 系列使用 high-noise 与 low-noise 两个专家。前者负责高噪声区间，后者负责低噪声区间；每次只执行一个约 14B 的模型，总参数容量则包含两套权重。flow 模型预测的具体量可以是速度场，按噪声阶段选择参数的逻辑相同。

[![Wan2.2 timestep MoE](/assets/img/blog/moe-we-like-you/10-wan2-2-timestep-moe.png)](/assets/img/blog/moe-we-like-you/10-wan2-2-timestep-moe.png)

*图 2：Wan2.2 沿 denoising timestep 划分 high-noise / low-noise experts。*

代码层面，Wan2.2 的 T2V/I2V A14B pipeline 创建 `low_noise_model` 和 `high_noise_model` 两个 DiT checkpoint。采样前先算：

```python
boundary = config.boundary * num_train_timesteps
```

采样循环里，每个 step 根据当前 scheduler timestep 选择模型：

```python
if t >= boundary:
    model = high_noise_model
else:
    model = low_noise_model
```

在该 CFG 路径中，conditional / unconditional 两次 forward 都使用当前选中的模型。高、低噪声专家各执行多少次，由阈值和 scheduler 共同决定；offload 配置则决定非当前专家的权重放在哪里。同一步内，所有视觉位置都经过同一个 denoiser。

### 1.2 MOVA：视频专家切换与音视频交互分别处理

[MOVA][mova-paper]在 Wan2.2 I2V A14B 视频主干之外，加入 1.3B 音频 DiT 和 2.6B 双向 Bridge，构成约 32B total / 18B active 的联合音视频生成模型。

[![MOVA 的视频塔、音频塔与双向 Bridge](/assets/img/blog/moe-we-like-you/10-mova-dual-tower-architecture.png)](/assets/img/blog/moe-we-like-you/10-mova-dual-tower-architecture.png)

*图 3：MOVA 的视频塔继承噪声阶段专家，音频塔与视频塔通过双向 cross-attention Bridge 交换隐藏状态。*

视频塔仍按高、低噪声阶段选择模型。音视频同步由另一条机制处理：在交互层中，音频隐藏状态作为视频的 cross-attention 输入，视频隐藏状态也反向注入音频塔；位置编码将两个 VAE 的不同时间采样率对齐到同一物理时间尺度。这里的两个 tower 是模态处理路径，视频 tower 内的两个 denoiser 才是按噪声阶段选择的专家。

这一划分也影响训练。MOVA 为适配 FSDP 的计算图要求，在奇数训练步为整个 batch 采样高噪声区间并更新 high-noise DiT，偶数步采样低噪声区间并更新 low-noise DiT；音频塔和 Bridge 每步都更新。音视频各自的噪声采样还可以解耦，不要求两个模态始终处于相同噪声状态。

模型级切换为不同阶段保存完整权重。若希望保留更多公共参数，也可以把时间选择下移到 block、通道或 adapter；这些结构还能进一步与逐位置的内容路由组合。

### 1.3 Block 内的时间门控：RAPHAEL、DTR 与 Switch-DiT

[RAPHAEL（NeurIPS 2023）][raphael]在 U-Net 的 Transformer block 中同时设置 space-MoE 和 time-MoE。space-MoE 根据文本概念及其对应的图像区域组织特征，time-MoE 则用时间门控为当前噪声阶段选择专家。相邻 block 可以选择不同的专家，整条采样轨迹因而由多层局部选择共同构成，不需要每个阶段都切换完整 U-Net。

[![RAPHAEL 的空间专家与时间专家](/assets/img/blog/moe-visual-generation/raphael-time-space.png)](/assets/img/blog/moe-visual-generation/raphael-time-space.png)

*图 4：RAPHAEL 在同一个 block 中分别处理文本区域与噪声阶段。右下的时间门控根据 timestep 选择处理视觉特征的专家。*

[Denoising Task Routing（DTR，ICLR 2024）][dtr]从共享程度出发：为各 timestep 预设通道 mask，让相邻时间步的激活窗口重叠，距离更远的任务共享更少通道。它用非均匀的窗口移动速度，为不同噪声阶段安排任务专属通道。这属于单一网络内的通道路径设计，没有增加一组独立 FFN 专家；固定激活比例与随时间变化的通道重叠也应分开理解。

[![DTR 的时间步通道掩码](/assets/img/blog/moe-visual-generation/dtr-timestep-channel-mask.png)](/assets/img/blog/moe-visual-generation/dtr-timestep-channel-mask.png)

*图 5：横轴为通道、纵轴为 timestep。黄色窗口随时间移动，窗口位置与移动速度决定不同任务共享多少通道。*

[Switch-DiT（ECCV 2024）][switch-dit]将时间选择变成可训练门控。第 $\ell$ 个 block 的 gate 只接收 timestep embedding $e_t$，输出专家分数并取 top-$k$；论文的主要配置为三个专家选两个。因此，同一噪声阶段的不同样本和空间位置使用同一组门控权重，而专家仍处理各自的输入特征。

[![Switch-DiT 的时间门控与通道调制](/assets/img/blog/moe-visual-generation/switch-dit-time-gating.svg)](/assets/img/blog/moe-visual-generation/switch-dit-time-gating.svg)

*图 6：Switch-DiT 用 timestep embedding 选择 MLP experts，再将加权输出用于 block 输入的逐元素调制，并保留互补残差路径。*

它比较了直接传递专家输出与通道调制等接法，优选设计用专家输出生成调制值，并通过残差保留另一部分信息。训练还用 diffusion prior loss 对齐 DTR 式的任务关系：相近阶段保留公共路径，冲突阶段允许分开。这样可缓解所有 timestep 挤到相同专家的退化，并稳定 EMA 模型的门控。

这里的“时间专用”来自显式输入 $e_t$ 和路径先验。普通 token router 即使不直接接收 $t$，也可能从已经包含噪声信息的 hidden state 中学到阶段偏好；两种现象不能仅凭最终路由图判成相同机制。

<a id="time-lora"></a>

### 1.4 TimeStep Master：共享主干上的时间 LoRA

[TimeStep Master（TSM，ICML 2025）][tsm]面向扩散模型的参数高效微调。若所有时间步只用同一份 LoRA，它必须同时适应多个噪声分布；为每段复制完整模型，又会扩大存储和训练成本。

TSM 先将时间轴分段，在不同区间分别训练 LoRA experts，再把不同尺度的区间专家组合起来。对于当前 $t$，最细区间对应的 LoRA 作为 core expert，直接加入更新；覆盖这个 $t$ 的较粗区间 LoRA 作为 context experts，其贡献由时间相关的 gate 加权。

[![TimeStep Master 的多尺度 core 与 context LoRA](/assets/img/blog/moe-visual-generation/timestep-master-core-context.svg)](/assets/img/blog/moe-visual-generation/timestep-master-core-context.svg)

*图 7：图中用 8、4、2、1 段四种区间尺度演示组合。当前最细区间的 core expert 不经门控，较粗尺度的 context experts 提供加权补充。*

主干权重因此继续共享，阶段差异由低秩更新表达。core 不经过 gate，保证细粒度阶段适配始终存在；context 则补充更大时间范围的共性。它扩展的是 adapter 的表达能力，不能把其微调收益直接解释成“每轮少执行几个完整 FFN”的加速。

<a id="temporal-budget"></a>

### 1.5 时间区间的计算预算：DiffPruning 与 ALTER

阶段专家也可以具有不同深度或宽度。[DiffPruning（ECCV 2024）][diffpruning]先比较各 timestep 训练目标的梯度相似性，据此划分区间；再为区间专家提供可裁剪的深度和宽度，由 Expert Routing Agent 在计算预算下选择网络配置，最后微调选定的专家。

[![DiffPruning 的专家网络配置选择](/assets/img/blog/moe-visual-generation/diffpruning-routing-agent.png)](/assets/img/blog/moe-visual-generation/diffpruning-routing-agent.png)

*图 8：Expert Routing Agent 选择各阶段专家的宽度与深度配置。推理使用选定的区间模型，而非每个 prompt 重新搜索网络。*

[ALTER（2025）][alter]将层剪枝、时间路由与权重微调放进同一优化过程。其 hypernetwork 同时学习各专家子网络的逐层保留决策，以及 timestep 到这些子网络的选择，减少先剪枝、后微调之间的不一致。

[![ALTER 的时间路由与层剪枝联合优化](/assets/img/blog/moe-visual-generation/alter-temporal-pruning.png)](/assets/img/blog/moe-visual-generation/alter-temporal-pruning.png)

*图 9：ALTER 联合学习时间步的专家选择和各专家的层级剪枝配置，再按当前阶段执行对应子网络。*

两者把阶段分工落实到了执行量：当前阶段既选择参数，也选择需要运行的层或通道。与固定 top-k 的时间专家相比，这类方法需要额外回答各阶段应该分到多少预算，以及剪掉某一步的计算会怎样影响后续采样轨迹。

<a id="track-routing"></a>

### 1.6 TRACK：用轨迹分歧选择大小模型

[TRACK（2026 年 9 月）][track]不重新训练阶段专家，而是在兼容的大、小视频模型之间选择。校准时先用大模型生成参考轨迹，再让两个模型在相同的 latent、timestep、条件与 guidance 下预测，统计各步骤的归一化分歧。给定大模型调用预算，分歧较大的步骤优先保留大模型，其余步骤交给小模型。

[![TRACK 的离线轨迹校准与在线模型切换](/assets/img/blog/moe-visual-generation/track-trajectory-routing.png)](/assets/img/blog/moe-visual-generation/track-trajectory-routing.png)

*图 10：上半部分在参考轨迹上校准分歧并形成切换策略；下半部分推理时每步只执行选中的一个模型。*

这些分歧在校准集上汇总，在线不需要先运行两个模型再决定选谁。它是模型级的阶段预算策略，和在 FFN 内训练 sparse MoE 有不同的部署形态。分歧也只是步骤敏感度的代理：后续混合轨迹会偏离全大模型参考，最终生成质量仍由混合采样实验验证。

这使时间设计包含了几种不同动作：选择整套模型、选择 block 路径、混合低秩更新，或改变当前阶段的实际计算量。接下来再引入空间位置，讨论同一步内的 token 怎样使用专家。

<a id="3-token-级选择在-dit-block-内替换-mlp"></a>

## 2. 视觉 Token 的专家选择与计算量分配

### 2.1 固定 Top-k：DiT-MoE 与 LingBot-Video

逐 token MoE 可以沿用 DiT 的时间条件注入，再由当前 token 的 hidden state 选择 FFN 专家。[DiT-MoE][dit-moe]在部分 DiT block 中用 sparse MoE 替换 MLP，保留 always-on shared experts，并通过 expert-level balance loss 约束专家使用率。对当前噪声阶段的一个 patch hidden state $h_j$，简化的输出是：

$$
\begin{aligned}
y_j&=\sum_{e\in\mathrm{TopK}(r(h_j))}p_{j,e}E_e(h_j)\\
&\quad+\sum_{s=1}^{n_s}S_s(h_j).
\end{aligned}
$$

$E_e$ 是 routed FFN，$S_s$ 是 shared FFN。每个位置独立选择 top-$k$ 个 routed experts，再加上共享路径；同一位置在不同采样轮中得到的 hidden state 不同，也可以换一组专家。

[![DiT-MoE 的 shared 与 routed FFN 路径](/assets/img/blog/moe-we-like-you/10-dit-moe-framework.png)](/assets/img/blog/moe-we-like-you/10-dit-moe-framework.png)

*图 11：DiT-MoE 将 FFN 容量拆为共享部分与按 token 选择的专家部分。*

论文在 ImageNet 条件生成上扩展到 16B 级别，并分析路由随空间位置、网络深度和噪声阶段的变化。被测模型中，专家对位置和 timestep 的偏好比类别条件更明显，深层路由也更分散。这里的 timestep 信息可以通过主干的条件调制进入 hidden state，router 并不需要一个显式的“阶段专家编号”。

[LingBot-Video][lingbot-video-paper]将这类 token 级结构用于 30B-A3B 视频模型，另提供 Dense 1.3B 版本。MoE 分支包含 shared experts 与 routed experts；router 使用 group-limited selection，先选专家组再在组内选 top-k，以限制分布式通信范围。训练中的在线 bias 调整只影响选择，输出混合仍使用原始 affinity。这让它同时涉及容量扩展、负载控制和通信约束。

[![LingBot-Video 的多模态 attention 与 sparse MoE](/assets/img/blog/moe-we-like-you/10-lingbot-video-architecture.png)](/assets/img/blog/moe-we-like-you/10-lingbot-video-architecture.png)

*图 12：LingBot-Video 的 single-stream diffusion Transformer 在 attention 之后执行 shared 与 top-k routed experts。*

它支持 T2I、T2V、TI2V，训练中加入 70,000+ 小时具身数据，MoE 版本还配有 refiner。物理合理性与任务完成率的结果对应这一整套训练和生成流程；专家分工本身应根据路由与消融分析。refiner 的额外 forward 也属于最终采样时间。

固定 top-k 可以改变每个位置使用的参数，但 routed expert 调用数仍为“位置数乘 $k$”。复杂区域选了一组不同的专家，并不意味着它获得了更多计算。下面两种设计进一步改变调用数量。

### 2.2 Expert-Choice：让专家选择位置

[EC-DiT][ecdit]把选择方向反过来：每个专家对候选 token 打分，再取其中分数最高的一部分。每个专家的容量固定，一个位置被多少专家选中则可以变化。

[![EC-DiT 的专家选位置与异构计算分配](/assets/img/blog/moe-visual-generation/ecdit-expert-choice.png)](/assets/img/blog/moe-visual-generation/ecdit-expert-choice.png)

*图 13：EC-DiT 中多个专家可以共同选择细节较多的位置，部分背景位置则不进入 routed expert 分支。*

用四个位置、两个专家、每专家容量为二的示例：专家 A 选位置 1、2，专家 B 选位置 2、3。四次调用中，位置 2 得到两次，位置 1、3 各一次，位置 4 为零。容量相同，分配到各位置的计算不同；未被选择的位置仍保留 Transformer 的其它路径。

EC-DiT 将文本条件和 timestep 信息融入视觉表示，再据此选择专家输入。因此计算分配可以随内容和采样阶段变化。不过，固定每专家容量时，整组候选的 routed 调用总量仍受容量约束：变化主要发生在位置之间的分配。

### 2.3 DiffMoE：跨样本竞争与推理时容量预测

[DiffMoE][diffmoe]训练时把 batch 中的视觉 token 展平成一个全局池，让每个专家从不同样本、不同噪声阶段的 token 中选择输入。每个专家仍有固定训练容量，但不同图像分到的计算不再相同。

[![DiffMoE 的训练全局池与推理容量预测](/assets/img/blog/moe-visual-generation/diffmoe-capacity-predictor.png)](/assets/img/blog/moe-visual-generation/diffmoe-capacity-predictor.png)

*图 14：左侧按全局 token pool 进行专家选择，同时训练 capacity predictor；右侧用预测值与专家阈值决定推理中的选择。*

推理时的样本组合未必复现训练时的全局竞争。DiffMoE 用训练路由产生的选择标签，通过二元交叉熵监督一个 capacity predictor；推理时根据预测输出和每专家阈值，确定当前输入需要的 token–expert 项。因此不同样本、采样阶段可以使用不同的实际容量，阈值也提供计算量与生成质量的调节接口。

这三类分配需要分开理解：固定 token-choice 保证每位置 top-k，expert-choice 固定每专家接收量，DiffMoE 则进一步学习推理时的动态选择。专家负载更整齐与每个位置计算相同，是两种不同约束。

<a id="diff-moe-conditioning"></a>

### 2.4 Diff-MoE：让专家自己适应时间条件

第 0.2 节的 adaLN 为整个 block 提供条件调制；[Diff-MoE（ICML 2025）][diff-moe-time]进一步给每个专家增加独立的 timestep conditioning。它保留空间 token 路由，同时让被选专家通过专属条件映射适应当前噪声阶段。它与前面名称接近的 DiffMoE 改动位置不同：后者主要调整 token–expert 选择与容量。

[![Diff-MoE 的专家专属时间条件与全局特征调制](/assets/img/blog/moe-visual-generation/diff-moe-time-conditioned-experts.png)](/assets/img/blog/moe-visual-generation/diff-moe-time-conditioned-experts.png)

*图 15：router 分配空间 token，专家内的独立时间条件路径负责阶段适应；全局上下文还参与通道特征重标定。*

若把普通专家写成 $E_e(h)$，这里更接近 $E_e(h;t)$：同一专家在两个噪声阶段被选中，仍能对输入执行不同的条件变换。论文用 expert-specific 条件路径调制专家，并以空间池化得到的全局表示重标定特征通道；新增映射再通过低秩分解控制参数开销。

作者希望由专家内部承担时间适应，让 router 更专注于空间特征分工。时间调制仍可能间接改变后续路由，但这与 Switch-DiT 直接用 $g_\ell(e_t)$ 选择专家不同。因此，分析时既要统计专家 ID，也要检查同一专家内部的条件计算，单看激活频率会遗漏后一类变化。

## 3. 视觉专家需要显式的语义分工吗

### 3.1 SplitMoE：语义专家与通用专家分开训练

视频中同一物体覆盖许多相关 patch，局部区域的出现频率又很不均匀。[SplitMoE][splitmoe]针对的冲突是：若所有专家承担相同角色，再强行把 token 均匀分散到整个池中，相关区域可能被拆到不同专家，专家也难以形成稳定的语义分工。

它将 routed experts 分为 semantic 和 generic 两组，另保留 shared expert。论文配置有 20 个语义专家、80 个通用专家，每个位置固定选择 2 个语义专家和 6 个通用专家。语义分组因此没有减少每位置的八次 routed 调用，改变的是这些调用承担的功能。

[![SplitMoE 的语义原型监督与两组专家](/assets/img/blog/moe-visual-generation/splitmoe-semantic-generic.png)](/assets/img/blog/moe-visual-generation/splitmoe-semantic-generic.png)

*图 16：训练时由干净 VAE 特征与原型构造语义路由目标；带噪 token 的语义 router 学习这些目标，通用分支负责其余表示需求。*

训练中，干净视频的 VAE 特征与可学习原型计算相似度，形成语义专家的软目标，带噪状态的 router 用对齐损失学习这个分布。原型的 pull 项让它们覆盖当前与近期 VAE 特征，push 项避免不同原型挤在一起。干净特征承担训练监督，生成时路由仍来自当前带噪 hidden state。

两组的负载规则也不同：generic experts 使用 loss-free bias 均衡，semantic experts 不施加同样的均匀负载约束，而通过语义对齐组织选择。论文从 Wan2.2 的 low-noise checkpoint 改造部分 FFN，观察到两组专家的贡献随去噪阶段变化；这属于训练后形成的行为，不是按 timestep 硬切换两个池。

### 3.2 PixelMoE：普通 Token-Choice 也能形成分工

[PixelMoE][pixelmoe]在像素空间采用另一种选择：保留早期 dense blocks，后续 FFN 使用细粒度 routed experts 加 shared expert，router 直接对当前 token hidden state 做 sigmoid 打分和固定 top-k。bias 用来调节选择，选中后的输出权重按原始分数归一化，不额外加入语义或 timestep 路由监督。

[![PixelMoE 的普通 token-choice MoE 与分辨率阶段](/assets/img/blog/moe-visual-generation/pixelmoe-token-choice.png)](/assets/img/blog/moe-visual-generation/pixelmoe-token-choice.png)

*图 17：右侧 MoE 使用固定 top-k 与共享专家；左侧先低分辨率生成整体结构，再在全分辨率继续细化，两阶段复用主干。*

PixelMoE 同时使用分层 flow matching，让早期采样处理较少的低分辨率 token，后期再增加空间分辨率。这会改变位置总数；固定 top-k 路由则决定每个位置调用哪些专家。二者一起影响计算，但不是同一个机制。

论文在其像素空间配置中观察到类别与去噪阶段上的专家分工，标准 token-choice 也优于被测 expert-choice 配置。它与 SplitMoE 的表示空间、数据和模型改造方式不同，不能据此把两者排成通用路由排名。更具体的问题是：当前 hidden state 是否已经包含足够的语义和噪声阶段信息，以及额外的路由监督是否仍能改善同预算下的生成质量。

<a id="4-子空间选择magi-2-的-head-local-experts"></a>

## 4. 长视频的专家粒度与通信：MAGI-2

前面讨论专家的功能与选择规则，专家本身的宽度也会影响执行。长视频的一轮 forward 包含大量 token，完整 hidden state 向多个专家重复分发，会放大通信量。MAGI-2 将专家放进 head-local latent space，连同通信方式一起调整。

[MAGI-2 Preview][magi2] 公开材料里，模型约 114B total，每 token 激活约 6B 参数；主体 36 层使用 Multi-Head MoE。它把 3072 hidden 先映射成 12 个 256 维 head，每个 head 有独立 router 和 256 个专家，每个 head top-6。也就是说，一个 token 会在 12 个低维子空间里各自选择专家，最后再 merge 回 hidden state，而不是只触发几个完整宽度 FFN。

[![MAGI-2 Multi-Head LatentMoE](/assets/img/blog/moe-we-like-you/10-magi-2-multi-head-latentmoe.png)](/assets/img/blog/moe-we-like-you/10-magi-2-multi-head-latentmoe.png)

*图 18：MAGI-2 将完整 hidden 拆成多个 latent heads，各 head 独立路由到 head-local experts。*

这条路来自 [Multi-Head LatentMoE and Head Parallel][latentmoe-hp] 的想法：把完整 hidden 拆成多个 latent heads，让不同子空间独立路由。通信也随之变化。普通 EP 的通信量随 top-k 增长，而且每个 batch 的 send/recv shape 由 routing 结果决定；Head Parallel 先按 head 做固定形状分发，再在 head owner 内部完成 routing 和 expert compute。跨设备通信边界更规则，动态 routing 的不规则性被留在本地。

例如，一个 token 在某个 head 中选择六个专家，这六次本地选择不再对应六次跨设备的完整 hidden 分发。跨设备先确定 head 归属，目标设备在该子空间内完成动态路由，再合并输出。专家变窄后，单次矩阵乘也变小，因而需要路由排序、连续布局和 grouped expert compute 配合；固定通信形状省下的时间，仍要扣除拆分、合并与小算子的执行开销。

<a id="6-应用案例图像视频与同步音视频"></a>

## 5. 文本条件与视觉 Token 怎样经过 MoE

### 5.1 HiDream-I1：双流阶段先分开处理，单流阶段再共享

[HiDream-I1][hidream-paper]是 17B 参数的 sparse DiT。它使用 CLIP、T5 与 Llama 的文本特征，图像则由带噪 VAE latent patchify 得到。双流 block 为文本和图像分别保留投影与 FFN，再在联合 attention 中交换信息；后续单流 block 将两类 token 拼成序列处理。

[![HiDream-I1 的双流、单流和 MoE 结构](/assets/img/blog/moe-visual-generation/hidream-dual-single-stream.png)](/assets/img/blog/moe-visual-generation/hidream-dual-single-stream.png)

*图 19：双流 block 中，文本使用 SwiGLU FFN，视觉使用 MoE；单流 block 在拼接序列上执行 MoE。底部展开 routed 与 shared expert 的计算。*

[官方实现][hidream-code]的双流 `HiDreamImageTransformerBlock` 中，`ff_i` 是 MoE，`ff_t` 是普通 SwiGLU。两种 token 先用各自的 Q/K/V 投影，拼接后做 attention，再回到各自的 FFN。文本可以影响视觉 hidden state，进而影响视觉路由，但文本本身在这个阶段不经过图像的 routed experts。

进入 `HiDreamImageSingleTransformerBlock` 后，文本与图像 hidden states 拼接，经过同一条 attention 和 MoE 路径；最终只取视觉位置预测图像更新量。shared expert 是对这一条 MoE 输入始终执行的 FFN，“单流共享处理”和“单个 shared expert”分别描述网络组织与专家结构。

### 5.2 HunyuanImage-3.0：一个 MoE 主干，两种预测目标

[HunyuanImage-3.0][hunyuan-report]从 Hunyuan-A13B MoE 主干扩展而来，约 80B total / 13B active，每层有 64 个 routed experts、每 token 激活 8 个，并有一个 shared MLP。它将文本 next-token prediction 与图像 diffusion prediction 放在同一个主干中。

[![HunyuanImage-3.0 的统一文本与视觉主干](/assets/img/blog/moe-visual-generation/hunyuanimage-multimodal.png)](/assets/img/blog/moe-visual-generation/hunyuanimage-multimodal.png)

*图 20：文本与连续视觉表示进入统一 MoE 主干，再分别用于语言和视觉预测。*

文本使用离散 tokenizer。生成图像的带噪 VAE 特征经 timestep-modulated projector 映射到主干宽度，timestep embedding 也进入序列；条件图像则结合 VAE 与 vision encoder 的特征，让语义信息和可生成的视觉表示一起参与上下文。

两种预测目标对应不同的 attention 可见性。文本位置只能读取前面的多模态内容；同一图像段内的视觉位置可以相互读取，并读取之前的文本或条件图像。这样，模型可以先产生文本内容，再对一整段连续图像 latent 迭代去噪。整体使用自回归序列框架，不意味着图像 patch 逐个做离散 next-token prediction。

[![HunyuanImage-3.0 的广义因果 attention mask](/assets/img/blog/moe-visual-generation/hunyuanimage-attention-mask.png)](/assets/img/blog/moe-visual-generation/hunyuanimage-attention-mask.png)

*图 21：文本保持因果可见性，图像段内部使用双向可见性。多张带噪生成图像共同训练时，还会屏蔽其作为后续上下文的连接。*

报告统计了 1,000 个 prompt 的生成路由：文本与图像激活专家的分布随深度逐渐拉开。这说明该配置中出现了模态偏好；候选专家并没有因此被定义成固定的“文本一半、图像一半”。模态专门化是训练后观察到的分布，与显式按模态限制专家资格是两种设计。

### 5.3 MoT：固定模态参数仍可保留跨模态交互

另一种基线是[Mixture-of-Transformers（MoT）][mot]：根据已知模态，为 token 选择对应的 attention projection、norm 和 FFN 参数，再把得到的 Q/K/V 放回完整序列计算全局 attention。其分支选择由模态类型决定，可以进一步与 learned FFN MoE 组合。

[![MoT 的模态专用参数与全局 attention](/assets/img/blog/moe-visual-generation/mot-modality-parameters.png)](/assets/img/blog/moe-visual-generation/mot-modality-parameters.png)

*图 22：不同模态使用各自的 Transformer 参数，但 attention 仍跨模态计算。*

因此，参数分开不等于信息隔离。HiDream 的双流和单流安排、HunyuanImage 的共享 MoE，以及 MoT 的固定模态分支，分别选择了不同的共享位置。比较它们时，问题应落实到哪些变换需要专用容量、语义交互发生在哪里，以及训练目标是否争用同一组参数。

<a id="mos-state-routing"></a>

### 5.4 MoS：去噪阶段决定读取哪些文本层

文本条件与噪声阶段还可以在特征融合处结合。[Mixture of States（MoS，CVPR 2026）][mos]保留理解塔各层的 hidden states，让生成塔在每轮去噪中重新选择需要的条件表示。这里被路由的是特征来源，而非一组 FFN 参数专家。

理解塔先对 prompt 执行一次前向，得到不同深度的文本表示。轻量 router 接收文本、当前带噪 latent 和 timestep，经两个 Transformer blocks 处理后，为每个文本 token 预测一张“理解层 × 生成层”的分数矩阵。对每个生成层，它沿理解层维度取 top-k，再加权聚合对应文本 token 的隐藏状态。

[![MoS 的带噪状态条件与逐 token 跨层路由](/assets/img/blog/moe-visual-generation/mos-dynamic-text-states.png)](/assets/img/blog/moe-visual-generation/mos-dynamic-text-states.png)

*图 23：理解塔的多层状态可重复使用，router 根据当前视觉状态和 timestep，决定各生成层接收哪些文本层的表示。*

聚合后的文本序列与该生成层的视觉 token 拼接，参与联合计算。下一个去噪步仍可复用理解塔已经算好的状态，但重新计算路由，从而改变条件信息的来源和权重。训练时还用少量随机选层鼓励探索，推理时使用确定的 top-k。

论文比较了只输入 prompt、加入 latent、再加入 timestep 的配置，完整状态方案在被测设置中最好。它提供了一个可检验的问题：生成不同噪声阶段是否需要不同层次的文本信息，以及相同时间步的不同内容是否应该共享融合方式。它改变的是每个生成层接收的条件表示；FFN 内是否使用 MoE，仍是另一项架构选择。

## 6. 视觉理解与生成应该共用哪些专家

### 6.1 自然形成的模态分工，不一定对应噪声阶段分工

[Beyond Language Modeling][beyond]在统一文本预测与视觉 flow matching 的预训练中，比较了专家粒度、稀疏度及 shared expert 设计。其被测配置中，每模态设置自己的 always-on shared expert，效果优于一个全局 shared expert；routed experts 则可以在共享池中学习分工。

[![统一多模态预训练中逐层形成的专家模态偏好](/assets/img/blog/moe-visual-generation/beyond-modality-specialization.svg)](/assets/img/blog/moe-visual-generation/beyond-modality-specialization.svg)

*图 24：被测模型中，浅层以文本偏好的专家为主，深层视觉和多模态专家的比例增加。*

论文分析了一个 13.5B total、1.5B active、256 experts 的模型，观察到明显的文本/视觉专家分工；同一模型的视觉理解与生成却使用高度重叠的专家，各层选择率相关系数达到 $r\geq0.90$。视觉专家跨十个 timestep 区间的选择率变化也较小，变异系数集中在约 0.15。

[![同一模型中视觉专家跨噪声阶段的选择率变化](/assets/img/blog/moe-visual-generation/beyond-timestep-specialization.svg)](/assets/img/blog/moe-visual-generation/beyond-timestep-specialization.svg)

*图 25：跨噪声阶段的选择率变异较小。这里统计的是专家选择频率，不表示逐 token 的专家 ID 完全不变。*

这个结果与显式阶段专家、PixelMoE 中的阶段偏好形成对照：模态分工、理解/生成分工和噪声阶段分工，需要各自分析，不能由其中一种直接推出另外两种。

### 6.2 Symbiotic-MoE：为冲突目标分组，再保留共享桥梁

若从已经训练好的 VLM 出发，加入图像生成目标，还会遇到已有理解能力被新梯度干扰的问题。[Symbiotic-MoE][symbiotic]先用文本与视觉理解数据统计原有专家的激活频率，再将每层 128 个专家中的 96 个用于理解，32 个用于生成。文本和 ViT 特征进入理解组，VAE 生成特征进入生成组，各组使用独立 router，并保留处理所有 token 的 shared expert。

[![Symbiotic-MoE 的理解组、生成组与共享专家桥梁](/assets/img/blog/moe-visual-generation/symbiotic-understanding-generation.png)](/assets/img/blog/moe-visual-generation/symbiotic-understanding-generation.png)

*图 26：专用专家分担理解与生成的不同需求，共享专家接受两类数据的训练，保留共用的语义变换。*

它还采用渐进训练、差异化学习率和早期梯度屏蔽，先保护预训练路径，再逐步引入生成学习。

两项工作的训练起点与组织方式不同。前者说明共享池可以自然学到兼容理解和生成的视觉专家，后者针对给已有 VLM 添加生成能力时的冲突，显式限制更新和专家资格。是否分组，应由共同训练或扩展训练中的干扰证据决定。

<a id="5-控制接口约束下的专家候选与混合权重"></a>

## 7. 控制条件怎样限制专家候选

前面的模态与任务约束，还可以用于视频世界模型的控制接口。相机轨迹、机器人动作与稠密动作图有不同的输入形式，模型需要共享视觉动力学，同时接收各自的控制信号。

[Worldscape-MoE][worldscape]基于 Wan2.2 的 5B video world model，把 camera trajectory、dual-arm robot action、dense action map、LIBERO single-arm action 放进一个统一 world-dynamics backbone。MoE 结构是 shared expert 加 modality-specific experts：每个样本只允许 shared expert 和由 control mask 选中的对应专家参与，router logits 从 sequence-mean pooled hidden states 计算。

[![Worldscape-MoE architecture](/assets/img/blog/moe-we-like-you/10-worldscape-moe-architecture.png)](/assets/img/blog/moe-we-like-you/10-worldscape-moe-architecture.png)

*图 27：Worldscape-MoE 用 shared expert 和 control-specific experts 吸收多种控制接口。*

在这里，已知的 control mask 先限制候选分支，router 再决定混合权重。它与完全开放的 patch-token top-k 不同，也与按 timestep 切换整套模型不同。专家分工的接口含义来自控制信号约束。

这里的 modality-specific 指控制类型，选择粒度是整个样本；文本/视觉 token 路由则可以逐位置变化。新增接口时，可以增加 control-specific expert，训练新分支并复用已有的 shared dynamics。

<a id="7-专家选择粒度对应的质量与性能指标"></a>

图像和视频 MoE 的设计由此包含几种独立选择：按噪声阶段选择参数与计算预算，按视觉位置分配专家与调用数，用语义或任务约束组织分工，再选择文本、视觉和控制路径之间共享哪些参数。选出的专家最终还要落到矩阵计算与数据传输上；完整采样时间包含这些开销，也包含 attention、VAE、CFG 和可能的 refiner。较小的 active 参数量能否转成更快的生成，取决于被减少的工作是否位于实际耗时的环节。

## 系列导航

上一篇：[第九篇：DeepEP v1/v2、MoonEP 与 EP 通信优化]({% post_url 2026-07-23-moe-ep-communication-optimization %})

<a id="延伸阅读"></a>

## 参考资料

- [Scalable Diffusion Models with Transformers][dit-paper]；[原始 DiT 实现][dit-code]
- [eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers][ediffi]
- [ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts][ernie-vilg]
- [Wan2.2 GitHub Repository][wan22]
- [Scaling Diffusion Transformers to 16 Billion Parameters][dit-moe]
- [HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer][hidream-paper]；[分支实现][hidream-code]
- [HunyuanImage 3.0 Technical Report][hunyuan-report]；[代码][hunyuan-image]
- [Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence][lingbot-video-paper]
- [MOVA: Towards Scalable and Synchronized Video-Audio Generation][mova-paper]
- [MAGI-2 Preview: Scaling Video Generation Models Efficiently][magi2]
- [Multi-Head LatentMoE and Head Parallel][latentmoe-hp]
- [Worldscape-MoE: A Unified Mixture-of-Experts World Model for Scalable Heterogeneous Action Control][worldscape-paper]
- [EC-DiT: Scaling Diffusion Transformers with Adaptive Expert-Choice Routing][ecdit]
- [DiffMoE: Dynamic Token Selection for Scalable Diffusion Transformers][diffmoe]
- [Breaking the Uniformity Trap: Scaling Video Diffusion Model via SplitMoE][splitmoe]
- [PixelMoE: Scaling Pixel-Space Diffusion with LLM-Style Mixture-of-Experts][pixelmoe]
- [Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models][mot]
- [Beyond Language Modeling: An Exploration of Multimodal Pretraining][beyond]
- [Symbiotic-MoE: Unlocking the Synergy between Generation and Understanding][symbiotic]

- [RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths][raphael]
- [Denoising Task Routing for Diffusion Models][dtr]
- [Switch Diffusion Transformer: Synergizing Denoising Tasks with Sparse Mixture-of-Experts][switch-dit]
- [TimeStep Master: Asymmetrical Mixture of Timestep LoRA Experts for Versatile and Efficient Diffusion Models in Vision][tsm]
- [Mixture of Efficient Diffusion Experts Through Automatic Interval and Sub-Network Selection][diffpruning]
- [ALTER: All-in-One Layer Pruning and Temporal Expert Routing for Efficient Diffusion Generation][alter]
- [Accelerating Video Diffusion via Training-Free Trajectory Routing][track]
- [Diff-MoE: Diffusion Transformer with Time-Aware and Space-Adaptive Experts][diff-moe-time]
- [Mixture of States: Routing Token-Level Dynamics for Multimodal Generation][mos]

[ediffi]: https://arxiv.org/abs/2211.01324
[dit-paper]: https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html
[dit-code]: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
[ernie-vilg]: https://arxiv.org/abs/2210.15257
[wan22]: https://github.com/Wan-Video/Wan2.2
[dit-moe]: https://arxiv.org/abs/2407.11633
[hidream]: https://github.com/HiDream-ai/HiDream-I1
[hidream-paper]: https://arxiv.org/abs/2505.22705
[hunyuan-image]: https://github.com/Tencent-Hunyuan/HunyuanImage-3.0
[lingbot-video]: https://github.com/Robbyant/lingbot-video
[lingbot-video-paper]: https://arxiv.org/abs/2607.07675
[mova]: https://github.com/OpenMOSS/MOVA
[mova-paper]: https://arxiv.org/abs/2602.08794
[magi2]: https://sand.ai/blog/magi-2-preview
[magi2-github]: https://github.com/SandAI-org/MAGI-2-preview
[latentmoe-hp]: https://arxiv.org/abs/2602.04870
[worldscape]: https://github.com/EmbodiedCity/Worldscape-MoE.code
[worldscape-paper]: https://arxiv.org/abs/2607.03964
[worldscape-arch]: https://github.com/EmbodiedCity/Worldscape-MoE.code/blob/main/docs/ARCHITECTURE.md
[lingbot-architecture]: https://arxiv.org/html/2607.07675v1#S2.F2

[ecdit]: https://arxiv.org/html/2410.02098v2
[diffmoe]: https://arxiv.org/html/2503.14487v1
[splitmoe]: https://arxiv.org/html/2609.38140v1
[pixelmoe]: https://arxiv.org/html/2610.18857v3
[hidream-code]: https://github.com/HiDream-ai/HiDream-I1/blob/5f92bab45f1dfb1e794ee357286a5b837eaf4400/hi_diffusers/models/transformers/transformer_hidream_image.py#L101
[hunyuan-report]: https://arxiv.org/html/2509.23951v2
[mot]: https://arxiv.org/html/2411.04996v2
[beyond]: https://arxiv.org/html/2603.03276v1
[symbiotic]: https://arxiv.org/html/2604.07753v2

[raphael]: https://arxiv.org/html/2305.18295v2
[dtr]: https://arxiv.org/html/2310.07138v3
[switch-dit]: https://arxiv.org/html/2403.09176v2
[tsm]: https://arxiv.org/html/2503.07416v1
[diffpruning]: https://arxiv.org/html/2409.15557v1
[alter]: https://arxiv.org/html/2505.21817v1
[track]: https://arxiv.org/html/2609.30096v1
[diff-moe-time]: https://proceedings.mlr.press/v267/cheng25d.html
[mos]: https://arxiv.org/html/2511.12207v2
