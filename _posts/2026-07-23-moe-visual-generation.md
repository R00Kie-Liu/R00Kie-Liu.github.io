---
layout: post
title: "MoE我们喜欢你（十）：视觉生成中的专家分工"
series: moe-we-like-you
series_order: 10
last_modified_at: 2026-10-26
date: 2026-07-23
tags: [MoE, Diffusion, Video, WorldModel]
description: "从 Wan2.2、DiT-MoE、HiDream-I1、LingBot-Video、MOVA、MAGI-2 和 Worldscape-MoE 出发，讨论图像、视频和世界模型里的 MoE 路由轴。"
---


## 0. 先分清专家在什么层级被选择

视觉生成中的 expert 可以是一整个 denoiser，也可以是 DiT block 内的 MLP、hidden 子空间中的小专家，或接入某种控制信号的分支。它们选择专家的单位、训练方式和系统成本并不相同。

下面先按这些机制展开，再看图像、视频和音视频模型如何使用它们。任务类别用于说明应用，不能直接推出专家究竟按什么分工。

## 1. 反复采样与空间时间结构带来什么约束

视频生成通常在大量空间时间 token 上反复执行 denoiser。因此，专家选择既可能随 token 改变，也可能随 denoising step 改变。应区分显式规则（例如按噪声阈值换模型）与观察到的专门化（例如 learned router 偏好某些位置或 timestep）；后者不等于显式为某种内容设置了专家。

系统评估也要覆盖整个 sampling pipeline。除 MoE FFN 外，attention、VAE、refiner、offload 和多次 denoiser forward 都可能占据主要时间。音视频同步或物理合理性则是任务目标，需要对应的数据和评价，不能仅由采用 MoE 推导出来。

## 2. 模型级选择：按 timestep 换 denoiser

更早的一条线是 denoising expert，它出现在 DiT-MoE 之前。这里要先区分实现层级：denoising expert 通常直接用 timestep 或 noise level 选择整套 denoiser，LLM MoE 那种 token router 不参与。

普通扩散模型可以写成一个 denoiser：

$$
\hat{\epsilon}=D_{\theta}(x_t,t,c)
$$

所有 timestep 共用同一套参数 $\theta$。timestep expert 做的是先把 $t$ 或 noise level $\sigma$ 切成几个连续区间 $S_1,\cdots,S_m$，再定义一个固定映射：

$$
g(t)=i,\quad t\in S_i,\qquad \hat{\epsilon}=D_{\theta_i}(x_t,t,c)
$$

训练时采样到哪个 $t$，就更新对应的 $D_{\theta_{g(t)}}$；推理时 scheduler 走到哪个 $t$，也只调用对应的 expert。这里没有 token-level top-k，也没有 per-patch all-to-all。每个 denoising step 的计算仍然近似是一次 denoiser forward，只是总 checkpoint/参数量变成多份；显存压力取决于这些 expert 是否同时常驻 GPU，还是在 high/low 阶段之间 offload。

[eDiff-I][ediffi] 的做法比较典型。它先训练一个覆盖全噪声分布 $p(\sigma)$ 的基线 denoiser，再用这个基线初始化专家，而不是从零训练多套模型。第一层分裂时，把 $p(\sigma)$ 按面积分成 low-noise 和 high-noise 两个区间；继续分裂时，主要保留低噪声、高噪声和中间区间这几类有用 expert。高噪声 expert 更负责从接近纯噪声的输入里定布局、吸收文本条件；低噪声 expert 更负责最后几步的锐化和细节恢复。推理时沿着采样轨迹切换 denoiser，所以容量变大，但每一步仍然只跑一个 expert。

[ERNIE-ViLG 2.0][ernie-vilg] 更直接：DDPM 有 1000 个训练 timestep，它把这些 timestep 均匀切成 $n$ 个连续 block，每个 block 对应一个 U-Net expert。官方模型里 $n=10$，text encoder 共享，U-Net expert 分开。训练也分两阶段：先训练一个共享 U-Net，再复制成 10 个 denoising experts；第二阶段采样到 timestep $t$ 时，只优化 $\lfloor t/100 \rfloor$ 对应的 expert。这个版本的 router 几乎就是一行代码：`expert_id = t // 100`。

这条线在 [Wan2.2][wan22] 上变得更工程化。Wan2.2 明确把 MoE 引入 video diffusion model，通过跨 timestep 分离 denoising process，让不同专家模型分别处理不同噪声阶段。A14B 系列仍然保持每步计算成本相近，但总容量变大。这个设计比 token-level top-k 更容易解释：扩散模型本来就不是一步生成，timestep 是天然的 routing coordinate。

![Wan2.2 timestep MoE](/assets/img/blog/moe-we-like-you/10-wan2-2-timestep-moe.png)


*图 1：Wan2.2 沿 denoising timestep 划分 high-noise / low-noise experts。图源：[Wan2.2][wan22] README。*

代码层面，Wan2.2 的 T2V/I2V A14B pipeline 会同时加载 `low_noise_model` 和 `high_noise_model` 两个 DiT checkpoint。采样前先算：

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

之后 conditional / unconditional 两次 forward 都走同一个被选中的 model，再做 classifier-free guidance。也就是说，Wan2.2 的 MoE 是“采样轨迹上的模型切换”：高噪声阶段调用 high-noise expert，低噪声阶段调用 low-noise expert；两者各占多少采样步，由阈值和 scheduler 决定。它不是在一个 DiT block 里放很多 FFN experts，也不会让同一帧里的不同 patch 选择不同专家。

这种专家选择不需要按 patch 做 top-k dispatch；denoiser 本身仍可使用其它分布式并行。它的分工粒度只有 timestep：同一个 step 内，不同空间区域、不同运动状态是否需要不同 expert，timestep expert 本身回答不了。这个问题要交给下一节的 token-level DiT-MoE。

## 3. Token 级选择：在 DiT block 内替换 MLP

如果把 MoE 放进 Diffusion Transformer block，问题就接近 LLM MoE 了：每个 image/video token 可以选择 routed experts。但视觉 token 有位置，denoising 有时间，expert specialization 会更复杂。

[DiT-MoE][dit-moe] 是这条线的重要坐标。它做的是 sparse DiT，提出 shared expert routing 和 expert-level balance loss，并在 ImageNet 条件生成上扩到 16B 级别。论文里最值得记的是路由分析：expert selection 对 spatial position 和 denoising timestep 有明显偏好，对 class condition 反而没那么敏感；越到深层，专家选择从位置特异逐渐变得更分散、更均衡。

![DiT-MoE framework](/assets/img/blog/moe-we-like-you/10-dit-moe-framework.png)


*图 2：DiT-MoE 把 sparse MoE 放进 DiT block，并保留 shared expert 路径。图源：[DiT-MoE][dit-moe]。*

与整套 denoiser 切换相比，DiT-MoE 在 block 内选择分支。它在部分 DiT block 内把 MLP 换成 MoE layer；采样轨迹仍由同一个 denoiser 走，变化发生在 token 的 MLP 分支上。对每个 image patch token $x_j$，router 先在 $n$ 个 routed experts 上算概率，选 top-$K$ 个 MLP；同时还有 $n_s$ 个 shared experts 是确定执行的。可以粗略写成：

$$
y_j=\sum_{i\in \mathrm{TopK}(r(x_j))}p_{j,i}E_i(x_j)+\sum_{s=1}^{n_s}S_s(x_j)
$$

这里 $E_i$ 是 routed MLP expert，$S_s$ 是 shared MLP expert。shared expert 是确定执行的公共路径，routed experts 则由 learned router 选择；具体专门化需要通过路由分布和消融确认。因为 router 的输入是 patch token hidden state，同一个 denoising step 里，不同空间位置也可以走不同 expert；同一个 patch 在不同 timestep 也可能切到不同 expert。这才是更接近 LLM MoE 的 token-level routing。

在这个被测模型中，路由分析应同时控制 timestep 与空间位置，再检查类别和内容的影响。位置偏好并不自动等于“纹理专家”或“运动专家”，后者需要额外的行为分析。

## 4. 子空间选择：MAGI-2 的 head-local experts

MAGI-2 和前面几类不太一样。它把 MoE 放进更细的 head-local latent space；timestep expert 和普通 token-level DiT-MoE 只是参照坐标。

[MAGI-2 Preview][magi2] 公开材料里，模型约 114B total，每 token 激活约 6B 参数；主体 36 层使用 Multi-Head MoE。它把 3072 hidden 先映射成 12 个 256 维 head，每个 head 有独立 router 和 256 个专家，每个 head top-6。也就是说，一个 token 会在 12 个低维子空间里各自选择专家，最后再 merge 回 hidden state，而不是只触发几个完整宽度 FFN。

![MAGI-2 Multi-Head LatentMoE](/assets/img/blog/moe-we-like-you/10-magi-2-multi-head-latentmoe.png)


*图 3：MAGI-2 将完整 hidden 拆成多个 latent heads，各 head 独立路由到 head-local experts。图源：[MAGI-2 Preview][magi2]。*

这条路来自 [Multi-Head LatentMoE and Head Parallel][latentmoe-hp] 的想法：把完整 hidden 拆成多个 latent heads，让不同子空间独立路由。通信也随之变化。普通 EP 的通信量随 top-k 增长，而且每个 batch 的 send/recv shape 由 routing 结果决定；Head Parallel 先按 head 做固定形状分发，再在 head owner 内部完成 routing 和 expert compute。跨设备通信边界更规则，动态 routing 的不规则性被留在本地。

这对视频生成很关键。长视频 token 多，top-k 大一点就会放大通信和 buffer 抖动。MAGI-2 选择把 expert 粒度切到低维 head subspace，再通过系统 co-design 把小 expert 执行、路由排序、通信和优化器一起处理。这里要同时评价专家计算与通信形状，不能只比较 active 参数量。

MAGI-2 也提醒我们，fine-grained expert 在视觉生成里会遇到比 LLM 更强的 kernel 压力。专家变窄后，单个 GEMM 可能更小；如果没有 fused routing、compact token layout、grouped expert compute 和稳定通信边界，稀疏化收益很容易被执行开销吃掉。

## 5. 条件分支：由控制接口限制专家候选

世界模型里的 MoE 又多了一层语义：expert 不一定对应视觉区域或噪声阶段，也可能对应 control modality。

[Worldscape-MoE][worldscape] 是一个干净的例子。它基于 Wan2.2 的 5B video world model，把 camera trajectory、dual-arm robot action、dense action map、LIBERO single-arm action 放进一个统一 world-dynamics backbone。MoE 结构是 shared expert 加 modality-specific experts：每个样本只允许 shared expert 和由 control mask 选中的对应专家参与，router logits 从 sequence-mean pooled hidden states 计算。

![Worldscape-MoE architecture](/assets/img/blog/moe-we-like-you/10-worldscape-moe-architecture.png)


*图 4：Worldscape-MoE 用 shared expert 和 control-specific experts 吸收多种控制接口。图源：[Worldscape-MoE][worldscape-paper] Fig. 2。*

在这里，已知的 control mask 先限制候选分支，router 再决定混合权重。它与完全开放的 patch-token top-k 不同，也与按 timestep 切换整套模型不同。专家分工的接口含义来自控制信号约束。

新增控制接口时，可以增加 modality-specific expert 并复用 shared dynamics。这个扩展需要新的控制数据和分支训练；新增接口后的共享与独立能力都要验证。

## 6. 应用案例：图像、视频与同步音视频

前面的机制可以出现在不同任务中。下面几组开源样本说明模型规模、数据和运行方式；只有公布了 router 与 expert 的实现，才进一步判断具体分工。

[HiDream-I1][hidream] 是 17B 参数的开源图像生成模型，采用 sparse DiT 与动态 MoE，提供了运行这类图像模型的开放样本。

[HunyuanImage-3.0][hunyuan-image] 则采用另一种生成框架。它是 80B total / 13B active 的开源 image generation MoE，64 experts，并提供 FlashInfer MoE 实现选项。不过它官方强调统一多模态/自回归框架，和典型 diffusion DiT MoE 有明显差别。因此它应作为自回归图像 MoE 的对照，不能用来佐证 diffusion timestep routing。

[LingBot-Video][lingbot-video] 提供面向具身场景的视频生成样本。它把 video pretraining 明确拉向 embodied intelligence：数据里加入 70,000+ 小时 embodied data，评价里加入 physical rationality 和 task completion。模型侧提供 Dense 1.3B 和 MoE 30B-A3B 版本，MoE 版还配 refiner，任务覆盖 T2I、T2V、TI2V。

![LingBot-Video architecture](/assets/img/blog/moe-we-like-you/10-lingbot-video-architecture.png)


*图 5：LingBot-Video 的 single-stream diffusion transformer。MoE branch 由 shared experts 和 top-k routed experts 组成。图源：[LingBot-Video][lingbot-video-paper] Fig. 2。*

这里需要把架构与任务证据分开：MoE 提供容量与条件计算，embodied data 和 physical rationality / task completion 评价定义了要学习什么。要判断某类专家是否专门承担物理能力，仍需路由分析或消融。

LingBot-Video 的工程侧也值得注意。它公开了 diffusers 路径和 SGLang Diffusion 路径，还提到 MoE FP8 expert、grouped expert execution、FSDP/CP8/refiner 的组合。部署时需要确认这些组件能否在目标 runtime 中组合运行，并单独计入 refiner 和专家执行成本。

[MOVA][mova] 则把问题换成 audio-video synchronization。它是 32B total / 18B active 的开源 video-audio generation model，目标是单次推理同时生成视频和音频，避免先生成视频再补音轨带来的误差累积。音画同步来自联合生成架构与训练目标；仅凭采用 MoE，还不能判断它是否显式按音频/视频模态路由。

![MOVA dual-tower architecture](/assets/img/blog/moe-we-like-you/10-mova-dual-tower-architecture.png)


*图 6：MOVA 用 video DiT tower、audio DiT tower 和 bidirectional Bridge 做同步音视频生成。图源：[MOVA][mova-paper] Fig. 2。*

MOVA 的 active ratio 较高。比较它与其它模型时，需要同时考虑音视频输出、质量目标和完整采样成本，不能只按稀疏比例排序。

## 7. 按机制选择评估方式

| 机制 | 选择单位 | 需要验证 |
|---|---|---|
| timestep 切换 | 整套 denoiser | 区间边界、各 expert 覆盖步数、切换与 offload 成本 |
| token-level MoE | block 内的 patch/token MLP | 相同 timestep 下的位置与内容偏好、质量、EP 与 GEMM 时间 |
| head-local MoE | hidden 子空间内的 experts | 每 head 负载、通信形状、小矩阵与合并成本 |
| control-aware 分支 | 由控制接口限制的候选专家 | 新旧控制能力、共享分支贡献、control mask 与混合权重 |

所有机制都应报告完整 sampling latency，并分解 denoiser、attention、MoE FFN、VAE、refiner 和 offload。视频/世界模型还需对应的时间一致性、物理合理性或任务完成指标。

先说明专家在什么层级被选择，再用实现与实验解释它承担了什么功能，才能把模型规模、任务目标与专家分工联系起来。

## 系列导航

上一篇：[第九篇：DeepEP v1/v2、MoonEP 与 EP 通信优化]({% post_url 2026-07-23-moe-ep-communication-optimization %})

## 延伸阅读

想看开源视频 MoE 的运行形态，可以先读 [Wan2.2 README][wan22]、[LingBot-Video README][lingbot-video] 和 [MAGI-2 README][magi2-github]。前者适合理解 timestep MoE，后两者更适合看大视频 MoE 在 inference runtime 里怎么落地。

如果更关心 world model，可以读 [Worldscape-MoE 架构说明][worldscape-arch]。它把 shared expert、control-specific experts、control mask 和 checkpoint migration 写得很直接。

## 参考资料

- [eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers][ediffi]
- [ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts][ernie-vilg]
- [RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths][raphael]
- [Wan2.2 GitHub Repository][wan22]
- [Scaling Diffusion Transformers to 16 Billion Parameters][dit-moe]
- [HiDream-I1: A High-Efficient Image Generative Foundation Model with Sparse Diffusion Transformer][hidream-paper]
- [HunyuanImage-3.0 GitHub Repository][hunyuan-image]
- [Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence][lingbot-video-paper]
- [MOVA: Towards Scalable and Synchronized Video-Audio Generation][mova-paper]
- [MAGI-2 Preview: Scaling Video Generation Models Efficiently][magi2]
- [Multi-Head LatentMoE and Head Parallel][latentmoe-hp]
- [Worldscape-MoE: A Unified Mixture-of-Experts World Model for Scalable Heterogeneous Action Control][worldscape-paper]

[ediffi]: https://arxiv.org/abs/2211.01324
[ernie-vilg]: https://arxiv.org/abs/2210.15257
[raphael]: https://arxiv.org/abs/2305.18295
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
