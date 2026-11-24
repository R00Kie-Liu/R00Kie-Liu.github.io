---
layout: post
title: "连续 dLLM 导读：从离散扩散到连续扩散与流"
date: 2026-11-19
last_modified_at: 2026-11-24
visibility: unlisted
sitemap: false
series: continuous-dllm
series_order: 0
tags: [dLLM, Diffusion, Flow Matching]
math: true
description: "面向离散 dLLM 研究者的连续扩散与流语言模型系列，包含 8 篇主线、2 篇选读、方法流程、完整推导、参考代码和实验表格。"
permalink: /2026/11/19/continuous-dllm-index.html
---

连续扩散语言模型需要回答的，不只是“怎样给 embedding 加噪声”，而是整段文本怎样通过连续状态共同形成。做离散 dLLM 时熟悉的逐位置 CE、并行预测和提交次序，到了这里仍然有用；变化发生在概率路径、网络预测与实际采样之间的连接。

这套 8 篇主线、2 篇选读面向已经研究离散 dLLM、但没有系统学过连续扩散的读者。正文从条件期望、Gaussian、ODE／SDE 讲起，再逐项拆开代表方法的状态、训练、梯度、采样和读出。需要比较的方法给出完整流程；容易混淆的公式配有手算例子、反例或参考代码。

可以把每篇当作一章教材阅读，也可以按下面的主题定位。基础概念和方法机制在系列内展开；论文链接用于核对版本、复现实现和继续追踪严格证明。

## 系列目录

### 主线 8 篇

- [第 1 篇 从离散状态走向连续状态]({% post_url 2026-11-19-continuous-dllm-01-continuous-state %})：沿着 AA／BB 的四次连续更新，理解模型保存什么、怎样用 token 概率移动状态，以及何时读出离散结果。

- [第 2 篇 从高斯去噪理解 score]({% post_url 2026-11-19-continuous-dllm-02-gaussian-score %})：推导条件均值与 score，走完 DDPM 的训练和采样；解释 Brownian 增量、Fokker–Planck、reverse SDE 与 probability-flow ODE，明确反向时间的符号。

- [第 3 篇 从交叉熵走到 Flow Matching]({% post_url 2026-11-19-continuous-dllm-03-flow-matching %})：用概率守恒推导联合速度，连接类别后验与连续更新；给出 one-hot 张量操作、可运行的 AA／BB 解析采样，以及 CE 误差到速度误差的界。

- [第 4 篇 文本应该放在哪个连续空间]({% post_url 2026-11-19-continuous-dllm-04-representations %})：从可辨认性与几何退化出发，逐条追踪 Diffusion-LM、CDCD、SED、ELF、LD4LG、LDLM、Cola DLM、CoBit 的表示来源、训练模块、梯度与解码成本。

- [第 5 篇 连续模型怎样训练好]({% post_url 2026-11-19-continuous-dllm-05-training %})：比较 CE、概率 MSE、embedding MSE；构造三种时间日程；拆解 AdaLN、条件注意力、tokenwise time、self-conditioning 和 NELBO，并给出含 clean prefix 的训练更新。

- [第 6 篇 连续模型怎样采样和控制]({% post_url 2026-11-19-continuous-dllm-06-sampling-guidance %})：从 Euler／Heun 走到连续外推、logit guidance、奖励梯度与自动机规则控制；解释异步提交、infilling、EOS、MBR、KV cache 和实际 NFE。

- [第 7 篇 从多步流走向一步生成]({% post_url 2026-11-19-continuous-dllm-07-flow-maps %})：推导双时间概率与复合软目标，按 FMLM v3 的 CE 默认方案讲训练；比较 CFM、DFM、pushforward generator、DBTM、AR coupling、固定点闭合和异步映射，并列出一步实验。

- [第 8 篇 从论文结果走向研究判断]({% post_url 2026-11-19-continuous-dllm-08-research-evaluation %})：用 RePlaid 的 scaling、PPL 表格与消融划分证据；比较从头训练与 AR 迁移；推导 MAUVE、energy distance、GM、SOL 的含义，再连接条件文本、谜题、生物序列与后训练。

### 选读 2 篇

- [选读 A 连续与离散的混合模型]({% post_url 2026-11-19-continuous-dllm-09-hybrid-models %})：从共享 latent 的相关性推到 CANDI、CADD、CCDD 的具体 kernel、训练目标和联合采样，解释何时连续辅助状态仍会在独立读出时丢掉联合依赖。

- [选读 B Simplex 与球面上的流]({% post_url 2026-11-19-continuous-dllm-10-simplex-sphere %})：区分 Gaussian logits、BFN belief 与 simplex 状态；算出 Dirichlet／Beta 速度、Fisher–Rao 距离与边界长度，再用 S-FLM 说明 CE 怎样提供球面速度、数值步怎样保持约束。

## 怎样安排阅读顺序

第一遍按 1→2→3 建立基础：要能说明“网络输出的是哪种条件统计量，为什么它能决定整个句子的人群分布”。第 2、3 篇的数值例子可以边读边算，第 3 篇的小程序不需要训练神经网络。

接着按 4→5→6 追踪一份真实模型：表示从哪里来，哪些模块训练，怎样抽时间，怎样输入条件，以及一次更新到底调用什么。第 4 篇的模型流程和第 5 篇的张量代码可对照阅读。

第 7 篇需要前面三条线一起使用：后验统计量、连续动力学和有限步数值误差。第 8 篇再回答这种设计究竟改善了分布、任务成功率还是计算效率。想尽早联系离散 dLLM，可在第 3 篇后读选读 A；几何选读 B 建议在第 4 篇后阅读。

## 从 survey 的主题定位到本系列

这里按原 survey 的章节提供入口，避免为了找一个术语重新通读它。

- **§2：factorization、连续状态与数学准备。** 第 1 篇解释单次转移的条件独立近似；第 2 篇给出 Gaussian 去噪、score、SDE／ODE；第 3 篇建立 FM 和联合生成的连接。
- **§3.1：七类表示与表示质量。** 第 4 篇比较固定、可学习、上下文、压缩 latent 和二进制表示；选读 B 处理 logits、simplex、球面，选读 A 处理混合状态。
- **§3.2–3.4：目标、日程与 self-conditioning。** 第 5 篇逐项推导，包含 token 可辨认率、条件熵曲线、估计方差三种日程，以及 likelihood bound 的各项职责。
- **§3.5–3.6：主干与次序。** 第 5 篇解释时间调制、注意力和 tokenwise 训练；第 6 篇解释推进、提交、修订、因果前缀与缓存。
- **§4：采样与控制。** 第 6 篇集中处理数值解法、随机更新、guidance、自动机约束、条件生成、最终读出、停止和完整计算成本。
- **§5：少步接口、目标、coupling 与映射采样。** 第 7 篇分别拆开普通 flow、flow map、双时间去噪器、直接生成器和自主终点映射；给出 CE 蒸馏、分布训练、AR 配对、固定点闭合及 γ-sampling 的对应位置。
- **§6：扩展训练。** 第 8 篇区分 compute-optimal、固定参数过度训练和预训练 AR 迁移，列出 RePlaid、Scaling CFM、Cola、PLM、Sigma、Lafa 的比较条件。
- **§7：指标与应用。** 第 8 篇区分测试 likelihood、样本 gPPL、词频熵和分布覆盖，解释分布指标，再按文本、程序／推理、谜题和生物序列整理实验证据。
- **§8：后续问题。** 第 4、5 篇分析表示与训练，第 8 篇进一步解释 post-training 的概率比难题、多模态时间分配和中间态监控。

如果问题是“某个方法到底改了哪里”，优先去对应的方法流程；如果是“一个公式为什么成立”，先看第 2、3、5 篇及选读 B；如果是“结果是否值得跟进”，去第 7、8 篇的实验和证据部分。

## 统一记号

主线让时间 t 从 0 走到 1：x₀ 是噪声，x₁ 是干净表示，xₜ 是中间状态。C 表示干净表示，ε 表示标准 Gaussian 噪声；L 是序列长度，V 是词表大小，d 是单个位置的表示维度。Dθ 预测干净表示，bθ 预测速度，sθ 预测 score。

qθ 通常是网络的 token 后验预测；它与连续状态 x 不同。选读 B 中的 p 则是 simplex 上的当前状态，它虽然也是一份概率向量，也不自动等于模型对 clean token 的后验 q。下标或局部记号会在使用处定义。

涉及 DDPM、CANDI 或其他沿“数据→噪声”定义的原始方法时，正文单独说明其时间方向。换记号前，应同时核对 α、σ、SNR／NSR 和时间导数的符号。

第一遍需要向量运算、导数、概率和交叉熵。条件期望、连续性方程、Brownian 增量、Fisher–Rao 度量等概念，会随相应问题解释。

## 配图、算例与代码怎样配合

概念图主要区分公式中的对象与运算次序：联合与独立采样、后验均值与样本、训练插值与生成轨迹、先奖励再平均与先平均再奖励，以及连续状态与 self-conditioning 草稿。第 3 篇的二维流图直接按解析速度计算。

数值例子承担另一项职责：你可以复核符号与数量级，例如 reverse SDE 的时间方向、短跳软目标的权重、CANDI 的揭示概率，以及 Dirichlet 和球面的一次数值更新。正文标为教学的分布、伪代码和参考更新，与论文报告的实验表格分别注明。

代码中的 network、sample_time 等接口需要按前文实现；它们用于展示张量、梯度和更新规则。第 3 篇的 NumPy 解析例子则可以直接运行，不依赖模型 checkpoint。

## 出处与版本

系列以 [Continuous Diffusion and Flow Language Models Survey](https://continuous-lm.github.io/) 的结构和文献图谱为起点，按理解依赖重新组织，并对照代表论文展开机制、训练步骤与实验条件。

Survey 的文献截止日期为 2026 年 11 月 2 日。本次核对还使用了之后发布的 [LangFlow v4（11 月 5 日）](https://arxiv.org/abs/2604.11748v4)、[ELF v3（11 月 5 日）](https://arxiv.org/abs/2605.10938v3)、[LDLM v2（11 月 8 日）](https://arxiv.org/abs/2605.07933v2) 与 [Sigma v2（11 月 12 日）](https://arxiv.org/abs/2610.02665v2)。

少步篇按 [FMLM v3](https://arxiv.org/abs/2602.16813v3) 的 CE 复合蒸馏讲默认实现；早期两阶段 MSE 方案放在版本对照中。评估篇的主要数表按 [RePlaid v3](https://arxiv.org/abs/2605.18530v3) 核对。理解算法时先沿本系列的状态、目标、更新和读出阅读；需要精确复现数据管线、超参数或严格证明时，再使用相应版本的原文与代码。
