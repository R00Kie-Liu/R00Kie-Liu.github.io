---
layout: post
title: "连续 dLLM 导读：从离散扩散到连续扩散与流"
date: 2026-11-19
last_modified_at: 2026-11-19
visibility: unlisted
sitemap: false
series: continuous-dllm
series_order: 0
tags: [dLLM, Diffusion, Flow Matching]
math: true
description: "面向离散 dLLM 研究者的连续扩散与流语言模型系列，包含 8 篇主线、2 篇选读、完整推导、数值例子和示意图。"
permalink: /2026/11/19/continuous-dllm-index.html
---

连续扩散语言模型需要理解的第一件事，是整段文本的连续状态怎样共同变化。离散 dLLM 中熟悉的逐位置分类、交叉熵和并行更新仍然有用；高斯噪声、score、速度场与数值积分，则提供了另一套描述生成过程的工具。

这套系列面向已经做过离散 dLLM、但还没有系统学习连续扩散的读者。8 篇主线从条件概率与去噪逐步走到训练、采样和一步生成；2 篇选读连接混合模型与几何方法。核心概念、推导中间步骤、数值例子和练习答案均在正文展开。

## 系列目录

### 主线 8 篇

- [第 1 篇 从离散状态走向连续状态]({% post_url 2026-11-19-continuous-dllm-01-continuous-state %})：从离散 dLLM 的逐位置预测出发，用联合分布反例解释连续状态与耦合动力学解决的问题。

- [第 2 篇 从高斯去噪理解 score]({% post_url 2026-11-19-continuous-dllm-02-gaussian-score %})：从高斯观测与条件均值推导 score，连接去噪目标、DDPM、ODE 与 SDE，并给出手算例子。

- [第 3 篇 从交叉熵走到 Flow Matching]({% post_url 2026-11-19-continuous-dllm-03-flow-matching %})：推导条件平均速度与连续性方程，解释逐位置交叉熵如何定义整段序列的联合生成动力学。

- [第 4 篇 文本应该放在哪个连续空间]({% post_url 2026-11-19-continuous-dllm-04-representations %})：比较 one-hot、embedding、binary 与 latent 表示，分析 token 可辨认性、压缩与连续状态成本。

- [第 5 篇 连续模型怎样训练好]({% post_url 2026-11-19-continuous-dllm-05-training %})：连接 CE、MSE、时间日程、信息分配、self-conditioning 与似然下界，给出训练接口和判断依据。

- [第 6 篇 连续模型怎样采样和控制]({% post_url 2026-11-19-continuous-dllm-06-sampling-guidance %})：解释 Euler 与高阶求解、网络调用成本、CFG、奖励引导和最终解码，区分目标分布与数值近似。

- [第 7 篇 从多步流走向一步生成]({% post_url 2026-11-19-continuous-dllm-07-flow-maps %})：从可精确求解的 ODE 理解 flow map，推导双时间预测、短跳复合与蒸馏，并讨论一步生成的多样性。

- [第 8 篇 从论文结果走向研究判断]({% post_url 2026-11-19-continuous-dllm-08-research-evaluation %})：拆解连续语言模型的 scaling、似然与生成指标，围绕质量、多样性、计算和比较条件组织研究判断。

### 选读 2 篇

- [选读 A 连续与离散的混合模型]({% post_url 2026-11-19-continuous-dllm-09-hybrid-models %})：解释共享潜变量、连续与离散过程的组合，以及混合模型如何表达逐位置预测之外的联合依赖。

- [选读 B Simplex 与球面上的流]({% post_url 2026-11-19-continuous-dllm-10-simplex-sphere %})：从概率约束推导 Dirichlet 速度、Fisher–Rao 度量与平方根坐标，说明球面流的训练和有限步更新。

## 怎样安排阅读顺序

- **建立基础**：先读第 1 至 3 篇，把联合分布、条件均值和连续动力学连起来。第 2、3 篇的手算例子适合边读边推。
- **理解模型设计**：第 4 至 6 篇依次处理表示、训练和采样，关注每个选择怎样改变可辨认性、优化目标和实际计算。
- **进入研究问题**：第 7、8 篇讨论少步生成与评估。想尽早联系离散 dLLM，可以在第 3 篇后先读选读 A；几何选读 B 建议在第 4 篇后阅读。

## 统一记号

主线让时间 t 从 0 走到 1：x₀ 是噪声，x₁ 是干净表示，xₜ 是中间状态。C 表示干净表示，ε 表示标准高斯噪声；L 是序列长度，V 是词表大小，d 是单个位置的表示维度。Dθ 预测干净表示，bθ 预测速度，sθ 预测 score。

涉及 DDPM 或其他原始方法时，正文会单独说明它们的时间约定。对照论文公式之前，先确认时间是从噪声走向数据，还是反过来。

第一遍需要的基础主要是向量运算、导数、概率和交叉熵。条件期望、ODE、SDE、Fisher–Rao 度量等概念，会在相应问题出现时解释。

## 配图与推导怎样配合

概念图重点解释公式中的对象与运算次序：联合采样为何不同于独立采样、后验均值为何未必是合法样本、训练插值与生成轨迹为何不同，以及奖励为何不能随意与平均交换。训练篇还用图区分连续状态和 self-conditioning 草稿，几何篇展示切向速度与有限步更新的差异。

第 3 篇的二维流图由正文给出的解析式计算，直接对应两位置模型。其余示意图用于解释关系，图注会说明读图方式；具体数值与适用条件以推导为准。

## 出处与资料范围

系列以 [Continuous Diffusion and Flow Language Models Survey](https://continuous-lm.github.io/) 的知识框架为起点，按理解依赖重新安排主题。Survey 的文献截止日期为 2026 年 11 月 2 日；各篇保留相关论文链接及具体版本说明。

正文用于建立完整的机制理解。需要精确复现某个实现、核对最新实验数字或研究边界处的严格证明时，再按具体问题回查原文和代码。
