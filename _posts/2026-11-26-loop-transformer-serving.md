---
layout: post
title: "计算的回路（四）：Loop Transformer 的推理系统——跨轮 KV 与投机解码"
date: 2026-11-26
last_modified_at: 2026-11-26
permalink: /blog/loop-transformer-serving/
visibility: unlisted
sitemap: false
series: 计算的回路
series_description: 循环模型、迭代生成、隐式推理与推理系统
series_order: 4
lang: zh-CN
math: true
tags: [Loop Transformer, KV Cache, Inference, Speculative Decoding]
description: "从历史 KV 的写入和读取出发，比较逐轮缓存、跨轮共享、浅轮自草拟和长上下文并发调度。"
---

> 计算的回路 · [一 · 循环架构]({% post_url 2026-10-19-loop-transformer-architecture %}) / [二 · 去噪与生成]({% post_url 2026-10-19-loop-denoising-flows %}) / [三 · 隐式推理]({% post_url 2026-10-19-loop-latent-reasoning %}) / **四 · 推理系统**

<a class="paper-link" href="https://vllm.ai/blog/2026-11-20-inkling2-mini" title="Inkling 2 Mini on vLLM: Day-0 Support for a Looped Model">Inkling 2 Mini</a><a class="paper-citation" id="cite-1-1" href="#ref-1" title="Inkling 2 Mini on vLLM: Day-0 Support for a Looped Model" aria-label="参考文献 1">[1]</a> 的中间 12 层要跑三遍。vLLM 把三次执行注册成不同的逻辑层，分别管理 KV 与卷积状态，同时把跨轮读消息的操作合进少数 kernel。<a class="paper-link" href="https://arxiv.org/abs/2611.03998" title="Dory1.0-4B: A Looped Normalized-GPT">Dory1.0-4B</a><a class="paper-citation" id="cite-2-1" href="#ref-2" title="Dory1.0-4B: A Looped Normalized-GPT" aria-label="参考文献 2">[2]</a> 则只保存末轮的历史 KV。同样是共享中间层，两款模型给历史状态定了不同的读写规则。

这会影响投机解码。一轮浅层预测可以提前启动下一个 token，但如果一个批次同时推进三四个深度，attention 也可能在同一次共享块调用中读三四份历史。短 prompt 上省掉的是串行权重访问。长上下文和多请求并发则可能更早耗尽 KV 带宽。

## 1. 每一轮究竟要存什么
{: #cross-loop-kv }

设循环块里物理注意力层 $$\ell$$ 在第 $$r$$ 次调用收到位置 $$i$$ 的状态 $$u_{\ell,r,i}$$。它写出的键值为

$$
K_{\ell,r,i}=W^K_\ell u_{\ell,r,i},\qquad
V_{\ell,r,i}=W^V_\ell u_{\ell,r,i}.
$$

投影权重可以复用。各轮的输入状态仍会变化，所以独立分轮缓存要区分层、轮次和位置。有 $$L$$ 个循环全注意力层，循环 $$R$$ 轮，批大小 $$B$$、序列长度 $$S$$ 时，静态占用约为

$$
M_{\mathrm{KV}}^{\mathrm{full}}\approx 2BSLR\,n_{\mathrm{kv}}d_h b.
$$

$$n_{\mathrm{kv}}$$ 是 KV head 数，$$d_h$$ 是 head dimension，$$b$$ 是每个元素的字节数。该式计算循环部分的张量容量。未循环层、局部窗口、其他 token mixer 的状态、页表与临时缓冲需要分别计入。模型按整栈循环或逐层循环时，权重访问顺序不同。只要两者都独立保存同样次数的全注意力调用，估算式就相同。

### 1.1 只存末轮，还是全局共享并保留窗口

| 方案 | 历史 KV 的读写 | 要实测的成本 |
|---|---|---|
| **逐轮独立** | Inkling 2 Mini 每个循环深度都写自己的历史，同深度 request 读取对应页。 | 逐轮缓存便于复用已有 cache manager。全局 attention 份数与独立深度有关，卷积状态也要分别计入。 |
| **末轮覆盖** | Dory1.0-4B 保存末轮历史。早轮处理新 token 时，可以读到既有 token 已经精炼的 KV。 | 历史容量按一轮计。实际收益受状态分布、上下文长度和末轮历史能否供浅轮使用影响。 |
| **共享全局历史，另存局部窗口** | <a class="paper-link" href="https://arxiv.org/abs/2610.02383v1" title="The Surprising Effectiveness of Shared Memory in Looped Transformers">LPT</a><a class="paper-citation" id="cite-3-1" href="#ref-3" title="The Surprising Effectiveness of Shared Memory in Looped Transformers" aria-label="参考文献 3">[3]</a> 保存第一轮的全局 KV，后续各轮另保留局部窗口。<a class="paper-link" href="https://www.zhipuai.cn/zh/research/164" title="GLM-5.4 官方技术公告">GLM-5.4</a><a class="paper-citation" id="cite-4-1" href="#ref-4" title="GLM-5.4 官方技术公告" aria-label="参考文献 4">[4]</a> 也采用首轮稀疏 KV 与后轮局部 KV，并复用首轮索引。 | 全局占用随上下文增长一次。每一轮仍需执行相应的 query 和 attention 读取，复用索引只省去了重复检索。 |
| **单条可更新状态** | <a class="paper-link" href="https://arxiv.org/abs/2605.07721" title="Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models">MELT</a><a class="paper-citation" id="cite-5-1" href="#ref-5" title="Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models" aria-label="参考文献 5">[5]</a> 为每层、每 token 保留一份 latent state，用门控更新共享记忆。 | 要训练模型适应新的读写关系，更新顺序也影响 prefill 并行方式。 |

以 LPT 为例，若第一个全局缓存保存 $$S$$ 个位置，每个后续窗口保留至多 $$w+1$$ 个位置，则循环块的 KV 容量约为

$$
M_{\mathrm{KV}}^{\mathrm{LPT}}\approx
2BL\bigl[S+(R-1)\min(S,w+1)\bigr]n_{\mathrm{kv}}d_h b.
$$

它说明窗口共享为何能降低峰值显存。要估算 decode 时间，还得数每轮实际扫描的 KV 字节数。首轮全局历史被读多次时，容量节省和带宽节省会有不同的比例。

![MELT 的固定大小 latent state 更新]({{ '/assets/img/blog/loop-series/melt_memory_efficient_loop.png' | relative_url }})

*MELT 用可更新的记忆保留跨轮信息。*

### 1.2 谁负责写全局历史

Causal encoder-decoder 把写入全局 KV 的工作集中在少数 encoder 层。<a class="paper-link" href="https://arxiv.org/abs/2611.06835" title="LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models">LooM</a><a class="paper-citation" id="cite-6-1" href="#ref-6" title="LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models" aria-label="参考文献 6">[6]</a> 在读取全局记忆的 decoder 侧增加局部循环，历史状态按写入规则只存一份。它为逐层执行提供不同预算，预算先在训练中采样，再在部署时选取。

<a class="paper-link" href="https://arxiv.org/abs/2611.13378" title="Looping Deeper Without Growing the KV Cache: Recurrent Depth for Causal Encoder-Decoders">Looped CED</a><a class="paper-citation" id="cite-7-1" href="#ref-7" title="Looping Deeper Without Growing the KV Cache: Recurrent Depth for Causal Encoder-Decoders" aria-label="参考文献 7">[7]</a> 把循环放到 encoder，让多轮计算改进最终写入的全局 KV。作者还用随机循环深度训练，使各轮可以共享末轮缓存。论文在 552B 级 DeepSeek-V4.1-Flash 的八轮改造和 4 张 GB200 上做了服务实验。独立分层、分轮缓存装不下一条 128K 请求，Looped CED 在相同硬件上容纳了 38 条。

还有两条可以叠加的做法。<a class="paper-link" href="https://arxiv.org/abs/2611.19446" title="Architectures and KV Replay for Depth-Recurrent Language Models">KV Replay</a><a class="paper-citation" id="cite-8-1" href="#ref-8" title="Architectures and KV Replay for Depth-Recurrent Language Models" aria-label="参考文献 8">[8]</a> 只在选中的轮写 KV，其余轮回放已有状态，并用周期性全注意力层保留长程访问。<a class="paper-link" href="https://arxiv.org/abs/2611.25369" title="Fold the Loop: Decoding-Time KV Memory for Looped Vision-Language Models">FoldCache</a><a class="paper-citation" id="cite-9-1" href="#ref-9" title="Fold the Loop: Decoding-Time KV Memory for Looped Vision-Language Models" aria-label="参考文献 9">[9]</a> 在 LoopVL 的完整 prefill 后折叠视觉 prefix KV，问题和答案仍按轮独立。后者约减半 prefix KV 的结果只覆盖所评测的视觉缓存和任务。

## 2. 提前退出会在历史上留下空缺
{: #adaptive-cache }

一个历史 token 在第二轮停止，后面一个 token 却跑到第四轮。第四轮 query 该读取哪一份历史？<a class="paper-link" href="https://arxiv.org/abs/2507.10524" title="Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation">MoR</a><a class="paper-citation" id="cite-10-1" href="#ref-10" title="Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation" aria-label="参考文献 10">[10]</a> 的稀疏分轮缓存只读取该深度仍活跃的位置。它也评估了首轮共享，即所有深度读第一轮完整历史。<a class="paper-link" href="https://arxiv.org/abs/2610.09013v1" title="Enabling Dynamic Computation in Looped LMs">Ragged KV</a><a class="paper-citation" id="cite-11-1" href="#ref-11" title="Enabling Dynamic Computation in Looped LMs" aria-label="参考文献 11">[11]</a> 则保留历史 token 已经算出的各轮 KV，缺少目标深度时读取最深的一份。设 token $$i$$ 在第 $$e_i$$ 轮退出，第 $$d$$ 轮查询使用

$$
(K_i,V_i)^{\min(d,e_i)}.
$$

按深度找页后，后续 token 可以继续利用已经退出的位置，系统也省下了为旧 token 补跑的计算。还得跟踪每轮活跃 token 数和实际执行的 kernel。若同一批中仍要为少数深轮请求运行整个张量，平均退出深度的下降可以先体现为逻辑 FLOPs 变化，墙钟时间则取决于如何把活跃行打包。

另一个选择是让 prefill 和 decode 使用不同的预算。<a class="paper-link" href="https://arxiv.org/abs/2611.07211" title="Not All Loops Cost the Same: Phase-Dependent Depth for Serving Looped Transformers">LoopDuet</a><a class="paper-citation" id="cite-12-1" href="#ref-12" title="Not All Loops Cost the Same: Phase-Dependent Depth for Serving Looped Transformers" aria-label="参考文献 12">[12]</a> 联合选择两阶段深度，并处理 prefill 结束和 decode 开始之间的交接。对长 prompt 来说，减少一个 prefill 循环会作用于整段输入。对长输出来说，减少一轮的收益会在生成中反复出现。若把阶段预算分开评测，就能判断开销变化落在首 token 之前还是之后。[第一篇的弹性深度和停机]({% post_url 2026-10-19-loop-transformer-architecture %}#depth-budget)讨论了这些预算怎样训练出来。

## 3. 用浅轮先启动下一个 token
{: #draft-types }

给 token $$x_i$$ 做全深度预测时，当前请求要等共享块执行完 $$R$$ 次。若第 $$d<R$$ 轮已经给出可信的候选，可以暂时把它作为 $$x_i$$，启动 $$x_{i+1}$$。之前的 token 同时向全深度推进。一旦验证拒绝，依赖该分支的推测状态需要回收。缓存管理因此要区分已提交的前缀、在途草稿和可以安全复用的 KV。

采样时要求更严格。设同一已确认前缀上的草稿分布为 $$q$$、目标分布为 $$p$$，从 $$q$$ 取得的候选 $$x$$ 以 $$\min(1,p(x)/q(x))$$ 的概率接受。拒绝时从归一化的 $$(p-q)_+$$ 取补偿 token。greedy 模式逐位置核对目标 argmax。两种验证都依赖一个明确的目标执行配置。若改了目标端的 KV 规则或轮数，应该先定义这个新配置的 AR 基线，再判断投机流程是否保持它的输出。

### 3.1 填满浅轮和深轮之间的空档

<a class="paper-link" href="https://arxiv.org/abs/2609.23033" title="WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models">WaveFront</a><a class="paper-citation" id="cite-13-1" href="#ref-13" title="WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models" aria-label="参考文献 13">[13]</a> 把浅轮草拟和深轮验证排在同一次共享块调用里。不同 token 像对角线上的位置，每次一起前进一步。<a class="paper-link" href="https://arxiv.org/html/2609.17184v1" title="LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers">LoopSpec</a><a class="paper-citation" id="cite-14-1" href="#ref-14" title="LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers" aria-label="参考文献 14">[14]</a> 在浅轮之后允许一次有条件的第二提案。起草得越早，流水线越长，但不足以预测下一个 token 的分支也越多。相对各自的全深度 AR，32 轮 Raven 最高快约 7 倍，四轮 Ouro 最高快约 3 倍。较深的模型留出了更多可重叠的串行轮次。

<a class="paper-link" href="https://arxiv.org/abs/2609.34538" title="Shallow Queries, Mature Values: Depth-Asynchronous Self-Speculation for Looped Transformers">DAS-Wave</a><a class="paper-citation" id="cite-15-1" href="#ref-15" title="Shallow Queries, Mature Values: Depth-Asynchronous Self-Speculation for Looped Transformers" aria-label="参考文献 15">[15]</a> 让浅层 query 读取已确认前缀的成熟 values，再由独立的全深度计算负责验收。

流水线也可以一次接收多个候选位置。<a class="paper-link" href="https://arxiv.org/abs/2610.21594" title="LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models">LoopAhead</a><a class="paper-citation" id="cite-16-1" href="#ref-16" title="LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models" aria-label="参考文献 16">[16]</a> 训练轻量 lookahead 模块预测多个后续 token，并用更深轮的反馈控制哪些分支继续执行。<a class="paper-link" href="https://arxiv.org/html/2611.19649v1" title="LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers">LoopFill</a><a class="paper-citation" id="cite-17-1" href="#ref-17" title="LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers" aria-label="参考文献 17">[17]</a> 让浅轮输出和后缀检索共同写入一棵草稿树，复用前缀中已经出现过的字符串。后缀匹配成功时，可以在一个浅轮预测后立即接上多个位置。在单张 L20X 上做单请求 greedy 解码时，LoopFill 在 Ouro-2.6B 和 Huginn-3.5B 上相对同引擎 AR 平均快约 5–6 倍。

### 3.2 目标有循环，草稿也可以来自专用模块

Inkling 2 Mini 自带 **DSpark**。vLLM 的实现会按固定的层执行位置取 drafter 输入，所以可变深度 checkpoint 改成其他轮数时，不能直接沿用默认深度的 DSpark。<a class="paper-link" href="https://huggingface.co/sensenova/SenseNova-Loop-9B" title="SenseNova-Loop-9B 官方模型卡">SenseNova-Loop-9B</a><a class="paper-citation" id="cite-18-1" href="#ref-18" title="SenseNova-Loop-9B 官方模型卡" aria-label="参考文献 18">[18]</a> 同样提供了 MTP 投机解码支持和匹配的量化草稿。两者都允许把目标的循环结构与草稿模块分别分析。[SenseSpec](https://github.com/OpenSenseNova/SenseSpec) 还提供了训练 DSpark 和 DFlash2 草稿的代码。

<a class="paper-link" href="https://arxiv.org/abs/2610.24766" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction">ECHO</a><a class="paper-citation" id="cite-19-1" href="#ref-19" title="ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction" aria-label="参考文献 19">[19]</a> 把未来 token 的训练目标放到骨干末层。推理时沿未来位置递归调用末层，生成候选后交给完整模型验证。LoopSpec 的浅轮状态来自当前 token 的执行深度，ECHO 的后续状态则随着草稿位置前移。[MTP 篇]({% post_url 2026-10-23-mtp-pretraining-self-drafting %}#54-echo未来监督是否应该直接约束主干最后一层)说明了各深度实际被执行的频率，[第一篇的 DDS 推导]({% post_url 2026-10-19-loop-transformer-architecture %}#dds-loss-weights)给出了对已执行的各步损失取平均后，每个深度的期望损失权重。另一条叫 LoopCD 的路线会对早晚轮 logits 做重加权，得到新的输出分布，详见[第三篇]({% post_url 2026-10-19-loop-latent-reasoning %}#readout-dynamics)。

## 4. 长上下文和高并发先检查 KV 流量
{: #concurrency }

WaveFront 把不同深度放进同一次共享块调用，能省掉部分权重读取。各深度的 attention 仍可能要装入各自的历史。序列变长、并发提高后，这部分读取足以占满带宽，单请求省掉的串行时间就难以外推到整个批次。

<a class="paper-link" href="https://arxiv.org/html/2611.24301v1" title="LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers">LoopCache</a><a class="paper-citation" id="cite-20-1" href="#ref-20" title="LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers" aria-label="参考文献 20">[20]</a> 把流量约束直接放进调度。它循环选择一轮 KV，把匹配到该轮的候选放在一起执行，避免每拍读齐所有深度的历史，少量敏感浅层仍保留独立缓存。置信度变化还可以触发在途草稿更新。论文在 8 张 A100 上测试了 4K–128K 上下文，相对同设置下的 AR 解码平均快约 5 倍。

<a class="paper-link" href="https://arxiv.org/abs/2611.30205" title="PackLoop: Row-Packed Self-Speculative Decoding for Looped Language Models under Concurrency">PackLoop</a><a class="paper-citation" id="cite-21-1" href="#ref-21" title="PackLoop: Row-Packed Self-Speculative Decoding for Looped Language Models under Concurrency" aria-label="参考文献 21">[21]</a> 优化并发时的另一种浪费。对同一个请求、同一个轮次，若多个候选分别占一行，就可能反复读相同历史。它让深轮积累一小批位置，再在一行里处理，并把上下文复制来的候选并入原本就会读取的首轮。在论文的并发负载下，相对针对每个负载点选出的最快已有调度，PackLoop 的加速中位数约为 25%。

<details markdown="1">
<summary>各论文的实测对照</summary>

| 方法 | 对照与报告值 |
|---|---|
| LoopSpec | 相对各模型的全深度 AR，四轮 Ouro 最高 3.36×，32 轮 Raven 最高 6.83×。 |
| LoopFill | 单张 L20X、单请求 greedy，七项任务上相对同引擎 AR 的平均加速为 Ouro-2.6B 5.85×、Huginn-3.5B 5.13×。 |
| LoopCache | 8 张 A100、4K–128K 上下文，对所实现的 AR、先草拟后验证、WaveFront、LoopSpec 的平均加速分别为 5.12×、2.52×、10.25×、14.56×。 |
| PackLoop | 164 个负载点中 161 个更快，相对逐点选出的最快已有调度，中位加速 1.25×、最高 1.76×。 |

</details>

| 要回答的问题 | 至少记录的观测量 |
|---|---|
| 一条请求是否更快 | 固定模型、轮数、上下文、输出长度和采样设置，分别报告 TTFT、每输出 token 延迟、完成时间。 |
| 浅轮是否适合起草 | 记录每个草拟深度的接受长度、分支回收量，以及额外 drafter 和读出开销。 |
| 长上下文还能省多少 | 同时记录每 token 总 KV 容量、每轮读取字节数、上下文长度及带宽利用率。 |
| 并发增加后还能得到什么 | 报告 batch 或请求到达率、总输出 tokens/s，以及在约定延迟下可服务的吞吐。 |

## 5. 两个模型的部署对照

Inkling 在逐轮 KV 的配置上已经有公开的 vLLM 实现，适合固定轮数，做 DSpark 开关对照。若要分离循环次数的影响，用研究版固定权重，分别测试一至三轮。vLLM 在 4K、单张 GB200、只计 forward 的结果里，三轮配置共享 KV 后，缓存容量从 175K 提至 194K tokens，单请求 decode step 仍约 3.85 ms。卷积状态按轮保留，解释了容量变化为何较小。

SenseNova 的开放权重让另一组对照可做。固定两轮 target 与采样设置，先切换 MTP，再改变 prompt 长度和 KV 精度，便能分辨收益来自多 token 验证、KV 带宽，还是显存多容纳了请求。4-bit 权重和 8-bit KV 下 256K 约需 15 GiB 是模型卡的部署估算。验证吞吐仍需报告硬件、并发、草稿接受长度和解码配置。

<details markdown="1">
<summary>其他会改变这组预算的做法</summary>

[HLA](https://arxiv.org/abs/2610.07940v1) 和 [ResidualQuant](https://arxiv.org/abs/2610.10381v1) 压缩历史表示。[RAMP](https://arxiv.org/abs/2610.13041v1) 给 prefill 的不同轮安排不同的计算精度。[LT2](https://arxiv.org/abs/2605.20670) 和 [MixerLoop](https://arxiv.org/abs/2608.18230) 调整循环内反复执行的算子。[LoopCoder-v2](https://arxiv.org/abs/2606.18023) 用位置偏移减少轮次之间的串行依赖，训练配置的变化需要用各自的 checkpoint 评估。

![LT2 中 loop 与 linear-time attention 的结合]({{ '/assets/img/blog/loop-series/lt2_linear_time_loop.png' | relative_url }})

</details>

## 参考文献

1. <span id="ref-1" class="reference-anchor"></span>[Inkling 2 Mini on vLLM: Day-0 Support for a Looped Model](https://vllm.ai/blog/2026-11-20-inkling2-mini)。 <a class="reference-backlink" href="#cite-1-1" aria-label="返回文献 1 首次引用处" title="返回正文">↩</a>
2. <span id="ref-2" class="reference-anchor"></span>[Dory1.0-4B: A Looped Normalized-GPT](https://arxiv.org/abs/2611.03998)。 <a class="reference-backlink" href="#cite-2-1" aria-label="返回文献 2 首次引用处" title="返回正文">↩</a>
3. <span id="ref-3" class="reference-anchor"></span>[The Surprising Effectiveness of Shared Memory in Looped Transformers](https://arxiv.org/abs/2610.02383v1)。 <a class="reference-backlink" href="#cite-3-1" aria-label="返回文献 3 首次引用处" title="返回正文">↩</a>
4. <span id="ref-4" class="reference-anchor"></span>[GLM-5.4 官方技术公告](https://www.zhipuai.cn/zh/research/164)。 <a class="reference-backlink" href="#cite-4-1" aria-label="返回文献 4 首次引用处" title="返回正文">↩</a>
5. <span id="ref-5" class="reference-anchor"></span>[Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models](https://arxiv.org/abs/2605.07721)。 <a class="reference-backlink" href="#cite-5-1" aria-label="返回文献 5 首次引用处" title="返回正文">↩</a>
6. <span id="ref-6" class="reference-anchor"></span>[LooM: Loop-Invariant Memory for Causal Encoder-Decoder Language Models](https://arxiv.org/abs/2611.06835)。 <a class="reference-backlink" href="#cite-6-1" aria-label="返回文献 6 首次引用处" title="返回正文">↩</a>
7. <span id="ref-7" class="reference-anchor"></span>[Looping Deeper Without Growing the KV Cache: Recurrent Depth for Causal Encoder-Decoders](https://arxiv.org/abs/2611.13378)。 <a class="reference-backlink" href="#cite-7-1" aria-label="返回文献 7 首次引用处" title="返回正文">↩</a>
8. <span id="ref-8" class="reference-anchor"></span>[Architectures and KV Replay for Depth-Recurrent Language Models](https://arxiv.org/abs/2611.19446)。 <a class="reference-backlink" href="#cite-8-1" aria-label="返回文献 8 首次引用处" title="返回正文">↩</a>
9. <span id="ref-9" class="reference-anchor"></span>[Fold the Loop: Decoding-Time KV Memory for Looped Vision-Language Models](https://arxiv.org/abs/2611.25369)。 <a class="reference-backlink" href="#cite-9-1" aria-label="返回文献 9 首次引用处" title="返回正文">↩</a>
10. <span id="ref-10" class="reference-anchor"></span>[Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524)。 <a class="reference-backlink" href="#cite-10-1" aria-label="返回文献 10 首次引用处" title="返回正文">↩</a>
11. <span id="ref-11" class="reference-anchor"></span>[Enabling Dynamic Computation in Looped LMs](https://arxiv.org/abs/2610.09013v1)。 <a class="reference-backlink" href="#cite-11-1" aria-label="返回文献 11 首次引用处" title="返回正文">↩</a>
12. <span id="ref-12" class="reference-anchor"></span>[Not All Loops Cost the Same: Phase-Dependent Depth for Serving Looped Transformers](https://arxiv.org/abs/2611.07211)。 <a class="reference-backlink" href="#cite-12-1" aria-label="返回文献 12 首次引用处" title="返回正文">↩</a>
13. <span id="ref-13" class="reference-anchor"></span>[WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models](https://arxiv.org/abs/2609.23033)。 <a class="reference-backlink" href="#cite-13-1" aria-label="返回文献 13 首次引用处" title="返回正文">↩</a>
14. <span id="ref-14" class="reference-anchor"></span>[LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers](https://arxiv.org/html/2609.17184v1)。 <a class="reference-backlink" href="#cite-14-1" aria-label="返回文献 14 首次引用处" title="返回正文">↩</a>
15. <span id="ref-15" class="reference-anchor"></span>[Shallow Queries, Mature Values: Depth-Asynchronous Self-Speculation for Looped Transformers](https://arxiv.org/abs/2609.34538)。 <a class="reference-backlink" href="#cite-15-1" aria-label="返回文献 15 首次引用处" title="返回正文">↩</a>
16. <span id="ref-16" class="reference-anchor"></span>[LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models](https://arxiv.org/abs/2610.21594)。 <a class="reference-backlink" href="#cite-16-1" aria-label="返回文献 16 首次引用处" title="返回正文">↩</a>
17. <span id="ref-17" class="reference-anchor"></span>[LoopFill: Pipelined Multi-Token Speculative Decoding for Looped Transformers](https://arxiv.org/html/2611.19649v1)。 <a class="reference-backlink" href="#cite-17-1" aria-label="返回文献 17 首次引用处" title="返回正文">↩</a>
18. <span id="ref-18" class="reference-anchor"></span>[SenseNova-Loop-9B 官方模型卡](https://huggingface.co/sensenova/SenseNova-Loop-9B)。 <a class="reference-backlink" href="#cite-18-1" aria-label="返回文献 18 首次引用处" title="返回正文">↩</a>
19. <span id="ref-19" class="reference-anchor"></span>[ECHO: Efficient Causal Self-Recursion Beyond Auxiliary Multi-Token Prediction](https://arxiv.org/abs/2610.24766)。 <a class="reference-backlink" href="#cite-19-1" aria-label="返回文献 19 首次引用处" title="返回正文">↩</a>
20. <span id="ref-20" class="reference-anchor"></span>[LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers](https://arxiv.org/html/2611.24301v1)。 <a class="reference-backlink" href="#cite-20-1" aria-label="返回文献 20 首次引用处" title="返回正文">↩</a>
21. <span id="ref-21" class="reference-anchor"></span>[PackLoop: Row-Packed Self-Speculative Decoding for Looped Language Models under Concurrency](https://arxiv.org/abs/2611.30205)。 <a class="reference-backlink" href="#cite-21-1" aria-label="返回文献 21 首次引用处" title="返回正文">↩</a>
{: .references }
