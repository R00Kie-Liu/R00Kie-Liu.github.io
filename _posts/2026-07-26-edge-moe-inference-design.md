---
layout: post
title: "MoE我们喜欢你（六）：端侧推理的权重读取与专家搬运"
series: moe-we-like-you
series_order: 6
last_modified_at: 2026-10-27
date: 2026-07-26
tags: [MoE, Infra, Inference]
description: "从权重读取与专家加载解释端侧 MoE 推理，分析 MobileMoE、MiniCPM-V 4.7 LES 与 BigMoMo，并简述投机验证中的专家复用问题。"
---

## 0. 从一次权重读取开始

一个 5B total、0.8B active 的 MoE，按纯 INT4 打包计算，完整权重约为 2.5GB；再按每份 active 权重读取一次的简化口径估算，单 token 的权重流量约为 400MB。这两个数字对应不同的开销：2.5GB 决定能否把模型留在内存里，400MB 决定生成一个 token 时需要读取多少权重。

如果 400MB 都从 DRAM 读取，有效带宽为 60GB/s，纯读取就需要约 6.7ms。如果其中一些专家留在闪存里，还要先把它们加载到内存。矩阵乘即使很快，也必须等输入和权重就绪才能开始。

端侧 MoE 的执行问题由此分成两种：完整权重能够常驻时，怎样减少每个输出 token 的读取与算子开销；权重放不下时，怎样让专家加载少阻塞计算。本文先讨论 MobileMoE 的内存常驻执行，再讨论 MiniCPM-V 4.7 的 LES 与 BigMoMo 怎样组织专家加载。

## 1. 一个专家从哪里读，什么时候能算

对某一层、某一个 token，专家路径可以写成：

```text
hidden state
  → router 产生专家编号与权重
  → 查找这些专家的权重位置
  → 准备尚未就绪的权重
  → 执行各专家 FFN
  → 按路由权重合并输出
```

“准备权重”在不同部署中包含不同操作：

| 权重所在位置 | 执行前需要做什么 |
|---|---|
| 手机 DRAM / 统一内存 | 计算单元读取权重；量化格式还可能要求解包或反量化 |
| 手机闪存 | 先读入 DRAM，再交给对应计算后端 |
| 独立 GPU 的 CPU 内存 | 经主机到设备的数据路径搬入显存 |

手机的 CPU、GPU 通常共享 DRAM，不应直接套用独立显卡的 CPU–GPU 拷贝模型。具体后端还可能增加自己的准备阶段。例如后文 BigMoMo 使用的 Hexagon NPU，需要将 DRAM 区域映射到 NPU 地址空间，再把计算所需的权重块搬到本地 VTCM 存储。

路由结果依赖当前层的 hidden state。按需加载的直接实现必须等 router 给出编号，才知道缺哪些专家；这段加载时间会阻塞后续 FFN。预取试图更早发出读取请求，缓存则试图让下一次请求直接命中已经加载的权重。两者都在缩短“路由完成后，计算还需要等多久”。

## 2. 完整权重常驻时，时间花在哪里

### 2.1 active 参数怎样换算成读取量

先假设每份 active 权重在一次 decode 中读取一次，没有跨 token 的缓存复用。忽略量化元数据，INT4 权重的读取量为：

$$
B_{\mathrm{weight}}=N_{\mathrm{active}}\times\frac{4}{8}\ \text{bytes}.
$$

用假设的有效带宽 60GB/s 估算：

| active 参数量 | 纯 INT4 权重字节量 | 读取时间 |
|---:|---:|---:|
| 0.3B | 150MB | 2.5ms |
| 0.8B | 400MB | 6.7ms |
| 1.0B | 500MB | 8.3ms |

这是一组教学估算，MB、GB 均使用十进制单位。它只计算权重流量；完整 decode 还要执行 attention、读取 KV、处理量化数据并调度算子。缓存命中可以减少 DRAM 读取，实际带宽也随访问方式变化，因此表中数值不能直接当作 token 延迟。

不过，这个估算解释了一个具体收益：在相同权重精度下，减少每步访问的专家参数，可以减少需要从 DRAM 读取的字节数。收益能保留多少，取决于后面的专家计算是否引入了过多小算子与调度开销。

### 2.2 同一个专家处理多少 token

设一个 expert 的输入宽度为 `H`，FFN 中间宽度为 `I`。单 token 的一次 projection 是：

```text
[1, H] @ [H, I] → [1, I]
```

若同一批次有 `m` 个 token 选择它，就可以将输入聚成 `[m, H]`，用同一份权重计算。这既增加矩阵乘的并行度，也提供了跨 token 复用权重的机会。

这里要统计同一层的一组 token 访问了多少个不同专家。用一个 top-2 的教学例子说明：

```text
token A → {E3, E17}
token B → {E17, E60}
```

一共发生 4 次 token–expert 调用，但只涉及 3 份唯一专家权重。若把两个 token 放在同一验证批次，并让 E17 的两个输入一起计算，在权重可复用的实现中，E17 可以服务两个 token 而只加载一次。如果 token B 改选 `{E8, E60}`，唯一权重就变成 4 份。

这个并集就是该批次的专家工作集。不同层的专家分别统计；跨层汇总时使用 `(layer, expert_id)` 标识一份权重。工作集给出了需要哪些权重，实际读取次数还取决于缓存容量、淘汰和计算分块。

普通 AR decode 在单请求下每次只推进一个 token，无法提前把后续 token 任意凑成 batch；prefill 已有多个输入位置，推测解码则先提出多个候选，为目标模型构造验证批次。后文 BigMoMo 利用这个窗口组织专家搬运与计算。

### 2.3 MobileMoE 怎样执行这些小矩阵

[MobileMoE][mobilemoe] 的 0.3B–0.9B active、1.3B–5.3B total 配置，面向完整量化权重能够放入手机内存的部署。其 INT4 QAT 版本的 S/M/L 权重占用分别为 0.68、1.48、2.75GB。

[![MobileMoE 的 routed experts、shared expert 与细粒度切分结构](/assets/img/blog/moe-we-like-you/06-mobilemoe-expert-architecture.png)](/assets/img/blog/moe-we-like-you/06-mobilemoe-expert-architecture.png)

*图 1：MobileMoE 的专家结构，展示 top-k 路由、细粒度专家切分和 shared expert。截取自[原论文 Fig. 2 左侧][mobilemoe-figure]。*

论文的 ExecuTorch CPU 实现先按 expert id 重排 token，使同一专家的输入连续存放；再把 top-k、dispatch、gate/up projection、SwiGLU、down projection 和加权回填放进一个 MoE 算子。这样既能按专家组织矩阵乘，也减少算子之间的调用开销。小 active 参数量与这条执行路径共同构成其端侧收益。

论文在 Galaxy S25 的 4 CPU 线程、iPhone 16 Pro 的 2 CPU 线程上，以 batch=1、INT4 权重和动态 INT8 激活测试。MobileMoE-S 相对 MobileLLM-Pro 报告约 2.2–3.4× decode 加速；两者权重占用分别为 0.68GB 和 0.55GB，QAT 后 14 项平均分为 44.0 和 45.5。这组比较对应较接近的权重容量与质量，不能解释为同一模型只替换了 kernel 后的加速。

## 3. LES：把下一层的加载提前到当前计算期间

模型扩大到几十 B total 后，完整权重可能超过应用可用内存。[MiniCPM-V 4.7][minicpm-v47] 的 35B-A3B 版本通过存储卸载保留完整专家池；[Lookahead Expert Scheduling（LES）][les] 用跨层路由预测提前加载下一层可能需要的专家。

[![LES 跨层专家预测、异步预取与就绪专家优先计算时间线](/assets/img/blog/moe-we-like-you/06-les-lookahead-expert-scheduling.png)](/assets/img/blog/moe-we-like-you/06-les-lookahead-expert-scheduling.png)

*图 2：LES 在当前层计算期间预测并预取下一层专家，下一层优先计算已就绪的真实路由专家，同时补载其余权重。图中 top-8、命中率与时长对应官方示例。图源：[LES 官方流程图][les-figure]。*

以一层 top-2 路由为例。下面的编号只用于解释流程：

```text
预测下一层会用 {E2, E5} → 提前发起加载
下一层真实路由选 {E2, E7}
  E2 已就绪 → 先计算 E2
  E7 未就绪 → 加载 E7，尝试与 E2 的计算重叠
  E5 未被选中 → 不参与本次输出
E2、E7 都完成后 → 按真实路由权重合并
```

预测错误有两种成本：多读了不需要的 E5，或者没有及时准备好必需的 E7。LES 仍会补载 E7，所以预测命中率影响等待时间，计算所用的专家集合则由真实 router 决定。

重叠能隐藏多少时间，还取决于已有多少可执行工作。用一个独立加载与计算可以并行的简化例子：E7 的剩余加载需要 5ms，E2 的计算需要 3ms，那么 E2 结束后还需等约 2ms。若 E2 只需要 0.5ms，等待就有约 4.5ms。预取不仅要尽量命中，还要足够早；已经就绪的计算也必须能覆盖一部分加载。

官方文档还将连续权重布局、减少读取调用、直接写入 GPU 可见内存、逐层混合精度量化和 DSpark 推测采样接入同一流程。在 iPhone 18 Pro Max（设备总内存 12GB）上的逐阶段结果是：

| 累积启用的优化 | decode token/s |
|---|---:|
| 从存储按需加载专家的基线 | 3.38 |
| lookahead expert scheduling | 6.57 |
| 再加调度与加载优化 | 11.34 |
| 再加逐层混合精度量化 | 17.16 |
| 再加 DSpark 推测采样 | 20.51 |

速度按生成 token 数除以总 decode 时间计算。单独 lookahead 阶段约为基线的 1.94×，完整流程约为 6.07×。前者主要改变等待，布局优化改变读取效率，量化减少权重字节量，DSpark 再减少每个输出 token 分摊的解码开销。这些收益不能全部归给路由预测。

## 4. 投机验证怎样改变专家工作集

投机解码先用较便宜的 draft 生成候选，再由目标模型并行验证。同一专家可以服务多个候选，摊薄权重读取成本；但候选越多，也可能引入更多不同专家。被拒绝的候选触发的读取与计算，最终没有转化为有效输出。

[S²-MoE（S2-MoE）][s2-moe] 专门处理 MoE 的自投机解码效率：扩展候选时同时考虑接受收益和新增专家成本，并通过路由 bias 促进候选之间复用专家。这里与端侧搬运直接相关的是，验证成本取决于整个候选批次的专家工作集。候选树扩展、draft 与目标模型的上下文对齐等机制，留到投机解码系列展开。

## 5. BigMoMo：把验证请求排成可执行的搬运流水

验证批次暴露了一组专家请求，但这些权重到达计算单元的时间不同。[BigMoMo][bigmomo] 将请求筛选、闪存布局与 NPU 执行调度接成一条流水。

[![BigMoMo 的请求筛选、闪存组织与专家执行系统总览](/assets/img/blog/moe-we-like-you/06-bigmomo-system-overview.png)](/assets/img/blog/moe-we-like-you/06-bigmomo-system-overview.png)

*图 3：BigMoMo 系统总览。上半部分从 draft 节点与专家筛选推进到目标验证，下半部分对应闪存布局和 NPU 执行调度。截取自[原论文 Fig. 6][bigmomo-overview-figure]。*

它面向 Snapdragon 的 Hexagon NPU，将权重准备过程细分为：

```text
Flash → DRAM → NPU 可访问的映射缓冲区 → VTCM → HMX 矩阵计算
```

VTCM 是 NPU 的本地存储，HMX 是其矩阵计算单元。一个 expert 已经在 DRAM 里，仍可能在等待映射或搬入 VTCM。若每层都等全部专家就绪再启动计算，较早到达的权重也只能闲置。

BigMoMo 先减少进入流水的请求：根据接受收益筛掉低价值 draft 节点，再结合 routing impact 与各存储层级的搬运成本裁剪部分专家激活。专家被裁掉后，其路由权重会分给保留的专家；这一步近似了目标模型计算，质量需要随裁剪预算一起评估。

剩下的请求还要解决两个问题。其一是闪存上的碎片读取：经常一起加载的专家若分散在文件中，就需要多次读取。系统根据近期共同加载记录调整专家布局，尽量将这类请求转成连续读取。其二是到达时间不同：调度器将已就绪专家组成计算组，在 VTCM 容量允许的范围内，让当前计算覆盖下一组权重的准备；两个 VTCM 权重缓冲区交替加载与计算。

[![BigMoMo 的 VTCM 双缓冲与跨层级搬运计算流水](/assets/img/blog/moe-we-like-you/06-bigmomo-movement-pipeline.png)](/assets/img/blog/moe-we-like-you/06-bigmomo-movement-pipeline.png)

*图 4：左侧是 VTCM 双缓冲，右侧比较按需执行、固定双缓冲与跨层级调度。粉色区域中的候选计算组用于覆盖后续权重准备，减少 HMX 空闲区间。截取自[原论文 Fig. 11][bigmomo-pipeline-figure]；原图沿用了 MoMoXL 名称。*

因此，流水中的优化包含减少请求、改善存储布局和隐藏剩余等待。即使两轮验证拥有同样大小的专家工作集，命中 DRAM 的比例、闪存布局和 NPU 准备状态不同，也会产生不同延迟。

论文基于 llama.cpp 的 Hexagon 后端，以四种 Q4_0 MoE、五类任务评估，报告相对按需卸载 AR 基线平均 4.83× 加速。这个基线本身需要从闪存加载专家，与 MobileMoE 的内存常驻 CPU 测试、LES 的 Apple 设备测试分别对应不同的执行路径。

## 6. 三种端侧执行路径分别减少了哪部分时间

| 工作 | 主要处理的开销 | 是否改变目标模型的专家选择 |
|---|---|---|
| MobileMoE | 用较小 active 配置与融合算子减少读取和执行开销 | 训练得到新的模型；运行时执行其自身路由 |
| MiniCPM-V 4.7 LES | 提前加载、优先执行就绪专家，并改善权重读取 | 预取与调度保留真实 router 的选择；量化另有数值影响 |
| BigMoMo | 减少低收益请求，改善闪存布局，重叠多级搬运与 NPU 计算 | 专家裁剪会改变验证时的专家混合 |

完整权重常驻时，这些优化减少专家权重读取与小矩阵执行开销；使用卸载时，还要缩短权重就绪前的等待。推测解码扩大了一次可执行的 token 集合，也扩大了可能触发的专家集合。只有新增的有效输出足以分摊这些读取与验证成本，加速才会成立。

## 系列导航

上一篇：[第五篇：100B 训练配置、并行策略与显存推演]({% post_url 2026-07-21-moe-100b-parallel-memory %})

下一篇：[第七篇：RL 训推不一致与路由对齐问题]({% post_url 2026-07-31-moe-rl-train-serving-mismatch %})

## 参考资料

- [MobileMoE: Scaling On-Device Mixture of Experts][mobilemoe]，尤其是第 4.3 节的量化与端侧执行实现。
- [MiniCPM-V 4.7 官方文档][minicpm-v47]；[LES 机制与逐阶段结果][les]。
- [S²-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices][s2-moe]；[官方代码][s2-moe-code]。
- [BigMoMo: Efficient Inference of Large-Scale MoE with Speculative Decoding on Mobile Devices][bigmomo]，尤其是第 4 节的请求筛选、存储布局和搬运调度。

[mobilemoe]: https://arxiv.org/abs/2605.27358
[s2-moe]: https://arxiv.org/abs/2608.15018
[s2-moe-code]: https://github.com/angerybob/S2-MoE/tree/orin
[minicpm-v47]: https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_v4dot7_en.md
[les]: https://github.com/OpenBMB/llama.cpp-omni/blob/main/docs/les/lookahead-expert-scheduling.md
[bigmomo]: https://arxiv.org/abs/2609.14643

[mobilemoe-figure]: https://arxiv.org/html/2605.27358v1#S3.F2
[les-figure]: https://github.com/OpenBMB/llama.cpp-omni/blob/main/docs/les/lookahead-expert-scheduling.png
[bigmomo-overview-figure]: https://arxiv.org/html/2609.14643v1#S3.F6
[bigmomo-pipeline-figure]: https://arxiv.org/html/2609.14643v1#S4.F11
