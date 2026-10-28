---
layout: post
title: "MoE我们喜欢你（六）：端侧推理优化与投机解码优化"
series: moe-we-like-you
series_order: 6
last_modified_at: 2026-10-29
date: 2026-07-26
tags: [MoE, Infra, Inference]
description: "从端侧 MoE 与服务器 GPU 的执行差异出发，解释权重读取和加载瓶颈、投机验证引起的专家并集膨胀，以及候选、专家预算、训练和搬运优化。"
---

## 0. 端侧 MoE 的读取、执行与加载等待

端侧 MoE 的 active 参数量很小，decode 仍可能很慢。原因在于：每层选出专家后，设备要读取相应权重、执行若干小矩阵乘，再合并结果。权重在 DRAM 中时，时间可能主要花在读取和小算子调度上；权重留在闪存时，还要等待冷专家加载。减少 FFN 的计算量，只缩短了这条路径的一部分。

```text
hidden state → router / top-k → 准备专家权重 → expert FFN → 加权合并
                                  ↑
                      读取常驻权重，或先加载缺失专家
```

本文以手机上的交互式 decode 为主。先比较它与服务器 GPU 推理的内存和执行差异，再讨论量化、融合与预取能解决什么。随后引入投机解码：用一次权重准备服务多个候选，进一步摊薄成本；但 MoE 的候选还会引入更多专家，因而需要额外的验证优化。

## 1. 端侧与服务器 GPU 的内存和执行差异

这里以权重常驻 HBM 的服务器 GPU 推理作比较基准。手机通常使用 CPU、GPU、NPU 共享的 DRAM，完整专家池放不下时再依赖闪存。两种部署中，专家就绪之前需要经过的路径不同。

| 约束 | 权重常驻的服务器 GPU | 手机端交互式推理 | 对 MoE 执行的影响 |
|---|---|---|---|
| 权重存储 | 专家权重已在 HBM 中 | 较小模型常驻 DRAM，大模型可能将冷专家留在闪存 | 端侧可能先遇到容量限制，再为每次专家 miss 支付加载时间 |
| 权重读取 | 通过高带宽显存向计算单元供数 | 从共享 LPDDR/DRAM 供数，应用可用带宽还受其它单元竞争；闪存加载更慢 | active 权重的字节量、读取方式和缓存命中共同决定时间 |
| 每专家 token 数 | 多请求 batching 可将同一专家的输入聚大 | 单用户 decode 常为 batch=1，每个专家只处理很少的行 | 小 GEMV/GEMM 的并行度低，调度和反量化成本更显眼 |
| 执行后端 | 可使用 GroupedGEMM、fused MoE 与多 GPU EP | 取决于移动 CPU/GPU/NPU 的算子和内存接口 | 需要适配 token 重排、量化布局及后端要求的权重准备 |

服务器 GPU 在 batch=1 时也可能受权重带宽限制；使用 CPU offload 时，也会出现加载等待。端侧更紧的容量、共享内存带宽和持续功耗预算，使这些成本更难通过扩大 batch 或增加设备来摊薄。prefill 已有多个输入位置可聚合，而交互式 decode 的后续 token 依赖先前输出，不能直接提前凑成大 batch。

具体后端还会增加额外准备阶段。[BigMoMo][bigmomo] 使用的 Hexagon NPU 就需要将 DRAM 区域映射到 NPU 地址空间，再搬入本地 VTCM 存储。因此，“权重已在 DRAM”与“矩阵单元现在能执行这个专家”仍是两个状态。后面讨论搬运流水时会继续使用这个区别。

## 2. 单 token 执行优化：量化、算子融合与专家预取

### 2.1 常驻权重的容量与读取量

先考虑完整权重常驻的方案。一个 5B total、0.8B active 的 MoE，纯 INT4 权重约为 2.5GB，这是权重容量；每步需要从中读取多少，则由实际激活路径决定。

若每份 active 权重在一次 decode 中读取一次，且没有跨 token 的缓存复用，忽略量化元数据后，INT4 权重的读取量为：

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

### 2.2 小矩阵与算子开销：MobileMoE

在单请求 decode 中，一个 expert 的 projection 可能只有 `[1, H] @ [H, I]` 这样的形状。如果 top-k、token 分派、各专家 projection 和输出合并分别调用，就要为这些小操作重复付出调度成本。MobileMoE 将它们集中到融合算子中执行。

[MobileMoE][mobilemoe] 的 0.3B–0.9B active、1.3B–5.3B total 配置，面向完整量化权重能够放入手机内存的部署。其 INT4 QAT 版本的 S/M/L 权重占用分别为 0.68、1.48、2.75GB。

[![MobileMoE 的 routed experts、shared expert 与细粒度切分结构](/assets/img/blog/moe-we-like-you/06-mobilemoe-expert-architecture.png)](/assets/img/blog/moe-we-like-you/06-mobilemoe-expert-architecture.png)

*图 1：MobileMoE 的专家结构，展示 top-k 路由、细粒度专家切分和 shared expert。*

论文的 ExecuTorch CPU 实现先按 expert id 重排 token，使同一专家的输入连续存放；再把 top-k、dispatch、gate/up projection、SwiGLU、down projection 和加权回填放进一个 MoE 算子。这样既能按专家组织矩阵乘，也减少算子之间的调用开销。小 active 参数量与这条执行路径共同构成其端侧收益。

论文在 Galaxy S25 的 4 CPU 线程、iPhone 16 Pro 的 2 CPU 线程上，以 batch=1、INT4 权重和动态 INT8 激活测试。MobileMoE-S 相对 MobileLLM-Pro 报告约 2.2–3.4× decode 加速；两者权重占用分别为 0.68GB 和 0.55GB，QAT 后 14 项平均分为 44.0 和 45.5。这组比较对应较接近的权重容量与质量，不能解释为同一模型只替换了 kernel 后的加速。

### 2.3 冷专家加载与预取：LES

模型扩大到几十 B total 后，完整权重可能超过应用可用内存。[MiniCPM-V 4.7][minicpm-v47] 的 35B-A3B 版本通过存储卸载保留完整专家池；[Lookahead Expert Scheduling（LES）][les] 用跨层路由预测提前加载下一层可能需要的专家。

[![LES 跨层专家预测、异步预取与就绪专家优先计算时间线](/assets/img/blog/moe-we-like-you/06-les-lookahead-expert-scheduling.png)](/assets/img/blog/moe-we-like-you/06-les-lookahead-expert-scheduling.png)

*图 2：LES 在当前层计算期间预测并预取下一层专家，下一层优先计算已就绪的真实路由专家，同时补载其余权重。图中 top-8、命中率与时长对应官方示例。*

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

官方文档还将连续权重布局、减少读取调用、直接写入 GPU 可见内存、逐层混合精度量化和 DSpark 投机采样接入同一流程。在 iPhone 18 Pro Max（设备总内存 12GB）上的逐阶段结果是：

| 累积启用的优化 | decode token/s |
|---|---:|
| 从存储按需加载专家的基线 | 3.38 |
| lookahead expert scheduling | 6.57 |
| 再加调度与加载优化 | 11.34 |
| 再加逐层混合精度量化 | 17.16 |
| 再加 DSpark 投机采样 | 20.51 |

速度按生成 token 数除以总 decode 时间计算。单独 lookahead 阶段约为基线的 1.94×，完整流程约为 6.07×。前者主要改变等待，布局优化改变读取效率，量化减少权重字节量，DSpark 再减少每个输出 token 分摊的解码开销。这些收益不能全部归给路由预测。

表中的 DSpark 已经引入投机解码。预取让单次 forward 尽量少等权重；投机解码进一步尝试让这次权重读取和计算产生多个有效输出。BigMoMo 与 S²-MoE 也利用这种多 token 验证窗口，但窗口扩大之后，MoE 会遇到新的专家访问问题。

## 3. 投机解码：权重复用收益与专家并集膨胀

### 3.1 多 token 验证怎样摊薄权重成本

投机解码先用较便宜的 draft 生成候选，再由目标模型并行验证，最终提交其中可接受的输出。draft 提供了一组暂定的未来 token，使目标模型能将多个位置放进一次 forward，而不必等待每个位置都完成一次独立的 AR step。

如果其中 $m$ 个位置选择同一专家，就可以将其输入组织成 `[m, H]`，让同一份权重处理更多行。对于带宽受限、计算单元仍有余量的执行路径，这能提高权重复用。在专家卸载场景中，多个位置也有机会分摊一次专家加载，并提供更多可用于隐藏搬运的计算。

单请求下，设一轮最终提交的平均输出 token 数为 $\bar A$，标准 AR 每 token 耗时为 $T_{\mathrm{AR}}$。把 draft、verify 与额外调度/采样的时间分别记为 $T_{\mathrm{draft}}$、$T_{\mathrm{verify}}$、$T_{\mathrm{extra}}$，则加速比可估为：

$$
\mathrm{speedup}\approx
\frac{\bar A\,T_{\mathrm{AR}}}
{T_{\mathrm{draft}}+T_{\mathrm{verify}}+T_{\mathrm{extra}}}.
$$

分子是实际提交的输出，分母包含 draft、目标验证和额外处理。投机长度增加，只有在有效输出的增长抵得上这些新增成本时才有收益。同样的接受长度不保证相同加速比，verify 访问的权重数量也会改变分母。

### 3.2 验证 token 数、专家并集与搬运量

对于 Dense FFN，多个验证位置使用相同的权重矩阵；对于 MoE，不同位置还可能选中不同专家。因此，扩大验证集合既增加 token 计算，也可能增加需要访问的唯一权重。这里新增的是 MoE 路由带来的专家并集成本；Dense 的验证时间同样会随计算量和 attention 开销变化。

设第 $\ell$ 层有 $E_\ell$ 个 routed experts，每个位置选 top-$k$，验证位置集合为 $\mathcal V$。该层实际访问的 routed expert 集合为：

$$
\mathcal U_\ell(\mathcal V)
=\bigcup_{v\in\mathcal V}\mathcal E_\ell(v),\qquad
k\leq|\mathcal U_\ell|\leq\min(E_\ell,k|\mathcal V|).
$$

用一层 8 个专家、top-2、4 个验证位置的教学例子说明：

| 验证位置 | 专家重叠较多 | 专家互不重叠 |
|---|---|---|
| v0 | E0、E1 | E0、E1 |
| v1 | E0、E2 | E2、E3 |
| v2 | E1、E2 | E4、E5 |
| v3 | E0、E2 | E6、E7 |
| 唯一专家数 | 3 | 8 |

两列都是 8 次 token–expert 调用，专家矩阵乘的总行数相同，涉及的唯一权重却相差很大。若同一专家的输入能聚合计算、权重在这一轮中只读取一次，左侧只需读取 3 份 routed expert 权重，右侧需要 8 份。shared expert 是固定的公共路径，另行计入。

专家并集也不直接等于卸载流量。设当前层在目标内存层级（例如 GPU 显存）中已驻留的专家集合为 $\mathcal R_\ell$，在缺失专家每轮只换入一次的假设下，换入量约为：

$$
B_{\mathrm{load},\ell}
\approx\sum_{e\in\mathcal U_\ell\setminus\mathcal R_\ell}W_{\ell,e},
$$

其中 $W_{\ell,e}$ 是该专家的权重字节量。相同大小的并集，因专家尺寸、精度和驻留状态不同，也可能有不同搬运成本。全 GPU 常驻时，这部分 host-to-device 换入可以为零，但专家权重仍要从 HBM 进入计算单元。跨层统计使用 `(layer, expert_id)`，每层的 E0 是不同权重。

### 3.3 Batch 与专家覆盖：MoESD

专家并集是否明显增加，还取决于 AR 原本已覆盖多少专家。[MoESD][moesd] 用 target efficiency 刻画目标验证的成本：$T_{\mathrm{target}}(B,1)/T_{\mathrm{target}}(B,m)$，即同一请求 batch 大小 $B$ 下，单位置 AR forward 耗时与每请求 $m$ 个位置的 verify forward 耗时之比。batch 大小会改变这两次 forward 之间新增的专家读取。

在均匀、跨位置独立的路由假设下，$n$ 个输入位置访问的期望专家数为：

$$
\mathbb E[|\mathcal U(n)|]
=E\left[1-\left(1-\frac{k}{E}\right)^n\right].
$$

以 $E=64,k=8$、每请求验证 4 个位置为例，可得下面的理论估算：

| 请求 batch 大小 | AR 输入位置数 | AR 期望专家数 | verify 输入位置数 | verify 期望专家数 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 8.0 | 4 | 26.5 |
| 32 | 32 | 63.1 | 128 | 64.0 |

第一行中，验证需要访问的专家明显增加；第二行中，AR 本来就接近覆盖整个专家池，验证只增加很少的唯一权重。若此时每个专家的矩阵乘仍有计算余量，投机验证就能增加权重复用。这是 MoESD 讨论的中等 batch 收益区间。继续增大 batch、进入 compute-bound 后，额外计算又会限制收益。

表中均匀独立路由用于解释饱和趋势，不是实际模型测量，也不意味着 batch=32 是通用阈值。真实 token 的路由相关性、专家粒度与 kernel 形状会改变这个区间。

## 4. MoE 投机解码的优化方法

针对这些新增成本，优化分别作用于验证前的候选选择、验证器的专家集合、训练时的路由模式，以及权重准备与执行调度。

### 4.1 减少低收益验证：Cascade 与 S²-MoE

[Cascade][cascade] 根据运行时收益决定是否投机，以及投机多长。它将每轮有效输出 token 数与相对 AR 的迭代耗时组成 utility，短暂测试若干长度，再在后续一段迭代中使用较好的长度；utility 低于 1 时回退到 AR，并通过 back-off 降低反复试错的成本。它保留目标模型的自然路由，主要避免把候选扩展到验证成本已经超过收益的区域。

[S²-MoE（S2-MoE）][s2-moe] 同样把候选扩展收益与新增专家成本联系起来，并通过路由 bias 促进跨候选复用。它还涉及 self-draft 和上下文对齐，这些细节留到投机解码系列展开。

### 4.2 减少候选引入的新专家：EcoSpec

[EcoSpec][ecospec] 进一步区分“验证多少候选”和“验证哪些候选”。它在 draft tree 已生成、目标模型尚未验证时，用训练好的轻量 predictor 估计各候选路径会触发的专家。选择候选时，同时考虑路径概率和相对已选集合新增的专家数量。

[![EcoSpec 的专家预测与候选路径选择流程](/assets/img/blog/moe-we-like-you/06-ecospec-cost-aware-selection.png)](/assets/img/blog/moe-we-like-you/06-ecospec-cost-aware-selection.png)

*图 3：EcoSpec 在 draft 与目标验证之间加入专家预测和候选选择。Global Expert Buffer 记录已选候选覆盖的预测专家集合。*

这里的 Global Expert Buffer 是选择过程维护的集合记录，不代表这些权重已经在 GPU cache 中。一个新节点还需要覆盖它的祖先路径，边际专家成本因此按整条路径计算。选入一条路径后，共享这些专家的其它候选会变得更便宜。最终 verify 仍使用目标模型的真实 router 与标准验证规则，预测只影响候选集合及执行成本。

### 4.3 限制验证专家集合：MoE-Spec、AcceptMoE

另一条路线保留较大的候选集合，直接限制 verify 可以使用哪些专家。

[MoE-Spec][moe-spec] 在每层聚合验证位置的 router probability，按专家重要性排序，再保留固定预算的 shortlist。原 top-k 落在 shortlist 外时，可以截断缺失专家的贡献，或从 shortlist 中选择替代专家。因此，即使候选树继续扩大，每层可用的专家总数也被预算限制。

[![MoE-Spec 的专家排序、预算限制与覆盖处理](/assets/img/blog/moe-we-like-you/06-moe-spec-expert-budget.png)](/assets/img/blog/moe-we-like-you/06-moe-spec-expert-budget.png)

*图 4：MoE-Spec 对验证专家排序并限制预算，再对缺失专家使用截断或替代。*

[AcceptMoE][acceptmoe] 针对固定预算与等权聚合继续调整：用离线估计的最终提交概率对各 draft 位置的专家需求加权，降低深层和旁支对专家选择的影响；再根据需求分布的集中程度，以熵的指数估计应保留的专家数量，使集合随当前验证块变化。使用卸载时，它还会结合驻留状态，在重路由约束下裁剪低收益的非驻留专家。

这两类方法改变了目标验证器可用的专家，输出概率也随之变化。接受规则仍然存在，并不能恢复被修改前的目标分布。它们的收益应理解为专家预算与生成质量之间的取舍，区别于只调整候选、保持目标路由的优化。

### 4.4 训练时增加跨 token 专家复用：Coactivation

[Expert Coactivation][coactivation] 将优化提前到模型训练阶段，让相邻位置更倾向于使用重叠的专家。论文结合 global load-balancing、shared experts、相邻 router 分布的一致性损失，以及 autoregressive expert selection。global balance 允许局部窗口集中使用少量专家，同时在更大的训练 batch 上约束总体使用率。

其中 autoregressive expert selection 直接限制相邻位置的专家变化：后一个位置沿用前一个位置的部分专家，再补入少量新专家。在长度为 $m$ 的相邻 token 窗口中，若首位置选择 $k$ 个 routed experts，后续每个位置最多引入 $r$ 个新专家，则并集有上界：

$$
|\mathcal U(m)|\leq\min(E,k+(m-1)r).
$$

每个位置仍执行 $k$ 个 routed experts，受控的是窗口内的唯一专家数，而非 token–expert 调用总数。代价是路由自由度改变，需要配套训练并检查模型质量；这是一种训练与推理共同设计，不能直接当作现成 checkpoint 的缓存配置。

### 4.5 将验证请求排成搬运流水：BigMoMo

对于使用闪存卸载的端侧 MoE，减少验证请求后，剩余专家仍可能到达得太晚。搬运调度需要把它们尽早变成可执行工作，并与当前计算重叠。

[BigMoMo][bigmomo] 用投机解码将多个候选放到同一轮目标验证中，因而同时暴露一批专家请求。各专家权重到达计算单元的时间不同，系统将请求筛选、闪存布局与 NPU 执行调度接成一条流水。

[![BigMoMo 的请求筛选、闪存组织与专家执行系统总览](/assets/img/blog/moe-we-like-you/06-bigmomo-system-overview.png)](/assets/img/blog/moe-we-like-you/06-bigmomo-system-overview.png)

*图 5：BigMoMo 系统总览。上半部分从 draft 节点与专家筛选推进到目标验证，下半部分对应闪存布局和 NPU 执行调度。*

它面向 Snapdragon 的 Hexagon NPU，将权重准备过程细分为：

```text
Flash → DRAM → NPU 可访问的映射缓冲区 → VTCM → HMX 矩阵计算
```

VTCM 是 NPU 的本地存储，HMX 是其矩阵计算单元。一个 expert 已经在 DRAM 里，仍可能在等待映射或搬入 VTCM。若每层都等全部专家就绪再启动计算，较早到达的权重也只能闲置。

BigMoMo 先减少进入流水的请求：根据接受收益筛掉低价值 draft 节点，再结合 routing impact 与各存储层级的搬运成本裁剪部分专家激活。专家被裁掉后，其路由权重会分给保留的专家；这一步近似了目标模型计算，质量需要随裁剪预算一起评估。

剩下的请求还要解决两个问题。其一是闪存上的碎片读取：经常一起加载的专家若分散在文件中，就需要多次读取。系统根据近期共同加载记录调整专家布局，尽量将这类请求转成连续读取。其二是到达时间不同：调度器将已就绪专家组成计算组，在 VTCM 容量允许的范围内，让当前计算覆盖下一组权重的准备；两个 VTCM 权重缓冲区交替加载与计算。

[![BigMoMo 的 VTCM 双缓冲与跨层级搬运计算流水](/assets/img/blog/moe-we-like-you/06-bigmomo-movement-pipeline.png)](/assets/img/blog/moe-we-like-you/06-bigmomo-movement-pipeline.png)

*图 6：左侧是 VTCM 双缓冲，右侧比较按需执行、固定双缓冲与跨层级调度。粉色区域中的候选计算组用于覆盖后续权重准备，减少 HMX 空闲区间。*

因此，流水中的优化包含减少请求、改善存储布局和隐藏剩余等待。即使两轮验证拥有同样大小的专家工作集，命中 DRAM 的比例、闪存布局和 NPU 准备状态不同，也会产生不同延迟。

论文基于 llama.cpp 的 Hexagon 后端，以四种 Q4_0 MoE、五类任务评估，报告相对按需卸载 AR 基线平均 4.83× 加速。这个基线本身需要从闪存加载专家，与 MobileMoE 的内存常驻 CPU 测试、LES 的 Apple 设备测试分别对应不同的执行路径。

## 5. 各类优化的作用与计算语义

| 工作 | 主要改变什么 | 目标模型与路由的变化 |
|---|---|---|
| MobileMoE | active 配置、量化与融合算子 | 训练得到面向端侧的新模型 |
| LES | 专家预取、读取布局与就绪调度 | 调度保留真实路由；量化有独立的数值影响 |
| BigMoMo | 请求筛选、闪存组织与多级搬运 | 专家裁剪会近似目标计算 |
| MoESD | 分析 batch、专家覆盖与 target efficiency | 分析本身不改模型 |
| Cascade | 投机开关与长度 | 保持自然路由与目标验证 |
| EcoSpec | 预测专家成本并选择候选路径 | 训练额外 predictor，目标模型与验证规则保留 |
| MoE-Spec | 每层验证专家的固定预算 | 截断或替代会改变目标计算 |
| AcceptMoE | 提交概率加权、自适应专家集合与驻留裁剪 | 限制 eligible experts，改变目标路由 |
| Coactivation | 训练目标与相邻位置的专家选择规则 | 需要训练或适配，形成新的路由行为 |
| S²-MoE | 候选扩展、专家复用与 self-draft | 完整配置中的 routing bias 会改变目标路由 |

端侧的读取、小算子和冷专家加载解释了为什么需要量化、融合与预取。投机解码在此基础上扩大一次 forward 的有效工作量，也可能扩大专家并集。候选优化控制哪些位置值得验证，专家预算控制验证器访问哪些权重，训练时共激活改变跨 token 的复用模式，搬运流水则缩短剩余权重的等待；它们分别作用于这条执行链的不同环节。

## 系列导航

上一篇：[第五篇：100B 训练配置、并行策略与显存推演]({% post_url 2026-07-21-moe-100b-parallel-memory %})

下一篇：[第七篇：RL 训推不一致与路由对齐问题]({% post_url 2026-07-31-moe-rl-train-serving-mismatch %})

## 参考资料

- [MobileMoE: Scaling On-Device Mixture of Experts][mobilemoe]
- [MiniCPM-V 4.7][minicpm-v47]；[LES][les]
- [BigMoMo: Efficient Inference of Large-Scale MoE with Speculative Decoding on Mobile Devices][bigmomo]
- [MoESD: Unveil Speculative Decoding’s Potential for Accelerating Sparse MoE][moesd]
- [Utility-Driven Speculative Decoding for Mixture-of-Experts（Cascade）][cascade]
- [Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts（EcoSpec）][ecospec]
- [MoE-Spec: Expert Budgeting for Efficient Speculative Decoding][moe-spec]
- [AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets for Efficient MoE Speculative Decoding][acceptmoe]
- [Efficient Mixture-of-Experts with Speculative Decoding via Expert Coactivation][coactivation]
- [S²-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices][s2-moe]；[官方代码][s2-moe-code]

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

[moesd]: https://arxiv.org/abs/2505.19645
[cascade]: https://arxiv.org/abs/2506.20675
[ecospec]: https://arxiv.org/abs/2607.12696
[moe-spec]: https://arxiv.org/abs/2602.16052
[acceptmoe]: https://arxiv.org/abs/2608.02989
[coactivation]: https://arxiv.org/abs/2609.22471
[ecospec-figure]: https://arxiv.org/html/2607.12696v1#S1.F2
[moe-spec-figure]: https://arxiv.org/html/2602.16052v1#S1.F1
