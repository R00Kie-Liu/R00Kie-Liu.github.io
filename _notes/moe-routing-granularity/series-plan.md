# MoE我们喜欢你：路由粒度与专家复用 番外专题提纲

检索截止日期：2026 年 11 月 9 日。

这个番外专题研究一组 token 怎样共同使用专家：共同选择候选池、限制局部专家并集、复用相邻位置和去噪步骤的路由结构，以及按语义组织专家。沿用“MoE我们喜欢你”系列名称和 `moe-we-like-you` 导航分组；前十篇之后以“番外一、番外二……”标注专题文章。第六篇保留端侧和投机解码的整体执行路径，番外展开路由算法、训练目标和专家集合的设计。

## 系列结构

| 番外篇次 | 标题 | 主要问题 | 重点工作 |
|---|---|---|---|
| 一 | Block、Chunk 与投机树的专家集合设计 | 相同 token-level top-k 为何有不同工作集？选池、训练局部性与候选选择分别改变什么？ | 以统一成本模型比较块路由、局部性训练、投机树和语义分组 |
| 二 | dLLM 的 Block 路由与动态专家池 | 专家池由谁选、选多大、何时重选？固定容量与 top-p 自适应各有什么代价？ | LLaDA2.2、DES、dMoE |
| 三 | 投机树验证的专家并集与候选选择 | 多节点共同 verify 的成本怎样计算？剪候选与改 verifier 路由有什么区别？ | S²-MoE、EcoSpec、EVICT、SpecFit、MoE-Spec、AcceptMoE；DraftExpert 作端侧对照 |
| 四 | 去噪步骤间的专家与计算复用 | 同一 block 反复前向时，哪些结构能复用，哪些值必须刷新？ | TEAM、Epoch |
| 五 | 端侧 Chunk 稀疏性与专家缓存 | 怎样让一个局部窗口的专家工作集适配内存？训练约束与缓存策略怎样配合？ | BlockFFN、Local Routing Consistency、Cache-Aware Joint Router Adaptation |
| 六 | 相邻 Token 的路由约束与联合训练 | 怎样增加专家复用，又保留位置之间的计算差异？软一致性与硬选择分别保证什么？ | StickyMoE、Expert Coactivation、Cacheable by Design 的负面结果 |
| 七 | Sentence 路由与语义 Chunk 设计 | 路由组应该由位置、句子还是语义决定？专家组、计算预算与生成顺序有什么区别？ | MoCE、DCDM；以 SeqTopK 和 Routes of Reasoning 辨析不同的 sequence/sentence-level 含义 |

番外一已整理为可发布 v1：`_posts/2026-11-07-moe-routing-granularity-expert-reuse.md`，作为路由专题的综述入口，导航排序接在原第十篇之后（`series_order: 11`）。后续文章按各自问题深入算法、实现与实验，不重复入口篇的分类和基础定义。

## 第一篇 Block、Chunk 与投机树的专家集合设计

用两个 top-2、四位置的例子说明：token–expert 调用都为 8，unique experts 可以是 2，也可以是 8。定义每层、每次前向的专家并集，再分别引入块级候选池、驻留集合与相邻位置新增专家数。

LLaDA2.2-flash 提供一个具体实现：32 个 token、256 个 routed experts、48 个候选专家、每位置 top-8。围绕执行场景与专家集合成本，比较 DES、dMoE 的选池方式，再连接局部性训练、跨去噪步骤复用、投机树候选选择与语义分组。正文的 22 项研究各有一张原文配图，另有一张原创投机树示意图及可逐步检查的数值例子。

## 第二篇 dLLM 的 Block 路由与动态专家池

以相同的两阶段表示比较三种方法：先选块级候选池，再做 token-level 路由。

- **LLaDA2.2**：max 聚合的 token racing 与固定容量；selection bias 与实际混合权重的区别；continued pretraining 对路由变化的适应。
- **DES**：DES-Seq 用各位置少量高分专家的并集形成 coreset；DES-Vote 对局部 top-k 之后的分数投票，再按预算选择 coreset。训练无需更新参数，推理路径会改变。
- **dMoE**：聚合 token-level 专家分数，以 top-p 决定动态池大小；在同一路由前向下训练并自蒸馏。比较固定容量与累计质量阈值对分布集中程度的响应。

具体推导 max 与 sum 聚合的区别：某个专家只被一个位置强烈需要，另一个专家被多个位置中等程度需要，两种聚合怎样排序。比较路由覆盖、unique expert 数、选择开销与任务质量，不跨硬件排列论文加速比。

## 第三篇 投机树验证的专家并集与候选选择

将一个 block 的位置集合推广为祖先闭合的候选树。验证支付所有保留节点的专家需求，输出通常只提交一条路径上的部分 token；同一深度的不同分支还可能使用互不重叠的专家。比较时同时记录节点数、逐层专家并集、实际搬运量和最终提交 token 数。

先解释候选的边际专家成本，再比较四类方法：EVICT 按 drafter 累计分数排序并选择收益/成本最好的验证规模；EcoSpec、SpecFit 显式考虑不同候选的专家重叠；S²-MoE 联合扩树与 reuse-aware gating；MoE-Spec、AcceptMoE 直接限制 verifier 的专家资格。DraftExpert 连接固定占用 self-drafter、扩张感知截断和专家预取。

S²-MoE 的 preferred-set `cap` 只限制被加 bias 的专家集合，池外专家仍可选，不能据此推出专家并集的硬上界。SpecFit 当前测量使用计时之外获取的 route atlas，在线 predictor 尚待补齐。EVICT 的节点数成本表是平均 profile，不能表述为每轮显式预测专家 ID 并集。

将相邻位置的复用约束推广到树时，需分别计算单路径与整树的上界：父子之间至多引入 r 个专家，并不保证所有兄弟节点共享同一专家池。详细资料与推导保存在 [投机树验证研究笔记](speculative-tree-expert-union.md)。

## 第四篇 去噪步骤间的专家与计算复用

同时使用位置索引与去噪迭代索引，区分相邻 token 的相关性与同一位置在不同迭代中的相关性。

**TEAM** 的 decoded-token 缓存、masked-token 分类和候选探索，分别改变哪些计算和接受机会？缓存专家输出与仅复用专家池，有哪些不同的数据依赖？

**Epoch** 把 diffusion block 作为规划单位。重点解释 Expert Atlas 的 active support、每步新算的 gate values、Liveness-Shard Parallelism 与 FreshLane Dispatch。被采样决定不再需要的 logits，不代表对应 token 可以从双向 attention 状态里删除。support 截断和缓存值有各自的近似边界；不能把所有优化统称为无损缓存。

## 第五篇 端侧 Chunk 稀疏性与专家缓存

以 **BlockFFN** 为中心，同时讲专家和 router 的设计：细粒度非门控 MLP expert、ReLU 与 RMSNorm router、activation locality loss、chunk sparsification loss，以及按专家合并多位置读取的执行路径。它直接训练 chunk-level activation sparsity，正好对应端侧小工作集的需求。

**Not All Models Suit Expert Offloading** 提供 SRP、SCH 两种局部路由一致性指标。解释固定专家集合覆盖一段 token，与有限缓存中的最优命中率是两个问题。使用未来路由信息的指标属于分析参照，不能当作可直接部署的在线策略。

**Cache-Aware Joint Router Adaptation** 学习跨 token 保留优先级，并可叠加因果的提前加载。保留 native top-k 规则不等于冻结 native router 参数；辅助缓存 router 也不等于将真实路由替换为常驻专家。

## 第六篇 相邻 Token 的路由约束与联合训练

**StickyMoE** 用相邻 softmax gate 分布的平方距离惩罚路由切换，研究预训练时让 router 与专家共同适应。需要写明其小模型实验范围。

**Expert Coactivation** 联合 global load balancing、shared experts、相邻 router 一致性和 autoregressive expert selection。若每步至多引入 r 个新专家，可以推导长度 m 的连续窗口满足专家并集上界 `min(E, k + (m - 1)r)`；这比单独约束 soft probabilities 更直接。

**Cacheable by Design?** 必须纳入对照。它在多域 137M 模型、联合缓存 miss 和困惑度的预注册标准下，没有达到严格的质量与局部性目标。讨论数据域、规模、损失强度及容许质量损失为何影响结论。该结果不能外推为所有局部性训练都无效，也不能被省略后写成局部性几乎免费。

## 第七篇 Sentence 路由与语义 Chunk 设计

**MoCE（EMNLP 2025）** 用独立 encoder 的 sequence embedding 做聚类和专家组分配，再在组内执行 token-level adapter expert 路由。把语义专门化与权重工作集联系起来，同时交代独立 encoder 的成本和 adapter expert 的实验对象。

**DCDM（2026 年 5 月）** 用学习到的语义 cluster 替代固定位置块，chunk 可不连续；chunk-causal attention 决定生成分组。其 DCDM-MoE 变体另有 token-level MoE。语义生成 chunk 与 FFN 专家集合没有自动一一对应关系。

**SeqTopK（2026 年 3 月修订）** 将序列的总 token–expert 调用预算重新分配给不同位置，主要目标是动态分配计算；它与限制 unique expert pool 的设计并列比较。AR 推理的在线实现必须单独解释。

**Routes of Reasoning（2026 年 11 月 4 日）** 是近期扩展：按句子标注推理行为，分析并干预 router 分数。它能讨论 expert 与 planning、reflection 等行为的关联，但没有将整句强制送入同一专家池。

## 扩展选题 路由相关性与专家部署

**Expert Coupling（2026 年 10 月）** 利用同一位置内的专家共选和跨层条件相关性，做 correlated placement、去重 dispatch 与 token shuffling。它不修改模型的路由决策，适合与“通过训练制造相关性”的方法对照。本文的 token shuffling 有 TP/SP 与 EP 布局前提，不能直接当作单设备端侧结果。

正篇第六篇保留端侧执行链及投机优化的整体介绍；番外第三篇深入候选树、专家并集与路由约束，分别分析可用信息和计算语义。LES 与搬运流水在相关位置做链接，通用投机采样算法留给投机解码系列。

## 文献与版本

下表记录本轮采用的原文版本。日期是对应版本日期；会议年份不替代首次发布或修订日期。

| 工作 | 版本或日期 | 原文及项目 |
|---|---|---|
| LLaDA2.2 | 2026 年 7 月发布；本轮读取官方报告第 2.2 节与模型代码 | [报告](https://github.com/inclusionAI/LLaDA2.X/blob/main/LLaDA2_2_tech_report.pdf)、[固定版本模型代码](https://huggingface.co/inclusionAI/LLaDA2.2-flash/blob/9b78da3f9149f1e239fb6d15f20a326fde574b9f/modeling_llada2_moe.py)、[配置](https://huggingface.co/inclusionAI/LLaDA2.2-flash/blob/9b78da3f9149f1e239fb6d15f20a326fde574b9f/config.json) |
| DES | arXiv v2，2026-11-03；v1 为 2026 年 1 月 | [论文](https://arxiv.org/html/2602.00879v2) |
| dMoE | arXiv v2，2026-06-20 | [论文](https://arxiv.org/html/2605.30876v2)、[代码](https://github.com/fscdc/dMoE/tree/master) |
| TEAM | arXiv v3，2026-08-03；ICML 2026 | [论文](https://arxiv.org/html/2602.08404v3) |
| Epoch | arXiv v1，2026-09-09 | [论文](https://arxiv.org/html/2609.09748v1) |
| BlockFFN | arXiv v2，2025-07-30；COLM 2025 | [论文](https://arxiv.org/html/2507.08771v2)、[代码](https://github.com/thunlp/BlockFFN) |
| Local Routing Consistency | arXiv v4，2026-02-28；ICLR 2026 | [论文](https://arxiv.org/html/2505.16056v4) |
| Cache-Aware Joint Router Adaptation | arXiv v2，2026-09-10 | [论文](https://arxiv.org/html/2609.04895v2) |
| Sticky Routing / StickyMoE | arXiv v1；原文标注 2026-06-12，编号为 2607.08780 | [论文](https://arxiv.org/html/2607.08780v1) |
| Expert Coactivation | arXiv v1，2026-09-18 | [论文](https://arxiv.org/html/2609.22471v1) |
| Cacheable by Design? | arXiv v1，2026-08-18 | [论文](https://arxiv.org/html/2608.18261v1) |
| MoCE | EMNLP 2025 | [论文](https://aclanthology.org/2025.emnlp-main.718/) |
| Dynamic Chunking / DCDM | arXiv v1，2026-05-15 | [论文](https://arxiv.org/html/2605.15676v1) |
| SeqTopK | arXiv v2，2026-03-27；v1 为 2025 年 11 月 | [论文](https://arxiv.org/html/2511.06494v2)、[代码](https://github.com/Y-Research-SBU/SeqTopK) |
| Routes of Reasoning | arXiv v1，2026-11-04 | [论文](https://arxiv.org/html/2611.05919v1) |
| Expert Coupling | arXiv v1，2026-10-07 | [论文](https://arxiv.org/html/2610.09372v1)、[作者说明](https://www.zyphra.com/our-work/expert-coupling-in-moe-pretraining) |
| S²-MoE | arXiv v2，2026-08-19 | [论文](https://arxiv.org/html/2608.15018v2)、[代码](https://github.com/angerybob/S2-MoE/tree/orin) |
| EcoSpec | arXiv v1，2026-07-14 | [论文](https://arxiv.org/html/2607.12696v1) |
| EVICT | arXiv v2，2026-09-05；Findings of EMNLP 2026 | [论文](https://arxiv.org/html/2605.00342v2)、[会议版本](https://aclanthology.org/2026.findings-emnlp.380/) |
| SpecFit | arXiv v1，2026-10-11 | [论文](https://arxiv.org/html/2610.15417v1) |
| MoE-Spec | arXiv v1，2026-02-17 | [论文](https://arxiv.org/html/2602.16052v1) |
| AcceptMoE | arXiv v1，2026-08-04 | [论文](https://arxiv.org/html/2608.02989v1) |
| DraftExpert | arXiv v1，2026-07-27 | [论文](https://arxiv.org/html/2607.24434v1) |

## 写作与证据要求

每篇至少有一个可逐步复现的路由例子或推导，配图解释实际的数据依赖与候选集合。涉及论文架构图时使用原图，图注保留技术内容，来源放在正文引用或文末。

统一记录被分组的对象、候选专家池的生命周期、token-level 选择规则、专家实际计算的对象、训练或适配需求。量化收益时注明硬件、基线、块长与质量变化；区分显存容量、权重流量、计算量和接受 token 数。

尚待后续逐篇细读的部分是各方法的完整实验表、代码执行路径与硬件测量条件。本提纲使用已核对的机制建立选题，不据此替代复现实验或作跨论文性能排序。
