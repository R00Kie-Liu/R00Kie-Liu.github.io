# 路由粒度综述 v1 审阅记录

日期：2026 年 11 月 9 日。

成稿：`_posts/2026-11-07-moe-routing-granularity-expert-reuse.md`。

系列归属：标题改为“MoE我们喜欢你（番外一）：Block、Chunk 与投机树的专家集合设计”，沿用 `series: moe-we-like-you`，以 `series_order: 11` 接在正篇第十篇之后。发布日期按用户要求设为 2026 年 11 月 7 日，文件名与日期元数据同步，正式 URL 为 `/2026/11/07/moe-routing-granularity-expert-reuse.html`。

采用仓库中的 `.agents/skills/polish-technical-blog-zh/SKILL.md`，分别执行起草、润色与审阅流程，并参照 `ai-tells-zh.md`、`anti-defensive-writing.md` 和 `reasoning-blog-style.md`。该 skill 的统一入口已包含审阅能力。

## 叙事与范围

文章先介绍常见的 token-level routing 及其与单请求 AR 解码的对应关系，再梳理端侧卸载、投机验证、block diffusion 与语义分组的不同动机。数值例子移到成本定义一节，后续依次分析共享候选池、去噪迭代复用、局部性训练、投机树和语义分组。每部分说明被改变的集合、时间范围和计算方式，以场景需求联系具体论文。

引言修订保留两个条件：端侧专家加载在普通 AR 中同样存在；语义分组不自动保证专家并集缩小。没有将组级路由的需求写成 diffusion 特有的问题，也没有将所有方案归为强制共享最终专家 ID。

模型范围修订：开头用场景表说明各自的执行单元、问题和预期收益，完整论文分类表移到第 7 节，避免在成本定义之前堆积论文名称。第 2、3 节明确针对 block diffusion；第 4、5 节明确为 AR MoE；第 6 节以子标题分开 MoCE、DCDM 和 SeqTopK。DES、dMoE、TEAM、Epoch 的研究对象补到首次介绍处，并区分 diffusion 生成块与 AR verification block。

本轮根据人工审阅修改正文：压缩开头与结尾的重复分类；第 4.2 节按软一致性损失、top-k 反例、硬选择上界推进；第 4.3 节集中讨论局部性与均衡的取舍；缓存适配移到驻留集合和加载量之后，并补明单次前向与跨 AR 步骤的统计区别。第 6 节先区分专家组、生成块和调用预算，再介绍对应工作。删去空泛总结，解释 coreset、hot/cold 和支持集，去掉未解释的 anchor、fresh 等用语。

补充 AR 动机：第 4 节在方法之前解释端侧 decode 的跨步缓存复用，以及 prefill、投机验证的块内权重复用和矩阵乘效率；标明普通 decode 中未来位置不可见，因此需要训练局部性或沿历史约束选择。第一张表新增“目的与预期收益”列，并把 AR prefill 单列，区分加速目标与语义专门化、泛化及生成分组目标。

最后检阅与配图：合并 dMoE 的重复训练说明，删去投机验证开头重复的 AR/diffusion 对照，用边际成本变化替换 coverage function 的跳跃术语；第 3 节分成 TEAM 与 Epoch 两个小节，第 5.2 节标题同时覆盖 S²-MoE 和 DraftExpert。第 6 节从执行窗口转向语义分组，补清与前文的关系。结合原图补充 AcceptMoE 的根位置初始集合、按其余需求的熵指数追加专家及驻留裁剪；保留裁剪可以移除初始集合中非驻留专家的条件。配图说明 S²-MoE 的 preferred set 仍可被池外专家突破。

Multi-View Latent MoE 未纳入本篇。投机解码部分聚焦候选与专家需求，不展开通用采样接受算法。旧系列十篇正文没有在这次修改中变动。

## 技术审阅结果

| 检查项 | 成稿处理 |
|---|---|
| 引言与场景动机 | 先介绍 token-level routing，再区分端侧卸载、投机验证、block diffusion 和语义分组；论文放在需求与成本定义之后 |
| 模型范围与 Block 含义 | LLaDA2.2、DES、dMoE、TEAM、Epoch 归入 block diffusion；DCDM 单列为语义 diffusion 分组；AR 局部性与投机验证分别标明 |
| 专家数量与计算量 | 区分逻辑调用数、独立专家数及实际搬运，标明驻留集合和单次加载假设；跨 AR 步骤需考虑驻留集合变化与重复加载 |
| LLaDA2.2 | 依据固定版本代码及配置核对 256/32/48/8；区分选择偏置和输出混合权重 |
| DES 与 dMoE | 区分固定 coreset、top-p 动态选池和训练适配；不将无需训练写成分布保持 |
| TEAM | 明确新解码 token 额外前向后复用 KV，避免误读成直接缓存所有 MoE 输出 |
| Epoch | 区分 support mask 与每轮重算的路由结果，保留 support 截断及缓存过时的影响 |
| BlockFFN | 说明 ReLU/RMSNorm 与非门控 expert；并集损失按可微代理解释；补充整块补算后 mask 的内核工作量 |
| Coactivation | 上界限定在选择规则生效的窗口内，保留窗口重置条件 |
| 软一致性 | 用接近的概率向量但不同 top-1 的反例说明不保证专家集合一致 |
| SimBal 补充 | 按用户提供的 v1，简述均衡训练与输入相似关系的动机、近正交初始化和 Gram 正则；与相邻 token 约束及并集预算区分 |
| 局部性训练证据 | 保留小模型与多域 137M 负面结果的实验范围；依据原文补明其联合标准包含困惑度增幅不超过 1% |
| 投机树 | 区分单路径与整树并集；新增深节点计入缺失祖先；同分排序保持祖先优先 |
| S²-MoE | preferred-set cap 不解释为并集硬上限；完整配置的 routing bias 会改变 target |
| EVICT 与 EcoSpec | 区分节点数平均成本 profile 和按路径预测专家 ID 的选择 |
| SpecFit | 明确 route atlas 在请求计时之外获取，未将结果写成包含在线预测器的完整部署收益 |
| MoE-Spec 与 AcceptMoE | 说明 eligible expert 变化会改变 verifier 分数；AcceptMoE 区分根位置初始集合、追加专家与驻留裁剪 |
| MoCE、DCDM、SeqTopK | 分开解释专家组、生成分组与总调用预算；标明 adapter 和在线因果信息条件 |

## 数学与引用核对

已核对 top-2 四位置示例的 8 次调用与 2/8 个专家，TLS/CLS 数值，Max/Sum 的不同排序，以及链与树的新增专家上界。LLaDA2.2 参数与已下载的官方配置一致。

正文引用 22 项研究，包含模型代码与配置共 24 个引用地址；均返回 HTTP 200。链接可访问性与机制核对分别执行，未以链接状态代替内容审阅。

机械审稿执行两轮，最终没有命中内置表达模式。人工审阅另行修正技术对象、排序条件和 SpecFit 段落中的写作过程用语；结果不作为 AI 检测概率。

本轮结构修改后再次执行两轮机械审稿，均无命中。与修改前快照逐项比较，frontmatter、24 个引用地址、全部展示公式及 3 张配图路径保持一致；正文缩短约 1,300 字符。核对结果保存在 `logic-polish-preservation-check.json`。

最后图文修订继续保持 frontmatter、24 个引用地址和全部展示公式一致。新增 9 张论文原图，原有 3 张图保留，图号连续为 1–12；7 个直接下载的图片/SVG 与原始文件逐字节一致，另 2 张从论文 PDF 按矩形裁剪渲染。技术核对包含 Coactivation 图中的 12 次调用、9/5 个独立专家，以及 S²-MoE 图中的池外 E5。最后再拆开 AcceptMoE 的长句，明确“计算熵，再取指数”的过程，两轮机械审稿均无命中。记录见 `final-illustrated-preservation-check.json`。

## 配图记录

| 文件 | 内容与取得方式 |
|---|---|
| `dmoe-block-expert-pool.png` | dMoE v2 的 `overview.png` 原始文件，1282 × 619；正文对应分数聚合、top-p 选池与训练 |
| `blockffn-chunk-kernel.png` | BlockFFN v2 PDF 第 6 页 Figure 2；裁剪矩形为 PDF 坐标 `(104,74,510,227)`，4.5 倍渲染 |
| `tree-expert-union.svg` | 原创工程示意图；根 `{0,1}`、孩子 `{0,2}` 和 `{1,3}`，说明路径覆盖与整树覆盖的区别 |
| `des-shared-pool.svg` | DES v2 Figure 2，原始 `https://arxiv.org/html/2602.00879v2/method_v4.svg`；比较逐位置选择、共享 coreset 与跨位置投票 |
| `team-overview.png` | TEAM v3 Figure 3，原始 `https://arxiv.org/html/2602.08404v3/3.png`；延迟缓存、hot/cold 分类与两轮路由 |
| `epoch-overview.png` | Epoch v1 Figure 6，原始 `https://arxiv.org/html/2609.09748v1/fig-overview.png`；调度计划与 FreshLane 执行流程 |
| `coactivation-expert-reuse.png` | Expert Coactivation v1 PDF 第 1 页 Figure 1(a)(b)；裁剪矩形 `(106,526,295,644)`，4 倍渲染；保留 9/5 个专家的示意 |
| `ecospec-overview.png` | EcoSpec v1 Figure 2，原始 `https://arxiv.org/html/2607.12696v1/framework.png`；专家预测、覆盖集合维护与候选选择 |
| `s2moe-reuse-gating.svg` | S²-MoE v2 Figure 5，原始 `https://arxiv.org/html/2608.15018v2/reuse.svg`；置信度加权聚合与软路由偏置 |
| `acceptmoe-expert-set.svg` | AcceptMoE v1 Figure 1，原始 `https://arxiv.org/html/2608.02989v1/algo1.svg`；提交概率加权、根位置初始集合与自适应追加 |
| `moce-hierarchical-routing.png` | MoCE EMNLP 2025 PDF 第 3 页 Figure 1；裁剪矩形 `(70,73,526,255)`，4 倍渲染；序列级分组与组内路由 |
| `dcdm-semantic-chunks.png` | DCDM v1 Figure 1，原始 `https://arxiv.org/html/2605.15676v1/overview.png`；内容分组与 chunk-causal attention |

图注只保留技术说明，原文链接在相邻正文及文献中。论文图均核对过图中标注，原创 SVG 的数值与正文一致；SVG 直接保留原始矢量文件，论文图可点击打开原尺寸。

### 全部论文配图核对

上一轮 12 张图只覆盖 11 项研究与 1 张原创示意图。本轮补齐剩余 11 项，正文的 22 项研究各有一张原文配图，含原创示意图共 23 张，图号已连续重排。逐项对应关系和所有原图来源记录在 `figure-coverage.json`，覆盖范围包含 LLaDA2.2 报告；模型代码与配置归入同一报告，不单独计算论文。

| 新增文件 | 原文与内容 |
|---|---|
| `cache-aware-router.svg` | Cache-Aware Joint Router Adaptation v2 Figure 1；原始 `spatio-temporal-router.svg`，缓存保留与因果预取 |
| `llada22-block-routing.png` | LLaDA2.2 报告第 4 页 Figure 3；矩形 `(343,114,542,342)`，5 倍渲染；块级候选池与池内路由 |
| `sticky-token-consistency.png` | StickyMoE v1 第 23 页 Figure 3(a)；矩形 `(120,153,244,280)`，5 倍渲染；只保留正文讨论的同层相邻位置约束 |
| `local-routing-cache-oracle.svg` | Local Routing Consistency v4 Figure 3；原始 `sch.svg`，需要未来窗口信息的缓存分析参照 |
| `simbal-expert-redundancy.png` | SimBal v1 第 7 页 Figure 3；矩形 `(108,76,504,224)`，4 倍渲染；PES 专家冗余对比，未扩展正文篇幅 |
| `cacheable-quality-locality.png` | Cacheable by Design? v1 第 8 页 Figure 5；矩形 `(123,69,488,208)`，4.5 倍渲染；局部性与质量的联合门槛 |
| `evict-selection.png` | EVICT v2 Figure 4；原始 `method.png`，验证规模选择与祖先闭合树裁剪 |
| `specfit-selection.png` | SpecFit v1 第 4 页 Figure 1；矩形 `(107,80,505,222)`，4.5 倍渲染；候选搜索、成本估计与原 target 验证 |
| `draftexpert-architecture.svg` | DraftExpert v1 Figure 6；原始 `draftexpert_architecture.svg`，离线蒸馏与在线扩张感知验证 |
| `moe-spec-budget.svg` | MoE-Spec v1 Figure 1；原始 `overview.svg`，候选树不变时的专家预算 |
| `seqtopk-budget-allocation.png` | SeqTopK v2 第 1 页 Figure 1(a)；矩形 `(55,174,211,302)`，5 倍渲染；固定总预算下的跨位置分配 |

图注保留条件：SCH 的 oracle 使用未来信息；负面结果图的 miss 与困惑度使用不同纵轴；SeqTopK 图解释训练中的整序列竞争；SpecFit 正文的离线路由信息条件继续保留。所有新增图已查看原尺寸，未生成替代图或改写图内标注。

## 构建与显示检查

Jekyll 构建通过。本轮修改后，Chromium 在 1440 像素桌面宽度和 390 像素手机宽度下均验证：70 处数学渲染、5 张表格、3 张图片，无 MathJax 错误、页面脚本错误或整页横向溢出。手机表格使用网站已有的内部滚动。检查场景表、章节顺序、软约束公式和文末模型分类表，并查看桌面及手机截图。

补充 AR 动机后重新构建并检查两种宽度，数学、配图和表格总数保持一致；首表为 5 行场景、4 列，新增收益列可通过表格内部横向滚动访问。检查输出见 `ar-motivation-browser-check.json`，本次机械审稿无命中，公式、引用、frontmatter 和配图路径与前一版一致。

上一轮重点配图版重新构建，1440 和 390 像素宽度均验证 70 处数学渲染、5 张表格、12 张图片，无 MathJax 错误、页面脚本错误或整页横向溢出。12 张图均加载成功，展示尺寸不超出正文，图片链接指向原尺寸资源。另查看桌面与手机截图，核对 SVG、方法图与紧邻说明。检查输出见 `final-illustrated-browser-check.json`。

补齐全部论文后再次构建，覆盖清单与正文 22 项引用逐一匹配，全部 23 张图片在桌面及手机宽度下加载成功，均可打开原尺寸。70 处数学、5 张表格、日期与系列元数据保持不变，无脚本或 MathJax 错误，也无整页横向溢出。已查看 LLaDA2.2、SimBal、负面结果曲线与 SeqTopK 的页面截图。检查输出见 `all-paper-figures-browser-check.json`。

## DCDM 机制补充

按原文第 4.1–4.3 节、Algorithm 1、附录 B 与 D.2 重写第 6.2 节，补充子空间映射的范数打分、带偏置的硬编号、按位置归一化的 soft attention 及其梯度路径，再解释 chunk-causal mask。正文将原文的子空间维度 h 记为 r，避免与隐藏状态 h_i 混淆；路由分数相应记为 s。软路径不加入硬分组偏置。

新增六位置编号 `(1,2,1,3,2,3)` 的可见性表，验证组 1 读取 `{1,3}`，组 2 读取 `{1,2,3,5}`，组 3 读取全部位置。解释推理单流规则与训练双流 mask、分组阶段 noise mask 的区别。DCDM-MoE 的 Qwen-style 逐 token top-k 和 shared expert 独立于分组层，不据此推导共享专家池。原文只明确推理 mask，没有展开跨轮固定或刷新分组的完整采样调度，正文保留此边界，未补造采样流程。

第 6.2 节之外的正文逐字保持不变，原有 23 张图、引用及 11 月 7 日日期保留。新增 3 个展示公式和 1 张可见性表，机械审稿无命中。

补充后 Jekyll 构建通过，1440/390 像素宽度检查 88 处数学渲染、6 张表格及 23 张图片引用，无 MathJax/脚本错误或整页横向溢出。新增公式按多行对齐，便于手机阅读。已查看分组公式和 mask 的页面截图，输出见 `dcdm-expanded-browser-check.json`。

## 系列叙事与重复审阅（2026-11-10）

范围：原系列十篇及番外一，检查问题提出、章节推进、共同概念的职责与文章之间的衔接。本轮更新 DCDM 的 block diffusion 分类；以下为审阅建议，未按去重方案改写原系列正文。审阅不替代全部引用论文和代码的重新事实核验。

### 每篇的职责

| 篇章 | 建议承担的主要问题 | 当前结构判断 |
|---|---|---|
| 一：共识设计 | 开放模型重复采用了什么，证据强弱如何 | 总览定位明确，但 LatentMoE、shared sink、端侧部分与第二篇反复解释机制；目录未列番外 |
| 二：架构与训练 | 固定预算后怎样选择和消融配置 | 预算到配置的主线成立；部分 shared/latent/zero expert 段落仍以动机与建议为主，和第一篇接近 |
| 三：Megatron 源码 | 配置如何落实到选择、分派、专家和合并 | 与第二篇的机制回顾有合理衔接；训练/推理差异和 trace/replay 应保留实现入口，RL 干预交给第七篇 |
| 四：性能诊断 | 怎样从 trace 区分通信、计算和负载等待 | profiler 视角清楚，与第九篇的后端/planner 深入介绍分工有效 |
| 五：100B 算例 | 模型配置怎样变成参数、process groups 与显存预算 | 算例持续使用同一配置，推进清楚；末尾第六篇链接名称仍为旧标题 |
| 六：端侧与投机优化 | 端侧读取、计算、加载等待怎样被摊薄或隐藏 | 前半部分问题驱动清楚，后半重复番外的专家集合教程与算法分类，是首要去重对象 |
| 七：RL 训推一致性 | 同策略执行偏差与新旧策略变化怎样分开诊断 | 源码机制用于具体 RL 问题，有新增分析；上一章链接名称仍为旧标题 |
| 八：架构扩展 | 深度、参数共享、attention、memory 等资源怎样组织 | 按资源轴组织有效，但第 8 节混入工作集复用与压缩、模块化训练，和番外形成第三处讲解入口 |
| 九：EP 后端与调度 | 数据如何移动，逻辑专家如何映射到物理副本 | 第四篇已有诊断入口，本篇增加实现/规划约束，重复以必要背景为主 |
| 十：视觉生成 | 整模型、token、子空间与控制接口分别如何选择专家 | 机制到应用的顺序清楚；不应把视觉噪声阶段专家与 DCDM 的语言生成块合并为一类 |
| 番外一：路由粒度 | 多个位置怎样共同约束候选、并集与生成依赖 | 选池→跨迭代→训练局部性→树验证→语义分组的比较链成立；与第六篇缺少分工和互链 |

### 优先级一：第六篇与番外的实质重复

两个正文共同引用 5 项工作：S²-MoE、EcoSpec、MoE-Spec、AcceptMoE、Expert Coactivation。共同引用本身不是删文依据，重复的推导和解释才是。

| 内容 | 第六篇位置 | 番外位置 | 建议 |
|---|---|---|---|
| 调用次数、唯一专家和驻留集合 | 3.2 节，含四位置 top-2 表与换入量公式 | 第 1 节，含四位置 top-2 表与换入量公式 | 第六篇保留短定义与 verify 成本直觉；完整集合定义、表格和驻留公式放番外 |
| EcoSpec 的 predictor、路径边际成本和覆盖集合 | 4.2 节 | 5.1 节 | 第六篇概述对端侧延迟的作用，算法和 ancestor closure 详解集中在番外 |
| MoE-Spec/AcceptMoE 的排序、预算、加权和质量边界 | 4.3 节 | 5.3 节 | 第六篇保留“专家资格变化会改目标计算”的结论及简表；详细选池放番外 |
| Coactivation 的局部保留规则与并集上界 | 4.4 节 | 4.2 节及第 5 节树扩展 | 第六篇说明训练提高复用；窗口上界、软硬约束及树上区别集中在番外 |
| S²-MoE 的扩展收益与路由 bias | 4.1 节简述 | 5.2 节 | 简述可以保留，但第六篇应明确链接番外的 expert-set 讨论；self-draft/context 与接受算法仍可留投机解码专题 |

两张四位置 top-2 表分别比较 3/8 与 2/8 个独立专家，教学职责相同，换一组数字没有增加新的推理。EcoSpec 和 MoE-Spec 还重复展示了同一篇论文的同一幅框架图。两篇正文目前没有互相链接。

建议让第六篇围绕端侧端到端执行收紧：保留 MobileMoE 的融合算子、LES 的加载时机、MoESD 的 batch/专家覆盖分析、Cascade 的投机开关与长度、BigMoMo 的 Flash→DRAM→映射→VTCM→HMX 流水。第 4 节可改为简短的优化位置对照，再深入 BigMoMo。番外则保留各类集合约束及跨场景比较；树分支与最终提交路径的差异是 beyond-token 路由的重要场景，不宜整体删除。

MoESD 的中等 batch 分析宜明确作为服务器对照：第六篇开头以手机交互式单请求为主，3.3 节进入 batch=32 的覆盖饱和后，需要回到端侧 batch=1 的专家扩张和搬运问题，衔接后续方法。

反向调整也有必要：番外 5.2 节中 DraftExpert 的常驻 drafter、自蒸馏、预取等端侧流水细节，更适合由第六篇承接；番外强调扩张感知截断与预测专家集合即可。不要只在一篇删字，让另一篇继续无边界吸收部署细节。

去重优先调整讲解职责，不删去番外已收录论文及其原图。番外的 22 项研究与 23 张图继续保留；同一论文在第六篇仍可用短介绍定位它改善执行链的哪个环节。

### 优先级二：第八篇的第三处入口

第八篇 8 节先讲 EMO 的文档级专家池，再讲 BAR、MoE-to-Dense，随后讲 XShare、DES、EcoSpec。它同时混合了训练分组、模块化后训练、模型导出和推理工作集策略。“不直接改变 Transformer block”的总括也不能覆盖把 MoE FFN 转为 dense FFN 的导出路线。

建议让第八篇继续承担资源组织方式的横向地图：EMO 用来引出文档级分工，DES/EcoSpec 只保留与工作集相关的一两句并链接番外；BAR 与 MoE-to-Dense 单独标为模块化训练与交付。无需把所有相关论文继续搬入番外。第八篇 MoRE 的 reuse block 沿层维度共享权重，与番外沿位置维度共享候选池不同，应该保留这种区别。

第八篇把 XShare、DES、EcoSpec 合写为“不改模型权重”的工作，还可补明计算语义：前两类约束 token 可选专家，EcoSpec 选择待验证候选、保留目标路由。冻结权重与保留原路由是两个判断，这样才与第六篇及番外的分类一致。

### 其它衔接与表达问题

- 第一篇 1 节的系列目录只列 1–10；第六、八篇正文也未引向番外，应增加与阅读问题对应的互链。
- 第五篇“下一篇”和第七篇“上一篇”仍称第六篇为“端侧 Active MoE 的推理瓶颈与设计取舍”，与实际标题不一致。
- 第一篇 10 节与第二篇 5 节都较详细介绍 LatentMoE 动机，1.11/2.8.2 重讲 shared sink，1.6/2.7 重讲控制器。第一篇宜压成设计信号与证据，第二篇用预算、配置与消融解释取舍；不必为去重删掉第二篇真正新增的约束。
- 第二篇部分“什么时候需要 / 什么时候不需要 / 建议”依赖宽泛情境，和第一篇的机制介绍接近。后续修改时优先落到固定预算、基线与可检验变量，避免再次扩成论文清单。
- 第三/七篇、第四/九篇具备明确递进：实现入口→RL 数值/策略问题，性能诊断→后端与物理调度。保留短背景与跨篇引用即可，不是本轮重点压缩对象。

### 建议的后续修改顺序

先明确第六篇与番外各自回答的问题并添加互链，再将同构集合算例和方法推导集中到番外，同时把 DraftExpert 的端侧执行细节归回第六篇。随后收紧第八篇的部署段落和第一/二篇的重述，最后统一目录、标题与导航。每篇仍保留独立阅读所需的一段背景，不要求读者先跳转才能理解核心问题。

DCDM 本轮已在小标题及分类表中明确为 block diffusion 的语义分组，并保留原小节锚点以兼容已发布链接。

已修正 Kramdown 对行内集合括号及绝对值竖线的解析，使用 `\lbrace`、`\rbrace`、`\lvert` 和 `\rvert`，避免生成错误表格。预览生成目录的文件权限单独调整以便本地 HTTP 服务读取，站点样式和脚本未作修改。

预览与校验输出保存在 `/tmp/moe-routing-research-20261109/`。可发布源稿已进入 `_posts`，此前为本地审阅；本轮按用户要求提交并推送到远程 main。
