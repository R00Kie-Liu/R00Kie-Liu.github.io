---
layout: post
title: "Towards Better Tokenizer (3): 固定分词之外，最新 Tokenizer 研究在关心什么"
date: 2026-06-22
tags: [Tokenizer, survey, Pretraining]
description: "梳理 2026 年 tokenizer 相关研究趋势：模型与 tokenizer co-design、tokenizer-free、动态 tokenization、多语种 token tax、旧 tokenizer 适配、工具/图/action tokenization 和多模态接口。"
published: false
---

系列一讲 tokenizer 如何影响预训练、长上下文和 agent 模型设计。系列二讲 agentic tokenizer：工具调用、thinking token、chat template、parser 和安全边界。

这一篇单独讲研究趋势。

截至 2026 年 6 月，tokenizer 研究已经不只是“BPE 怎么调得更好”，而是在讨论更大的问题：

> tokenizer 应该如何和模型、语料、语言、公平性、工具、图结构和多模态接口一起设计？

我会把最近的趋势分成七类：

1. tokenizer 被重新定义为模型设计的一部分；
2. tokenizer-free / 自适应 byte-level 建模；
3. 动态 / 历史感知 tokenization；
4. 多语种 tokenizer 与 Token Tax；
5. 旧 tokenizer 的适配、扩词表和剪枝；
6. 工具、图、CAD 等非文本对象的 tokenization；
7. 多模态和语音里的 tokenizer-free / modality-specific tokenizer。

## 1. 先给结论

2026 年 tokenizer 研究的主线，不是“固定 tokenizer 会不会立刻消失”，而是：

| 层次 | 判断 |
|---|---|
| 现实层 | 更好的 BPE 仍然重要 |
| 系统层 | tokenizer 要和模型、parser、template、runtime 一起设计 |
| 未来层 | tokenizer 会更动态、更公平、更像 action space |

短期内，BPE、byte-level BPE、tiktoken-style BPE 仍然是主流工程方案。中期会出现更多领域扩词表、动态 tokenization、多语种公平优化。长期看，tokenizer-free、tool/graph/action tokenization、多模态接口会让 tokenizer 的边界越来越模糊。

## 2. Tokenizer 被重新定义为模型设计的一部分

过去很多系统把 tokenizer 当成预处理组件：先训练一个 BPE 或 SentencePiece，然后模型只是被动接受 token 序列。

2026 年一类工作开始明确反对这种看法。例如 [*Stop Taking Tokenizers for Granted*](https://arxiv.org/abs/2601.13260) 直接把 tokenization 说成 LLM 的核心设计决策，强调 tokenizer 会影响公平性、稳定性、安全性、语言覆盖和领域适配。

另一个角度来自信息论。[*An Information-Theoretic Perspective on LLM Tokenizers*](https://arxiv.org/abs/2601.09039) 从压缩和信道容量角度分析 tokenizer，提出 compression-aware BPE 和 capacity-utilization 这类指标。这类工作不只是问“切得短不短”，而是问 tokenizer 是否有效利用了模型容量。

这类研究真正重要的地方不在于提出某个新 tokenizer，而是改变了问题表述：

> 旧问题是：这个 tokenizer 压缩率高不高？  
> 新问题是：这个 tokenizer 是否适合我的模型、语料、任务、语言和部署场景？

这也意味着 tokenizer 选型应该进入模型设计阶段，而不是训练脚本开头的一行配置。

## 3. Tokenizer-Free / 自适应 byte-level 建模

另一条很明显的主线是 tokenizer-free。

[Byte Latent Transformer, BLT](https://arxiv.org/abs/2412.09871) 把讨论推向 raw bytes 和动态 patch：模型不再依赖固定 subword tokenizer，而是在 byte 层级上形成更高层的计算单元。

2026 年的 [ByteFlow](https://arxiv.org/abs/2603.03583) 更进一步，提出让模型直接从 raw byte stream 中学习自己的分段和压缩方式。它的核心主张是：固定 tokenizer 把模型限制在预先决定好的粒度上，而 byte-level 自适应建模可以让模型根据输入自己形成语义单元。

[T-FREE](https://arxiv.org/abs/2406.19223) 则走另一条弱 tokenizer 路线，用 character triplets 的稀疏激活模式替代传统 subword 词表，试图减少 embedding 参数和固定词表带来的语料偏置。

这些工作共同说明：**固定 vocab 不是唯一答案**。长期看，模型可能会把 tokenization 变成内部动态计算，而不是外部静态文件。

不过这并不意味着工程上可以马上抛弃 tokenizer。今天的主流训练、推理、serving、缓存、计费、上下文管理都仍然围绕 token 展开。Tokenizer-free 更像一个长期方向，而不是短期替代。

## 4. 动态 / 历史感知 tokenization

固定 tokenizer 的问题是：一旦训练完，边界就固定了。可是不同输入、不同领域、不同 batch 的高频片段并不一样。

ACL 2025 的 [Dynamic Tokenization](https://aclanthology.org/2025.acl-long.1444/) 尝试在已有 LLM 上动态调整 token 边界：如果一个 batch 里某些 subword 序列频繁出现，就临时合并它们，并用 hypernetwork 预测新 token embedding。

2026 年的相关方向继续往 context-aware 走：tokenizer 不再只看当前字符串，而是参考上下文、历史 tokenization 结果、输入分布甚至熵信号，决定什么时候合并、什么时候重置。

这对企业内部场景很有吸引力。比如：

- 医学报告里反复出现长术语；
- 法律合同里有固定条款；
- 代码仓库里有大量项目内 API 名；
- 日志里有重复模板；
- agent trace 里有固定 tool schema。

这些文本不一定值得重新训练一个 foundation model，但值得动态减少 token 成本。

动态 tokenization 的核心想法是：**tokenizer 不再是一次性训练好的静态对象，而是可以随输入分布自适应的系统组件。**

## 5. 多语种 tokenizer 与 Token Tax

Tokenizer 不是中立的。某些语言会因为 tokenization 更碎，承担更高的训练、推理和 API 成本。

这就是 [Token Tax](https://arxiv.org/abs/2509.05486) 的问题：同样一句话，如果某个语言需要更多 token，它就会在以下方面更贵：

- 训练 token 预算；
- 推理延迟；
- API 计费；
- 上下文容量；
- 长文档 RAG；
- 多轮对话历史。

2026 年多语种 tokenizer 研究也在变得更系统。[MUTANT](https://openreview.net/forum?id=oEpMYAs10U) 这类工作不再只说“把词表加大”，而是从 vocabulary size、训练数据、language-aware pre-tokenization、subword/multiword-aware training 等角度给出 multilingual tokenizer recipe。

这对中文模型尤其重要。我们不能只看英文 benchmark，也不能只看平均 token 数。一个 tokenizer 可能英文很好，但对中文、印地语、阿拉伯语、泰语或低资源语言非常不公平。

好的多语种 tokenizer 需要回答：

- 不同语言的 token fertility 是否均衡？
- 非拉丁文字是否被切得过碎？
- 低资源语言是否只是 byte fallback？
- 代码和多语种文本混排时是否稳定？
- 语言之间如何分配 merge 和 token 预算？

在这个视角下，tokenizer 设计已经带有公平性和成本分配的含义。

## 6. 旧 tokenizer 的适配、扩词表和剪枝

现实中，很多团队不会从零训练 foundation model。更常见的问题是：已有模型的 tokenizer 不适合新领域，怎么办？

2026 年有一类工作关注“教旧 tokenizer 新词”。例如 [*Teaching Old Tokenizers New Words*](https://aclanthology.org/2026.findings-eacl.341/) 提出 continued BPE training：不是训练一个全新 tokenizer 再把不重叠 token 硬塞进去，而是在已有 BPE merge 基础上继续训练，让新增 token 更可达、更有效。[Vocabulary Customization](https://arxiv.org/abs/2509.26124) 则从领域部署角度讨论如何追加领域高频 token，降低特定领域文本的 token fertility。

它还讨论了 vocabulary pruning：删除冗余 token，减少浪费。

这个方向很实用，因为很多领域都有大量高频长片段：

- 医学术语；
- 法律条款；
- 芯片设计信号名；
- 企业内部 API；
- 金融合约；
- 代码仓库私有类名；
- 工具 schema 和日志模板。

但扩词表不能随便做。至少要回答：

- 新 token embedding 如何初始化？
- LM head 如何扩展？
- 是否继续训练？
- 是否损害原有语言能力？
- 新 token 是否真的可达？
- serving runtime 是否支持新 tokenizer？

还有 [R-BPE](https://aclanthology.org/2025.emnlp-main.1169/) 这类工作关注如何复用已有 BPE 词表来改善目标语言，说明“适配旧 tokenizer”不只适用于领域，也适用于语言迁移。

我的看法是：预训练前应该尽量把 tokenizer 设计好；预训练后扩词表可以做，但应该被视为领域或语言适配，而不是常规操作。

## 7. 工具、图、CAD 等非文本对象的 tokenization

2026 年另一个很有意思的趋势是：tokenizer 不再只处理自然语言。

[Graph Tokenization](https://arxiv.org/abs/2603.11099) 把图结构可逆地序列化，再结合 BPE，让标准 Transformer 能处理图数据。它关心的不是“句子怎么切”，而是“图结构如何变成离散 token 序列”。

[GRAFT](https://arxiv.org/abs/2605.11706) 更贴近 agent：它把 tool graph 中的每个工具节点映射成 dedicated special token，并学习工具之间的依赖关系，用于 dependency-aware tool planning。

[Toolscaler / SGTC](https://aclanthology.org/2025.findings-emnlp.30/) 则从工具调用角度提出结构化语义 tokenization，让相似工具共享 subtokens，压缩工具空间，并帮助模型泛化到新工具。

这给 agentic tokenizer 一个很重要的启发：

> agent 的词表未来不只是自然语言词表，而可能是 action vocabulary。

工具、API、环境状态、图结构、任务依赖，都可能被 token 化。

类似地，[CAD-Tokenizer](https://arxiv.org/abs/2509.21150) 为 CAD 序列设计 modality-specific tokenizer，把草图和拉伸操作等 CAD primitive 压缩成离散 token。这说明 tokenizer 正在变成一种更通用的“离散接口设计”。

## 8. 多模态和语音里的 tokenizer-free / modality-specific tokenizer

Tokenizer 的概念也在进入语音、视觉、CAD 等模态。

[VoxCPM2](https://github.com/OpenBMB/VoxCPM/) 这类 tokenizer-free TTS 系统直接生成连续语音表示，绕过传统离散 speech token。[CAD-Tokenizer](https://arxiv.org/abs/2509.21150) 则反过来为 CAD 这种结构化模态设计专门 token。

这两个方向看似相反，其实都在回答同一个问题：

> 一个模态最自然的建模单元是什么？

文本里可能是 subword、byte、character n-gram；语音里可能是连续表示；CAD 里可能是 primitive；工具调用里可能是 action 或 tool node；图里可能是子结构。

因此，未来 tokenizer 的边界会越来越模糊。它不只是“文本切词器”，而是不同模态和不同动作空间进入 Transformer 的接口。

这一节只讨论研究趋势：tokenizer-free TTS、modality-specific tokenizer、图像/语音/CAD 的离散接口为什么重要。具体 VLM、Omni、image/video/audio processor、merger/resampler 和多模态 token budget 的工程设计，放到系列四单独展开。

## 9. Scaling 视角：哪些方向可能进入大规模基座训练？

论文里的 tokenizer 方法很多，但真正放到大规模基座模型训练里，会遇到一组更现实的约束：

- **训练管线是否稳定**：数据清洗、去重、packing、loss mask、checkpoint、resume 都依赖 tokenization 结果。
- **硬件效率是否可控**：序列长度、batch packing、embedding lookup、LM head、通信和 kernel 都会受影响。
- **推理栈是否兼容**：vLLM、SGLang、Transformers、TGI、TensorRT-LLM、KV cache、speculative decoding、streaming parser 都要支持。
- **收益是否足够大**：如果 token 数只省 2%，但训练和推理系统复杂度增加一倍，大规模训练很难接受。
- **风险是否可回滚**：基座模型训练一旦开始，tokenizer 基本不能改。越底层的 tokenizer 变化，风险越大。

所以，从 scaling 角度看，这些研究可以分成三类。

### 9.1 最有机会进入大规模基座训练的方向

第一类是 **改进固定 tokenizer，但不破坏现有训练/推理范式**。

包括：

- 更好的 byte-level BPE / tiktoken-style BPE；
- 多语种 tokenizer recipe；
- language-aware pre-tokenization；
- compression-aware BPE；
- 更系统的 special token / agent token 规划；
- 更合理的 reserved token 区间；
- tokenizer 与 chat template / parser / processor 成套发布。

这些方向最容易进入大规模训练，因为它们仍然产出一个稳定词表和稳定 token id 序列。训练系统、数据管线、推理框架、计费方式都不需要根本改变。

我认为未来一两代大规模基座模型最可能采用的是这条路线：**不是抛弃 tokenizer，而是把 tokenizer 训练得更系统、更公平、更任务感知。**

多语种 tokenizer 也是这里最有现实价值的方向。因为 token tax 会直接影响训练 token 预算、推理成本和用户体验，而改进 tokenizer 的收益可以在全链路放大。

### 9.2 有机会在后训练、领域模型、企业模型中落地的方向

第二类是 **不一定适合从零训练超大基座，但很适合领域适配或中小规模模型**。

包括：

- continued BPE training；
- vocabulary customization；
- vocabulary pruning；
- dynamic tokenization；
- 领域高频术语扩词；
- 针对代码库、日志、schema、工具集合的局部 token 优化。

这些方法的优势是收益更集中。比如医学、法律、芯片、金融、企业代码库里有大量重复术语和模板，扩展少量 token 可能明显降低 token fertility。

但它们放到大规模通用基座训练里会更困难：

- 扩词表会改变 embedding 和 LM head；
- 动态 token 会影响 batching、KV cache、speculative decoding 和 serving；
- 不同领域的最优 token 可能相互冲突；
- 很难保证不损害通用能力；
- tokenizer 版本管理会更复杂。

所以这类方向更可能先在 **领域模型、企业私有模型、继续预训练、推理时压缩、RAG/agent 专用系统** 中落地，而不是马上成为通用基座模型默认方案。

### 9.3 更像中长期研究或特定模态系统的方向

第三类是 **有想象力，但离主流大规模文本基座训练还有距离**。

包括：

- tokenizer-free byte-level 模型；
- raw byte 自适应 patch；
- character triplet sparse representation；
- graph tokenization；
- tool graph / action tokenization；
- CAD tokenizer；
- tokenizer-free TTS；
- modality-specific tokenizer。

这些方向真正有趣，因为它们挑战了“文本 subword token 是唯一接口”的假设。长期看，它们可能改变模型架构。

但短期放到大规模基座模型训练里，难点很多：

- byte-level 输入会显著拉长序列，需要新的架构来抵消计算成本；
- 动态 patch 会改变 kernel、packing、cache、并行策略；
- tokenizer-free 方法和现有数据 pipeline、评测、serving、计费体系不完全兼容；
- graph/tool/action tokenization 需要专门数据和任务，未必能直接受益于通用网页语料；
- modality-specific tokenizer 往往和对应 encoder/decoder 绑定，不能只作为文本 tokenizer 替换。

因此，我更倾向于把它们看作 **架构级研究** 或 **特定模态/特定任务系统** 的方向。它们未必会马上替代 BPE，但会影响未来模型如何定义“输入单元”和“行动单元”。

### 9.4 一个简化判断表

| 方向 | 大规模基座训练可行性 | 更可能首先落地的场景 | 主要瓶颈 |
| --- | --- | --- | --- |
| 更好的 byte-level BPE / tiktoken BPE | 高 | 通用基座模型 | 语料配比和多语种公平 |
| 多语种 tokenizer recipe / Token Tax 优化 | 高 | 多语种基座模型 | 语言间 token 预算分配 |
| agent special token / chat template co-design | 高 | agentic 基座和后训练模型 | parser/runtime 对齐 |
| compression-aware BPE | 中高 | 新基座 tokenizer 设计 | 指标和下游效果的一致性 |
| continued BPE / 扩词表 | 中 | 领域模型、继续预训练 | embedding 初始化和能力保持 |
| vocabulary pruning | 中 | 小模型、端侧、领域模型 | 删除 token 后的兼容性 |
| dynamic tokenization | 中低 | 企业领域推理、RAG、代码库 | batching、KV cache、serving 复杂度 |
| tokenizer-free byte/patch 模型 | 中长期 | 新架构研究 | 序列长度、训练效率、生态兼容 |
| graph/tool/action tokenization | 中长期 | agent、规划、工具系统 | 数据、评测、runtime 协议 |
| CAD/语音/多模态 tokenizer | 任务相关 | 特定模态模型 | 与 encoder/decoder 强绑定 |

### 9.5 我的判断

如果目标是训练下一代大规模文本基座模型，最现实的探索顺序是：

1. **先把固定 tokenizer 做到足够好**：byte-level、150K 左右、多语种公平、代码/JSON/agent 轨迹充分覆盖。
2. **把 agent 协议前置到 tokenizer 设计里**：role、tool、thinking、media、FIM、reserved token 一起规划。
3. **用小规模 ablation 验证 tokenizer 选择**：不要只看压缩率，要看 loss、下游任务、生成稳定性、训练吞吐。
4. **再考虑领域扩词表和 pruning**：用于企业/领域模型，而不是一开始就污染通用基座词表。
5. **持续关注 tokenizer-free / dynamic tokenization**：但把它们视为架构实验，而不是短期默认选项。

换句话说，大规模基座训练最怕“底层不稳定”。Tokenizer 越靠近输入底层，越需要保守。真正能进 scaling pipeline 的，往往不是最激进的论文想法，而是 **收益明确、风险可控、能和现有训练推理系统兼容** 的方法。

## 10. 对实践有什么启发

我会把启发分成三层。

### 10.1 现实层：更好的 BPE 仍然重要

短期内，绝大多数 LLM 仍然会使用 BPE、byte-level BPE、tiktoken-style BPE 或 Qwen2Tokenizer 这类成熟路线。

所以实践上仍然要做好：

- 分桶训练 tokenizer；
- 中文、多语种、代码、JSON、tool trace 都进入 tokenizer 训练语料；
- 评估 token/char、token/byte、tool call tokens/call；
- 控制词表大小和 embedding/logits 成本；
- 预留 role/tool/thinking/media/FIM token。

### 10.2 系统层：tokenizer 要和 parser、template、runtime 一起设计

对 agent 模型来说，tokenizer 不能单独看。

它必须和这些东西一起设计：

- chat template；
- tool parser；
- thinking parser；
- media processor；
- streaming state machine；
- serving runtime；
- 安全 escaping；
- SFT/RL 数据格式。

很多 agent 失败不是模型不会用工具，而是 tokenizer、模板、parser、serving 的协议没有对齐。

### 10.3 未来层：tokenizer 会更动态、更公平、更像 action space

长期看，研究趋势大概会走向：

- tokenizer-free byte-level / patch-level 模型；
- dynamic tokenization；
- 多语种 token tax 优化；
- 领域扩词表和剪枝；
- tool graph / action tokenization；
- modality-specific tokenization；
- tokenizer 和模型 co-design。

也就是说，未来 tokenizer 不一定是一个固定 vocab 文件，而可能是模型内部的动态边界选择机制，也可能是工具、动作、图和多模态对象的统一接口。

## 11. 总结

系列一讲的是 tokenizer 如何影响预训练、长上下文、中文和 agent 模型。系列二讲的是 agentic tokenizer 如何变成工具调用和 thinking 的协议层。

系列三想补上的，是一个更宽的视角：

> tokenizer 正在从“文本切分器”变成“模型与世界之间的离散接口”。

它可以是文本 token，也可以是 byte patch、character n-gram、tool node、graph substructure、CAD primitive、media placeholder、action token。

从研究趋势看，固定 tokenizer 正在受到挑战，但这不是一个简单的“会不会被替代”的问题。更准确的说法是：tokenizer 会变得更动态、更公平、更领域化，也更靠近模型和环境交互的边界。

所以，今天设计 tokenizer 时，我们仍然要把 BPE、词表大小、压缩率做好；但也要意识到，未来的 tokenizer 可能不仅服务于语言建模，还会服务于工具调用、图推理、多模态生成和行动规划。

## 12. 参考资料

- Byte Latent Transformer: Patches Scale Better Than Tokens: <https://arxiv.org/abs/2412.09871>
- T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings: <https://arxiv.org/abs/2406.19223>
- Retrofitting Large Language Models with Dynamic Tokenization: <https://aclanthology.org/2025.acl-long.1444/>
- Toolscaler: Semantic Graph-Based Tool Calling for Large Language Models: <https://aclanthology.org/2025.findings-emnlp.30/>
- The Token Tax: <https://arxiv.org/abs/2509.05486>
- Vocabulary Customization for Efficient Domain-Specific LLM Deployment: <https://arxiv.org/abs/2509.26124>
- R-BPE: <https://aclanthology.org/2025.emnlp-main.1169/>
- Stop Taking Tokenizers for Granted: <https://arxiv.org/abs/2601.13260>
- An Information-Theoretic Perspective on LLM Tokenizers: <https://arxiv.org/abs/2601.09039>
- ByteFlow: <https://arxiv.org/abs/2603.03583>
- MUTANT: A Modular Multilingual Tokenizer Recipe: <https://openreview.net/forum?id=oEpMYAs10U>
- Teaching Old Tokenizers New Words: <https://aclanthology.org/2026.findings-eacl.341/>
- Graph Tokenization: <https://arxiv.org/abs/2603.11099>
- GRAFT: Graph-Tokenized LLMs for Tool Planning: <https://arxiv.org/abs/2605.11706>
- CAD-Tokenizer: Towards Text-based CAD Prototyping via Modality-Specific Tokenization: <https://arxiv.org/abs/2509.21150>
- VoxCPM2: Tokenizer-Free TTS: <https://github.com/OpenBMB/VoxCPM/>
