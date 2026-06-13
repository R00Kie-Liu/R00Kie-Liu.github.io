---
layout: post
title: "2025-2026开源LLM演进综述：From Scaling to Agentic system"
date: 2026-06-13
tags: [survey]
description: "梳理 2025-2026 年开源、开放权重和开放报告 LLM 在 MoE、长上下文、RL、Agentic 系统和多模态方向的演进。"
---

## 写在前面

到写下这篇文章时，我即将作为一名预训练算法工程师工作满一年。过去这一年，LLM 领域给我的直观感受并不是线性的“模型又大了一点、榜单又高了一点”，而更像是一场高密度的技术竞速：开源和开放权重模型快速追赶，MoE、长上下文、合成数据、可验证 RL、Agentic workflow、多模态和推理系统几乎同时加速；每隔几周就会出现新的技术报告、新的模型家族和新的训练范式。

所以这篇综述更像是一份阶段性笔记：我想把过去一年观察到、读到过、也在工作中反复感受到的模型扩张和技术快速演进做一次系统整理。它不是严格意义上的学术 survey，而是站在预训练研究和工程实践交界处，对 2025-2026 年开源 LLM 演进路径的一次复盘。

## 摘要

2025-2026 年的开源、开放权重和开放报告大语言模型呈现出清晰的代际变化：能力提升不再主要依赖单一 dense 模型的参数堆叠，而是来自 MoE 稀疏激活、长上下文注意力、数据合成、可验证强化学习、agentic 任务环境和推理系统的共同演进。DeepSeek-V4、GLM-5、Kimi K2/K2.5、MiniMax-M2/M3、LongCat-Flash、MiMo-V2-Flash、Step-3/3.5、Qwen3/Qwen3.5、Nemotron、Gemma 4、Ant Ling、Hy3-preview 等模型共同表明，开放模型已经进入“训练流程 + 数据工程 + 架构 + 系统”协同优化阶段。ERNIE 5.0、MAI-Thinking-1 这类非 open-weight 但披露较完整技术报告的模型，则作为本文的对照样本，用来补充统一多模态、自回归生成和大规模 RL 系统侧的观察。

本文基于公开技术报告、官方 model card/reference 和少量高质量解释性资料，对最近一年开放模型的主要进展进行梳理。重点关注五条主线：各家模型谱系演进；预训练、mid-training、后训练和 RL 的阶段化分工；优化器、learning rate、辅助损失与能力整合 recipe；MoE 架构与负载均衡；长上下文、多模态和 agentic 能力的系统化发展。

> 说明：本文把“官方技术报告/论文”和“model card、blog、visual guide”等资料分层处理。后者只用于补充架构理解或产品定位，不用于推断未披露的数据配比、RL rollout 规模或完整训练 recipe。

> 术语说明：标题中的“开源 LLM”采用宽口径写法，覆盖 fully open、open-weight、open-report/reference 三类资料。严格 fully open 的模型只占少数；更多模型是开放权重但不开放完整数据/日志，也有 ERNIE 5.0、MAI-Thinking-1 这类不开放权重但技术报告信息量很高的对照样本。正文会尽量区分这些证据边界。

## 本文核心判断

- **Scaling 的重心从参数规模转向系统协同。** 新一代模型不再只比 total params，而是比 active params、MoE 路由、attention/KV cache、低精度、kernel 和 serving topology 能否一起工作。
- **Mid-training 成为能力迁移和扩长的关键阶段。** 长上下文、代码、agentic workflow、多模态长音视频和新 attention 往往不是依靠最后 SFT 补齐，而是在 mid-training/continued pretraining 中重新组织数据和训练目标。
- **Agentic 能力是跨阶段注入的。** Pretraining/mid-training 建立代码、工具、网页、GUI、长上下文和跨模态先验；SFT/Distillation 教格式和协议；RL/环境 rollout 才把这些先验转成可执行行为。
- **SFT 没有消失，而是重新分工。** 通用聊天 SFT 的边际作用下降，但 thinking/non-thinking 控制、工具协议、long-CoT cold start、specialist consolidation 和 omni interaction alignment 变得更重要。
- **RL 的核心瓶颈越来越像基础设施问题。** Verifier、sandbox、rollout 长尾、异步调度、off-policy MoE mismatch、数据分布偏移和 reward 审计，往往比单个算法名更决定可扩展性。
- **优化 recipe 正在成为显性竞争变量。** Muon/MuonClip/Muon Split、WSD/cosine/constant LR、MTP loss weight、load-balance coefficient 和 OPD/MOPD/consolidation SFT 共同决定训练稳定性、token efficiency 和多专家能力能否合并。
- **长上下文竞争从窗口长度转向 utility per dollar。** 256K/1M 只是表面指标，关键是长文、代码仓库、工具轨迹和多模态序列能否以可承受的 prefill/decode/KV 成本被持续利用。
- **多模态正在从“输入扩展”变成 agent 的感知与行动层。** Qwen3.5-Omni、Kimi K2.5、ERNIE 5.0、LongCat/Step/MiMo 分支说明，视觉、音频、视频、GUI 和生成能力正在和工具调用、代码、搜索、交互体验耦合。
- **公开资料仍有明显盲区。** 精确数据配比、SFT/RL rollout 规模、base model agentic eval、packing 后的真实 token mixture、dataloader 消费顺序和 contamination audit 仍普遍披露不足。

## 核心模型速览

| 模型 | 参数/激活规模 | 上下文 | 核心技术关键词 | 训练/后训练主线 |
|---|---:|---:|---|---|
| Qwen3/Qwen3.5/Omni | Qwen3-235B-A22B；Qwen3.5-397B-A17B；Omni Plus/Flash | Qwen3 长上下文版本到 256K/1M 级；Qwen3.5-Omni 256K | thinking/non-thinking, Gated DeltaNet + MoE, Thinker-Talker, AuT, OPD | Qwen3 36T 预训练 + Long-CoT/RL；Qwen3.5 主线暂无 standalone tech report；Omni 4T multimodal S2 + Specialist Distillation/OPD/Interaction-Aligned RL |
| DeepSeek-V4 | Pro 1.6T/49B，Flash 284B/13B | 1M | CSA/HCA, mHC, Muon, DeepSeekMoE | 32T/33T 预训练，domain experts，GRPO，on-policy distillation |
| Kimi K2/K2.5 | 1.04T/32B lineage | K2 预训练 4K、扩到 128K；K2.5 eval context 256K | MuonClip, QK-Clip, ultra-sparse MoE, MLA, MoonViT-3D, Agent Swarm/PARL | K2 15.5T 预训练 + agentic synthesis；K2.5 约 15T mixed vision-text tokens + zero-vision SFT + joint multimodal RL |
| GLM-5 | 744B/40B | 4K -> 200K mid-training | DSA, MLA-256, Muon Split, async RL | 27T base，28.5T total，Reasoning RL -> Agentic RL -> General RL |
| MiniMax-M2/M2.7/M3 | M2/M2.7 为 229.9B/9.8B；M3 约 428B/23B | M2/M2.7 为 192K native；M3 为 1M | M2: fine-grained experts, sigmoid gating, Forge；M3: MSA, 128 experts/top-4 + 1 shared, sigmoid routing, native multimodality | M2: 29.2T 预训练，agent-native RL；M3: coding/agentic + 1M context + multimodal；MSA 报告补齐 attention/kernel，但未披露完整 M3 数据/RL |
| LongCat-Flash | 560B/18.6B-31.3B dynamic | 128K | zero-computation experts, ScMoE, PID expert bias | 20T+ 预训练，reasoning/code mid-training，multi-agent synthesis |
| MiMo-V2-Flash | 309B/15B | 32K -> 256K | hybrid SWA/global, MTP, MOPD | 27T 预训练，SFT，teacher RL/SFT，on-policy distillation |
| Step-3/Step 3.5 Flash | Step-3 321B VLM / 316B LLM / 38B active；Step 3.5 Flash 196B/11B | Step-3 系统报告重点分析 4K/8K/32K decoding；Step 3.5 Flash 128K，评测到 256K | MFA, Attention-FFN Disaggregation, FP8, 3:1 SWA/full attention, MTP-3, MIS-PO | Step-3 强调模型-系统协同和低 TPOT；Step 3.5 Flash 17.2T 预训练 + 128K mid-training + SFT/domain RL/MIS-PO |
| Nemotron 3 Ultra | 550B/55B | 1M | hybrid Mamba-Attention, LatentMoE, NVFP4, MTP | 20T 预训练，SFT/RLVR/MOPD，开源数据、recipe 与 checkpoint |
| ERNIE 5.0 | trillion-parameter ultra-sparse MoE | 8K -> 32K/128K | unified autoregressive multimodal, modality-agnostic routing, elastic training, UM-RL | text/image/video/audio 从头统一训练，SFT + unified multimodal RL；非 open-weight 对照 |
| Gemma 4 | E2B/E4B/12B/31B dense；26B A4B MoE | 128K-256K | local/global interleaving, p-RoPE, K=V global attention, per-layer embeddings, encoder-free 12B, MTP drafter | 官方暂无 tech report PDF；model card + visual guide 可作为端侧高效、多模态和 speculative decoding 案例 |
| MAI-Thinking-1 | 1T/35B | 256K；30T@16K -> 3.4T@64K -> 150B@256K | MoE thinking model, enterprise-grade data | 30T 预训练，reasoning-focused post-training；非 open-weight 对照 |

## 阶段化训练框架

| 阶段 | 主要目标 | 最新进展 | 代表模型 |
|---|---|---|---|
| Pretraining | 建立通用语言、代码、数学、多语言和世界知识能力 | 15T-36T token 成为常见区间；FP8/Muon/MTP/高稀疏 MoE 提高 token efficiency；多模态模型开始披露 trillion-token 跨模态 mixture | Qwen3/Qwen3.5, Qwen3.5-Omni, DeepSeek-V4, Kimi K2/K2.5, GLM-5, MiniMax-M2/M3, MiMo-V2-Flash, Step 3.5 Flash, ERNIE 5.0 |
| Mid-training | 扩展上下文、强化代码/推理/agentic workflow、适配新 attention | 4K -> 200K、32K -> 256K、128K -> 1M；DSA/CSA/HCA/SWA/YaRN 等通过 continued pretraining 或 context activation 适配 | GLM-5, Kimi K2.5, Qwen3.5-Omni, LongCat-Flash, MiMo-V2-Flash, DeepSeek-V4, Step 3.5 Flash, ERNIE 5.0 |
| 跨阶段优化 recipe | 提高 token efficiency、稳定 MoE/attention、控制阶段遗忘和能力合并 | Muon/MuonClip/Muon Split 与 AdamW exceptions；WSD/cosine/constant LR 按阶段切换；MTP/balance/dropout/loss weighting 随阶段调参；OPD/MOPD/consolidation SFT 合并专家能力 | DeepSeek-V4, Kimi K2, GLM-4.5/5, ERNIE 5.0, MAI-Thinking-1, MiMo-V2-Flash, Qwen3.5-Omni |
| SFT | 冷启动推理格式、工具协议、指令跟随、对话风格 | 从通用 SFT 转向 domain-specific SFT、long-CoT cold start、zero-vision SFT 和 specialist distillation | Qwen3, Qwen3.5-Omni, Kimi K2.5, DeepSeek-V4, MiMo-V2-Flash, Step 3.5 Flash |
| RL/RLVR | 通过可验证任务提升数学、代码、证明、工具、视觉和交互能力 | GRPO、MIS-PO、RLVR、rule/model-based reward、verifier/sandbox、visual/outcome reward 成为主流 | Qwen3, DeepSeek-V4, Kimi K2/K2.5, Qwen3.5-Omni, Step 3.5 Flash, Magistral, ERNIE 5.0 |
| Agentic RL | 训练长轨迹、多工具、多环境任务执行能力 | rollout 服务、异步 RL、Agent-as-Verifier、Docker/browser/terminal/GUI/omni 环境成为关键基础设施 | GLM-5, Kimi K2.5, Qwen3.5-Omni, MiniMax-M2/M3, LongCat-Flash, MiMo-V2-Flash, Step 3.5 Flash |
| Distillation | 合并专家能力、压缩小模型、避免阶段间遗忘 | on-policy distillation、cross-stage distillation、MOPD、strong-to-weak distillation、跨模态 specialist distillation | DeepSeek-V4, GLM-5, Qwen3/Qwen3.5-Omni, MiMo-V2-Flash |

## 1. 总体趋势

这一代开放模型的核心变化可以概括为五点。

第一，MoE 成为 frontier open-weight 模型的默认扩展路径。DeepSeek-V4-Pro 达到 1.6T total / 49B active，Kimi K2/K2.5 为约 1T / 32B active lineage，GLM-5 为 744B / 40B active，Qwen3-235B-A22B 为 235B / 22B active，Qwen3.5-397B-A17B 为 397B / 17B active，LongCat-Flash 为 560B total 且动态激活约 18.6B-31.3B，MiMo-V2-Flash 为 309B / 15B active，Step 3.5 Flash 为 196B / 11B active，MiniMax-M2 为 229.9B / 9.8B active，MiniMax-M3 为约 428B / 23B active。模型竞争的重点从“总参数越大越好”转向“在固定 active params 下获得更好的能力和吞吐”。

第二，训练流程被明确拆成 pretraining、mid-training、SFT、RL、agentic RL 和 distillation 多阶段。GLM-5 把 4K 到 200K 的 context extension 明确放入 mid-training；LongCat-Flash 在 mid-training 中强化 reasoning/coding 并扩到 128K；MiMo-V2-Flash 先以 32K native context 预训练，再扩展到 256K；Kimi K2.5 和 Qwen3.5-Omni 则把 context extension 扩展到长视频、长音频和 visual/omni agent 场景。MiniMax-M3 的 MSA 则说明 MiniMax 开始从 M2 的 full attention + GQA 路线转向百万上下文 sparse attention。后训练部分，Qwen3、DeepSeek-V4、GLM-5、MiniMax-M2/M3、Kimi K2/K2.5、Qwen3.5-Omni、MiMo-V2-Flash 都显式使用 RL、distillation 或 RL-like pipeline。

第三，数据工程从通用语料清洗转向可生成、可验证、可执行的数据系统。MiniMax-M2 的 SWE/AppDev/Terminal/Cowork 管线、Kimi K2 的 agentic data synthesis、Kimi K2.5 的 GUI/action trajectories 与 Agent Swarm prompts、LongCat-Flash 的 multi-agent synthesis、Qwen3 的 query-verifier pairs、Qwen3.5-Omni 的跨模态 teacher distillation 与 Interaction-Aligned RL、MiMo-V2-Flash 的 teacher reward 和 outcome reward，都表明“任务环境 + verifier + rollout”正在成为高价值数据来源。

第四，长上下文设计从 RoPE scaling 转向架构与系统联合优化。DeepSeek-V4 用 CSA/HCA 将 1M context 下的 FLOPs 和 KV cache 大幅压缩；GLM-5 用 DSA 进行 continued pretraining；MiMo-V2-Flash 用 5:1 的 SWA/global hybrid attention 和 128-token window；Nemotron 用 hybrid Mamba-Transformer；MiniMax 和 LongCat 则从训练、推理和 agent workload 角度优化长上下文成本。

第五，多模态从 VLM 进入 omni/agent 阶段。Qwen3.5-Omni、Qwen3-Omni、LongCat-Omni、Step-Audio、MiMo-Audio、LongCat-Video、Step-Video、GLM-5V、STEP3-VL、MiMo-VL 等模型说明，多模态已经不只是输入扩展，而是在扩展 agent 的感知、交互和动作空间。ERNIE 5.0 进一步提供了一个非 open-weight 但报告细节丰富的统一自回归对照：它将 text、image、video、audio 的理解和生成放进同一个 Next-Group-of-Tokens 目标和同一个 MoE backbone 中，而不是只给语言模型外挂模态 decoder。

## 2. 各家模型谱系演进

### 2.1 Qwen：从通用基座到 thinking、coder、omni

Qwen3 是一个 dense + MoE 并行的模型家族，突出 36T tokens、多语言、数学和代码能力，以及 thinking/non-thinking 双模式。其后训练设计很有代表性：旗舰模型采用四阶段流程，包括 Long-CoT Cold Start、Reasoning RL、Thinking Mode Fusion 和 General RL。Qwen3 还通过 strong-to-weak distillation 把大型模型能力迁移给小模型，降低小模型重复完整后训练流程的成本。

Qwen3 的数据构造也展示了当前 reasoning 模型的典型做法：长 CoT cold start 数据覆盖数学、代码、逻辑推理和 STEM，样本需要 verified answers 或 code-based test cases；Reasoning RL 使用 query-verifier pairs，并通过 GRPO 更新模型；General RL 使用 rule-based reward、带参考答案的 model-based reward 和无参考答案的 preference reward。Qwen3.5 则进一步扩展到 vision-language 和 omni，但主线 Qwen3.5 与 Qwen3.5-Omni 需要分开写：前者是 Qwen3.5-397B-A17B 这样的 native vision-language MoE，后者是基于 Qwen3.5 继续扩展到 text、image、audio、audio-video 和实时交互的 omni 模型。

从模型家族演进看，Qwen 的特点是把“全尺寸谱系”和“训练流程复用”结合得最完整。Qwen3 同时发布 dense 与 MoE：dense 覆盖 0.6B、1.7B、4B、8B、14B、32B，MoE 覆盖 30B-A3B 和 235B-A22B。旗舰 Qwen3-235B-A22B 使用 128 experts、top-8，不使用 Qwen2.5-MoE 中的 shared expert，并把 maximum context length 设为 128K。它不是单点模型，而是一套从大模型到小模型的训练/蒸馏体系：大模型先完整走 post-training，小模型则用 strong-to-weak distillation 复用 teacher 的 thinking/non-thinking 能力。

预训练方面，Qwen3 的 36T tokens 被拆成三段：S1 general stage 超过 30T tokens，覆盖语言能力、世界知识和通用文本；S2 reasoning stage 提高 STEM、coding、reasoning 和 synthetic data 比例；S3 long-context stage 将 sequence length 从 4K 提升到 32K，长上下文语料中约 75% 为 text，并包含高质量长文档数据。Qwen 的数据工程重点不是只扩大网页语料，而是通过 Qwen2.5-VL 从 PDF/视觉文档中抽取文本，用 Qwen2.5-Math/Qwen2.5-Coder 生成数学和代码 synthetic data，并用 multilingual annotation system 做 instance-level 标注与小模型消融，指导数据过滤和组合。

Qwen3 的后训练可以概括为两个目标：thinking control 与 strong-to-weak distillation。前两阶段建立 thinking：Long-CoT Cold Start 使用经过 query/response 过滤的数学、代码、逻辑、STEM 样本；Reasoning RL 使用 query-verifier pairs，报告披露了 3,995 个 query-verifier pairs，并用 GRPO、大 batch、多 rollout 和 off-policy 采样提高训练效率。后两阶段把 thinking/non-thinking 统一：Thinking Mode Fusion 用 `/think` 和 `/no_think` 模板把两种模式合入同一模型，General RL 再修复指令跟随、格式、知识、安全和长程决策等通用行为。thinking budget 是这一流程的自然产物：模型学会在未完整展开 CoT 时也能给出答案，因此可以用 token budget 控制推理深度。

Qwen3.5 主线在本文中按“official reference/model card”而不是 standalone technical report 处理。截至本文整理时，未找到 Qwen3.5 general model 的独立 PDF；官方 reference 指向 Qwen3.5 release blog、HF collection 和 Qwen3.5-397B-A17B model card。其关键架构信息已经足以支撑本文的架构讨论：Qwen3.5-397B-A17B 是 397B total / 17B active，采用 Gated DeltaNet + sparse MoE，512 experts，10 routed + 1 shared expert。它不宜被简化为“Qwen3 的同构升级”：Qwen3-235B-A22B 是 128 experts、top-8、无 shared expert 的传统 sparse MoE，而 Qwen3.5 把 Gated DeltaNet 与 MoE 结合，目标更偏向 native vision-language agent 的吞吐和长序列效率。

Qwen3-Omni 到 Qwen3.5-Omni 的演进可用于说明“omni 模型”如何从多模态拼接走向原生交互。Qwen3-Omni 已采用 Thinker-Talker MoE：Thinker 负责文本/多模态理解和文本生成，Talker 负责流式语音 token 生成，并通过 AuT 音频编码器、多码本 codec、MTP 和 ConvNet Code2Wav 降低首包延迟。Qwen3.5-Omni 进一步把 Thinker 和 Talker 都升级为 Hybrid-Attention MoE，并引入 Gated Delta Net 以提高长音视频序列建模效率；它支持 256K token、10 小时以上音频理解和约 400 秒 720P 视频输入。

Qwen3.5-Omni 并非与 Qwen3.5 主线无关的独立架构。报告明确说 Thinker/Talker built upon Qwen3.5 的 Hybrid MoE architecture，并且 pretraining S1 中 Qwen3.5-Omni is initialized with parameters from Qwen3.5，同时视觉编码器来自 Qwen3.5-VL。因此更准确地说：Qwen3.5-Omni 是以 Qwen3.5 为语言/多模态基座初始化，再经过 Encoder Alignment、General Multimodal Pretraining 和 Long Context Stage 进行 omni 加训/重训式扩展。它的 S2 general pretraining 约 4T tokens，其中 text 0.92T、audio 1.99T、image 0.95T、video 0.14T、video-audio 0.29T；S3 再把上下文从 32K 扩到 262K，并提高长音频、长视频比例。

后训练方面，Qwen3.5-Omni 分三步：先用 text、vision、audio 等 specialist teachers 做 Specialist Distillation；再通过 On-Policy Distillation 把文本查询下更强的回答能力蒸馏到音频查询；最后用 Interaction-Aligned RL 处理多轮语音交互中的语言切换、persona 一致性和长上下文指令跟随问题。这个流程说明，omni 模型的 post-training 不只是“多模态 SFT”，而是在解决不同模态条件下的响应质量差异。

### 2.2 DeepSeek：从 V3 的高效 MoE 到 V4 的 million-token context

DeepSeek-V3 以 DeepSeekMoE、MLA、MTP、FP8 和 aux-loss-free load balancing 为核心，是这一轮开放 MoE 的关键起点。DeepSeek-V4 在此基础上把目标推进到 million-token context。V4 包含 Pro 和 Flash 两个版本：Pro 为 1.6T total / 49B active，Flash 为 284B total / 13B active。其关键改动包括 CSA/HCA hybrid attention、mHC residual connection 和 Muon optimizer。

DeepSeek-V4 的价值在于，它把长上下文问题明确拆成 attention computation、KV cache、MoE kernel、cache storage 和 post-training infrastructure。报告称在 1M context 下，V4-Pro 相比 V3.2 只需要约 27% single-token inference FLOPs 和 10% KV cache，V4-Flash 进一步降到约 10% FLOPs 和 7% KV cache。这说明长上下文竞争不只是最大窗口长度，而是单位 token 成本和长期任务可持续性。

V4 的模型演进可以看作 DeepSeek-V3/V3.2 路线的“长上下文重构版”。V3 奠定了 DeepSeekMoE、MLA、MTP、FP8 和 aux-loss-free load balancing 的效率基线；V4 没有只把参数继续放大，而是围绕 1M 上下文重新设计 attention 与残差路径。Hybrid CSA/HCA 是核心：CSA 将每 m 个 KV 压成一个 compressed entry，再用 lightning indexer 选择 top-k compressed KV 做 sparse attention，同时保留 sliding window 分支；HCA 用更大的 compression rate 做更激进的压缩，用于降低远程上下文成本。两者与 SWA/uncompressed tail state 共同形成 heterogeneous KV cache，这也解释了为什么 V4 报告花了大量篇幅讨论 cache layout、cache hit/eviction 和 on-disk KV cache。

DeepSeek-V4 的另一个重要信号是 Muon 正式进入 frontier-scale open-weight 训练。V4 对大多数模块使用 Muon，对 embedding、output head、mHC 和 RMSNorm 等保留 AdamW，并设计 hybrid ZeRO bucket assignment 以处理 Muon 需要完整梯度矩阵的问题。mHC 则是对 residual connection 的升级：通过 manifold-constrained residual mapping 提高深层网络稳定性，但它会增加 activation memory 和通信，因此报告又配套了 fused kernel 和 memory-efficient implementation。换句话说，V4 的架构创新不是“一个 attention trick”，而是一组 attention、optimizer、residual、kernel 和 cache 管理的协同改造。

后训练方面，DeepSeek-V4 采用“domain experts + on-policy distillation”的两阶段范式：先分别训练数学、代码、agent、instruction following 等领域专家，每个专家经过 domain SFT 和 GRPO；再通过 on-policy distillation 将专家能力合并到统一模型中。这是当前多能力模型避免互相干扰的一条重要路线。

DeepSeek 的 OPD 也值得和 Qwen/MiMo 对比。Qwen 的蒸馏重点是把大模型 thinking/non-thinking 迁移到小模型，MiMo 的 MOPD 强调多 teacher token-level reward 与 outcome reward 的统一，而 DeepSeek-V4 的 multi-teacher OPD 更像“领域专家合并器”：student 采样自己的 on-policy 轨迹，再对齐多个 domain teachers 的 full-vocabulary logits，以减少 off-policy 静态蒸馏带来的分布偏移。报告还将 1M-context RL framework、Quick Resumption、sandbox 和 rollout data format 作为后训练基础设施来描述，说明 DeepSeek 已经把后训练系统看作和模型架构同等重要的能力来源。

### 2.3 GLM：从 ARC 到 Agentic Engineering

GLM-4.5 把 agentic、reasoning、coding 统一成 ARC 路线。GLM-5 进一步将目标定义为从 vibe coding 到 agentic engineering。其 base model 训练包含 27T corpus，并扩展到 28.5T total token budget。GLM-5 明确把训练拆成 general/coding pretraining 和 agentic/long-context mid-training：先重视 code 和 reasoning，再用 long-context agentic data 将上下文从 4K 扩到 200K。

架构上，GLM-5 扩展到 744B total / 40B active，并采用 DSA 降低长上下文训练和推理成本。报告中还讨论了 MLA 与 GQA 的取舍，并提出 Muon Split 让 MLA 在 Muon optimizer 下稳定训练。后训练上，GLM-5 采用 Reasoning RL -> Agentic RL -> General RL 的顺序，并使用 On-Policy Cross-Stage Distillation 减轻灾难性遗忘。

GLM-5 的另一个特点是异步 RL 基础设施：通过解耦 generation 和 training，提高大规模 agent trajectory 探索时的 GPU 利用率。这与 MiniMax Forge、DeepSeek rollout service、MiMo R3 等形成了同一趋势：RL 系统本身正在成为模型能力的一部分。

更具体地说，GLM-5 是从 GLM-4.5 的 ARC 路线演进而来，但把重心从“agentic/reasoning/coding 三能力统一”推进到“agentic engineering”。预训练从 27T corpus 开始，并有意识地在早期提高 code 和 reasoning 的优先级；mid-training 则分 32K/128K/200K 三段，token 规模分别约为 1T、500B、50B，重点是 long-context agentic data。报告披露了软件工程数据构造：沿用 repo-level code、commit、issue/PR 等开发上下文，过滤后 issue-PR 部分约 160B unique tokens；长上下文数据则混合书籍、论文、通用文档和 synthetic long-context 数据，并在 200K stage 加入少量 MRCR-like 数据以增强多轮/多证据检索能力。

GLM-5 的 attention 演进也值得单独展开。报告先比较 GQA、MLA 与 MLA-256：MLA 在 KV cache 上有优势，但用 Muon 训练时需要 Muon Split，把 projection weights 切成独立矩阵以稳定优化；MLA-256 则在保持效果的同时降低 decoding compute。随后 GLM-5 引入 DSA，并强调它不是从零训练 sparse attention，而是从 dense/MLA base 做 continued pretraining。DSA warmup 阶段冻结 base、训练 indexer，随后 sparse adaptation 只用约 20B tokens 就接近原 MLA long-context 表现。这一结果的意义在于，它提供了一条低成本替换 attention 的工程路线。

后训练方面，GLM-5 的三阶段 RL 顺序非常清晰：Reasoning RL -> Agentic RL -> General RL。Reasoning RL 基于 GRPO，并使用 IcePop 处理 MoE 路由带来的 rollout/training 分布差异；DSA 训练中还发现 indexer top-k 的非确定性会严重影响 RL 稳定，因此默认使用 deterministic `torch.topk` 并冻结 indexer。Agentic RL 部分则采用 fully asynchronous and decoupled RL framework，面向 SWE、terminal 和 multi-hop search 构造超过 10K 个可验证训练环境，并用 Multi-Task Rollout Orchestrator 将不同工具/任务的轨迹标准化为 unified message-list。General RL 负责基础正确性、情商与人类风格偏好，同时使用 On-Policy Cross-Stage Distillation 恢复前序 SFT/RL 阶段学到的能力，缓解多阶段优化造成的遗忘。

### 2.4 Kimi：从 K2 的 agentic intelligence 到 K2.5/K2.6

Kimi K2 是一个 1.04T total / 32B active 的 ultra-sparse MoE。它的亮点不是 thinking model，而是 non-thinking agentic/coding 能力。预训练使用 15.5T tokens，并引入 MuonClip。Muon 本身有 token efficiency 优势，但大规模训练时容易出现 attention logits explosion；Kimi 用 QK-Clip 对 query/key projection weights 进行约束，解决 Muon scaling 的稳定性问题。

后训练方面，Kimi K2 强调 large-scale agentic data synthesis：构造工具、agent、任务和轨迹，并结合 real/synthetic environments。RL 方面，它结合 RLVR 和 self-critique rubric reward，让模型既能从外部 verifier 学习，也能通过 rubric 评价开放式任务输出。这代表了“从静态 imitation 到交互式 agent learning”的转变。

Kimi K2 的架构路线与 DeepSeek/GLM/Qwen 有明显区别：它把模型定位为 open agentic intelligence，而不是显式 thinking model。K2 是 1.04T total / 32B active，hidden dimension 7168，MoE expert hidden dimension 2048，384 experts、top-8、1 shared expert，sparsity ratio 达到 48。报告用 sparsity scaling law 说明，在固定 active FLOPs 下增加总专家数可以降低训练 loss；因此 K2 相比 DeepSeek-V3 的 256 experts 增加到 384 experts，但保持 top-8 不变。与此同时，报告也指出更高稀疏会增加 agentic long-context 场景中的推理 FLOPs/通信压力，所以 K2 并不是无限追求稀疏，而是在训练 loss、active compute 和服务成本之间折中。

Kimi 的 optimizer 贡献适合在综述中单列。Muon 具有更好的 token efficiency，但在大规模 MoE 中可能触发 attention logits explosion。Kimi 提出 MuonClip：在 Muon 中加入 weight decay、RMS matching 和 QK-Clip；QK-Clip 在 query/key projection weights 更新后按 head rescale，使最大 attention logits 不超过阈值。K2 使用 τ=100，15.5T token 预训练全程没有 loss spike；报告还指出 QK-Clip 在训练早期会触发，约 70K steps 后所有 head 的 logits 都已低于阈值，clip 机制基本自停。这是一个典型案例：优化器 innovation 不只是提高收敛速度，也需要包含稳定性约束。

K2 的 agentic data synthesis 管线可以概括为“工具规格 -> agent/task 构造 -> 轨迹生成与验证”。工具来自真实世界工具和 LLM 合成工具，agents 具有不同能力、领域和行为模式，tasks 则覆盖多步工具使用、错误修正、执行反馈和真实交互。RL 部分先用 verifiable rewards gym 覆盖数学、STEM、代码、工具和单元测试可验证任务，再通过 self-critique rubric reward 扩展到创意写作、开放问答、faithfulness 等无法直接验证的任务。K2 的经验表明，agentic post-training 需要同时解决“可验证能力”和“开放式质量判断”，否则模型只能在窄 verifier 任务上变强。

Kimi K2.5 则是 K2 路线向 native multimodal 和 visual agentic intelligence 的延伸。它建立在 Kimi K2 的 1.04T total / 32B active MoE 语言模型上，使用约 15T mixed visual-text tokens 进行 joint pretraining，并通过 MoonViT-3D 将图像和视频放入统一视觉编码空间。K2.5 的一个值得注意的结论是：在固定 vision-text token budget 下，早期、低比例的 vision fusion 比晚期、高比例注入更好；因此它不是把视觉作为后置 adapter，而是在训练全程以较温和比例混合文本和视觉 token。

K2.5 的后训练同样有代表性。它提出 zero-vision SFT，即仅用 text SFT 激活视觉推理和工具使用能力；报告认为，joint pretraining 已建立足够强的 vision-text alignment，人工设计的视觉轨迹反而可能伤害泛化。随后，K2.5 使用 outcome-based visual RL 和 joint multimodal RL，在 visual grounding/counting、chart/document understanding、vision-critical STEM 等任务上训练。一个重要观察是，visual RL 不仅没有损害文本能力，反而提升了 MMLU-Pro、GPQA-Diamond 和 LongBench v2，说明跨模态 RL 可以产生正向迁移。

Agent Swarm 是 K2.5 区别于 K2 的另一个关键。传统 agent 以顺序 tool call 为主，延迟随推理步数线性增长；K2.5 通过 Parallel-Agent Reinforcement Learning (PARL) 训练一个 orchestrator，让它动态创建冻结的 subagents 并并行分解任务。训练时只优化 orchestrator，不端到端更新 subagents，从而降低 credit assignment ambiguity 和训练不稳定性。它还引入 critical steps 作为并行 agent 的资源约束，避免简单增加 subagents 造成 reward hacking。这一路线说明 agentic scaling 不只来自更长 CoT，也可以来自更好的并行任务编排。

Kimi K2.6 进一步强调 long-horizon coding、agent swarm、长时间工具调用稳定性和 multimodal agentic 能力；本文将其作为官方 blog/model-card reference，而不是独立技术报告来处理。

### 2.5 MiniMax：从长上下文 reasoning 到 mini activation agent，再到 MSA 百万上下文

MiniMax-M1 代表长上下文 reasoning 路线，M2/M2.7 则转向 mini activations：229.9B total、仅 9.8B active、192K native context。M2 使用 256 fine-grained experts、top-8 routing、sigmoid gating 和 expert-specific bias，尽量降低 per-token compute。

MiniMax-M2 信息量较高的是数据和 RL 系统。其 agentic coding 数据覆盖 SWE、AppDev 和 terminal interaction。SWE 管线从 GitHub PR 和 issue 中抽取任务，构造 Docker 环境和 test-based reward；AppDev 管线通过 expert meta queries 生成前后端/移动端/桌面应用任务，再用 Agent-as-a-Verifier 从 execution、interaction、visual aesthetics 三层验证；Terminal-Gym 从 StackOverflow 等场景生成可执行终端任务。Forge RL 则支持 white-box 和 black-box agents，并用 windowed-FIFO、prefix-tree merging 和训练/推理/agent 解耦提升长轨迹 RL 稳定性。

MiniMax-M3 最初通过官方 blog/model page 发布，随后 HF model card 披露了更多结构信息并链接 arXiv 技术报告。M3 是约 428B total / 23B active 的 native multimodal MoE，支持 1M context；config 显示其文本模型为 60 层，前 3 层 dense、后续 MoE，使用 128 local experts、top-4 routing、1 shared expert、sigmoid scoring 和 routing bias。它以 coding/agentic、1M context 和 native multimodality 为三个核心能力，并引入 MSA（MiniMax Sparse Attention）。需要注意的是，当前 arXiv PDF 更准确地说是 MSA 技术报告，而不是完整 M3 训练报告：它验证的主实验模型是 109B total / 6B active 的 native multimodal MoE，而非 428B/23B 的生产 M3，因此不能从中直接推出 M3 的完整数据配比、RL pipeline 或 token budget。

MSA 的核心设计是“Index Branch + Main Branch”。Index Branch 在 GQA 上增加轻量 query/key 投影，为每个 GQA group 独立选择 KV blocks；Main Branch 再对被选中的 blocks 做精确 block-sparse softmax attention。报告中的主设置是 block size 128、top-16 blocks，因此每个 query/GQA group 只看约 2048 个 KV tokens，并强制包含局部 self/local block。由于 Top-k 选择不可微，MSA 用 KL alignment loss 让 Index Branch 对齐 full attention 分布，并用 stop-gradient、indexer warmup 和 local block 三个机制稳定训练。109B 实验中，MSA-PT 从头训练，先做 40B-token indexer warmup 再进入 sparse training；MSA-CPT 则从 2.6T full-attention checkpoint 出发，继续 400B tokens，其中前 40B warmup，之后 sparse CPT，并额外用约 140B tokens 做 long-context extension。

从系统角度看，MSA 报告的主要参考价值在于把 attention sparsity 和 kernel co-design 讲清楚：exp-free Top-k、KV-outer sparse attention、query gathering、两阶段 combine、KL loss 的 LSE fusion，以及 backward persistent load balancing。报告称在 1M context 下，109B 实验模型的 per-token attention FLOPs 相比 GQA 降低 28.4x，H800 上 wall-clock speedup 为 14.2x prefill 和 7.6x decode；HF/model card 另称生产 M3 相比 M2 获得 9x prefill、15x decode 加速，并将 per-token compute 降到 1/20。两组数字口径不同，前者来自 MSA 报告实验，后者来自 M3 model card，应分开引用。M3 支持 thinking on/off/adaptive：thinking on 面向复杂推理、agentic tasks 和长程协作，thinking off 面向对话、代码补全等低延迟场景。

### 2.6 LongCat：动态计算 MoE 与多模态分支扩张

LongCat-Flash 是美团 LongCat 系列的语言基座，560B total，动态激活 18.6B-31.3B，平均约 27B。它的两个核心架构创新是 zero-computation experts 和 shortcut-connected MoE。zero-computation experts 允许不同 token 使用不同计算预算；为了控制平均计算负载，LongCat 用 PID-controller 调整 expert bias。负载均衡上，它还引入 device-level load balance loss、router-gradient balancing 和 hidden z-loss。ScMoE 则扩大通信和计算重叠窗口，直接服务训练和推理效率。

训练方面，LongCat-Flash 使用 20T+ tokens，并通过 two-stage pretraining data fusion 提高 reasoning-intensive domain 数据占比。mid-training 强化 reasoning/coding 并扩展到 128K。post-training 使用 reasoning/code、tool use、general capability 多阶段训练，并通过 multi-agent synthesis 生成复杂任务，任务难度由 information processing、tool-set complexity、user interaction 三个轴控制。

LongCat 系列的另一个特点是分支完整：Flash-Thinking、Flash-Prover、Flash-Omni、Next、Image、Video、AudioDiT、Video-Avatar 1.5 覆盖 reasoning、formal proof、omni、图像、视频、音频和数字人。

### 2.7 MiMo：从 7B reasoning 到 V2-Flash 与 V2.5

小米 MiMo 先以 MiMo-7B 展示 reasoning/RL 路线，随后 MiMo-V2-Flash 扩展到 309B total / 15B active。MiMo-V2-Flash 的关键是 hybrid SWA/global attention：8 个 hybrid blocks，每个 block 中 5 个 SWA 层接 1 个 global attention 层，sliding window 为 128。报告显示该设计在长上下文中能显著降低 KV cache 和 attention cost，同时保持甚至提升部分长上下文和 reasoning 任务表现。

训练上，MiMo-V2-Flash 使用 27T tokens，native 32K context，并扩展到 256K。后训练提出 MOPD：先 general SFT，再训练 domain-specific teachers，最后学生模型同时学习 teacher 的 dense token-level reward 和 outcome-based verifiable reward。这个范式试图解决 RL 学习低效和能力不均衡问题，也体现出 teacher/student + on-policy 的新后训练形态。

MiMo 的演进线和其他 1T 级 MoE 不同：它从 MiMo-7B 的 reasoning/RL 经验出发，转向“更小 active params 的高吞吐 agent 模型”。MiMo-V2-Flash 只有 15B active，却用 309B total 保留容量，并通过 48 层结构实现 39 层 SWA + 9 层 global attention。每个 hybrid block 是 5 个 SWA 层加 1 个 GA 层，第一层为 GA + dense FFN 以稳定早期表征；SWA/GA 都配 sparse MoE FFN，MTP block 则刻意使用 dense FFN 和 SWA，让它足够轻量以服务 speculative decoding。

MiMo 对 hybrid attention 的结论适合作为综述案例：SWA 不是窗口越大越好。报告比较 128 和 512 window，发现 128-token window 配合 attention sink bias 在长上下文 extension 与 long-context SFT 后更稳，原因是小窗口迫使 SWA 专注局部依赖，把长程依赖交给 global attention；窗口太大会模糊 local/global 分工。预训练阶段分三段，前 26T tokens 建立 32K native context，最后 1T tokens 做 256K context extension，并上采样 long-range dependency data。报告称在 32K 到 256K retrieval 上接近 100% success rate，这使 MiMo 成为 sparse/hybrid attention 在生产型 agent 模型中的重要样本。

MiMo 的 MOPD 可以看作后训练范式的一个强对照。传统做法要么直接合并多个 teacher 的离线数据，要么用 outcome reward 做稀疏 RL；MOPD 则让 student 采样 on-policy 轨迹，并从 domain teachers 获得 token-level reverse-KL reward，同时可叠加 ORM/GRPO 这类 outcome reward。报告中的 domain teachers 覆盖 math、coding、agentic、instruction following、safety 等领域，MOPD 后多个 benchmark 能接近甚至超过最强 teacher。Agentic RL 方面，MiMo 在大规模 GitHub issues、web development、视觉 verifier、真实/合成环境上扩展，并指出 code-agentic RL 的提升会泛化到其他 agentic 任务、数学和代码任务。MTP 则是训练系统侧的加速器：3-layer MTP 在不同 batch size 下带来最高约 2.6x decoding speedup，对小 batch、长尾 rollout 的 RL 场景尤其有价值。

### 2.8 Microsoft MAI：非开放但重要的 reasoning 对照

MAI-Thinking-1 不属于本文严格意义上的 open-weight 主线，但可放入厂商演进中作为对照样本。一方面，它采用 1T total / 35B active 的 MoE thinking model 设计，训练 token 规模为 30T，和 Kimi K2、DeepSeek-V4、GLM-5 等开放 MoE 模型处于相近的规模区间；另一方面，MAI 的定位更偏 Microsoft 自有平台和 enterprise 场景，强调 clean enterprise-grade data、reasoning-focused post-training 和服务化集成，而不是直接开放完整权重生态。

MAI 报告的独特价值在于，它把模型开发显式描述为一个 hill-climbing machine：由数据管线、训练基础设施、RL 环境、奖励、评测和安全测试共同构成的持续优化系统。其 pretraining 数据使用公开与授权的人类生成数据，包括 web、public GitHub、books、academic papers、news、多语言文本和领域材料；报告明确表示预训练阶段不使用 LLM 生成的 synthetic data，并努力去除 AI-generated content。这与许多开放模型依赖合成数据提升 token efficiency 的路线形成鲜明对照。

MAI-Base-1 是 35B active / 1T total sparse MoE，采用 8/512 experts 激活、local/global attention 以及高稀疏 MoE 与小型 dense FFN 交替的结构。其主预训练为 30T tokens，随后还有 3.55T tokens 的 mid-training，mid-training 后上下文达到 256K。RL 部分则强调从零开始学习 reasoning traces，而不是继承第三方模型蒸馏；它训练 STEM reasoning、agentic coding/tool use、helpfulness/safety 三类 specialist，再合并能力。

MAI 对本文的参考价值主要不在榜单分数，而在于它对 data mixture 与 long-context extension 的实验化描述。数据侧，MAI 把训练数据处理成可解释 bucket，通过质量过滤、精确/模糊/语义去重、跨数据集 drop-order 去重、embedding/metadata 分类和 NLL evaluation suite 来支撑 data mixture search；报告中提到曾在 3 个尺度上训练 183 个小模型、覆盖 61 种 Web/Code/other 混合比例，用小尺度前沿来指导大模型 mixture 选择。长上下文侧，MAI 的经验是：不必在全程用最大上下文训练。最终 recipe 是 16K pretraining、64K mid-training，再用约 140B tokens 做 256K extension；报告中的 ablation 显示，较短上下文 mid-train 后接短程 extension 可以接近更昂贵的长上下文 mid-train，并且 extension 早期 1%-10% 迭代已完成多数 NLL 改善。这一点对综述很重要：long-context training 的核心有时不是学习新知识，而是校准已有表征在新位置分布下的使用方式。

因此，MAI-Thinking-1 在本文中的角色不是“开放模型代表”，而是“非 open-weight 平台模型如何采用同类技术路线”的参照：同样使用大规模 MoE、长推理和后训练系统，但开放程度、数据策略和部署方式与 open-weight 模型明显不同。它也说明，合成数据不是唯一答案；clean human-generated data、可控数据来源和企业级数据治理，仍然可能是模型训练的重要差异化路径。

### 2.9 StepFun：从 Step-3 系统协同到 Step 3.5 agentic Flash

Step-3 是 321B total / 38B active 的多模态 MoE，并配套系统报告。Step 3.5 Flash 则是更偏 reasoning/agentic 的高密度 sparse MoE：196B total / 11B active，使用 3:1 Sliding Window/Full Attention、head-wise gated attention、GQA-8 和 MTP-3，把重点放在多轮 agent 的低延迟执行。STEP3-VL、Step-Audio、Step-Video 则说明 StepFun 在 agent、多模态理解、语音推理和视频生成方向并行推进。

Step-3 可作为解码成本优化案例。它的技术报告不是单纯追求更低 active params，而是明确指出 total params 或 active params 不是解码成本的充分指标。Step-3 使用 Multi-Matrix Factorization Attention (MFA) 降低 KV cache 和 attention cost，同时提出 Attention-FFN Disaggregation (AFD)，将 attention 和 FFN 部署到不同 GPU 子系统上，以便分别匹配计算、带宽和并行策略。这一思路说明：MoE 推理成本不只是模型结构问题，也是 serving topology 问题；过度稀疏的 MoE 在纸面上 active params 很低，但如果不适配硬件计算、内存带宽和网络带宽，未必获得真实成本优势。

Step-3 System 报告把 AFD 进一步落实为可部署系统。Attention instances 负责 attention、KV cache 和非专家计算，FFN instances 负责纯 MoE 计算和 TP/EP 通信；A->F 传输 FP8 token，F->A 返回 BF16 activation，以保留 residual 精度。系统用多阶段 pipeline 重叠 attention、FFN 和通信，并开发 StepMesh 通信库，通过 GPUDirect RDMA、CPU-side 异步通信线程、预注册 tensor、拓扑感知 RoCE 部署和双 NIC 流量均衡来满足约 50ms TPOT 的目标。报告还给出一个具有参考价值的反例：过度追求更稀疏的 FFN 会让 all-to-all、专家不均衡和硬件利用率抵消理论收益，所以 MoE 稀疏度需要和网络、batch、EP/TP 拓扑一起设计。

Step 3.5 Flash 报告补上了 StepFun 在 post-training 和 agent 方向的另一面。它预训练 17.2T tokens，并用 dedicated mid-training 将 context 从 32K 扩到 128K，强化 long-horizon reasoning、code agent、search agent 和 tool-use。后训练采用两阶段 SFT，第一阶段覆盖 Math、Code、STEM、Logic、General QA、Code Agent、Tool-use、Search Agent 和 Long Context Understanding，表中披露 870,687 samples / 7.23B tokens；随后通过 domain-specific RL、self-distillation 和 MIS-Filtered Policy Optimization (MIS-PO) 进行可扩展 RL。MIS-PO 的核心是用 token/trajectory-level binary masking 过滤 off-distribution samples，降低 MoE off-policy 训练的梯度方差和 routing mismatch。该报告可作为“低 active agentic MoE + scalable RL”的案例，与 MiniMax-M2、MiMo-V2-Flash 和 GLM-5 形成对照。

StepFun 在本文中应被视为“工程细节挖掘型”模型家族：Step-3 提供模型-系统协同和 serving topology 的案例，Step 3.5 Flash 提供低 active agentic MoE、SFT/RL 和 MIS-PO 的案例，STEP3-VL/Step-Audio/Step-Video 则提供多模态理解、语音推理和视频生成的专项案例。它的价值不只是榜单分数，而是把模型结构、推理系统、长上下文和 agentic 后训练放在同一条产品化路线里观察。

### 2.10 ERNIE 5.0：统一自回归多模态对照

ERNIE 5.0 虽然不是 open-weight 模型，但它的技术报告可在本文中作为“统一多模态训练”的对照样本。和许多 VLM/omni 模型从已有语言模型出发再接入视觉或音频模块不同，ERNIE 5.0 报告强调从头把 text、image、video、audio 都序列化到统一 token space，并用 Next-Group-of-Tokens Prediction 目标同时训练多模态理解和生成。这使它更接近“一个统一自回归多模态基座”，而不是“语言模型 + 若干模态 adapter”的组合。

架构上，ERNIE 5.0 使用 trillion-parameter ultra-sparse MoE，并采用 modality-agnostic expert routing：router 不根据显式模态 id 分配专家，而是基于统一 token representation 将文本、图像、视频和音频 token 路由到同一个专家池。报告中的专家可视化显示，专家会自然形成跨模态共享与模态/任务特化，而不是人为指定某些专家只服务某个模态。负载均衡上，它采用 auxiliary-loss-free load balancing，并在 8K pre-training 到 32K/128K mid-training 时降低 bias update speed，以缓解大规模 MoE 训练中的 iteration-level oscillation。

训练流程上，ERNIE 5.0 分为 8K pre-training 和 32K/128K mid-training：8K 阶段使用 WSD learning rate schedule，global batch size 从 14M tokens 增至 56M tokens，并从一开始将 RoPE base 设为 1,000,000，方便后续扩长；mid-training 阶段切到 cosine schedule，并把学习率从 1e-4 anneal 到 1e-5。它还提出 elastic training：在同一次预训练中同时训练 elastic depth、elastic width 和 elastic sparsity，让模型可以在不同层数、专家数和 top-k routing 下抽取子模型。报告称，降低 routing top-k 到 25% 可以带来 15%+ decoding speedup；depth/width/sparsity 的 elastic 组合能以 53.7% activated parameters 和 35.8% total parameters 保持接近完整模型的性能。

后训练方面，ERNIE 5.0 采用 SFT + Unified Multimodal RL。它的 RL 细节和 StepFun、MiniMax、GLM 一样值得关注：Unbiased Replay Buffer 用于提高 rollout 效率并避免短样本先进入训练造成的数据难度偏移；Multi-granularity Importance Sampling Clipping、Well-learned Positive Sample Mask 和 Adaptive Hint-based RL 分别处理 entropy collapse、过度优化已掌握样本和稀疏奖励 hard queries。系统侧，它还把 tokenizers 与 MoE backbone 分离部署，使用 FlashMask 支持跨模态异构 attention pattern，并设计 disaggregated RL infrastructure、统一 FP8 stack 和 elastic CPU pooling。对本文来说，ERNIE 5.0 的主要参考价值在于：omni 模型竞争不只是“能听能看能生成”，而是统一 tokenization、MoE routing、elastic serving 和 RL infrastructure 的整体系统问题。

### 2.11 Ant Ling、Nemotron、Gemma、OLMo、Hy3

Ant Ling/BaiLing 和 Ling 2.0 体现了高稀疏 MoE、MTP、FP8、reasoning-oriented data、DFT/Evo-CoT 等方向。Ling 2.6 当前更多依赖 model card 和官方文档，本文暂不按 standalone technical report 级别展开。

Nemotron 3 Ultra/Nano 代表 NVIDIA 在 hybrid Mamba-Transformer、NVFP4、1M context 和长程 agent 推理上的系统路线。Gemma/Ministral 则代表端侧和小模型路线。OLMo 2/3 是 fully open 对照组，其主要价值不是榜单，而是开放数据、训练代码、日志、中间 checkpoint 和 model flow。Hy3-preview 是腾讯最新 MoE 基座之一，295B total / 21B active、256K context，但目前缺独立 tech report PDF。

Gemma 4 虽然没有官方 standalone 技术报告，但官方 model card、release/blog 和 Maarten Grootendorst 的 visual guide 已经足够支撑一个“端侧高效架构”案例。Gemma 4 系列包含 E2B、E4B、12B、31B dense 和 26B A4B MoE：小模型强调 effective parameters / per-layer embeddings，26B A4B 强调 26B total / 4B active 的 MoE，12B 则强调 encoder-free multimodality。它和 Qwen/Kimi/MiniMax 这类 frontier MoE 的差异在于，Gemma 4 的重点不是把 total params 推到 1T，而是在 laptop/mobile/edge 约束下同时处理长上下文、多模态和低延迟 decoding。

架构上，Gemma 4 延续 Gemma 3 的 local/global attention interleaving，但做了几处更面向长上下文和端侧服务的调整：最后一层保证为 global attention；local attention 使用 sliding window，小模型 visual guide 口径为 512，大模型为 1024；global attention 进一步用更高比例的 GQA、K=V 和 p-RoPE 降低 KV/cache 与长距离位置建模成本。多模态侧，普通 Gemma 4 仍可看作带视觉/音频 encoder 的 VLM/omni-ish family，而 Gemma 4 12B 更激进：它把 image/audio 输入改成更轻的 split/project/embed 流程，让非文本 token 更快进入 LLM backbone，因此可与 Qwen3.5-Omni、Kimi K2.5 的“强 encoder + native multimodal training”路线对比。推理侧，Gemma 4 还配套 MTP drafter，用 target activations、KV cache sharing 和 efficient embedder 做 speculative decoding，这让它成为小模型部署章节里一个值得保留的系统优化样本。

需要特别说明的是，Nemotron 这类模型未必是社区讨论热度最高的模型，但它的技术报告包含更细的系统、推理、硬件和多模态工程信息。Nemotron 的 hybrid Mamba-Attention、NVFP4、1M context、LatentMoE 和 long-running agents 设计，为综述提供了超出榜单比较的技术观察，也和 StepFun 一起构成“讨论热度不高但信息密度较高的报告”的代表。

Nemotron 3 Ultra 则适合作为硬件友好训练与推理的案例。它是 550B total / 55B active 的 hybrid Mamba-Attention LatentMoE，108 层、512 experts/layer、top-22、MoE latent size 2048，并使用两层 shared-weight MTP。报告披露了 15T broad/diverse data + 5T high-quality data 的两阶段预训练，phase 1 偏 diversity，phase 2 偏 quality；其公开配比显示 web/synthetic crawl 仍是最大组成，但 code、math、finepdfs、multilingual、legal、SFT-style data 都被系统纳入。1M long-context extension 也很具体：LC phase 只有 33B tokens，其中 long-context data 46%、phase-2 data 54%；92% iteration 用 1,048,576 context，8% iteration 用 4K，并只在 4K iteration 中放 math/code SFT-style data 以维护短上下文 benchmark。它的关键启示是：开放模型的竞争不只在模型权重，还在可复用数据、recipe、低精度训练、推理框架和硬件栈。

## 3. 预训练：token 规模之外的数据工程

公开报告显示，主流 frontier open-weight 模型的预训练 token 数集中在 15T 到 36T：

- Qwen3：36T
- DeepSeek-V4：32T/33T
- GLM-5：27T base，28.5T total
- MiniMax-M2：29.2T
- MiniMax-M3：HF model card 未披露完整 token budget；模型页称重建数据管线并扩展到 100T+ pretraining data；MSA 报告披露的是 109B 实验模型的 3T 训练预算、MSA-CPT 的 400B 继续训练和约 140B long-context extension，而不是生产 M3 的完整训练配方
- MiMo-V2-Flash：27T
- Step 3.5 Flash：17.2T；另有 128K mid-training
- LongCat-Flash：20T+
- Kimi K2：15.5T
- Kimi K2.5：在 K2 checkpoint 上继续约 15T mixed vision-text tokens，另有约 1T ViT training
- Qwen3.5 主线：暂无 standalone general tech report PDF，官方 model card/reference 未披露完整 token budget
- Qwen3.5-Omni：S2 general multimodal pretraining 约 4T tokens，另有 100M+ 小时音视频数据
- ERNIE 5.0：报告披露 trillions of text tokens and multimodal instances，但未给完整 token budget 或跨模态精确配比；作为非 open-weight 统一多模态对照

但 token 数已经不是唯一解释变量。更关键的问题是：哪些 token 在什么阶段出现，是否可验证，是否能产生 agent 轨迹，是否能支持长上下文和工具使用。

从数据来源看，最新模型通常包含以下类别：

1. 通用文本：网页、书籍、百科、论文、论坛、问答。
2. 代码数据：仓库、PR、issue、测试、代码解释、竞赛编程。
3. 数学和 STEM：竞赛题、证明题、逻辑推理、科学问答。
4. 多语言数据：英语、中文和低资源语言。
5. 长上下文数据：长文档、PDF、packed documents、代码仓库、跨文档任务。
6. 合成数据：由上一代模型或 specialist models 生成。
7. 可验证数据：答案、单元测试、Docker 环境、verifier、sandbox。
8. Agentic 数据：工具调用、多轮交互、终端、浏览器、GUI、office workflow。
9. 多模态数据：图文、音频、视频、音视频、GUI、文档和场景数据。

数据配比方面，多数报告只披露类别和阶段重点，不披露完整比例。因此本文避免过度精确化，并将披露程度分为：精确披露、阶段重点披露、仅类别披露、未披露。OLMo 3 在这方面最有研究价值，因为它提供更完整的 data/model flow。

目前公开报告中可直接写入正文的精确或半精确配比如下：

| 模型/阶段 | 可写配比 | 综述意义 |
|---|---|---|
| Qwen3.5-Omni General Stage / S2 | S2 是第二阶段 general multimodal pretraining，约 4T tokens：text 0.92T、audio 1.99T、image 0.95T、video 0.14T、video-audio 0.29T | 少数披露跨模态 token 预算的 omni 报告，可用于分析多模态 mixture |
| Qwen3-Omni AuT | 20M 小时音频，80% 中英伪标 ASR、10% 其他语言 ASR、10% audio understanding | 说明音频 encoder 先被单独做成强通用表征 |
| Qwen3.5-Omni AuT | 40M 小时 audio-text，中文/英文/多语言约 3.5:3.5:3 | 显示从 Qwen3-Omni 到 Qwen3.5-Omni 的多语言音频扩展 |
| Nemotron 3 Ultra pretraining | 15T diversity-biased + 5T quality-biased；phase 1/2 最大项为 quality-filtered/synthetic web crawl，约 49%/38% | 展示 two-phase curriculum 和公开数据 recipe |
| Nemotron 3 Ultra LC phase | 33B tokens；46% long-context data + 54% phase-2 data；92% 1M context + 8% 4K context | 说明 1M extension 可用短而集中的 CPT 完成，同时保留短任务能力 |
| MAI long-context recipe | 这不是数据配比，而是长度阶段 recipe：16K pretraining、64K mid-training、约 140B tokens 的 256K extension；报告未披露各长度阶段的 domain/token 配比 | 支持“短上下文高 MFU mid-train + 末端 extension”的成本经验 |

不同模型的数据哲学正在分化。Qwen、Kimi、MiniMax、LongCat、Nemotron 等模型积极使用合成数据、可验证任务和 agent 轨迹来扩展能力；Qwen3.5-Omni 和 Kimi K2.5 进一步把数据工程推进到跨模态 token 配比、长音视频、GUI/action trajectories、image-code pairs 和 visual agent data；MAI-Thinking-1 则强调预训练不使用 LLM-generated synthetic data，并尽量过滤 AI-generated content；OLMo 追求 fully open data/model flow。这个差异说明，评价数据工程不能只问“数据有多少”，还要问“数据如何被信任、验证、复用和治理”。

## 4. Mid-training：近期值得单独讨论的阶段

Mid-training 或 continued pretraining 已成为连接 base pretraining 和 post-training 的关键阶段。过去的“预训练 -> SFT/RL”二分法已经不够解释最新模型，因为许多关键能力既不是从随机初始化的大规模预训练中直接获得，也不是只靠后训练纠正出来，而是在 base model 已具备通用表征后，通过定向 token budget 做能力再塑形。它往往承担四类任务：扩展上下文、强化代码/推理、引入 agentic workflow，以及适配新的 attention/context 架构。

### 4.1 Mid-training 的边界：不是 SFT，也不只是继续多训

Mid-training 和 SFT 的区别在于训练目标与数据形态。SFT 通常是 instruction/ChatML/trajectory 格式，目标是让模型学会回答协议、工具格式和交互风格；mid-training 仍更接近 language modeling 或 continued pretraining，数据可以是长文档、代码仓库、repo issue-PR 序列、packed papers、合成长依赖样本、agent workflow 文本化轨迹，目标是改变模型的基础分布和上下文利用能力。

它也不是简单“再训练一些 token”。最近报告中的 mid-training 通常有明确设计变量：sequence length 如何从 4K/16K/32K 逐步拉长；是否上采样代码、数学、STEM、长文档、agentic data；是否改变 RoPE base、attention mask、sparse indexer、MTP loss weight；是否保持短上下文 benchmark 不退化。也就是说，mid-training 是 pretraining 与 post-training 之间的“能力定向再预训练”。

### 4.2 长上下文扩展：从全程长训转向末端适配

扩展 context window 是 mid-training 最显性的用途。GLM-5 从 4K 预训练基础上做 32K、128K、200K 三段 mid-training，token 规模分别约为 1T、500B、50B；MiMo-V2-Flash 前 26T token 建立 native 32K context，最后 1T token 扩到 256K；Step 3.5 Flash 在 dedicated mid-training 中从 32K 扩到 128K，并加入 code/search/tool-use 等 agentic domain data；Qwen3.5-Omni 的 S3 从 32,768 提升到 262,144，并提高长音频、长视频样本比例；Nemotron 3 Ultra 在预训练末端用 33B tokens 做 1M LC phase；MAI 则采用 16K pretraining、64K mid-training，再用约 140B tokens 做 256K extension。

这些案例共同指向一个成本原则：不必把整个训练过程都放在最大上下文长度上。长上下文训练的 MFU 低、batch 小、通信成本高，尤其在 MoE 和多模态模型中更明显。更可行的做法是先在较短或中等上下文上学习主体能力，再用相对较少 token 校准 positional/attention 机制。MAI 报告尤其明确：他们尝试过在 256K extension 阶段上调 long-context documents 或调整 domain mixture，但最终发现最简单的方法，即使用上一阶段 mid-training mixture 重新 pack 到 256K，就足够有效；这说明 extension 很多时候是在校准位置外推与检索行为，而不是重新学习完整能力。

### 4.3 数据形态：长文档、代码仓库和 agent workflow 成为主料

Mid-training 的数据不只是“更长的网页”。GLM-5 的 software engineering data 把 repo-level code files、commit diffs、GitHub issues、pull requests 和相关源码文件拼成统一序列，过滤后 issue-PR 部分约 160B unique tokens；这类数据让模型在一个长上下文中看到问题描述、历史修改、文件关系和解决补丁。Long-context data 则混合自然数据和合成数据：自然数据来自书籍、论文和通用预训练文档，合成数据通过构造长程依赖、interleaved packing 和 MRCR-like 变体来缓解 lost-in-the-middle。

MiMo-V2-Flash 的 Stage 2 mid-training 上采样 code-centric data，并加入约 5% synthetic reasoning data；Stage 3 context extension 沿用 Stage 2 数据分布，同时上采样 long-range dependency data。Step 3.5 Flash 的 mid-training 则明确保留 open-domain data，同时混入 code agent、search agent 和 tool-use domain-specific data，把扩长和 agent 初始化绑在一起。Qwen3 的 pretraining S2 也具有 mid-training 的味道：提高 STEM、coding、reasoning 和 synthetic data 比例，再在 S3 做 long-context stage；Qwen3.5-Omni 的 S3 则把 32K 扩到 262K，并提高长音频/长视频比例。Kimi K2 和 MiniMax/LongCat 虽然报告口径不完全相同，但也都把 agentic data synthesis、tool-use demonstrations、代码和 workflow 数据前置到 SFT/RL 之前；Kimi K2.5 进一步把 image-code pairs、GUI/action trajectories、long video 和 visual agent data 放入预训练/延续训练，为后训练提供可学习的多模态 agentic 先验。

### 4.4 架构适配：efficient attention 通常靠 continued pretraining 接入

Mid-training 还是替换 attention 架构的缓冲带。GLM-5 的 DSA 是披露较清晰的例子：它不是从头训练 sparse attention，而是在 dense/MLA base 后做 DSA warmup 和 sparse adaptation。warmup 阶段冻结 base、训练 indexer；随后 sparse adaptation 使用约 20B tokens 让 DSA 接近原 MLA 的 long-context 表现。这个流程说明，efficient attention 的引入可以被看作“架构迁移 + 短程 CPT”，而不是需要重跑完整预训练。

DeepSeek-V4 则把 CSA/HCA、SWA state cache、heterogeneous KV cache 与 1M training/inference 作为整体设计；MiMo-V2-Flash 在 256K extension 阶段调整 GA 的 RoPE base，并降低 expert bias update factor，以稳定长上下文 MoE 路由；Nemotron 的 1M LC phase 明确混合 1M 和 4K iteration，其中 8% 4K iteration 只放 math/code SFT-style data，以维持短任务能力。这些细节说明，long-context mid-training 同时是 attention 校准、position scaling、router 稳定和短长能力兼容的工程问题。

### 4.5 Mid-training 与后训练的接口

Mid-training 的一个关键作用是降低后训练难度。Reasoning RL 如果直接作用在缺乏代码、数学或长上下文先验的 base model 上，rollout 成本会很高，reward 也更稀疏。GLM-5 先通过 agentic long-context mid-training 稳定复杂 workflow，再进入 Reasoning RL、Agentic RL 和 General RL；MiMo-V2-Flash 先通过 Stage 2 强化 code/reasoning，再在 MOPD 中整合 domain teachers；Qwen3 先用 pretraining S2/S3 建立 reasoning/long-context 基础，再用 Long-CoT Cold Start 和 GRPO 提升 thinking 能力；Qwen3.5-Omni 和 Kimi K2.5 则说明，多模态后训练前也需要先通过跨模态 pretraining/mid-training 建立音视频、GUI、视觉 grounding 和工具使用先验，否则 visual/omni RL 会在更稀疏的奖励空间里冷启动。

因此，mid-training 可以被视为“后训练前的能力地形整理”：它让模型在进入 SFT/RL 前已经见过足够多的代码结构、长文档布局、工具轨迹和可验证任务模式。后训练负责把这些潜能变成可控行为，mid-training 则负责让这些行为不至于完全从稀疏奖励中硬学。

### 4.6 小结：mid-training 正在成为模型差异化的主战场

如果预训练决定模型的知识和语言底座，后训练决定交互行为，那么 mid-training 正在决定模型能否承受长上下文、复杂代码库、多文件任务和 agentic workflow。它的核心变量包括：长度 curriculum、domain upsampling、synthetic long-dependency data、repo/issue/PR packing、attention adaptation、RoPE/context scaling、短上下文能力保持和与 RL 的接口设计。

GLM-5、MiMo-V2-Flash、Step 3.5 Flash、MAI-Thinking-1、Nemotron 3 Ultra、Qwen3.5-Omni、Kimi K2.5 和 ERNIE 5.0 构成了 mid-training 的代表性样本。它们分别对应 agentic engineering、hybrid SWA/global、低 active agentic MoE 的 128K 扩长、短上下文高 MFU + 末端 extension、1M LC phase、多模态长音视频扩展、visual agentic context activation/YaRN 扩长，以及统一多模态自回归模型的 8K -> 32K/128K 扩长。

## 5. 后训练与 RL：从偏好对齐到能力生成

后训练的角色正在变化。早期 SFT/DPO 主要负责“让模型会聊天”，而 2025-2026 年的开放模型把后训练变成能力生成系统：SFT 建立格式和冷启动行为，RLVR 提升数学、代码和可验证任务，Agentic RL 训练多步工具执行，Distillation/OPD/MOPD 合并多个专家模型的能力，rollout infrastructure 则决定这些算法能否规模化落地。

一个重要变化是，post-training 不再只是 base model 的末端修饰。对于 Qwen3/Qwen3.5-Omni、DeepSeek-V4、GLM-5、Kimi K2/K2.5、MiniMax-M2、MiMo-V2-Flash、Step 3.5 Flash、Nemotron 3 Ultra 和 MAI-Thinking-1，后训练阶段都被设计成多轮闭环：生成数据、训练 specialist、rollout、验证、过滤、蒸馏、再训练。它和 pretraining/mid-training 的关系也更紧：mid-training 给模型打下代码、长上下文、agent workflow、音视频和 GUI/视觉环境先验，post-training 再通过奖励和交互把这些先验转为稳定行为。

Step 3.5 Flash 也应纳入这个闭环：它先用 unified SFT foundation 建立多域行为，再通过 domain-specific RL 训练 Math、Code、STEM、Tool-use、Long Context、Human Preference 和 Agentic Reasoning 等专家能力，最后以 self-distillation 和 MIS-PO 统一能力。它的特点不是提出一个新 reward，而是把 off-policy MoE RL 的稳定性问题显式化，用 routing confidence 和 MIS filtering 控制训练/推理分布偏移。

### 5.1 SFT：格式、协议与冷启动

SFT 仍然是必要阶段，但它的作用从“教聊天”变成了“建立行为坐标系”。Qwen3 的 Long-CoT Cold Start 用数学、代码、逻辑推理和 STEM 样本建立 reasoning pattern，要求样本配有 verified answers 或 code-based test cases；后续 Thinking Mode Fusion 又用 `/think` 和 `/no_think` 模板把 thinking 与 non-thinking 统一到一个模型中。这里的 SFT 不只是拟合答案，而是在定义模型何时思考、如何暴露思考、如何在 token budget 中停止思考。

DeepSeek-V4 的 SFT 更偏 domain specialization。它先分别训练 math、code、agent、instruction following 等 domain experts，每个专家经过 domain-specific SFT 后再进入 GRPO。这个策略避免把所有能力一开始混在一个模型里竞争，也为后续 OPD 提供 teacher pool。MiMo-V2-Flash 也先进行 general SFT，再训练 domain-specific teachers；Nemotron 3 Ultra 则在 SFT 中保留 shared-weight MTP objective，并加入长上下文 SFT data，为后续 RLVR/MOPD 准备 student。

Agentic 模型的 SFT 还有一个特殊职责：把工具协议和轨迹格式内化。MiniMax-M2、LongCat、GLM-5、Kimi K2 的 agent 数据通常包含工具 schema、环境观察、错误修正、终止条件和最终产物；Step 3.5 Flash 的两阶段 SFT 也明确覆盖 Code Agent、Tool-use、Search Agent 和 Long Context Understanding，第一阶段披露 870,687 samples / 7.23B tokens；Kimi K2.5 的 zero-vision SFT 则说明，在 joint pretraining 已建立视觉-文本对齐后，text-only SFT 也可能激活视觉工具能力，避免低质量人工视觉轨迹伤害泛化；Qwen3.5-Omni 则用 specialist distillation/OPD 合并 text、vision、audio、agentic、coding 和 reasoning teachers。没有这一步，RL 会在极稀疏奖励下同时学习“怎么调用工具”和“怎么解决任务”，样本效率很低。因此，agent/omni SFT 或 distillation 更像 trajectory grammar learning 与 modality interface alignment。

### 5.2 RLVR：可验证任务成为训练燃料

RLVR 的核心是把 reward 从人类偏好扩展到可执行验证。数学、代码、形式证明、终端任务和工具任务都适合 RLVR，因为它们可以构造 verifier：数学标准答案、单元测试、Lean checker、Docker 环境、browser/terminal 状态、artifact validation、页面渲染检查、格式约束等。相比纯 preference reward，RLVR 的优势是信号更稳定、更可扩展，也更容易形成自动化数据飞轮。

Qwen3 使用 query-verifier pairs 并用 GRPO 训练 reasoning；DeepSeek-V4 在每个 domain expert 中使用 GRPO，并将 domain-specific reward models 用于数学、代码、agent 和 instruction following；Kimi K2 的 verifiable rewards gym 覆盖 math、STEM、代码、工具任务和单元测试；Kimi K2.5 进一步使用 outcome-based visual RL 和 joint multimodal RL，reward 覆盖 visual grounding/counting、document/chart/STEM、agentic behaviors 与 PARL 并行调度；MiniMax-M2 从 GitHub PR/issue 构造 SWE 任务，用 Docker 环境和 test-based reward 判断是否修复；MiMo-V2-Flash 把 outcome-based verifiable reward 与 teacher token-level reward 结合；Nemotron 3 Ultra 则进行 multi-environment RLVR，覆盖 reasoning、agentic、code、safety、usability 和 chat。

RLVR 也带来新的风险：reward hacking 和 benchmark leakage。MiMo-V2-Flash 报告中特别提到 SWE-Bench 官方镜像中 ground truth commits 未正确删除可能导致模型利用未清除的目标提交信息，并据此修复训练镜像。GLM-5 在 slides/前端类 RL 中也观察到模型可能通过硬截断、隐藏超长内容等方式利用渲染或规则奖励漏洞。综述中应强调：verifier 越自动化，越需要对环境、镜像、隐藏状态和评分漏洞做审计。

### 5.3 Agentic RL：长轨迹、环境和系统

Agentic RL 的难点不是单步奖励，而是长轨迹环境。模型需要计划、调用工具、观察反馈、修正错误，并在几十到几百步后拿到奖励。各家的解决方案开始系统化：

- GLM-5：异步 RL，generation 与 training 解耦。
- MiniMax-M2：Forge 支持 white-box/black-box agents，windowed-FIFO 缓解轨迹长度方差。
- DeepSeek-V4：million-token RL framework、sandbox infrastructure、fault-tolerant rollout。
- Kimi K2：real/synthetic environments，self-critique rubric reward。
- Kimi K2.5：Unified Agentic RL Environment，text-vision joint RL，PARL 与 Agent Swarm。
- MiMo-V2-Flash：MOPD，将 dense teacher reward 与 verifiable outcome reward 结合。
- Qwen3.5-Omni：Specialist Distillation -> OPD -> Interaction-Aligned RL，专门解决音频查询响应质量和长程语音交互稳定性。
- Step 3.5 Flash：domain-specific RL + self-distillation + MIS-PO，用 token/trajectory-level binary filtering 缓解 off-policy MoE RL 的 routing mismatch 和高方差。
- Nemotron 3 Ultra：SFT -> multi-environment RLVR -> MOPD，把 reasoning、agentic、code、safety、usability 和 chat 环境整合进统一 post-training。
- MAI-Thinking-1：从零学习 reasoning traces，并训练 STEM reasoning、agentic coding/tool use、helpfulness/safety 三类 specialist，再做能力合并。

Agentic RL 的数据来源也在形成几个固定模式。第一类是真实软件工程环境：GitHub issue、PR、commit diff、测试、Docker image、repo 文件检索。MiniMax、GLM、MiMo、DeepSeek 和 MAI 都在使用这一类。第二类是终端/浏览器/搜索环境：terminal tasks、multi-hop search、web browsing、tool calling。第三类是合成 agent 任务：由 LLM 生成工具、用户需求、agent persona、任务约束和验证器。Kimi K2 的 agentic data synthesis、LongCat 的 multi-agent synthesis、GLM-5 的 search task synthesis 都属于这一类。第四类是多模态/GUI/并行 agent 环境：Kimi K2.5 的 GUI screenshots/action trajectories、Agent Swarm prompts 和 subagent scheduling，以及 Qwen3.5-Omni 的 audio-visual tool use/omni interaction，都说明 agentic 数据正在从文本工具轨迹扩展到更丰富的环境状态。

系统上，Agentic RL 的瓶颈已经从算法公式转向 rollout 服务和调度。长轨迹导致 tail latency 极重，环境执行和 reward computation 也会拖慢 GPU。GLM-5 采用 fully asynchronous and decoupled RL，把 generation 和 training 解耦，并用 Multi-Task Rollout Orchestrator 统一不同任务的轨迹格式；MiniMax Forge 用 windowed-FIFO、prefix-tree merging 和训练/推理/agent 解耦处理长轨迹；Kimi K2.5 的 Agent Swarm/PARL 则把系统瓶颈进一步推到并行 subagent 调度、critical steps 和 orchestrator reward；Step 3.5 Flash 把问题落到 off-policy MoE RL 的稳定性，使用 Routing Confidence 估计 rollout/training 分布差异，并用 MIS-PO 过滤不稳定样本；MiMo 使用 R3/Rollout Routing Replay、动态采样和 toolbox/tool manager；DeepSeek-V4 强调 preemptible/fault-tolerant rollout、sandbox 和 Quick Resumption；MAI 则把 RL environments、rewards、evals、infrastructure 和 safety testing 一起纳入 hill-climbing machine。

这说明 agentic capability 很难只由模型参数解释。一个模型是否强，取决于它能否高效地产生长轨迹、保留中间状态、复用 KV/cache、执行工具、恢复中断、审计 reward，并把这些结果重新转化为训练信号。

### 5.4 RL 基础设施：比算法名更接近真实瓶颈

从最近的技术报告看，RL 系统的核心 lessons 可以总结成五条。

第一，rollout 长尾是第一等瓶颈。Agentic/RLVR 任务的响应长度高度不均，少数长轨迹会拖住整个 batch。ERNIE 5.0 的 Unbiased Replay Buffer、MiniMax Forge 的 windowed-FIFO、GLM-5 的异步 generation/training 解耦，本质上都在处理“GPU 等待最长样本”的问题。RL 算法如果不解决吞吐和数据到达顺序，很难稳定扩展。

第二，环境和 verifier 比 reward 公式更重要。SWE、terminal、browser、search、GUI、audio-visual tool use 都需要可复现环境、状态清理、测试执行、artifact validation 和污染审计。MiMo 对 SWE-Bench 镜像泄漏的修复、MiniMax 的 Docker/test-based reward、DeepSeek 的 sandbox 和 fault-tolerant rollout，都说明 verifier 本身是训练系统的一部分。

第三，MoE RL 有额外的训练/推理分布偏移。Step 3.5 Flash 用 routing confidence 和 MIS-PO 过滤 high mismatch 样本，MiMo 用 R3/Rollout Routing Replay 维护 rollout/training route consistency，ERNIE 5.0 也强调统一 FP8 stack 和 rollout/training 数值一致性。对 MoE 来说，off-policy 不只是 token 分布偏移，还包含 routed experts 和低精度执行路径的偏移。

第四，异步 RL 会引入数据分布偏移。异步 rollout 能提高吞吐，但短样本可能更早进入训练，长样本被延后，从而改变训练难度分布。ERNIE 5.0 的 U-RB、GLM-5 的 Multi-Task Rollout Orchestrator、MiniMax Forge 的任务调度，都在试图让吞吐提升不破坏训练分布。

第五，hard query 需要 scaffold，而不是只靠更多 rollout。ERNIE 5.0 的 Adaptive Hint-based RL、Qwen3 的 Long-CoT cold start、MAI 的 specialist climb 和 DeepSeek 的 domain experts 都指向同一个问题：如果 base/SFT 模型在某类任务上 pass rate 太低，纯 RLVR 很容易全零奖励，训练信号会消失。因此，SFT、hint、teacher、domain expert 和 curriculum 不是可省略的前置环节，而是让 RL 真正启动的脚手架。

所以，后训练章节需要保留的核心观点是：RL 不是一个孤立算法模块，而是数据、环境、verifier、rollout scheduler、低精度推理、MoE routing、长上下文缓存和失败样本回流组成的系统。模型是否具备 agentic 能力，很多时候取决于这套系统能否稳定产生高质量训练信号。

### 5.5 Distillation/OPD/MOPD：多专家能力合并成为主线

Distillation 正在从“小模型压缩”扩展为“多能力合并”。Qwen3 的 strong-to-weak distillation 仍然是经典用途：旗舰模型完整走 post-training，0.6B 到 30B-A3B 等小模型通过 off-policy 和 on-policy distillation 获得 thinking/non-thinking 能力，避免每个尺寸都重复完整 RL。到 Qwen3.5-Omni，这一方向变成跨模态 specialist distillation 和 OPD：多个从 Qwen3.5 base fine-tune 的 text/vision/audio/agentic/coding/reasoning teachers 被合并到统一 omni 模型，并把文本条件下更强的响应迁移到音频条件。它的意义不再只是降低模型家族训练成本，也是在解决不同模态条件下能力不一致的问题。

DeepSeek-V4 的 OPD 是另一种用途：合并 domain experts。它先训练多个 specialist，再让统一 student 在自己的 on-policy 轨迹上对齐多个 teacher 的 full-vocabulary logits。这样比静态离线蒸馏更贴近 student 真实分布，也能减轻不同领域能力互相干扰。GLM-5 的 On-Policy Cross-Stage Distillation 则服务于多阶段后训练中的遗忘问题：Reasoning RL、Agentic RL、General RL 顺序优化时，后一个阶段可能损伤前一阶段能力，因此需要跨阶段 teacher 帮模型恢复已学技能。

MiMo-V2-Flash 和 Nemotron 3 Ultra 的 MOPD 更进一步。MOPD 把多 teacher distillation 写成 on-policy RL objective：student 生成轨迹，domain teachers 给 token-level dense guidance，outcome reward/verifier 再提供任务级稀疏奖励。MiMo 把它用于 math、coding、agentic、instruction following、safety 等领域；Nemotron 则用 10+ domain-specialized teachers，通过 MOPD warmup、asynchronous on-policy distillation 和 MTP boosting 整合能力。MOPD 的关键启示是：RLVR 给的 outcome reward 很珍贵但稀疏，teacher logits 能提供密集 token-level credit assignment，两者结合比单独 RL 或单独离线蒸馏更稳。

### 5.6 Reward source：从答案正确到体验与安全

后训练 reward source 可以分成四层。第一层是 rule/programmatic reward：数学答案、格式约束、单元测试、编译、Lean proof、artifact render。第二层是 model-based reward with reference：有标准答案或 rubric，模型 judge 辅助评分。第三层是 model-based reward without reference：开放问答、creative writing、helpfulness、faithfulness、安全等需要偏好模型或 rubric judge。第四层是 interaction reward：多轮对话稳定性、工具调用成功率、用户体验、语音自然度、persona 一致性、语言切换控制。

不同模型在 reward 设计上分化明显。Qwen3 同时使用 rule-based、带参考答案的 model-based reward 和无参考答案的 preference reward；Kimi K2 用 self-critique rubric reward 把 RL 从可验证任务扩展到主观任务；Kimi K2.5 则把 reward 扩展到 visual grounding/counting、document/STEM、GRM 细粒度评分和 PARL 的 parallelism/finish/task outcome；Qwen3.5-Omni 的 Interaction-Aligned RL 针对多轮语音交互中的 code-switching、persona inconsistency 和长上下文指令跟随；Step 3.5 Flash 同时使用 rule/checker、model-based STEM verifier、GenRM/MetaRM 和 agent reward，覆盖数学、代码、工具、搜索、报告生成和偏好任务；MAI 把 helpfulness/safety specialist 与 STEM、agentic coding specialist 并列训练；Nemotron 把 safety、usability、chat 也纳入 multi-environment RLVR。由此可见，post-training 的目标已经从“答案对”扩展到“任务完成、过程可靠、交互自然、安全可控和并行执行效率”。

### 5.7 后训练范式对比

| 范式 | 核心信号 | 优势 | 风险/限制 | 代表模型 |
|---|---|---|---|---|
| Long-CoT cold start | 高质量 CoT/SFT 样本 | 建立推理格式和思考习惯 | 依赖样本质量，容易格式过拟合 | Qwen3 |
| RLVR/GRPO | verifier 或 outcome reward | 可扩展、可自动化、适合数学/代码/视觉 grounding | reward hacking、环境污染、稀疏 credit assignment | Qwen3, DeepSeek-V4, Kimi K2/K2.5, MiMo, Step 3.5 Flash, Nemotron |
| MIS-PO | token/trajectory-level distributional filtering | 稳定大规模 off-policy MoE RL，降低 routing mismatch 和高方差梯度 | 需要估计训练/推理分布差异并设定过滤边界 | Step 3.5 Flash |
| Agentic RL | 长轨迹环境 + 工具反馈 | 训练真实任务执行能力 | rollout 成本高，tail latency 和环境稳定性难 | GLM-5, Kimi K2.5, MiniMax-M2, MiMo, DeepSeek-V4, Step 3.5 Flash, MAI |
| Strong-to-weak distillation | 大模型 teacher 输出/ logits | 降低小模型训练成本 | teacher bias 会被继承 | Qwen3 |
| OPD / specialist distillation | student on-policy 轨迹 + teacher logits/outputs | 减少分布偏移，适合专家合并和跨模态能力合并 | teacher 选择和调度复杂 | DeepSeek-V4, Qwen3 小模型, Qwen3.5-Omni |
| Cross-stage distillation | 前序阶段 teacher | 缓解多阶段 RL 遗忘 | 需要维护多阶段 checkpoint/teacher | GLM-5 |
| MOPD | 多 teacher token-level reward + outcome reward | 密集 credit assignment + 可验证任务信号 | 系统复杂，teacher 质量决定上限 | MiMo-V2-Flash, Nemotron 3 Ultra |
| PARL / Agent Swarm | 并行 subagent 环境 + orchestration reward | 降低复杂任务 latency，扩大 agentic search/execution 范围 | credit assignment 和资源预算更复杂 | Kimi K2.5 |

### 5.8 小结：后训练正在系统工程化

RL 不应只被理解为 GRPO/PPO/DPO 的算法比较。最近一年的模型表明，后训练的关键差异来自五个系统问题：是否有可扩展的 verifier；是否能构造足够多样的 agentic environments；是否能处理长轨迹 rollout 的调度和中断恢复；是否能用 distillation 合并 specialist 而不遗忘；是否能把 reward 从 correctness 扩展到 usability、安全和交互体验。

因此，开放模型的 post-training 已从“偏好对齐阶段”演变为“任务环境驱动的能力生产线”。未来模型的差距很可能不只来自 pretraining token 数，而来自谁能更快地产生高质量任务、验证它们、用 RL 消化它们，并把多个 specialist 的能力稳定合并回统一模型。

## 6. 模型优化：optimizer、learning rate 与能力合并 recipe

最近一年的技术报告有一个明显变化：optimizer、learning rate schedule、辅助损失权重和能力整合策略开始从“附录超参”变成正文技术点。当模型进入 1T total params、高稀疏 MoE、百万上下文和长轨迹 RL 后，训练是否稳定、每个 token 是否高效、多个 specialist 能否被统一到一个可服务模型中，往往不再由一个架构名决定，而是由整套 optimization recipe 决定。

### 6.1 一张 recipe 对照表

| 模型 | Optimizer / LR | 辅助目标与稳定性 | 能力整合方式 |
|---|---|---|---|
| Kimi K2 | MuonClip；WSD：500 steps warmup，前 10T tokens 保持 2e-4，后 5.5T cosine decay 到 2e-5；长上下文 activation 再降到 7e-6 | QK-Clip、weight decay、RMS matching；τ=100 控制 attention logits explosion | post-training 继续使用 Muon；agentic data synthesis + RL |
| DeepSeek-V4 | 大多数模块用 Muon，embedding/output head/mHC/RMSNorm 等保留 AdamW；2000 steps warmup + stable/decay | hybrid ZeRO 支撑 Muon；MTP loss 从 0.3 降到 0.1；mHC/attention/kernel 联动 | domain experts + full-vocabulary OPD |
| GLM-4.5/5 | GLM-4.5 使用 Muon + cosine decay，LR 0 -> 2.5e-4 -> 2.5e-5；GLM-5 引入 Muon Split | Muon Split 稳定 MLA；前 15T 使用 loss-free balance bias，后续关闭 bias update，保留 seq aux loss；MTP 0.3 -> 0.1 | GLM-5 用 on-policy cross-stage distillation 缓解阶段遗忘 |
| ERNIE 5.0 | 8K pretraining 用 WSD，warmup 到 1e-4 后保持；32K/128K mid-training 切 cosine，1e-4 -> 1e-5 | bias update speed 1e-4 -> 1e-5；MTP 0.3 -> 0.1；posterior-based loss weighting 平衡多模态 loss | SFT + unified multimodal RL；elastic training 支撑多部署形态 |
| MAI-Thinking-1 | AdamW；pretraining warmup 约 12B tokens 后 cosine，2e-4 -> 2e-5；RL climb constant 1e-6，长输出降到 9e-7 | grad clip、dropout 0.15；self-distillation 和 RL 使用不同 load-balance coefficient | self-distillation 携带不稳定 RL run 的进步，再 consolidation SFT 合并 specialist |

### 6.2 三个判断

第一，Muon 已经进入 frontier-scale 训练，但没有成为直接替代 AdamW 的答案。Kimi K2 用 MuonClip/QK-Clip 处理 attention logits explosion；DeepSeek-V4 保留 AdamW exceptions 并为 Muon 设计 hybrid ZeRO；GLM-5 用 Muon Split 解决 MLA 在 Muon 下的稳定性问题。也就是说，optimizer innovation 需要和 attention、residual、kernel、ZeRO sharding 一起看。

第二，LR schedule 越来越阶段化。Kimi K2 和 ERNIE 5.0 保留 WSD 或 stable stage，GLM-4.5 则明确改用 cosine 并指出 WSD 在一些 general benchmark 上 underfit；MAI 把 pretraining、mid-training、RL climb 和 self-distillation 的 LR 分开设置。对长 rollout RL 来说，LR 甚至要随最大输出长度和 off-policiness 调整。

第三，辅助目标不是小细节。MTP loss weight 从 0.3 降到 0.1，在 GLM、ERNIE、DeepSeek 中都出现；MoE balance bias、seq aux loss、load-balance coefficient 也会按阶段启停或降权。多模态模型还需要处理不同模态 loss scale，ERNIE 5.0 的 posterior-based loss weighting 就是这个方向的例子。

### 6.3 小结：优化 recipe 是训练系统的控制面板

如果把架构看作“模型能表达什么”，优化 recipe 更像“模型是否能稳定到达那里”。主文中不必展开所有 OPD/MOPD/consolidation 细节，因为这些已经放在后训练章节讨论；但在优化章节里需要保留一个判断：能力整合不是简单参数融合，而是依赖 on-policy freshness、teacher scheduling、loss balance、LR 和遗忘控制的训练过程。

## 7. MoE 架构与负载均衡

MoE 的目标是用较低 active params 获得大模型容量。但最近一年的报告说明，MoE 已经不只是“把 FFN 换成专家层”。它同时涉及参数规模、激活预算、路由函数、负载均衡、专家通信、低精度训练、serving topology 和 RL rollout 一致性。一个 MoE 模型的真实效率，往往取决于 active params 能否转化为硬件上的高 MFU，而不是纸面上激活了多少参数。

### 7.1 从 total params 竞争转向 active params 预算

Frontier open-weight 模型已经形成一个清晰区间：total params 常在数百 B 到 1T+，active params 多在 10B-50B。DeepSeek-V3 是 671B/37B，DeepSeek-V4-Pro 是 1.6T/49B，Kimi K2/K2.5 沿用 1.04T/32B active lineage，GLM-4.7 是 355B/32B，GLM-5 是 744B/40B，Qwen3-235B-A22B 是 235B/22B，Qwen3.5-397B-A17B 是 397B/17B，LongCat-Flash 是 560B total 且动态激活约 18.6B-31.3B，Nemotron 3 Ultra 是 550B/55B，MiMo-V2-Flash 是 309B/15B，MiniMax-M2 是 229.9B/9.8B，MiniMax-M3 是约 428B/23B。Qwen3.5-Omni 这类 omni MoE 报告虽然没有单独给出 Plus/Flash 的 total/active 参数表，但明确继承 Qwen3.5 Hybrid MoE，并把 MoE 扩展到 Thinker/Talker、长音视频和流式语音生成。这个区间反映出一个共同目标：用接近 dense 10B-50B 的 per-token compute 获取更接近大模型的知识容量。

| 模型 | Total / Active | 专家与路由 | MoE 设计重点 |
|---|---:|---|---|
| DeepSeek-V3 | 671B/37B | DeepSeekMoE，fine-grained routed/shared experts | MLA、MTP、FP8 mixed precision 与 aux-loss-free load balancing，奠定 V4 前的高效 MoE 基线 |
| DeepSeek-V4-Pro/Flash | 1.6T/49B；284B/13B | DeepSeekMoE | 继承 fine-grained MoE 与 aux-loss-free 路由，配合 CSA/HCA 降低长上下文成本 |
| Kimi K2/K2.5 | 1.04T/32B lineage | 384 experts，top-8，1 shared expert | ultra-sparse MoE，K2 强调 sparsity scaling/MuonClip；K2.5 复用 backbone 并转向 visual agentic/PARL |
| GLM-4.7 | 355B/32B | GLM-4.x MoE；公开模型表标注 355B-A32B | 继承 GLM-4.5/4.6 ARC 路线，强化 agentic coding、tool use、Interleaved/Preserved/Turn-level Thinking |
| GLM-5 | 744B/40B | 256 experts，top-8，1 shared expert | 扩专家数但减少层数到 80，以降低 expert-parallel 通信 |
| Qwen3-235B-A22B | 235B/22B | 128 experts，top-8，无 shared expert | Qwen3 MoE 取消 Qwen2.5-MoE 的 shared expert，并使用 global-batch load balancing loss |
| Qwen3.5-397B-A17B / Qwen3.5-Omni | Qwen3.5 主线 397B/17B；Omni Plus/Flash 未单独披露 total/active | Gated DeltaNet + sparse MoE；512 experts，10 routed + 1 shared；Omni 的 Thinker/Talker 均为 Hybrid-Attention MoE | Qwen3.5 主线以 Gated DeltaNet + MoE 提高吞吐；Omni 进一步用 GDN、ARIA、多码本 codec 与 chunked prefill 支撑 256K omni context、长音视频和低延迟语音输出 |
| Step 3.5 Flash | 196B/11B | 45 layers；42 MoE layers；288 routed experts + 1 shared；top-8 | 低 active agentic MoE，3:1 SWA/full attention、head-wise gated attention、EP-level balance loss 和 MTP-3 共同降低多轮 agent 延迟 |
| LongCat-Flash | 560B/18.6B-31.3B | zero-computation experts + FFN experts | token-level dynamic compute allocation |
| Nemotron 3 Ultra | 550B/55B | LatentMoE，512 experts/layer，top-22 | 用 latent bottleneck 提升 accuracy per parameter |
| MiMo-V2-Flash | 309B/15B | 256 experts，top-8，无 shared expert | 小 active MoE + hybrid SWA/global + MTP |
| MiniMax-M2 | 229.9B/9.8B | 256 fine-grained experts，top-8 | mini activation，sigmoid gating，expert-specific bias |
| MiniMax-M3 | 约 428B/23B | 128 local experts，top-4，1 shared expert；sigmoid scoring + routing bias | MSA + native multimodality + 1M context；7 个 MTP modules；前 3 层 dense、后续 MoE；MSA 报告的主实验为同路线 109B/6B 模型 |
| Gemma 4 26B A4B | 26B/4B | 128 experts，top-8，1 shared expert | 面向端侧/本地高效推理的小 active MoE；与 E2B/E4B/12B/31B dense 共同组成混合产品线 |

这张表里需要注意的是，active params 并不是越低越好。Gemma 4 26B A4B 代表 4B active 的端侧 MoE，MiniMax-M2、Step 3.5 Flash、MiMo-V2-Flash 和 Qwen3.5-397B-A17B 追求 10B-17B active 的较低推理成本，适合高吞吐 agent/coding/多模态服务；Qwen3-235B-A22B、DeepSeek-V3、Kimi K2/K2.5、GLM-4.7 和 GLM-5 保持 20B-40B active，以换取更强通用/agentic 能力；Nemotron 3 Ultra 则用 55B active 与 LatentMoE、NVFP4 和 Mamba-Attention 共同优化吞吐/精度边界。Qwen3.5-Omni 则说明 MoE 的比较不能只看语言 backbone 的 active params：omni 模型还要把稀疏计算、长序列 attention、音视频 encoder、codec 和流式生成拆成一个协同系统。也就是说，MoE 的设计目标正在分化为端侧小 active、mini activation、frontier sparse capacity、hardware-friendly sparse serving 和 multimodal/agentic orchestration 五条路线。

### 7.2 专家粒度：fine-grained、ultra-sparse、dynamic compute、LatentMoE

第一条路线是 fine-grained experts。DeepSeekMoE、MiniMax-M2/M3、MiMo-V2-Flash、GLM-5 都使用较多小专家和 top-k routing。它的好处是专家更容易分工，active compute 可控；难点是 all-to-all 通信和 router 稳定性。MiniMax-M3 相比 M2 从 256 experts/top-8 转为 128 local experts/top-4 + 1 shared expert，active params 从 9.8B 提高到约 23B，同时把效率重心从极低 active MoE 转向 MSA 百万上下文。Kimi K2 在此基础上进一步走 ultra-sparse：384 experts、top-8，sparsity ratio 为 48。报告认为在 fixed active parameters / constant FLOPs 下，增加专家总数能降低训练 loss，但也承认更高稀疏会给长上下文 agent 服务带来额外通信和推理压力。Kimi K2.5 继承这个 MoE backbone 后，把效率问题从单模型稀疏计算扩展到 MoonViT-3D、DEP 和 Agent Swarm/PARL，说明新一代 MoE agent 的效率不只由 router 决定，也由视觉 encoder、上下文长度和并行任务编排共同决定。

第二条路线是 dynamic compute。LongCat-Flash 的 zero-computation experts 具有代表性：部分专家不做真实 FFN 计算，相当于允许不同 token 使用不同计算预算。它不只是节省 FLOPs，也是在让模型根据 token 难度分配计算。为了控制平均激活和稳定训练，LongCat 使用 PID-controller 调整 expert bias，并配合 device-level balance loss、router-gradient balancing 和 hidden z-loss。

第三条路线是 LatentMoE。Nemotron 3 Ultra 和 MAI-Base-1 都采用 LatentMoE，但具体配置不同。Nemotron 3 Ultra 是 512 experts/layer、top-22、MoE latent size 2048；MAI-Base-1 是 8/512，并明确说明 routing 基于原始 representation，dispatch 前做 shared down-projection，combine 后再投回原维度。LatentMoE 的核心不是“更多专家”本身，而是把专家计算放在压缩 latent space 中，用较低专家计算成本换取更高专家容量。

### 7.3 路由与负载均衡：aux loss 之外的阶段化策略

负载均衡是 MoE 训练的核心问题。传统 auxiliary load balancing loss 可以让专家利用率更均匀，但可能干扰主语言建模目标，因此新模型开始减少对 aux loss 的依赖。DeepSeek-V3/V4 使用 aux-loss-free load balancing；MiniMax-M2 使用 expert-specific bias 和 sigmoid gating；LongCat 用 PID expert bias 控制平均计算负载；MiMo-V2-Flash 使用 sequence auxiliary loss 和 expert bias update，并在 256K extension 阶段降低 expert bias update factor；MAI 强调 global-batch load balancing，即跨 DP workers 和 micro-batches 聚合 expert frequency。

GLM-4.5 是阶段化负载均衡的一个简洁例子：架构上采用 loss-free balance routing 和 sigmoid gates，但训练 recipe 中只在前 15T tokens 以 0.001 更新 balance bias，后续关闭 bias update，并保留 0.0001 权重的 sequence-level balance loss 来抑制单序列内的极端不均衡。也就是说，它不是全程依赖 loss-free balancing，而是在 pre-training 前段和后续阶段切换了 MoE 稳定策略。

另一个容易混淆的点是 gating 函数本身。Softmax gating 会让专家在同一个概率单纯形上竞争，天然有“此消彼长”的归一化约束；sigmoid gating 更像对每个专家独立打分，再做 top-k 选择或归一化，通常更容易和 expert bias、aux-loss-free balancing 搭配，允许多个专家同时获得较高 affinity。下表把技术报告明确披露和 HF config / Transformers modeling 代码可推断的 gating 选择合并整理；代码推断项在备注中标明，仍未确认的模型不做确定性归类。

| 模型/家族 | Gating 选择 | 负载均衡搭配 | 依据/备注 |
|---|---|---|---|
| DeepSeek-V3/V4 | sigmoid gating / top-k affinity normalization | auxiliary-loss-free bias + sequence-wise aux loss | DeepSeek-V3 报告明确从 softmax 改为 sigmoid；V4 延续 aux-loss-free 路线 |
| GLM-4.5 | sigmoid gates | 前 15T 更新 loss-free balance bias，后续关闭；保留 seq aux loss | 报告披露；HF config 只显示 top-8、norm_topk_prob、160 routed experts + 1 shared expert，不单独标注 scoring_func |
| MiniMax-M2 | sigmoid gating | expert-specific bias | 报告披露；HF config 同时标注 `scoring_func=sigmoid` 与 `shared_moe_mode=sigmoid` |
| MiniMax-M3 | sigmoid scoring/gating | routing bias；MSA 报告不披露 MoE load balancing 细节 | HF config 披露 `scoring_func=sigmoid`、`use_routing_bias=true`、128 local experts、top-4、1 shared expert；PDF 重点是 attention indexer 而非 MoE router |
| MAI-Thinking-1 / MAI-Base | softmax gating | global-batch load balancing loss；强调跨 DP worker 和 micro-batch 聚合 expert frequency | 报告披露；LatentMoE，8/512 experts |
| LongCat-Flash | softmax router | PID-controlled expert bias + device-level balance loss + router-gradient balancing + hidden z-loss | Transformers modeling 显示 `router_logits.softmax(dim=-1)` 后 top-k；zero-computation experts 让激活计算量动态变化 |
| Qwen3 MoE | softmax gating + top-k normalization | global-batch load balancing loss | Transformers `qwen3_moe` modeling 显示 `softmax(router_logits)` 后 top-k；HF config 披露 128 experts、top-8、`norm_topk_prob=true`、无 shared expert |
| Qwen3.5-397B-A17B / Qwen3.5-Omni | softmax gating + top-k normalization | router aux loss；共享专家另有 gate | Transformers `qwen3_5_moe` modeling 显示 `softmax(router_logits)` 后 top-k 并归一化；HF config 披露 512 experts、top-10、1 shared expert、Gated DeltaNet + sparse MoE |
| Kimi K2/K2.5 | sigmoid scoring/gating + top-k normalization | ultra-sparse top-8/384；训练稳定性重点在 MuonClip/QK-Clip | Kimi K2 HF config 标注 `scoring_func=sigmoid`、`norm_topk_prob=true`、`topk_method=noaux_tc`；K2.5 复用 K2 MoE backbone |
| Step 3.5 Flash | sigmoid router activation | loss-free load balancing + EP-level balancing loss；RL 阶段用 Routing Confidence/MIS-PO 约束 routing mismatch | HF config 标注 `moe_router_activation=sigmoid`、`norm_expert_weight=true`、`use_moe_router_bias=true`、FP32 gate |
| MiMo-V2-Flash | sigmoid scoring/gating + top-k normalization | sequence aux loss + expert bias update；RL 关注 rollout-router 一致性 | HF config 标注 `scoring_func=sigmoid`、`norm_topk_prob=true`、`topk_method=noaux_tc`；报告强调 FP32 router |
| Nemotron 3 Ultra | 未确认 softmax/sigmoid；已确认 top-22 + top-k normalization | 关注 MaxVio/dead experts、hot expert replication 与 LatentMoE 专家健康 | HF config / model card 披露 512 routed experts、1 shared expert、top-22、`norm_topk_prob=true`、LatentMoE；未在公开 config 中标注 scoring_func |

负载均衡方法可以分成几类：

- auxiliary load balancing loss：传统方法，但可能影响主目标。
- aux-loss-free strategy：通过 bias 或 routing 调节减少 auxiliary loss。
- stage-dependent balancing：GLM-4.5 显示 pre-training 与 mid-training 可采用不同策略，前段使用 loss-free balance bias update，后续关闭 bias update 并保留 seq aux loss。
- expert bias：直接在 router score 上调节专家利用率。
- PID-controller：LongCat 用控制论方式稳定平均激活专家数。
- device-level balance loss：防止 EP group 的极端不均衡。
- router-gradient balancing / z-loss：抑制 router 或 hidden activation 异常。
- deterministic computation：用于复现和 SDC 检测。
- global-batch load balancing：MAI 报告认为 aggregation strategy 比 load-balancing loss 类型更关键，需要跨 DP worker 和 micro-batch 聚合 expert frequency。
- dropless MoE：MAI 的经验是有限 capacity/token dropping 会改变负载均衡实验结论，因此收敛到支持 variable all-to-all message size 的 fully dropless MoE。
- EP-level balancing loss：Step 3.5 Flash 在 loss-free load balancing 之外显式加入 rank-level utilization 约束，说明专家均衡已经从“专家频率均匀”扩展到“EP rank 负载均匀”。
- routing confidence / MIS filtering：Step 3.5 Flash 把 MoE routing 不确定性和 off-policy RL 高方差联系起来，用 activated experts 的概率质量作为稳定性 proxy，再通过 token/trajectory-level binary mask 过滤高 mismatch 样本。
- MaxVio/专家健康监控：Nemotron 使用 MaxVio 监测专家最大负载相对均值，并关注 imbalanced/dead experts，说明 MoE 训练健康度已成为持续监控指标。
- LatentMoE：以 Nemotron 3 Ultra 和 MAI-Base-1 为代表，关注 expert capacity 与 latent bottleneck 的结合，重点不是负载均衡技巧本身，而是提高 MoE 参数效率和硬件友好性。
- modality-agnostic routing：ERNIE 5.0 将文本、图像、视频和音频 token 送入同一个专家池，专家特化由 token representation 和任务需求自然形成，而不是由人工指定模态专家；这对 omni MoE 的 router 设计有参考意义。
- elastic depth/width/sparsity：ERNIE 5.0 在一次预训练中训练可裁剪的层数、专家宽度和 routing top-k，使同一个 super-network 能服务不同延迟/内存预算，这比单纯比较 total/active params 更接近部署侧问题。

### 7.4 MoE 通信与推理系统：all-to-all 是真实瓶颈

MoE 的服务成本常常被 active params 低估，因为专家并行引入 all-to-all 通信、专家热度不均、batch 形态变化和跨节点网络瓶颈。Step-3 System 报告虽然重点是 AFD，但它的观点对所有 MoE 都成立：过度稀疏的 FFN 如果不适配硬件计算、内存带宽和网络带宽，理论 FLOPs 优势可能无法兑现。AFD 将 attention 和 FFN 分开部署，目的之一就是让 FFN 在合适 batch 下保持高 MFU，同时让 attention 独立扩展以处理动态 context length。

LongCat 的 ScMoE 试图扩大通信和计算重叠窗口；DeepSeek-V4 通过 fused MoE kernel 重叠 compute/communication/memory；GLM-5 把专家数扩到 256 但减少层数到 80，以降低 expert-parallel 通信；Nemotron 3 Ultra 在 wide expert parallelism 下讨论 hot expert replication 和 per-rank token balance；ERNIE 5.0 通过 tokenizer-backbone disaggregation 将多模态 tokenizer 从 MoE backbone 中拆出独立部署，并使用 64-way expert parallelism、context parallelism、FlashMask 和统一 FP8 stack 支撑多模态/长上下文/RL 训练；MiMo 的 R3/Rollout Routing Replay 则说明，RL 训练中还要保持 rollout 与 training 的 routed experts 一致，否则 MoE 路由差异会放大 off-policy 问题。

### 7.5 MoE 与低精度训练/推理

MoE 也在推动低精度训练和推理。DeepSeek-V3/V4 以 FP8/FP4 路线降低训练和推理成本；Nemotron 3 Ultra 使用 NVFP4 pretraining 和 NVFP4 checkpoint，并保留 final layers、部分 projections、MTP 和 embedding 的较高精度；Step-3 使用 full FP8 quantization 讨论 Hopper 上的 decoding throughput；GLM-5 在部署侧讨论 INT8/W8A8 与专家压缩；MiMo-V2-Flash 结合 FP8 mixed precision 和 MTP。低精度对 MoE 的挑战在于 router、专家权重、activation scale 和 communication dtype 都会影响稳定性，因此不能只看 GEMM precision。

### 7.6 小结：MoE 的下一步不是更多专家，而是更可服务的稀疏性

未来 MoE 研究的重点可能不是单纯增加专家数，而是让稀疏性更可训练、更可服务、更适配 RL。MoE 的核心问题可以归纳为五个：第一，active params 与 total params 的效率边界；第二，router 和负载均衡是否稳定且不损害主目标；第三，all-to-all 通信能否被重叠、压缩或拓扑化；第四，低精度专家和 router 是否稳定；第五，MoE 在 agentic RL 中能否保持 rollout/training 一致性。

因此，2025-2026 年的开放模型已经从“MoE 参数扩容”进入“稀疏计算系统工程”。谁能把路由、通信、低精度、attention 和 serving topology 一起设计，谁就更可能把纸面 active params 转化为真实吞吐和可持续训练效率。

## 8. 长上下文与 attention 设计

长上下文能力可以拆成四个问题：训练时如何看到长序列，推理时如何承担 KV cache，模型如何保留远距离信息，agent/RL rollout 如何负担长轨迹。最近一年的模型显示，最大 context window 本身已经不是最关键指标；关键在于每个输出 token 的成本、远程信息检索能力、长程 reasoning 稳定性、cache 复用和多轮 agent 的可持续运行。

### 8.1 长上下文不是单一能力

长上下文至少包含五种能力。第一是 retrieval：能否从几十万到百万 token 中找回证据。第二是 reasoning over context：能否对长文档、代码库或多轮轨迹做归纳、比较和推理。第三是 generation under long prefix：长 prefix 下输出质量是否稳定。第四是 long-horizon agent state：工具观察、错误、文件状态和中间计划能否被保留。第五是 serving efficiency：KV cache、attention FLOPs、prefill/decode latency 和吞吐能否支撑真实业务。

因此，综述中不应只比较“128K/256K/1M”。例如 Qwen3 的 RULER 结果显示 thinking mode 在极长输入上可能因 verbose reasoning 干扰 long-context 表现；MAI 的实验显示长上下文 extension 很快改善 NLL，说明很多长上下文能力是在校准位置和 attention 使用；DeepSeek-V4、Step-3、Nemotron 则直接把 long-context 讨论落到 KV/cache 和 decoding throughput 上。

### 8.2 Attention 路线对比

| 路线 | 代表模型 | 核心思想 | 主要收益 | 主要代价 |
|---|---|---|---|---|
| MLA / low-rank KV | DeepSeek, Kimi, GLM | 压缩 KV 表示，降低 cache | 长上下文推理省内存 | decoding compute 和实现复杂度需要优化 |
| CSA/HCA | DeepSeek-V4 | compressed sparse + heavily compressed attention | 1M context 下显著降低 FLOPs/KV cache | heterogeneous KV cache 管理复杂 |
| DSA | GLM-5 | content-based sparse attention，通过 continued pretraining 接入 | 比固定 sparse pattern 更接近 dense 表现 | indexer/top-k 确定性影响 RL 稳定 |
| MSA | MiniMax-M3 | GQA-based block sparse attention；Index Branch 为每个 GQA group 选择 top-16 个 128-token blocks，Main Branch 对选中 blocks 做精确 sparse softmax；每个 query/group 约 2048 KV tokens | MSA 报告称 1M 下 attention FLOPs 降低 28.4x，H800 上 14.2x prefill、7.6x decode；M3 model card 另称相比 M2 为 9x prefill、15x decode | 需要 KL alignment、stop-gradient、40B-token indexer warmup 和 local block 稳定训练；报告主体是 109B/6B 实验模型，不是完整 M3 recipe |
| Hybrid SWA/global | MiMo-V2-Flash, MAI | 多个局部层配一个全局层 | 训练/推理成本低，长上下文可扩展 | 需要 attention sink、RoPE 和层比例调参 |
| Full attention + GQA | MiniMax-M2 | 保留 full attention，靠系统和 GQA 控成本 | 可靠、简单、适合复杂 agent workload | 超长上下文成本高 |
| Hybrid Mamba-Attention | Nemotron 3 Ultra | 状态空间层承担长序列状态，attention 作为 anchor | decode cost 对长度更友好 | Mamba cache、rollback、prefix reuse 更复杂 |
| MFA + AFD | Step-3 | 降低 KV/attention cost，并把 attention/FFN 分开服务 | decoding cost 与 serving topology 联合优化 | 系统实现复杂，依赖网络和部署拓扑 |
| 3:1 SWA/full + head-wise gated attention | Step 3.5 Flash | 三个 SWA 层配一个 full attention 层，SWA window 512，并用 head-wise gate 补偿 SWA 质量损失 | 大幅降低长上下文 prefill/decode cost，适合多轮 agent 和 MTP-3 speculative decoding | 比全 attention 需要更细的 head 数、RoPE 和 MTP 配置调参 |
| Hybrid-Attention MoE + GDN | Qwen3.5-Omni | Thinker/Talker 都采用 Hybrid-Attention MoE，Gated Delta Net 负责更高效的长音视频序列建模 | 支持 256K、10 小时以上音频和约 400 秒 720P 视频输入 | 需要同时处理跨模态同步、流式输入输出和 Talker 低延迟生成 |
| Local/global interleaving + p-RoPE | Gemma 4 | local sliding-window attention 与 global attention 交错，最后一层为 global；global attention 使用 K=V 和 p-RoPE | 在 128K-256K 长上下文和端侧内存约束下保持可服务性 | 不是内容自适应 sparse attention，长程信息仍依赖周期性 global 层整合 |

这些路线的分歧可以理解为不同约束下的最优解。DeepSeek-V4 面向 1M context 和低 KV cache，选择 CSA/HCA；GLM-5 需要从已有 MLA base 平滑迁移到 efficient attention，选择 DSA continued pretraining；MiniMax-M3 则从 M2 的 full attention + GQA 转向 MSA，把百万上下文直接绑定到 coding/agentic 和 native multimodality。MSA 和 DSA/MoBA 的差异也值得单独标出：MSA 是按 GQA group 做 block-level Top-k selection，DSA 更接近 MLA/MQA 场景下的 token-level/shared-index selection，MoBA 则依赖更粗的 block/key aggregation；因此 MSA 的设计目标不是单纯“稀疏”，而是在训练期就学习每个 GQA group 的长程 block routing，并通过 kernel 规避 irregular sparse attention 的系统开销。MiMo-V2-Flash 追求 15B active 的高吞吐 agent 模型，选择 5:1 SWA/global；Step 3.5 Flash 同样追求低 active agent 延迟，但选择 3:1 SWA/full、head-wise gated attention 和 MTP-3；Gemma 4 则更像端侧/本地长上下文路线，用 local/global interleaving、K=V global attention、p-RoPE 和 MTP drafter 把成本压到消费级设备可以承受的范围。MiniMax-M2 更看重生产级 agent/coding 的可靠性，因此保留 full attention + GQA；Nemotron 走硬件友好 Mamba-Attention，并把 NVFP4、MTP 和 TRT-LLM 一起优化；Step-3 则把 attention 设计直接绑定到 AFD serving。Qwen3.5-Omni 提醒我们，长上下文 attention 已经不只服务文本，还要在同一个 Thinker/Talker 系统中处理长音频、长视频和流式语音输出。

Kimi K2.5 的路线不应被归入单一 attention 范式。它更像跨模态 context pipeline：在 K2 MoE backbone 外接 MoonViT-3D、NaViT packing 和 4-frame video compression，并用 DEP 解耦视觉 encoder 训练。其长上下文效率取决于视觉 encoder、packing、YaRN 扩长和 Agent Swarm/PARL 调度的整体配合，而不是某个 attention kernel 本身。

### 8.3 Context extension recipe：长度 curriculum 比单点窗口更重要

第 4 节已经讨论过 mid-training，但在长上下文章节还需要强调 recipe。GLM-5 的 32K/128K/200K 三段式，MiMo-V2-Flash 的 32K native + 256K extension，Step 3.5 Flash 的 32K -> 128K agentic mid-training，Qwen3.5-Omni 的 32,768 -> 262,144，Kimi K2.5 的 15T vision-text @4K 后接 long-context activation/YaRN 顺序扩长，Nemotron 3 Ultra 的 33B-token 1M LC phase，MAI 的 64K mid-training + 140B-token 256K extension，都表明模型通常通过 curriculum 而不是一次性跳到最大长度。

长度 curriculum 的核心是平衡三件事：长上下文能力、短上下文能力和训练效率。Nemotron 在 1M LC phase 中保留 8% 的 4K iteration，并只放 math/code SFT-style data，目的就是维持短任务指标；MAI 最终没有为 256K extension 特意改变 domain mixture，而是把上一阶段 mixture repack 到目标长度；MiMo 在 Stage 3 调整 GA RoPE base 并降低 expert bias update factor，以避免长上下文 extension 破坏 MoE 路由。这些细节说明，context extension 不是只改 RoPE scale，而是长度、数据、optimizer、router 和 attention 的联动。

多模态扩长还多了一层复杂性：长序列不只是长文档，也可能是长音频、长视频、GUI 轨迹、图文 interleaving 或视觉 agent 任务状态。Qwen3.5-Omni 在 S3 提高长音频和长视频比例，并把上下文扩到 262K；Kimi K2.5 则在 MoonViT-3D/NaViT packing 和 4-frame video compression 之上做 long-context activation，使长文本理解、长视频理解和 visual agentic workflow 共享同一个扩长 recipe。这个趋势说明，下一代 context extension 的关键变量会从“文本长度”扩展到“跨模态 token 如何被压缩、排列、同步和复用”。

### 8.4 KV cache 与 serving：长上下文的真实成本中心

长上下文推理的成本中心往往不是预训练 FLOPs，而是 decoding 阶段的 KV cache 和 per-token attention。DeepSeek-V4 报告称，在 1M context 下，V4-Pro 相比 V3.2 约为 27% single-token inference FLOPs 和 10% KV cache，V4-Flash 进一步约为 10% FLOPs 和 7% KV cache；其代价是需要 heterogeneous KV cache layout，分别管理 CSA/HCA compressed KV、SWA state cache 和 uncompressed tail states，并支持 on-disk KV cache 来复用 shared-prefix requests。

Step-3 的观察更直接：attention design 可能比 activated parameter count 更支配 decoding cost。它用 MFA 降低 KV cache 和 attention computation，并用 AFD 将 attention instances 与 FFN instances 分开部署，让 attention 按 context length 扩展、FFN 按 batch/MFU 优化。报告中 8K/32K 的 MFA/MLA/GQA latency 对比说明，长上下文 attention 不能只看理论复杂度，还要看具体 kernel、硬件、batch 和网络。

Step 3.5 Flash 则把服务成本与 agentic workload 直接绑定：196B total / 11B active 的 MoE 选择 3:1 SWA/full attention、SWA window 512、head-wise gated attention 和 MTP-3，目标是降低多轮 agent 的长上下文 latency。报告还披露其在线部署首周在 Hopper GPU 上可达到约 170 tokens/s，这说明 StepFun 的路线不是只追求小 active params，而是把 attention、MTP、MoE 均衡和在线吞吐一起优化。

Gemma 4 提供了另一类更贴近本地设备的 speculative decoding 样本。其 MTP drafter 不是简单挂一个小模型，而是围绕 target model 做了多处工程化适配：复用 target activations，复用局部/全局 attention 的 KV cache，并用 efficient embedder 降低小 drafter 在大词表 logits 上的成本。这个例子说明，MTP 的收益不只来自“多预测几个 token”，还来自 drafter 与 target backbone 的缓存、embedding 和维度接口是否足够贴合。

Nemotron 的 hybrid Mamba-Attention 说明另一类 cache 问题。Mamba 层让 per-step decode cost 对 sequence length 更友好，但 Mamba SSM state 本身也需要缓存、量化和 rollback；在 speculative/MTP decoding 中，如果 draft token 被拒绝，Mamba state 如何回滚比纯 attention 的 KV truncation 更麻烦。因此长上下文架构越“非标准”，serving framework 的适配越重要。

### 8.5 长上下文与 agentic RL

Agentic RL 把长上下文问题进一步放大。一次 SWE、terminal、browser 或 search 任务可能包含长 problem statement、多文件代码、工具观察、错误日志、测试输出和中间推理。模型如果不能有效管理上下文，就会在长轨迹中遗忘早期约束或被无关观察干扰。

GLM-5 在 search agents 中使用 context management 策略，例如保留最近若干轮观察、折叠早期 observation，并选取 32K 作为某些搜索 agent 的策略参数；DeepSeek-V4 的 Quick Resumption 和 KV cache 保存面向长 rollout 中断恢复；MiniMax 和 MiMo 的 RL infra 都要处理长尾轨迹和 prefix/route 复用。也就是说，长上下文能力和 agentic RL 不是两个独立章节：前者决定后者的环境状态是否能被持续利用，后者又推动模型对长上下文提出更真实的压力测试。

### 8.6 评测：从 needle retrieval 到真实工作流

长上下文评测正在从 needle-in-haystack 转向多维任务。RULER、MRCR、CorpusQA、LongBench v2、RepoQA、GSM-Infinite、NoLiMa 等 benchmark 各测不同侧面：检索、压缩、多轮引用、代码仓库、数学长上下文或干扰鲁棒性。MAI 用 Code NLL、Retrieval NLL 和 Generative QA 分析 256K extension；GLM-5 用 RULER/MRCR/HELMET/RepoQA 比较 DSA、SWA 和 full attention；MiMo 用 retrieval 和 GSM-Infinite 验证 hybrid SWA；DeepSeek-V4 直接评估 MRCR 1M 和 CorpusQA 1M。

但这些 benchmark 仍无法完全代表 agentic workload。真实 agent 任务的长上下文不仅有长文本，还有状态变化、工具 side effects、文件系统、网页、终端和多轮用户意图。因此更稳妥的表述是：长上下文 benchmark 用于验证基础能力，真实软件工程、搜索和多模态 agent 任务用于验证长上下文能否转化为行动能力。

### 8.7 小结：长上下文正在系统化

长上下文的下一阶段不是简单把窗口从 256K 推到 1M 或 2M，而是让长上下文变得可训练、可推理、可服务、可用于 agent。模型需要在 attention/KV cache 上降低单位成本，在 mid-training 上设计长度 curriculum，在 RL infra 中支持长轨迹与恢复，在评测上从检索走向真实工作流。

因此，长上下文章节可以提出一个总判断：2025-2026 年的开放模型已经从“支持更长输入”转向“能以可承受成本使用长上下文”。关键竞争点是 long-context utility per dollar，而不是 context window headline。

## 9. 多模态与 Omni Agent

多模态开放模型的发展已经从 VLM 扩展到 omni agent。它的本质变化不是“支持更多输入格式”，而是模型获得了更多环境状态和动作空间：图像、文档、GUI、网页、音频、视频和设备状态都可以变成 agent 的观察，语音、工具调用、代码、搜索、编辑和内容生成则变成可执行动作。

### 9.1 三层结构：VLM、Omni、生成/场景模型

第一层是 VLM：Qwen3.5 主线、Qwen3-VL、GLM-5V、STEP3-VL、MiMo-VL、Gemma 3/4。它们关注图文理解、文档、GUI、OCR、视觉推理和 grounding。VLM 的核心评价通常是文本回答质量、视觉定位、图表/文档解析、GUI 操作理解和多图推理。Qwen3.5-397B-A17B 暂无 standalone technical report，但 official reference 已经说明 Qwen 主线从 Qwen3 的 text/reasoning MoE 进一步推进到 native vision-language MoE。Gemma 4 的特别之处在于，它把多模态能力与本地部署强绑定：E2B/E4B/12B 支持 audio 输入，所有 Gemma 4 模型支持 image 输入，12B 更进一步采用 encoder-free multimodal 设计，把视觉/音频输入变成更轻的 split/project/embed 流程后交给 LLM backbone 处理。

第二层是 Omni：Qwen3-Omni/Qwen3.5-Omni、LongCat-Flash-Omni、Step-Audio、MiMo-Audio、Nemotron Nano Omni。它们把文本、图像、音频、视频或音视频理解整合进统一交互。Omni 的难点不只是编码更多模态，而是实时性、语音生成、跨模态对齐、长音视频理解和多轮交互稳定性。ERNIE 5.0 虽然不是开放权重，但其报告给出了另一种更激进的统一路线：理解和生成都放入同一自回归 backbone，而不是把生成能力交给外接的 modality-specific generator。

第三层是生成与场景化模型：LongCat-Image、LongCat-Video、LongCat-AudioDiT、Step-Video、ERNIE-Image、MiMo-VL-Miloco、LongCat-Video-Avatar。它们说明模型家族正在从“回答问题”扩展到“生成内容、操作场景、驱动角色和设备”。这类模型未必都是 LLM backbone 的直接延伸，但它们会和 LLM agent 形成工作流闭环。

### 9.2 Qwen3.5-Omni：从多模态理解到实时 omni interaction

Qwen3.5-Omni 是当前披露较完整的 omni 报告之一，因为它同时给出架构、数据配比和交互后训练。它的关键不是“支持更多模态”，而是把模态差异显式分配到 Thinker/Talker、AuT、multi-codebook codec、ARIA 和 streaming concurrency 中：Thinker 负责长音视频上下文中的理解/推理/工具调用，Talker 负责低延迟语音生成，ARIA 用自适应 interleave 约束同步文本与 speech units，Interaction-Aligned RL 再从体验层处理多轮稳定性。

数据侧，Qwen3.5-Omni 披露了少见的跨模态 token 预算：S2 general multimodal pretraining 约 4T tokens，其中 text 0.92T、audio 1.99T、image 0.95T、video 0.14T、video-audio 0.29T；AuT 音频编码器使用 4000 万小时 audio-text 数据训练，整体报告还提到 1 亿小时以上音视频内容。训练流程是 Encoder Alignment -> General Multimodal Pretraining -> Long Context Stage，后训练是 Specialist Distillation -> On-Policy Distillation -> Interaction-Aligned RL。这个设计说明 omni 模型需要同时优化 recognition、reasoning、generation、latency 和 conversation UX，单一 benchmark 很难覆盖。

### 9.3 Kimi K2.5：visual agentic model 与 Agent Swarm

Kimi K2.5 代表了另一种形态：visual agentic model。它不是只把视觉作为问答输入，也不是纯生成模型，而是把视觉纳入 agentic workflow：图像/视频理解、视觉 grounding、文档/图表解析、computer use、search 和 coding 可以放在统一 agent 任务中。它继承 Kimi K2 的 1.04T/32B active MoE backbone，使用约 15T mixed visual-text tokens 做 joint pretraining，并通过 MoonViT-3D 统一图像和视频编码空间。

K2.5 的一个关键经验是早期低比例 vision fusion 优于晚期高比例视觉注入。这说明当语言模型已经足够强时，多模态不应只是后置 adapter，而应在训练全程以较温和比例融合，让文本和视觉表征共同适配。后训练上，K2.5 提出 zero-vision SFT：只用 text SFT 激活视觉推理和工具能力，认为 joint pretraining 已建立足够强的 vision-text alignment，人工视觉轨迹反而可能伤害泛化。随后通过 outcome-based visual RL 和 joint multimodal RL 增强 visual grounding/counting、chart/document understanding 和 vision-critical STEM。

Agent Swarm 是 K2.5 对 agentic scaling 的进一步探索。PARL 训练一个 orchestrator，让它动态创建冻结 subagents 并并行分解任务；训练时只优化 orchestrator，以降低 credit assignment ambiguity。critical steps 则约束并行 agent 的资源使用，避免简单增加 subagents 造成 reward hacking。这一路线说明，多模态 agent 的扩展不只来自更长 CoT，也可以来自更好的并行任务编排。

### 9.4 ERNIE 5.0：统一理解与生成的另一条路线

ERNIE 5.0 与 Qwen3.5-Omni、Kimi K2.5 形成了清晰对照。Qwen3.5-Omni 更强调实时交互和 Thinker/Talker 分工，Kimi K2.5 更强调 visual agentic 和并行 Agent Swarm；ERNIE 5.0 则把重点放在统一自回归目标上：image generation 用 Next-Scale Prediction，video generation 扩展为 Next-Frame Prediction，audio generation 用 Next-Codec Prediction，最终都落到统一的 Next-Group-of-Tokens 预测范式中。

这个设计的关键启示是，omni 模型不一定要靠多个专用生成器拼起来。ERNIE 5.0 让视觉、音频和文本 token 在同一个 MoE backbone 中交互，专家路由不显式区分模态，而是在训练中自然形成共享/特化模式。它还专门处理了多模态 tokenization 和 attention 的工程问题：tokenizer 与 backbone 分离部署，视觉局部双向 attention 和文本/音频因果 attention 由 FlashMask 支撑，视频/音频生成则通过逐尺度、逐帧、逐 codec depth 的结构化预测降低序列建模难度。

如果说 Qwen3.5-Omni 更像“实时交互型 omni agent”，Kimi K2.5 更像“视觉 agentic model”，ERNIE 5.0 则更像“统一自回归理解/生成模型”的系统实验。它不是本文的开源主线，但可纳入综述作为多模态架构范式对照。

### 9.5 多模态数据：从图文 pair 到音视频/GUI/工具轨迹

多模态数据构造正在从 image-text pairs 扩展到 heterogeneous trajectories。VLM 阶段主要依赖图文、OCR、文档、图表、GUI 和 grounding 数据；Omni 阶段需要 audio-text、video-text、audio-video、video-audio-text、speech dialogue、captioning 和 streaming interaction 数据；visual agent 阶段还需要 computer use、browser、search、文档编辑、代码和视觉验证任务。

这也使数据配比更难披露和比较。Qwen3.5-Omni 的跨模态 token 配比信息量较高，因为多数报告只说“图文/音视频/多模态数据”而不给比例。Kimi K2.5 给出 15T mixed visual-text tokens、ViT training 和 long-context mid-training 阶段，但视觉任务数据细节仍不完整。STEP3-VL 披露 1.2T multimodal tokens，说明中等规模 VLM 也开始使用 trillion-token multimodal pretraining。ERNIE 5.0 披露的是更高层的数据组织方式：text data 覆盖多语言网页、书籍、论文、代码和结构化知识，multimodal data 覆盖 image-text、video-text、audio-text 与 interleaved multimodal sequences，并从训练开始就混入多模态数据；但它没有给出像 Qwen3.5-Omni 那样的详细跨模态 token 配比。多模态数据更适合按 pair、instruction、trajectory、environment、generation feedback 分层，而不是只写“multimodal data”。

### 9.6 多模态 RL 与评测

多模态后训练不再只是视觉问答 SFT。Qwen3.5-Omni 用 Interaction-Aligned RL 优化真实语音/音视频交互质量；Kimi K2.5 用 visual RL 和 joint multimodal RL，且报告观察到 visual RL 不仅未损害文本能力，还提升 MMLU-Pro、GPQA-Diamond 和 LongBench v2；MiMo-VL、Step-Audio-R1、Step-Audio-R1.5 等报告也说明音频/视觉 reasoning 开始进入 RL 阶段。ERNIE 5.0 的 Unified Multimodal RL 进一步说明，RL 系统需要处理 rollout 长尾、异步训练分布偏移、entropy collapse、hard query 稀疏奖励和多模态 verifier，这些问题在 text-only RLVR 中已经存在，但在 omni 模型中会被放大。

评测上，传统 VQA/OCR/Video QA 已不足以覆盖 omni agent。需要加入 audio reasoning、speech dialogue、audio-visual grounding、captioning、tool-use、computer use、visual coding、GUI 操作和多轮交互体验。Qwen3.5-Omni 的 Audio-Visual Vibe Coding 是一个值得注意的信号：模型可以根据音视频指令直接生成可执行代码，说明多模态能力正在和 coding/agentic workflow 耦合。

### 9.7 小结：多模态正在变成 agent 的感知层

多模态的本质变化是：模型不再只是多一种输入，而是获得更多环境状态和动作空间。这对 agentic RL 意义很大，因为网页、GUI、视频、语音和设备状态都可以成为交互环境的一部分。Qwen3.5-Omni 代表实时 omni interaction，Kimi K2.5 代表 visual agentic workflow，LongCat/Step/MiMo 则展示模型家族向图像、视频、音频和 avatar 扩展的趋势。ERNIE 5.0 则说明，另一条路线是把理解、生成、tokenization、routing 和 RL infra 更彻底地统一到同一个自回归系统里。

因此，开放模型的多模态路线正在从“视觉问答能力”转向“agent 感知与执行基础设施”。未来的关键不是模型能否看图听声，而是能否把视觉、音频、视频状态转化为可靠行动，并在长上下文、多工具、多轮环境中持续使用这些状态。

## 10. 作为预训练研究员，我感兴趣的点

这一节不按模型家族展开，而按研究问题展开。每个问题先给一句当前证据能支持的结论，再展开模型案例、证据边界和仍未披露的缺口。它的目的不是给出完全定论，而是把现有技术报告中可支持的证据、尚未披露的缺口和可能的解释整理出来，形成更有研究味的综述讨论。

### 10.1 哪些模型是 think model，agentic model 都是 think model 吗？

**结论：think 与 agentic 是两条相交但不等价的轴。** Think model 主要控制显式推理轨迹和推理预算；agentic model 主要解决工具、环境和长轨迹执行。强 agentic model 不一定是显式 think model，think/non-think 混合也更多由后训练格式、控制 token、system prompt 和 test-time compute 策略实现。

现有报告显示，think model 和 agentic model 是相关但不同的概念。Think model 强调显式长推理、thinking mode、reasoning traces 或可控 thinking budget；agentic model 强调工具使用、环境交互、任务执行和长轨迹。二者可以重叠，但并不等价。

| 模型 | Think/Non-think 定位 | Agentic 定位 | 备注 |
|---|---|---|---|
| Qwen3 | 同一模型支持 thinking/non-thinking，并有 thinking budget | 有 tool-calling、多轮环境和 agent 任务 | thinking control 是后训练核心目标 |
| Qwen3.5 | 未见 standalone technical report 披露 thinking-control pipeline | native vision-language MoE；agentic 细节主要来自 official reference/model card | 应与 Qwen3.5-Omni 区分：主线提供 Gated DeltaNet + sparse MoE 基座，Omni 报告披露更多交互和后训练细节 |
| DeepSeek-V4 | Non-think / Think High / Think Max 多模式 | 强 agentic search、coding、sandbox 和工具环境 | 报告明确讨论 interleaved thinking 与工具调用 |
| GLM-5 | 强调 interleaved thinking，Reasoning RL -> Agentic RL -> General RL | 明确定位 agentic engineering | thinking 与 tool call 在 workflow 中交织 |
| Kimi K2 | 明确是 open-source non-thinking model | 强 agentic/coding/tool-use | 说明 agentic model 不一定是 think model |
| Kimi K2.5 | 集成 thinking/instant modes，但报告主线是 multimodal agentic 和并行 agent orchestration | visual agentic + Agent Swarm/PARL | 最新证据更强调“多模态 + 多 agent + 并行执行” |
| MiniMax-M3 | HF model card/chat template 披露 thinking enabled/disabled/adaptive | coding/agentic、1M context、native multimodality | 本文已覆盖 arXiv PDF 中的 MSA 部分；think 控制来自 model card/chat template，训练 recipe 仍未披露 |
| Qwen3.5-Omni | 未以经典 think/non-think 控制为主线 | native omnimodal agent，支持 WebSearch、audio-visual tool use、audio-visual code generation | 最新 Qwen 证据从文本 thinking control 扩展到 omni agent |
| MiMo-V2-Flash | 强 reasoning/agentic，但未以显式 think/non-think 为主线 | code/web/agentic RL | 重点是 MOPD 和小 active 高吞吐 |
| MAI-Thinking-1 | 明确 thinking model | STEM、agentic coding/tool use、helpfulness/safety specialists | 报告强调从零学习 reasoning traces |
| Nemotron 3 Ultra | 后训练含 reasoning effort/control 相关设计 | multi-environment RLVR 与 long-running agents | 更偏系统化 reasoning/agentic open-weight |
| LongCat-Flash / Thinking | Flash 本身是通用 MoE，另有 Thinking 分支 | tool-use 与 multi-agent synthesis | thinking 能力通过分支模型显式化 |

因此，第一个结论是：agentic model 不一定是 think model。Kimi K2 是典型反例：它的报告反复强调 non-thinking setting，但它在 agentic coding 和 tool-use 上能力突出。反过来，think model 也不自动等于强 agentic model；think 能提升复杂推理，但如果缺少工具协议、环境数据、rollout infra 和 verifier，未必能稳定执行多步任务。

更细地看，think/non-think 混合不是一种单一机制，而是至少包含五种产品和训练形态。这里最关键的区分是：有些模型是在同一个 checkpoint 内做 mode switching，有些是同一家族内发布 thinking/non-thinking 分支，有些则根本不暴露显式 thinking，但通过 agentic data 和 RL 让模型能完成长轨迹任务。

| 形态 | 代表模型 | 实现方式 | 适用场景 |
|---|---|---|---|
| 单 checkpoint 混合 think/non-think | Qwen3, MiniMax-M3 | Qwen3 经过 Thinking Mode Fusion；用户 query 或 system message 中用 `/think`、`/no_think` 控制；非 thinking 样本保留空 `<think></think>`；多轮对话中随机插入 flags，并让模型服从最后一个 flag。M3 model card/chat template 披露 `thinking_mode=enabled/disabled/adaptive`，开启用于复杂推理/agentic/长程协作，关闭用于对话和代码补全等低延迟场景 | 一个模型同时服务快速回答和复杂推理，降低维护两套模型的成本 |
| 多档 reasoning effort | DeepSeek-V4 | Non-think / Think High / Think Max；用 `<think>...</think>` 标记 reasoning path；Think Max 通过特殊 system prompt 推高 thoroughness；RL 中为不同模式设置不同 length penalty 和 context window，评测中也使用不同 output budget | 按任务难度、延迟和成本选择推理强度 |
| 流程内 interleaved/preserved thinking | GLM-5, DeepSeek-V4 | 在 tool call 前后保留或插入思考；GLM-5 的 Interleaved Thinking 让模型在每次响应和工具调用前思考，Preserved Thinking 在 coding agent 场景保留已有 thinking；DeepSeek-V4 讨论跨工具轮次保留 reasoning traces | 多步 agent/coding/search 工作流 |
| 分支式 thinking model | LongCat-Flash / Thinking, 部分 Qwen/DeepSeek 分支 | 通用模型和 thinking 分支分开发布或分开模式化训练 | 同一模型家族覆盖快答与推理产品 |
| test-time deep think / reasoning scaling | Step 3.5 Flash | 报告不以产品化 think/non-think 混合为主线，但在评测中使用 PaCoRe deep think inference 和 tool-integrated PaCoRe | 把 thinking level 体现为并行/协调推理与工具交互，而不是固定单链条 CoT |
| 非显式或弱显式 thinking agentic | Kimi K2, Kimi K2.5, MiniMax-M2, Qwen3.5-Omni | 不把显式 thinking mode 作为唯一主线，而是强调 agentic data、tool-use、multimodal perception、RLVR、self-critique、PARL 或 Forge/RL 系统 | 低延迟 agent/coding/omni 服务，避免把所有能力都压到长 CoT |

因此，“混合模型”也可以再拆两层。第一层是 behavior mixture：同一个模型既能输出无显式 CoT 的答案，也能进入 `<think>` 格式。Qwen3 是较明确案例，non-thinking 不是简单不给模型思考，而是在 chat template 中保留空 thinking block，从格式上约束模型不要展开推理。第二层是 capability mixture：模型同时具备 reasoning、tool use、coding、search、general chat、多模态感知和长轨迹执行等能力，但是否显式 thinking 由任务、产品模式或 system prompt 决定。Kimi K2 属于 capability mixture 突出但不显式 thinking 的路线；Kimi K2.5 则把这一点推进到视觉 agent 和 Agent Swarm：报告强调并行 sub-agent 调度、视觉工具使用和 PARL，而不是把能力全都描述成单链条 CoT。Qwen3.5-Omni 也类似，它把 Qwen3 的 thinking-control 叙事推进到 native omni agent，重点是 Thinker/Talker、长音视频上下文、WebSearch 和跨模态 tool use。

不同 level 的 think 通常不是靠“预训练出多个大脑”，而是靠后训练和推理控制实现。可以概括为以下几种旋钮。

| 控制旋钮 | 代表模型 | 具体做法 | 对 level 的影响 |
|---|---|---|---|
| 控制 token / chat template | Qwen3 | `/think`、`/no_think`、空 `<think></think>`、默认 thinking mode | 决定是否展开显式 reasoning |
| Thinking budget | Qwen3 | 当 thinking 长度达到用户阈值时，插入停止思考指令，让模型基于已有部分 reasoning 直接回答 | 形成 low/medium/high 之间的连续预算控制 |
| System prompt | DeepSeek-V4 | Think Max 额外注入“absolute maximum”等强指令 | 把普通 thinking 推到更彻底、更慢的模式 |
| RL 长度与上下文配置 | DeepSeek-V4 | 不同 reasoning effort 使用不同 length penalty、context window 和 output budget | 在训练和评测两侧同时塑造不同 effort level |
| 工具上下文策略 | GLM-5, DeepSeek-V4 | tool call 前思考、跨轮保留 thinking、coding agent 场景保留完整推理状态 | 让 thinking level 随 workflow 深度变化，而不只是随单轮问答变化 |
| Agent orchestration / 并行预算 | Kimi K2.5, Step 3.5 Flash | Kimi K2.5 的 Agent Swarm 中由 trainable orchestrator 创建 frozen subagents，并通过 PARL 学习何时并行、如何分解任务；Step 3.5 Flash 的 PaCoRe 用并行协调推理扩展 test-time compute | thinking level 不只表现为输出 token 长短，也表现为并行 agent 数、critical steps、工具预算和并行 reasoning trajectories |
| Omni interaction context | Qwen3.5-Omni | Thinker 处理长音视频/文本上下文，Talker 处理流式语音；Interaction-Aligned RL 优化多轮交互稳定性 | thinking/control 与实时交互、语音/视频上下文和工具调用耦合 |
| 分支 checkpoint / 产品路由 | LongCat-Flash / Thinking, 部分模型家族 | 快答模型、thinking 模型、agent 模型分开服务 | 用模型路由而非单模型控制实现 level |

基于现有报告，可以形成一个判断：目前公开报告中的 thinking level 更像“test-time compute management”，而不是一个纯粹的预训练能力标签。预训练和 mid-training 提供数学、代码、知识、长上下文与 agentic priors；SFT/RL 决定模型是否以 `<think>` 形式暴露这些能力；部署侧的 budget、system prompt、工具状态和路由策略决定每次请求实际消耗多少 reasoning compute。这也解释了为什么 agentic model 不必然是 think model：agentic 的核心是环境交互和任务完成，thinking 只是其中一种可选的计算分配方式。

### 10.2 Think 数据会加入预训练吗？各家公司如何优化 think 数据？

**结论：预训练会加入 reasoning-oriented data，但公开报告很少证明“显式长 CoT think traces”是预训练主料。** 更常见的做法是在 pretraining/mid-training 中加入数学、代码、STEM、逻辑、合成题解和高质量 problem/solution，真正的 thinking format、budget control 和 overthinking 修正主要发生在 SFT/RL/Distillation 阶段。

从现有披露看，显式 reasoning traces / long-CoT 数据主要出现在 post-training，而不是主预训练。Qwen3 的 Long-CoT Cold Start 在后训练阶段；DeepSeek-V4 的 think/non-think/high/max 是后训练和推理模式管理；MAI 明确表示 RL climb 从零学习 reasoning traces，pre- 和 mid-training 给 base model 提供 broad predictive competence 和 STEM/coding 基础，但不让模型提前看 reasoning traces；Kimi K2 则将主线设为 non-thinking agentic model。

预训练阶段会加入“reasoning-oriented data”，但这通常不是显式 CoT traces，而是数学、代码、STEM、逻辑、合成问答、高质量 problem/solution 或知识密集数据。例如 Qwen3 S2 提高 STEM、coding、reasoning 和 synthetic data 比例；Qwen3.5 主线的完整数据 recipe 尚未以独立技术报告披露，但架构上已经转向 Gated DeltaNet + sparse MoE 的 native vision-language 基座；Qwen3.5-Omni 从 Qwen3.5 base checkpoint 出发，在 S2 使用约 4T 跨模态 tokens，并把 text、audio、image、video、video-audio 共同纳入 general multimodal pretraining；Kimi K2.5 在 Kimi K2 语言基座上加入约 15T mixed vision-text tokens，文本侧继续强调 Web、Code、Math、Knowledge，视觉侧加入 caption、interleaving、OCR、knowledge、perception、video 和 agent data。GLM-5 在 early pretraining 就提高 code/reasoning 权重；Step 3.5 Flash 预训练 17.2T tokens，并在 mid-training 中显式加入 long-horizon reasoning、code agent、search agent 和 tool-use；MiMo Stage 2 加入约 5% synthetic reasoning data；Nemotron 的 pretraining mixture 含 math、code、SFT-style data；DeepSeek-V4 强调 mathematical contents、code、long documents 和 high-quality categories。这些更像“给 base model 建 reasoning/agentic/multimodal substrate”，而不是直接教模型输出长 CoT。

各家公司对 think 数据的优化集中在后训练：Qwen3 用 query/response filtering、verified answers、code test cases、GRPO 和 thinking budget；Qwen3.5-Omni 则把后训练重心放在 Specialist Distillation、OPD 和 Interaction-Aligned RL，用 domain teachers 覆盖文本、视觉、音频、agentic、coding 和 reasoning，再把文本条件下更强的响应蒸馏到音频条件。DeepSeek-V4 用 think mode 标签、interleaved thinking、Think High/Max 以及工具调用中的 reasoning persistence；GLM-5 用 interleaved thinking、Preserved Thinking、Reasoning RL 和 Cross-Stage Distillation；MAI 用从零开始的 RL climb 让模型学 reasoning traces；Kimi K2.5 则进一步显示，zero-vision SFT + outcome-based visual RL + joint multimodal RL 可以激活和强化视觉 reasoning/tool-use，而不必先用大量人工视觉 CoT 轨迹。

Overthink 问题确实被提及，但披露程度不一。Qwen3 的 thinking budget 本质上是在控制“想太多”的成本，并且报告提到 thinking mode 在极长输入上可能因 verbose reasoning 影响 long-context 表现；DeepSeek-V4 在用户反馈里提到 occasional over-thinking，并区分 non-think、high、max 以匹配不同场景；Kimi K2 选择 non-thinking agentic 路线，某种程度上也是对 test-time compute 和长推理成本的工程取舍。可写成综述判断：overthinking 已从“行为缺陷”变成“推理预算管理问题”。

### 10.3 Agentic 能力如何在 pretraining/mid-training 注入？

**结论：agentic priors 可以在 base/mid-training 阶段注入，但可执行 agentic behavior 通常要到 SFT/RL 环境中成型。** 公开报告支持“代码、repo、网页、工具、GUI、视觉环境和长上下文数据会提前进入训练”，但很少系统评估 SFT 前 base model 的 agentic execution 能力。

Agentic 能力的注入大致分三层。第一层是 pretraining 中的代码、网页、文档、工具相关自然语料、GUI/视觉环境数据和 synthetic data，它提供 API、代码结构、任务描述、命令行语义、网页/桌面状态和视觉 grounding 等基础知识。Kimi K2 报告明确说 pre-training must endow agentic capability priors，同时引入 synthetic data generation；Kimi K2.5 则把 agentic priors 扩展到多模态：预训练语料包含 GUI screenshots、desktop/mobile/web action trajectories、image-code pairs、长视频和 grounding/perception 数据。Qwen3.5-Omni 也明确将模型设计成 native omni agent，训练早期就混入 unimodal 和 cross-modal data，并支持 audio-visual tool use。DeepSeek-V4 在 data construction 中提到加入 agentic data during mid-training；GLM-5 把 agentic/long-context 放入 mid-training；Step 3.5 Flash 在 dedicated mid-training 中混入 code agent、search agent、tool-use 和 long-horizon reasoning；MiMo 在 Stage 2 上采样 code-centric data，并在 Stage 3 上采样 long-range dependency data。

第二层是 mid-training 中的 workflow 级数据。GLM-5 是代表性例子：repo-level code files、commit diffs、GitHub issues、pull requests 和 relevant source files 被拼接为统一训练序列，issue-PR 数据过滤后约 160B unique tokens；long-context data 还包含 synthetic long-range dependency 和 MRCR-like 变体。DeepSeek-V4 强调 long-document curation、code、math 和 agentic mid-training data。Step 3.5 Flash 的 32K -> 128K mid-training 则说明，扩长阶段可以和 agent 初始化绑在一起：一方面提高长上下文承载能力，另一方面让 code/search/tool-use 的长轨迹在 SFT/RL 前就进入模型分布。MiMo 的 22T-26T mid-training 上采样 code-centric data 并加入 5% synthetic reasoning，26T-27T context extension 继续使用 Stage 2 分布并上采样 long-range dependency。

第三层是 post-training/RL 中的可执行环境与轨迹。Kimi K2 的 agentic data synthesis 通过 tool spec generation、agent/task generation 和 trajectory generation 生成工具使用示范；Kimi K2.5 的 Unified Agentic RL Environment 支持 text-vision joint RL 和 PARL，Agent Swarm 通过 create_subagent/assign_task 这类接口学习任务分解和并行调度；Qwen3.5-Omni 使用 Specialist Distillation、OPD 和 Interaction-Aligned RL，使 omni 模型在音频查询、视觉/音频理解、coding、agentic tasks 和长期语音交互中保持一致行为。Step 3.5 Flash 用 domain-specific RL 覆盖 Math、Code、STEM、Tool-use、Long Context、Human Preference 和 Agentic Reasoning，并用 MIS-PO 解决 off-policy MoE RL 的训练/推理分布偏移。MiniMax-M2 的 SWE/AppDev/Terminal 数据管线用 Docker、test execution 和 Agent-as-Verifier；GLM-5 构造 10K+ SWE/terminal/search 可验证环境；MiMo 使用 GitHub issues、web development tasks 和视觉 verifier；DeepSeek-V4 使用 sandbox infrastructure 和 fault-tolerant rollout。稳定的强 agentic 能力通常需要第三层，单靠 pretraining/mid-training 很难完成。

会不会在 base model 上评估 agentic 能力？现有报告披露并不充分。多数 base model evaluation 仍集中在语言、知识、数学、代码和 long-context benchmark。Kimi K2 对 base model 有 perplexity/generation 类 benchmark，并主要把 agentic 结果放在 Instruct/Post-trained 模型；Kimi K2.5 和 Qwen3.5-Omni 的强 agentic 指标也主要出现在 post-trained/agentic evaluation 中，例如 visual-to-code、WebSearch、OmniGAIA、audio-visual tool use、Agent Swarm 等。DeepSeek-V4 base evaluation 覆盖知识、reasoning、code 和 long context，agentic 能力更多在后训练模型中展示；GLM-5 的 agentic engineering 评测面向 post-trained/产品化模型。合理结论是：agentic priors 可以在 base/mid-training 中注入，但公开报告较少系统评估 SFT 前 base model 的 agentic execution 能力；agentic benchmark 仍主要是 post-training 后能力。

### 10.4 扩长阶段需要大量长文数据吗？数据比例披露到什么程度？

**结论：扩长需要高质量长程依赖数据，但不一定需要把大量 token 都放在最大上下文长度上训练。** 更常见的高性价比做法是短/中上下文主体训练，加少量最大长度校准、repacking、长程依赖上采样和短上下文能力保持；精确长文数据比例仍是披露盲区。

答案是：不一定需要“极大量”专门长文数据，但需要足够高质量、可重新打包、能覆盖长程依赖的数据。现有报告反而支持一种更节省的 recipe：主体能力在短/中等上下文训练，末端用相对较少 token 做 context extension 和位置/attention 校准。

| 模型 | 扩长 recipe | 长文数据比例/构成披露 | 可支持的结论 |
|---|---|---|---|
| MAI-Thinking-1 | 30T@16K -> 3.4T@64K -> 150B@256K；本文按 64K mid-training + 约 140B 256K extension 表述 | 明确说 64K/256K 阶段 re-pack 原 mixture，不修改 mixture weights；尝试过上调 long-context docs 和调 domain ratio，但无明显收益 | 不一定需要大幅改变数据配比；repack 高质量数据即可显著减少截断和校准长位置 |
| Nemotron 3 Ultra | 33B tokens LC phase；92% iteration 用 1M，8% 用 4K | 46% long-context data + 54% phase-2 data；长上下文包含 document QA 和 long-context SFT-style data；4K iteration 放 math/code SFT-style | 很少 token 也可做 1M extension，但需要混合短上下文保持能力 |
| GLM-5 | 32K 1T -> 128K 500B -> 200K 50B | 披露自然长文来自 books/papers/general corpora；合成数据构造 long-range dependencies；200K stage 加少量 MRCR-like data；未给完整比例 | 长度越长 token 越少，长文类型强调自然+合成+代码库/workflow |
| MiMo-V2-Flash | 22T-26T mid-training；26T-27T context extension 到 256K | Stage 2 加约 5% synthetic reasoning；Stage 3 沿用 Stage 2 分布并上采样 long-range dependency data；未披露长文占比 | 扩长可作为 1T token 末段完成，长程依赖数据上采样比绝对量更关键 |
| Step 3.5 Flash | pretraining 后 dedicated mid-training，从 32K 扩到 128K；PaCoRe/YaRN 评测到 256K | 披露加入 code agent、search agent、tool-use 和 long-context data，但未给长文/agent 数据比例 | 扩长与 agentic 初始化结合，目标不是单纯检索长度，而是多轮 agent 的长轨迹承载 |
| Kimi K2 | 15.5T 主预训练 4K；末端 400B@4K + 60B@32K；YaRN 扩到 128K | 未披露完整长文比例；披露 32K activation token 数和 128K evaluation context | K2 的长上下文不是主叙事，但有明确 activation/YaRN pipeline |
| Kimi K2.5 | 近末端 K2 checkpoint 上继续 15T vision-text tokens@4K；第三阶段用高质量 mid-training data 和 long-context activation，经 YaRN 顺序扩长 | 披露文本四类 Web/Code/Math/Knowledge，视觉七类 caption/interleaving/OCR/knowledge/perception/video/agent；未给完整比例 | 最新 Kimi 证据显示长上下文扩展同时服务长文本理解、长视频理解和 visual agentic workflow |
| Qwen3.5-Omni | S2 32K；S3 262K | 披露 S2 跨模态 token 配比；S3 提高长音频/长视频比例但未给精确比例 | 多模态扩长还要考虑长音频/视频，而不只是长文本 |
| ERNIE 5.0 | 8K pre-training；32K/128K mid-training | 披露 text data 与 multimodal data 两大类、trillions text tokens 和 multimodal instances，但未给精确模态比例 | 从一开始混入 text/image/video/audio，并用统一 token space + RoPE base 1,000,000 支撑后续扩长 |
| DeepSeek-V4 | 训练长度逐步到 16K/64K/1M | 强调 long-document curation，包含 math、code、web pages、long documents 等；未给长文比例 | 重点是 CSA/HCA 架构与 cache 系统，数据比例披露较少 |
| MiniMax-M3 / MSA | 109B MSA-CPT 从 2.6T full-attention checkpoint 继续 400B，其中前 40B indexer warmup；之后约 140B long-context extension | 披露实验模型是 text + image/video native multimodal mixture，但未给生产 M3 或 109B 实验的 domain/token ratio | MSA 证据支持“先让 selector 对齐 full attention，再 sparse CPT/扩长”的架构校准路线；不能据此推断完整 M3 数据配比 |

因此，对于“扩长阶段是否需要大量长文数据”，目前最稳妥的回答是：需要长程依赖数据和高质量长文档，但不一定需要在最大长度上投入巨大 token 预算。MAI、Nemotron 和 MiniMax-M3/MSA 都支持“短/中等上下文主体训练 + 末端相对短程 extension”的路线；GLM-5 和 MiMo 则显示长文、代码仓库、合成长依赖和 MRCR-like 数据会随上下文长度上调而被上采样。各类长文数据的精确比例大多未披露，Nemotron 和 Qwen3.5-Omni 是少数给出较具体阶段比例的报告；MiniMax 的 MSA 报告给出了 140B long-context extension 预算，但没有给出长文类型占比。

### 10.5 SFT 阶段是越来越轻量吗，还是各有侧重？

**结论：通用聊天 SFT 相对变轻，但 SFT 的职责没有变少，而是转向格式、协议、冷启动、能力合并和多模态/agent 接口。** 也就是说，SFT 不再负责生成主要能力上限，却仍然决定模型能否进入正确的 post-training 行为分布。

从现有报告看，SFT 不是简单越来越轻量，而是发生了分工变化。通用 instruction SFT 的边际作用在下降：模型的大部分知识、代码、数学和长上下文能力来自 pretraining/mid-training，能力爬升更多交给 RLVR、agentic RL、OPD/MOPD 和 specialist distillation。但 SFT 并没有变得可有可无，它正在转向更精细的接口层：负责 chat format、thinking/non-thinking 切换、工具协议、long-CoT cold start、specialist consolidation、多模态激活和 RL 起点稳定。

| 模型 | SFT 阶段特点 | 是否“轻量” | 主要作用 |
|---|---|---|---|
| Qwen3 | Long-CoT Cold Start + Thinking Mode Fusion；SFT 数据混合 thinking/non-thinking，并训练 `/think`、`/no_think` 切换 | 不是简单轻量，而是高度格式化 | 建立 CoT 格式、双模式控制和 thinking budget 基础 |
| Qwen3.5-Omni | Specialist Distillation -> OPD -> Interaction-Aligned RL；teachers 均从 Qwen3.5 base fine-tune，覆盖 text、vision、audio、agentic、coding、reasoning | 不是单一 SFT，而是多 teacher/多模态能力合并 | 把文本条件下更强的响应蒸馏到音频条件，并优化多轮交互稳定性 |
| ERNIE 5.0 | SFT 后接 Unified Multimodal RL；RL 使用 U-RB、MISC、WPSM、AHRL 和统一 verifier 系统 | SFT 是多模态 RL 前的能力与格式底座 | 强化统一多模态 reasoning/generation，并处理 hard query sparse reward 与 entropy collapse |
| DeepSeek-V4 | 每个 domain expert 先做 domain-specific SFT，再 GRPO，最后 OPD 合并 | 专家化而非通用大一统 | 给数学、代码、agent、instruction 等专家打底 |
| GLM-5 | multi-task SFT 引入 interleaved thinking；SFT 覆盖 agent/coding/general，并把 SFT context 扩到约 202K | SFT 仍很重要 | 引入工具前思考、preserved thinking 和复杂 workflow 格式 |
| Kimi K2 | multi-stage post-training，agentic tool-use demonstrations 依赖合成轨迹 | 披露更偏 agentic data/RL，SFT 不是唯一主角 | 教工具协议、agent/task/trajectory 格式 |
| Kimi K2.5 | zero-vision SFT：text-only SFT 激活视觉/工具能力；后续 joint multimodal RL 和 PARL | 视觉 SFT 反而被刻意减轻，但 RL/agent orchestration 变重 | 避免人工视觉轨迹伤害泛化，依赖 joint pretraining alignment，再用 RL 强化视觉与并行 agent 能力 |
| Step 3.5 Flash | 两阶段 SFT；第一阶段披露 870,687 samples / 7.23B tokens，覆盖 Math、Code、STEM、Logic、General QA、Code Agent、Tool-use、Search Agent 和 Long Context | 不算轻量，属于 agentic/RL 前的统一行为底座 | 让低 active MoE 在进入 domain-specific RL 和 MIS-PO 前先掌握多域格式、工具协议和长上下文任务 |
| MiMo-V2-Flash | Stage 1 general SFT，随后 domain-specialized RL/SFT teachers，再 MOPD；SFT 还关注 MoE 训练稳定性，例如 num-zeros 指标 | 通用 SFT 是底座，能力整合靠 MOPD | 建 instruction-following 基础，激活 latent capability，并提供 teacher/student 起点 |
| MAI-Thinking-1 | specialist RL 后用 consolidation SFT 合并三类 specialists；披露 mixture：STEM/coding 56% sample / 89% token，agentic 11% / 9%，helpfulness/safety 33% / 2%；最终还有 lightweight RL | consolidation SFT 精细但非最大成本中心 | 把 STEM、agentic coding/tool use、helpfulness/safety 合成单模型，并解决长 trace token 占比不均 |
| Nemotron 3 Ultra | SFT 后接 multi-environment RLVR/MOPD；SFT 还保留 MTP objective 和长上下文数据 | SFT 是 pipeline 中一环 | 对齐指令、长上下文和 MTP/post-training 接口 |

因此，一个更准确的判断是：SFT 的“通用能力生成”作用在下降，但“格式、协议、冷启动、稳定性和能力合并”作用在上升。以前 SFT 可能承担大量聊天能力和 instruction following；现在，强模型更倾向于用 SFT 建立可训练格式，再用 RL 和 distillation 做能力爬升。Qwen3 用 SFT 建 thinking/non-thinking 控制，Qwen3.5-Omni 用 specialist distillation/OPD 合并跨模态 teachers，GLM-5 用 SFT 建 interleaved thinking，MAI 用 consolidation SFT 合并 specialists，MiMo/Nemotron 用 SFT 给 MOPD/RLVR 提供 student 起点，Kimi K2.5 则说明在多模态场景下 SFT 可以更克制，但 joint multimodal RL 和 PARL 会变得更重。

围绕 SFT 是否越来越轻量，可以拆成四个趋势。

| 趋势 | 是否变轻 | 解释 |
|---|---|---|
| 通用聊天 SFT | 相对变轻 | base model 已经更强，通用对话格式不再需要承担主要能力学习；过重 SFT 还可能压制 base capability 或带来 style overfitting |
| Reasoning / thinking SFT | 不一定变轻 | 需要 cold start、格式学习、thinking/non-thinking fusion、trace filtering 和 budget control；它更像 RL 前的轨道铺设 |
| Agentic / tool / omni SFT | 更专业化 | 工具调用、终端、浏览器、代码仓库、SWE issue、search trajectory、音视频交互和视觉工具使用都有严格协议，SFT/Distillation 用于教会模型“怎么和环境说话” |
| Consolidation SFT | 变得更重要 | 当能力来自多个 specialist 或多个 RL climb，SFT 常用于把 teacher rollouts 合并回单一模型；MAI、DeepSeek、MiMo 都体现了这个方向 |

这背后还有一个数据配比启示：SFT 的 sample ratio 和 token ratio 会严重分离。MAI 的 consolidation SFT 中 helpfulness/safety 有 33% 样本权重但只有 2% token 权重，而 STEM/coding 只有 56% 样本权重却占 89% token 权重，原因是 reasoning traces 更长。因此，后训练报告如果只披露“样本占比”或只披露“token 占比”，都不足以判断真实训练重心。SFT 数据配比至少需要区分 sample mixture、token mixture、domain mixture 和 trace-length effect。

这个趋势对预训练研究员的启示是：如果 base/mid-training 已经足够强，SFT 不必承担所有能力学习；它更像 post-training 系统的接口层，负责让模型进入正确的行为分布。更昂贵且更影响上限的部分，正在转向可验证任务、环境 rollout、teacher/student 合并和 reward 设计。

### 10.6 预训练阶段会做数据增广吗？packing、消费顺序和 dataloader 有披露吗？

**结论：预训练增广不是普遍充分披露的标准项，但 packing/repacking 已经从工程填充变成能力构造手段。** Kimi K2 的 rephrasing 是较明确的 pretraining augmentation 案例；MAI、DeepSeek、GLM、MiniMax 等则显示 packing、repacking、repository-level sequence 和确定性 dataloader 正在成为长上下文训练的关键变量。

这一点需要单独讨论，因为最近的 pretraining recipe 已经不只是“清洗网页、按比例采样、拼成 sequence”。公开报告里能看到三类细节：第一，少数模型明确做 pretraining data augmentation；第二，长上下文模型越来越重视 packing/repacking 的构造方式；第三，高效 dataloader 和数据消费顺序披露较少，但 MAI、DeepSeek、GLM-5 给出了一些系统级线索。

先看数据增广。Kimi K2 是披露较明确的 pretraining augmentation 案例：它为了提高高质量 token 的 token utility，引入 rephrasing-based augmentation，而不是简单多 epoch 重复。知识数据用 style- and perspective-diverse prompting、chunk-wise autoregressive rewriting 和 fidelity verification；报告还比较了原始数据重复、多次 rephrasing 与多 epoch 的效果，并说明每个 corpus 最多 rephrase 两次。Kimi K2.5 延续 K2 的数据处理方法，但把增广/构造扩展到多模态：视觉数据包含 synthetic captions、OCR、grounding、image-code pairs、GUI screenshots/action trajectories 和长视频数据，同时对 synthetic captions 设置限制以缓解 hallucination。这个做法的本质是：对高质量知识文本做“保真改写”，对多模态 agent 则构造能连接视觉、代码、动作和工具的训练语境。

数学、代码和 agentic 数据的增广更常出现在 mid-training 或 post-training，但会反过来影响 base 能力。DeepSeek-V4 继承 token-splitting 和 FIM，并在 mid-training 加入 agentic data；MiniMax-M2 的 post-training 数据中有 PR task transformations、bug injection、test-writing 和 code review tasks；GLM-5 用 synthetic long-range dependencies 和 MRCR-like 数据扩展长上下文能力；Qwen3 在 pretraining S2 提高 synthetic data 比例，并用 instance-level annotation/ablation 优化 mixture；Qwen3.5-Omni 则使用更广泛自然语言 prompts、早期混合 unimodal/cross-modal data，并通过 timestamp text strings/random audio timestamps 降低长音视频时序建模的数据构造成本。严格说，Kimi K2 的 rephrasing 更接近“预训练增广”，而 DeepSeek/GLM/MiniMax/Qwen/Qwen3.5-Omni 的很多做法更像“能力定向数据合成、模态对齐或 curriculum data construction”。

| 模型 | 预训练/延续训练中的增广或构造 | 是否属于严格 pretraining augmentation | 综述判断 |
|---|---|---|---|
| Kimi K2 | knowledge/math rephrasing；chunk-wise rewriting；fidelity verification；每个 corpus 最多 rephrase 两次 | 是，报告明确写在 pre-training data | 用改写提高高质量 token utility，替代简单重复 |
| Kimi K2.5 | synthetic captions、OCR、grounding、image-code pairs、GUI/action trajectories、long video；控制 synthetic caption 比例 | 是多模态数据构造/增广，但不完全等同文本 rephrasing | 最新 Kimi 证据显示 agentic 数据构造已进入视觉、GUI、视频和代码-视觉对齐 |
| Qwen3 | 30T+ tokens 做 multilingual annotation；instance-level mixture optimization；S2 提高 STEM/coding/reasoning/synthetic data | 部分是数据选择与合成，不完全是 augmentation | 数据增益来自细粒度标注、过滤和配比，而不只是扩量 |
| Qwen3.5-Omni | 更广泛自然语言 prompts；早期混入 unimodal/cross-modal data；视频/音视频用文本 timestamp，音频随机插入 timestamp；S2 披露 4T 跨模态 token 配比 | 更像多模态时序/交互数据构造 | 降低长音视频时序学习的数据构造成本，并强化 omni agent 输入形式 |
| ERNIE 5.0 | text/image/video/audio 从训练早期进入统一序列；视觉/音频 tokenizer 逐步切换或结构化预测；视频历史 token corruption、随机历史帧 masking | 更像统一自回归多模态数据构造和 tokenizer curriculum | 目标是让理解与生成共享 backbone，并缓解长视觉/音频序列的训练不稳定 |
| DeepSeek-V4 | 继承 token-splitting、FIM；过滤 auto-generated/templated web；pack documents from different sources；mid-training 加 agentic data | FIM/token splitting 属于格式/任务增广，agentic data 更偏 mid-training | 更关注长上下文、代码和数学数据质量 |
| GLM-5 | synthetic long-range dependency；相似文本 interleaved packing；200K stage 加 MRCR-like data | 主要是 mid-training 长上下文构造 | 增广目标是让长序列有真实依赖，而不是随机变长 |
| MiniMax-M2 | code concatenation、PDF 长文、thematically related document packing；post-training PR task transformations / bug injection | pretraining 侧偏 packing 和 domain upsampling；任务增广多在 post-training | agentic 能力的数据增广主要发生在环境和任务层 |
| MiniMax-M3 / MSA | Indexer warmup、KL alignment、local block 更像 attention selector 的训练构造；报告没有披露完整数据增广、packing 或 dataloader | 不是数据增广报告 | 提供的是“稀疏 attention 如何被训练成可用”的方法，而不是数据 recipe |
| Step 3.5 Flash | mid-training 中加入 code agent、search agent、tool-use；后训练使用两阶段 SFT、domain RL/self-distillation/MIS-PO | 更像 agentic curriculum 和 RL 稳定化，不是披露充分的 pretraining augmentation | 最新 StepFun 证据强调长轨迹和 MoE off-policy 稳定性，而不是文本保真改写 |
| MAI-Thinking-1 | 明确不使用 LLM-generated synthetic data 做 pretraining，并过滤 AI-generated content | 否 | 代表 clean/human-generated 数据路线，和 Kimi 的 rephrasing 路线形成对照 |

再看 packing。普通短文本 packing 的目标是减少 padding、提高 token 利用率；长上下文时代的 packing 还承担另一个任务：构造长程依赖和减少 truncation。MAI 披露较具体：数据经过 exact/fuzzy/semantic dedup 和 quality binning 后，在每个 dataset 内 greedily packed into fixed-length sequences；代码文件按 repository 分组、按目录深度优先顺序排序，再拼成 repository-level sequences。到 256K extension 时，MAI 没有改变 mixture weights，而是把上一阶段 mixture repack 到目标长度。这个结论信息量较高：长上下文扩展有时不是改数据比例，而是改 packing 长度和截断方式。

DeepSeek-V4 也明确提到 packing：为了减少 sample truncation，它把不同来源的 documents pack into appropriate sequences，并在 pretraining 中使用 sample-level attention masking。由于 CSA/HCA 会压缩 KV，报告还讨论了 packed multiple sequences 在 context parallelism 下带来的 compressed KV length 不均和 rank 边界问题。GLM-5 则从能力角度使用 interleaved packing：把高度相似文本聚合成序列，用来缓解 lost-in-the-middle 并增强长上下文任务表现。MiniMax-M2 把高质量代码拼接、自然长 PDF 和 thematically related document packing 作为长上下文训练样本来源。它们共同说明，packing 已经从“工程填满 batch”变成“数据语义组织方式”。

| Packing 方式 | 代表模型 | 目的 |
|---|---|---|
| Greedy fixed-length packing | MAI-Thinking-1 | 减少 padding，提高 token 利用率，并保留 dataset 内质量采样结构 |
| Repository-level packing | MAI、GLM-5、MiniMax-M2 | 保留代码仓库结构、commit/PR/issue 上下文和文件间依赖 |
| Cross-source / appropriate sequence packing | DeepSeek-V4 | 减少样本截断，适配长上下文 pretraining |
| Interleaved packing of similar texts | GLM-5 | 构造长程依赖，缓解 lost-in-the-middle |
| Thematically related document packing | MiniMax-M2 | 为长上下文训练提供语义相关的长序列 |
| Repacking / curriculum at longer sequence length | MAI、Qwen3.5-Omni、Kimi K2.5、GLM-5 等 | 扩长阶段重新组织同一 mixture、提高长音视频/长文比例，或通过 YaRN/context activation 顺序扩长 |

最后是消费顺序和 dataloader。多数报告不会披露完整 data loader，因为这涉及训练系统内部实现和数据资产管理。但有几个信号值得记录。MAI 明确为了 bitwise deterministic training 固定 training micro-batches 的顺序；即使 DP workers 异步 prefetch batches，也保证 pending batch queue 在不同 run 和 restart 中一致，并在 checkpoint 中保存 dataloader progress 和 RNG。GLM-5 在长序列训练中提到 workload-aware sequence reordering、dynamic redistribution of attention computation 和可变 context-parallel group，用来解决长序列造成的 DP/PP 负载不均。DeepSeek-V4 在 million-token RL/OPD 里把 rollout data 拆成 lightweight metadata 和 heavy per-token fields：metadata 用于 global shuffling 和 packing layout computation，heavy fields 通过 shared-memory dataloader 按 mini-batch 粒度加载和释放，以降低 CPU/GPU 内存压力。虽然这是 RL 阶段，不是主预训练，但它代表了长序列 dataloader 的方向。

因此，对这个开放问题的初步回答是：公开报告已经开始披露“数据如何被组织成序列”，但很少完整披露“训练时按什么顺序消费所有数据”。Kimi K2 证明预训练阶段可以对高质量文本做保真 rephrasing augmentation；Kimi K2.5 和 Qwen3.5-Omni 进一步说明，新一代多模态 agent 的“增广”已经转向跨模态语义组织、视觉-代码对齐、GUI/action trajectories、长音视频 timestamp 和 agent data。MAI 证明 packing/repacking 本身就是长上下文 recipe 的关键变量；GLM-5 和 MiniMax-M2 说明 packing 可以服务长程依赖与 agentic/code workflow；DeepSeek-V4 和 MAI 则说明高效 dataloader 正在和 determinism、global shuffling、shared-memory loading、restart recovery 绑定。对预训练研究员来说，后续可以继续追问四个更具体的问题：sample-level mixture 如何映射到 token-level mixture，packing 是否跨文档/跨域/跨仓库/跨模态，长上下文扩展是否改变 mixture 或只 repack，dataloader 是否能在异步 prefetch 和故障恢复下保持确定性。

### 10.7 小结：六个问题的综述判断

第一，think 与 agentic 不等价。Think model 解决推理预算和显式 reasoning traces，agentic model 解决工具、环境和长轨迹执行；Kimi K2 说明 non-thinking agentic model 也可以能力突出，而 Kimi K2.5、Qwen3.5-Omni 和 Step 3.5 Flash 进一步说明 agentic 正在扩展到多模态感知、工具调用、并行推理和低延迟长轨迹执行。第二，think/non-think 混合主要通过后训练格式、控制 token、system prompt、thinking budget、工具上下文、PaCoRe/PARL 这类 test-time compute 策略和 agent orchestration 实现，而不是在预训练阶段直接固化。第三，agentic 能力通常通过 pretraining/mid-training 的代码、workflow、long-context、GUI/视觉环境和 cross-modal data 建立先验，再通过 SFT/RL 的工具轨迹和可执行环境转化为行为；公开报告中 base model 的 agentic execution 评估仍不足。第四，长上下文扩展不一定需要海量最大长度数据，关键是高质量数据 repacking、长程依赖上采样、少量最大长度校准和短上下文能力保持；Qwen3.5-Omni、Kimi K2.5 和 Step 3.5 Flash 说明扩长还要服务长音视频、视觉 interleaving、visual agentic workflow 和 code/search/tool-use 轨迹。第五，SFT 没有消失，也不只是变轻，而是从“大而全的聊天调优”转向“格式/协议/冷启动/合并接口”；Qwen3.5-Omni 的 specialist distillation/OPD、Kimi K2.5 的 zero-vision SFT/joint RL 和 Step 3.5 Flash 的 7.23B-token agentic SFT 都是最新例子。第六，预训练数据工程正在从“样本过滤和配比”扩展到“保真增广、语义 packing、跨模态数据构造、长序列 repacking、agentic curriculum 和确定性 dataloader”，但各家对数据消费顺序的披露仍明显不足。

## 11. 开放程度与可复现性

需要区分 open weights 和 fully open。大多数模型开放权重和技术报告，但不开放完整训练数据、配比、日志和中间 checkpoint。OLMo 2/3 是少数 fully open 代表，适合研究数据配比、训练流程和复现实验。

本文将开放程度粗分为四层：

1. Fully open：OLMo 2/3。
2. Open weights + tech report：DeepSeek、Qwen、GLM、Kimi、MiniMax-M2/M3、LongCat、StepFun、MiMo、Nemotron、Mistral。
3. Open weights + model card/reference：Gemma 3/4 reference、Hy3-preview、MiMo-V2.5、Ling 2.6、Qwen3.5 主线。
4. 非 open-weight/API/platform reference：ERNIE 5.0、MAI-Thinking-1、Step 3.7 Flash、ERNIE 5.1。

开放生态的一个矛盾是：模型越来越强，但数据和训练细节未必更开放。因此，本文区分“能力开放”和“可复现开放”：前者强调权重、API 或模型能力可用，后者强调数据、代码、日志、checkpoint 和 recipe 能否支持外部复现。

Nemotron 3 Ultra 处在一个值得注意的位置：它是 open-weight，同时还释放部分 training data、posttraining data、recipes、base/post-trained/quantized checkpoints 和 RL environments。它不是 fully open 到 OLMo 那种程度，但比许多只给权重和报告的模型更接近“工程可复用”。这类中间形态适合在开放程度章节单独标注。

## 12. 未充分披露的问题

本次整理也暴露出几个共同缺口。

第一，数据配比普遍不完整。多数报告会说使用代码、数学、合成数据、agentic data，但不披露比例；即使披露阶段 token 数，也常常缺少 domain mixture、sample mixture 和 token mixture 的对应关系。

第二，后训练数据规模常常不清楚。RL rollout 数、环境数量、reward model 构成、失败样本处理方式通常只部分披露；agentic RL 还很少披露长尾轨迹、失败轨迹和环境污染审计的完整统计。

第三，MoE 负载均衡细节披露不均。DeepSeek、LongCat、MiniMax、Step 3.5 和 MAI 披露较多，其他模型往往只给 experts/top-k/active params；softmax gating、sigmoid gating、expert bias、aux-loss-free bias、seq aux loss 和 EP/rank-level balancing 的组合关系仍不够完整。

第四，优化 recipe 的披露仍不均衡。少数报告会给出 optimizer 参数、LR schedule、MTP/loss-balance 权重和 self-distillation 超参，但多数报告仍不会完整说明哪些参数用 AdamW/Muon、LR 何时切换、辅助损失何时降权，以及多专家/多阶段能力整合到底依赖离线蒸馏、on-policy distillation、MOPD 还是 consolidation SFT。

第五，长上下文评测仍不统一。1M context、256K context、128K context 的实际能力取决于任务类型，不应只比较窗口长度。

第六，多模态模型的训练数据和评测污染更难判断，尤其是 audio/video/GUI/网页任务。

第七，Qwen3.5 主线、Kimi K2.6、Ling 2.6、Hy3-preview、Gemma 4 等新模型目前更多依赖 model card、blog 或 repo reference，缺少统一 standalone technical report。Gemma 4 的 visual guide 解释性很强，适合补架构理解，但不能替代官方训练报告。MiniMax-M3 已有 arXiv PDF，但该 PDF 主要是 MSA attention/infra 报告，能够支撑 MSA 架构、indexer 训练和 kernel 效率分析，却不能补齐生产 M3 的完整数据配比、训练流程和后训练细节。

第八，base model 的 agentic 能力评估仍不足。多数报告在 base 阶段评估语言、数学、代码和 long-context benchmark，而把工具调用、SWE、browser/search、GUI 和多模态 agent 指标放在 post-trained 模型上，导致“agentic priors 在预训练阶段注入到什么程度”仍难以量化。

第九，packing、数据消费顺序和 dataloader 设计披露很少。MAI、DeepSeek 和 GLM-5 给出了 determinism、shared-memory dataloader、sequence reordering 等线索，但大多数报告仍未说明 sample-level mixture 如何映射到 token-level mixture，以及长上下文 repacking 是否改变了训练数据的真实消费分布。

## 13. 结论

2025-2026 年的开放大模型已经进入一个新的阶段：模型能力不再由单一因素驱动，而是由 MoE 稀疏扩容、长上下文 attention、合成与可验证数据、optimizer/LR recipe、能力合并策略、agentic RL、低精度训练推理和系统调度共同决定。

从技术路线看，未来一年值得关注的方向包括：

1. 1T 级 MoE 与 10B-50B active params 的效率边界。
2. 动态计算 MoE，例如 zero-computation experts 和 token-level compute allocation。
3. million-token context 下的 KV cache、FLOPs 和 agent rollout 成本。
4. 从 RLVR 到 agentic RL 的环境构造和 reward 可靠性。
5. Muon/AdamW、LR schedule、MTP/loss balance 和 optimizer-aware attention 稳定性的组合 recipe。
6. teacher/student、on-policy distillation、MOPD、consolidation SFT 等能力合并范式。
7. 多模态从理解走向执行，尤其是 GUI、网页、视频、音频和设备场景；ERNIE 5.0 这类统一自回归报告还提示，理解/生成/路由/RL infra 可能会进一步收敛。
8. fully open 数据/日志/checkpoint 是否能追上 open-weight 模型能力。

用一句话概括：开放模型的前沿已经从“训练一个会回答的大模型”转向“构建一个能在长上下文、多工具、多模态环境中持续行动和自我改进的系统”。

## 附录 A：延伸阅读顺序

第一组是本文最核心的 frontier MoE 与 agentic 训练报告：

1. DeepSeek-V4：million-token context、CSA/HCA、mHC、Muon、domain expert + OPD。
2. GLM-5：27T base corpus、4K -> 200K mid-training、DSA、异步 RL、三段式 RL。
3. Kimi K2/K2.5：MuonClip/QK-Clip、15.5T text tokens、agentic data synthesis；K2.5 的 15T vision-text tokens、zero-vision SFT、joint multimodal RL 与 Agent Swarm/PARL。
4. MiniMax-M2/M2.7/M3：M2/M2.7 的 9.8B active agentic MoE、Forge RL、SWE/AppDev/Terminal 数据管线；M3 的 428B/23B MoE、1M context、native multimodality 和 thinking on/off；MSA 报告需重点读 Index Branch/Main Branch、KL alignment、indexer warmup、kernel co-design 和 109B/6B 实验。
5. LongCat-Flash：zero-computation experts、PID expert bias、ScMoE、multi-agent synthesis。
6. MiMo-V2-Flash：hybrid SWA/global attention、MTP、MOPD、R3。

第二组用于补齐多模态和系统侧：

1. Qwen3.5 主线 reference、Qwen3.5-Omni、Qwen3-Omni、Qwen3-VL。
2. Step-3、STEP3-VL、Step-Audio、Step-Video。
3. LongCat-Omni、LongCat-Next、LongCat-Image/Video/AudioDiT。
4. MiMo-VL、MiMo-Audio、MiMo-VL-Miloco。
5. ERNIE 5.0、Nemotron 3 Ultra/Nano/Omni。

第三组用于复现性和对照：

1. OLMo 2/3：fully open model flow。
2. Gemma 3/4：端侧和高效模型路线；Gemma 4 重点读 local/global attention、p-RoPE、per-layer embeddings、26B A4B MoE、12B encoder-free multimodality 和 MTP drafter。
3. Ministral/Magistral/Devstral：小模型、reasoning 和 coding agent 对照。
4. Hy3-preview、Ling 2.6、MiMo-V2.5、ERNIE 5.1：近期重要但 PDF 披露不完整的 model-card/reference 条目；MiniMax-M3 的 MSA PDF 已纳入本文，但仍缺完整 M3 数据/RL technical report。

## 附录 B：补充材料

### B.1 统一引用表

本文需要把“模型能力讨论”和“资料来源”分开。建议用一张统一引用表管理发布时间、license/access、weights 链接、repo/project 链接和技术报告位置。当前可以把它视为资料索引表：已经由官方报告或 model card 支撑的条目标为 high/medium-high，仍依赖 blog、HF model card 或 repo reference 的条目标为 medium/low-medium，其中 license 与发布时间仍建议在引用前逐项核验。

| 模型家族 | 代表模型 | 资料状态 | 写作处理 |
|---|---|---|---|
| Qwen / DeepSeek / Kimi / GLM | Qwen3、DeepSeek-V4、Kimi K2/K2.5、GLM-5 | 有技术报告或较完整官方报告 | 可作为主线模型，正文详写训练流程/架构/RL |
| MiniMax / StepFun / MiMo / Nemotron | MiniMax-M2/M3、Step 3.5、MiMo-V2-Flash、Nemotron 3 Ultra | 报告细节丰富，但覆盖面不同 | 适合作为技术案例；注意 M3 是 MSA 报告而非完整 M3 recipe |
| Gemma / Hy3 / Qwen3.5 主线 / Ling 2.6 | Gemma 4、Hy3-preview、Qwen3.5-397B、Ling 2.6 | 更多依赖 model card、blog、repo reference | 可写架构和产品定位，避免写死数据配比/RL recipe |
| OLMo / ERNIE / MAI | OLMo 3、ERNIE 5.0、MAI-Thinking-1 | 一个 fully open，两个非 open-weight 但报告详细 | 分别作为复现性上界、统一多模态对照和闭源平台参照 |

### B.2 MoE 负载均衡细表

本文将 MoE 设计拆成 router/gate、top-k/shared expert、expert bias/aux loss、capacity/dispatch、device/EP balance 和通信/serving 策略。这个拆法比“参数/激活规模”更接近训练研究员关心的问题：router 如何产生分配，负载均衡信号是否干扰 LM loss，expert bias 是否在不同训练阶段启停，EP/device 层面的热点如何被控制，以及 serving 侧是否把稀疏计算转化为吞吐。

| 路线 | 代表模型 | Router / Balance 关键词 | 主要观察 |
|---|---|---|---|
| aux-loss-free / expert bias | DeepSeek-V3/V4、GLM-4.5、MiniMax-M2、MiMo、Step 3.5、ERNIE 5.0 | sigmoid gate、balance bias、sequence aux loss、expert bias | 负载均衡从静态 aux loss 转向阶段化 bias 与轻量约束；ERNIE 5.0 还展示了多模态 shared expert pool 下的无辅助损失均衡 |
| device/EP-level balance | LongCat-Flash、Step 3.5 | device-level balance loss、EP-level balance loss、PID expert bias | 高吞吐 MoE 不能只看 token-level expert usage，还要看设备和 EP rank 热点 |
| global-batch balance | Qwen3、MAI | global-batch load balancing | 对大规模 DP/EP 训练，micro-batch 内均衡不够，跨 worker 聚合更重要 |
| latent/dynamic compute | Nemotron 3 Ultra、MAI、LongCat、ERNIE 5.0 | LatentMoE、zero-computation experts、elastic depth/width/sparsity | 专家计算不一定发生在原 hidden size，也不一定每个 token 用同等 FFN compute；elastic training 进一步把可裁剪部署纳入预训练 |
| 未充分披露但架构重要 | Kimi K2/K2.5、MiniMax-M3、Gemma 4 | ultra-sparse MoE、routing bias、edge MoE | 可讨论结构和 active params，但负载均衡细节不能过度推断 |

### B.3 RL 系统细表

本文将后训练/RL 拆成 reward 类型、verifier/judge、环境、rollout 调度、distillation/consolidation、是否 on-policy。这个表的用处是避免把所有 RL 都写成“GRPO/RLVR”：同样是 RL，Qwen3 更像 thinking control 与 verified reasoning，DeepSeek-V4 更强调 million-token RL infra 与 domain experts，Kimi K2/K2.5 更强调 agentic/visual/PARL，MiniMax-M2 更强调可执行 SWE/AppDev/Terminal 环境，Step 3.5 则强调 off-policy MoE RL 的 MIS-PO 稳定性。

| RL 设计轴 | 代表模型 | 典型做法 |
|---|---|---|
| Verifiable reasoning | Qwen3、DeepSeek-V4、Nemotron、MAI | verified answers、rule/checker rewards、GRPO/RLVR、STEM feedback |
| Executable agent environments | MiniMax-M2、GLM-5、MiMo、DeepSeek-V4 | Docker tests、terminal/browser/search/SWE、sandbox、tool manager |
| Multimodal/visual agentic RL | Kimi K2.5、Qwen3.5-Omni、MiMo-VL、ERNIE 5.0 | visual grounding/counting/document rewards、audio-visual tool use、GUI/action trajectories、unified multimodal verifier |
| Distillation/consolidation | DeepSeek-V4、MiMo、Nemotron、MAI、Qwen3.5-Omni | OPD/MOPD、specialist distillation、teacher token-level reward、consolidation SFT |
| RL infra and scheduling | GLM-5、MiniMax Forge、Step 3.5、Kimi K2.5、ERNIE 5.0 | async RL、windowed-FIFO、MIS-PO、PARL/Agent Swarm、Unbiased Replay Buffer |

### B.4 技术报告与官方资料链接

下面按模型系列整理本文使用过的官方技术报告、官方 blog、model card 或 repo。带有“未见独立 PDF”的条目，表示当前写作中只把它作为官方 reference/model card 资料使用，不从中推断完整训练 recipe、数据配比或 RL 细节。

<details markdown="1">
<summary>展开技术报告、blog、model card 与 repo 链接表</summary>

| 系列 | 模型/资料 | 链接 | 说明 |
|---|---|---|---|
| Qwen | Qwen3 | [Technical Report PDF](https://arxiv.org/pdf/2505.09388) / [GitHub](https://github.com/QwenLM/Qwen3) / [HF](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Qwen3 dense + MoE 主报告 |
| Qwen | Qwen3.5 main | [Release blog](https://qwen.ai/blog?id=qwen3.5) / [HF collection](https://huggingface.co/collections/Qwen/qwen35) / [Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) | 未见独立 general tech report PDF |
| Qwen | Qwen3.5-Omni | [Technical Report PDF](https://arxiv.org/pdf/2604.15804) | Omni 技术报告；不能完全替代 Qwen3.5 mainline 报告 |
| Qwen | Qwen3-Omni | [Technical Report PDF](https://arxiv.org/pdf/2509.17765) | Thinker-Talker omni-modal report |
| Qwen | Qwen3-VL | [Technical Report PDF](https://arxiv.org/pdf/2511.21631) | 长上下文多模态模型报告 |
| Qwen | Qwen3-Coder | [Release blog](https://qwen.ai/blog?id=qwen3-coder) | agentic coding 主线；当前按官方 blog/page 处理 |
| Qwen | Qwen3-Coder-Next | [Technical Report PDF](https://arxiv.org/pdf/2603.00729) | 80B/3B active coding MoE |
| DeepSeek | DeepSeek-V3 | [Technical Report PDF](https://arxiv.org/pdf/2412.19437) / [GitHub](https://github.com/deepseek-ai/DeepSeek-V3) / [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3) | V3 baseline：MLA、MTP、FP8、load balance |
| DeepSeek | DeepSeek-V4 | [Official HF PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf) / [HF Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | V4-Pro/V4-Flash；1M context |
| Kimi | Kimi K2 | [Technical Report PDF](https://arxiv.org/pdf/2507.20534) / [GitHub](https://github.com/MoonshotAI/Kimi-K2) / [HF](https://huggingface.co/moonshotai/Kimi-K2-Base) | agentic MoE 主线 |
| Kimi | Kimi K2.5 | [Technical Report PDF](https://arxiv.org/pdf/2602.02276) | multimodal agentic extension |
| Kimi | Kimi K2.6 | [Official blog](https://www.kimi.com/blog/kimi-k2-6) | 当前按官方 blog/model-card reference 处理 |
| GLM / Z.ai | GLM-4.5 | [Technical Report PDF](https://arxiv.org/pdf/2508.06471) / [Z.ai HF](https://huggingface.co/zai-org) | agentic/reasoning/coding MoE |
| GLM / Z.ai | GLM-5 / GLM-5.1 | [Technical Report PDF](https://arxiv.org/pdf/2602.15763) / [GitHub](https://github.com/zai-org/GLM-5) / [Z.ai HF](https://huggingface.co/zai-org) | 4K -> 200K mid-training、DSA、async RL |
| GLM / Z.ai | GLM-5V-Turbo | [Technical Report PDF](https://arxiv.org/pdf/2604.26752) | multimodal agent model |
| MiniMax | MiniMax-M1 | [Technical Report PDF](https://arxiv.org/pdf/2506.13585) | long-context reasoning baseline |
| MiniMax | MiniMax-M2 / M2.7 | [Technical Report PDF](https://arxiv.org/pdf/2605.26494) / [HF org](https://huggingface.co/MiniMaxAI) / [GitHub org](https://github.com/MiniMax-AI) | agent-native RL、Forge、192K context |
| MiniMax | MiniMax-M3 | [MSA Technical Report PDF](https://arxiv.org/pdf/2606.13392) / [Official blog](https://www.minimax.io/blog/minimax-m3) / [Model page](https://www.minimax.io/models/text/m3) / [HF](https://huggingface.co/MiniMaxAI/MiniMax-M3) / [GitHub](https://github.com/MiniMax-AI/MiniMax-M3) | PDF 主要是 MSA attention/infra，不是完整 M3 training report |
| LongCat | LongCat-Flash | [Technical Report PDF](https://arxiv.org/pdf/2509.01322) / [HF org](https://huggingface.co/meituan-longcat) / [GitHub org](https://github.com/meituan-longcat) | zero-computation experts、ScMoE |
| LongCat | LongCat-Flash-Lite | [Technical Report PDF](https://arxiv.org/pdf/2601.21204) | efficient LongCat language model |
| LongCat | LongCat-Flash-Thinking | [Technical Report PDF](https://arxiv.org/pdf/2509.18883) | thinking model 与 RL |
| LongCat | LongCat-Flash-Prover | [Official PDF](https://raw.githubusercontent.com/meituan-longcat/LongCat-Flash-Prover/main/LongCat_Flash_Prover_Technical_Report.pdf) | Lean4/formal reasoning |
| LongCat | LongCat-Flash-Omni | [Technical Report PDF](https://arxiv.org/pdf/2511.00279) | omni-modal audio-visual interaction |
| LongCat | LongCat-Next | [Official PDF](https://raw.githubusercontent.com/meituan-longcat/LongCat-Next/main/tech_report.pdf) | lexicalized multimodality |
| LongCat | LongCat-Image | [Technical Report PDF](https://arxiv.org/pdf/2512.07584) | image generation |
| LongCat | LongCat-Video | [Technical Report PDF](https://arxiv.org/pdf/2510.22200) | video generation |
| LongCat | LongCat-AudioDiT | [Technical Report PDF](https://arxiv.org/pdf/2603.29339) | waveform-latent diffusion TTS |
| LongCat | LongCat-Video-Avatar 1.5 | [Technical Report PDF](https://arxiv.org/pdf/2605.26486) | audio-driven avatar video generation |
| StepFun | Step-3 | [Technical Report PDF](https://arxiv.org/pdf/2507.19427) / [System Report PDF](https://raw.githubusercontent.com/stepfun-ai/Step3/main/Step3-Sys-Tech-Report.pdf) / [HF org](https://huggingface.co/stepfun-ai) / [GitHub org](https://github.com/stepfun-ai) | 模型报告 + 系统报告 |
| StepFun | Step 3.5 Flash | [Technical Report PDF](https://raw.githubusercontent.com/stepfun-ai/Step-3.5-Flash/main/step_3p5_flash_tech_report.pdf) | MIS-PO、agentic MoE |
| StepFun | Step 3.7 Flash | [Platform page](https://platform.stepfun.com/) | 未见独立 PDF；按平台模型 reference 处理 |
| StepFun | STEP3-VL-10B | [Technical Report PDF](https://arxiv.org/pdf/2601.09668) | compact multimodal model |
| StepFun | Step-Audio 2 | [Technical Report PDF](https://arxiv.org/pdf/2507.16632) | speech conversation、tool calling |
| StepFun | Step-Audio-R1 | [Technical Report PDF](https://arxiv.org/pdf/2511.15848) | audio reasoning |
| StepFun | Step-Audio-R1.5 | [Official PDF](https://raw.githubusercontent.com/stepfun-ai/Step-Audio-R1/main/Step-Audio-R1.5.pdf) | latest audio reasoning update |
| StepFun | Step-Video-T2V | [Technical Report PDF](https://arxiv.org/pdf/2502.10248) | text-to-video baseline |
| StepFun | Step-Video-TI2V | [Technical Report PDF](https://arxiv.org/pdf/2503.11251) | image-to-video baseline |
| MiMo / Xiaomi | MiMo-7B | [Technical Report PDF](https://raw.githubusercontent.com/XiaomiMiMo/MiMo/main/MiMo-7B-Technical-Report.pdf) | reasoning-focused 7B |
| MiMo / Xiaomi | MiMo-V2-Flash | [Technical Report PDF](https://arxiv.org/pdf/2601.02780) / [HF org](https://huggingface.co/XiaomiMiMo) / [GitHub org](https://github.com/XiaomiMiMo) | 309B/15B active MoE、MOPD |
| MiMo / Xiaomi | MiMo-V2.5 / V2.5-Pro | [MiMo-V2.5 page](https://mimo.xiaomi.com/mimo-v2-5) / [MiMo-V2.5-Pro page](https://mimo.xiaomi.com/mimo-v2-5-pro) / [HF V2.5](https://huggingface.co/XiaomiMiMo/MiMo-V2.5) / [HF V2.5-Pro](https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro) | 未见独立 PDF；按官方 page/model card 处理 |
| MiMo / Xiaomi | MiMo-VL | [Technical Report PDF](https://arxiv.org/pdf/2506.03569) | VLM + GUI grounding |
| MiMo / Xiaomi | MiMo-VL-Miloco | [Technical Report PDF](https://arxiv.org/pdf/2512.17436) | home-centric on-device VLM |
| MiMo / Xiaomi | MiMo-Audio | [Technical Report PDF](https://arxiv.org/pdf/2512.23808) | audio language model |
| Nemotron / NVIDIA | Nemotron 3 Ultra | [Official NVIDIA PDF](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) / [NVIDIA HF](https://huggingface.co/nvidia) | hybrid Mamba-Attention LatentMoE |
| Nemotron / NVIDIA | Nemotron 3 Nano | [Technical Report PDF](https://arxiv.org/pdf/2512.20848) | efficient hybrid Mamba-Transformer MoE |
| Nemotron / NVIDIA | Nemotron 3 Nano Omni | [Technical Report PDF](https://arxiv.org/pdf/2604.24954) | omni multimodal model |
| Gemma / Google | Gemma 3 | [Technical Report PDF](https://arxiv.org/pdf/2503.19786) | dense edge/multimodal baseline |
| Gemma / Google | Gemma 4 | [Official releases](https://ai.google.dev/gemma/docs/releases) / [Model card](https://ai.google.dev/gemma/docs/core/model_card_4) / [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/) / [LiteRT notes](https://developers.google.com/edge/litert-lm/models/gemma-4) / [Visual guide](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4) / [Visual guide 12B](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4-12b) | 未见官方 tech report PDF；visual guide 仅作二级解释性资料 |
| Ant / InclusionAI | Ant Ling / BaiLing | [Technical Report PDF](https://arxiv.org/pdf/2503.05139) | Ling-Lite / Ling-Plus |
| Ant / InclusionAI | Ant Ling 2.0 | [Technical Report PDF](https://arxiv.org/pdf/2510.22115) / [Ling-V2 GitHub](https://github.com/inclusionAI/Ling-V2) | high-sparsity MoE、MTP、FP8 |
| Ant / InclusionAI | Ant Ling 2.6 | [Model docs](https://developer.ant-ling.com/en/docs/models/ling/) / [HF collection](https://huggingface.co/collections/inclusionAI/ling-26) / [Ling-2.6-1T](https://huggingface.co/inclusionAI/Ling-2.6-1T) / [Ling-2.6-flash](https://huggingface.co/inclusionAI/Ling-2.6-flash) | 未见独立 PDF；按官方 docs/model card 处理 |
| Tencent | Hy3-preview | [GitHub](https://github.com/Tencent-Hunyuan/Hy3-preview) / [HF](https://huggingface.co/tencent/Hy3-preview) / [HF Base](https://huggingface.co/tencent/Hy3-preview-Base) / [Website](https://hy.tencent.com/hy3-preview) | 未见独立 PDF；295B/21B active MoE reference |
| Baidu | ERNIE 5.0 | [Technical Report PDF](https://arxiv.org/pdf/2602.04705) | unified multimodal autoregressive MoE |
| Baidu | ERNIE 5.1 | [Official release blog](https://ernie.baidu.com/blog/posts/ernie-5.1-0508-release/) | 未见独立 PDF；非 open-weight reference |
| Baidu | ERNIE-Image | [GitHub](https://github.com/baidu/ERNIE-Image) | image generation reference |
| Microsoft | MAI-Thinking-1 | [Official PDF](https://microsoft.ai/pdf/mai-thinking-1.pdf) / [Microsoft AI](https://www.microsoft.ai) | 非 open-weight，但报告细节丰富 |
| AI2 | OLMo 2 | [Technical Report PDF](https://arxiv.org/pdf/2501.00656) / [GitHub](https://github.com/allenai/OLMo) / [HF org](https://huggingface.co/allenai) | fully open baseline |
| AI2 | OLMo 3 | [Technical Report PDF](https://arxiv.org/pdf/2512.13961) / [GitHub](https://github.com/allenai/OLMo) / [HF org](https://huggingface.co/allenai) | fully open model flow |
| Mistral | Magistral | [Technical Report PDF](https://arxiv.org/pdf/2506.10910) | reasoning model reference |
| Mistral | Devstral | [Technical Report PDF](https://arxiv.org/pdf/2509.25193) | coding agent reference |
| Mistral | Ministral 3 | [Technical Report PDF](https://arxiv.org/pdf/2601.08584) | small dense model reference |
| Meta | Llama 4 Scout / Maverick | [Official model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md) | 未见官方 tech report PDF；作为 model card reference |

</details>
