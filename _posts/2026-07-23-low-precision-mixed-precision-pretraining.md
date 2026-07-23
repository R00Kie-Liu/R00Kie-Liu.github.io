---
layout: post
title: "从 BF16 到 FP8/FP4：大语言模型低精度训练与混合精度预训练"
date: 2026-07-23
tags: [Pretraining, Mixed Precision, Megatron, MoE, Infra]
description: "记录 Megatron/MoE 预训练里的 BF16 baseline、FP8/FP4 recipe、router 精度边界，以及 NVIDIA、AMD、Intel 训练栈。"
---

## 写在前面

我在预训练里默认一直使用 BF16。最近 profile 之后发现，单纯开启 FP8 并没有带来 MFU 提升；但另一方面，开源更大规模的模型和训练报告里又越来越多地使用 FP8 甚至 FP4/NVFP4 做训练。

所以这份笔记主要想弄清楚：低精度/混合精度预训练到底是怎么做的，它和 Megatron/MoE、通信、router、optimizer、kernel 之间有什么关系，以及 FP8/FP4 是否一定能提升训练效率。它和之前的 [2025-2026 开源 LLM 演进综述]({% post_url 2026-06-13-open-llm-survey %}) 互补：那篇看模型演进，这篇只看训练精度和系统 recipe。

## 先说结论

如果站在“我已经有一个 Megatron MoE + BF16 baseline”的起点看，低精度训练不是要立刻把所有 tensor 都压到更低位宽，而是先判断哪些部分值得冒险，哪些部分应该保守：

- **BF16 仍然是预训练最可靠的地板。** 相较 FP16，BF16 保留了接近 FP32 的指数范围，通常更少依赖 loss scaling；做 MoE 预训练时，它也是最适合先拿来对齐 loss、吞吐、resume 和数据 pipeline 的 baseline。
- **FP8 值得关注，但它不是一个单独的 dtype 开关。** NVIDIA Hopper/Blackwell、AMD MI300/MI350、Intel Gaudi 等硬件与 Transformer Engine 类库把 FP8 推到了可规模化尝试的位置；真正要验证的是 format、scale、amax history、kernel、通信和 checkpoint 能不能组成稳定 recipe。
- **FP4、NVFP4、MXFP4/6、1-bit/1.58-bit 仍偏前沿。** 它们有明显效率潜力，但相比 BF16/FP8，对 scaling、随机舍入、块粒度、优化器和稳定性策略依赖更强。
- **MoE 的低精度边界要比 dense 模型更小心。** Expert FFN GEMM 可以积极试 FP8；router logits、gate、top-k、load-balance loss 和 expert bias/balance bias 我会优先保留 FP32，至少不低于 BF16。
- **端到端速度要看 time-to-loss，而不只是 tokens/s。** FP8/FP4 可能提高 GEMM 吞吐，但如果瓶颈在 all-to-all、小 expert GEMM、kernel fallback、scale 开销或收敛退化，最终未必比 BF16 更划算。

## 给自己的几条结论

- 先把 BF16 baseline 跑稳，再谈 FP8/FP4；低精度实验要和 loss 曲线、MFU、通信占比、resume 一起记录。
- 在 Megatron 里，低精度会和 TP/PP/DP/EP/CP、distributed optimizer、param gather、activation recompute 一起作用，不要只盯着一个 `--fp8-format`。
- MoE 里最值得先试的是 expert 侧 GEMM 和部分通信，最不值得冒进的是 router 决策路径。
- NVIDIA/Megatron/Transformer Engine 是资料最完整的主线；ROCm TE 和 Intel Gaudi TE 更适合关注它们能否复用或迁移这些 recipe。
- DeepSeek-V3 这类案例值得读，不只是因为“用了 FP8”，而是因为它把 FP8、MoE、通信重叠和大规模并行放在同一个训练系统里验证。

## 1. 问题边界：训练、微调、推理量化不要混在一起

低精度相关术语容易混用，我先把它们拆成三类任务：

| 场景 | 目标 | 常见技术 | 风险点 |
|---|---|---|---|
| 预训练 mixed precision | 从头训练或持续预训练时降计算/显存/带宽 | BF16、FP16、FP8 GEMM、低精度通信、低精度优化器状态 | loss spike、NaN/Inf、scale 不稳、梯度 underflow/overflow、长程训练偏差 |
| 微调/后训练低精度 | 降低 SFT/RLHF/LoRA 成本 | BF16/FP16、QLoRA NF4、8-bit optimizer、FP8 fine-tuning | 小数据集过拟合、adapter 精度、梯度累计误差 |
| 推理量化 | 降低部署显存和吞吐成本 | INT8/INT4/FP8 weight-only、KV cache quant、AWQ/GPTQ/SmoothQuant | 精度退化、长上下文误差、kernel/硬件适配 |

这份笔记重点讨论第一类：**LLM 低精度训练与混合精度预训练**。微调和推理量化只在必要时作为对照。

## 2. 精度格式与数值特点

![浮点精度格式示意图](/assets/img/blog/low-precision/precision-format-explainer-gpt-image-2-v2.png)

先把命名拆开看会容易很多。一个浮点数通常可以粗略分成三部分：

- **Sign**：符号位，表示正负。
- **Exponent**：指数位，主要决定这个格式能表示多大或多小的数，也就是动态范围。
- **Mantissa**：尾数位，也叫 significand/fraction，主要决定两个相邻可表示数字之间有多细，也就是精度。

所以 **E4M3** 的意思就是：这个 FP8 格式里有 **4 个 exponent bits** 和 **3 个 mantissa bits**。再加上 1 个 sign bit，正好是 8 bit：`1 + 4 + 3 = 8`。类似地，**E5M2** 就是 `1 sign + 5 exponent + 2 mantissa`。

直觉上，E4M3 把更多 bit 留给 mantissa，因此数值刻度更细，常用于 forward 的权重和激活；E5M2 把更多 bit 留给 exponent，因此动态范围更大，常用于 backward 的梯度。Transformer Engine 里常说的 FP8 hybrid recipe，核心就是在不同方向/不同张量上混用 E4M3 和 E5M2。

### 2.1 FP32、TF32、FP16、BF16、FP8、FP4

| 格式 | 典型位宽 | 特点 | LLM 训练中的常见用途 |
|---|---:|---|---|
| FP32 | 32-bit | 动态范围和精度都高，成本高 | master weights、优化器内部关键计算、loss/metric、部分归一化或累加 |
| TF32 | 19-bit-ish compute format | NVIDIA Ampere 后 Tensor Core 默认加速 FP32 matmul 的一种路径 | 某些 FP32 matmul 加速，LLM 预训练主流已更多转向 BF16/FP8 |
| FP16 | 16-bit | mantissa 比 BF16 多，指数范围小 | 早期 mixed precision 训练，常需 dynamic loss scaling |
| BF16 | 16-bit | 指数范围接近 FP32，mantissa 较少 | 当前 LLM 预训练最常见 baseline |
| FP8 E4M3 | 8-bit | 4 位指数、3 位尾数，精度相对更好，动态范围较小 | forward 权重/激活常用 |
| FP8 E5M2 | 8-bit | 5 位指数、2 位尾数，动态范围更大，精度较低 | backward 梯度常用 |
| MXFP8 / microscaling FP8 | 8-bit + block scale | 以小块为单位缩放，降低 per-tensor scaling 的误差 | Blackwell/新 TE recipe，适合更细粒度 FP8 |
| NVFP4 / MXFP4 | 4-bit + microscale | NVFP4 是 E2M1 元素 + 16 元素 block scale + tensor scale；比 FP8 更依赖 recipe | Blackwell/TE FP4 训练研究，需要 RHT、2D scaling、随机舍入和高精度保护层 |
| INT8/INT4 | integer | 通常不保留浮点指数，依赖量化 scale/zero point | 以推理和 QAT/微调为主，纯预训练更难 |

### 2.2 为什么 BF16 适合 LLM 预训练

FP16 的主要问题是指数范围窄，梯度分布在深层网络中容易出现 underflow/overflow，因此常配合 dynamic loss scaling。BF16 虽然尾数更短，但指数位与 FP32 相同数量级，能覆盖更大动态范围，因此在大模型预训练中通常更稳。实践中，许多框架会保留 FP32 master parameters 和优化器状态，同时用 BF16 参数副本做 forward/backward。

一个常见 BF16 mixed precision recipe：

```text
模型参数计算副本：BF16
Forward GEMM / Attention / MLP：BF16 Tensor Core
Backward GEMM：BF16 Tensor Core
梯度通信：BF16 或 FP32，取决于稳定性和通信开销
Master weights：FP32
Adam 一阶/二阶矩：FP32，或用更激进的低精度 optimizer state
LayerNorm/RMSNorm、Softmax、Loss：按框架策略保留高精度或混合实现
```

### 2.3 FP8 的关键：格式 + scaling + kernel + 框架

FP8 的 E4M3/E5M2 格式本身不是完整方案。因为 8-bit 浮点无法同时覆盖所有 layer、所有 step、所有张量分布，必须搭配 scaling：

| scaling 策略 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| per-tensor current scaling | 当前 step 用当前 tensor amax 计算 scale | 简单，响应快 | 可能引入同步/统计开销，局部异常值影响大 |
| delayed scaling | 用历史 amax 窗口计算下一步 scale | 工程上稳定，TE 常用 | 对分布突变反应慢，需要 amax history |
| blockwise / subchannel scaling | 按矩阵块、通道或子通道缩放 | 减少单个异常值污染整张量，适合 FP8/FP4 | 需要硬件/kernel/框架支持 |
| microscaling | 更细粒度的小块缩放，如 MXFP8 | 精度-效率折中更好 | 工程复杂度更高，依赖新硬件 |

NVIDIA Transformer Engine 的 FP8 primer 明确提到：forward 中权重和激活通常更需要精度，E4M3 更适合；backward 的梯度更需要动态范围，E5M2 更合适。Transformer Engine 的 delayed scaling recipe 会维护 amax history、FP8 format 等元信息。  
参考：[NVIDIA Transformer Engine FP8/FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)

### 2.4 NVFP4 不是“直接把 FP8 换成 FP4”

NVIDIA 的 `Pretraining Large Language Models with NVFP4` 技术报告很值得单独看。它的重点不是证明 FP4 一定端到端更快，而是给出一套能让 4-bit pretraining 长程稳定的 recipe。报告里训练了一个 12B hybrid Mamba-Transformer 模型到 10T tokens，并把 NVFP4 训练曲线和 FP8 baseline 对齐到可比较的范围。

我从这篇里记下的 NVFP4 recipe：

1. **NVFP4 格式本身**：E2M1 FP4 元素，16 个元素共享一个 E4M3 block scale，再加一个 FP32 tensor-level scale；相比 MXFP4 的 32 元素 block 和 UE8M0 scale，NVFP4 的 block 更小、scale 更细。
2. **高精度保护层**：不是所有 Linear 都降到 FP4。报告建议保留一部分数值敏感 Linear 在高精度，尤其是网络后段；Nemotron 3 Ultra 里还把 final 15% network、Mamba output projection、latent projection、QKV/attention projection、MTP layer、embedding 等保在更高精度。
3. **Random Hadamard Transform, RHT**：主要用于 weight-gradient GEMM 的输入，目的是把 block-level outlier 打散，降低局部 amax 对 FP4 block 的污染。
4. **2D weight scaling**：权重用 16x16 的二维 block scaling，让 forward/backward 看到的量化表示更一致；activation/gradient 则使用 1x16 scaling。
5. **stochastic rounding on gradients**：随机舍入主要放在梯度上；weights/activations 使用 round-to-nearest-even。报告里的 ablation 显示，RHT、2D scaling、随机舍入、高精度保护层缺一项都会让收敛变差。

所以 NVFP4 对我的启发不是“FP4 会自动比 BF16/FP8 快”，而是：**bit-width 越低，recipe 越像一个训练算法**。FP8 还可以被理解成 GEMM/scaling/communication recipe；FP4 已经需要把量化格式、RHT、block scaling、rounding、敏感层保护和训练后段是否切回高精度一起设计。

参考：[Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)

## 3. 混合精度训练的系统组成

大规模 LLM 训练中的“低精度”至少包含以下层次：

1. **算子计算精度**：Linear、GEMM、attention、MLP 是否使用 BF16/FP8/FP4 Tensor Core 或 matrix core。
2. **累加精度**：低精度输入的 matmul 是否用 FP16/FP32 累加，累加后是否截断。
3. **参数存储精度**：模型权重主副本、训练副本、分片参数的 dtype。
4. **梯度精度**：局部梯度、累积梯度、梯度 reduce-scatter/all-reduce dtype。
5. **优化器状态精度**：AdamW 的 `exp_avg`、`exp_avg_sq` 和 master params 是否 FP32、BF16、FP16、FP8。
6. **通信精度**：TP/DP/EP/CP 下 all-reduce、all-gather、reduce-scatter、all-to-all 是否压缩。
7. **激活存储精度**：activation checkpointing/recompute 与 FP8 activation cache 的关系。
8. **特殊模块精度**：LayerNorm/RMSNorm、Softmax、RoPE、cross entropy、router、MoE gate 是否保留高精度。
9. **checkpoint 精度与可恢复性**：保存的是 FP32 master、BF16 权重、FP8 权重，还是分布式状态。

## 4. Megatron 系列框架：低精度预训练的主战场

这里不重新介绍 TP/PP/EP 是什么，重点是把低精度会碰到的边界标出来：哪些开关只是改变 GEMM dtype，哪些会影响通信、分片参数、optimizer state 和 checkpoint 恢复。

### 4.1 Megatron 中 mixed precision 的典型路径

Megatron/Megatron Bridge 中常见的精度 recipe：

| recipe | 适用阶段 | 说明 |
|---|---|---|
| `fp32` | debug、小模型基线 | 最稳但成本最高 |
| `fp16_mixed` | 较老 GPU 或已有 FP16 pipeline | 需要 loss scaling，LLM 大规模下不如 BF16 稳 |
| `bf16_mixed` | 大多数 LLM 预训练 baseline | 计算使用 BF16，保留 FP32 master/optimizer |
| `bf16_with_fp8_*` | Hopper/MI300/Gaudi 等 FP8 硬件 | BF16 框架下部分 GEMM/通信/参数 gather 使用 FP8 |
| `mxfp8` / `nvfp4` | Blackwell/新硬件实验 | 更激进，依赖特定 TE recipe 和 kernel |

Megatron Bridge 文档说明 mixed precision config 会自动更新模型、优化器和 DDP 设置；半精度训练中层计算使用 FP16/BF16，同时模型状态如 optimizer states 和 master parameters 保留单精度。  
参考：[Megatron Bridge Mixed Precision Training](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/mixed-precision.html)

### 4.2 一个 Megatron BF16 baseline 配置骨架

下面是 Llama/Qwen/DeepSeek-like decoder-only 预训练的简化配置骨架：

```bash
torchrun \
  --nproc_per_node 8 \
  pretrain_gpt.py \
  --use-mcore-models \
  --num-layers 32 \
  --hidden-size 4096 \
  --ffn-hidden-size 14336 \
  --num-attention-heads 32 \
  --seq-length 8192 \
  --max-position-embeddings 8192 \
  --position-embedding-type rope \
  --normalization RMSNorm \
  --swiglu \
  --disable-bias-linear \
  --attention-backend fused \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --bf16 \
  --grad-reduce-in-bf16 \
  --use-distributed-optimizer \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --sequence-parallel \
  --overlap-grad-reduce \
  --overlap-param-gather \
  --cross-entropy-loss-fusion \
  --log-throughput
```

这个 baseline 的目标是先验证：

- loss 曲线正常；
- 吞吐、显存、MFU 可测；
- checkpoint 可保存和恢复；
- 数据 pipeline、tokenizer、长上下文 attention mask 都正确。

### 4.3 在 BF16 baseline 上启用 FP8

NVIDIA Megatron-LM 的 Llama3 H100 FP8 示例在 BF16 配置基础上叠加了：

```bash
--fp8-format hybrid
--fp8-amax-history-len 1024
--fp8-amax-compute-algo max
--fp8-param-gather
```

这些参数背后的含义：

| 参数 | 作用 |
|---|---|
| `--fp8-format hybrid` | 通常表示 forward/backward 使用不同 FP8 格式组合，例如 E4M3/E5M2 |
| `--fp8-amax-history-len` | delayed scaling 中保存 amax 历史窗口 |
| `--fp8-amax-compute-algo max` | 用历史 amax 的最大值计算 scale |
| `--fp8-param-gather` | 参数 gather 阶段使用 FP8，降低通信/显存压力 |

我会避免一开始就全量开启所有 FP8 能力。更稳的顺序是：

1. BF16 全流程跑通；
2. 开启 FP8 GEMM；
3. 开启 FP8 param gather；
4. 观察 loss spike、amax、scale、grad norm；
5. 再评估 optimizer state 或通信进一步压缩。

### 4.4 并行策略与低精度的耦合

低精度训练的收益在大规模并行下会被放大，也会暴露更多风险：

| 并行策略 | 与低精度相关的关键点 |
|---|---|
| TP, tensor parallel | matmul 切分后 scale 统计粒度变化；TP all-reduce 可成为通信瓶颈 |
| PP, pipeline parallel | micro-batch 更小会影响 batch 统计和吞吐；bubble 与 recompute 策略相关 |
| DP / distributed optimizer | 梯度 reduce-scatter dtype、参数 all-gather dtype、optimizer state dtype 影响显存和稳定性 |
| EP, expert parallel | MoE all-to-all 通信巨大，低精度通信很有吸引力；router/gate 通常需要高精度保护 |
| CP, context parallel | 长上下文训练下 attention/activation 显存压力大，低精度和 recompute 更关键 |
| sequence parallel | 降低 activation 显存，常与 TP 和 BF16/FP8 同时启用 |

对 MoE 模型，FP8 的难点更大：expert 的 token 分布不均、router logits、all-to-all 通信、expert GEMM 形状变化都会影响 scale 和 kernel 效率。DeepSeek-V3 的工程价值很大程度就在于它把 FP8 与 MoE、大规模并行、通信重叠一起验证。

MoE 混合精度里最需要保守处理的是 router。dense 层的低精度误差通常是连续扰动，而 router 的误差会被 top-k/argmax 放大成离散 expert 选择变化，进一步改变 expert load、expert specialization、token drop、all-to-all 形状和稀疏梯度分布。因此我会让 router 的关键决策路径保持 FP32，至少不低于 BF16：router linear 可以用 BF16 计算，但 logits 最好 cast 到 FP32 后再做 softmax/sigmoid、top-k、load-balance loss 和 expert bias/balance bias 更新。不要轻易把 router logits 或 top-k 前 gate score 放到 FP8。

## 5. 代表性工作与论文脉络

### 5.1 Mixed Precision Training

经典 mixed precision training 思路是：把适合低精度的运算放到 FP16/BF16，保留 FP32 master weights 或关键累加，并通过 loss scaling 避免 FP16 梯度 underflow。对现代 LLM 来说，这一思想演化为 BF16 baseline 和更复杂的 FP8 recipe。

### 5.2 FP8 Formats for Deep Learning

FP8 格式论文提出 E4M3 和 E5M2 两种 8-bit 浮点格式。它的关键贡献不是说“所有地方都用 FP8”，而是定义了适合深度学习训练/推理交换的低位浮点格式，为硬件、编译器和框架对齐提供基础。

参考：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

### 5.3 Transformer Engine FP8

Transformer Engine 将 FP8 训练工程化，核心包括：

- FP8-aware Linear/LayerNorm/Attention 模块；
- E4M3/E5M2 hybrid format；
- delayed scaling、current scaling、block scaling；
- amax history 维护；
- Hopper/Blackwell Tensor Core 支持；
- PyTorch/JAX API；
- 与 Megatron/NeMo 等框架集成。

参考：

- [NVIDIA Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)
- [NVIDIA Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)

### 5.4 DeepSeek-V3：超大规模 FP8 预训练案例

DeepSeek-V3 是公开资料中最重要的 FP8 预训练案例之一：

- 总参数：671B；
- 每 token 激活参数：37B；
- 预训练数据：14.8T tokens；
- 架构：MoE；
- 训练系统：FP8 mixed precision、DualPipe、通信计算重叠、跨节点 all-to-all 优化；
- 我会重点看：公开技术报告里，FP8 mixed precision training 不只是小规模 ablation，而是被放到了超大规模 MoE 预训练系统里验证。

DeepSeek-V3 对低精度训练的启示：

1. FP8 需要与 MoE 通信优化一起设计；
2. 需要细粒度量化策略，而不是粗暴 per-tensor；
3. 累加精度和 scale 粒度决定稳定性上限；
4. 训练框架要能把 pipeline、expert parallel、通信 overlap 和 dtype recipe 统一调度；
5. 低精度不只是吞吐优化，也影响可训练模型规模和单位 token 成本。

参考：[DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)

### 5.5 MS-AMP：FP8 与 ZeRO/DeepSpeed 路线

MS-AMP 代表了另一条 FP8 系统化路线：在 PyTorch/DeepSpeed/Megatron 生态上扩展 automatic mixed precision，把 FP8 用到权重、梯度、通信、优化器状态和 ZeRO 并行训练中。

它的价值在于说明 FP8 训练不应只看 forward GEMM，还要覆盖：

- gradient communication；
- optimizer state；
- ZeRO partition；
- distributed checkpoint；
- 与既有训练框架的兼容。

参考：[MS-AMP Introduction](https://azure.github.io/MS-AMP/docs/introduction/)

### 5.6 NVIDIA NVFP4：4-bit 预训练 recipe

NVIDIA 的 NVFP4 报告可以看作 FP4 预训练从“格式/硬件潜力”走向“训练 recipe”的标志性工作。它的实验对象是 12B hybrid Mamba-Transformer，训练到 10T tokens，并与 FP8 baseline 对齐；报告强调自己主要验证算法和训练方法，而不是做系统吞吐 benchmark。

它对这篇笔记最有用的地方是把 FP4 训练拆成了几个必须同时考虑的组件：

- NVFP4 格式：E2M1 FP4 元素、16 元素 block、E4M3 block scale、FP32 tensor scale；
- 高精度保护层：保留数值敏感 linear，尤其是网络后段；
- RHT：对 Wgrad 输入做 Random Hadamard Transform，缓解 block-level outlier；
- 2D weight scaling：减少 forward/backward 中权重量化表示不一致；
- gradient stochastic rounding：降低梯度量化偏差；
- 后段切高精度：如果 FP4 和高精度之间仍有 loss gap，可以在 decay 前后切回 BF16/更高精度补一段。

这让我对 FP4 的态度更保守：FP4 是值得跟的方向，但它更像“新训练算法 + 新 kernel + 新硬件”的组合，而不是 Megatron 脚本里多开一个 dtype flag。

参考：

- [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)
- [NVIDIA Transformer Engine NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

### 5.7 BitNet b1.58 与 1-bit/ternary LLM

BitNet b1.58 路线探索极低比特权重训练，常见形式是 ternary weights，例如 {-1, 0, +1}。这一方向的吸引力是推理和训练计算都可能大幅下降，但它不是 BF16/FP8 的直接替代，而是更深的模型结构与训练范式变化。

适合作为研究参考的问题：

- 低比特权重是否需要改变 optimizer？
- 低比特模型的 scaling law 是否不同？
- 低比特训练是否能保持 reasoning/coding 能力？
- 极低比特训练与 MoE、长上下文、多模态是否兼容？

## 6. 厂商路线：NVIDIA、Intel、AMD

### 6.1 NVIDIA：最完整的 FP8/FP4 全栈

NVIDIA 的路线最完整，覆盖硬件、kernel、库、框架和模型 recipe：

| 层级 | 代表组件 |
|---|---|
| 硬件 | A100 TF32/BF16，H100 FP8 Transformer Engine，Blackwell MXFP8/NVFP4/FP4 |
| kernel/library | cuBLASLt、FlashAttention/fused attention、Transformer Engine |
| 训练框架 | Megatron Core、Megatron-LM、NeMo、Megatron Bridge |
| recipe | BF16 mixed、BF16+FP8 delayed/current/subchannel scaling、MXFP8、NVFP4 |
| 模型生态 | Llama/Qwen/DeepSeek/MoE recipe、checkpoint conversion、performance summary |

H100 官方资料强调第四代 Tensor Core 和 FP8 Transformer Engine；Blackwell 进一步引入 micro-tensor scaling，面向 MXFP8 和 FP4/NVFP4。  
参考：

- [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Transformer Engine FP8/FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [NVFP4 Transformer Engine Documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

我的理解是：NVIDIA 的优势在于生态闭环。麻烦也在这里，许多最先进 recipe 对硬件和 TE 版本绑定较深，迁移到其他平台时不能只搬配置，需要重新验证 kernel、scale 和通信策略。

### 6.2 Intel：Gaudi FP8 + Xeon BF16/AMX

Intel 的低精度路线分两条：

1. **CPU/XPU 侧**：Xeon + AMX/BF16，配合 Intel Extension for PyTorch、oneDNN、Neural Compressor，用于 CPU 训练/推理优化和部分 XPU 工作流。
2. **Gaudi 加速器侧**：Gaudi 2/3 支持 BF16/FP8 训练，提供 Intel Gaudi Transformer Engine、Optimum Habana、Gaudi Software Suite。

Intel Gaudi Transformer Engine 文档说明其提供 FP8 PyTorch 模块和类似 AMP 的 API，用于配置和控制 FP8-enabled modules。Gaudi 3 白皮书强调 FP8/BF16 compute、HBM 容量和带宽。  
参考：

- [Intel Gaudi Transformer Engine FP8 Training](https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html)
- [Intel Gaudi 3 AI Accelerator White Paper](https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf)
- [Intel Neural Compressor Mixed Precision](https://intel.github.io/neural-compressor/latest/docs/source/mixed_precision.html)
- [Hugging Face Accelerate: Intel Gaudi](https://huggingface.co/docs/accelerate/usage_guides/gaudi)

我的理解是：Intel 的机会在于成本和开放框架适配，尤其是 Gaudi 在云服务中的性价比路线。挑战在于生态规模、kernel 覆盖和与 Megatron/DeepSpeed 主线 recipe 的一致性验证。

### 6.3 AMD：ROCm + MI300/MI350 + TransformerEngine

AMD 的路线是通过 ROCm 生态追赶并兼容主流训练栈：

| 层级 | 代表组件 |
|---|---|
| 硬件 | MI250、MI300X、MI325X、MI350 系列 |
| 软件 | ROCm、hipBLASLt、Composable Kernel、AITER、ROCm TransformerEngine |
| 框架 | ROCm/Megatron-LM、PyTorch ROCm、vLLM ROCm、DeepSpeed 适配 |
| 低精度 | BF16、FP8、MXFP8/MXFP4 等新格式逐步支持 |

ROCm TransformerEngine 仓库说明其目标是在 AMD GPU 上加速 Transformer，包括在 MI300 GPU 上使用 FP8 precision，以降低训练和推理的内存占用并提升性能。  
参考：

- [ROCm TransformerEngine GitHub](https://github.com/ROCm/TransformerEngine)
- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/docs-6.1.5/how-to/tuning-guides/mi300x/index.html)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

我的理解是：AMD 的重要性在于提供 NVIDIA 之外的大规模训练硬件选择。短期关注点是 ROCm TE 与 Megatron/DeepSpeed recipe 的稳定性、FlashAttention/attention backend 覆盖、通信库 RCCL 的规模化表现，以及 MI300X/MI350 上 FP8 kernel 的成熟度。

## 7. 开源模型与低精度训练参考

### 7.1 直接与低精度训练强相关

| 模型/工作 | 规模 | 低精度相关点 | 参考价值 |
|---|---:|---|---|
| DeepSeek-V3 | 671B total, 37B active | FP8 mixed precision pretraining, MoE, DualPipe | 超大规模 FP8 预训练系统样本 |
| Nemotron 3 Super/Ultra | Ultra: 550B total, 55B active | NVFP4 pretraining, LatentMoE, MTP, high-precision protected layers | 公开 NVFP4 预训练 recipe 和 Megatron/NVIDIA 生态样本 |
| BitNet b1.58 系列 | 2B 级公开实验较典型 | 1-bit/ternary 权重训练 | 极低比特训练范式参考 |
| NVIDIA NVFP4 work | 12B 级实验等 | 4-bit pretraining recipe | FP4 从推理走向预训练的前沿 |
| MS-AMP examples | 框架实验 | FP8 weights/grad/comm/optimizer/ZeRO | FP8 系统化训练组件参考 |

### 7.2 与 2025-2026 开源 LLM 综述的关系

上一篇开源 LLM 综述的核心判断是：2025-2026 年开放模型已经进入“训练流程 + 数据工程 + 架构 + 系统”的协同优化阶段，能力提升不再只靠 dense 参数堆叠，而是由 MoE、长上下文、数据工程、后训练、RL、低精度和 serving topology 一起决定。对低精度预训练来说，这意味着模型案例不能只按 benchmark 排序，而要按“架构是否适合低精度、训练系统是否披露、MoE/通信是否成为瓶颈、硬件/框架 recipe 是否可复用”来组织。

其中有三条线索和这份笔记最相关：

1. **Frontier open-weight 模型默认走 MoE**：DeepSeek、Qwen、Kimi、GLM、MiniMax、LongCat、MiMo、Step、Nemotron 等都在 total params 与 active params 之间寻找成本/能力平衡。
2. **低精度与系统调度绑定**：DeepSeek-V3 的 FP8 training、Nemotron 的 NVFP4 pretraining、Step-3 的 FP8 serving/AFD，说明低精度已经成为训练和 serving topology 的一部分；但训练 recipe 和推理量化要分开记。
3. **优化器与低精度同样重要**：Kimi K2 的 MuonClip/QK-Clip、GLM 的 Muon Split、DeepSeek-V4 的 Muon + AdamW exceptions，都提示低精度训练稳定性不能只看 dtype，也要看 optimizer、LR schedule、router/balance 和 attention logits。

### 7.3 明确披露低精度预训练 recipe 的模型/报告

这里不再列完整开源模型表。很多 open-weight 模型虽然重要，但如果没有披露预训练低精度 recipe，就不适合作为这一节的主参考；否则容易把“架构/optimizer/serving 很有参考价值”和“低精度训练 recipe 明确”混在一起。

| 模型/报告 | 低精度 recipe | 我主要看什么 |
|---|---|---|
| DeepSeek-V3 Technical Report | FP8 mixed precision pretraining；FP8 GEMM；tile-wise/block-wise scaling；FP32 accumulation；BF16/FP32 保护 embedding、output head、MoE gate、norm、attention、master weights/gradients；FP8 activation cache 和 MoE dispatch，combine 保 BF16 | 超大规模 MoE FP8 预训练如何和 DualPipe、all-to-all overlap、aux-loss-free balance 放在一起 |
| Nemotron 3 Ultra Technical Report | NVFP4 pretraining；TE/cuBLAS NVFP4 GEMM for fprop/dgrad/wgrad；E2M1 + 2D block weight quant；RHT on Wgrad inputs；gradient stochastic rounding；final 15% network、Mamba output、latent projection、QKV/attention、MTP、embedding 保高精度 | NVFP4 recipe 如何从 12B/10T 验证扩展到 550B-A55B/20T 的 open model |
| NVIDIA `Pretraining Large Language Models with NVFP4` | 12B/10T NVFP4 vs FP8 baseline；高精度保护层、RHT、2D weight scaling、gradient stochastic rounding、后期切 BF16 | FP4 预训练的算法组件和 ablation，作为理解 Nemotron NVFP4 的基础 |

参考：

- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
- [Nemotron 3 Ultra Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf)
- [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)
- [NVIDIA Nemotron 3: Efficient and Open Intelligence](https://arxiv.org/abs/2512.20856)
- 个人网站文章：[2025-2026开源LLM演进综述：From Scaling to Agentic system]({% post_url 2026-06-13-open-llm-survey %})

### 7.4 一个可复用的低精度标注表

如果后续继续维护开源 LLM 综述，可以在每个模型条目后追加下面四列，把“模型能力进展”和“训练系统进展”对齐：

```text
训练精度：BF16 / FP8 / unknown
训练框架：Megatron / DeepSpeed / internal / unknown
硬件：H100 / B200 / MI300X / Gaudi / unknown
低精度亮点：FP8 training / FP8 weights / FP8 comm / FP4 / low-bit optimizer / unknown
```

低精度训练专门表里，优先补 DeepSeek-V3 和 Nemotron 3 Super/Ultra 就够了；Step-3 更适合放到 serving topology/AFD 笔记里，Qwen/Kimi/GLM 更适合放到 MoE 架构、optimizer 或长上下文综述里。

## 8. 实验备忘：从 Megatron BF16 MoE baseline 往低精度走

### 8.1 实验路线

如果已有 BF16 训练配置，我会按四阶段拆，而不是直接把脚本改成“全量 FP8”：

1. **BF16 短程 sanity run**  
   目的：验证数据、tokenizer、loss、mask、checkpoint 和 resume。只有定位数值问题时，才需要把局部模块退回 FP32 debug。

2. **BF16 baseline**  
   目的：得到稳定收敛曲线和吞吐 baseline。记录 tokens/s、TFLOPs、MFU、显存峰值、通信占比。

3. **BF16 + FP8 GEMM**  
   目的：验证 FP8 scaling recipe 对 loss 的影响。观察 amax/scale、NaN/Inf、grad norm、loss spike。

4. **FP8 通信/参数 gather/优化器状态**  
   目的：进一步降低内存和带宽。逐项打开，不要一次性全开。

### 8.2 FP8/FP4 一定比 BF16 快吗

不一定。更低 bit-width 通常意味着更高理论算力、更低显存占用和更低带宽压力，但这些收益只有在实际瓶颈被它们命中时才会转化成端到端速度。训练里至少要区分两个指标：

- **step throughput**：每秒能处理多少 token，或者每 step 多快。
- **time-to-loss / time-to-quality**：训练到同等 loss 或同等下游能力需要多久。

FP8/FP4 可能让单步更快，但如果数值噪声导致 loss spike、收敛变慢，或者需要更保守的 learning rate、更长 warmup、更多重训 token，那么最终 time-to-loss 未必优于 BF16。MoE 里还要额外考虑 all-to-all、router/gate、expert imbalance 和小 expert GEMM 的 kernel 利用率。

一个简单判断表：

| 场景 | FP8/FP4 相比 BF16 更可能加速吗 | 原因 |
|---|---|---|
| 大 GEMM 占主导，硬件有原生 FP8/FP4 Tensor Core 或 Matrix Core | 更可能 | 低精度计算吞吐能真正吃满 |
| 显存或参数/激活带宽是瓶颈 | 可能 | 低精度减少读写和 buffer 压力 |
| DP/TP/EP 通信是瓶颈 | 不一定 | 低精度通信有帮助，但通信调度、拓扑和 overlap 仍决定端到端速度 |
| MoE all-to-all 或 router 成为瓶颈 | 不一定 | expert GEMM 可能快，但 dispatch/combine 和负载不均可能抵消收益 |
| 小 batch、小 expert token 数、小 GEMM shape | 可能不快 | kernel 吃不满，cast/scale 开销占比上升 |
| FP8 scaling、amax 统计、格式转换开销明显 | 可能不快 | FP8 是 recipe，不是免费 dtype |
| FP4 缺少成熟 kernel 或需要频繁 unpack/dequant | 可能更慢 | 软件开销可能抵消理论吞吐 |
| 收敛变差，需要更多 token 才到同等质量 | 可能更慢 | 单步快不等于训练目标更快 |

因此我自己的实验记录里要同时放 BF16 baseline、FP8/FP4 step throughput、validation loss 曲线、loss spike、收敛到同等 loss 的 token 数和总 wall-clock。只报 tokens/s 很容易高估低精度收益。

### 8.3 监控指标

| 指标 | 为什么重要 |
|---|---|
| train loss / validation loss | 首要收敛指标 |
| loss spike 次数和恢复时间 | 低精度不稳定常先体现在 spike |
| grad norm | 检测梯度爆炸/消失 |
| NaN/Inf 计数 | 直接数值异常 |
| FP8 amax / scale 分布 | 判断 scaling 是否健康 |
| overflow/underflow 统计 | FP16/FP8 重要 |
| tokens/s | 端到端吞吐 |
| MFU | 模型 FLOP 利用率 |
| HBM 利用率 | 显存和带宽瓶颈 |
| NCCL/RCCL/HCCL 通信时间 | 分布式瓶颈 |
| all-to-all 时间 | MoE 模型尤其重要 |
| checkpoint 恢复一致性 | 大规模训练容错必需 |
| time-to-loss | 判断低精度是否真的比 BF16 更快 |

### 8.4 哪些模块要谨慎降精度

如果只想先定一条保守边界，我会从这些模块开始保护：

- embedding 和 lm_head；
- final logits 和 cross entropy；
- LayerNorm/RMSNorm 统计；
- attention softmax；
- MoE router/gate logits；
- RoPE 相关位置计算；
- loss accumulation；
- optimizer update 内部计算；
- very small batch 或长尾 expert 的 token group。

MoE 场景可以按下面的精度边界起步：

| MoE 模块 | 我的起步精度 | 说明 |
|---|---|---|
| expert FFN GEMM | BF16 baseline；稳定后试 FP8 | 这是 MoE 中最值得低精度化的主要计算部分 |
| router linear | BF16 可接受；追求稳定可 FP32 | router 计算量小，没必要过度压榨 |
| router logits | FP32 优先，至少 BF16 | 低精度扰动会改变 top-k expert 选择 |
| softmax/sigmoid gate | FP32 推荐 | gate 分数直接决定 dispatch 和 combine 权重 |
| top-k / argmax / sorting | logits 转 FP32 后做 | 离散决策对量化误差特别敏感 |
| load-balance loss / aux loss | FP32 推荐 | 长期影响 expert load 和 specialization |
| expert bias / balance bias 更新 | FP32 推荐 | bias 更新是负载均衡控制环，建议高精度 |
| dispatch mask / expert indices | integer | 不涉及浮点精度，但要保证确定性和一致性 |
| token dispatch/combine activation 通信 | BF16 起步；稳定后试 FP8 | all-to-all 是 MoE 大头，但 FP8 通信需要观察 loss 和 expert load |
| expert optimizer state | FP32/BF16 起步 | expert 稀疏更新，冷门 expert 的 Adam 统计更容易受低精度影响 |

### 8.5 常见问题与排查

| 现象 | 可能原因 | 排查方向 |
|---|---|---|
| loss 一开始就 NaN | dtype 配置过激、scale 初始化不当、数据异常 | 先退回 BF16；检查 tokenizer/mask/loss；关闭 FP8 param gather |
| 周期性 loss spike | delayed scaling 窗口不合适、amax 异常值、MoE expert 分布突变 | 记录 amax history；改 current/block scaling；保护 router |
| 吞吐没提升 | kernel fallback、GEMM shape 不友好、通信主导 | profile kernel；检查 TE 是否生效；提升 batch/sequence；优化 overlap |
| 显存没下降 | master params/optimizer state 仍 FP32、activation 占主导 | 开 distributed optimizer/FSDP；activation recompute；检查 param gather dtype |
| 多机不稳定 | 通信 dtype/scale 不一致、checkpoint 恢复问题 | 固定 seed；检查分布式 scale 同步；做短程 resume 测试 |
| MoE all-to-all 爆炸 | expert imbalance、token dispatcher 不优 | router loss/bias；expert parallel 配置；all-to-all overlap |

## 9. 对研究工作的可能切入点

可以从以下方向选题：

1. **FP8 scaling 粒度与稳定性**  
   比较 per-tensor、per-channel、blockwise、microscaling 对 LLM loss spike 和 downstream benchmark 的影响。

2. **FP8 optimizer state**  
   研究 AdamW 一阶/二阶矩低精度存储对长期预训练的影响，尤其是 warmup、lr decay 和 loss spike 阶段。

3. **低精度通信**  
   比较 BF16/FP8 gradient reduce、param gather、MoE all-to-all 的通信收益与收敛退化。

4. **MoE + FP8**  
   DeepSeek-V3 之后，MoE 成为 FP8 系统研究的核心场景。router、expert GEMM、token dispatcher 都值得研究。

5. **长上下文 + FP8**  
   context parallel、sequence parallel、attention backend 与低精度 activation 的结合。

6. **FP4/NVFP4/MXFP4 预训练**  
   关注 micro-block scaling、stochastic rounding、high-precision block encoding 等策略。

7. **低比特训练 scaling law**  
   比较 BF16、FP8、FP4、1-bit 模型在相同 token budget 下的 loss scaling 和 downstream 能力。

8. **跨硬件一致性**  
   同一模型 recipe 在 H100/B200、MI300X/MI350、Gaudi 2/3 上的稳定性和吞吐差异。

## 10. 小结：回到 Megatron MoE 实验

如果从一个 Megatron BF16 MoE baseline 出发，我不会把目标写成“尽快全量 FP8/FP4 化”。更合理的路线是：先保住 BF16 的收敛曲线和 resume 能力，再 profile 瓶颈在哪里；如果瓶颈确实在大 GEMM 或参数/激活带宽，再逐步打开 FP8 GEMM、FP8 param gather 或低精度通信。

MoE 这块尤其要克制。Expert FFN GEMM 是最值得先试低精度的地方；router/gate/top-k/load-balance loss 则不应该为了省一点计算就过早降到 FP8。router 的计算量本来不大，但它一旦改变 expert 选择，就会影响 expert load、all-to-all shape、token drop 和长期 specialization，这些后果比一次 GEMM 的量化误差更难 debug。

所以我最后会把低精度实验记成一张 checklist，而不是一个 dtype 表：改了哪些 Megatron flag；TE 是否真的生效；amax/scale 是否健康；all-to-all 和 GEMM 各占多少；loss spike 有没有变多；同等 validation loss 需要多少 token 和 wall-clock。只要这些问题没有一起回答，“FP8/FP4 比 BF16 快”就还只是局部结论。

对我来说，这篇文章最重要的结论是：BF16 是底线，FP8/FP4 是可以逐步引入的系统 recipe；MoE 里 expert 可以激进一点，router 要保守一点；最终比较对象不是单步 tokens/s，而是稳定训练到同等质量的总成本。

## 延伸阅读

### 基础与框架

- PyTorch AMP: <https://docs.pytorch.org/docs/main/amp.html>
- PyTorch AMP recipe: <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- DeepSpeed Training API: <https://deepspeed.readthedocs.io/en/latest/training.html>
- MS-AMP Introduction: <https://azure.github.io/MS-AMP/docs/introduction/>
- MS-AMP GitHub: <https://github.com/Azure/MS-AMP>

### NVIDIA

- NVIDIA Transformer Engine Documentation: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>
- NVIDIA Transformer Engine FP8/FP4 primer: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html>
- NVIDIA Transformer Engine GitHub: <https://github.com/NVIDIA/TransformerEngine>
- Megatron-LM GitHub: <https://github.com/NVIDIA/Megatron-LM>
- Megatron Bridge Mixed Precision: <https://docs.nvidia.com/nemo/megatron-bridge/latest/training/mixed-precision.html>
- H100 GPU: <https://www.nvidia.com/en-us/data-center/h100/>
- Blackwell Architecture: <https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/>
- NVFP4 TE documentation: <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html>

### Intel

- Intel Gaudi Transformer Engine FP8 Training: <https://docs.habana.ai/en/latest/PyTorch/PyTorch_FP8_Training/index.html>
- Intel Gaudi 3 AI Accelerator White Paper: <https://cdrdv2-public.intel.com/817486/gaudi-3-ai-accelerator-white-paper.pdf>
- Intel Neural Compressor Mixed Precision: <https://intel.github.io/neural-compressor/latest/docs/source/mixed_precision.html>
- Hugging Face Accelerate Intel Gaudi: <https://huggingface.co/docs/accelerate/usage_guides/gaudi>

### AMD

- ROCm TransformerEngine: <https://github.com/ROCm/TransformerEngine>
- AMD MI300X tuning guides: <https://rocm.docs.amd.com/en/docs-6.1.5/how-to/tuning-guides/mi300x/index.html>
- AMD Instinct MI300X: <https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html>

### 论文与模型

- FP8 Formats for Deep Learning: <https://arxiv.org/abs/2209.05433>
- DeepSeek-V3 Technical Report: <https://arxiv.org/pdf/2412.19437>
- Pretraining Large Language Models with NVFP4: <https://arxiv.org/abs/2509.25149>
- NVIDIA Nemotron 3 Ultra Technical Report: <https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf>
- NVIDIA Nemotron 3: Efficient and Open Intelligence: <https://arxiv.org/abs/2512.20856>

### 相关文章

- [2025-2026 开源 LLM 演进综述]({% post_url 2026-06-13-open-llm-survey %})

## 还可以继续展开的问题

- DeepSeek-V3 FP8 的 tile-wise/block-wise quantization 可以单独画图拆解。
- Megatron、DeepSpeed、FSDP 三套训练配置可以做一张横向对比表。
- H100/B200、MI300X/MI350、Gaudi 2/3 的硬件规格和 FP8/FP4 支持值得单独整理。
- FP4/NVFP4/MXFP4 预训练的公开实验仍在快速变化，需要持续跟进。
- 如果后续要真正跑实验，可以把 Megatron BF16 baseline 和 FP8 recipe 写成独立配置模板。
