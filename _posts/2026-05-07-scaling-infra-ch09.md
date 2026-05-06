---
layout: post
title: "Scaling Book 入门第 9 章：实战 — 训练 LLaMA 3 的分片决策"
date: 2026-05-07
tags: ['LLM', 'Infra', 'Scaling', '训练', 'Megatron']
---

# Scaling Book 入门第 9 章：实战 — 训练 LLaMA 3 的分片决策

> **本章目标**：将前几章的理论应用到具体模型（LLaMA 3）上，做端到端的训练配置推演和成本估算。
>
> **对应原书**：Chapter 6 (Training LLaMA 3 on TPUs)  
> **优先级**：⭐⭐ 中 | **建议时间**：Day 9, 约 2 小时

---

## 9.1 LLaMA 3 的模型规格

![LLaMA 配置](/assets/scaling-book/img/llama-json.png)

| 参数 | LLaMA 3-8B | LLaMA 3-70B | LLaMA 3-405B |
|------|-----------|------------|-------------|
| 层数 L | 32 | 80 | 126 |
| 隐藏维度 D | 4096 | 8192 | 16384 |
| FFN 维度 F | 14336 | 28672 | 53248 |
| 注意力头数 H | 32 | 64 | 128 |
| KV 头数 | 8 (GQA) | 8 (GQA) | 8 (GQA) |
| 每头维度 K | 128 | 128 | 128 |
| 词表 V | 128256 | 128256 | 128256 |
| 总参数量 | ~8B | ~70B | ~405B |

---

## 9.2 参数量验证

以 LLaMA 3-70B 为例：

**每层参数**：
- Attention：Wq[8192, 8192] + Wk[8192, 1024] + Wv[8192, 1024] + Wo[8192, 8192]
  = 8192×(8192 + 1024 + 1024 + 8192) = 8192 × 18432 ≈ 151M
- FFN（SwiGLU）：3 × [8192, 28672] = 3 × 234M ≈ 704M
- 每层总计：~855M

**总参数**：80 × 855M + embedding(128256 × 8192) ≈ 68.4B + 1.1B ≈ **69.5B** ✓

---

## 9.3 训练 FLOPs 和时间估算

**训练总 FLOPs**：

假设训练 15T tokens（LLaMA 3 论文的设置）：

$$C = 6 \times N \times P = 6 \times 15 \times 10^{12} \times 70 \times 10^9 = 6.3 \times 10^{24} \text{ FLOPs}$$

**训练时间估算**（假设使用 16K 张 H100，MFU=40%）：

$$T = \frac{C}{\text{GPU数} \times \text{FLOPs/s} \times \text{MFU}} = \frac{6.3 \times 10^{24}}{16000 \times 9.9 \times 10^{14} \times 0.4}$$

$$= \frac{6.3 \times 10^{24}}{6.34 \times 10^{18}} \approx 10^6 \text{ 秒} \approx 11.5 \text{ 天}$$

> 🔗 **与你的联系**
>
> 你做 Scaling Law 实验时应该很熟悉这种计算。关键的新知识是 **MFU（Model FLOPs Utilization）**：
> - 理想 MFU = 100%（所有硬件 FLOPs 都在做有用计算）
> - 实际 MFU 受通信、bubble、内存带宽限制
> - 好的训练配置可以达到 40-55% MFU
> - 差的配置可能只有 20-30%

---

## 9.4 分片策略推演

### 内存分析：能否用纯 DP？

70B 模型训练需要 ~840 GB（参数 140 + 梯度 140 + Adam 560）。

H100 有 80 GB HBM。每卡需要放下 840/N GB：
- N=16：52.5 GB/卡 → 可以放下，但没有空间给激活值
- 需要至少 N=16-32 张卡才能用纯 FSDP

### 推荐配置：128 张 H100

```
TP = 8（节点内 NVLink）
PP = 2（跨 2 个节点，每 stage 40 层）
DP = 8（128 / 8 / 2 = 8）
```

**内存估算**（每卡）：
- 参数：140 GB / 8 (TP) / 2 (PP) = 8.75 GB
- 优化器：560 GB / 8 (DP with dist-optimizer) = 70 GB → 太多
- 使用 distributed-optimizer：560 GB / 8 / 8 (TP) / 2 (PP) ≈ 4.4 GB

实际还需考虑激活值和 KV cache。

### 通信分析

![通信和计算时间](/assets/scaling-book/img/math-comms-time.png)

**TP 通信**：
- 每层 4 次 AllReduce，每次 `4 × batch_local × 8192` bytes
- 在 900 GB/s NVLink 上，批大小合理时可被计算掩盖

**PP 通信**：
- 每 micro-batch 传输 `bf16[micro_bs × seq_len, 8192]`
- 很小，通常不是瓶颈

**DP 通信**：
- 梯度 AllReduce 可以和反向计算重叠

---

## 9.5 MFU 计算

$$\text{MFU} = \frac{\text{实际训练 FLOPs/s}}{\text{硬件峰值 FLOPs/s}} = \frac{6BP / T_{\text{step}}}{N_{\text{gpu}} \times \text{Peak FLOPs/s}}$$

其中：
- B = 每步总 token 数（global batch size × seq_len）
- P = 模型参数量
- $T_{\text{step}}$ = 每步训练时间

> 🛠️ **实践：Megatron**
>
> **Megatron 的 MFU 监控**：
> - Megatron 在训练日志中会输出 `throughput (TFLOP/s/GPU)` 和 `MFU`
> - 也可以用 Weights & Biases 监控：`wandb.log({"mfu": mfu})`
>
> **LLaMA 3-70B 的 Megatron 配置参考**：
>
> ```bash
> # 模型参数
> --num-layers 80
> --hidden-size 8192
> --ffn-hidden-size 28672
> --num-attention-heads 64
> --group-query-attention
> --num-query-groups 8
> --seq-length 8192
> --max-position-embeddings 8192
>
> # 并行策略
> --tensor-model-parallel-size 8
> --pipeline-model-parallel-size 2
> # DP = world_size / (8 × 2)
>
> # 内存优化
> --use-distributed-optimizer
> --recompute-granularity selective
> --sequence-parallel
> --use-flash-attn
>
> # 训练超参
> --micro-batch-size 1
> --global-batch-size 1024
> --bf16
>
> # 通信优化
> --overlap-grad-reduce
> --overlap-param-gather
> ```
>
> **调优技巧**：
> 1. 先固定 TP=8，PP=1，看内存是否够用
> 2. 如果 OOM，增加 PP 或启用 recompute
> 3. 调 micro-batch-size 使 GPU 利用率最高
> 4. 监控 MFU，目标 > 40%
> 5. 如果 MFU 低，检查是否通信瓶颈（`nsys` profile）

---

## 9.6 成本估算

| 配置 | GPU 数 | 时间 | 成本（$2/GPU·h） |
|------|--------|------|-----------------|
| 16K H100, MFU=40% | 16,000 | ~12 天 | ~$9.2M |
| 8K H100, MFU=45% | 8,000 | ~21 天 | ~$8.1M |
| 2K H100, MFU=50% | 2,000 | ~73 天 | ~$7.0M |

**权衡**：更多 GPU → 更快但 MFU 可能更低（通信开销增加）→ 成本可能反而更高。

---

## 关键要点

- [ ] 实际训练配置是 TP/PP/DP 的组合，由内存和通信约束共同决定
- [ ] MFU 是衡量配置好坏的核心指标，目标 > 40%
- [ ] 训练时间 = 6NP / (GPU数 × Peak FLOPs/s × MFU)
- [ ] Megatron 配置的关键：先确定 TP（通常 8），再确定 PP，最后 DP 自动计算
- [ ] `--use-distributed-optimizer` + `--recompute-activations` + `--sequence-parallel` 是标准内存优化组合

---

## 进一步阅读

- 原书 Chapter 6: Training LLaMA 3 on TPUs
- [LLaMA 3 论文](https://arxiv.org/abs/2407.21783) 的训练细节章节
- [Megatron-LM GitHub 中的示例脚本](https://github.com/NVIDIA/Megatron-LM/tree/main/examples)

