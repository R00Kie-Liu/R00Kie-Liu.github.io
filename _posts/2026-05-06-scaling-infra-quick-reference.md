---
layout: post
title: "LLM Infra 快速参考卡片"
date: 2026-05-06
tags: [Infra, Scaling Book]
---

# LLM Infra 快速参考卡片

> 本文是 [LLM Infra 入门手册]({% post_url 2026-05-06-scaling-infra-index %}) 的速查表，汇总了关键数字、公式和配置模板。

---

## 关键数字速记

### 硬件带宽层级（从快到慢）

| 链路 | 带宽量级 | 用途 |
|------|----------|------|
| VMEM/SMEM → 计算单元 | ~20-40 TB/s | 片上缓存 |
| HBM → 计算单元 | ~1-9 TB/s | 主内存 |
| NVLink（节点内） | ~900 GB/s | GPU 间高速互联 |
| ICI（TPU 芯片间） | ~90 GB/s/轴 | TPU 直连 |
| InfiniBand（节点间） | ~50 GB/s | 跨节点网络 |
| PCIe | ~50 GB/s | CPU-GPU |
| DCN | ~6 GB/s | 跨 Pod |

### Roofline 临界值

- **TPU v5e**：240 FLOPs/Byte
- **H100**：295 FLOPs/Byte
- **含义**：matmul 的 per-replica batch size 需要 > 这个值才 compute-bound

### 训练内存占用（每参数）

| 组件 | 字节数 |
|------|--------|
| 权重 (bf16) | 2 |
| 梯度 (bf16) | 2 |
| Adam 优化器 (fp32) | 8 |
| **总计** | **12** |

加上激活值：~16-20 bytes/参数

### 推理 KV Cache（LLaMA 70B）

- 每 token 每层：4 KB
- 序列长度 4096：1.34 GB/请求
- 256 并发：343 GB

---

## 并行策略速查

| 策略 | 内存节省 | 通信量 | 适用互联 | 何时使用 |
|------|---------|--------|---------|---------|
| **DP** | 无 | 2P | 任意 | 模型能放单卡 |
| **FSDP** | ÷N | 3P | 任意 | 节省内存 |
| **TP** | ÷N | 16BD/层 | 高带宽（节点内） | 模型太大 |
| **PP** | ÷N | 小 | 中等带宽 | 跨节点扩展 |

---

## Megatron 配置模板

```bash
# 3D 并行：TP=8, PP=4, DP=自动
--tensor-model-parallel-size 8
--pipeline-model-parallel-size 4

# 内存优化三件套
--use-distributed-optimizer
--recompute-activations
--sequence-parallel

# 通信优化
--overlap-grad-reduce
--overlap-param-gather

# Flash Attention
--use-flash-attn
```

---

## SGLang 配置模板

```bash
# 基础部署
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3-70B \
  --tp-size 8 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --max-running-requests 256

# FP8 量化
--quantization fp8 --tp-size 4
```

---

## 常见问题诊断

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| MFU < 30% | TP 跨节点 | TP ≤ 节点内 GPU 数 |
| OOM | 激活值太多 | `--recompute-activations` |
| 通信 >> 计算 | DP AllReduce 慢 | Gradient accumulation |
| Decode 延迟高 | Memory-bound | 增大 TP 或量化 |
| TTFT 高 | 长 prompt | `--chunked-prefill-size` |

---

## 公式速查

### 训练 FLOPs

$$C = 6 \times N \times P$$

N = 训练 token 数，P = 参数量

### 训练时间

$$T = \frac{C}{\text{GPU数} \times \text{FLOPs/s} \times \text{MFU}}$$

### Decode 延迟

$$T_{\text{decode}} = \frac{2P_{\text{bytes}}}{\text{HBM BW} \times N_{\text{gpu}}}$$

### Compute-bound 条件（TP）

$$D > \frac{2 \times TP \times \text{FLOPs/s}}{B_{\text{link}}}$$

---

## 学习路径建议

### 5 天速成（核心）

1. [第1章：硬件基础]({% post_url 2026-05-06-scaling-infra-ch01 %})
2. [第2章：Roofline]({% post_url 2026-05-06-scaling-infra-ch02 %})
3. [第8章：训练并行]({% post_url 2026-05-06-scaling-infra-ch08 %})
4. [第10章：推理基础]({% post_url 2026-05-06-scaling-infra-ch10 %})
5. [第11章：推理优化]({% post_url 2026-05-06-scaling-infra-ch11 %})

### 训练方向（7 天）

第1-6章 + 第8-9章

### 推理方向（7 天）

第1-2章 + 第10-12章

### 完整学习（13 天）

查看 [总览]({% post_url 2026-05-06-scaling-infra-index %}) 按顺序阅读
