---
layout: post
title: "Scaling Book 入门第 12 章：实战 — Serving LLaMA 3"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '推理', 'SGLang']
---

# Scaling Book 入门第 12 章：实战 — Serving LLaMA 3

> **本章目标**：将推理理论应用到 LLaMA 3-70B 的实际 serving 中，分析延迟/吞吐量权衡，给出 SGLang 的实际部署配置。
>
> **对应原书**：Chapter 8 (Serving LLaMA 3-70B on TPUs)  
> **优先级**：⭐⭐ 中 | **建议时间**：Day 11-12, 约 2 小时

---

## 12.1 LLaMA 3-70B 的推理内存分析

### 模型权重

| 精度 | 每卡（TP=8） | 总量 |
|------|-------------|------|
| bf16 | 17.5 GB | 140 GB |
| fp8 | 8.75 GB | 70 GB |
| int4 | 4.4 GB | 35 GB |

### KV Cache

每 token 每层：`4 × Kv_heads × K = 4 × 8 × 128 = 4096 bytes = 4 KB`

| 序列长度 | 每请求 KV cache | 256 并发 |
|----------|----------------|---------|
| 2048 | 0.67 GB | 171 GB |
| 4096 | 1.34 GB | 343 GB |
| 8192 | 2.68 GB | 686 GB |

**关键约束**：HBM 总量 = 权重 + KV cache + 其他开销

8 张 H100（每张 80 GB）= 640 GB 总 HBM：
- bf16 权重：140 GB
- 可用于 KV cache：~480 GB
- 序列长度 4096：最多 ~358 个并发请求

---

## 12.2 延迟分析

### Prefill 延迟（TTFT）

$$T_{\text{prefill}} = \frac{2 \times S_{\text{prompt}} \times P}{\text{FLOPs/s} \times N_{\text{gpu}}}$$

LLaMA 70B，prompt 2048 tokens，8 张 H100：

$$T_{\text{prefill}} = \frac{2 \times 2048 \times 70 \times 10^9}{9.9 \times 10^{14} \times 8} \approx 36 \text{ ms}$$

### Decode 延迟（TPOT）

$$T_{\text{decode}} = \frac{2P_{\text{bytes}}}{\text{HBM BW} \times N_{\text{gpu}}}$$

bf16，8 张 H100：

$$T_{\text{decode}} = \frac{140 \times 10^9}{3.35 \times 10^{12} \times 8} \approx 5.2 \text{ ms/token}$$

即约 **192 tokens/s**（单请求 decode 速度）。

### Batch 对延迟的影响

![Batch 对延迟的影响](/assets/scaling-book/img/batch-scaling-latency.png)

随着 batch size 增加：
- 每 token 的 decode 延迟**几乎不变**（直到 compute-bound 临界点）
- 因为权重只加载一次，多个请求共享

超过临界 batch size 后，延迟开始线性增长。

---

## 12.3 吞吐量分析

![Batch 对吞吐量的影响](/assets/scaling-book/img/batch-scaling-throughput.png)

| 阶段 | Batch 小 | Batch 大 |
|------|---------|---------|
| Memory-bound | 吞吐量 ∝ batch（线性增长） | — |
| 临界点 | batch ≈ FLOPs/s ÷ HBM BW ≈ 295 | — |
| Compute-bound | — | 吞吐量 ≈ 常数 |

理论最大吞吐量（decode）：

$$\text{Max throughput} = \frac{\text{FLOPs/s} \times N_{\text{gpu}}}{2P_{\text{flops}}}$$

8 张 H100，LLaMA 70B：

$$\text{Max throughput} = \frac{9.9 \times 10^{14} \times 8}{2 \times 70 \times 10^9} \approx 56,571 \text{ tokens/s}$$

但实际受 KV cache 内存限制，batch 到不了这么大。

---

## 12.4 延迟 vs 吞吐量的权衡

![延迟-成本权衡](/assets/scaling-book/img/latency-cost.png)

核心权衡：
- **更多 GPU**（更大 TP）→ 延迟更低，但成本更高
- **更大 batch**→ 吞吐量更高，但单请求延迟可能增加
- **量化**→ 同时降低延迟和成本，但可能影响质量

| 场景 | 优化目标 | 推荐配置 |
|------|---------|---------|
| 聊天机器人 | 低延迟 (TPOT < 30ms) | TP=8, 小 batch |
| Batch 处理 | 高吞吐 | TP=4, 大 batch |
| API 服务 | 平衡 | TP=8, 中等 batch + continuous batching |

---

## 12.5 SGLang 部署 LLaMA 3-70B

> 🛠️ **实践：SGLang**
>
> ### 基础部署
>
> ```bash
> # bf16，8 卡张量并行
> python -m sglang.launch_server \
>   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
>   --tp-size 8 \
>   --port 30000
> ```
>
> ### 优化部署
>
> ```bash
> python -m sglang.launch_server \
>   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
>   --tp-size 8 \
>   --port 30000 \
>   --mem-fraction-static 0.85 \        # 85% HBM 给权重+cache
>   --chunked-prefill-size 8192 \       # chunked prefill
>   --max-running-requests 256 \        # 最大并发
>   --context-length 8192               # 最大上下文长度
> ```
>
> ### FP8 量化部署（降低成本）
>
> ```bash
> python -m sglang.launch_server \
>   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
>   --tp-size 4 \                       # fp8 更小，4 卡够了
>   --quantization fp8 \
>   --port 30000
> ```
>
> ### 关键调优参数
>
> | 参数 | 说明 | 建议值 |
> |------|------|--------|
> | `--tp-size` | 张量并行度 | bf16: 8, fp8: 4, int4: 2 |
> | `--mem-fraction-static` | 权重内存比例 | 0.8-0.9 |
> | `--max-running-requests` | 最大并发数 | 取决于 KV cache 内存 |
> | `--chunked-prefill-size` | Prefill chunk 大小 | 4096-16384 |
> | `--context-length` | 最大上下文长度 | 按需设置 |
> | `--schedule-policy` | 调度策略 | "lpm" (longest prefix match) |
>
> ### 性能监控
>
> ```python
> # 查看 metrics
> import requests
> metrics = requests.get("http://localhost:30000/get_server_info").json()
> # 关注：
> # - cache_hit_rate：prefix cache 命中率
> # - num_running_reqs：当前并发数
> # - token_usage：KV cache 使用率
> ```
>
> ### 常见问题排查
>
> 1. **OOM**：减小 `--max-running-requests` 或增大 `--tp-size`
> 2. **TTFT 高**：检查是否有很长的 prompt，考虑增大 `--chunked-prefill-size`
> 3. **吞吐量低**：检查 batch utilization，可能需要增加并发请求
> 4. **Cache hit rate 低**：确认请求是否有共享前缀，调整 `--schedule-policy lpm`

---

## 关键要点

- [ ] LLaMA 70B bf16 需要 8 张 H100（权重 140 GB + KV cache）
- [ ] Decode 延迟 ≈ 模型大小 / (HBM 带宽 × GPU 数) ≈ 5ms/token @8xH100
- [ ] 吞吐量随 batch 线性增长直到 compute-bound 临界点（~295 tokens/batch）
- [ ] 量化（fp8/int4）可以减少 TP 需求并降低延迟
- [ ] SGLang 部署关键：`--tp-size` + `--mem-fraction-static` + `--chunked-prefill-size`

---

## 进一步阅读

- 原书 Chapter 8: Serving LLaMA 3-70B on TPUs
- [SGLang 官方文档](https://sglang.readthedocs.io/)
- [LLaMA 3 模型卡](https://huggingface.co/meta-llama/Meta-Llama-3-70B)

