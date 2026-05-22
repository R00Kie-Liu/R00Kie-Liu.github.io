---
layout: post
title: "LLM Infra 入门手册：How to Scale Your Model — 中文导读"
date: 2026-05-06
tags: [LLM, Infra, Scaling, 教程索引]
---

# LLM Infra 入门手册：How to Scale Your Model

> 基于 Google DeepMind《How to Scale Your Model》定制化中文导读  
> 面向背景：预训练研究员 | 熟悉 Scaling Law / 数据管线 / 模型架构  
> 工具栈：训练用 Megatron-LM，推理用 SGLang

---

## 学习路线图（14天）

### 第一周：硬件与通信基础（Day 1-5）

| Day | 章节 | 优先级 | 预计时间 |
|-----|------|--------|----------|
| 1-2 | [第1章：硬件基础 — GPU 与 TPU 是什么]({% post_url 2026-05-06-scaling-infra-ch01 %}) | ⭐⭐⭐ | 3h |
| 2 | [第2章：性能分析基石 — Roofline 模型]({% post_url 2026-05-06-scaling-infra-ch02 %}) | ⭐⭐⭐ | 2h |
| 3 | [第3章：内存层级与带宽 — 数据如何流动]({% post_url 2026-05-06-scaling-infra-ch03 %}) | ⭐⭐⭐ | 2h |
| 3-4 | [第4章：芯片互联与集群拓扑]({% post_url 2026-05-06-scaling-infra-ch04 %}) | ⭐⭐⭐ | 2.5h |
| 4 | [第5章：集合通信原语]({% post_url 2026-05-06-scaling-infra-ch05 %}) | ⭐⭐⭐ | 2h |
| 5 | [第6章：分片矩阵乘法 — 分布式计算的核心]({% post_url 2026-05-06-scaling-infra-ch06 %}) | ⭐⭐⭐ | 3h |

### 第二周：训练与推理（Day 6-14）

| Day | 章节 | 优先级 | 预计时间 |
|-----|------|--------|----------|
| 6 | [第7章：Transformer FLOPs/参数量/内存精确计算]({% post_url 2026-05-06-scaling-infra-ch07 %}) | ⭐⭐ | 2.5h |
| 7-8 | [第8章：训练并行策略 — DP/FSDP/TP/PP]({% post_url 2026-05-06-scaling-infra-ch08 %}) | ⭐⭐⭐ | 4h |
| 9 | [第9章：实战 — 训练 LLaMA 3 的分片决策]({% post_url 2026-05-06-scaling-infra-ch09 %}) | ⭐⭐ | 2h |
| 9-10 | [第10章：推理基础 — Prefill vs Generation]({% post_url 2026-05-06-scaling-infra-ch10 %}) | ⭐⭐⭐ | 3h |
| 10-11 | [第11章：推理优化 — KV Cache / Batching / 量化]({% post_url 2026-05-06-scaling-infra-ch11 %}) | ⭐⭐⭐ | 3h |
| 11-12 | [第12章：实战 — Serving LLaMA 3]({% post_url 2026-05-06-scaling-infra-ch12 %}) | ⭐⭐ | 2h |
| 13 | [第13章：性能调优 — Profiling 与调试]({% post_url 2026-05-06-scaling-infra-ch13 %}) | ⭐ | 2h |
| 14 | [第14章：JAX 并行编程入门（选读）]({% post_url 2026-05-06-scaling-infra-ch14 %}) | ⭐ | 2h |

---

## 图例说明

本手册中的标记含义：

> 📋 **背景知识** — 你可能不了解的前置概念

> 🔗 **与你的联系** — 将新概念与你已有的 CV/预训练经验关联

> 🛠️ **实践：Megatron** — 该知识点在 Megatron-LM 中的对应实现和使用技巧

> 🛠️ **实践：SGLang** — 该知识点在 SGLang 推理引擎中的对应实现

---

## 核心阅读建议

1. **如果只有5天**：重点读第1、2、8、10、11章
2. **如果对训练更感兴趣**：第1-6章 + 第8-9章
3. **如果对推理更感兴趣**：第1-2章 + 第10-12章
4. 每章末尾的"关键要点"可作为快速回顾的 checklist

---

## 快速参考

查看 [快速参考卡片]({% post_url 2026-05-06-scaling-infra-quick-reference %}) 获取：
- 关键数字速记（带宽层级、Roofline 临界值）
- 并行策略速查表
- Megatron/SGLang 配置模板
- 常见问题诊断清单
- 核心公式汇总

---

## 原书信息

- 书名：How to Scale Your Model — A Systems View of LLMs on TPUs
- 作者：Jacob Austin, Sholto Douglas, Roy Frostig 等 (Google DeepMind)
- 在线地址：[https://jax-ml.github.io/scaling-book/](https://jax-ml.github.io/scaling-book/)
