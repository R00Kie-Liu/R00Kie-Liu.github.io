---
layout: post
title: "从 DFlash 到 dLLM Self-Spec（一）：算法收益如何落到 Serving 加速"
date: 2026-06-29
tags: [Infra, Speculative Decoding]
description: "从 DFlash 的 block draft、target verify、KV injection 和 SGLang Spec V2 出发，解释投机解码的算法收益如何在真实 serving 中兑现。"
---

## TL;DR

- DFlash 的加速不是简单来自“draft model 更小”，而是来自 block draft、target verify 摊薄、KV injection 和 serving 系统优化的叠加。
- target verify 可以并行，是因为 draft tokens 已经给出；但它仍然像短 prefill/extend 一样消耗 target model 计算，并不是免费。
- SGLang Spec V2 的核心价值是把 draft、verify、KV 写入、调度准备和 CPU copy 之间的等待压低，让算法收益在真实 serving 热路径里兑现。
- 高并发、低 acceptance、长上下文和显存紧张都会削弱 DFlash 收益；实际调参时要同时看 accepted length、verify latency、draft cost 和 KV headroom。

## 相关链接

- [LMSYS Blog: DFlash and Spec V2](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [DFlash paper](https://arxiv.org/abs/2602.06036)
- [Xiaomi MiMo-V2.5-Pro-UltraSpeed blog](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps)
- [SGLang MiMo-V2.5 deployment documentation](https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5)

## 第一章：工程优化概要

DFlash 类方法的工程直觉是：它不是“多跑一个 draft model 反而更慢”，而是用更便宜的 block draft 和一次性 target verify，把 target model 的逐 token 成本摊薄到多个输出 token 上。

### 1.1 DFlash 方法简述

普通 LLM decode 是逐 token 自回归的，每个输出 token 都要跑一次 target model。Speculative decoding 用 draft model 先猜一段 token，再让 target model 一次 verify 这段候选，最后提交被接受的前缀和一个 bonus token。

传统 EAGLE/MTP 类方法虽然减少了 target model 的逐 token forward，但 draft 侧常常仍然是逐 token 生成：`draft step 1 -> d1 -> draft step 2 -> d2 -> draft step 3 -> d3 ...`。

DFlash 的核心变化是把 draft 阶段改成 block 形式：

- 输入：上一轮 verified/bonus token + 若干 mask token。
- 输出：整个 block 的 hidden states。
- 然后通过 target `lm_head` 得到 block 内 draft tokens。

直观上：

| 方法 | draft 形态 |
| --- | --- |
| EAGLE/MTP 类 draft | `d1 -> d2 -> d3 -> d4` |
| DFlash draft | `[d1, d2, d3, d4]` in one block |

这里 target verify 之所以可以并行，是因为 draft tokens 已经给出来了。target model 不需要自己逐步生成 `d1, d2, d3...`，而是把它们当成一段已知输入，像 prefill/teacher forcing 一样一次 forward 得到每个位置的 next-token logits：

| 已知输入 | target 一次 forward 得到 | verify 比较 |
| --- | --- | --- |
| `context` | `p1 = P(next | context)` | `d1 == argmax(p1)?` |
| `context, d1` | `p2 = P(next | context, d1)` | `d2 == argmax(p2)?` |
| `context, d1, d2` | `p3 = P(next | context, d1, d2)` | `d3 == argmax(p3)?` |
| `context, d1, d2, d3` | `p4 = P(next | context, d1, d2, d3)` | `d4 == argmax(p4)?` |

所以 verify 的“并行”不是取消因果依赖，而是利用因果 mask 在一次批量 forward 里同时评估多个已知位置。

从时间成本上看，target verify 更像一次很短的 prefill/extend，而不是普通逐 token decode：

| 阶段 | 计算形态 |
| --- | --- |
| 普通 decode | 每个请求每轮通常只算 1 个新 token。 |
| target verify | 每个请求每轮算 `block_size` 个已知 draft tokens；prefix KV cache 已经存在，只对 draft block 做一次短 extend-style forward。 |

因此它通常比连续跑 `block_size` 次 target decode 更便宜，但不是免费。`block_size` 越大，verify token 数、attention/MLP 计算和 KV 写入都会增加。Speculative decoding 能赚，是因为：

- 一次 `target verify(block_size tokens)` 的目标是显著低于连续 `block_size` 次普通 target decode。
- accepted tokens 足够多。
- draft cost、KV injection cost、scheduler cost 足够低。

换句话说，verify 的并行性解决的是“串行深度”问题，不是消除计算量。target model 仍然要对 `batch_size * block_size` 个位置做完整 forward。它更像短 prefill/extend：prefix KV 已经存在，不用重算 prompt，但 attention 仍要读取已有 KV，MLP/logits/KV write 仍要对 draft block 内每个位置执行。

因此 verify 也可能成为瓶颈。典型情况包括：

- `block_size` 过大，单轮 verify token 数太多。
- acceptance length 低，verify 了整块 token，却只提交很少 token。
- 高并发 batch 已经很大，`batch_size * block_size` 变成一次很重的 target extend。
- 上下文很长，verify 虽然只处理短 block，但每个位置仍要 attend 到很长的 prefix KV。
- target model 本身很重，或者 verify 后还要做 top-k/top-p、penalty、logprob、rejection sampling 等后处理。

所以 DFlash 的关键不是“verify 并行所以免费”，而是“verify 成本能被足够多 accepted tokens 摊薄”。

DFlash 的第二个关键点是 KV injection：

KV injection 的路径是：`target hidden states -> draft_model.project_target_hidden(...) -> draft 每层 KV projection / norm / RoPE -> 写入 draft KV cache`。

这让 draft model 不必从零理解上下文，而是复用 target model 已经计算出的上下文表示。

因此 DFlash 可以简化理解为：

- **block diffusion draft**：降低 draft 成本。
- **KV injection**：提高 draft token 被 target 接受的概率。
- **target verify**：一次验证多个候选 token，摊薄大模型 decode 成本。

![DFlash architecture](/assets/dflash-self-spec/dflash-arch-diagram.webp)

图中最值得关注的是 target model hidden states 到 draft model KV cache 的注入路径：draft model 不是独立重建上下文，而是利用 target model 已经形成的上下文表示，专注于预测下一段 block tokens。

### 1.2 加速收益概览

DFlash 的实际加速不是单纯来自“draft model 比 target model 小”，而是来自下面几类收益叠加：

1. target model 的逐 token decode 被改成 block verify，昂贵的大模型 forward 成本被多个 token 摊薄。
2. DFlash 的 draft 侧使用 block diffusion 风格，一次生成一个 token block，避免 EAGLE/MTP 类方法中 draft 侧仍然逐 token 自回归的问题。
3. KV injection 让 draft model 复用 target model 的上下文表示，提高 draft token 的接受长度。
4. SGLang Spec V2 用 overlap scheduling、FutureMap、KV over-allocation、copy stream 等机制，减少 host/device 同步和 scheduler 开销。
5. DFlash 实现中还有针对热路径的 Triton kernel、fused KV materialization、CUDA Graph 等工程优化，降低每轮 speculative step 的固定成本。

一个粗略的性能模型是：

> speedup ~= baseline target decode per-token cost * average committed tokens per round / speculative round total cost

其中一轮 speculative 的成本主要包括：DFlash draft block、target verify block、accept/bonus 计算、target hidden 到 draft KV 的 materialization，以及 scheduler、KV allocation、D2H、同步等系统开销。

所以 DFlash 是否快，关键看五件事：

- 每轮能提交多少 token。
- draft block 有多便宜。
- target verify 比逐 token target decode 便宜多少。
- KV injection/materialization 成本能否被接受长度摊薄。
- Spec V2 是否把 CPU 调度和 GPU forward 重叠起来。

### 1.3 DFlash 的加速来源分层

#### 1.3.1 一级收益：决定上限

这些决定 DFlash 能不能有大幅 speedup：

| 来源 | 机制 | 影响 |
| --- | --- | --- |
| target verify 摊薄 | 一次 target forward 验证 block | 降低大模型 per-token 成本 |
| block draft | draft 侧一次 forward 产生 block | 避免 draft 自回归 loop |
| KV injection | target hidden 写入 draft KV | 提高 acceptance length |

如果 acceptance length 很低，后面的系统优化只能止损，很难产生大幅加速。

#### 1.3.2 二级收益：决定能否在 serving 中兑现

| 来源 | 机制 | 影响 |
| --- | --- | --- |
| overlap scheduling | CPU 准备和 GPU forward 重叠 | 减少 host overhead |
| FutureMap | GPU buffer 跨 iteration relay | 减少同步和数据搬运 |
| copy stream | D2H result copy 与下一轮 forward 重叠 | 降低输出处理阻塞 |
| KV over-allocation | 提前预留 block KV slots | 避免热路径 allocator 抖动 |
| CPU seq_lens 滞后 | DFlash 特化跳过部分 D2H | 减少 per-step sync |

这些二级收益本质上不是改变 DFlash 的数学算法，而是让“算法上省下来的 target decode”不要被 serving 系统里的同步、调度、拷贝和内存管理吃掉。DFlash 类投机解码每轮不只是一次 target decode，它通常包含 draft block、target verify、accept/bonus 计算、target hidden 写入 draft KV、下一轮 draft input 准备等多个阶段。因此，只要其中某个阶段强制 CPU 等 GPU，或者强制下一轮等上一轮所有尾部工作结束，理论 speedup 就会被明显打折。

**overlap scheduling**

普通串行路径可以理解为：GPU forward 完成后，CPU 才处理结果、更新请求状态、规划 KV、构造下一轮 batch，然后再启动下一轮 GPU forward。DFlash 每轮的尾部工作更多，如果完全串行，GPU 很容易在两轮之间等 CPU。

Spec V2 的 overlap scheduling 试图把这些阶段拆开：当前轮还在 forward stream 做尾部工作时，scheduler 已经可以利用提前发布的状态准备下一轮；结果拷贝也可以放到 copy stream 上。这对 DFlash 很重要，因为 DFlash 的收益来自“少跑多次 target decode”，但它同时引入了更多“每轮状态协调工作”。overlap 的作用就是把这些协调工作尽量藏到 GPU 计算背后。

这里的“每轮状态协调工作”指的是每一轮 speculative step 之间，为了让 draft、target verify、KV cache 和 scheduler 状态保持一致而必须做的额外操作。普通 decode 一轮通常是 target model 生成 1 个 token，然后更新请求长度和 KV 状态；DFlash 一轮则要经历 draft block、target verify、accept/bonus 计算、target hidden 写入 draft KV、`new_seq_lens` 发布、下一轮 `DFlashDraftInputV2` 构造、batch/KV 重新规划等步骤。这些步骤不一定计算量很大，但很容易制造 GPU/CPU 同步、小 tensor gather/scatter、KV slot 分配、`req_to_token` 更新和结果 D2H copy。换句话说，DFlash 是“少跑大模型”，但代价是“每轮要组织更多中间状态”。

**FutureMap**

投机解码的下一轮输入依赖上一轮结果，例如 `new_seq_lens`、`verified_id`、accepted tokens、采样相关 buffer。如果每轮都把这些数据同步回 CPU，再由 CPU 重新组织后传回 GPU，就会产生 D2H/H2D 往返和同步点。

FutureMap 的价值是让跨 iteration 的状态继续留在 GPU buffer 里，通过 request slot 或 future index 做 relay。对 DFlash 来说，这尤其重要，因为下一轮 draft block 的第一个 token 就是上一轮 verify 得到的 bonus/verified token，下一轮长度也依赖上一轮 `commit_lens`。这些状态越少经过 CPU，decode 热路径越不容易被小同步卡住。

**copy stream**

serving 不是只要 GPU 算出 logits 就结束，还要把 token id、accept length、finish reason、logprob 等结果搬到 CPU，交给 scheduler 和上层请求处理。普通 decode 每轮输出少，copy 成本可能不显眼；DFlash 每轮可能提交多个 token，同时还要处理 speculative 的 `accept_lens` 和输出 token block。

如果结果 D2H copy 占住主 stream，下一轮 forward 就要等它。copy stream 的意义是把“给 CPU 看结果”这件事和“GPU 继续跑下一轮”分离。只要依赖关系允许，主 forward stream 可以更早启动下一轮，CPU 可见结果稍后到达。这类优化不会提升单个 kernel 的速度，但能降低端到端 serving 的空泡。

**KV over-allocation**

DFlash 每轮要验证一个固定 block，所以它不能只给“当前真正已提交的 token”分配 KV slot，还要为即将写入的 verify block 准备位置。若每轮都临时分配 KV、更新 `req_to_token`、处理 page allocator，再遇到请求过滤、合并或释放，allocator 和元数据维护会变成热路径上的固定成本。

KV over-allocation 的做法是提前把未来一到两个 block 的 KV 空间预留好，让 worker 在 draft/verify 时可以直接从 `req_to_token` 找到 cache loc。它用更多显存换更平滑的调度路径。对 DFlash 来说这很划算，因为 block verify 的形状固定，提前预留可以减少每轮 allocator 抖动；但它也解释了为什么 DFlash 会更吃显存余量。

**CPU seq_lens 滞后**

`seq_lens` 是每个请求当前长度。GPU 上的 attention 和 verify 需要它，CPU scheduler 有时也需要 host-side `seq_lens_cpu` 做规划。如果每轮 accept/bonus 之后都立刻把新长度从 GPU 拷回 CPU，就会形成一个很硬的同步点：CPU 必须等 GPU 算完，GPU 下一轮又可能等 CPU 规划完。

DFlash Spec V2 的处理方式是让 GPU 侧 `new_seq_lens` 及时推进，而 CPU 侧长度允许滞后一轮；下一轮规划需要的安全上界由 `planning_seq_lens_cpu` 和 `reserved_seq_lens_cpu` 承担。这样做的关键前提是：DFlash 每轮最多推进一个固定 block，scheduler 可以用保守上界规划 KV 和 attention metadata。它减少了每轮 D2H 同步，是 DFlash 能在连续 decode 中保持吞吐的重要细节。

#### 1.3.3 三级收益：减少固定成本

| 来源 | 机制 | 影响 |
| --- | --- | --- |
| dflash_prepare_block Triton kernel | 一次生成 block ids、positions、cache loc | 减少小 tensor op |
| dflash_accept_bonus Triton kernel | 一次计算 accept、bonus、new seq lens | 减少每轮后处理成本 |
| fused_kv_materialize | 跨层 KV projection + norm + RoPE 融合 | 降低 KV injection 成本 |
| CUDA Graph | 固定 block shape 更易 replay | 降低 kernel launch/metadata 成本 |
| hidden_states 不回 CPU | verify 后清空 hidden_states | 避免大 tensor CPU copy |

#### 1.3.4 Serving 部署中的关键瓶颈地图

如果已经熟悉 DFlash / block diffusion，本系列最值得关注的不是“为什么能 draft 出一段 token”，而是这些 token 进入真实 serving engine 后会卡在哪里。

| 瓶颈 | 为什么在 DFlash 类方法里更明显 | 常见优化思路 | 需要观察的指标 |
| --- | --- | --- | --- |
| target verify capacity | 每轮不再只处理 `batch_size` 个 decode token，而是处理 `batch_size * verify_length` 个 verify token。 | 控制 block size / verify length；让低价值 suffix 少进 target；高并发下做动态 verify budget。 | target verify latency、verify token 数、accepted tokens / verified tokens。 |
| draft path latency | 多跑一个 draft path，收益必须靠更高 accepted length 摊回来。 | draft model 轻量化；固定 block shape；CUDA Graph；避免 draft 自回归 loop。 | draft block latency、draft / target 时间占比。 |
| KV materialization | target hidden 要写入 draft KV，跨层 projection / norm / RoPE 容易变成一串小 kernel。 | fused KV materialization；减少 Python loop；避免 hidden states 回 CPU。 | KV materialization latency、kernel launch 数、显存带宽。 |
| scheduler / CPU-GPU sync | 每轮 speculative step 需要维护 `verified_id`、`new_seq_lens`、accepted tokens、KV slots。 | `on_publish`、FutureMap、CPU `seq_lens` 滞后、overlap scheduling。 | GPU idle gap、D2H/H2D 次数、scheduler step latency。 |
| allocator / KV pool pressure | DFlash 需要 draft KV、target KV、未来 block 预留和中间 buffer。 | KV over-allocation、稳定 `req_to_token`、预留 headroom、控制 `max_running_requests`。 | KV pool available size、OOM / eviction、可跑 batch size。 |
| result copy / postprocess | 每轮可能提交多个 token，还要处理 accept length、finish reason、logprob 等 metadata。 | copy stream；只拷 CPU 必需结果；把后处理和下一轮 forward overlap。 | copy_to_cpu latency、output queue delay。 |
| CUDA Graph shape stability | block verify 形状相对固定，但请求过滤、变长 verify、动态 batch 都会破坏 capture/replay 条件。 | 固定关键 shape；bucket；区分固定 block DFlash 和变长 DSpark 的 runtime 路径。 | graph replay 命中率、fallback eager 次数。 |

所以实际部署时，不能只问“平均 acceptance length 多高”。更有用的问题是：每轮 target verify 花了多少 target capacity，draft/KV/scheduler/copy 这些附加成本有没有被 overlap 掉，以及显存余量是否允许更大的 running batch。

### 1.4 为什么 DFlash 相比 EAGLE/MTP 更容易在大模型上赢

从工程角度看，DFlash 更容易在大模型上赢，是因为它同时压低了 draft 成本、提高了 acceptance length，并让 Spec V2 有机会把 scheduler/KV/copy 开销藏到 overlap 里。

简化成一句话：EAGLE/MTP 常受 draft 自回归成本限制；DFlash 用 block draft 控制 draft 成本，用 KV injection 保住接受率。

### 1.5 什么时候 DFlash 可能不快

DFlash 并不保证所有场景都快。常见不利条件：

#### 1.5.1 Acceptance length 低

如果每轮 block size 是 16，但平均只接受 1 个 draft token，那么：

`draft cost + target verify cost + KV materialization cost` 都会变成额外负担。

这通常发生在：

- draft model 与 target/domain 不匹配。
- temperature/sampling 较激进。
- 任务分布变化大。
- 上下文中约束很强。

#### 1.5.2 输出很短

如果请求只输出几个 token，prefill、draft KV 初始化、CUDA Graph warmup、调度成本还没有摊开，请求就结束了。

DFlash 更适合长输出、代码生成、推理生成、多 token 连续文本。

#### 1.5.3 baseline batch 已经很满

高并发下，普通 decode 的 GPU 利用率本来就更高。Speculative decoding 的边际收益会下降，因为 target decode 的小 batch 问题被 batching 缓解了。

低并发或单请求时，普通 decode 往往是一个请求每轮只生成 1 个 token，GPU kernel 很难吃满。DFlash 把多轮小 decode 合并成 draft block + target verify，减少 target forward 次数，因此收益更明显。

但高并发时，普通 decode 已经可以把很多请求合成一个较大的 decode batch，例如 100 个请求每轮一起跑 target forward。此时 baseline 的 per-token 成本已经被 batching 降低，DFlash 再减少 target decode 轮数的边际收益会变小。同时，DFlash 自己还要付出 draft model、target verify block、accept/bonus、KV injection、`req_to_token` 更新、额外 KV cache 和调度协调成本。

所以投机解码在高并发下不是完全没用，而是更依赖几个条件：

- acceptance length 足够高。
- draft block 和 verify block 足够便宜。
- overlap scheduling 能把 CPU 调度、KV 准备和 D2H copy 藏起来。
- 显存还能支撑 draft KV、over-allocation 和较大的 running batch。
- workload 是代码、推理、结构化自然文本这类高接受率场景，而不是随机 token 或高度发散的采样。

#### 1.5.4 显存更紧

DFlash 需要额外容纳 target model weights、draft model weights、target KV、draft KV、over-allocated KV headroom、fused materialization workspace。

DFlash 的内存压力主要来自两类：一类是“多一套东西”，例如 draft model weights 和 draft KV cache；另一类是“提前留位置”，例如 Spec V2 为未来 DFlash block 预留的 KV slots。除此之外，每轮 verify 还会产生 draft tokens、positions、cache loc、verify hidden states、accept/bonus 等中间 buffer；如果启用 CUDA Graph，也会为固定 shape 保留可复用 buffer。

显存压力可能降低可跑的 `max_running_requests` 或可用 context length。

#### 1.5.5 功能限制

当前 DFlash 实现对一些功能有限制，例如：

- `return_logprob`
- overlap 下的 `return_hidden_states`
- grammar-constrained decoding
- DP attention
- pipeline parallel size > 1

这些限制来自 DFlash verify、KV injection、overlap scheduling 对状态一致性的要求。

### 1.6 实际调参时该观察什么

#### 1.6.1 Acceptance length

最重要的是 `average accepted drafts per round`，以及 `commit_lens = accept_len + 1`。

如果 acceptance length 低，先不要盲目调 kernel/backend，应该优先检查：

- draft model 是否匹配 target。
- block size 是否过大。
- 采样参数是否太激进。
- 数据域是否偏离训练分布。

#### 1.6.2 Block size

block size 越大：

| 变化 | 影响 |
| --- | --- |
| 潜在每轮提交 token 越多 | 上限更高 |
| target verify block 越大 | verify 成本更高 |
| draft block 成本越大 | draft 侧开销更高 |
| KV over-allocation 越大 | 显存压力更高 |
| acceptance tail 越难保持 | 后半段 draft 更容易被拒 |

经验上可以扫 `block_size = 4, 8, 16`。

不要只看 acceptance length，也要看 end-to-end throughput 和 target verify latency。一个较大的 block size 可能提高平均接受长度，但如果 verify block 本身变成瓶颈，端到端吞吐仍然可能下降。

#### 1.6.3 Draft attention backend

DFlash draft worker 支持的 attention backend 有自己的限制。SGLang 当前实现会避免 `trtllm_mha` 作为 draft backend，因为 DFlash draft path 需要 per-layer DFlash attention。常见 fallback 是 CUDA 上的 `flashinfer`、ROCm 上的 `triton`，也可能使用 `fa3` / `fa4`。

target backend 和 draft backend 可以不同。实际性能要分别看 `target verify latency`、`draft block latency`、`KV materialization latency`。

#### 1.6.4 是否启用 overlap plan stream

DFlash 的 `DFlashDraftInputV2.prepare_for_decode` 支持额外 plan stream：

**配置片段：overlap plan stream 开关**

```bash
SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
```

这个开关让部分 prepare / KV allocation 相关 work 放到 plan stream 上，与主 forward stream 更好地配合。但它也要求事件和共享 tensor 写入顺序正确，所以实现里有 `verify_done` event 和 stream wait。

#### 1.6.5 显存余量

DFlash 不是只省计算。它会增加内存压力。需要观察：

- KV pool available size
- `max_running_requests`
- `mem_fraction_static`
- draft KV layers
- over-allocation headroom
- CUDA Graph capture 后剩余显存

如果这些指标已经很紧，DFlash 的吞吐收益可能会被更小的 batch、更短的可用 context 或更频繁的 KV 回收抵消。

### 1.7 落地案例与后续形态

DFlash / Spec V2 的主干机制落到真实系统后，瓶颈会继续向量化、runtime codesign、verify budget 和模型内 self-spec 迁移。MiMo、DSpark、I-DLM / Nemotron 分别代表几类典型方向。

#### 1.7.1 MiMo-V2.5-Pro-FP4-DFlash：DFlash + 量化 + runtime codesign 案例

小米 MiMo-V2.5-Pro-UltraSpeed 可以看作 DFlash 类方法在 1T MoE 模型上的工程化案例。它的重点不是单靠 DFlash 一个点跑快，而是把模型量化、draft/verify 算法和推理运行时一起做 codesign。

| 层次 | 做法 | 对推理端的意义 |
| --- | --- | --- |
| target model | MoE experts 使用 FP4/MXFP4 量化，其他模块保留更高精度 | 降低权重体积和显存带宽压力，尤其适合 experts 占参数大头的 MoE。 |
| draft/verify | DFlash block-level masked parallel prediction | draft 一次填充一个 block，减少 draft 自回归开销；target 一次 verify 多个候选。 |
| draft model 设计 | BF16 DFlash draft generator，SWA-only，block size 8 | draft 成本更接近常数，不随完整 context 线性增长；block size 8 用来平衡 acceptance length、verify 成本和并发。 |
| 系统运行时 | TileRT persistent engine kernel、warp specialization、定制 kernel/编译 | 在 1000 TPS 级别，operator launch、同步和 global memory round-trip 会变成显性空泡，运行时需要尽量消除这些 operator boundary gaps。 |

小米博客给出的平均 acceptance length 很有参考价值：coding 约 6.30，math/reasoning 约 5.56，agent 约 4.29；同时 block size 限制为 8。这个选择说明 DFlash 落地时必须同时看接受长度、verify 成本和并发，而不是单纯把 block 拉长。Hugging Face model card 更适合作为 checkpoint 结构和配置线索，不能直接等同于线上 1000 TPS 的完整部署 recipe。

#### 1.7.2 DeepSeek-V4-Pro-DSpark：DFlash 之后的负载感知投机解码

DeepSeek 的 `deepseek-ai/DeepSeek-V4-Pro-DSpark` 不是一个新 base model，而是同一个 DeepSeek-V4-Pro checkpoint 加上额外 speculative decoding module。它更关注 DFlash 之后的两个线上痛点：

| DFlash 痛点 | DSpark 的处理 |
| --- | --- |
| block 后半段 token 缺少足够的 token 间依赖，acceptance tail 容易下降。 | 保留 DFlash 并行 backbone，但增加 Markov/RNN 这类轻量 sequential head，让 draft token 采样左到右带上前一个 token 的影响。 |
| 固定长度 target verify 在高并发下容易浪费 batch capacity。 | confidence head 预测 prefix survival probability，hardware-aware scheduler 根据负载动态决定每个请求 verify 多长。 |

公开配置里的 `dspark_block_size = 5`、`dspark_target_layer_ids = [58, 59, 60]`、`dspark_markov_rank = 512`，说明它延续了“target hidden feature + draft module”的路线，但把 verify budget 变成了调度资源。

#### 1.7.3 模型内 self-spec：下一类落地优化形态

dLLM / AR unified model 代表另一种落地形态：同一个模型同时支持 block diffusion draft 和 AR verify。serving 问题从“如何绑定轻 draft model 和大 target model”，变成“如何让同一个模型在不同 forward phase 里扮演 draft / verify 两种角色”。

| 代表工作 | self-spec 形态 | 对 serving 的新问题 |
| --- | --- | --- |
| I-DLM / ISD | 一个 forward 同时 verify 上一轮 speculative tokens，并从 MASK 位置生成下一轮 draft tokens。 | SGLang 需要维护 per-request pending/spec 状态、变长 advance、KV trim；verify 和 sample 尽量合并成一批 GPU work。 |
| Nemotron tri-mode | 同一个模型通过切换 attention pattern 支持 AR、diffusion decoding 和 linear self-spec。 | runtime 需要处理 causal / bidirectional attention 切换、shared KV cache、可选 draft LoRA、CUDA Graph phase hooks 和按并发选择 block size。 |

I-DLM 和 Nemotron tri-mode 的共同点是把 self-spec 下沉到模型结构和 runtime 边界：attention metadata、KV pool、graph capture、LoRA/weight view、scheduler state 都会进入热路径。

## 结语

DFlash 的算法收益可以概括为：便宜地提出一批 token，并让 target 一次性验证。

SGLang Spec V2 的系统收益可以概括为：把 draft、verify、KV 写入、scheduler 准备、CPU copy 之间的等待压到最低。

两者合起来，才是 DFlash 在实际 serving 中能跑出高 throughput 的原因。前者决定理论上限，后者决定这个上限能不能在真实 serving 热路径里兑现。

## 术语速查

<details markdown="1">
<summary>展开术语表</summary>

| 术语 | 简单解释 |
| --- | --- |
| prefill | 处理 prompt 的阶段。模型一次性读入用户输入，生成第一批 KV cache 和第一个输出 token。 |
| extend | 在已有 prefix KV cache 的基础上，批量处理一小段新增 token；可以理解为“接在 prefix 后面的短 prefill”。 |
| decode | 逐步生成输出 token 的阶段。每一步通常基于已有 KV cache 生成下一个 token。 |
| target model | 真正要服务的主模型。输出质量以它为准。 |
| draft model | speculative decoding 中用来提前猜 token 的小模型或辅助模型。 |
| target verify | target model 对 draft tokens 做批量验证，判断哪些 draft token 可以接受。 |
| bonus token | draft token 第一次不匹配时，target model 在该位置给出的 token；通常会和已接受前缀一起提交。 |
| accept length | 一轮 speculative decoding 中被 target model 接受的 draft token 数。 |
| block size | DFlash 每轮 draft/verify 的固定 token 窗口大小，例如 8 或 16。 |
| hidden states | 模型中间层产生的向量表示。DFlash 会用 target hidden states 帮 draft model 理解上下文。 |
| KV cache | attention 中历史 token 的 Key/Value 缓存。decode 时复用它，避免每步重算完整上下文。 |
| KV injection | DFlash 把 target hidden states 转成 draft model 的 KV cache，让 draft model 复用 target 的上下文表示。 |
| req_to_token | SGLang 中从请求 slot 到 token/KV slot 的映射表，用来找到每个请求的 KV cache 位置。 |
| token_to_kv_pool | 存放实际 KV cache 的内存池。`req_to_token` 记录“请求的第几个 token 对应池里的哪个位置”。 |
| radix cache / prefix cache | 复用相同 prompt 前缀的 KV cache，避免重复 prefill。 |
| FutureMap | Spec V2 overlap 中跨 iteration 传递 GPU 数据的结构，例如 new seq lens、verified id、next draft input。 |
| on_publish | worker 在 forward 还没完全结束时提前发布 `new_seq_lens`，让 scheduler 可以开始准备下一轮。 |
| overlap schedule | 把 GPU forward、CPU 调度、KV 准备、CPU copy 尽量重叠起来，减少串行等待。 |
| CUDA Graph | 把固定形状的 GPU 执行流程录下来反复 replay，减少 kernel launch 和调度开销。 |
| Triton kernel | 用 Triton 写的自定义 GPU kernel，常用于融合小操作、减少 Python/Torch eager 开销。 |
| D2H / H2D | Device to Host / Host to Device，即 GPU 到 CPU、CPU 到 GPU 的数据拷贝。 |
| TP | Tensor Parallel，把模型权重切到多个 GPU 上并行计算。 |
| attention backend | SGLang 中 attention 计算的具体实现后端，例如 FlashInfer、Triton、fa3/fa4、TRTLLM MHA。 |

</details>

## 参考资料

1. LMSYS Blog: The next generation of speculative decoding: DFlash and Spec V2
   https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/

2. DFlash paper
   https://arxiv.org/abs/2602.06036

3. Xiaomi MiMo-V2.5-Pro-UltraSpeed blog
   https://mimo.xiaomi.com/blog/mimo-tilert-1000tps

4. XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash Hugging Face model card
   https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash

5. TileRT: Two Leaps to 1000 Tokens/s on a 1T-Parameter Model
   https://www.tilert.ai/blog/breaking-1000-tps.html

6. SGLang MiMo-V2.5 deployment documentation
   https://docs.sglang.io/cookbook/autoregressive/Xiaomi/MiMo-V2.5

7. DeepSeek-V4-Pro-DSpark Hugging Face model card
   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark

8. DeepSpec GitHub repository
   https://github.com/deepseek-ai/DeepSpec

9. DSpark paper
   https://arxiv.org/abs/2606.19348

10. SGLang main branch source files
   https://github.com/sgl-project/sglang

11. Key SGLang implementation paths
   `python/sglang/srt/speculative/dflash_worker_v2.py`
   `python/sglang/srt/speculative/dflash_info_v2.py`
   `python/sglang/srt/speculative/dflash_info.py`
   `python/sglang/srt/speculative/dflash_utils.py`
   `python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py`
   `python/sglang/srt/speculative/triton_ops/dflash_accept_bonus.py`
   `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
   `python/sglang/srt/managers/scheduler.py`
   `python/sglang/srt/managers/overlap_utils.py`

12. I-DLM: Introspective Diffusion Language Models
    https://arxiv.org/abs/2604.11035

13. Introspective-Diffusion/I-DLM GitHub repository
    https://github.com/Introspective-Diffusion/I-DLM

14. NVlabs/Nemotron-Labs-Diffusion GitHub repository
    https://github.com/NVlabs/Nemotron-Labs-Diffusion

15. SGLang DLLM onboarding issue
    https://github.com/sgl-project/sglang/issues/25802
