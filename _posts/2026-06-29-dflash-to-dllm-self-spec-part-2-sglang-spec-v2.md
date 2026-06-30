---
layout: post
title: "从 DFlash 到 dLLM Self-Spec（二）：SGLang Spec V2 实现剖析"
date: 2026-06-29
tags: [Infra, Speculative Decoding, SGLang]
description: "结合 SGLang Spec V2 源码路径，拆解 DFlash decode、target verify、KV injection、overlap、FutureMap 和 KV over-allocation 如何协同。"
---

## TL;DR

- SGLang Spec V2 的重点不是改写 DFlash 的数学规则，而是减少 speculative decoding 每轮之间的同步、拷贝、调度和 KV 管理空泡。
- 一轮 DFlash decode 大致包含 prepare block、draft block forward、target verify、accept/bonus、target hidden 写入 draft KV、下一轮输入准备。
- `FutureMap`、`on_publish`、copy stream 和 CPU `seq_lens` 滞后一轮，都是为了避免每轮都让 CPU 和 GPU 硬同步。
- KV over-allocation 用更多显存换更稳定的热路径，适合 DFlash 这种 fixed-block verify 形态，但会提高显存余量要求。

## 相关链接

- [LMSYS Blog: DFlash and Spec V2](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [DFlash paper](https://arxiv.org/abs/2602.06036)
- [SGLang speculative runtime source](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/speculative)
- [SGLang DFlashWorkerV2 implementation](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/dflash_worker_v2.py)

## 第二章：结合源码剖析推理优化流程

SGLang Spec V2 的执行路径把 DFlash 的算法收益落到 runtime 热路径上。重点不是逐行复刻源码，而是解释这些结构为什么存在、它们在哪个阶段减少了等待或固定成本。

### 2.1 源码入口和关键结构

以下代码路径基于 SGLang `84a7a84` 整理。后面的代码块是为了讲清主路径而做过删节的源码片段，省略了异常处理、fallback 分支和部分参数。

- `python/sglang/srt/speculative/dflash_worker_v2.py`
- `python/sglang/srt/speculative/dflash_info_v2.py`
- `python/sglang/srt/speculative/dflash_info.py`
- `python/sglang/srt/speculative/dflash_utils.py`
- `python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py`
- `python/sglang/srt/speculative/triton_ops/dflash_accept_bonus.py`
- `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/overlap_utils.py`
- `python/sglang/srt/managers/schedule_batch.py`

核心类：

| 类/结构 | 职责 |
| --- | --- |
| `DFlashWorkerV2` | 负责 DFlash speculative decoding 的主执行逻辑。 |
| `DFlashDraftInputV2` | 在 Spec V2 overlap 下跨 iteration 携带 draft 状态、`seq_lens`、`verified_id`、KV allocation watermark。 |
| `DFlashVerifyInput` | 把 target verify 包装成 `ForwardMode.TARGET_VERIFY`。 |
| `FutureMap` | 在 overlap 模式下跨 iteration 传递 `new_seq_lens`、verified token、draft input 等 GPU 数据。 |

### 2.2 一轮 DFlash decode 在 SGLang 中大致怎么跑

#### 2.2.1 Prefill 阶段

在 prefill/extend 阶段，DFlash worker 首先让 target model 正常处理 prompt，同时要求捕获 hidden states：

**源码片段：Prefill 阶段捕获 hidden states**

```python
model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
batch_output = target_worker.forward_batch_generation(model_worker_batch)
```

然后把 prompt 对应的 target hidden states 立即 materialize 到 draft KV cache：

**源码片段：将 target hidden 写入 draft KV**

```python
_append_target_hidden_to_draft_kv_by_loc(
    target_hidden=logits_output.hidden_states,
    cache_loc=model_worker_batch.out_cache_loc,
    positions=positions,
)
```

这里的“立即 materialize”很关键。原因是 SGLang 的 radix cache/prefix cache 后续可能更新，如果 target hidden states 只临时保存而不立刻写入 draft KV，容易破坏 prefix cache 复用和 KV 状态一致性。

部署时，这一步是在用一次 prefill 阶段的额外写入，换后续 draft path 的上下文质量。如果这里延迟太高，短输出请求很容易还没摊开成本就结束；如果这里不写稳，后续 prefix cache / draft KV 一致性会变成更麻烦的问题。

prefill 结束后，DFlash 会生成下一轮 decode 需要的 `DFlashDraftInputV2`：

**源码片段：构造下一轮 DFlashDraftInputV2**

```python
verified_id = next_token_ids
new_seq_lens = model_worker_batch.seq_lens
```

#### 2.2.2 Decode 第 1 步：准备 DFlash block

每轮 decode 时，DFlash 使用上一轮的 `verified_id` 作为 block 第 0 个 token，后面填 mask token：

**源码片段：DFlash block token 初始化**

```python
block_ids[:, 0] = verified_id
block_ids[:, 1:] = mask_token_id
```

同时构造 `positions_2d` 和 `verify_out_cache_loc_2d`。

SGLang 为这个步骤实现了专门的 Triton kernel：`triton_ops/dflash_prepare_block.py`。这个 kernel 一次性完成 `block_ids`、`positions`、`cache_loc` 的准备，避免 Python eager 路径里多个 tensor op、gather、scatter 和 kernel launch。

这里主要防的是“小操作把大模型省下来的时间吃掉”。这些 tensor shape 不大，但每轮都会发生；如果散落成多个 eager op，就会制造 kernel launch、同步和 Python 调度噪声。

#### 2.2.3 Decode 第 2 步：draft model 一次跑完整 block

DFlash 构造一个 `ForwardBatch`，使用 `ForwardMode.TARGET_VERIFY` 风格的固定 block 形态跑 draft model：

**源码片段：draft model block forward**

```python
forward_batch = ForwardBatch(
    forward_mode=ForwardMode.TARGET_VERIFY,
    input_ids=block_ids.flatten(),
    positions=positions,
    input_embeds=input_embeds,
    spec_algorithm=SpeculativeAlgorithm.DFLASH,
    capture_hidden_mode=CaptureHiddenMode.NULL,
)

draft_logits_output = draft_model_runner.forward(forward_batch).logits_output
```

这里 draft model 返回 block 内 hidden states。SGLang 随后用 target model 的 `lm_head` 从这些 hidden states 中得到 draft tokens：

**源码片段：用 target lm_head 采样 draft tokens**

```python
draft_next = _greedy_sample_from_vocab_parallel_head(
    hidden_states=draft_hidden[:, 1:, :],
    lm_head=target_lm_head,
)
```

这个设计有两个含义：

1. draft model 主要负责快速产生 block hidden states。
2. token id 的 vocab projection 使用 target lm_head，避免 draft vocab/head 与 target 不一致带来的额外复杂性。

draft path 的部署目标不是“尽可能强”，而是“足够便宜且接受率足够高”。一旦 draft block latency 接近 target verify latency，或者需要复杂的逐 token draft loop，投机解码的净收益就会被迅速压缩。

#### 2.2.4 Decode 第 3 步：target verify

DFlash 把 draft tokens flatten 后交给 target model verify：

**源码片段：target verify forward**

```python
verify_input = DFlashVerifyInput(
    draft_token=verify_input_ids,
    positions=positions,
    draft_token_num=block_size,
    capture_hidden_mode=CaptureHiddenMode.FULL,
)

verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(...)
target_out = target_worker.forward_batch_generation(
    forward_batch=verify_forward_batch,
    is_verify=True,
    skip_attn_backend_init=True,
)
```

`DFlashVerifyInput` 的作用是：

1. 设置 `ForwardMode.TARGET_VERIFY`。
2. 把 input ids 改成 draft tokens。
3. 让 target model 返回 block 内每个位置的 logits。
4. 捕获 hidden states，供下一步写入 draft KV cache。

verify 是最容易被误解的阶段：它可以并行，但仍然消耗 target model capacity。高并发时要重点看 `batch_size * block_size` 形成的 verify token 数，以及这些 token 最后有多少真正变成 committed tokens。

#### 2.2.5 Decode 第 4 步：计算 accept length 和 bonus token

greedy 模式下，DFlash 的验证规则是：

- `accept while candidates[:, 1:] == target_predict[:, :-1]`
- `bonus = target_predict[accept_len]`
- `commit_len = accept_len + 1`

也就是：

给定 `candidates = [current, d1, d2, d3, ...]` 和 `target_predict = [p1, p2, p3, p4, ...]`，如果 `d1 == p1` 就接受 `d1`，如果 `d2 == p2` 就继续接受 `d2`，直到第一次不匹配。最后附加 target 在该位置预测出的 bonus token。

SGLang 为这个步骤实现了 Triton kernel：`triton_ops/dflash_accept_bonus.py`。它一次性输出 `accept_len`、`commit_lens`、`bonus_ids`、`out_tokens`、`new_seq_lens`。

这个 kernel 的收益不是改变算法复杂度，而是减少每轮 decode 中非常频繁的小 tensor 操作。

accept/bonus 是 scheduler 下一轮能否尽早推进的关键信号源。它输出的不只是 token，还包括 `commit_lens` 和 `new_seq_lens`；这些数据越少经过 CPU 同步，下一轮越不容易被卡住。

#### 2.2.6 Decode 第 5 步：把 target hidden 写入 draft KV cache

target verify 后，DFlash 取 target hidden states：

**源码片段：读取 target verify hidden states**

```python
hidden = logits_output.hidden_states
hidden = hidden.view(bs, block_size, -1)
```

然后只把 committed prefix 对应的部分写入 draft KV cache：

**源码片段：把 verify hidden 写入 draft KV**

```python
_append_target_hidden_to_draft_kv_by_loc(
    target_hidden=hidden.reshape(-1, hidden_dim),
    cache_loc=verify_out_cache_loc,
    cache_loc_2d=verify_out_cache_loc_2d,
    positions=positions,
    commit_lens=commit_lens,
)
```

这一步就是 DFlash 的 KV injection 在 serving 里的核心实现。

如果启用了 fused KV materialization，SGLang 会走 `triton_ops/fused_kv_materialize.py`。它把跨层 KV projection、K norm、RoPE、写入 KV pool 等步骤融合或批处理。

收益是降低逐层 Python loop 和多个小 kernel 的成本。

这一步防的是 KV injection 变成新的瓶颈。DFlash 靠 KV injection 提高 acceptance length，但如果 materialization 要跑大量逐层小操作、占用主 stream 或触发额外拷贝，理论上的接受率收益会被实现成本抵消。

### 2.3 Spec V2 overlap 如何放大 DFlash 收益

#### 2.3.1 Spec V2 的核心目标

DFlash 算法本身减少了 target decode 次数，但如果每轮都被 CPU scheduler、KV allocation、D2H copy 卡住，实际吞吐仍然会掉。

Spec V2 的目标是：

- 减少 host-device synchronization。
- 让 CPU scheduling、KV planning、result copy 与 GPU forward 重叠。

SGLang 的 scheduler overlap 路径大致是：

**源码骨架：scheduler overlap 简化路径**

```python
with forward_stream:
    resolve_forward_inputs(batch, future_map)
    batch_result = model_worker.forward_batch_generation(
        batch,
        on_publish=future_map.publish(...)
    )
    future_map.stash(...)

with copy_stream:
    batch_result.copy_to_cpu(...)
```

这段路径对应的 serving 目标很直接：让 GPU 少等 CPU，让下一轮少等上一轮的尾部工作。DFlash 每轮多出来的状态协调越多，overlap 的价值越大。

#### 2.3.2 on_publish：提前发布 new_seq_lens

Spec V2 worker 可以在 forward 内部调用：

**源码片段：提前发布 new_seq_lens**

```python
on_publish(new_seq_lens)
```

DFlash 在 accept/bonus 算完后就 publish `new_seq_lens`。这意味着 scheduler 不必等 DFlash worker 完整返回，下一轮的一些准备工作可以更早开始。

DFlash 之后还要做 `target hidden -> draft KV materialization`、`verify_done event record`、`next_draft_input construction`。如果没有 overlap，这些尾部工作会把 scheduler 串住。Spec V2 把一部分“下一轮准备”提前和这些尾部工作重叠。

`on_publish` 解决的是“结果已经足够规划下一轮，但 worker 还没完全收尾”的问题。它把 `new_seq_lens` 这种关键状态提前暴露给 scheduler，减少 round boundary 上的空泡。

#### 2.3.3 FutureMap：避免每轮同步搬运跨 iteration 数据

overlap 模式下，上一轮输出不会简单地立刻同步回 scheduler，然后再作为下一轮输入。SGLang 使用 `FutureMap` 按 `req_pool_indices` 存储跨 iteration 数据：

常见 buffer 包括 `new_seq_lens_buf`、`output_tokens_buf`、`verified_id_buf`、`topk_p_buf`、`topk_index_buf`、`hidden_states_buf`。

对 DFlash 来说，常见路径更轻：`DFlashDraftInputV2` 直接携带 `verified_id` 和 `new_seq_lens`，当 `direct_carry_valid` 为真时可以跳过 FutureMap 的额外 gather。这样可以减少很多小规模 GPU gather、D2H 同步和 Python 调度成本。

FutureMap 的核心价值是让跨轮状态尽量留在 GPU 侧。如果每一轮都要把 `verified_id`、长度、采样 buffer 搬回 CPU 再搬回来，投机解码会在小同步上失血。

#### 2.3.4 DFlash 特化：CPU seq_lens 允许滞后一轮

很多 attention backend 或 scheduler 逻辑需要 host-side `seq_lens_cpu`。如果每轮 decode 都要把 GPU 上的新 `seq_lens` 拷到 CPU，就容易形成同步点。

DFlash 在 `FutureMap.resolve_seq_lens_cpu` 里有专门处理：

**源码片段：DFlash 跳过 CPU seq_lens 同步的分支**

```python
if spec_algo.is_dflash() and direct_carry_valid:
    batch.seq_lens = draft_input.new_seq_lens
    return
```

也就是说，DFlash 只更新 GPU `seq_lens`，CPU 侧长度可以保持滞后。下一轮规划需要的 host-side upper bound 放在 `DFlashDraftInputV2` 的 `planning_seq_lens_cpu` 和 `reserved_seq_lens_cpu` 里。

这避免了 decode 热路径上的频繁 D2H。

这是一个典型的“用保守规划换少同步”的设计。只要 scheduler 有足够安全的长度上界，就不必为了得到精确 host-side `seq_lens_cpu` 每轮都阻塞 GPU。

#### 2.3.5 KV over-allocation：用显存换调度顺滑

`DFlashDraftInputV2.prepare_for_decode` 里会提前为下一轮 DFlash block 预留 KV slots：

**源码片段：KV over-allocation 规划公式**

```python
planning_len = committed_len + block_size
reserved_len = max(cur_alloc_len, committed_len + 2 * block_size)
```

然后更新 shared `req_to_token`：

**源码片段：更新 req_to_token 映射**

```python
assign_req_to_token_pool_func(...)
```

这叫 over-allocation。它的目的不是省显存，而是避免在 draft/verify 热路径里反复发生 allocator alloc、backup/restore、`req_to_token` rebuild、host/device sync。

代价是更多 KV headroom；收益是 decode step 更少被 allocator 和 host metadata 卡住。

KV over-allocation 是吞吐和显存之间的交换：它能让热路径更稳定，但会降低可承载的 running requests 或 context headroom。实际部署时要把它和 `max_running_requests`、KV pool 剩余量、CUDA Graph buffer 一起看。

### 2.4 执行时序图

**时序图：Spec V2 + DFlash 执行链路**

![Spec V2 + DFlash 执行时序图](/assets/dflash-self-spec/spec-v2-dflash-timeline.svg)

Spec V2 的核心不是让单个 GPU kernel 变快，而是让这些阶段不要完全串行。

## 结语

DFlash 的算法收益可以概括为：便宜地提出一批 token，并让 target 一次性验证。

SGLang Spec V2 的系统收益可以概括为：把 draft、verify、KV 写入、scheduler 准备、CPU copy 之间的等待压到最低。

两者合起来，才是 DFlash 在实际 serving 中能跑出高 throughput 的原因。前者决定理论上限，后者决定这个上限能不能在真实 serving 热路径里兑现。

## 参考资料

1. LMSYS Blog: The next generation of speculative decoding: DFlash and Spec V2
   https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/

2. DFlash paper
   https://arxiv.org/abs/2602.06036

3. SGLang main branch source files
   https://github.com/sgl-project/sglang

4. Key SGLang implementation paths
   `python/sglang/srt/speculative/dflash_worker_v2.py`
   `python/sglang/srt/speculative/dflash_info_v2.py`
   `python/sglang/srt/speculative/dflash_info.py`
   `python/sglang/srt/speculative/dflash_utils.py`
   `python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py`
   `python/sglang/srt/speculative/triton_ops/dflash_accept_bonus.py`
   `python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`
   `python/sglang/srt/managers/scheduler.py`
   `python/sglang/srt/managers/overlap_utils.py`
