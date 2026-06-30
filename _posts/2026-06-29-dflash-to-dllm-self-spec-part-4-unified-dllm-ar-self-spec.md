---
layout: post
title: "从 DFlash 到 dLLM Self-Spec（四）：Self-Spec dLLM & AR Uni-Model"
date: 2026-06-29
tags: [Infra, Speculative Decoding, dLLM]
description: "从 I-DLM 和 Nemotron tri-mode 出发，分析 dLLM / AR unified model 如何在同一个模型内实现 self-spec，以及 serving 侧的 KV、scheduler 和 phase switching 瓶颈。"
---

## TL;DR

DFlash / DSpark 属于“额外 draft/spec module + target verify”的落地路径。I-DLM 和 Nemotron tri-mode 代表另一条路线：模型本身同时具备 block diffusion draft 与 AR verify 能力，self-spec 发生在同一个模型内部。

这类方法的 serving 重点不再是“如何部署一个 draft model”，而是：

- 如何在同一个 runtime 里切换 causal / bidirectional attention。
- 如何让 draft phase 与 verify phase 共享或修剪 KV cache。
- 如何让 scheduler 理解每轮 accepted tokens 可能变长。
- 如何在 CUDA Graph、LoRA/weight view、attention metadata 和 KV pool 上控制 phase-switch 成本。
- 如何在高并发时动态缩短 block，避免 block diffusion 的并行 draft 把 target/main model capacity 消耗在低价值 suffix 上。

## 相关链接

- [I-DLM paper](https://arxiv.org/abs/2604.11035)
- [I-DLM SGLang integration](https://github.com/Introspective-Diffusion/I-DLM/tree/main/sglang)
- [Nemotron LinearSpec examples](https://github.com/NVlabs/Nemotron-Labs-Diffusion/tree/main/chat)
- [SGLang DLLM onboarding issue](https://github.com/sgl-project/sglang/issues/25802)
- [SGLang Nemotron diffusion runtime PR](https://github.com/sgl-project/sglang/pull/25803)
- [Your LLM Knows the Future](https://arxiv.org/abs/2507.11851)

## 第四章：dLLM / AR unified model 与 self-spec

### 4.1 从外置 draft 到模型内 self-spec

DFlash 的 serving 形态可以概括为：轻量 draft model 并行提出一个 block，target model 一次 verify 多个 token。DSpark 继续把 fixed block verify 改成 confidence-scheduled verify，让 target capacity 更少浪费在低接受率 suffix 上。

I-DLM 和 Nemotron tri-mode 把问题又往前推进一步：如果模型本身就能在 diffusion-style block draft 和 AR-style verification 之间切换，self-spec 就可以不再依赖独立 draft model。它的潜在收益不是“少启动一个模型”这么简单，而是：

| 对比维度 | DFlash / DSpark | I-DLM / Nemotron tri-mode |
| --- | --- | --- |
| draft 来源 | 额外 draft/spec module，通常比 target 小。 | 同一个 unified model 的 diffusion 或 masked block mode。 |
| verify 来源 | target AR model。 | 同一个模型的 AR/causal phase。 |
| 核心工程问题 | draft/target overlap、KV injection、verify budget、scheduler 与 CUDA Graph。 | attention mode switching、shared KV、phase-specific graph capture、LoRA/weight view、变长 block 调度。 |
| serving 风险 | draft 太弱导致 acceptance 低；target verify 高并发下仍可能满。 | 同一个模型承担 draft 和 verify，phase 切换与 KV 正确性更难；draft 不一定便宜。 |

因此，unified self-spec 不是对 DFlash / DSpark 主线的替代，而是把同一个问题推进到模型结构和 runtime 交界处：仍然是为了少跑昂贵的 AR decode step，但系统瓶颈从 draft/target 协同，迁移到了 attention mode、KV 语义、CUDA Graph phase 和 per-request state。

### 4.2 先看 draft / verify 边界：I-DLM vs Nemotron

两者都可以叫 self-spec，但 draft/verify 的边界不一样。I-DLM 更像“跨轮次的内生 speculative decoding”：上一轮 MASK draft，下一轮同一个 forward 先 verify 旧 specs、再 draft 新 specs。Nemotron tri-mode 更像“同一模型的双模式 speculative decoding”：先用 diffusion/bidirectional mode draft，再切到 AR/causal mode verify。

| 维度 | I-DLM / ISD | Nemotron tri-mode / linear self-spec |
| --- | --- | --- |
| draft 从哪里来 | 同一个模型在 MASK positions 上采样，得到 speculative tokens 和 draft 分布 `q`。 | 同一个模型切到 diffusion/block mode，在一个 block 内并行提出候选 tokens。 |
| verify 从哪里来 | 下一轮把旧 specs 放到 clean positions，同一个模型给出 anchor 分布 `p`，用 `p/q` 或近似准则接受。 | 切回 causal/AR view，用 AR logits 从左到右验证 diffusion draft 的候选前缀。 |
| draft/verify 时间关系 | verify 旧 specs 和 draft 新 specs 被合并在同一轮 forward/后处理里。 | draft phase 和 verify/causal KV update 更像两个 mode/phase，runtime 需要显式切换。 |
| attention 边界 | SGLang 实现里强制 causal view，依靠 strict causal/introspective 训练让 MASK draft 与 clean verify 一致。 | 同一模型显式切换 bidirectional/block diffusion attention 与 causal attention。 |
| KV 处理 | MASK KV 不持久化；被拒绝或未提交的 suffix 通过 `_kv_trim_info` 释放。 | diffusion draft 与 AR continuation 的 KV 语义不同，可能需要 causal KV refresh/update。 |
| 分布保证 | 严格路径依赖 speculative decoding 的 `p/q` 接受准则，目标是保持 AR 分布。 | 依赖 AR verify 作为分布锚点；具体实现更强调 mode switching 与 shared KV 的正确性。 |
| serving 成本重心 | 单轮 forward 内 logits gather、p/q verify、sampling、KV trim 和 per-request state 更新。 | attention mode 切换、causal KV update、draft LoRA/weight view、CUDA Graph phase capture。 |

更直观地说，I-DLM 的 draft 和 verify 是“错开一轮但揉在一次执行里”：第 `t` 轮留下 specs，第 `t+1` 轮验证它们，同时继续产生下一批 specs。它的好处是 forward 组织很紧凑；难点是请求状态更细，`pending/spec/draft_probs/advance/trim` 都要跟着每个请求走。

Nemotron 的 draft 和 verify 是“同一模型的两个运行姿态”：draft 时更像 block diffusion，verify 时更像 AR decode。它的好处是算法语义清楚，可以把 AR view 当作校验锚点；难点是 serving runtime 不能只把它当普通模型 forward，需要正确切换 attention metadata、KV 语义、LoRA/weight view 和 CUDA Graph phase。

所以二者的共同点是“没有独立 target/draft 两模型”；真正的差异是 self-spec 的边界在哪里。I-DLM 把边界放在轮次和 token layout 上，Nemotron 把边界放在 attention phase 和 model mode 上。

#### 4.2.1 Draft stride：下一轮 window 如何推进

`block_size` / `window_size` 并不等于每轮一定推进这么多 token。它更像每轮可尝试的最大窗口、kernel shape 或 KV 预留上限；真实推进量由本轮 committed tokens 决定。接受结果通常先更新 committed prefix、释放 rejected suffix，下一轮 draft / verify window 再从新的 request state 出发。

| 方法 | 当前轮做什么 | window size 的含义 | 下一轮 draft 从哪里开始 |
| --- | --- | --- | --- |
| DFlash | draft model 生成固定线性 block，target 一次 verify，接受连续 prefix 并追加 target bonus。 | verify block 的最大宽度；每轮实际推进 `accept_len + 1`。 | 用本轮 bonus token 作为下一轮 block 第 0 位，后面重新填 mask。 |
| DSpark | DFlash-style block backbone 仍并行算 hidden/base logits，Markov/RNN head 在轻量 sampling 层左到右修正；confidence 决定 verify prefix length。 | 最大 draft / verify 上限；实际 verify 长度可由 confidence 和负载动态缩短。 | 从新的 committed prefix 开始，由 scheduler 重新分配下一轮 verify budget。 |
| I-DLM / ISD | 当前 forward 验证上一轮 specs，同时从后续 MASK positions 采样下一轮 clean/spec tokens。 | `2 * gen_block_size - 1` 是容纳旧 specs 和新 MASK draft 的最大 layout，不是固定推进量。 | 如果旧 specs 全接受，下一轮从缓存的 `pending/spec_tokens/draft_probs` 进入 verify；如果拒绝，则从 correction token cold start。 |
| Nemotron LinearSpec | diffusion/block mode 产生候选，causal/AR view verify 或识别 SOL path。 | diffusion draft / verify window 的最大上限；实际提交由 causal verify / compaction 返回结果决定。 | scheduler 提交 returned tokens，释放 suffix；下一轮基于新的 `output_ids` 构造 mask block。 |

因此，这几条线真正不同的不是“每轮是否按 block size 固定迭代”，而是 draft/verify 的时间边界和 window 起点：DFlash 和 DSpark 是本轮 draft、本轮 target verify；Nemotron 是本轮 diffusion draft、本轮 causal verify；I-DLM 则是本轮 verify 旧 specs，同时 draft 新 specs，形成跨轮次流水。`block_size` 负责限制和预留，`committed_tokens_this_round` 才决定 request 真正前进多少。

![I-DLM 与 Nemotron draft / verify 触发时间流水线](/assets/dflash-self-spec/idlm-nemotron-draft-verify-timeline.svg)

I-DLM 的 `pending/spec_tokens/draft_probs` 会跨轮次携带，下一轮用同一次 forward 同时 verify 旧 specs 并 draft 新 specs；Nemotron LinearSpec 则在同一轮里先 diffusion draft，再切回 causal/AR view verify，commit 后下一轮重新从新的 committed prefix 构造 block。

### 4.3 I-DLM / ISD：跨轮次 self-spec 如何服务化

I-DLM 的关键是 Introspective Strided Decoding（ISD）。它把每轮输入组织成两类位置：

- 已经生成、准备被校验的 clean/spec positions。
- 接下来要并行预测的 MASK positions。

![I-DLM overview](/assets/dflash-self-spec/idlm-overview-teaser.png)

算法上可以把 ISD 理解成“延后一轮校验”的 self-spec。第 `t` 轮先从 MASK 位置并行采样出一组 speculative tokens，并保存这些 token 的 draft 分布 `q`；第 `t+1` 轮再把上一轮 speculative tokens 当成 clean positions 放进输入，让同一个模型在 causal view 下给出 anchor 分布 `p`。如果某个 speculative token 在 `p/q` 接受准则下通过，它就被提交；如果被拒绝，则用 target/anchor 分布给出的 correction token 回退并重新冷启动。

![I-DLM decoding comparison](/assets/dflash-self-spec/idlm-decoding-comparison.png)

这和 DFlash 的相似点是：都在尝试“一轮提交多个 token”。差异在于，I-DLM 没有外置 draft model；draft 分布 `q` 和 verify 分布 `p` 都来自同一个 introspective diffusion model，只是来自不同输入位置和不同轮次的模型视图。因此 ISD 的算法核心不是“两个模型谁更便宜”，而是“同一个模型能否对自己上一轮的 MASK draft 保持足够高的一致性”。

一个简化轮次如下：

| 轮次 | 输入形态 | 模型做的事 | 本轮输出 |
| --- | --- | --- | --- |
| cold start | `[t0, M, M, ...]` | 从 MASK 位置并行采样 clean token 和 speculative tokens。 | 提交少量 token，缓存下一轮要 verify 的 specs。 |
| verify round | `[pending, spec0, spec1, ..., M, M, ...]` | 用 clean/spec positions 的 logits 校验旧 specs，同时从后续 MASK 位置采样新 specs。 | 提交 accepted prefix 或 correction token，并缓存新 specs。 |

#### 4.3.1 p/q verify、per-request state 与 KV trim

I-DLM 最关键的算法前提是 introspective consistency：模型要能“认可自己上一轮生成的 token”。如果 diffusion model 在 MASK 位置采样出的 token，下一轮换成 clean position 后自己并不认同，那么 acceptance length 会很低，self-spec 就只会制造额外 forward 和 KV 压力。I-DLM 通过 strict causal masking、logit shift 和 all-masked training，让 MASK draft 和 clean/causal verify 之间尽量对齐。

这里的 strict causal 很重要。I-DLM 虽然叫 diffusion LLM，也使用 MASK positions 做 block draft，但它不是任意 bidirectional denoising；它刻意让 masked token 和 clean token 都服从 causal 约束。这样 serving 侧才能把它接到 AR runtime 上，继续复用 paged KV cache、continuous batching 和 CUDA Graph，而不是为完全 bidirectional denoising 另写一条 serving 栈。

`p/q` acceptance 是它和 speculative decoding 对齐的地方。`q` 是上一轮 MASK draft 产生某个 speculative token 的概率；`p` 是下一轮把该 token 放进 clean/causal view 后，模型作为 anchor 给出的概率。接受概率可以理解成 `min(1, p/q)`：如果 anchor view 认为这个 token 不比 draft view 更离谱，就接受；如果拒绝，则用 correction token 回到 anchor 分布。这就是 I-DLM 能把 self-spec 和 AR 分布保证连起来的关键。

在 SGLang 实现里，`IDLMBlockN` 的注释已经把这个执行形态写得很直白：

```python
"""
Block-N speculative generation: 1 forward -> 1 to N output tokens.

Two modes per round (blk = 2*N - 1 positions):
  Verify:     [pending, spec0, ..., spec(N-2), M, M, ..., M]
              -> verify specs left-to-right, then sample N new tokens
  Cold start: [t0, M, M, ..., M]
              -> sample 1 clean + (N-1) spec, hold specs for next verify

MASK KV is always freed after the forward (never persists in cache).
Rejected specs cause a fallback to cold start with the corrected token.
"""
```

这段实现的 serving 含义比算法公式更重要：

| 代码机制 | serving 含义 |
| --- | --- |
| `block_size = 2 * gen_block_size - 1` | 一次 forward 既要容纳上一轮 `pending/spec`，也要容纳下一轮 MASK draft。KV pool 需要按 worst-case block 预留。 |
| `_pending` / `_spec_tokens` / `_spec_draft_probs` | speculative state 变成 per-request runtime state，而不是独立 draft model 的输出缓存。 |
| `use_spec_verify` / `verify_alpha` / `fast_verify` | verify 可以是严格 p/q，也可以用更便宜的近似路径；这直接影响延迟与分布精确性。 |
| `_dllm_write_override` | 输出给请求的 tokens 不一定等于本轮完整 block，需要算法告诉 SGLang 写哪些 token。 |
| `_advance_override` | 每轮推进长度是变长的，scheduler 不能假设一个 block 全部 commit。 |
| `_kv_trim_info` | 没被接受的 spec/MASK KV 必须释放，否则 KV pool 会被 speculative suffix 吃掉。 |

从代码路径看，I-DLM 的一轮不是“先 draft forward，再 verify forward”的两模型流水，而是把 verify、correction、sampling 和 KV trim 尽量放在同一轮 GPU/CPU 后处理里：

```python
# Phase 2: Forward
forward_batch.dllm_force_causal = True
out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
forward_batch.dllm_force_causal = False

# Phase 3: Batched post-forward -- verify + sample + trim
full_logits = out.logits_output.full_logits
```

这解释了它为什么更接近 “self-spec”：同一个模型前向产生 clean logits 来校验旧 speculative tokens，同时从 MASK positions 里采样新 speculative tokens。后续的 p/q verify 或 fast verify 仍然要消耗 logits gather、softmax/top-k、correction 和 CPU 状态更新，所以 verify 并不是免费的；只是它被更紧凑地揉进同一个 forward / 后处理窗口里。

这里还有一个容易忽略的 trade-off：下一轮 specs 是在“旧 specs 尚未完全确认”的输入 layout 后半段 draft 出来的。如果旧 specs 最终全部或大部分被接受，后半段 MASK 看到的上下文就接近真实 prefix，新 specs 质量通常更稳定；如果旧 specs 很早被拒绝，那么后半段 MASK draft 实际上建立在错误上下文之后，这批新 specs 不能安全复用，通常要丢弃并回到 correction / cold start 路径。也就是说，I-DLM 把 verify 和 draft 合到一次 forward，收益来自减少单独 draft round，但代价是下一轮 draft 质量强依赖上一轮 spec acceptance。

I-DLM 对 SGLang 的启发主要有三点：

- self-spec 的关键不只是算法 acceptance，而是 request state 是否能表达 `pending -> spec -> accepted/rejected -> trim`。
- 如果 MASK KV 不及时释放，高并发下 speculative block 会非常快地放大 KV pressure。
- “AR-compatible serving” 的本质是尽量复用 paged KV cache、continuous batching、CUDA Graph 这些 AR serving 基础设施，而不是为 dLLM 另做一套完全不同的 serving stack。

### 4.4 Nemotron tri-mode：mode switching 形态的 self-spec

Nemotron-Labs-Diffusion 的 README 把模型定位成 tri-mode：AR decoding、diffusion-based parallel decoding、self-speculation。它的核心不是额外训练一个独立 draft model，而是同一个模型在推理时通过 attention pattern 切换模式：

| 模式 | 推理形态 | 对 self-spec 的作用 |
| --- | --- | --- |
| `ar` | 标准 causal AR decode。 | 作为 verify / fallback 的语义锚点。 |
| `dlm` | block diffusion parallel decode。 | 一次填充多个 MASK positions。 |
| `linear_spec` | diffusion draft + AR verification。 | 同一个模型自己 draft，再用 causal view 校验。 |

#### 4.4.1 open-source LinearSpec：diffusion draft + causal verify

算法上，Nemotron tri-mode 的 self-spec 更像“同一模型内的 DFlash 变体”：draft phase 用 diffusion/block mode 一次提出多个候选 token；verify phase 切回 causal/AR mode，从左到右检查这些候选是否符合 AR 分布。被接受的前缀进入输出；第一次不匹配的位置由 AR view 给出 correction 或 bonus token。这样做的目标是保留 AR decode 的分布锚点，同时利用 diffusion block draft 减少逐 token 生成的串行深度。

从 HF remote code 抓到的 `linear_spec_generate` 看，当前公开实现是更基础的 LinearSpec：draft phase 在 diffusion mode 下填完整个 mask block；verify phase 切回 causal mode，接受连续匹配 prefix，并追加一个 AR bonus token。

```python
# HF remote code: modeling_nemotron_labs_diffusion.py
block = torch.full((1, block_length), token_mask_id, dtype=torch.long, device=device)
block[0, 0] = next_token.item()

# draft phase: diffusion / bidirectional view
_set_diffusion_lm(True)
_toggle_adapters(True)
...
block[is_mask] = draft_tokens[is_mask]

# verify phase: causal / AR view
_set_diffusion_lm(False)
_toggle_adapters(False)
enc_out = self.encoder(
    input_ids=block,
    past_key_values=past_key_values,
    use_cache=True,
    use_causal_mask=True,
)
ar_tokens = verify_logits.argmax(dim=-1)

accepted = 0
for i in range(block_length - 1):
    if ar_tokens[0, i].item() == block[0, i + 1].item():
        accepted += 1
    else:
        break
accepted += 1

accepted_toks = ar_tokens[:, :accepted]
_crop_dynamic_cache(past_key_values, cache_len + accepted)
next_token = ar_tokens[:, accepted - 1 : accepted]
```

论文或技术材料里的 SOL / recursive dynamic compaction 可以放在这个基础实现之后理解：普通 `linear_spec_generate` 只做 longest-prefix verify；SOL 思路则尝试在候选 block 内通过 mask / compaction 保留仍然能和 AR target 对齐的 suffix。

![Nemotron recursive dynamic compaction SOL path](/assets/dflash-self-spec/nemotron-recursive-dynamic-compaction-sol.png)

当前 HF remote code 和可见的 SGLang staging 分支里，没有公开显式的 SOL-path search 实现；SOL / recursive dynamic compaction 更适合理解为论文里的进一步优化方向，而不是当前开源 `linear_spec_generate` 的逐行行为。

recursive dynamic compaction / SOL path 并不是在当前 block 内每找到一个 accepted position，就立刻从这个位置后面重新开一个完整 `block_size` 的 draft。它是在当前候选 block 内确定可提交的 path；本轮结束后，runtime 才把 accepted tokens 写入 `output_ids`，释放 rejected suffix，下一轮再基于新的 committed prefix 构造 `origin_input_ids + output_ids + [MASK] * active_block_size`。

这个思路和 *Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential* 里的 Quadratic Decoding 有相似的动机：二者都不满足于朴素 longest-prefix verify，因为线性 verify 一旦中间位置失败，后面的 future proposal 就会被整段丢掉。Quadratic Decoding 通过在 speculative tokens 之间插入二次数量的 mask tokens，把 future proposals 展成更像 tree 的 layout；即使某个候选位置验证失败，后续仍然有新的 mask 分支可以在下一步继续提供 speculative tokens。Nemotron 的 recursive dynamic compaction / SOL path 则更像在 diffusion block 已经产生的候选里做路径回收：通过 compaction 找到仍然能和 AR view 对齐的线性提交路径，减少 suffix 被无条件丢弃。

两者的 serving 含义也不同。Quadratic Decoding 是提前展开更多候选，用更宽的 mask / attention layout 换更稳定的后续 verify；Nemotron SOL 是对当前 block 内已算出的候选做路径压缩和回收。前者的成本主要体现在 `k^2` 级别的并行 token layout，后者的成本主要体现在 compaction、path search、metadata 更新和可能的额外 verify 逻辑。共同点是：投机解码的收益不只取决于 draft 准不准，还取决于 reject 之后能不能少浪费已经并行算出来的 future tokens。

这条路线的关键前提是模型本身同时懂两种 attention 语义：

| 能力 | 算法作用 | serving 影响 |
| --- | --- | --- |
| causal AR view | 提供 verify / correction / continuation 的分布锚点。 | 需要 causal attention metadata 和与 AR prefix 一致的 KV。 |
| bidirectional/block diffusion view | 在一个 block 内并行补全 MASK tokens。 | 可以提高单轮 draft token 数，但会制造 speculative KV 和 block 内状态。 |
| linear self-spec path | 把 diffusion draft 与 AR verify 组合成一轮 self-spec。 | 需要 runtime 在两种 view 之间切换，并把接受结果反馈给 scheduler。 |

#### 4.4.2 从算法形态到 serving 约束

Nemotron 的重点不是把一个 AR 模型外接成 draft/target 两模型，而是让同一个模型天然具备三种使用方式：`ar` 走 causal decode，`dlm` 走 diffusion-style block denoise，`linear_spec` 把 diffusion draft 和 AR verify 拼成 self-spec。换句话说，tri-mode 的“统一”发生在模型结构和 attention pattern 层，而不是 serving 进程里临时组合两个 checkpoint。

在 `linear_spec` 里，draft phase 借助 block diffusion 的并行性一次提出候选 token；verify phase 切回 causal/AR 视角，把这些候选当作已知输入，从左到右验证 prefix。这个设计和 DFlash 的精神相似：都把“先猜一段，再一次性验证一段”作为提速来源。但 Nemotron 的 draft 不来自外部轻模型，而来自同一个模型的另一个 attention mode。

draft LoRA 可以看作这条路线的一个工程化补丁：draft phase 需要更擅长快速提出可接受的 block，verify phase 需要保持 AR anchor 的语义稳定。因此 LoRA 只在 draft 侧打开、verify 侧关闭是合理的设计。但这也把算法选择变成了 runtime 问题：weight view 不同，CUDA Graph capture、graph replay、LoRA toggle 和 attention metadata 都要按 phase 对齐。

公开 chat 示例里也能看到这个思路：`linear_spec_generate` 先在 bidirectional attention 下抽取 diffusion draft，再用 causal attention 做 AR 验证，并且每轮还能产生 bonus token。带 LoRA 的版本会把 `linear_spec_lora` adapter 作为 draft-side adapter 接上。

### 4.5 SGLang DLLM runtime：两类 self-spec 的共同底座

SGLang DLLM runtime 不是只为 I-DLM 写的，也不是只为 Nemotron 写的。它更像一类把 dLLM/block generation 接进 SGLang AR serving 底座的扩展能力。I-DLM 和 Nemotron 的公开代码来源、集成形态和成熟度不同，但都在考验同一组 serving 抽象：request state、scheduler、ForwardBatch metadata、algorithm plugin、KV/output processing。

从 serving 热路径看，这套 DLLM runtime 可以拆成五层：

| 层次 | runtime 负责什么 | I-DLM 侧压力 | Nemotron 侧压力 |
| --- | --- | --- | --- |
| request state | 给每个请求维护 `dllm_phase`、`dllm_ids`、block offset、active block size。 | 要表达 `pending/spec/draft_probs` 这类跨轮次 speculative state。 | 要表达 prefill、diffusion extend、causal update 等 phase。 |
| scheduler | 决定请求进入 DLLM prefill 还是 DLLM extend，组 batch，处理 preemption。 | accepted length 变长，不能假设每轮推进固定 block。 | 高并发时可能按 running batch size 调整 active block size。 |
| ForwardBatch metadata | 把 block input、KV slot、req pool index、attention flags 传给 model runner。 | 需要支持 `dllm_force_causal`、输入 layout 和 per-request trim 信息。 | 需要支持 `dllm_causal_kv_update` 这类 attention mode flag。 |
| algorithm plugin | 在 DLLM runtime 上挂不同 decoding algorithm。 | `IDLMBlockN` 做 p/q verify、sampling、write override、KV trim。 | `FastDiffuser` 做 block denoise、confidence unmask、causal KV refresh。 |
| output / KV processing | 把算法输出写回请求，释放未提交 KV，更新 finished/stream/metrics。 | `_dllm_write_override`、`_advance_override`、`_kv_trim_info` 是关键接口。 | `new_tokens < block_size` 时要释放 rejected suffix；causal KV update 要保持 prefix 语义一致。 |

“共同底座”不是指二者已经共享同一份完全一致的 upstream 实现，而是指它们需要的 serving 能力高度重叠。I-DLM 更像在测试 runtime 的 state machine 和 output processing：每个请求都有 pending/spec 状态，每轮接受长度可能不同，KV trim 必须精确到 speculative suffix。Nemotron 更像在测试 runtime 的 phase switching：同一个模型要在 diffusion draft、AR verify、causal KV update、draft LoRA/weight view 之间切换，同时还希望 CUDA Graph 和 continuous batching 不被打碎。

一个简化链路是：`ReqDllmMixin` 先把请求变成 DLLM 请求；scheduler 走 DLLM path 并调用 `ScheduleBatch.prepare_for_dllm_block_extend`；`ForwardBatch` 携带 block input、KV slot 和 attention flags；具体算法由 `DllmAlgorithm` 插件实现，例如 `IDLMBlockN` 或 `FastDiffuser`；最后 `process_batch_result_dllm` 写回输出、释放 KV、处理 stream 和 metrics。

这里的关键不是类名本身，而是它把“生成一个 token”的 AR 请求模型，扩展成了“每轮可能处理一个 block、提交变长 token、释放部分 KV、切换 attention phase”的请求模型。

Nemotron 侧最值得看的不是入口参数，而是 DLLM runtime 如何表达 mode switching。第一处是 attention 决策。`get_dllm_causal_attention` 里把 causal attention 变成一个 runtime 条件：只有在 `causal_context` 打开，并且当前不是普通 DLLM extend，或者正在做 `dllm_causal_kv_update` 时，才回到 causal view。

```python
def get_dllm_causal_attention(layer, forward_batch, dllm_config, default_causal):
    if dllm_config is None or layer.attn_type != AttentionType.ENCODER_ONLY:
        return default_causal
    if layer.is_cross_attention:
        return False
    if not dllm_config.causal_context:
        return False
    return (
        not forward_batch.forward_mode.is_dllm_extend()
        or forward_batch.dllm_causal_kv_update
    )
```

第二处是 `FastDiffuser`。它会在 block 内多步填 MASK；如果模型需要 causal context，则在 block 被接受后触发一次 causal KV refresh/update：

```python
if self.causal_context:
    forward_batch.dllm_causal_kv_update = True
out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
if self.causal_context:
    forward_batch.dllm_causal_kv_update = False
```

这说明 shared KV 并不等于“无脑共用同一份缓存”。diffusion draft phase 可能使用非 causal / bidirectional view，但后续 AR verify 或 causal continuation 需要一份与 causal prefix 一致的 KV。runtime 必须知道什么时候刷新、什么时候提交、什么时候释放 speculative suffix。

第三处是 CUDA Graph 与 weight view。SGLang DLLM 分支里有 `DllmGraphPhaseHooks`：

```python
@dataclass(frozen=True)
class DllmGraphPhaseHooks:
    before_draft: Callable[[], None] = _noop
    before_verify: Callable[[], None] = _noop
    after_capture: Callable[[], None] = _noop
```

这类 hook 的意义是：draft phase 和 verify phase 可能需要不同的 weight view，例如 Nemotron 的 draft LoRA 只在 diffusion draft 时打开，而 causal verify 时关闭。如果要用 CUDA Graph，capture 的 graph 必须和当时的 weight/LoRA/attention metadata 状态一致，否则图能不能复用、复用后是否正确都会变成问题。

第四处是并发感知的 block size。`DllmConfig` 支持 `block_size_tiers`：高并发时可以选择更小的 active block size，但静态 `block_size` 仍按 tier 中最大值给 KV pool 做 worst-case sizing。这一点和 DSpark 的 confidence-scheduled verify 方向相似：block 越长，单轮潜在输出越多，但高并发时也越容易占满 KV 和 main model capacity。

### 4.6 KV 生命周期：self-spec 最容易踩坑的地方

Unified self-spec 里最容易被低估的不是 logits 后处理，而是 KV 生命周期。因为 draft/verify 都发生在同一个模型或同一套 runtime 里，很多人会直觉以为 KV 可以自然共享；但实际 serving 中，KV 能否共享取决于它对应的 token 是否已经提交、attention 语义是否一致、以及后续 AR continuation 是否能安全复用。

I-DLM 的 KV 生命周期可以简化成：

| KV 类型 | 是否应该持久化 | 原因 |
| --- | --- | --- |
| prompt / accepted prefix KV | 是 | 已经进入输出前缀，后续 decode 必须复用。 |
| pending / accepted specs 对应 KV | 视接受结果而定 | 被提交的部分进入 prefix；被拒绝后的 suffix 不能留下。 |
| MASK positions KV | 否 | MASK 只是本轮 draft 工作区，`IDLMBlockN` 明确要求 MASK KV forward 后释放。 |
| rejected speculative suffix KV | 否 | 这些 token 没有成为真实输出，留下会污染后续 attention，也会浪费 KV pool。 |

这就是 `_kv_trim_info` 和 `_advance_override` 重要的原因。前者告诉 runtime 哪些 KV slot 要释放；后者告诉请求下一轮应该前进多少 token。没有这两个信号，runtime 可能会出现两类问题：要么把未接受的 speculative KV 当作真实 prefix，导致生成语义错误；要么保守地不释放，导致高并发下 KV pool 被 MASK/suffix 快速吃满。

Nemotron 的 KV 问题不完全一样。它的难点不是“MASK KV 是否持久化”这么单一，而是 diffusion draft phase 和 AR continuation 的 KV 语义不同。diffusion/block attention 可以在 block 内看见更多位置；AR continuation 需要的是严格 causal prefix。如果 draft phase 的 KV 被直接当作 causal prefix KV 使用，就可能出现 attention 语义不一致。因此 `FastDiffuser` 在 `causal_context` 打开时会触发 `dllm_causal_kv_update`，让 accepted block 刷新成与 causal view 一致的 KV。

这也是为什么 shared KV cache 不等于无成本共享。shared 的是 KV pool 和请求映射基础设施，不是说所有 phase 产生的 KV 都可以直接互换。真正的 serving 约束是：

- accepted prefix KV 才能长期存在。
- speculative suffix KV 必须能按请求、按 block、按 accepted length 释放。
- diffusion view 产生的 KV 是否可复用，要看后续是不是需要 causal view。
- block size 越大，临时 KV 和 rejected suffix 的峰值越高。
- 高并发时，KV correctness 和 KV pressure 会一起变成吞吐上限。

### 4.7 这类 unified self-spec 的关键瓶颈

把 I-DLM 和 Nemotron tri-mode 放在一起看，可以得到一张更贴近 serving 的瓶颈图：

| 瓶颈 | 为什么会出现 | 应该观察什么 |
| --- | --- | --- |
| phase switch overhead | 同一个模型要在 draft / verify / KV update 之间切换 attention、LoRA、metadata 或 graph。 | CUDA Graph 命中率、graph capture shape 数量、LoRA toggle 是否导致额外同步。 |
| KV correctness | diffusion draft 的 KV、AR verify 的 KV、accepted prefix 的 KV 语义不同。 | rejected suffix 是否及时 free，MASK KV 是否持久化，causal KV update 是否成为额外 forward。 |
| verify 仍然消耗主模型 capacity | self-spec 没有独立 target，但 verify/correction 仍要用同一个大模型 logits。 | 高并发下每秒 main-model forward 次数、每轮 accepted length、verify 后处理耗时。 |
| dynamic block scheduling | block 大可以提高单轮输出上限，但会放大 KV pressure 和低价值 suffix。 | block size tiers、running batch size、acceptance tail、per-request latency。 |
| sampling / p/q 后处理 | 严格 verify 需要 p/q、softmax、correction；近似 verify 又会引入质量/分布 trade-off。 | full logits gather、softmax/top-k kernel 时间、CPU 状态更新和同步点。 |
| continuous batching 复杂度 | 每个请求处在不同 phase，accepted length 也不同。 | scheduler 是否能合并同相位请求，是否频繁 preempt，是否出现小 batch graph miss。 |

这也回答了一个容易误解的点：unified self-spec 不意味着 verify 不再是瓶颈。它减少的是跨模型部署与 draft/target 协调成本，但 verify/correction 仍然要占用同一个模型的 forward 与 logits 后处理。高并发时，如果每个请求都带着较大的 speculative block 进入 verify 或 causal KV update，main model capacity 一样会被打满。

### 4.8 小结：统一模型 self-spec 的 serving 边界

Unified model self-spec 的关键变化是：draft/verify 边界不再主要落在两个模型之间，而是下沉到同一个模型的 attention mode、KV 语义、graph phase 和 request state 里。

| 路线 | serving 问题 |
| --- | --- |
| DFlash / Spec V2 | 外置 draft module 如何与 target verify、KV injection、overlap scheduler 配合。 |
| DSpark | DFlash 之后，verify budget 如何变成负载感知资源。 |
| dLLM / AR unified self-spec | 同一个模型如何通过 attention/KV/graph/scheduler 组合出 draft + verify。 |

![I-DLM 与 Nemotron self-spec 边界对比](/assets/dflash-self-spec/unified-self-spec-boundary.svg)

这类 runtime 最需要处理的是下面这些 infra 问题：

1. I-DLM 的 `IDLMBlockN` 为什么要用 `2*N-1` 的 block layout。
2. ISD 如何把 verify 和下一轮 draft 放进一次 forward。
3. Nemotron tri-mode 如何用 attention pattern 切换 AR / DLM / self-spec。
4. shared KV cache 为什么仍然需要 causal KV update 和 speculative suffix trim。
5. block size tiers 为什么是高并发 serving 中的重要旋钮。
6. draft LoRA / graph phase hooks 为什么会影响 CUDA Graph capture 与复用。

### 4.9 当前源码索引

I-DLM 侧：

- `Introspective-Diffusion/I-DLM`
- `inference/sglang/sglang/srt/dllm/algorithm/idlm_blockN.py`
- `inference/sglang/sglang/srt/dllm/config.py`
- `inference/sglang/sglang/srt/dllm/mixin/req.py`
- `inference/sglang/sglang/srt/dllm/mixin/scheduler.py`

I-DLM 当前公开材料里更直接的实现入口是 I-DLM repo 自带的 SGLang integration，而不是 `sgl-project/sglang` 上已经合并的 upstream PR。因此这里把它作为 bundled integration 来引用，不把它误写成 SGLang upstream PR。

Nemotron / SGLang DLLM 侧：

- `NVlabs/Nemotron-Labs-Diffusion`
- `xp/dlm_api/dlm_generate/nemotron.py`
- `chat/chat_linear_spec.py`
- `chat/chat_linear_spec_lora.py`
- `python/sglang/srt/dllm/algorithm/fastdiffuser.py`
- `python/sglang/srt/dllm/attention.py`
- `python/sglang/srt/dllm/config.py`
- `python/sglang/srt/dllm/graph.py`
- `python/sglang/srt/dllm/mixin/scheduler.py`

Nemotron 相关的 SGLang upstream / staging PR stack（状态截至 2026-06-29）：

| 链接 | 状态 | 相关性 |
| --- | --- | --- |
| `sgl-project/sglang#25802` | tracking issue | Nemotron Labs Diffusion upstreaming stack，总入口。 |
| `sgl-project/sglang#25803` | upstream PR，open | Shared DLLM runtime、Nemotron model implementation、FastDiffuser、focused tests、B200 validation。 |
| `hutm/sglang#2` | staging PR，closed | Onboard Nemotron Labs Diffusion algorithms。 |
| `hutm/sglang#3` | staging PR，closed | LoRA-aware LinearSpec execution。 |
| `hutm/sglang#4` | staging PR，closed | FA4 attention for DLLM graphs。 |
| `hutm/sglang#5` | staging PR，closed | LinearSpec scheduler hot path optimization。 |
| `hutm/sglang#6` | staging PR，closed | Dynamic LinearSpec block tiers。 |
| `hutm/sglang#7` | staging PR，closed | ModelOpt NVFP4 on SM100；更偏目标部署/量化路径，不是 self-spec 核心算法。 |

## 参考资料

1. I-DLM: Introspective Diffusion Language Models
   https://arxiv.org/abs/2604.11035

2. Introspective-Diffusion/I-DLM GitHub repository
   https://github.com/Introspective-Diffusion/I-DLM

3. NVlabs/Nemotron-Labs-Diffusion GitHub repository
   https://github.com/NVlabs/Nemotron-Labs-Diffusion

4. Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential
   https://arxiv.org/abs/2507.11851

5. SGLang DLLM onboarding issue
   https://github.com/sgl-project/sglang/issues/25802

6. SGLang PR: Add Nemotron diffusion runtime and FastDiffuser
   https://github.com/sgl-project/sglang/pull/25803

7. SGLang Diffusion LLM roadmap issue
   https://github.com/sgl-project/sglang/issues/14199

8. Staging PR: Onboard Nemotron Labs Diffusion algorithms
   https://github.com/hutm/sglang/pull/2

9. Staging PR: Add LoRA-aware LinearSpec execution
   https://github.com/hutm/sglang/pull/3

10. Staging PR: Enable FA4 attention for DLLM graphs
   https://github.com/hutm/sglang/pull/4

11. Staging PR: Optimize LinearSpec scheduler hot path
    https://github.com/hutm/sglang/pull/5

12. Staging PR: Add dynamic LinearSpec block tiers
    https://github.com/hutm/sglang/pull/6

13. Staging PR: Support ModelOpt NVFP4 on SM100
    https://github.com/hutm/sglang/pull/7
