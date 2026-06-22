---
layout: post
title: "Towards Better Tokenizer (4): 多模态大模型如何把世界变成 Token"
date: 2026-06-22
tags: [Tokenizer, Multimodal, Pretraining]
description: "从 image token、audio token、video token、media placeholder、processor、position encoding 和原生多模态设计出发，讨论多模态大模型 tokenizer 的设计取舍。"
---

系列一讲文本 tokenizer：BPE、pre-tokenizer、词表规模、压缩率和预训练成本。

系列二讲 agentic tokenizer：工具调用、thinking token、chat template、parser 和 serving runtime。

系列三讲 tokenizer 研究趋势：tokenizer-free、动态 tokenization、多语种 token tax、领域扩词表和 action tokenization。

这一篇继续往外走：**当输入不再只是文本，而是图像、视频、音频、屏幕、坐标和动作时，tokenizer 到底变成了什么？**

我的判断是：多模态模型里的 tokenizer 已经不是一个单独的 `tokenizer.json` 文件，而是一条完整的 modality-to-token pipeline：

| 组件 | 作用 |
|---|---|
| text tokenizer | 处理文本、代码、JSON、tool trace |
| media placeholder token | 在文本上下文中标记图像、视频、音频位置 |
| image / audio / video processor | 把原始模态转成模型可消费的输入 |
| encoder / projector / resampler | 抽取、压缩并对齐 soft tokens |
| spatial / temporal position encoding | 表达空间位置和时间顺序 |
| chat template / runtime parser | 让多模态输入输出和 serving 协议对齐 |
| multimodal token budget | 控制上下文、延迟和 KV cache 成本 |

文本模型里的 tokenizer 主要决定“语言如何被压缩”。多模态模型里的 tokenizer / processor 决定的是：**世界如何被序列化成模型能预测、能对齐、能推理、能执行的 token 流。**

## 1. 先给结论

如果说文本 tokenizer 的核心问题是：

> 一句话应该切成多少 token？

那么多模态 tokenizer 的核心问题变成了：

> 一张图、一段视频、一段声音、一个 GUI 状态，应该以什么粒度进入上下文？

这背后有几组关键取舍：

- 图像要保留多少空间细节；
- 视频要保留多少帧和时间顺序；
- 音频要保留波形、语音、声纹还是事件；
- 多模态 token 会吃掉多少 context window；
- vision/audio token 是否进入 KV cache；
- 多图、多视频、多轮对话时边界是否清晰；
- 坐标、OCR、grounding、tool call 是否能稳定表达；
- processor、chat template、runtime 是否和训练格式一致。

所以，一个好的多模态 tokenizer 不是“把图片压得越短越好”，而是要在下面几件事之间平衡：

- 感知细节；
- 空间 / 时间定位；
- token 成本；
- 长上下文可扩展性；
- 训练稳定性；
- 推理延迟；
- agentic 可执行性。

## 2. 多模态 Tokenizer 不是 BPE 的简单扩展

文本 tokenizer 的输入是字符串，输出是 token id：

> `"北京天气怎么样？" -> [token_1, token_2, token_3, ...]`

多模态模型面对的输入更复杂：

```text
用户：请看这张图，找出图中的表格，并把第二行金额加总。
图像：一张截图或照片
```

模型实际看到的往往不是“图片文件”，而是一个被序列化后的混合上下文：

```text
<|im_start|>user
请看这张图，找出图中的表格，并把第二行金额加总。
<image>
<|im_end|>
```

这里的 `<image>` 通常只是 placeholder。真正的图像内容会由 processor 读入，经过 vision encoder、projector 或 resampler，变成一串 soft visual embeddings，再插入到文本序列对应的位置。

也就是说，多模态模型里至少有两类 token：

```text
hard token: tokenizer 输出的离散 token id，例如 <image>、<|user|>、普通文字
soft token: 图像/音频/视频 encoder 产生的连续 embedding，未必对应词表里的 token id
```

这就是多模态 tokenizer 最容易让人误解的地方：很多时候，我们说 image token，并不是说图像被 BPE 切成了词表里的离散 token，而是说图像被变成了一组能和文本 token 一起进入 Transformer 的 embedding slot。

## 3. 一张图是怎么进入 LLM 的

主流 VLM 的 pipeline 大致是：

```text
image
  -> resize / crop / patchify
  -> vision encoder
  -> merger / projector / resampler
  -> visual embeddings
  -> 插入到 <image> placeholder 位置
  -> LLM decoder 继续做 next-token prediction
```

这里每一步都是 tokenizer 设计的一部分。

### 3.1 Placeholder Token

`<image>`、`<|image|>`、`<start_of_image>` 这类 token 的作用不是保存图像像素，而是告诉模型和 processor：

```text
这里有一段视觉内容，视觉 embedding 应该插入到这个位置。
```

placeholder 的设计会影响三个问题：

- 多张图如何排序；
- 文本和图像如何交错；
- 工具调用或最终回答能否引用对应图片。

例如：

```text
用户：比较图一和图二的价格差异。
<image_1>
<image_2>
```

如果只有一个通用 `<image>`，模型需要靠上下文顺序理解图一、图二；如果有更结构化的 image block，系统可以更稳定地维护多图索引。

Gemma 4 的官方 prompt formatting 文档明确把 `<|image|>`、`<|audio|>` 这类 token 放进控制 token 协议，同时把 tool、tool_call、tool_response、thinking/channel token 也纳入同一套格式。这说明在新一代多模态模型里，media placeholder 已经和 agentic control token 进入了同一个协议层。

参考：[Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)

### 3.2 Patch 和视觉 token 数量

图像进入模型前通常会被切成 patch。最朴素的想法是：

```text
image H x W
  -> patch size P x P
  -> (H/P) * (W/P) 个视觉 token
```

如果图像分辨率越高、patch 越小，视觉 token 越多，模型能看到更多细节，但代价也更大：

- prefill 更慢；
- KV cache 更大；
- context window 被视觉 token 占掉；
- 多图和视频场景成本迅速膨胀。

这就是为什么 VLM 不能只追求“高分辨率”。高分辨率截图、OCR、图表、GUI 操作当然需要细节；但如果所有图片都按最高分辨率展开，长上下文和多轮 agent 会很快不可用。

一个好的多模态 tokenizer / processor 往往要支持动态策略：

```text
简单图片 -> 少量视觉 token
复杂图表 -> 更多视觉 token
长截图 -> 分块或动态裁剪
视频 -> 降帧 + 压缩 + 时间建模
```

### 3.3 Merger / Projector / Resampler：视觉 Tokenizer 的压缩层

vision encoder 输出的 patch embeddings 通常不能直接喂给 LLM，需要一个中间模块做对齐和压缩。不同模型会叫它 projector、merger、resampler、abstractor，名字不完全一样，但核心职责很接近：

```text
vision hidden size -> LLM hidden size
many patch tokens -> fewer visual tokens
```

其中 **merger** 通常更强调“合并视觉 patch”。例如若干相邻 patch feature 先被 concat、pooling 或 MLP 处理，再变成更少的 visual tokens：

```text
patch_1, patch_2, patch_3, patch_4
  -> merger
  -> visual_token_1
```

projector 更强调 hidden size 对齐，resampler 更强调用 learned queries 或 attention 从大量视觉特征里采样固定数量 token。现实实现里，这几个职责经常混在同一个模块中。

可以粗略区分为：

```text
vision encoder:
  从像素提取 patch features

merger:
  合并 / 降采样视觉 patch，减少 visual token 数

projector:
  把视觉特征映射到 LLM hidden size

resampler / abstractor:
  用 attention 或 learned queries 从视觉特征中抽取固定数量 token
```

这些模块不是 tokenizer 文件的一部分，但它们承担了“把图像变成 LLM token 流”的关键职责。换句话说，文本 tokenizer 决定一句话切成多少 token，vision merger / resampler 决定一张图展开成多少 visual tokens。

严格说，merger / projector / resampler 主要是 **encoder-based VLM** 里的接口层：

```text
image -> vision encoder -> merger/projector/resampler -> LLM
```

因为这里有一个独立 vision encoder，模型必须解决两个接口问题：

```text
视觉 patch tokens 太多，需要压缩；
vision hidden size 和 LLM hidden size 不同，需要对齐。
```

到了 native / unified multimodal 路线，这个接口层不一定还叫 merger、projector 或 resampler，甚至可能不再以独立模块出现。它可能被 direct projection、discrete image tokenizer、semantic image tokenizer、audio frame projection 等机制替代。

但它们共同解决的是同一个问题：

```text
如何把非文本信号压缩成模型可消费的 token / embedding stream？
```

如果 resampler 压得太狠，OCR、小字、表格、坐标和局部细节会丢；如果压得太少，推理成本会爆炸。

这也是为什么多模态 tokenizer 的评价不能只看图像 benchmark 分数，还要看：

- 每张图平均视觉 token 数；
- 单图和多图延迟；
- 高分辨率文档理解；
- 长视频上下文；
- grounding 坐标精度；
- agent 执行链路中的稳定性。

如果看最新代表性工作，图像 tokenizer 大致有三类思路：

```text
VLM 理解路线：
  图像 -> ViT / vision encoder -> merger / resampler -> LLM visual embeddings

原生生成路线：
  图像 -> semantic image tokenizer -> discrete image tokens -> autoregressive model

混合生成路线：
  图像/文本条件 -> LLM 内部生成图像表示 -> diffusion / VAE decoder 还原图像
```

Qwen、InternVL、MiniCPM-V 这类 VLM 更接近第一类，重点是如何控制 visual token budget。X-Omni 明确使用 semantic image tokenizer、language-image autoregressive model 和 diffusion decoder，代表“离散图像 token + 自回归生成”的方向。HunyuanImage 3.0 则把理解和生成放到 native multimodal / autoregressive 框架里，关注图像 token、2D 位置、attention mask 和生成质量之间的平衡。

参考：[X-Omni](https://arxiv.org/abs/2507.22058)、[HunyuanImage 3.0](https://arxiv.org/abs/2509.23951)

## 4. 视频 Tokenization：时间维度比图像更难

视频不是很多张图片简单拼接。视频多了一个时间轴：

```text
frame_1, frame_2, frame_3, ..., frame_T
```

如果每一帧都按图像 token 展开，成本会非常高。假设一帧需要 512 个视觉 token，100 帧就是 51200 个 token，还没算文本、音频和对话历史。

因此视频 tokenizer / processor 要回答：

- 采样多少帧；
- 是否按场景变化自适应采样；
- 是否压缩时间维度；
- 是否保留动作顺序；
- 是否能和音频对齐；
- 是否能引用某一帧或某个时间段。

常见方案包括：

```text
固定 FPS 抽帧
关键帧抽取
短 clip encoder
3D resampler
时间位置编码
多阶段摘要
```

视频 tokenization 的难点是，压缩掉的帧可能正好包含关键事件。比如“杯子什么时候被拿起”“人有没有越过门线”“屏幕上哪一刻弹出错误提示”，这些都依赖时间顺序和局部瞬间。

所以视频 tokenizer 的目标不是把所有帧都保留下来，而是让模型在可承受的 token budget 内保留足够的事件结构。

最近的视频 tokenization / token compression 工作，基本都在围绕一个问题展开：

```text
视频 token 不能固定按帧铺开，而要按信息量、时间连续性和音视频线索动态分配。
```

例如 InfoTok 从信息论角度做 adaptive discrete video tokenizer，根据视频复杂度分配不同 token 长度；ForestPrune 面向 Video-MLLM 做高比例视觉 token pruning，用时空结构减少冗余 token；OmniZip 则把问题推进到 audio-video joint compression，用音频线索指导音视频 token 压缩。

这些工作说明，视频 tokenizer 的核心已经不是“抽几帧”，而是：

```text
哪些帧、哪些区域、哪些时间片真的值得占用上下文？
```

参考：[InfoTok](https://research.nvidia.com/labs/cosmos-lab/infotok/)、[ForestPrune](https://arxiv.org/abs/2603.22911)、[OmniZip](https://arxiv.org/abs/2511.14582)

## 5. 音频 Tokenization：语音不等于文本

音频有两种很不同的处理方式。

第一种是把音频先转成文本：

```text
audio -> ASR -> text -> text tokenizer
```

这对语音问答、会议纪要、字幕生成很有效，但会丢掉很多非文本信息：

- 语气；
- 停顿；
- 重音；
- 情绪；
- 说话人；
- 背景声；
- 音乐和环境事件。

第二种是把音频直接变成模型可消费的 audio embeddings 或 discrete audio tokens：

```text
audio waveform / spectrogram
  -> audio encoder / projection
  -> audio tokens
  -> LLM
```

Gemma 4 12B Unified 是一个很值得关注的新案例。Google 的开发者博客把它描述为 encoder-free multimodal model：视觉上使用 raw `48x48` pixel patches 投影到 LLM hidden dimension；音频上把 16kHz audio 切成 40ms frames，再线性投影到 LLM input space。它的重点不是“先把音频转文字”，而是让 audio/image/text 更直接地共享同一个 decoder backbone。

参考：[Gemma 4 12B: The Developer Guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/)

这类设计对 tokenizer 的启发是：音频 tokenization 不应该只服务 ASR，它还可能服务更广义的世界建模。

最新音频 tokenizer 也在分化成两条路线。

第一条是 **离散音频 token**：把 speech / audio 编成语义层、声学层、韵律层等 token，再用语言模型做理解或生成。SpeechTokenizer 用 RVQ 分层表示语义和声学信息；UniAudio 2.0 提出 text-aligned factorized audio tokenization，目标是让音频 token 同时服务理解和生成；MOSS-Audio-Tokenizer 则强调可扩展的离散 audio tokens，让纯自回归 TTS 和 ASR 都能共享音频 token 接口。

第二条是 **tokenizer-free / continuous representation**：不把语音先离散成 token，而是直接建模连续语音表示。VoxCPM2 就把自己定位成 tokenizer-free TTS，绕开传统离散 speech tokenizer，用端到端 diffusion autoregressive 架构生成连续语音表示。

因此音频 tokenizer 的设计问题可以概括为：

```text
如果追求 LLM 式统一建模，离散 audio tokens 更自然；
如果追求语音自然度和表现力，continuous / tokenizer-free 路线也很有吸引力。
```

参考：[SpeechTokenizer](https://0nutation.github.io/SpeechTokenizer.github.io/)、[UniAudio 2.0](https://arxiv.org/abs/2602.04683)、[MOSS-Audio-Tokenizer](https://arxiv.org/abs/2602.10934)、[VoxCPM2](https://github.com/OpenBMB/VoxCPM/)

## 6. 两条路线：Encoder-Based 与 Native / Unified Multimodal

现在主流多模态模型大致有两条路线。

### 6.1 Encoder-Based VLM

这是当前最常见的工程路线：

```text
text tokenizer + vision encoder + projector/resampler + LLM
```

代表包括 Qwen3-VL / Qwen3.5、InternVL3.5、MiniCPM-V 4.x、LLaVA-OneVision、GLM-4.6V、Kimi K2.5 / K2.6 等。

它的优点很现实：

- 可以复用强大的 text LLM；
- 可以复用成熟 vision encoder；
- 训练相对稳定；
- 数据和工程生态成熟；
- 推理框架容易接入。

缺点也明显：

- vision encoder 和 LLM 之间有模态断层；
- 视觉 token 的压缩策略决定细节上限；
- 多图/视频会迅速吃掉上下文；
- 对 grounding、GUI、OCR 等任务，processor 的细节影响很大；
- 后训练时需要同时照顾视觉 encoder、projector、LLM、chat template。

这个路线很适合今天的大规模开源 VLM，因为它风险低、迁移快、生态兼容性好。

### 6.2 Native / Unified Multimodal

另一条路线更激进：把文本、图像、视频、音频甚至 action 都视为统一序列的一部分。

```text
text / image / audio / video / action
  -> unified token stream or unified embeddings
  -> decoder-only model
  -> next-token prediction / next-action prediction
```

Emu3 / Emu3.5、X-Omni、HunyuanImage 3.0 这类工作把图像、文本、视频或图像生成中间表示放进统一自回归序列，再做 next-token prediction 或原生多模态生成。Gemma 4 12B Unified 则走 encoder-free 的方向，把视觉和音频更直接地投影进 LLM input space。

参考：[Emu3: Next-Token Prediction is All You Need](https://arxiv.org/abs/2409.18869)、[Emu3.5: Native Multimodal Models are World Learners](https://arxiv.org/abs/2510.26583)、[X-Omni](https://arxiv.org/abs/2507.22058)、[HunyuanImage 3.0](https://arxiv.org/abs/2509.23951)

这条路线的吸引力在于：

- 模态之间的接口更统一；
- 生成图像/视频/动作更自然；
- 多模态预训练目标更接近统一建模；
- agent action、GUI 操作、robotics token 更容易并入同一序列。

但它的风险也更大：

- 训练成本高；
- 数据配比难；
- tokenizer / processor 设计错误会影响所有模态；
- 生成质量、理解能力、长上下文成本之间更难平衡；
- 推理生态还没有 encoder-based VLM 成熟。

所以短期内，大多数开源多模态模型仍会采用 encoder-based 路线；但长期看，native / unified multimodal 很可能成为更重要的研究和大模型预训练方向。

## 7. 主流开源多模态模型的 Token Pipeline 对比

这一节不追求穷尽所有 VLM，也不重点比较 benchmark 分数。这里更关心一个 tokenizer 视角的问题：

```text
每个模型到底如何把 image / video / audio / action 接到文本 token 流里？
```

所以我会按四个维度看：

```text
1. media placeholder / chat template 怎么设计；
2. visual/audio/action token 怎么产生；
3. token budget 怎么控制；
4. 是否和 tool / grounding / runtime parser 形成协议闭环。
```

先给一个横向读法：

| 模型 / 路线 | 这一节关注的 tokenizer 问题 |
|---|---|
| Qwen3-VL / Qwen3.5 / Qwen-VLA | media token 如何接入既有 text / tool / thinking / action 协议 |
| InternVL3.5 | 视觉 token 粒度是否能随内容和分辨率动态变化 |
| MiniCPM-V 4.x | 如何用更少 visual tokens 保留足够视觉信息 |
| Kimi K2.5 / K2.6 | 原生多模态模型里 text / vision / tool / thinking 如何共享上下文协议 |
| GLM-4.6V | 视觉观察如何转成坐标、结构化字段和可执行 tool call |
| Gemma 4 | media / thought / tool control token 如何进入同一套 prompt 协议 |
| HunyuanImage 3.0 / X-Omni | 生成端的离散图像 token 应该如何设计 |
| DeepSeek-OCR | 是否可以用视觉 token 反向压缩长文本上下文 |

如果按模态拆开，可以先看一个更窄的表。这里不追求逐项列全，而是看每条路线最关键的模态入口：

| 模型 / 路线 | 主要覆盖模态 | tokenizer / processor 处理方式 |
|---|---|---|
| Qwen3-VL / Qwen3.5 / Qwen-VLA | 文本、图像、视频、音频、action | 文本复用 Qwen tokenizer；图像/视频/audio 由 processor 变成 soft tokens；action/trajectory 接到 VLA 接口 |
| InternVL3.5 | 文本、图像、视频帧 | 文本走 LLM tokenizer；图像/视频帧走 vision encoder，并用 Visual Resolution Router 动态调 visual token 压缩率 |
| MiniCPM-V 4.x | 文本、图像、视频 | 小 LLM tokenizer + SigLIP/ViT 视觉特征；intra-ViT early compression 降低 visual token 成本 |
| Kimi K2.5 / K2.6 | 文本、图像、视频、tool trace | 长上下文文本/tool/thinking 格式 + 原生多模态视觉通路；visual tokens、thinking、tool trace 共享预算 |
| GLM-4.6V | 文本、图像、tool/action | GLM 文本/agent 协议 + 多模态输入；native Function Calling 把视觉感知接到可执行 tool call |
| Gemma 4 | 文本、图像、音频、tool | role/thought/tool control token；`<|image|>`、`<|audio|>` 触发内部 soft embeddings |
| HunyuanImage 3.0 / X-Omni | 文本、图像生成 | 文本 prompt / CoT schema 进入自回归序列；图像由 semantic image tokenizer 离散化，再由 decoder 还原 |
| DeepSeek-OCR | 文本、页面图像 | 把长文本页面渲染为图像，再压缩成视觉 token 做 context compression |

### 7.1 Qwen3-VL / Qwen3.5 / Qwen-VLA：把多模态 token 接入现有 agent 协议

Qwen 系列最值得关注的不是某一个单独的 `<image>` token，而是它把文本 tokenizer、chat template、vision processor、thinking/tool token、代码/FIM token 放进了相对统一的工程生态里。

从 tokenizer 设计看，Qwen 的优势是：

```text
稳定文本 tokenizer
+ media placeholder 进入 chat template
+ vision/audio processor 负责把媒体展开成 soft tokens
+ 动态 visual token budget
+ 多模态长上下文
+ tool / thinking / code token 复用文本模型协议
+ Qwen-VLA 进一步把 action / trajectory 接入输出协议
```

这里的 tokenizer 取舍是：**不把多模态 tokenizer 做成孤岛，而是让 media token 成为原有 LLM 协议的一部分。** 这对 agent 很重要，因为一个真实任务往往不是“看图回答”，而是：

```text
看图 / 看视频
  -> 思考
  -> 调工具
  -> 读工具返回
  -> 继续看屏幕或执行 action
```

如果 image/video placeholder、tool_call、thinking、role boundary 使用完全不同的协议，runtime 很难稳定串起来。

Qwen3.5 和 Qwen3.5-Omni 的启发更进一步：当 text、image、audio、audio-visual 都进入统一系统后，tokenizer 设计问题就不再是“视觉 token 怎么接到文本模型后面”，而是“不同模态如何共享一个可训练、可推理、可部署的上下文协议”。Qwen-VLA 则把这个问题推到 action：轨迹和动作也需要被 token 化或结构化输出。

按模态看，Qwen 的处理方式比较完整：文本走 Qwen tokenizer 和 chat template；图像/视频走 processor 变成 visual tokens；音频在 Omni 分支里成为 audio/audio-visual token 流；action 在 Qwen-VLA 里进入 action-and-trajectory prediction 框架。

对预训练团队来说，Qwen 路线的结论是：

```text
media token 要和 text/tool/action token 协同设计，而不是后期补丁式加入。
```

参考：[Qwen3-VL](https://qwenlm.github.io/blog/qwen3-vl/)、[Qwen3.5](https://qwen.ai/blog?id=qwen3.5)、[Qwen3.5-Omni](https://qwen.ai/blog?id=qwen3.5-omni)、[Qwen-VLA](https://github.com/QwenLM/Qwen-VLA)

### 7.2 InternVL3.5：native multimodal pretraining、视觉路由和效率

InternVL 系列更适合讨论“视觉 token 粒度”本身。它的核心问题不是有没有 `<image>`，而是每张图应该展开成多少视觉 token、哪些区域应该给更多 token、不同分辨率如何进入同一个 LLM。

InternVL3 强调 native multimodal pretraining，意味着视觉 token 不是只在后训练阶段临时适配，而是在预训练阶段就参与模型能力形成。InternVL3.5 的 Visual Resolution Router 则更直接地触碰 tokenizer 设计：**视觉 token budget 不应该固定，而应该随输入内容动态变化。**

按模态看，InternVL3.5 的重点仍然是视觉：文本由 LLM tokenizer 承载，图像和视频帧进入 vision encoder 后再经过动态分辨率/压缩路由，音频不是这条路线的核心。

这可以理解成一种多模态 pre-tokenizer：

```text
输入图像
  -> 判断内容复杂度 / 分辨率需求
  -> 选择视觉 token 展开策略
  -> 把不同粒度的 visual tokens 送入 LLM
```

它解决的是文本 tokenizer 里也会遇到的问题：不同样本的信息密度不同，不应该用同一种切分粒度。自然图片 caption 可能不需要很多视觉 token，但文档 OCR、表格、代码截图、GUI 屏幕就需要更高分辨率和更细粒度的 token。

InternVL 这一路线的 tokenizer 结论是：

```text
视觉 tokenization 应该是内容自适应的，而不是固定分辨率、固定 token 数。
```

这也解释了为什么多模态 tokenizer 的 benchmark 不能只看准确率，还要同时报告 token 数、分辨率路由策略和复杂文档上的失败模式。

参考：[InternVL3 technical report](https://arxiv.org/abs/2504.10479)、[InternVL3.5 technical report](https://arxiv.org/abs/2508.18265)

### 7.3 MiniCPM-V 4.x：极致压缩和端侧部署

MiniCPM-V 系列最适合作为“token 压缩效率”的案例。它提醒我们：视觉 token 不是越多越好，真正重要的是单位 token 的信息密度。

从 tokenizer 视角看，MiniCPM-V 的问题是：

```text
如何用尽可能少的 visual tokens，保留足够多的视觉语义？
```

MiniCPM-V 4.5 已经把高效视频理解、混合 fast/deep thinking、复杂文档解析作为重点；MiniCPM-V 4.6 进一步强调 intra-ViT early compression，支持混合 `4x/16x` visual token compression rate。

按模态看，MiniCPM-V 4.x 的核心是图像和视频：文本走小 LLM backbone 的 tokenizer，图像/视频帧走视觉 encoder，并在 ViT 内部尽早压缩 visual tokens，音频和 action 不是它的主要 tokenizer 设计目标。

这类设计的 tokenizer 含义很明确：压缩不应该只发生在 LLM 前面的 projector，也可以更早发生在 ViT 内部。换句话说，视觉 tokenizer 不一定是：

```text
完整 vision features -> resampler 压缩
```

也可以是：

```text
ViT 内部提前压缩
  -> 更少 visual tokens
  -> 更低 LLM prefill / KV cache 成本
```

它的代价是：压缩太早可能丢掉细节，尤其是 OCR、小字、表格线、UI 控件和局部坐标。因此 MiniCPM-V 路线的关键不是单纯减少 token，而是要回答：

```text
哪些视觉信息可以早压缩？
哪些信息必须保留到 LLM 阶段？
```

这对端侧模型尤其重要。端侧 VLM 的 tokenizer 设计，本质上是用 token budget 换 latency、memory 和电量。

参考：[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)

### 7.4 Kimi K2.5 / K2.6：从 Kimi-VL 到原生多模态

Kimi-VL 可以作为早期 VLM 背景，但在这篇文章里不应该再作为主案例。Kimi K2.5 / K2.6 已经更适合放在“原生多模态 + agentic 协议”的位置：视觉能力不再只是一个外接 VL 模块，而是和长上下文、tool calling、thinking 模式、agent trace 放在同一套模型能力里考虑。

从 tokenizer 视角看，Kimi K2.5 / K2.6 的关键问题不是“图片如何插到文本后面”，而是：

```text
text tokens
visual tokens
thinking tokens
tool trace tokens
agent state tokens
如何共享同一个长上下文预算？
```

这和 Kimi-VL 的区别在于：Kimi-VL 更像“长上下文 VLM”，重点是图像、文档页、截图如何作为视觉证据进入上下文；Kimi K2.5 / K2.6 更像“原生多模态 agent 模型”，重点是文本、视觉、工具调用、思考模式和任务状态如何在同一协议里协作。

长文档、网页、表格、代码截图、多轮工具调用同时出现时，至少需要这些边界：

```text
page boundary
image boundary
region boundary
OCR text boundary
tool observation boundary
reasoning / answer boundary
```

否则模型可能知道“上下文很长”，但不知道第几页、第几张图、第几个区域和当前推理步骤之间的关系。

Kimi K2.5 / K2.6 还提示了另一个 tokenizer 问题：thinking token 会和 visual tokens、tool trace tokens 抢上下文。长图像序列 + 长思考链 + 多轮工具调用同时出现时，tokenizer / processor 必须考虑上下文预算分配，而不是只优化单轮 VQA。

它也很适合放进 tokenizer 系列的另一个原因是：Kimi K2.5 的开源说明里明确提到 media token / chat template 的细节会影响模型行为，例如 `<|media_start|>` 被修正为 `<|media_begin|>`。这类小改动看起来只是模板字符串，但在多模态 agent 模型里，它决定了 runtime 是否能把视觉内容放到模型预期的位置。

按模态看，Kimi K2.5 / K2.6 的重点是文本、图像/视频和 agent trace：文本和 thinking 走长上下文协议，图像/视频作为视觉证据进入原生多模态通路，tool call / tool result / agent state 则和视觉证据一起竞争上下文预算。音频不是本文讨论的重点。

参考：[Kimi K2.5](https://moonshotai.github.io/Kimi-K2.5/)、[Kimi K2.5 technical report](https://arxiv.org/abs/2602.02276)、[Kimi K2.6](https://moonshotai.github.io/Kimi-K2.6/)、[Kimi K2.6 model card](https://huggingface.co/moonshotai/Kimi-K2.6)

### 7.5 GLM-4.6V：面向 agent 和复杂任务的多模态接口

GLM-V 更适合从“视觉 token 如何变成可执行工具协议”来看。GLM 系列在文本 agent token 上已经比较显式；GLM-4.6V 又强调 native Function Calling，这让它成为“多模态 + agentic tokenizer”的代表案例。

多模态 agent 里的 tokenizer 问题不是：

```text
模型是否看懂图片？
```

而是：

```text
视觉观察如何被稳定转写成坐标、结构化字段和 tool call？
```

例如：

```text
看图 / 看屏幕
  -> 定位目标
  -> 生成坐标或结构化描述
  -> 调用工具或执行 action
  -> 读取返回结果
  -> 继续下一步
```

这意味着 image token、coordinate representation、tool token、role token 必须能在同一套协议里工作。否则模型可能能描述屏幕，却不能稳定点击；能识别图表，却不能稳定生成工具参数；能看懂 PDF，却不能把证据放进可审计的 tool trace。

按模态看，GLM-4.6V 的核心是“视觉输入 + 工具协议”：文本和 tool call 使用 GLM 的 agent 格式，图像/截图/文档页可以作为多模态输入或工具参数进入模型，输出侧则要稳定变成 function calling、坐标或结构化字段。

GLM-4.6V 这类模型对 tokenizer 设计的启发是：

```text
多模态 tokenizer 的输出协议要面向 execution，而不只是 caption。
```

参考：[GLM-4.6V](https://github.com/zai-org/GLM-V)、[GLM-4.6V docs](https://docs.z.ai/guides/vlm/glm-4.6v)

### 7.6 Gemma 4：control token 与 encoder-free 的一个例子

Gemma 4 可以作为一个小例子：如果图像、音频、文本、工具调用、thinking channel 都进入同一个 decoder-only backbone，那么 tokenizer 不再只是文本分词器，而会变成模型输入输出协议的一部分。

Gemma 4 的关键点有三个：

```text
1. <|image|> / <|audio|> 是 tokenizer 级 control token；
2. 图像和音频更直接地投影到 LLM input space；
3. tool_call / tool_response / thought channel 也在同一套 control token 里。
```

按模态看，Gemma 4 的文本、图像、音频和工具调用都被放进同一套 prompt/control-token 协议：文本走 role/channel token，图像和音频由 placeholder 触发内部 soft embeddings，工具调用由 tool_call/tool_response token 交给 runtime parser。

这意味着 Gemma 4 不是只在文本 prompt 里加一个 `<image>`，而是把 media token、reasoning token、tool token 放进一套 prompt formatting 协议中。它对 tokenizer 设计的启发是：

```text
未来的 tokenizer 可能不是 text tokenizer + 若干额外 special token，
而是统一管理 text / image / audio / tool / thought / channel 的协议层。
```

这也解释了为什么 serving runtime 往往需要模型专用 parser。只要 tool_call 格式、string delimiter、channel 边界和 tokenizer control token 不一致，模型输出就很难被稳定执行。

参考：[Gemma 4 Prompt Formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)

### 7.7 HunyuanImage 3.0 / X-Omni：离散自回归图像生成里的 tokenizer 设计

混元图像 3.0 和 X-Omni 适合放在 native / unified multimodal 路线里，因为它们关注的是另一类 tokenizer 问题：**如果模型不仅要理解图像，还要生成图像，图像 token 应该怎么设计？**

传统扩散模型并不一定需要把图像变成 LLM 词表附近的离散 token；但 native multimodal / autoregressive image generation 路线必须回答：

```text
图像如何离散化？
图像 token 是否和语言 token 在同一个自回归序列里？
图像 token 是否能承载语义、布局、文字渲染和审美偏好？
生成后的离散 token 如何被 decoder 还原成高质量图像？
```

X-Omni 的框架很典型：它包含 semantic image tokenizer、统一的 language-image autoregressive model，以及离线 diffusion decoder。这里的 image tokenizer 不是普通 VLM 里的 `<image>` placeholder，而是把图像内容压成可自回归预测的离散语义表示。它的目标不只是“看懂图片”，而是让模型像生成文本 token 一样生成图像 token，再交给 decoder 还原成图像。

HunyuanImage 3.0 的技术报告则把自己定位成 native multimodal model，在自回归框架里统一多模态理解和生成，并引入 native Chain-of-Thoughts schema、渐进式预训练和后训练。它对 tokenizer 设计的启发是：当图像生成、图像编辑、文本理解和推理链放在同一个系统中时，tokenizer 不只是输入端的压缩器，也决定了生成端的 action space。

按模态看，这条路线的重点是文本和图像生成：文本 prompt / CoT schema 进入自回归序列，图像被 semantic image tokenizer 离散化，模型生成 image tokens，再由 diffusion 或 image decoder 还原成图像。

这条路线和 Qwen/InternVL/MiniCPM-V 这类 VLM 有明显区别：

```text
VLM tokenizer:
image -> visual embeddings -> text answer

Native image-generation tokenizer:
text / image condition -> autoregressive image tokens -> image decoder
```

它的优势是统一：图像和语言可以在同一个 next-token prediction 框架里建模。它的难点也很清楚：离散图像 token 如果语义不够强，会影响指令跟随、文字渲染、布局稳定性和审美质量；如果 token 太多，生成成本和长上下文成本又会变高。

所以混元图像 3.0 / X-Omni 应该放进系列四讨论，因为它们把 tokenizer 问题从“如何观察世界”推进到“如何生成世界”：

```text
observation tokenization -> generation tokenization
```

参考：[HunyuanImage 3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)、[HunyuanImage 3.0 Technical Report](https://arxiv.org/abs/2509.23951)、[X-Omni](https://arxiv.org/abs/2507.22058)、[X-Omni repo](https://github.com/X-Omni-Team/X-Omni)

### 7.8 DeepSeek-OCR / DeepSeek-OCR2：把视觉当作长文本压缩介质

DeepSeek-OCR 是一个很适合 tokenizer 系列的非典型案例，因为它反过来问了一个问题：

```text
长文本一定要按文本 token 读进去吗？
```

DeepSeek-OCR 的目标不是传统 OCR 系统，而是验证 context optical compression：把长文本页面编码成少量视觉 token，再由 decoder 还原文本。它的问题意识非常 tokenizer：当长文档直接按文本 token 进入 LLM 太贵时，是否可以把文本渲染成二维视觉空间，用视觉 encoder 做压缩？

按模态看，DeepSeek-OCR 主要处理文本和图像这两个模态，但方向是反过来的：它不是把图像解释成文本给 LLM 看，而是把文本页面变成图像，再用视觉 token 压缩长文本上下文。

这类方法不一定会直接替代通用 tokenizer，但很适合作为“多模态 tokenization 可以反过来压缩文本上下文”的案例。

它的 tokenizer 含义非常明确：

```text
文本 tokenization: 字符串 -> subword tokens
OCR-style optical tokenization: 页面图像 -> visual tokens -> 文本恢复 / 理解
```

这条路线的优势是可以利用二维排版结构压缩长文档；风险是细节恢复、可编辑性、引用定位和错误可解释性都更复杂。它更像一种面向长文档的特殊 context compression tokenizer，而不是通用文本 tokenizer 的替代品。

参考：[DeepSeek-OCR](https://arxiv.org/abs/2510.18234)、[DeepSeek-OCR repo](https://github.com/deepseek-ai/DeepSeek-OCR)、[DeepSeek-OCR2](https://github.com/deepseek-ai/DeepSeek-OCR-2)

## 8. 多模态 Agent：从 Image Token 到 Action Token

多模态 tokenizer 最终会和 agent 结合。

一个屏幕操作 agent 的上下文可能长这样：

```text
<|system|>
你是一个 GUI agent，可以观察屏幕并点击、输入、滚动。

<|user|>
帮我把这张发票里的总金额填到报销系统。

<|observation|>
<image>

<|assistant|>
<think>
需要先识别发票总金额，再定位表单金额输入框。
</think>
<tool_call>
{"name": "ocr", "arguments": {"region": "invoice"}}
</tool_call>

<|tool|>
<tool_response>
{"total": "1234.56", "currency": "CNY"}
</tool_response>

<|assistant|>
<tool_call>
{"name": "click", "arguments": {"x": 842, "y": 391}}
</tool_call>
<tool_call>
{"name": "type", "arguments": {"text": "1234.56"}}
</tool_call>
```

这个例子里，多模态 tokenizer 需要同时表达：

- 屏幕图像；
- OCR 结果；
- 坐标；
- 工具调用；
- 工具返回；
- thinking；
- 用户目标；
- 多轮状态。

因此，多模态 agent 的 tokenizer 不是 image token 单点问题，而是：

```text
observation tokenization + action tokenization + tool protocol + state machine
```

这也是为什么系列二和系列四是连在一起的。Agentic tokenizer 解决“动作如何被系统解析”，multimodal tokenizer 解决“世界状态如何被模型观察”。

## 9. 设计多模态 Tokenizer 时应该做哪些 Benchmark

多模态 tokenizer 的 benchmark 不能只看 VQA 分数。更合理的是围绕 token budget 和任务能力一起评估。

### 9.1 Token Budget Benchmark

统计不同输入下的 token 成本：

```text
单张自然图像
高分辨率截图
扫描 PDF
10 页文档
1 分钟视频
5 分钟视频
带音频视频
多图多轮对话
```

要记录：

- text token 数；
- visual/audio token 数；
- prefill latency；
- KV cache 占用；
- 吞吐；
- 多轮上下文增长速度。

这个 benchmark 能回答一个很工程的问题：

```text
这个多模态模型在真实产品里用得起吗？
```

### 9.2 细节保真 Benchmark

压缩率高不等于好。要看压缩后还能不能保留关键细节：

- 小字 OCR；
- 表格单元格；
- 图表坐标轴；
- UI 按钮；
- 公式；
- 代码截图；
- 地图和路线；
- 医学影像局部区域。

同样是 256 个视觉 token，自然图片 caption 可能够用，但发票、表格、GUI 和文档截图可能不够。

### 9.3 空间定位 Benchmark

多模态 agent 需要定位：

```text
点击哪个按钮？
框选哪块区域？
表格第几行第几列？
图片左上角的对象是什么？
视频第几秒发生了什么？
```

这类 benchmark 可以测试：

- grounding box 精度；
- 坐标输出稳定性；
- 多对象引用；
- 多图引用；
- UI 操作成功率。

### 9.4 长视频和音频 Benchmark

视频和音频要测试：

- 时间顺序；
- 事件检测；
- 跨帧对象追踪；
- 音画同步；
- 说话人分离；
- 长时间记忆；
- 关键帧遗漏率。

只测短视频 QA 不够，因为真实 agent 经常面对长会议、长录屏、长操作轨迹。

### 9.5 多轮 Agent Benchmark

多模态 tokenizer 最终要放进多轮系统里测：

```text
observe -> reason -> tool call -> tool response -> observe again -> action
```

要看：

- tool call 是否稳定；
- 坐标是否漂移；
- 视觉证据是否被保留；
- tool response 是否和视觉上下文对齐；
- 多轮后是否忘记最初目标；
- 用户输入和工具返回是否隔离。

这类 benchmark 能判断 tokenizer / processor / chat template / parser 是否真的成套工作。

## 10. 大规模预训练里应该怎么选

如果我是预训练团队，我会把多模态 tokenizer 设计拆成五个问题。

### 10.1 先定产品目标，再定视觉 token budget

不同目标需要不同 tokenizer：

```text
通用聊天 VLM -> 中等视觉 token，重视泛化
文档模型 -> 高分辨率 OCR 和布局
视频模型 -> 时间压缩和长上下文
GUI agent -> 坐标和小 UI 元素
端侧模型 -> 极致 token 压缩
生成式多模态 -> 更统一的离散/连续 token 空间
```

不要先拍一个固定 image token 数，然后让所有任务适配它。视觉 token budget 是产品能力边界的一部分。

### 10.2 文本 tokenizer 要和 media/control token 一起设计

多模态模型仍然离不开文本 tokenizer。文本 tokenizer 至少要支持：

- 多语种；
- 代码；
- JSON / XML / schema；
- tool call；
- thinking / channel；
- media placeholder；
- 坐标和结构化输出；
- FIM / code edit。

如果文本 tokenizer 对结构化格式不友好，多模态 agent 会在工具调用和坐标输出上吃亏。

### 10.3 Processor 必须成为发布物的一部分

多模态模型不能只发布权重和 tokenizer，还必须发布：

- image processor；
- video processor；
- audio processor；
- chat template；
- placeholder token 规则；
- visual token 展开规则；
- 分辨率策略；
- tool parser；
- streaming 行为；
- 多图和视频排序规则。

否则下游部署时很容易出现“模型能跑，但格式不对、视觉错位、工具调用解析失败”的问题。

### 10.4 Native / unified 路线值得探索，但要控制风险

Gemma 4、Emu3 / Emu3.5、HunyuanImage 3.0 / X-Omni 这类工作说明，统一多模态建模是很有吸引力的方向。但对大规模基座模型训练来说，它不是无脑替代 encoder-based VLM。

更现实的判断是：

```text
短期产品：encoder-based VLM 仍然更稳
中期探索：部分 encoder-free / direct projection 值得做
长期研究：native multimodal token stream 会越来越重要
```

如果团队资源有限，应该先把 processor、token budget、数据配比、chat template 和 tool protocol 打磨好，再考虑完全 native multimodal。

### 10.5 多模态 tokenizer 要和 agent 数据一起训练

如果目标是 multimodal agent，训练数据不能只有 image-caption 和 VQA，还需要：

- OCR -> tool call；
- screenshot -> action；
- chart -> calculation；
- video -> event reasoning；
- audio -> dialogue state；
- GUI trajectory；
- 多轮 observe/action trace；
- 失败恢复轨迹。

否则模型可能“看得懂”，但不会“用得起来”。

## 11. 一个实用 Checklist

设计多模态 tokenizer / processor 时，我会检查这些问题：

```text
文本层：
- 词表是否支持多语种、代码、JSON、tool call、坐标输出？
- media placeholder 是否是稳定 special token？
- thinking / tool / role token 是否和 chat template 对齐？

图像层：
- 单图平均视觉 token 数是多少？
- 高分辨率文档是否会被过度压缩？
- 多图顺序和引用是否稳定？
- OCR、小字、表格、图表是否保真？

视频层：
- 帧采样策略是什么？
- 时间位置如何编码？
- 长视频 token budget 是否可控？
- 音画同步如何处理？

音频层：
- 是 ASR-first，还是 audio-native？
- 语气、说话人、环境声是否会保留？
- audio token 是否和文本/视频对齐？

Agent 层：
- observation、tool_call、tool_response、action 是否在同一协议里？
- 坐标输出是否稳定？
- runtime parser 是否支持该格式？
- streaming 时 tool call 是否能增量解析？

发布层：
- tokenizer、processor、chat template、parser 是否成套发布？
- vLLM / SGLang / Transformers 是否支持？
- 多轮多模态测试用例是否覆盖？
```

## 12. 延伸阅读：Merger、对齐与长视频 Tokenizer

如果把这篇文章继续往下读，我建议不要按“模型名字”继续堆论文，而是按三个问题来读：视觉 token 如何压缩、跨模态 embedding 如何对齐、长视频和音视频流如何保持 token budget 可控。

### 12.1 Merger / Projector / Resampler 设计

这一组工作回答的是：视觉 encoder 的 patch features 到底应该怎样进入 LLM？

| 方向 | 代表工作 | 适合关注的问题 |
|---|---|---|
| 简单 projector | [LLaVA](https://arxiv.org/abs/2304.08485), [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | 一个 MLP / linear projector 能把视觉特征接到 LLM 上，说明“接口简单”本身有很强工程价值 |
| query-based bridge | [BLIP-2 / Q-Former](https://arxiv.org/abs/2301.12597) | learned queries 如何从大量视觉特征中抽取少量语义 token |
| attention resampler | [Flamingo / Perceiver Resampler](https://arxiv.org/abs/2204.14198) | 如何把任意数量视觉 patch 压到固定数量 token，并服务 few-shot 多模态上下文 |
| 动态视觉 token | [Qwen2-VL](https://arxiv.org/abs/2409.12191), [Qwen3-VL](https://arxiv.org/abs/2509.16276), [InternVL3.5](https://arxiv.org/abs/2508.18265) | 分辨率、crop、visual token budget 和任务难度如何联动 |
| 早期压缩 | [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | 在 ViT 内部或 projector 前就减少 token，如何换取端侧和低延迟部署能力 |

从 tokenizer 角度看，这些论文不是在研究文本分词，而是在研究一种更广义的 tokenization：

> 视觉信息先被 encoder 表示成 dense patch features，再由 projector / merger / resampler 决定哪些信息进入语言模型上下文。

SigLIP、CLIP 这类工作通常更适合放在“对齐”部分理解。它们本身不是 merger / projector / resampler，但很多 VLM 会把这类模型作为 vision encoder：它们决定 patch features 的语义质量，而 merger / projector 决定这些 features 怎样被压缩并接入 LLM。

因此读这类论文时，不要只看 benchmark 分数，还要看三个细节：每张图最终进入 LLM 的 token 数、压缩发生在 encoder 前后还是 LLM 前、压缩模块是否会破坏 OCR / 坐标 / 小物体等局部信息。

### 12.2 多模态对齐：不只是 contrastive learning

对齐相关论文通常不直接叫 tokenizer paper，但它们会决定视觉 token 进入 LLM 后“像不像一种语言”。如果 projector 对齐不好，模型表面上有 image tokens，实际表现可能是：

- 图像语义能说大概，但细节不稳定；
- OCR、表格、坐标和 GUI 元素容易漂移；
- 多轮对话中视觉引用无法保持一致；
- tool call 参数来自视觉信息时更容易出错。

值得继续看的方向包括：

| 方向 | 代表工作 | 和 tokenizer 设计的关系 |
|---|---|---|
| vision-language pre-alignment | [CLIP](https://arxiv.org/abs/2103.00020), [SigLIP](https://arxiv.org/abs/2303.15343), [EVA-CLIP](https://arxiv.org/abs/2303.15389) | 先把视觉 encoder 训练到图文语义空间里，降低后续 projector / instruction tuning 的对齐难度 |
| visual instruction tuning | [LLaVA](https://arxiv.org/abs/2304.08485), [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | projector 不是孤立模块，它要和视觉问答、指令数据一起训练 |
| native multimodal pretraining | [InternVL3.5](https://arxiv.org/abs/2508.18265), [Qwen3-VL](https://arxiv.org/abs/2509.16276) | 对齐从后训练前移到预训练阶段，token budget 和数据配比更重要 |
| latent / feature-space alignment | [Fill the GAP](https://arxiv.org/abs/2605.12374) | 当模型开始在视觉 latent 上推理，embedding 空间是否匹配会直接影响稳定性 |
| projector stability | [SineProject](https://cvpr.thecvf.com/virtual/2026/poster/40190) | projector 不只是一个小接头，它的数值稳定性会影响跨模态语义是否漂移 |
| VLA / action alignment | [Evo-1](https://github.com/MINT-SJTU/Evo-1) | 对机器人和 GUI agent 来说，视觉 token 最终要对齐到 action token，而不只是 caption |

这部分对预训练尤其重要。大模型团队在设计 tokenizer / processor 时，最好把 alignment 看成 token 接口的一部分：图片怎样展开、视觉 token 怎样压缩、文本里怎样引用图像区域、action 或 tool call 怎样绑定视觉证据，应该一起设计。

### 12.3 长视频 Tokenizer 与音视频压缩

长视频是多模态 tokenizer 最容易暴露成本问题的场景。图像是一张图占几百到几千个视觉 token，视频则是这个数字再乘以帧数；如果再加音频，token budget 会很快失控。

近两年的代表性工作可以按三条路线看：

| 路线 | 代表工作 | 核心思路 |
|---|---|---|
| adaptive video tokenizer | [InfoTok](https://openreview.net/forum?id=JEYWpFGzvn) | 不再用固定压缩率处理所有视频，而是按信息密度动态分配 token |
| training-free video pruning | [ForestPrune](https://openaccess.thecvf.com/content/CVPR2026F/html/Ju_ForestPrune_High-ratio_Visual_Token_Compression_for_Video_Multimodal_Large_Language_CVPRF_2026_paper.html) | 根据时空关系构造 token forest，在不重训模型的前提下做高比例剪枝 |
| audio-guided compression | [OmniZip](https://arxiv.org/abs/2511.14582) | 用音频显著性指导视频 token 保留，适合 audio-video understanding |

这些工作给 tokenizer 设计的启发是：长视频不应该被看成“很多张图片”。更合理的接口应该同时考虑：

- 静止片段可以少给 token；
- 快速运动、转场、关键动作需要更多 token；
- 音频事件可以帮助定位哪些视频片段重要；
- 长视频问答需要保留时间索引，而不只是视觉语义；
- agent 场景还要保留可回溯证据，方便后续 tool call 或 action。

不过，在大规模基座模型训练里，这些方法的落地难度不同。固定或动态帧采样、简单 token pruning、分辨率路由比较容易进入产品系统；完全重新训练 video tokenizer、把音视频 token 做统一离散化，则更适合作为中长期预训练探索。

## 13. 总结

多模态 tokenizer 的核心不是“图片怎么切 token”，而是：

> 如何把世界状态变成模型可学习、可推理、可执行的上下文协议。

传统文本 tokenizer 解决的是语言压缩问题；agentic tokenizer 解决的是动作协议问题；多模态 tokenizer 解决的是观察协议问题。

真正好的多模态模型，需要把这三件事合在一起：

| 层次 | 目标 |
|---|---|
| language tokenization | 压缩和表达语言 |
| world tokenization | 序列化图像、视频、音频、屏幕和环境状态 |
| action tokenization | 表达工具调用、坐标、轨迹和可执行动作 |

这也是为什么 unified multimodal 这类新模型值得关注。它们不是简单多支持一种输入，而是提示了一个方向：未来的 tokenizer 可能不再是文本前处理器，而是大模型理解世界、调用工具、生成动作的统一接口。
