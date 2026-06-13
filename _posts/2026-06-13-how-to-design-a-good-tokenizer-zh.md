---
layout: post
title: "如何设计一个好的 Tokenizer：面向预训练、长上下文和 Agentic 模型的实践指南"
date: 2026-06-13
tags: [Tokenizer, Pretraining]
description: "从压缩率、训练效率、工程兼容性、协议表达和长期演进能力出发，总结面向中文、代码、长上下文和 Agent 模型的 tokenizer 设计方法。"
---

如果把大模型看成一个语言操作系统，tokenizer 就是它的字节码编译器。它决定模型看到什么、如何计费、上下文能装下多少真实信息，也决定工具调用、思考过程、多模态占位这些“协议”如何被模型学习和执行。

很多人讨论 tokenizer 时只看一个指标：同一段文本切成多少 token。这个指标当然重要，但它只是表面。真正好的 tokenizer，应该同时满足压缩率、训练效率、工程兼容性、协议表达能力和长期演进能力。

这篇文章基于我对 Qwen3、DeepSeek-V3/V4、Gemma4、GLM-4.x/5.0、Kimi-K2.x、MiMo-V2.x 等新一代开源权重大模型 tokenizer 的技术报告和本地 benchmark 分析，总结一套更通用的 tokenizer 设计方法。

## 先给结论

面向 2026 年的中文、代码、长上下文和 agent 模型，我会把默认方案设成：

```text
150K-165K byte-level BPE / tiktoken-style BPE
+ 明确 role/tool/thinking/FIM/media token 规划
+ 标准 tokenizer.json 或稳定可复现的 tiktoken.model
+ 官方 chat template 和 tool parser 同步发布
+ 预留至少 256 个 token 给协议演进
```

如果目标是成本敏感的推理模型，可以考虑更克制的 128K byte-level BPE。DeepSeek-V3/V4 就是这种路线的代表：基础词表更小，embedding 和 logits 成本更低。

如果目标是覆盖优先，可以走 256K 以上的大词表路线。但要注意，大词表不等于压缩率一定更好，也不等于模型一定更强。Gemma4 的词表很大，但在我测试的中文、代码、JSON、emoji/URL 混合样本里，token 数并不是最少。

## 先补一点基础概念

在进入设计细节之前，先把几个常见词讲清楚。

### Token 是什么

大模型并不是直接看“字”或“词”，而是看 token。Tokenizer 会把文本切成一串 token id，再交给模型。

例如一句话：

```text
我喜欢 tokenizer。
```

可能被切成：

```text
["我", "喜欢", " tokenizer", "。"]
```

也可能被切成：

```text
["我", "喜", "欢", " token", "izer", "。"]
```

不同 tokenizer 的切法不同，token 数也不同。token 数越多，同样内容占用的上下文越多，训练和推理成本也越高。

### Vocab 是什么

Vocab，也就是词表，是 tokenizer 认识的一组 token。每个 token 都对应一个整数 id。

例如：

```text
"hello" -> 15339
"世界" -> 3574
"<tool_call>" -> 151657
```

词表越大，越有机会把常见片段合成更长 token，从而降低 token 数。但词表越大，模型的 embedding 和输出层也越大，训练和推理成本更高。

所以词表不是越大越好，而是要和模型规模、语料分布、部署成本匹配。

### BPE 是什么

BPE 是 Byte Pair Encoding 的缩写，可以理解为一种“从小片段逐步合并出常见大片段”的算法。

一个极简例子：

```text
初始：t o k e n i z e r
发现 en 经常一起出现，合并成 en
发现 token 经常一起出现，继续合并成 token
发现 izer 经常一起出现，合并成 izer
最终：token izer
```

真实训练中，BPE 会在大规模语料上统计哪些相邻片段最常一起出现，然后反复合并，直到达到目标词表大小。

BPE 的好处是简单、确定、速度快，而且非常适合现代 LLM。它的缺点是，如果训练语料不平衡，某些语言或格式会被切得很碎。例如只用英文网页训练出来的 BPE，往往对中文、代码、数学符号、JSON 不友好。

### Byte-level BPE 是什么

普通 BPE 通常从字符或预切分后的文本片段开始。Byte-level BPE 则从 UTF-8 字节开始。

它的最大好处是：**几乎没有 OOV**。不管输入是中文、emoji、罕见符号、乱码、控制字符，最终都能被编码，因为任何文本都能表示成字节。

这对真实互联网文本很重要。用户输入不会总是干净的英文句子，代码、日志、URL、emoji、混合语言、奇怪空白都很常见。

但 byte-level BPE 也有代价：

- token 可读性更差；
- 有些 token 可能对应不完整的字符片段；
- 调试时不如“按字/词切分”直观。

Qwen、DeepSeek、Kimi 这类新模型都明显受益于 byte-level 或 tiktoken-style BPE 的鲁棒性。

### Pretokenizer 是什么

Pretokenizer 是 BPE 真正合并之前的“预切分规则”。它决定文本先被切成哪些大块，再交给 BPE 做 merge。

例如，同样一句：

```text
hello世界123
```

一种 pretokenizer 可能先切成：

```text
["hello", "世界", "123"]
```

另一种可能切成：

```text
["hello", "世", "界", "1", "2", "3"]
```

后面的 BPE merge 会基于这些初始片段继续学习。因此 pretokenizer 会强烈影响最终 tokenizer 的行为。

对大模型来说，好的 pretokenizer 往往会显式考虑：

- 汉字和非拉丁文字；
- 英文大小写和词缀；
- 数字、日期、版本号；
- 标点、换行、空白；
- 代码标识符、路径、URL；
- emoji 和特殊 Unicode。

这也是为什么两个模型都说自己用 BPE，实际表现可能差很多。

### Special Token 是什么

Special token 是 tokenizer 里带特殊语义的 token。它们不只是文本片段，而是模型协议的一部分。

常见 special token 包括：

```text
<bos>                序列开始
<eos>                序列结束
<|user|>             用户角色
<|assistant|>        助手角色
<tool_call>          工具调用开始
</tool_call>         工具调用结束
<think>              思考内容开始
</think>             思考内容结束
<image_pad>          图像占位
```

现代 agent 模型越来越依赖这些 token。它们告诉模型：“这里是用户消息”“这里是工具调用”“这里是工具返回”“这里是思考过程”。

### Chat Template 是什么

大多数模型训练时看到的不是简单字符串，而是带格式的多轮对话。例如：

```text
<|im_start|>system
你是一个助手。
<|im_end|>
<|im_start|>user
今天北京天气怎么样？
<|im_end|>
<|im_start|>assistant
```

Chat template 就是把结构化 messages：

```json
[
  {"role": "system", "content": "你是一个助手。"},
  {"role": "user", "content": "今天北京天气怎么样？"}
]
```

转换成模型实际输入字符串的模板。

这件事非常重要。很多 agent 模型失败，不是因为模型不会用工具，而是因为部署时 chat template、tool parser 和训练格式不一致。

## Tokenizer 不是预处理工具，而是模型协议

Tokenizer 至少影响八件事。

1. **有效训练量**

同样 10T 字符，用不同 tokenizer 会得到不同 token 数。token 数不同，训练预算、数据重复率、学习节奏都会变。

2. **有效上下文长度**

标称 128K context 只是 token 数。中文、代码、JSON、日志、工具轨迹到底能塞多少真实内容，取决于 tokenizer 的压缩率。

3. **模型参数和推理成本**

词表越大，embedding 和 LM head 越大。粗略说：

```text
embedding 参数量 ≈ vocab_size × hidden_size
```

如果输出层不共享权重，还要再付一份类似成本。每步 logits 计算也随词表大小增长。

4. **多语种公平性**

一个英文友好的 tokenizer，可能让中文、日文、韩文、泰文承担很高 token 成本。多语种模型不能只看英文压缩率。

5. **代码能力**

缩进、换行、路径、包名、snake_case、camelCase、JSON、YAML、XML、日志和错误栈都会影响代码训练效率。

6. **长上下文能力**

长上下文模型的瓶颈不只是 RoPE、attention 或 KV cache。Tokenizer 切得碎，标称上下文再长也会被浪费。

7. **Agentic 协议**

现代模型要学会使用工具、解析工具返回、输出思考过程、做多轮规划。role、tool call、tool response、thinking、media、FIM 这些 token，是 agent 行为的语法。

8. **安全和鲁棒性**

如果用户输入里出现 `<tool_call>` 或 `<think>` 这样的字符串，模型和 parser 会不会误判？特殊 token 是否能被注入？这不是后处理能完全解决的问题。

所以，tokenizer 应该在预训练前设计好，而不是训练后临时补。

## 怎么评估一个 Tokenizer

我的 benchmark 思路很简单：用同一组样本喂给每个 tokenizer，看它们把相同内容切成多少 token。

样本需要覆盖常见真实场景，而不只是普通自然语言：

- 中文说明文本
- 英文技术文本
- 中英混排
- Python 代码
- JSON 工具调用
- 数学表达式
- emoji + URL
- 缩进、tab、连续空白

核心指标有三个：

```text
tokens = tokenizer.encode(sample) 的 token 数
chars/token = 字符数 / token 数
bytes/token = UTF-8 字节数 / token 数
```

`tokens` 越少，表示同样文本占用上下文越少。`chars/token` 和 `bytes/token` 越高，表示压缩率越好。其中 `bytes/token` 对中英文混合比较更公平，因为中文字符通常占 3 个 UTF-8 字节。

但这只是 tokenizer 层面的评估，不是模型能力评测。Tokenizer 压缩率高，不代表模型生成质量、工具调用成功率或推理能力一定更好。正式预训练选型时，应该在目标训练语料上做更大规模的分桶 benchmark，并配合小规模预训练 ablation。

## 从最新模型看设计趋势

观察 Qwen3、DeepSeek、GLM、Kimi、MiMo、Gemma4，可以看到一个明显趋势：

- **128K** 正在成为新一代大模型 tokenizer 的成本友好下限。
- **150K-165K** 是中文、代码、agent 综合模型的主流甜点区间。
- **260K+** 更像覆盖优先路线，而不是压缩率优先路线。

在一组覆盖中文、英文、代码、JSON、数学、emoji/URL、空白缩进的样本上，我看到的现象大致是：

- Qwen3、MiMo-V2.x、GLM-5、Kimi-K2.x 都处在压缩率第一梯队。
- DeepSeek-V3/V4 token 数略高，但基础词表只有 128K，是更克制的工程折中。
- Gemma4 词表最大，但在这组代码/混合文本样本上并不省 token。
- Kimi-K2.x 中文压缩率很强，但工程上依赖 `tiktoken.model` 和自定义 tokenizer 实现，而不是标准 `tokenizer.json`。
- MiMo-V2.x 与 Qwen tokenizer 行为高度接近，说明 tokenizer 兼容性也可以成为工程资产。

这些观察说明一件事：**词表大小只是容量，不是质量本身。训练语料、pretokenizer、merge 分布、special token 规划和服务协议同样重要。**

## 主流模型 Tokenizer 设计观察

下面用几个新一代开源权重模型做案例。这里不讨论模型整体能力，只看 tokenizer 设计。

### Qwen3：中文、代码和 agent 场景的均衡方案

Qwen3 使用 Qwen 系 tokenizer，技术报告中明确提到 byte-level BBPE，词表规模约 151K 级。它的实际表现很符合这个定位：中文、代码、JSON、URL、emoji 的综合压缩率都比较稳。

Qwen3 的另一个特点是 tokenizer 协议非常完整。它不仅有常规 chat token，也有：

- `<|im_start|>` / `<|im_end|>`
- `<tool_call>` / `</tool_call>`
- `<tool_response>` / `</tool_response>`
- `<think>` / `</think>`
- FIM 相关 token
- vision/image/video pad token

这说明 Qwen3 的 tokenizer 不是只为普通对话设计，而是为 thinking、tool use、代码补全、多模态占位一起设计的。

优点是生态友好，文件形态标准，适合 HF、vLLM、SGLang 等工具链。缺点是词表比 128K 更大，embedding 和 logits 成本高于 DeepSeek 这类克制路线。

### DeepSeek-V3/V4：128K 的成本克制路线

DeepSeek-V3 技术报告明确说明使用 128K byte-level BPE，并且修改了 pretokenizer 和 tokenizer 训练数据来优化多语种压缩。

这是一条很有代表性的路线：不把词表扩到 150K 或 260K，而是在 128K 内尽量做好多语种和结构化文本。它的好处是成本低。词表更小，embedding、LM head、logits 计算都更轻。

在我的样本里，DeepSeek-V3/V4 的 token 数略高于 Qwen/GLM/Kimi/MiMo 第一梯队，但差距不大。换来的好处是更小的基础词表。

DeepSeek-V3 的 chat template 里直接包含 user、assistant、tool calls、tool outputs 等协议边界。V4 本地 tokenizer 与 V3 基础 BPE 行为一致，但 added-token 区域更大，说明基础 tokenizer 可以稳定复用，而 agent 协议还会继续演化。

适合场景：成本敏感、推理部署敏感、想保持 128K 词表但又需要多语种和代码能力的模型。

### Gemma4：覆盖优先的大词表路线

Gemma4 的本地 tokenizer 文件显示 BPE 词表达到 262,144，是这组模型里最大的。

大词表的直觉优势是覆盖更广，尤其对长尾字符、多语种、特殊符号、多模态生态继承可能有帮助。但大词表也有明显代价：embedding 和输出层更大，低频 token 学习更难，推理时 logits 也更重。

在我的混合样本里，Gemma4 的总 token 数反而最多，尤其代码样本较贵。这说明大词表不天然等于高压缩率。它可能是为了覆盖更多场景，而不是针对中文/代码/JSON 压缩率做极致优化。

Gemma4 也有 tool call、tool response、think/channel 相关 token，并和 response schema / processor 绑定。这种设计更像“完整协议栈”，只看 tokenizer 文件是不够的，还要看 processor 和 parser。

适合场景：覆盖优先、多模态生态继承、长尾符号和多语种广覆盖。若目标是成本敏感的代码/agent 文本压缩，它未必是最佳选择。

### GLM-4.x / GLM-5：150K 级 BPE 的稳定演进

GLM-4.5 和 GLM-4.7 的 tokenizer 在本次解析字段和 benchmark 结果上保持一致，说明同系列内部比较稳定。GLM-5 的词表略增，进入 154K 级别，仍属于 150K BPE 设计家族。

GLM-4.x 的 agent token 很直接：

- `<|system|>`
- `<|user|>`
- `<|assistant|>`
- `<think>` / `</think>`
- `<tool_call>` / `</tool_call>`
- `<tool_response>` / `</tool_response>`
- `/nothink`

这种设计的好处是协议清晰，thinking 和 tool use 都被显式 token 化。它在代码和 JSON 样本上表现也很强，GLM-5 在我的样本中和 Qwen/MiMo/Kimi 并列第一梯队。

风险在于，不同服务框架必须正确实现对应 chat template 和 parser。Tokenizer 文件里有 token，不代表部署时自然会按训练格式组织上下文。

适合场景：中文、代码、agent 综合模型，尤其适合希望在 150K 级词表内保持较好压缩率的路线。

### Kimi-K2.x：压缩率强，但工程接入更挑剔

Kimi-K2.x 使用 `TikTokenTokenizer` 和 `tiktoken.model`，基础词表为 163,840。它不是常规 `tokenizer.json` 路线，而是更接近原生 tiktoken-style BPE。

它的优点是压缩率强。在我的样本里，Kimi-K2.x 总 token 数进入第一梯队，中文样本 token 数最低。

Kimi 的 agent token 设计非常细：

- `<|im_user|>`
- `<|im_assistant|>`
- `<|im_system|>`
- `<|tool_calls_section_begin|>`
- `<|tool_call_begin|>`
- `<|tool_call_argument_begin|>`
- `<|tool_call_end|>`

Kimi-K2.5 又增加了：

- `<|media_begin|>`
- `<|media_content|>`
- `<|media_end|>`
- `<|media_pad|>`
- `<think>` / `</think>`

这说明 Kimi 的路线很 agentic：工具调用边界、参数边界、media、thinking 都被纳入 tokenizer 协议。

缺点是工程接入更挑剔。没有标准 `tokenizer.json` 时，推理框架需要支持 tiktoken model 或远程 tokenizer 代码。对线上服务来说，这不是小事。

适合场景：追求中文/agent/coding 压缩率，且团队能控制 tokenizer runtime 和 parser 的系统。

### MiMo-V2.x：Qwen 兼容路线上的 agent 强化

MiMo-V2/V2.5 本地 tokenizer 显示为 Qwen2Tokenizer 风格，基础 BPE 词表大小与 Qwen3 一致。在我的样本里，MiMo 与 Qwen3 的 token 数完全相同。

这说明 MiMo 的 tokenizer 选择不是靠新切分算法差异化，而是选择了一个成熟兼容的 tokenizer 基础，再在 chat template、system prompt、tool parser、thinking 规则和后训练数据上强化 agent 能力。

MiMo-V2.5 的 chat template 很重，包含默认 system prompt、工具说明、参数渲染、thinking 控制和多轮工具调用处理。它也强调长轨迹和大量 tool calls。

这给 tokenizer 设计一个很重要的启发：**同样的基础 BPE，可以通过不同 chat template、parser 和训练数据，形成不同 agent 行为。**

适合场景：希望复用 Qwen 生态兼容性，同时强化长轨迹 agent、工具调用和 thinking 控制的模型。

### 小结：不同路线的取舍

| 路线 | 代表模型 | 核心优点 | 主要代价 |
| --- | --- | --- | --- |
| 128K byte-level BPE | DeepSeek-V3/V4 | 成本低，工程克制 | 中文/JSON/URL 压缩率可能略逊 |
| 150K 级 Qwen2Tokenizer/BPE | Qwen3、MiMo、GLM | 中文/代码/agent 综合表现强，生态较友好 | 成本高于 128K |
| 164K tiktoken-style BPE | Kimi-K2.x | 压缩率强，agent token 细 | runtime/parser 依赖更强 |
| 262K 大词表 BPE | Gemma4 | 覆盖广，长尾友好 | embedding/logits 成本高，未必更省 token |

如果只看 tokenizer 设计，我会这样总结：

- **Qwen3 / MiMo**：最省心的中文、代码、agent 综合选择。
- **DeepSeek**：最克制的 128K 工程折中。
- **GLM-5**：150K 级路线里压缩率很强的选择。
- **Kimi-K2.x**：压缩率和 agent 协议都强，但工程要求高。
- **Gemma4**：覆盖优先，适合从整体生态和多模态角度评估，而不是只看 tokenizer 压缩率。

## 词表大小怎么选

可以先用下面这个表做初判。

| 词表大小 | 适合场景 | 风险 |
| --- | --- | --- |
| 64K 以下 | 端侧、小模型、英文为主 | 中文、多语种、代码、agent 文本会明显吃亏 |
| 128K | 成本敏感的大模型，通用对话，推理部署 | 如果中文/代码/agent 占比很高，可能不够省 token |
| 150K-165K | 中文、多语种、代码、agent 综合模型 | embedding/logits 成本高于 128K |
| 256K+ | 覆盖优先、多语种长尾、多模态生态继承 | 成本高，低频 token 学习不足，未必更省 token |

我倾向于把 **150K-165K** 视为中文/代码/agent 模型的默认起点。这个区间比 128K 更照顾中文和结构化文本，又没有 256K 那么重。

当然，最终决策不能靠经验值。应该训练多个候选 tokenizer，例如 128K、151K、164K、256K，然后在目标语料上比较：

- 中文 tokens/char
- 英文 tokens/char
- 多语种 token inflation
- 代码 tokens/line
- JSON/tool call tokens/call
- URL、日志、数学、空白缩进的表现
- embedding 和 logits 成本
- 小规模预训练 loss 和下游效果

## 算法路线怎么选

新一代模型主要有几类路线。

### Byte-level BPE

优点是无 OOV，任意 UTF-8 输入都能编码，适合真实互联网文本、多语种、emoji、代码和奇怪符号。

缺点是 token 可读性较差，某些 token 是字节片段，调试和人工分析不直观。

### tiktoken-style BPE

优点是快，适合大规模服务，也适合通过正则 pretokenizer 精细控制中文、数字、空白、标点等结构。Kimi-K2.x 就是这一类。

缺点是运行时需要支持。如果没有标准 `tokenizer.json`，很多工具链需要额外适配。

### Qwen2Tokenizer 风格

Qwen3、MiMo-V2.x 都采用类似路线。优点是文件形态标准，`vocab.json`、`merges.txt`、`tokenizer.json`、`tokenizer_config.json` 齐全，HF/vLLM/SGLang 生态接入比较友好，中文和代码表现也不错。

代价是词表比 128K 更大，成本高于 DeepSeek 这类克制路线。

### 大词表 BPE

Gemma4 这类 262K 级别词表覆盖更广，但成本也更高。它适合“覆盖优先”或生态继承，不一定适合追求代码/JSON 压缩率和部署成本的场景。

## Tokenizer 训练语料怎么配

Tokenizer 的语料配比，会决定谁更省 token。不要只拿普通网页文本训练 tokenizer，尤其不要只拿英文网页。

建议至少准备这些桶：

- 中文：新闻、百科、论坛、社媒、技术文档、繁体、中英混排。
- 英文：网页、书籍、论文、技术文档。
- 多语种：日韩、东南亚语言、欧洲语言、阿拉伯语、印地语等。
- 代码：Python、JavaScript、TypeScript、Java、C++、Go、Rust、Shell、SQL。
- 结构化文本：JSON、YAML、XML、Markdown、HTML、日志、protobuf、OpenAPI schema。
- 数学：LaTeX、Unicode 数学符号、公式、长数字。
- Agent 数据：role 消息、tool schema、tool call、tool response、错误栈、多轮轨迹。
- 多模态占位：image/video/audio/media placeholder 和相关 caption。

关键点是：**如果模型未来要做 agent，tool call 和 tool response 必须进入 tokenizer 训练语料**。否则工具参数、XML/JSON 包裹、错误日志会被切得很碎，agent 轨迹会变贵。

## Pretokenizer 很重要

同样是 BPE，pretokenizer 不同，结果会差很多。

好的 pretokenizer 至少要考虑：

- 中文连续汉字不能退化成大量 byte 级碎片。
- 数字要合理分组，避免长数字、日期、版本号完全不可控。
- 换行、tab、缩进要稳定，代码模型尤其依赖这些结构。
- URL、路径、包名、snake_case、camelCase 要在训练语料中足够常见。
- 特殊协议 token 不应被普通文本轻易触发。

Kimi 的 tokenizer 就显式使用了针对 Han 字符、英文大小写、数字、标点、换行和空白的正则分支。DeepSeek-V3 技术报告也强调修改 pretokenizer 和 tokenizer 训练数据来优化多语种压缩。这些细节往往比“词表大小是多少”更关键。

## Agentic 模型需要什么 Token

Agentic 能力不是只靠后训练数据堆出来的。Tokenizer 需要给模型提供一套稳定、低成本、可解析的协议。

至少应该考虑这些 token：

- 对话角色：system、user、assistant、tool。
- turn 边界：message start/end、EOT。
- 工具调用：tool_call begin/end、function name、arguments begin/end。
- 工具返回：tool_response begin/end。
- 思考控制：think begin/end、no_think、thinking mode switch。
- 代码补全：FIM prefix/middle/suffix/pad。
- 多模态：image/video/audio/media begin/content/end/pad。
- 保留区：给未来协议演进预留空间。

不同模型已经在这样做：

- Qwen3 / MiMo 使用 `<|im_start|>`、`<|im_end|>`、`<tool_call>`、`<tool_response>`、`<think>`、FIM token、vision/image/video pad，并通过 chat template 支持 thinking/non-thinking 和工具调用。
- DeepSeek-V3 的 chat template 直接编码 user、assistant、tool calls、tool outputs 等边界。
- GLM-4.x 显式包含 `<|system|>`、`<|user|>`、`<|assistant|>`、`<think>`、`<tool_call>`、`<tool_response>`、`/nothink`。
- Kimi-K2 把 tool call section、tool call begin、argument begin 等边界设计得非常细；Kimi-K2.5 又增加 media 和 thinking token。
- Gemma4 也有 tool call、tool response、think/channel 相关 token，并和 processor/schema 绑定。

这说明 tokenizer 已经不只是文本切分器，而是 agent 协议的一部分。

## Agentic Tokenizer 的几个原则

### 工具调用边界必须成对

例如：

```text
<tool_call> ... </tool_call>
<tool_response> ... </tool_response>
```

或者更细的 begin/end/argument begin 分层。边界不清楚，多工具、多轮、流式输出时很容易解析错。

### 参数区域必须保真

工具参数通常是 JSON、XML、TypeScript-style schema 或其他结构化文本。空格、换行、引号、转义符都要稳定，不能被 chat template 意外改写。

### Thinking 和最终答案要可分离

Qwen、MiMo、Kimi、GLM 都已经把 `<think>` 作为协议的一部分。训练和服务时必须明确：

- thinking 内容是否展示给用户
- thinking 内容是否放回多轮历史
- non-thinking 模式是否仍输出空 `<think></think>`
- parser 如何区分 reasoning 和 final answer

### Special token 不是安全边界

如果用户文本里出现 `<tool_call>` 字符串，模型或 parser 可能误判。安全做法是 role 隔离、escaping、结构化消息对象和严格 parser，而不是只靠 prompt 约束。

### Chat template 是 tokenizer 的一部分

很多 agent 失败不是模型不会用工具，而是模板错了。训练时模型看到一种格式，部署时 serving 层用了另一种格式，tool call 行为自然会漂移。

所以 tokenizer 发布时，应该同步发布：

- tokenizer 文件
- chat template
- tool parser
- thinking parser
- media processor
- serving 示例

## 不要在预训练后随便换 Tokenizer

预训练完成后更换 tokenizer 通常是灾难：

- embedding 和 LM head 对不上。
- 旧 token 的语义分布被破坏。
- chat template、tool parser、special token ID 都会变化。
- SFT、RL、蒸馏数据需要重新格式化。

更现实的做法是：预训练前把 tokenizer 设计好，预留足够 special token；后续只在保留区增加少量协议 token，并尽量保持基础 BPE 不变。

DeepSeek-V4 和 Kimi-K2.5 都体现了这个趋势：基础 tokenizer 大体稳定，但 added-token 协议继续演进。

## 常见坑

### 只看词表大小

大词表不必然更省 token。词表大小只是容量，真正表现取决于训练语料、pretokenizer、merge 分布和协议 token。

### 只看平均 token 数

平均值会掩盖问题。一个 tokenizer 可能英文很好、中文很差；代码很好、JSON 很差；普通对话很好、tool call 很差。必须分桶评估。

### 训练后临时加 agent token

临时加 `<tool_call>`、`<think>`、`<image>` 这类 token，embedding 往往是随机初始化，模型未必能稳定使用。更好的方式是在预训练前预留，并让相关格式进入训练数据。

### 忽略运行时生态

Tokenizer 不是只给训练用，也要服务于推理。vLLM、SGLang、Transformers、TGI、TensorRT-LLM 是否支持？是否需要 `trust_remote_code`？是否有标准 `tokenizer.json`？这些都会影响落地。

### 忽略安全注入

工具 token、thinking token、media token 都可能出现在用户输入中。parser 必须区分用户文本和协议 token。

## 一个实用检查清单

进入主预训练前，建议逐项确认：

- 中文、英文、多语种、代码、数学、结构化文本、agent 轨迹都有独立采样桶。
- 至少训练 128K、150K、164K 等多个候选 tokenizer。
- 对每个候选跑分桶压缩率，而不是只看整体平均。
- 单独统计中文、代码、JSON/tool call、URL、数学符号、空白缩进。
- 检查 byte fallback 或未知字符处理，确保任意 UTF-8 输入可编码。
- 检查 normalization 规则，避免全角/半角、emoji、组合字符被破坏。
- role token、turn boundary、BOS/EOS/PAD/UNK 语义明确。
- tool call 和 tool response 有稳定边界。
- thinking token 与 final answer 边界清楚。
- FIM token、media token 是否需要提前预留。
- added token 的 `special=true/false` 语义明确。
- chat template、tool parser、tokenizer 文件、serving 示例成套发布。
- 所有数据 pipeline 已经用最终 tokenizer 重新统计长度。
- tokenizer 版本号、hash、文件列表写入训练配置。
- 一旦进入主预训练，不再修改基础 vocab/merges。

## 总结

一个好的 tokenizer，不是“把文本切得尽量短”这么简单。它应该同时满足：

- 对目标语料压缩率高
- 对中文、多语种、代码和结构化文本公平
- 词表大小和模型成本匹配
- 任意输入可编码
- agent 协议清晰、可解析、可防注入
- 与训练数据、chat template、tool parser、serving 框架一致
- 预留未来扩展空间

如果你正在为一个中文、代码、长上下文、agentic 模型设计 tokenizer，我的建议是：

先从 150K-165K byte-level BPE / tiktoken-style BPE 开始，认真设计 role/tool/thinking/FIM/media token；然后用目标语料做分桶 benchmark，再用小规模预训练验证 loss 和下游表现。不要只看词表大小，也不要把 tokenizer 当成训练完之后还能轻松替换的外部零件。

Tokenizer 是模型的第一层架构。设计得好，后面的训练和部署都会顺；设计得草率，问题会一路传到预训练、SFT、RL、agent parser 和线上服务。
