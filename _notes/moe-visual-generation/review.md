# 第十篇重写与检阅记录

日期：2026-11-15。正文仍使用 2026-07-23 的发布日期及原 URL，标题改为“MoE我们喜欢你（十）：视觉生成与原生多模态中的专家设计”。

## 范围与结构

按用户确认的范围，聚焦图像/视频生成、噪声阶段与空间位置上的专家选择，以及原生多模态中的文本/视觉处理。未加入扩散语言模型，未新增相关番外。

正文从生成过程中的容量需求出发，依次讨论整模型阶段切换、token 选择与调用量、语义引导、head-local 粒度、文本/视觉路径、理解/生成任务以及控制接口。取消独立应用清单与末尾通用评测清单。MOVA 接入 Wan 的阶段专家讨论，LingBot 接入 token MoE。原 MAGI-2 和 Worldscape 的机制、原有实验规模与图保留；删除“视觉生成比 LLM 有更强 kernel 压力”等没有比较证据的概括。

## 主要事实核对

- HunyuanImage-3.0 v2：文本 next-token prediction，图像 diffusion/flow prediction，共享 Hunyuan-A13B MoE 主干；使用广义因果 attention。修正旧文“自回归图像 MoE”的分类，按 §3 与 §5.3 解释模态路由统计。
- HiDream-I1：核对官方 `transformer_hidream_image.py`，固定提交 `5f92bab45f1dfb1e794ee357286a5b837eaf4400`。双流 `ff_i` 为 MoE、`ff_t` 为普通 SwiGLU；单流将拼接后的视觉和文本 token 交给同一 MoE。正文未将“双流、单流都有 MoE”改写成“所有阶段的所有模态都使用 MoE”。
- MOVA v1：Wan2.2 I2V A14B 视频塔、1.3B 音频塔和 2.6B Bridge；按 §4.5 说明交替优化噪声阶段专家，音频塔和 Bridge 每步更新。
- EC-DiT v2：每专家容量与每 token 调用数量分开；四位置示例明确为教学示例。
- DiffMoE v1：训练使用 batch-level token pool；capacity predictor 学习选择标签，推理按预测及专家阈值调整选择。未声称推理固定复制训练 batch 的竞争结果。
- SplitMoE v1：从 Wan2.2 low-noise checkpoint 改造部分 FFN；20 semantic、80 generic，逐位置 2+6，并有 shared expert。干净 VAE 原型指导用于训练；generic 使用 loss-free balance，semantic 不套用相同均匀约束。专家贡献变化未表述为每步改变 top-k 数量。
- PixelMoE v3（2026-11-05）：标准 TC、细粒度专家、shared 与 bias balance；分层分辨率改变 token 数，与 top-k 专家选择区分。与 SplitMoE 按不同表示空间和实验配置比较，未排列跨论文质量/速度名次。
- MoT v2：模态专用投影、norm 和 FFN 配合全局 attention，未将参数隔离等同于信息隔离。
- Beyond Language Modeling v1：区分模态、任务与 timestep 的统计。CV 约 0.15 表示选择频率变化小，不推导逐 token 专家 ID 不变；理解/生成选择率相关性保留在被测模型范围内。
- Symbiotic-MoE v2：理解组含 Text+ViT，生成组处理 VAE，96/32 来自已有专家统计；shared expert 表示共用变换与训练参数，不表述成 FFN 自身跨 token 传递信息。

所有上述论文链接位于正文参考资料；新增图片为论文原始图文件，共 11 张，来源、版本和图片路径见 `figure-sources.json`。原有 6 张图保留，正文按出现顺序编号 1–17，无装饰性“图源/截取自”说明。删除旧文未在正文展开的 RAPHAEL 书目，未引入额外论文清单。

## 润色与验证

按仓库 `polish-technical-blog-zh` skill 深改，人工检查动机、机制、转场与证据范围，删除重复的预防性辩解和模型宣传式概述。机械审稿最终无命中，未把该结果解释为 AI 检测概率。

Jekyll 构建通过。1440×1000 与 390×844 下检查 17 张图片解码、16 处数学渲染、1 张表格，MathJax 与脚本无错误，整页无横向溢出；长公式拆行。三篇涉及导航的文章共检查 79 个站内链接、27 个片段链接。原章节锚点以别名保留。检查脚本与截图位于执行环境 `/tmp/moe-visual-edit-20261115/`，不作为站点内容发布。
