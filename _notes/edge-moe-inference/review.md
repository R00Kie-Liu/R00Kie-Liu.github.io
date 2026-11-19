# 第六篇 Expert Streaming 补充与写作检阅

日期：2026-11-19。发布日期仍为 2026-07-26，URL 与系列编号不变。

## 结构与范围

将 Expert Streaming 纳入第 2 节，按卸载数据路径、slot pool 与专家算子、LES 跨层预取、缓存及主机调度开销展开。案例范围增加 Mac/MPS，保留手机 CPU 与 Hexagon NPU 的设备和实验条件。后文继续沿投机验证、专家并集与 BigMoMo 多级搬运推进，未扩写投机树构造规则。

按用户要求，正文不介绍 PR 状态或 API 合入流程，也没有添加安装教程。将整篇作为同一次编辑检阅：收紧引言、带宽估算后的重复解释、LES 结果归因、投机解码转场与结尾；保留影响实验解释的限定。

## 来源与事实核对

- 原文：https://huggingface.co/blog/moe-expert-streaming ，2026-11-18 发布。
- 项目细节：https://tarekziade-moe-expert-streaming.static.hf.space/details.html 。
- 当前结果与实现：https://tarekziade-moe-expert-streaming.static.hf.space/index.html 、https://github.com/tarekziade/moe-expert-streaming 。核对的 README blob SHA 为 `11ebd2466d9813078028a8dec864e5230237a553`。
- 区分独立显存上的 CPU expert compute、主机权重供 GPU 使用，以及文件支持的设备缓存；统一内存设备仍需为全部驻留权重付出物理内存。
- slot pool 共享槽位存储，各层专家参数独立；真实 expert ID 与 slot ID 重映射时保持路由分数配对。未量化原型复用 `grouped_mm`，量化路径使用对应的索引与分块 kernel，未将所有后端表述成同一个算子。
- 保留文件读取与物理 SSD I/O 的区别。43GB 专家文件、12GiB 专家缓存和约 51GB n-gram 表分别说明，未将缓存预算等同于总内存占用。
- 约 13.9 token/s decode 与 228 token/s prefill 引用原文及当前项目结果，属于 M5/32GB、指定量化和专用 kernel 的整套实现；没有将其归因于单独的缓存或预取。OLMoE 的 10.1、34.6、38.1 单列为缓存与算子原型测量。
- 跨两层预测约 11.1、关闭后约 11.7 的消融来自 GGUF 主干配置；原文图中的单专家猜测预读作为另一条路径说明，未将前者描述为所有平台的预取结论。
- prefill 的缓存扫描、冷端安装、MPS command buffer 提交与 shared expert 覆盖主机管理分别解释，避免把命中率当作完整延迟。

## 写作与验证

使用仓库 `polish-technical-blog-zh` skill 及中文、反防御参照，人工核对问题动机、概念定义、论证顺序、数字归属和上下文转场。正文机械审稿无命中；该结果不代表 AI 检测概率。

原有 3 个展示公式、3 个代码块和 19 个参考链接目标均保留；5 张旧图保持相对顺序，新增原文 SVG 动画后共 6 张图。图片来源、原始文件哈希与动画保留情况见 `figure-sources.json`。

Jekyll 构建通过。1440×1000 与 390×844 下，6 张图片全部解码，14 处数学渲染无错误，6 张表格未造成整页横向溢出，未出现脚本错误。检查 26 个站内链接与 11 个片段链接，32 个旧正文 ID 均保留且无重复。原文 SVG 的动画仍正常运行，并已人工检查桌面、手机截图与图文对应。
