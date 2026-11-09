# 投机树验证的专家并集与候选选择

资料日期：2026 年 11 月 9 日。这里的 node 指 speculative tree 中的候选 token 节点。

## 树验证与 Block 路由的共同问题

一次验证选择候选树 $\mathcal T$，第 $\ell$ 层节点 $v$ 使用的 routed experts 为 $S_{\ell,v}$。本轮触及的专家并集为：

$$
U_\ell(\mathcal T)=\bigcup_{v\in\mathcal T}S_{\ell,v}.
$$

专家计算覆盖整棵保留的树，最终提交的 draft tokens 通常只来自其中一条路径。未提交分支的专家计算和权重准备也已经发生。不同分支的 hidden states 不同，候选置信度接近时，路由成本仍可能差很多。

对两个同为 top-2 的候选，设现有专家并集为 $\{0,1\}$。候选 A 使用 $\{0,1\}$，候选 B 使用 $\{2,3\}$：两者各增加两次 token–expert 调用，但 A 没有新增专家，B 新增两个。若选择更深的节点，还必须计入尚未选中的祖先，保持树的祖先闭合性。

设加入节点及缺失祖先的集合为 $P(v)$，其边际独立专家权重需求可写为：

$$
\Delta W(v\mid\mathcal T)=
\sum_\ell\sum_{e\in U_\ell(P(v))\setminus U_\ell(\mathcal T)}W_{\ell,e}.
$$

这个量描述新增的专家权重集合。对某一卸载边界，实际新增加载还要扣除已经驻留的专家，重复加载与逐出也会影响流量。添加候选即使不增加专家并集，仍会增加输入行计算和其它逐节点工作。

验证前无法直接知道所有候选的真实 target routing。候选选择需要 draft 路由代理、额外预测器或平均成本 profile；论文的测量若预先获取了 target routes，应单独交代。

## 方法比较

| 工作 | 主要决策 | 专家成本来自哪里 | 对 verifier 的影响 |
|---|---|---|---|
| [S²-MoE](https://arxiv.org/html/2608.15018v2) | 用收益/边际成本控制扩树，再以共享偏好促进专家复用 | self-draft 路由代理与校准的新增专家成本；verify 中聚合 target gating logits | reuse-aware bias 可以改变 target 路由 |
| [EcoSpec](https://arxiv.org/html/2607.12696v1) | 在 draft 后按路径概率与边际专家成本选择候选 | 训练好的 expert predictor；动态记录已覆盖专家 | 保留 target router 与验证规则 |
| [EVICT](https://arxiv.org/html/2605.00342v2) | 按累计置信度排序，选择预计提交收益/成本最好的前缀规模 | 初始化时测量不同验证节点数的平均整轮成本 | 保留 target 路由；不在线预测本轮具体专家并集 |
| [SpecFit](https://arxiv.org/html/2610.15417v1) | 在有界候选池中搜索祖先闭合子树，并控制 drafting depth | 逐层预测专家并集与冻结的节点数/专家数成本系数；当前实验使用预先取得的 route atlas | 选中节点由未修改的 target 完整验证 |
| [MoE-Spec](https://arxiv.org/html/2602.16052v1) | 每层只允许固定预算的 shortlist | 静态校准、聚合 router scores 或 oracle 排序对照 | 截断或替代池外专家，改变 target 计算 |
| [AcceptMoE](https://arxiv.org/html/2608.02989v1) | 按提交概率加权需求，自适应选择 eligible experts，卸载时结合驻留状态裁剪 | target-router scores、离线估计的提交概率与缓存状态 | 改变 expert eligibility 及输出分布 |
| [DraftExpert](https://arxiv.org/html/2607.24434v1) | 固定占用 self-drafter、置信度与扩张联合截断、专家预取 | router-agreement 蒸馏得到的 draft 路由预测 | 完整 target 验证；主要以多 token block 展开，不应一概称为树搜索 |

### S²-MoE 的扩树与路由 Bias

S²-MoE 第 4.1 节用路径置信度估计候选接受收益。成本同时计入继续 drafting 的成本和新增专家带来的验证成本；新增专家以 draft-time 路由作代理，并换算为硬件校准的时间。收益超过成本时继续扩展。

第 4.2 节在 verify 中对 target gating logits 做跨候选的置信度加权聚合，选出至多 `cap` 个 preferred experts，再给这些专家增加 bias。边界附近的选择因此更容易向共有专家集中，每个位置仍执行 top-k。

`cap` 限制的是获得 bias 的专家数。池外专家仍然可以进入 top-k，因此它没有提供 `|U| <= cap` 的硬保证。它与 LLaDA2.2 的“池外屏蔽”应分开比较。完整方法改变 target 路由，原分布的精确性不能仅靠保留接受步骤来推出。

### EVICT 与 EcoSpec 选择的对象不同

EVICT 第 3.2 节先按 drafter 的累计分数排序，再从这条排序中选择最划算的前 k 个节点。因子节点累计分数不超过祖先，前缀自然满足祖先闭合。其成本表反映节点数增加后的平均时间，包含 MoE 专家扩张效应，但不区分两个等大小候选集合在本轮触发的具体专家。

EcoSpec 则预测候选路径的专家集合，选择时维护已经覆盖的专家记录。同样数量的候选可以因专家重叠不同而被区别对待。记录已覆盖专家的 Global Expert Buffer 是集合元数据，不能视为权重驻留缓存。

### SpecFit 的集合优化与实验条件

SpecFit 将预计提交 token 数除以预计整轮时间作为当前轮的选择目标。时间模型包含节点数量项和逐层专家并集项；它在当前有界候选池中枚举合法子树，联合选择节点身份和数量。其精确搜索针对该候选池与估计目标，不代表任意大树或整个解码过程的全局吞吐最优。

固定节点路由估计后，专家并集是单调次模的 coverage function：已选集合越大，一个新增节点引入的新专家通常越少。其共享效应要求比较候选组合，单独给每个节点打一个不随集合变化的成本分数会遗漏这部分结构。

论文第 3.3、4.6 节写明，当前实验的路由来自参考 prefix 上对完整候选池的单独 target 查询，即 route atlas；获取过程未计入请求计时。文中还测试模拟预测误差并估计 predictor 成本，在线路由预测器训练留给后续。其结果支持集合选择器的设计，但尚不能作为包含在线预测器的完整部署收益。

## 相邻路由约束从路径推广到树

假设每个节点选择 k 个专家，且每条父子边最多引入 r 个新专家：

$$
|S_v\setminus S_{\operatorname{parent}(v)}|\le r.
$$

一条含 d 个节点的路径有上界 `k + (d - 1)r`，而一棵含 N 个节点的树只能据此得到 `k + (N - 1)r` 的上界，二者还要与专家总数 E 取最小值。兄弟节点可以分别引入不同专家，所以树深度小不保证整树工作集小。

例如根使用 `{0,1}`，两个孩子分别使用 `{0,2}`、`{1,3}`。每条边只引入一个新专家，每条根到叶路径各覆盖 3 个专家，整树却覆盖 4 个。若目标是限制整个 verify 的工作集，需要跨分支的共享池、全树聚合或候选集合选择。将连续 token 的 coactivation 规则用于 tree attention 时，父子关系与张量存储顺序也必须分清。

## 与新系列的衔接

这一主题放在 dLLM block routing 之后，可以直接比较多位置专家池与多分支专家池：二者都支付专家并集，但 speculative tree 还需要考虑祖先依赖、低提交概率的旁支和候选选择。后续的相邻路由训练篇再展开路径局部性与整树复用之间的差异。

通用 speculative sampling、接受规则和 self-draft KV 管理仍由投机解码系列展开。本篇聚焦候选集合、专家资格和路由局部性，不重复旧系列第六篇的完整端侧执行链。
