---
layout: post
title: "Scaling Book 入门第 14 章：JAX 并行编程入门（选读）"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '工具']
---

# Scaling Book 入门第 14 章：JAX 并行编程入门（选读）

> **本章目标**：了解 JAX 的并行编程 API，作为理解 TPU 编程范式的补充。如果你主要使用 PyTorch/Megatron，此章可选读。
>
> **对应原书**：Chapter 10 (Programming TPUs in JAX)  
> **优先级**：⭐ 低 | **建议时间**：Day 14, 约 2 小时

---

## 14.1 为什么了解 JAX 仍有价值

> 🔗 **与你的联系**
>
> 虽然你主要使用 Megatron（PyTorch），但了解 JAX 的并行编程模型有两个好处：
> 1. 很多前沿研究论文的代码用 JAX 写（如 Google 的工作）
> 2. JAX 的分片概念更显式，有助于深化对并行原理的理解
> 3. TPU 的最佳性能通常通过 JAX 获得

---

## 14.2 JAX 中的三种并行模式

### 模式 1：Auto Sharding

JAX 可以自动决定如何分片：

```python
import jax
import jax.numpy as jnp

# 定义计算
@jax.jit
def matmul(x, w):
    return x @ w

# JAX 自动决定如何在设备间分布
x = jnp.ones((1024, 4096))
w = jnp.ones((4096, 4096))
result = matmul(x, w)
```

编译器自动选择分片策略，但不保证最优。

### 模式 2：Explicit Sharding（显式分片）

程序员通过 `PartitionSpec` 指定每个张量的分片方式：

```python
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

# 定义设备网格
devices = jax.devices()
mesh = Mesh(devices.reshape(2, 4), axis_names=('dp', 'tp'))

# 显式指定分片
x_sharding = NamedSharding(mesh, P('dp', None))   # X 沿 dp 轴分片第一维
w_sharding = NamedSharding(mesh, P(None, 'tp'))    # W 沿 tp 轴分片第二维

x = jax.device_put(x, x_sharding)
w = jax.device_put(w, w_sharding)

# JAX 自动插入必要的通信
result = x @ w  # JAX 知道需要什么 AllGather/ReduceScatter
```

这对应第6章学的分片矩阵乘法：程序员指定"怎么切"，编译器自动处理"需要什么通信"。

### 模式 3：Manual Sharding（shard_map）

完全手动控制每个设备的计算：

```python
from jax.experimental.shard_map import shard_map

@shard_map(
    mesh=mesh,
    in_specs=(P('dp', None), P(None, 'tp')),
    out_specs=P('dp', 'tp')
)
def manual_matmul(x_shard, w_shard):
    # 这里的代码在每个设备上独立运行
    # x_shard 和 w_shard 已经是本地分片
    local_result = x_shard @ w_shard
    # 程序员手动调用通信原语
    result = jax.lax.psum(local_result, axis_name='dp')
    return result
```

`shard_map` 给予最大控制权，但需要程序员自己管理通信。

---

## 14.3 JAX vs PyTorch 的并行编程对比

| 特性 | JAX | PyTorch (Megatron) |
|------|-----|-------------------|
| 分片声明 | 声明式（PartitionSpec） | 命令式（手动切分） |
| 通信调度 | 编译器自动插入 | 程序员手动调用 NCCL |
| 编译 | XLA 全图编译 | Eager 模式 + 可选 torch.compile |
| 灵活性 | shard_map 可完全手动 | 完全手动（但更灵活） |
| 调试 | 编译后难调试 | Eager 模式易调试 |
| 适用场景 | TPU 优先，追求极致性能 | GPU 优先，大规模工程化 |

> 📋 **背景知识：JAX 的函数式范式**
>
> JAX 和 PyTorch 的核心区别是编程范式：
> - PyTorch：命令式（一步步执行）→ 灵活但编译器难以全局优化
> - JAX：函数式（声明计算图）→ 编译器可以全局优化但限制更多
>
> 例如 JAX 不允许在 `@jit` 函数中有数据依赖的控制流（if/for），必须用 `jax.lax.cond/scan` 替代。

---

## 14.4 对 Megatron 用户的启示

虽然你用 Megatron/PyTorch，但 JAX 的一些概念可以帮助理解：

1. **PartitionSpec 的思维**：思考每个张量"应该怎么分"，而非"怎么手动切和通信"
2. **编译器优化**：`torch.compile` 正在朝 XLA 的方向发展（全图编译）
3. **通信重叠**：JAX/XLA 编译器自动做的 overlap，在 PyTorch 中需要手动实现（Megatron 的 `--overlap-*` 参数）

---

## 14.5 实践建议

如果你想尝试 JAX：

```python
# 在 Google Colab 免费使用 TPU
import jax
print(jax.devices())  # 显示可用的 TPU 设备

# 简单的 Data Parallel 示例
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

mesh = Mesh(jax.devices(), axis_names=('dp',))
data_sharding = NamedSharding(mesh, P('dp',))

@jax.jit
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # JAX 自动在 dp 轴上 AllReduce 梯度
    return loss, grads
```

---

## 关键要点

- [ ] JAX 用 PartitionSpec 声明分片意图，编译器自动处理通信
- [ ] 三种模式：auto sharding → explicit sharding → shard_map（控制度递增）
- [ ] JAX 全图编译可以自动优化通信重叠，PyTorch 需要手动实现
- [ ] 对于 Megatron 用户：理解 JAX 概念有助于理解并行原理
- [ ] Google Colab 提供免费 TPU 可以用来实验

---

## 进一步阅读

- 原书 Chapter 10: Programming TPUs in JAX
- [JAX 官方教程: Distributed Arrays and Sharding](https://jax.readthedocs.io/en/latest/sharded-computation.html)
- [shard_map JEP](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html)

