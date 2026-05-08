---
layout: post
title: "Scaling Book 入门第 14 章：JAX 并行编程入门（选读）"
date: 2026-05-06
tags: ['LLM', 'Infra', 'Scaling', '工具']
---


> **本章目标**：了解 JAX 的并行编程 API，作为理解 TPU 编程范式的补充。如果你主要使用 PyTorch/Megatron，此章可选读。
>
> **对应原书**：[Chapter 10 (Programming TPUs in JAX)](https://jax-ml.github.io/scaling-book/jax-stuff)  
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

## 14.2 JAX 编译模型基础

在深入并行编程之前，先理解 JAX 的核心概念。

### jax.jit：即时编译

`jax.jit` 做两件事：
1. **编译**：将 Python 函数通过 tracing → StableHLO → XLA → 机器码
2. **分布**：如果输入有分片信息，自动将计算分布到多设备

```python
import jax
import jax.numpy as jnp

@jax.jit
def f(x, y):
    return jnp.einsum('bf,fd->bd', x, y)

# 第一次调用触发编译（较慢）
result = f(jnp.ones((128, 256)), jnp.ones((256, 64)))
# 后续调用复用编译结果（快）
result = f(jnp.ones((128, 256)), jnp.ones((256, 64)))
```

### Mesh：设备网格

Mesh 将物理设备组织为逻辑网格，每个轴有名字：

```python
# 8 个设备排成 2×4 网格
mesh = jax.make_mesh(
    axis_shapes=(2, 4),
    axis_names=('dp', 'tp')
)
# dp 轴有 2 个设备：可用于 Data Parallelism
# tp 轴有 4 个设备：可用于 Tensor Parallelism
```

### PartitionSpec：分片规约

`PartitionSpec` 描述张量的每个维度沿哪个 mesh 轴分片：

```python
from jax.sharding import PartitionSpec as P

P('dp', 'tp')   # 第 0 维沿 dp 分片，第 1 维沿 tp 分片
P('dp', None)   # 第 0 维沿 dp 分片，第 1 维不分片（复制）
P(None, 'tp')   # 第 0 维不分片，第 1 维沿 tp 分片
P()              # 全部复制到每个设备
```

> 📋 **背景知识：与 Megatron 的概念映射**
>
> | JAX | Megatron-LM |
> |-----|-------------|
> | `mesh = (dp=4, tp=8)` | `--data-parallel-size 4 --tensor-model-parallel-size 8` |
> | `P('dp', None)` | 输入 batch 沿 DP 切分 |
> | `P(None, 'tp')` | 权重沿 TP 切分（列并行） |
> | `jax.lax.psum` | `torch.distributed.all_reduce` |
> | `jax.lax.all_gather` | `torch.distributed.all_gather` |

---

## 14.3 三种并行模式

JAX 支持三种并行编程范式，控制粒度递增：

| 模式 | 视角 | 显式分片？ | 显式通信？ | 适用场景 |
|------|------|----------|----------|---------|
| **Auto** | 全局 | ❌ | ❌ | 快速原型 |
| **Explicit** | 全局 | ✅ | ❌ | 生产代码 |
| **Manual (shard_map)** | 单设备 | ✅ | ✅ | 性能优化 |

### 模式 1：Auto Sharding

最简单——让 XLA 编译器自动决定一切：

```python
mesh = jax.make_mesh((4, 2), ('X', 'Y'))
jax.set_mesh(mesh)

In = jnp.zeros((8, 2048), dtype=jnp.bfloat16,
               device=jax.NamedSharding(mesh, P('X', 'Y')))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16,
              device=jax.NamedSharding(mesh, P('Y', None)))

def matmul_square(In, W):
    return jnp.einsum('bd,df->bf', jnp.square(In), W)

jit_matmul = jax.jit(matmul_square,
                     out_shardings=P('X', None)).lower(In, W).compile()
out = jit_matmul(In, W)
```

**背后发生了什么？**

1. `In[B_X, D_Y]` 沿 X 轴分 batch，沿 Y 轴分 hidden dim
2. `W[D_Y, F]` 沿 Y 轴分 contracting dim
3. 由于指定 `out_shardings=P('X', None)`，输出在 X 上分片但 Y 上复制
4. 这需要一个 **AllReduce** 沿 Y 轴！

XLA 自动插入了这个 AllReduce。查看编译后的 HLO：

```c
%fusion = bf16[2,8192]{...} fusion(
    bf16[2,1024]{...} %param,
    bf16[8192,1024]{...} %copy-done)  ← 本地 matmul

ROOT %AllReduce = bf16[2,8192]{...} AllReduce(
    bf16[2,8192]{...} %fusion)        ← XLA 自动添加
```

注意 shape 是 **本地视角**：batch 8 / 4 设备 = 2，hidden 2048 / 2 = 1024。

**问题**：编译器可能犯错——有时插入不必要的 AllGather，占用 80% 的 profile 时间。此时需要 `jax.lax.with_sharding_constraint` 来"纠正"编译器：

```python
def matmul(x, Win, Wout):
    hidden = jnp.einsum('bd,df->bf', x, Win)
    hidden = jax.lax.with_sharding_constraint(hidden, P('X', 'Y'))  # 约束中间张量
    return jnp.einsum('bf,df->bd', hidden, Wout)
```

### 模式 2：Explicit Sharding

Explicit sharding 将分片信息纳入 JAX 的类型系统——当通信决策不明确时，JAX 会 **报错而非猜测**：

```python
import jax.sharding as shd

mesh = jax.make_mesh(
    axis_shapes=(2, 2), axis_names=('X', 'Y'),
    axis_types=(shd.AxisType.Explicit, shd.AxisType.Explicit))
jax.set_mesh(mesh)

In = jnp.zeros((8, 2048), dtype=jnp.bfloat16, out_sharding=P('X', 'Y'))
W = jnp.zeros((2048, 8192), dtype=jnp.bfloat16, out_sharding=P('Y', None))

@jax.jit
def matmul_square(In, W):
    return jnp.einsum('bd,df->bf', jnp.square(In), W)

matmul_square(In, W)  # 报错！
```

**错误信息**：`Contracting dimensions are sharded and it is ambiguous how the output should be sharded.`

这很好！因为输出可以是：
- `P('X', 'Y')` → ReduceScatter（更省内存）
- `P('X', None)` → AllReduce（输出复制）

程序员必须显式选择：

```python
@jax.jit
def matmul_square(In, W):
    return jnp.einsum('bd,df->bf', jnp.square(In), W,
                      out_sharding=P('X', 'Y'))  # 选择 ReduceScatter
```

> 💡 **Explicit vs Auto 的核心区别**
>
> - **Auto**：编译器猜测通信方式 → 可能猜错
> - **Explicit**：歧义时报错要求程序员决定 → 不会猜错
>
> 对应 PyTorch 的类比：Auto ≈ `torch.compile` 自动优化；Explicit ≈ 手动选择 `all_reduce` vs `reduce_scatter`

### 模式 3：shard_map（手动分片）

`shard_map` 给每个设备一个 **本地视角**：你看到的是自己 shard 的数据，通信必须自己写。

**基础示例**：

```python
@jax.shard_map(
    in_specs=P(('x', 'y')),
    out_specs=P()
)
def slice_and_average(x):
    # x 是 1/8 的原始数组（本地 shard）
    assert x.shape == (512 // 8,)
    # 手动调用 AllReduce 求平均
    return jax.lax.pmean(x[:4], axis_name=('x', 'y'))

x = jnp.arange(0, 512, dtype=jnp.int32, out_sharding=P(('x', 'y')))
out = slice_and_average(x)  # shape: (4,)
```

这等效于 `mean(x[:4], x[64:68], x[128:132], ...)`——在 `jax.jit` 全局视角下很难表达的操作。

**shard_map 可用的通信原语**：

| 原语 | 等价于 | 功能 |
|------|--------|------|
| `jax.lax.psum(x, axis)` | AllReduce | 所有设备的 x 求和 |
| `jax.lax.pmean(x, axis)` | AllReduce + 除 N | 所有设备的 x 求平均 |
| `jax.lax.all_gather(x, axis)` | AllGather | 收集所有 shard 的 x |
| `jax.lax.ppermute(x, axis, perm)` | 环形通信 | 按指定排列发送数据 |
| `jax.lax.all_to_all(x, axis, ...)` | All-to-All | 每设备发一块给每个其他设备 |
| `jax.lax.axis_index(axis)` | — | 当前设备在该轴上的索引 |
| `jax.lax.axis_size(axis)` | — | 该轴的设备数 |

**何时使用 shard_map vs jax.jit？**

| 场景 | 推荐 | 原因 |
|------|------|------|
| 标准 matmul + TP/DP | jax.jit (explicit) | 编译器能自动处理 |
| 需要通信重叠 | shard_map | 编译器可能不做 overlap |
| 非标准通信模式（MoE routing） | shard_map | 编译器无法推导 |
| 需要 shard 内局部操作 | shard_map | 全局视角不方便表达 |
| 快速原型 | jax.jit (auto) | 不需要思考分片 |

> 🛠️ **实践：Megatron vs JAX 的 TP 实现对比**
>
> **Megatron（手动通信）**：
> ```python
> # megatron/core/tensor_parallel/layers.py
> class ColumnParallelLinear:
>     def forward(self, input):
>         input_parallel = copy_to_tensor_model_parallel_region(input)
>         output = F.linear(input_parallel, self.weight)  # 本地 matmul
>         output = gather_from_tensor_model_parallel_region(output)  # AllGather
>         return output
> ```
>
> **JAX（Explicit sharding）**：
> ```python
> @jax.jit
> def column_parallel_linear(x, w):
>     return jnp.einsum('bd,df->bf', x, w,
>                       out_sharding=P('dp', None))  # 自动 AllReduce
> ```
>
> **JAX（shard_map）**：
> ```python
> @jax.shard_map(in_specs=(P('dp', None), P(None, 'tp')),
>                out_specs=P('dp', None))
> def column_parallel_linear(x, w):
>     local_out = x @ w
>     return jax.lax.psum(local_out, 'tp')  # 手动 AllReduce
> ```
>
> JAX 的 Explicit 模式最简洁；shard_map 和 Megatron 一样手动但更声明式。

---

## 14.4 Collective Matmul：通信-计算重叠

这是 shard_map 最重要的实际应用——实现通信与计算的重叠。

### 问题

模型并行中，常见操作：`A[B_X, D_Y] × W[D, F_Y] → Out[B_X, F_Y]`

朴素实现需要先 AllGather 再 matmul：

```
步骤 1: A[B_X, D] = AllGather_Y(A[B_X, D_Y])    ← 阻塞通信
步骤 2: Out[B_X, F_Y] = A[B_X, D] × W[D, F_Y]  ← 计算
                         ↑ 必须等通信完成才能开始
```

### Collective Matmul 解决方案

将 AllGather 拆成多步，每步通信一个 chunk 的同时计算上一个 chunk 的 matmul：

```
设备 0:  [通信 chunk0][计算 chunk0 + 通信 chunk1][计算 chunk1 + 通信 chunk2]...
设备 1:  [通信 chunk1][计算 chunk1 + 通信 chunk2][计算 chunk2 + 通信 chunk3]...
                      ↑ 通信和计算重叠！
```

### shard_map 实现

```python
import functools
import jax
import jax.numpy as jnp
import numpy as np

mesh = jax.make_mesh(axis_shapes=(2, 4), axis_names=('X', 'Y'))
jax.set_mesh(mesh)

B, D, F = 1024, 2048, 8192
A = jax.device_put(jnp.zeros((B, D)), P('X', 'Y'))
W = jax.device_put(jnp.zeros((D, F)), P(None, 'Y'))

def collective_matmul_allgather(lhs, rhs):
    """通信-计算重叠的 AllGather Matmul"""
    axis_size = jax.lax.axis_size('Y')  # 4
    idx = jax.lax.axis_index('Y')
    chunk_size = lhs.shape[1]

    def f(i, carrys):
        accum, lhs = carrys
        # 计算当前 chunk 的 matmul
        rhs_chunk = jax.lax.dynamic_slice_in_dim(
            rhs, (idx + i) % axis_size * chunk_size, chunk_size)
        update = lhs @ rhs_chunk
        # 同时做环形 permute（通信）
        lhs = jax.lax.ppermute(
            lhs, axis_name='Y',
            perm=[(j, (j - 1) % axis_size) for j in range(axis_size)])
        return accum + update, lhs

    accum = jnp.zeros((lhs.shape[0], rhs.shape[1]), dtype=lhs.dtype)
    accum, lhs = jax.lax.fori_loop(
        0, axis_size - 1, f, (accum, lhs), unroll=True)

    # 最后一个 chunk
    i = axis_size - 1
    rhs_chunk = jax.lax.dynamic_slice_in_dim(
        rhs, (idx + i) % axis_size * chunk_size, chunk_size)
    return accum + lhs @ rhs_chunk

jit_collective = jax.jit(jax.shard_map(
    collective_matmul_allgather,
    in_specs=(P('X', 'Y'), P(None, 'Y')),
    out_specs=P('X', 'Y')))
```

### 性能对比

| 实现方式 | 时间 | 说明 |
|---------|------|------|
| 无分片 matmul | 224 μs | 单设备基线（无通信） |
| jax.jit + auto | 311 μs | 有阻塞 AllGather |
| collective matmul | 244 μs | 通信重叠！ |

**Collective matmul 将通信开销从 ~87 μs 降到 ~20 μs**（几乎完全被计算掩盖）。

### ReduceScatter Collective Matmul

MLP 的 down-projection 需要 ReduceScatter：`Tmp[B_X, F_Y] × W_down[F_Y, D] → Out[B_X, D_Y]`

与 AllGather 版本互补——每步计算一个 chunk 的 partial result，同时 permute 部分结果：

```python
@jax.shard_map(
    in_specs=(P('X', 'Y'), P('Y', None)),
    out_specs=P('X', 'Y'))
def collective_reduce_scatter_matmul(lhs, rhs):
    axis_size = jax.lax.axis_size('Y')
    idx = jax.lax.axis_index('Y')
    chunk_size = rhs.shape[1] // axis_size  # D / Y

    def f(i, carrys):
        accum, partial = carrys
        # 计算当前 chunk
        rhs_chunk = jax.lax.dynamic_slice_in_dim(
            rhs, ((idx - i) % axis_size) * chunk_size, chunk_size, axis=1)
        update = lhs @ rhs_chunk  # [B/X, D/Y]
        # Permute partial results
        partial = partial + update
        partial = jax.lax.ppermute(
            partial, 'Y',
            perm=[(j, (j + 1) % axis_size) for j in range(axis_size)])
        return accum, partial

    accum = jnp.zeros((lhs.shape[0], chunk_size))
    _, result = jax.lax.fori_loop(0, axis_size, f, (accum, accum), unroll=True)
    return result
```

### 端到端 MLP 性能

将 AllGather collective matmul（up-proj）+ ReduceScatter collective matmul（down-proj）组合：

| 实现 | Up-proj 时间 | Down-proj 时间 | 总时间 | 通信开销占比 |
|------|-------------|---------------|--------|------------|
| jax.jit (auto) | 250 μs | 260 μs | 510 μs | ~35% |
| collective matmul | 230 μs | 240 μs | 470 μs | ~8% |
| 无分片基线 | 224 μs | 224 μs | 448 μs | 0% |

Collective matmul 将通信开销从 35% 降到 8%！

> 📋 **背景知识：为什么通信可以被"掩盖"？**
>
> TPU/GPU 有独立的计算单元（MXU/SM）和通信单元（ICI/NVLink DMA）。只要两者操作不同的内存区域，就可以同时工作。Collective matmul 利用这一点：
>
> ```
> MXU:   [计算 chunk i] [计算 chunk i+1] [计算 chunk i+2]
> DMA:   [发送 chunk i] [发送 chunk i+1] [发送 chunk i+2]
>         ↑ 同时进行，互不干扰
> ```
>
> 在 Megatron 中，这对应 `--overlap-grad-reduce` 和 `--overlap-param-gather` 选项——原理相同，但由 Megatron 的通信调度器实现而非 XLA。

---

## 14.5 JAX vs PyTorch 并行编程对比

| 特性 | JAX | PyTorch (Megatron) |
|------|-----|-------------------|
| 分片声明 | 声明式（PartitionSpec） | 命令式（手动切分） |
| 通信调度 | 编译器自动插入 / shard_map 手动 | 程序员手动调用 NCCL |
| 编译 | XLA 全图编译 | Eager 模式 + 可选 torch.compile |
| 灵活性 | shard_map 可完全手动 | 完全手动 |
| 调试 | 编译后难调试 | Eager 模式易调试 |
| 通信重叠 | 编译器自动 or collective matmul | 手动 `--overlap-*` |
| 适用场景 | TPU 优先，追求极致性能 | GPU 优先，大规模工程化 |

> 📋 **背景知识：JAX 的函数式范式**
>
> JAX 和 PyTorch 的核心区别是编程范式：
> - **PyTorch**：命令式（一步步执行）→ 灵活，调试容易，但编译器难以全局优化
> - **JAX**：函数式（声明计算图）→ 编译器可以全局优化，但限制更多
>
> JAX 的限制：不允许在 `@jit` 中使用数据依赖的控制流（Python 的 if/for），必须用：
> - `jax.lax.cond` 替代 `if`
> - `jax.lax.scan` / `fori_loop` 替代 `for`
>
> 这些限制让编译器能"看到"完整计算图 → 做更激进的优化。

### PyTorch 正在向 JAX 靠拢

`torch.compile` 正在朝 XLA 全图编译的方向发展：

| 特性 | JAX jit | torch.compile | Megatron (eager) |
|------|---------|---------------|-----------------|
| 全图编译 | ✅ | 部分（有 graph break） | ❌ |
| 自动 fusion | ✅ | ✅ | ❌ |
| 自动通信插入 | ✅ | ❌ | ❌ |
| 自动 layout 优化 | ✅ | 部分 | ❌ |

> 🛠️ **实践：Megatron 用户如何借鉴 JAX 思维**
>
> 1. **PartitionSpec 思维**：在设计并行策略前，先为每个张量画出 `(batch, hidden, ff)` 的分片方式，然后推导需要什么通信。这就是 JAX 的 explicit sharding 思维。
>
> 2. **通信重叠**：JAX 的 collective matmul 在 Megatron 中通过以下方式实现：
>    ```bash
>    --overlap-grad-reduce      # backward 时 AllReduce 与计算重叠
>    --overlap-param-gather     # FSDP 的 AllGather 与计算重叠
>    ```
>
> 3. **全图优化**：即使不用 JAX，理解"编译器需要看到完整计算图才能优化"这个概念，有助于写出对 `torch.compile` 更友好的代码（减少 graph break）。

---

## 14.6 Pallas：自定义 Kernel

当 XLA 编译器生成的 kernel 不够优时，可以用 Pallas 写自定义 kernel（类似 CUDA kernel 但面向 TPU）。

```python
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    # 在 VMEM/寄存器中直接操作
    o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // 128,),  # 并行 tile 数
        in_specs=[
            pl.BlockSpec(block_shape=(128, x.shape[1]), memory_space=pl.TPUMemorySpace.VMEM),
            pl.BlockSpec(block_shape=(128, y.shape[1]), memory_space=pl.TPUMemorySpace.VMEM),
        ],
        out_specs=pl.BlockSpec(block_shape=(128, x.shape[1]), memory_space=pl.TPUMemorySpace.VMEM),
    )(x, y)
```

**Pallas 的 GPU 等价物**：Triton（由 OpenAI 开发）。

| 方面 | Pallas (TPU) | Triton (GPU) |
|------|-------------|-------------|
| 抽象级别 | 操作 VMEM tiles | 操作 SRAM tiles |
| 编程模型 | 函数式 | 类 Python |
| Flash Attention | `pallas.flash_attention` | `triton.ops.flash_attention` |
| 自定义 kernel | 写 Pallas 函数 | 写 `@triton.jit` 函数 |

> 📋 **背景知识：为什么需要自定义 Kernel？**
>
> XLA 编译器生成的 kernel 在以下场景可能不够优：
>
> 1. **Flash Attention**：需要精细的 tiling 和在线 softmax，XLA 无法自动生成
> 2. **Quantization kernel**：int4/int8 的混合精度 matmul 需要特殊处理
> 3. **Sparse operations**：如 MoE 的 token routing，标准 matmul 无法高效处理
> 4. **Custom fusion**：某些操作组合（如 RMSNorm + matmul）需要手动 fusion
>
> 实际中 99% 的 LLM 工作不需要自己写 kernel——框架已经提供了优化实现。但理解 Pallas/Triton 的存在有助于理解：
> - 为什么某些操作（如 Flash Attention）不是简单地"让编译器处理"
> - 性能优化的最终手段是什么
> - 框架开发者如何工作

### 与 SGLang/vLLM 的关联

SGLang 和 vLLM 底层大量使用 Triton kernel：

```python
# SGLang 中的 Flash Attention 调用链
# sglang/srt/layers/attention/flashinfer_backend.py
#   → flashinfer（CUDA kernel）
#   → 或 triton_attention（Triton kernel）

# vLLM 中的 PagedAttention
# vllm/attention/ops/paged_attn.py
#   → 自定义 CUDA kernel 实现 paged KV cache 访问
```

当你在 SGLang 中设置 `--attention-backend triton` 时，就是在使用 Triton 写的自定义 attention kernel。

---

## 14.7 实践建议

### 在 Google Colab 免费使用 TPU

```python
# Colab 提供免费的 TPU v2-8（8 个 TPU 核心）
import jax
print(f"Devices: {jax.devices()}")
print(f"Platform: {jax.devices()[0].platform}")  # 'tpu'

# 或模拟多设备测试（不需要真实 TPU）
import jax
jax.config.update('jax_num_cpu_devices', 8)
```

### Data Parallel 完整示例

```python
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

mesh = jax.make_mesh((8,), ('dp',))
jax.set_mesh(mesh)

def loss_fn(params, batch):
    logits = params['w'] @ batch['x'].T
    return jnp.mean((logits - batch['y']) ** 2)

@jax.jit
def train_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # JAX 自动在 dp 轴上 AllReduce 梯度
    params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    return params, loss

# 参数复制到所有设备，数据沿 dp 分片
params = {'w': jnp.ones((10, 32), out_sharding=P())}
batch = {
    'x': jnp.ones((1024, 32), out_sharding=P('dp',)),
    'y': jnp.ones((1024, 10), out_sharding=P('dp',)),
}
params, loss = train_step(params, batch)
```

### Tensor Parallel MLP 示例

```python
mesh = jax.make_mesh((4, 2), ('dp', 'tp'))
jax.set_mesh(mesh)

@jax.jit
def tp_mlp(x, w_up, w_down):
    # x: [B, D] → P('dp', None)
    # w_up: [D, F] → P(None, 'tp')  列并行
    hidden = jnp.einsum('bd,df->bf', x, w_up,
                        out_sharding=P('dp', 'tp'))
    hidden = jax.nn.relu(hidden)
    # w_down: [F, D] → P('tp', None)  行并行
    out = jnp.einsum('bf,fd->bd', hidden, w_down,
                     out_sharding=P('dp', None))  # AllReduce
    return out

# 初始化分片
x = jnp.ones((1024, 8192), out_sharding=P('dp', None))
w_up = jnp.ones((8192, 32768), out_sharding=P(None, 'tp'))
w_down = jnp.ones((32768, 8192), out_sharding=P('tp', None))

out = tp_mlp(x, w_up, w_down)
print(f"Output shape: {out.shape}")       # (1024, 8192)
print(f"Output sharding: {out.sharding}") # P('dp', None)
```

### 检查编译后的 HLO

```python
# 查看编译后的 HLO（包括通信操作）
compiled = jax.jit(tp_mlp).lower(x, w_up, w_down).compile()
print(compiled.as_text())
# 应该看到：
# 1. fusion（matmul x × w_up）
# 2. fusion（relu）
# 3. fusion（matmul hidden × w_down）
# 4. AllReduce（沿 tp 轴）
```

### JAX Profiling 快速入门

```python
# Profile 一个训练步骤
with jax.profiler.trace("/tmp/jax_trace"):
    for i in range(5):
        out = tp_mlp(x, w_up, w_down)
        out.block_until_ready()  # 必须等待完成才能获取准确时间

# 在 TensorBoard 中查看
# tensorboard --logdir=/tmp/jax_trace
```

---

## 14.8 Worked Problems

### Q1：shard_map 基础练习

设 `A` 是 float32[S_X, D_Y] 的分片数组，`X * Y = N`。

**(a)** 用 shard_map 计算每个 shard 的平均值，返回 [X, Y] 的数组。

<details markdown="1">
<summary>点击查看答案</summary>

```python
average_shmap = jax.shard_map(
    lambda x: x.mean(keepdims=True),
    mesh=mesh,
    in_specs=P('X', 'Y'),
    out_specs=P('X', 'Y')
)
# 每个设备独立计算本地 shard 的平均值 → 无通信
```

用 jax.jit 的等价实现更复杂（需要 reshape）：

```python
def average(x):
    X, Y = mesh.axis_sizes
    return x.reshape(X, x.shape[0]//X, Y, x.shape[1]//Y).mean(axis=(1, 3))
```

</details>

**(b)** 用 shard_map 实现 `roll(x, shift, axis=0) - x`，仅在每个 X shard 内做 roll。

<details markdown="1">
<summary>点击查看答案</summary>

```python
def shift_shmap(x, shift):
    return jax.shard_map(
        lambda x: jnp.roll(x, shift, axis=0),
        mesh=mesh,
        in_specs=P('X', 'Y'),
        out_specs=P('X', 'Y')
    )(x)

# 关键：roll 只在本地 shard 内部做，不跨设备 → 无通信
```

</details>

---

### Q2：MoE 路由实现

设 `W[E_X, D, F]` 是 E 个 expert 矩阵，`A[S_X, D]` 是激活，`B[S_X]` 是路由分配（B[i] ∈ [0, E)）。实现 `Out[i] = W[B[i]] @ A[i]`。

**(a)** 先忽略分片，写单设备版本。

<details markdown="1">
<summary>点击查看答案</summary>

```python
def moe_local(W, A, B):
    S, _ = A.shape
    E, _, F = W.shape

    def expert_forward(carry, e):
        output = carry
        mask = (B == e)[:, None]         # [S, 1]
        expert_result = A @ W[e]          # [S, F]
        return output + expert_result * mask, None

    output = jnp.zeros((S, F))
    output, _ = jax.lax.scan(expert_forward, output, jnp.arange(E))
    return output
```

注意：不要一次性创建 [S, D, F] 张量（太大）！scan 逐 expert 处理。

</details>

**(b)** 将 (a) 用 jax.jit 运行，profile 它做了什么通信？

<details markdown="1">
<summary>点击查看答案</summary>

jax.jit 会 AllGather 整个 `A` 到每个设备 → 非常昂贵。因为每个 expert 可能需要任意 token，而 token 分布在不同设备上。

解决方案：用 shard_map + `all_to_all` 做 token 路由，只发送每个 expert 需要的 token。

</details>

---

### Q3：Collective AllReduce Matmul

实现 `A[B_X, D_Y] × W[D_Y, F] → Out[B_X, F]` 的通信重叠版本。

提示：沿 output dimension 分 tile，用 `jax.lax.psum` 做 AllReduce。

<details markdown="1">
<summary>点击查看答案</summary>

思路：将 F 维分成 Y 个 chunk，每步计算一个 chunk 的 partial sum 并同时 permute：

```python
@jax.shard_map(
    in_specs=(P('X', 'Y'), P('Y', None)),
    out_specs=P('X', None))
def collective_allreduce_matmul(lhs, rhs):
    # lhs: [B/X, D/Y], rhs: [D/Y, F]
    local_out = lhs @ rhs           # [B/X, F] partial sum
    return jax.lax.psum(local_out, 'Y')  # AllReduce
```

要做 overlap，需要将 rhs 沿 F 分 chunk，逐 chunk 计算 + permute。但由于 XLA 的调度方式，实际性能提升可能不明显。

</details>

---

### Q4：端到端 Transformer MLP

实现完整的 MLP 块：`In[B_X, D_Y] × W_up[D, F_Y] × W_down[F_Y, D] → Out[B_X, D_Y]`

分别用 jax.jit (explicit sharding) 和 shard_map (collective matmul) 实现，并比较 profile。

<details markdown="1">
<summary>点击查看答案</summary>

**jax.jit 版本**（最简单）：

```python
@jax.jit
def mlp_jit(x, w_up, w_down):
    h = jnp.einsum('bd,df->bf', x, w_up, out_sharding=P('X', 'Y'))
    h = jax.nn.gelu(h)
    return jnp.einsum('bf,fd->bd', h, w_down, out_sharding=P('X', 'Y'))
```

**shard_map + collective matmul 版本**：
将 14.4 的 AllGather collective matmul 用于 up-projection，ReduceScatter collective matmul 用于 down-projection，两者的通信都与 matmul 重叠。

理论上比 jax.jit 快 10-30%（取决于通信/计算比）。

</details>

---

## 关键要点

| 概念 | 要点 |
|------|------|
| 三种模式 | Auto → Explicit → shard_map，控制粒度递增 |
| PartitionSpec | 声明式描述每个维度的分片方式 |
| Auto sharding | 编译器自动插入通信，可能犯错 |
| Explicit sharding | 歧义时报错要求程序员决定，更安全 |
| shard_map | 本地视角 + 手动通信，最大控制权 |
| Collective Matmul | 通信-计算重叠的核心技术，将 AllGather 延迟从 87μs 降到 20μs |
| JAX vs PyTorch | JAX 更声明式/编译器驱动，PyTorch 更手动/灵活 |
| Pallas/Triton | 自定义 kernel 的最后手段，通常不需要 |
| torch.compile | PyTorch 正在向 JAX 的全图编译方向发展 |

---

## 进一步阅读

- [原书 Chapter 10: Programming TPUs in JAX](https://jax-ml.github.io/scaling-book/jax-stuff)
- [JAX 官方教程: Distributed Arrays and Sharding](https://jax.readthedocs.io/en/latest/sharded-computation.html)
- [shard_map JEP](https://jax.readthedocs.io/en/latest/jep/14273-shard-map.html)
- [shard_map 示例笔记本](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html)
- [Explicit Sharding 文档](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html)
- [Pallas 文档](https://jax.readthedocs.io/en/latest/pallas/tpu/details.html)
- [Collective Matmul 论文 (Wang et al., 2023)](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)

