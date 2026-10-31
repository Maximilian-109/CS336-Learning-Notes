# 第 02 讲详解：PyTorch、张量与资源核算

> 对应 Stanford CS336，Spring 2026，Lecture 2：PyTorch, resource accounting。  
> 整理日期：2026-11-01。本文根据学习对话整理，包含自编教学例子、逐步推导与工程补充。  
> 阅读目标：理解训练时数据怎样变成张量，参数、激活和梯度怎样产生，以及计算量、显存和训练速度应该怎样估算。

本文参考[官方第二讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)和 [lvsir18 的第二讲中文笔记](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes/blob/main/lectures/02-pytorch-resource-accounting.md)。后者适合复习；这里把结论之间省略的步骤展开。本文不是官方讲义的逐字翻译。

第一讲解释文字怎样变成 token，以及语言模型在预测什么。第二讲进一步问：**这些 token 进入模型以后，计算机实际存了什么、算了什么，为什么会慢，为什么会爆显存？**

文中算例统一注明假设。容量默认使用十进制 `MB = 10^6 byte`、`GB = 10^9 byte`；`GiB = 2^30 byte` 会另外标明。代码用于教学；CUDA 基准示例未在本文整理环境中做 GPU 实测，不代表某块硬件的实测性能。

## 目录

1. [这一讲的主线与阅读顺序](#1-这一讲的主线与阅读顺序)
2. [先读懂符号和一次训练的数据流](#2-先读懂符号和一次训练的数据流)
3. [张量的 shape 和每个轴的含义](#3-张量的-shape-和每个轴的含义)
4. [dtype 与张量大小](#4-dtype-与张量大小)
5. [storage、stride 与视图](#5-storagestride-与视图)
6. [reshape、transpose 和 contiguous](#6-reshapetranspose-和-contiguous)
7. [device、广播与 einsum](#7-device广播与-einsum)
8. [注意力中的张量形状逐步变化](#8-注意力中的张量形状逐步变化)
9. [Autograd 与反向传播的数值例子](#9-autograd-与反向传播的数值例子)
10. [训练 API：清梯度、更新参数和关闭求导](#10-训练-api清梯度更新参数和关闭求导)
11. [Transformer 参数量逐项推导](#11-transformer-参数量逐项推导)
12. [矩阵乘和反向传播的 FLOPs](#12-矩阵乘和反向传播的-flops)
13. [6ND 的来源、用途与边界](#13-6nd-的来源用途与边界)
14. [训练显存从哪里来](#14-训练显存从哪里来)
15. [混合精度为何不保证总显存减半](#15-混合精度为何不保证总显存减半)
16. [算术强度与 Roofline](#16-算术强度与-roofline)
17. [MFU：模型利用了多少峰值算力](#17-mfu模型利用了多少峰值算力)
18. [梯度累积：小批次怎样组成大批次](#18-梯度累积小批次怎样组成大批次)
19. [Activation checkpointing：少存一些、重新算一些](#19-activation-checkpointing少存一些重新算一些)
20. [怎样正确测量训练速度](#20-怎样正确测量训练速度)
21. [遇到 OOM 或训练慢时怎样定位](#21-遇到-oom-或训练慢时怎样定位)
22. [术语表与自测解答](#22-术语表与自测解答)
23. [参考资料](#23-参考资料)

## 1. 这一讲的主线与阅读顺序

可以用一次训练把本讲串起来：

```text
token ID
   ↓ embedding：从参数表里查向量
每个位置的向量
   ↓ Transformer：矩阵乘、注意力、FFN 等
每个位置对词表的预测分数 logits
   ↓ 与正确的下一个 token 比较
标量 loss
   ↓ backward：沿计算关系求导
参数梯度
   ↓ optimizer.step：使用梯度和优化器状态
更新后的参数
```

沿这条链，需要回答四组问题：

| 问题 | 需要理解的概念 |
| --- | --- |
| 数字在哪里、怎样排列？ | shape、dtype、device、storage、stride |
| 模型究竟学习哪些数字？ | 参数、激活、梯度、autograd |
| 训练要多少资源？ | 参数量、FLOPs、优化器状态、峰值显存 |
| 怎样更快或更省显存？ | 算术强度、梯度累积、重计算、正确计时 |

建议分三遍阅读：

1. **第一遍看第 2—10 节**：先把张量、注意力形状和训练流程连起来。
2. **第二遍看第 11—15 节**：跟着算一遍参数量、计算量和显存。
3. **第三遍看第 16—22 节**：理解性能、显存优化和实际排查。

初学时不用先掌握 CUDA 编程。先能说明每个张量的含义、每项资源从哪里来，再学习底层优化会更容易。

## 2. 先读懂符号和一次训练的数据流

### 2.1 符号表

| 符号 | 含义 | 本文常用例子 |
| --- | --- | --- |
| $B$ | 当前一次前向处理的序列条数，batch size | 2 |
| $T$ | 每条序列处理的 token 数，sequence length | 4 |
| $d$ | 每个位置的隐藏向量长度，模型宽度 | 768 |
| $h$ | 标准多头注意力的 head 数 | 12 |
| $d_h$ | 每个 head 的特征维度，$d_h=d/h$ | 64 |
| $d_{ff}$ | FFN 内部的中间宽度 | 2,048 |
| $L$ | Transformer block 数量 | 12 |
| $V$ | 词表大小 | 10,000 |
| $N$ | 按指定架构统计的独立可训练参数个数 | 约 1 亿 |
| $D$ | 累计用于训练的 token 数 | 例如 100 亿 |
| $b$ | 每张设备每个 microbatch 的序列条数 | 2 |
| $a$ | 每次参数更新累积的 microbatch 次数 | 4 |
| $p$ | 数据并行设备数 | 8 |

**小写 $d$ 是向量维度，大写 $D$ 是 token 总量。词表大小 $V$ 与注意力里的 Value 张量也要按上下文区分。**

### 2.2 从整数 ID 到标量 loss

假设一批输入是两段各含 4 个 token 的文本：

```text
inputs =
[[11, 28, 35, 19],
 [42, 17, 93,  8]]
```

输入形状是 `[2,4]`。经过不同模块后：

| 阶段 | 形状 | 每一项代表什么 |
| --- | --- | --- |
| token ID | `[B,T]` | 某条序列某个位置的整数编号 |
| embedding 输出 | `[B,T,d]` | 该位置的向量 |
| Transformer 输出 | `[B,T,d]` | 结合上下文更新后的向量 |
| LM head 输出 | `[B,T,V]` | 该位置对词表各 token 的分数 |
| 每个位置的 loss | `[B,T]` 或展平后的 `[BT]` | 预测与正确答案的差距 |
| 平均 loss | `[]`，零维张量 | 当前训练目标的一个标量 |

这里假设使用常见的逐 token 预测损失；padding 等无效位置是否参与平均，必须由实际损失函数和 mask 决定。

### 2.3 序列条数与 token 数不同

`B=8,T=2048` 表示 8 条序列，每条处理 2,048 个位置。在没有 padding 等无效位置时：

$$
\text{当前 batch 的 token 数}=BT=16,384
$$

如果存在被忽略的位置，有效 loss token 数可能小于 `BT`。后面讨论梯度累积和吞吐统计时，这个区别会直接影响结果。

## 3. 张量的 shape 和每个轴的含义

### 3.1 张量先看作带有轴含义的数字容器

一个张量可以是：

```text
标量：              3.2                  shape = []
向量：              [1,2,3]              shape = [3]
矩阵：              两行三列             shape = [2,3]
一批序列的隐藏表示： batch、位置、特征     shape = [B,T,d]
```

维数更多，并不意味着需要想象一个高维几何物体。通常只需要理解每个下标在给谁编号。

对于：

```python
x.shape == (2, 4, 768)
```

下面几种索引分别表示：

| 表达式 | 含义 | 形状 |
| --- | --- | --- |
| `x[0]` | 第 1 条序列的所有位置 | `[4,768]` |
| `x[0,2]` | 第 1 条序列第 3 个位置的向量 | `[768]` |
| `x[0,2,5]` | 该位置向量的第 6 个特征 | `[]` |
| `x[:,:,5]` | 所有序列所有位置的第 6 个特征 | `[2,4]` |

### 3.2 最后一维不是另外一串 token

`[B,T,d]` 中，`T` 是序列位置，`d` 是每个位置的特征数。

例如：

```text
第 3 个 token 的向量：
[0.12, -0.48, 1.03, ..., 0.07]  ← 共 768 个数
```

这里的 768 个数不是 768 个 token，而是同一个 token 在当前层的 768 个特征分量。

### 3.3 logits 的每个轴

若词表大小为 10,000：

```python
logits.shape == (2, 4, 10000)
```

则：

```python
logits[0, 2, :]
```

是第一条序列第 3 个位置，对全部 10,000 个候选 token 的预测分数。

训练时，通常用该位置的向量预测它后面的正确 token。**logits 是分数，经过 softmax 才是概率；交叉熵实现通常可以直接接收 logits。**

## 4. dtype 与张量大小

### 4.1 同样的数字个数，不同格式占用不同空间

| 常见 dtype | 位数 | 每个元素的字节数 | 常见用途 |
| --- | --- | --- | --- |
| `torch.float32`，FP32 | 32 | 4 | 参数、部分统计、较高精度计算 |
| `torch.float16`，FP16 | 16 | 2 | 低精度计算与存储 |
| `torch.bfloat16`，BF16 | 16 | 2 | 深度学习中的低精度计算 |
| `torch.int64` | 64 | 8 | token ID、索引等 |

一个 byte 是 8 bit。这里的 16 或 32 指保存一个元素使用的 bit 数。

对于独立存储的普通稠密张量：

$$
M=\operatorname{numel}(x)\times\operatorname{element\_size}(x)
$$

例如 `[2,4,768]` 的 FP32 张量：

$$
M=2\times4\times768\times4=24,576\ \text{byte}
$$

PyTorch 中可以检查：

```python
print(x.shape)
print(x.dtype)
print(x.numel())
print(x.element_size())
print(x.numel() * x.element_size())
```

最后一个结果是张量的**逻辑数据量**。多个视图可能共享同一份存储，因此不能无条件把每个张量的这个数相加。

### 4.2 BF16 和 FP16 同样大，为何还要区分？

浮点数的位会分配给符号、指数和有效数字。指数范围影响能表示多大、多小的数；有效数字位数影响相近数值之间分得有多细。

BF16 与 FP16 都占 2 byte，但位分配不同：BF16 的指数范围接近 FP32，而有效数字更少。FP16 的指数范围较小，一些很小的梯度可能下溢，一些大数可能溢出。

因此，“更省空间”和“训练足够稳定”是两个需要同时考虑的问题。BF16 也不是在所有意义上都等价于 FP32；矩阵乘、归一化、统计和参数更新可以使用不同的精度组合。

### 4.3 改 dtype 不是免费操作

```python
y = x.to(torch.bfloat16)
```

若确实需要改变格式，通常要为目标数据分配空间并逐元素转换。转换期间，源和目标还可能同时存在。

这就是为什么资源核算不能只看“最终结果占多少”，还要看**运行过程中哪几份数据同时存在**。

## 5. storage、stride 与视图

### 5.1 形状背后是一份底层存储

建立一个 FP32 张量：

```python
import torch

x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
```

显示出来是：

```text
[[0, 1, 2],
 [3, 4, 5]]
```

底层存储可以理解为：

```text
[0, 1, 2, 3, 4, 5]
```

PyTorch 需要用 shape、stride 和起始偏移等元数据，解释怎样从这排数字中找某个元素。

### 5.2 stride 是沿某个轴移动一步，要跨多少元素

对于上述 `x`：

```text
shape  = (2,3)
stride = (3,1)
```

这表示第一维下标增加 1，要跨过 3 个元素；第二维下标增加 1，要跨过 1 个元素。

忽略本例为零的起始偏移：

$$
\operatorname{offset}(x[i,j])=3i+j
$$

例如：

$$
\operatorname{offset}(x[1,2])=1\times3+2\times1=5
$$

所以取到底层第 6 个元素，也就是数字 5。

**PyTorch 的 stride 通常以元素为单位。若要换算字节偏移，还要乘 `element_size()`。**

### 5.3 转置可以只改变读法

```python
y = x.transpose(0, 1)
```

`y` 看起来是：

```text
[[0, 3],
 [1, 4],
 [2, 5]]
```

但普通稠密张量的这种转置可以共享原存储，不必先把所有数字重新复制排列。

此时：

```text
shape  = (3,2)
stride = (1,3)
```

取 `y[2,1]` 时：

$$
\operatorname{offset}(y[2,1])=2\times1+1\times3=5
$$

仍取到 5。

这就叫视图：**多个张量对象，可以按不同规则解释同一份底层数据。**

### 5.4 共享存储的两个直接后果

第一，视图很便宜，但不代表得到一份独立副本。修改共享的数据，可能影响其他视图。需要独立数据时，应明确考虑 `clone()` 等复制操作。

第二，一个很小的切片也可能让原本很大的存储继续存活。例如从大张量取一小块并保留该视图，底层整块存储可能仍不能释放。

因此，计算 tensor 的数量、逻辑元素数量和实际存储大小，是三个相关但不完全相同的问题。更多机制见 [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)。

## 6. reshape、transpose 和 contiguous

### 6.1 形状相同，不代表含义相同

继续使用：

```text
x =
[[0,1,2],
 [3,4,5]]
```

执行：

```python
x.reshape(3, 2)
```

结果是：

```text
[[0,1],
 [2,3],
 [4,5]]
```

而：

```python
x.transpose(0, 1)
```

结果是：

```text
[[0,3],
 [1,4],
 [2,5]]
```

两个结果的 shape 都是 `[3,2]`，但数字位置不同。

`reshape` 是按原张量的逻辑元素顺序重新组织形状；`transpose` 是交换轴的语义。**要把 batch 轴与位置轴交换，不能只因为维度相同就用 reshape。**

### 6.2 三类常见操作

| 操作 | 含义 | 普通稠密张量下是否复制 |
| --- | --- | --- |
| `transpose` / `permute` | 交换两个轴或重排多个轴 | 通常返回共享存储的视图 |
| `view` | 用兼容的形状重新解释原存储 | 不复制；布局不兼容时会报错 |
| `reshape` | 得到指定形状 | 可以返回视图，也可能复制 |
| `contiguous` | 得到指定连续内存布局 | 已满足时返回自身，否则复制 |

不要把“非连续”直接等同于“任何 view 都不能做”。是否可以 view，取决于具体 stride 与目标形状是否兼容。

### 6.3 contiguous 为什么可能导致显存峰值？

假设 `x` 占 2 GB：

```python
y = x.transpose(1, 2)
```

可能只新增少量元数据。

接着：

```python
z = y.contiguous()
```

如果 `y` 的布局不符合要求，就可能分配新的 2 GB 数据区并复制。在旧存储还活跃时，两份存储会同时存在。

因此，性能排查应结合：

```python
print(x.shape)
print(x.stride())
print(x.is_contiguous())
print(x.dtype)
print(x.device)
```

需要复制并不代表代码一定写错。问题是要知道复制发生在哪里、数据有多大，以及是否有更合适的布局。

## 7. device、广播与 einsum

### 7.1 device 决定数据在哪里

普通默认创建的张量通常在 CPU 上；创建时指定设备或改变默认设备，会影响这一行为。

```python
x_cpu = torch.zeros(2, 4, 768)

# 以下两行需要可用的 CUDA 设备。
x_gpu = x_cpu.to("cuda")
y_gpu = torch.zeros(2, 4, 768, device="cuda")
```

`to("cuda")` 不只是给张量贴一个标签。需要跨设备时，要真正把数据送到目标设备。

同一个运算的张量需要满足设备兼容要求。不要在循环中反复进行不必要的 CPU/GPU 来回复制；同时，也不需要为了“全部放 GPU”而无条件把整个训练数据集装进显存。

### 7.2 广播让较小张量匹配较大张量的轴

例如：

```text
x：[B,T,d]
b：[d]
x + b → [B,T,d]
```

表示每个位置都加上同一个长度为 `d` 的向量。

广播规则通常从末尾的轴开始匹配：对应维度相同，或其中一个维度为 1，才能兼容。广播不一定需要先物理复制一个完整的大输入张量，但运算输出及后续操作仍可能分配空间。

广播也会掩盖形状错误。例如本来想按位置缩放，却意外按特征缩放，某些维度碰巧相等时，程序可能照样运行。

### 7.3 einsum 把轴的关系写进表达式

普通矩阵乘：

```python
y = x @ w
```

可以写成：

```python
y = torch.einsum("mk,kn->mn", x, w)
```

意思是：

$$
Y_{m,n}=\sum_k X_{m,k}W_{k,n}
$$

`m,n` 保留在输出里，`k` 没出现在输出里，被收缩求和。

注意力中：

```python
scores = torch.einsum("bhtc,bhsc->bhts", q, k)
```

表示保留 batch、head、Query 位置和 Key 位置，对头内部的特征 `c` 求和。

不要仅检查最终 shape。测试可以用 `B=2,T=5,h=3,d_h=4` 这种各轴大小不同的配置，更容易暴露把位置轴和 head 轴弄反的问题。

## 8. 注意力中的张量形状逐步变化

本节按标准多头自注意力 MHA 说明，暂不引入 GQA、MQA、KV cache 等变体。主例子为：

```text
B=2，T=4，d=768，h=12，d_h=64
```

### 8.1 先为每个位置生成 Q、K、Value

输入：

```text
X：[B,T,d] = [2,4,768]
```

进行三次线性投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad \mathrm{Value}=XW_V
$$

数学上把这三个权重都写成 `[d,d]`，投影后形状仍是 `[B,T,d]`。

先用一个工作解释理解它们：

| 张量 | 可以理解成 |
| --- | --- |
| Q，Query | 当前这个位置用什么特征去寻找信息 |
| K，Key | 每个位置提供什么特征供其他位置匹配 |
| Value | 匹配以后实际要取回的信息 |

这个解释是帮助理解数据流的类比，不代表一个向量的单个维度有固定的人类语义。

### 8.2 拆成多个 head

把 768 维拆成 `12×64`：

```text
[2,4,768]
      ↓ 拆最后一维
[2,4,12,64]
      ↓ 交换位置轴与 head 轴
[2,12,4,64]
```

代码：

```python
q = q.reshape(B, T, h, d_h).transpose(1, 2)
k = k.reshape(B, T, h, d_h).transpose(1, 2)
v = v.reshape(B, T, h, d_h).transpose(1, 2)
```

也可借助 einops 表达拆分约束：

```python
from einops import rearrange

q = rearrange(q_before_split, "b t (h c) -> b h t c", h=h)
```

拆 head 不会凭空把特征元素增加 12 倍。`12×64` 总共仍然是 768；真正新增计算和数据的是投影、注意力等运算。

### 8.3 为什么分数矩阵是 T 乘 T？

只看一个样本的一个 head：

```text
Q：[T,d_h] = [4,64]
K：[T,d_h] = [4,64]
```

匹配分数：

$$
S=QK^\top
$$

形状：

```text
[4,64] @ [64,4] → [4,4]
```

`S[t,s]` 表示第 `t` 个位置的 Query 与第 `s` 个位置的 Key 的点积。

例如小向量的点积：

$$
[1,2]\cdot[3,4]=1\times3+2\times4=11
$$

每个位置都要与所有位置配对，所以是 `T×T`。恢复 batch 和 head 轴：

```text
scores：[B,h,T,T] = [2,12,4,4]
```

### 8.4 缩放、因果 mask 和 softmax

注意力权重为：

$$
P=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)
$$

这里 `M` 是因果 mask：允许的位置加 0，禁止看到未来的位置加负无穷，使 softmax 后这些位置的权重为 0。

`P[b,h,t,:]` 是某个 Query 位置，对各 Key 位置分配的权重。softmax 在最后的 Key 位置轴上进行。

### 8.5 用权重取回 Value 并合并各 head

$$
O=P\,\mathrm{Value}
$$

一个 head 内部：

```text
[T,T] @ [T,d_h] → [T,d_h]
```

所有 head 的输出：

```text
[B,h,T,d_h]
     ↓ 换轴
[B,T,h,d_h]
     ↓ 合并 head 和特征轴
[B,T,d]
```

再经过输出投影 $W_O$，得到注意力模块的输出。

下面是便于看形状的教学代码，省略 dropout、位置编码和完整模块封装；`Wq/Wk/Wv/Wo` 按数学约定存为 `[d,d]`，且需已正确初始化、放到兼容设备：

```python
import math
import torch

def attention_shape_demo(x, Wq, Wk, Wv, Wo, h):
    B, T, d = x.shape
    assert d % h == 0
    d_h = d // h

    q = (x @ Wq).reshape(B, T, h, d_h).transpose(1, 2)
    k = (x @ Wk).reshape(B, T, h, d_h).transpose(1, 2)
    v = (x @ Wv).reshape(B, T, h, d_h).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_h)
    future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    weights = scores.softmax(dim=-1)

    heads = weights @ v
    merged = heads.transpose(1, 2).reshape(B, T, d)
    return merged @ Wo
```

这是显式形成 `T×T` 中间矩阵的实现，适合帮助理解。它不是推荐用于所有大规模训练场景的高效注意力实现。

## 9. Autograd 与反向传播的数值例子

### 9.1 参数、激活和梯度先分清

| 名称 | 意思 | 例子 |
| --- | --- | --- |
| 参数 parameter | 模型通过训练调整的数字 | 线性层权重 $W$ |
| 激活 activation | 当前输入经过模型产生的中间结果 | 隐藏表示、Q/K/Value、FFN 中间结果 |
| 梯度 gradient | loss 对某个变量的导数 | $\partial\mathcal L/\partial W$ |
| 优化器状态 optimizer state | 优化器为更新参数保存的历史统计 | Adam 的 $m,v$ |

换一批输入，模型参数可以不变，但激活一般会变化。梯度是在当前参数和当前训练数据下算出的更新信息。

### 9.2 只有一个参数的模型

设：

$$
\hat y=wx,\qquad \mathcal L=\frac12(\hat y-y)^2
$$

取：

```text
x=3，w=2，正确答案 y=10
```

前向得到：

$$
\hat y=2\times3=6
$$

$$
\mathcal L=\frac12(6-10)^2=8
$$

### 9.3 用链式法则计算梯度

沿着：

```text
w → 预测值 y_hat → loss
```

计算：

$$
\frac{\partial\mathcal L}{\partial w}
=\frac{\partial\mathcal L}{\partial\hat y}
\frac{\partial\hat y}{\partial w}
=(\hat y-y)x
$$

代入数值：

$$
\frac{\partial\mathcal L}{\partial w}=(6-10)\times3=-12
$$

梯度为负，意味着在当前位置附近把 `w` 增大，可以降低 loss。

若使用学习率为 0.1 的简单梯度下降：

$$
w_{\text{new}}=w-0.1\times(-12)=3.2
$$

更新后预测变成 `3.2×3=9.6`，新 loss 为 `0.5×(9.6-10)²=0.08`，比 8 小很多。

### 9.4 PyTorch 怎样做同一件事

```python
w = torch.tensor(2.0, requires_grad=True)
prediction = w * 3
loss = 0.5 * (prediction - 10) ** 2

loss.backward()

print(w.grad)  # 本例应得到 -12
print(w)       # backward 本身没有把 w 更新成 3.2
```

`requires_grad=True` 表示需要沿相关计算关系求导。训练中的 `nn.Parameter` 通常默认需要梯度。

Autograd 在前向期间记录必要的运算关系，反向时按链式法则传播梯度。它通常不会显式构造整个模型的巨大完整 Jacobian，而是计算反向传播需要的梯度乘积。

### 9.5 为什么前向结果需要保存到反向？

刚才的公式：

$$
\frac{\partial\mathcal L}{\partial w}=(wx-y)x
$$

需要用到输入 `x` 和前向相关结果。

对于深层网络：

```text
输入 → 第一层输出 → 第二层输出 → 第三层输出 → loss
```

反向会使用其中某些张量来计算各层梯度。不是每个操作都必须保存所有输入输出，但所需数据必须能在反向时获得。

因此，训练前向不能总是“算完一层就把这层全部中间数据丢掉”。**保存反向需要的激活，是训练比单纯推理更占显存的重要原因。**

更多机制见 [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)。

## 10. 训练 API：清梯度、更新参数和关闭求导

### 10.1 标准的一次参数更新

```python
optimizer.zero_grad(set_to_none=True)

prediction = model(inputs)
loss = loss_fn(prediction, targets)

loss.backward()
optimizer.step()
```

| 调用 | 作用 |
| --- | --- |
| `zero_grad` | 清除上次遗留的参数梯度 |
| `model(inputs)` | 前向，计算预测 |
| `loss_fn(...)` | 计算训练目标 |
| `backward()` | 计算并累积梯度 |
| `step()` | 优化器根据梯度更新参数 |

**backward 不等于 step。计算出梯度，并不等于已经更新参数。**

### 10.2 为什么要清梯度？

PyTorch 通常会把本次反向得到的梯度加到已有 `.grad` 中，而不是自动替换。

如果不清梯度，多轮得到的结果会累加。梯度累积正是有意利用这个机制；普通每轮独立更新时，则需要在合适位置清梯度。

`set_to_none=True` 把相关 `.grad` 设为 `None`，与给现有梯度数组逐元素写 0 并不完全相同，也可能减少一些梯度内存和操作开销。

不要通过反复对同一份已释放中间结果的计算图调用 backward，来演示普通的多轮训练。通常每轮要重新前向构建本轮计算关系。

### 10.3 no_grad 与 eval 不同

```python
model.eval()

with torch.no_grad():
    prediction = model(inputs)
```

这里有两个独立动作：

- `eval()`：切换模块行为，例如关闭训练态 dropout；某些层还会改变统计量使用方式。
- `no_grad()`：这段计算不建立通常用于反向求导的计算图。

**只调用 eval，不代表关闭 autograd。** 后续继续训练时，应按需要调用 `model.train()`。

### 10.4 detach 是切断梯度路径，不是复制数据

```python
fixed_target = target.detach()
```

之后把 `fixed_target` 用作目标值，梯度不会再沿 `target` 原来的生成路径传回去。

它通常仍然共享底层存储。因此：

```python
target.detach()
```

不等于：

```python
target.detach().clone()
```

第二种还建立了独立数据副本。不要把“当作常量”“切断求导关系”和“物理复制数据”混为一谈。

## 11. Transformer 参数量逐项推导

### 11.1 先固定架构假设

本节统计下面这类 decoder-only 模型：

- 标准 MHA，Q/K/Value 的总投影宽度均为 $d$。
- 不带 bias。
- FFN 使用三个权重矩阵的 SwiGLU。
- 每个 block 有两个 RMSNorm，每个含 $d$ 个缩放参数。
- 总共 $L$ 个 block，末尾另有一个 RMSNorm。
- embedding 和 LM head 使用独立权重。
- 不额外加入可学习位置 embedding 等参数。

更换这些假设，参数量就要跟着修改。尤其不能把本节 `4d²` 无条件用于 GQA 或 MQA。

### 11.2 一个线性层的参数量

数学上写：

$$
Y=XW,\qquad W\in\mathbb R^{d_{\text{in}}\times d_{\text{out}}}
$$

则：

$$
N_W=d_{\text{in}}d_{\text{out}}
$$

如果带 bias，再加 $d_{\text{out}}$。

注意 PyTorch 的 `nn.Linear(in_features,out_features)` 实际把 `weight` 存成 `[out_features,in_features]`，前向相当于乘它的转置。这个存储约定不会改变参数个数。

### 11.3 注意力为什么是 4d 平方？

四个主要权重为：

$$
W_Q,\ W_K,\ W_V,\ W_O
$$

每个都是 $d\times d$，因此：

$$
N_{\text{attention}}=4d^2
$$

固定总宽度 $d$ 时，把 head 分得更多，单个 head 也会更窄；标准 MHA 的这四个大矩阵不因此按 head 数再额外乘一遍。

`[B,h,T,T]` 注意力分数不是参数。它由当前输入和参数计算出来，是激活。

### 11.4 SwiGLU 为什么是 3d d_ff？

一种数学写法是：

$$
\operatorname{FFN}(X)
=\left[\operatorname{SiLU}(XW_1)\odot(XW_3)\right]W_2
$$

其中：

| 矩阵 | 形状 | 参数个数 |
| --- | --- | --- |
| $W_1$ | $[d,d_{ff}]$ | $dd_{ff}$ |
| $W_3$ | $[d,d_{ff}]$ | $dd_{ff}$ |
| $W_2$ | $[d_{ff},d]$ | $dd_{ff}$ |

因此：

$$
N_{\text{FFN}}=3dd_{ff}
$$

若取 $d_{ff}=8d/3$：

$$
N_{\text{FFN}}=3d\times\frac83d=8d^2
$$

加上注意力，主要矩阵参数为 $4d^2+8d^2=12d^2$。

### 11.5 Norm、embedding 和输出层

每层两个 RMSNorm：

$$
N_{\text{norm/block}}=2d
$$

embedding 有 $V$ 行、每行 $d$ 个数：

$$
N_{\text{embedding}}=Vd
$$

独立 LM head 将 $d$ 维投影到 $V$ 维，也有 $Vd$ 个参数。最终 RMSNorm 再加 $d$。

整合后：

$$
N=L(4d^2+3dd_{ff}+2d)+2Vd+d
$$

### 11.6 完整算例：约 1 亿参数怎样算出来？

取：

```text
d=768，L=12，V=10000，d_ff=2048
```

每个 block：

| 部分 | 计算 | 参数个数 |
| --- | --- | --- |
| 注意力 | $4\times768^2$ | 2,359,296 |
| SwiGLU | $3\times768\times2048$ | 4,718,592 |
| 两个 RMSNorm | $2\times768$ | 1,536 |
| block 合计 | 三项相加 | 7,079,424 |

12 个 block：

$$
12\times7,079,424=84,953,088
$$

embedding 和独立 head：

$$
2\times10000\times768=15,360,000
$$

最后的 RMSNorm：768。

$$
N=84,953,088+15,360,000+768=100,313,856
$$

即约 **100.314M，或者 1.003 亿参数**。

### 11.7 哪些改动会改变参数量？

| 改动 | 参数量的主要变化 |
| --- | --- |
| batch size 增大 | 固定架构的参数量不变 |
| 输入长度增大 | 固定架构通常不变；若同时扩展可学习位置表，需另算 |
| 层数 $L$ 翻倍 | block 部分约翻倍 |
| 宽度 $d$ 翻倍，$d_{ff}$ 同比变大 | 主体矩阵参数约四倍 |
| 词表 $V$ 翻倍 | embedding 与 head 部分翻倍 |
| embedding/head 权重共享 | 减少一份独立 $Vd$ 参数 |

共享权重时，LM head 的投影运算仍要执行；**独立参数个数减少，不代表这项计算消失。**

## 12. 矩阵乘和反向传播的 FLOPs

### 12.1 工作量与速度不要混淆

| 写法 | 含义 | 类比 |
| --- | --- | --- |
| FLOP | 一次浮点运算的计数单位 | 一件工作 |
| FLOPs | 浮点运算总次数 | 总共有多少工作 |
| FLOP/s | 每秒浮点运算次数 | 每秒完成多少工作 |
| TFLOP/s | $10^{12}$ FLOP/s | 很大的计算速度单位 |

本文按一次乘法记 1 FLOP、一次加法记 1 FLOP 的常见口径。一次融合乘加即使由一条硬件指令完成，通常仍按 2 FLOPs 统计。

### 12.2 为什么矩阵乘约为 2mkn？

考虑：

$$
A_{m\times k}W_{k\times n}=Y_{m\times n}
$$

输出有 $mn$ 个元素。每个元素是长度为 $k$ 的点积：

$$
Y_{ij}=\sum_{r=1}^{k}A_{ir}W_{rj}
$$

需要 $k$ 次乘法、$k-1$ 次加法，因此精确的朴素点积计数是：

$$
mn(2k-1)
$$

大矩阵下忽略小边界项：

$$
F_{\text{matmul}}\approx2mkn
$$

### 12.3 线性层的前向计算量

把 `[B,T,d_in]` 的前两个轴合并，看成 $BT$ 行：

```text
[BT,d_in] @ [d_in,d_out] → [BT,d_out]
```

所以：

$$
F_{\text{linear,forward}}\approx2BTd_{\text{in}}d_{\text{out}}
$$

取 `B=2,T=4,d_in=d_out=768`：

$$
F=2\times2\times4\times768^2=9,437,184
$$

约 944 万 FLOPs。

### 12.4 反向为什么多出两次矩阵乘？

设：

$$
Y=XW,\qquad G=\frac{\partial\mathcal L}{\partial Y}
$$

上游已经给了输出梯度 $G$。本层需要计算：

$$
\frac{\partial\mathcal L}{\partial W}=X^\top G
$$

$$
\frac{\partial\mathcal L}{\partial X}=GW^\top
$$

若 `X:[m,k]`、`W:[k,n]`、`G:[m,n]`，则：

| 计算 | 矩阵形状 | 近似 FLOPs |
| --- | --- | --- |
| 前向 $XW$ | `[m,k] @ [k,n]` | $2mkn$ |
| 权重梯度 $X^\top G$ | `[k,m] @ [m,n]` | $2mkn$ |
| 输入梯度 $GW^\top$ | `[m,n] @ [n,k]` | $2mkn$ |

合计：

$$
F_{\text{forward+backward}}\approx6mkn
$$

即：

$$
F_{\text{training}}\approx3F_{\text{forward}}
$$

**训练合计约为前向的 3 倍；反向本身约为前向的 2 倍。**

这个近似针对主要矩阵乘法。如果某个输入不需要梯度、某些参数被冻结、使用重计算、还要考虑优化器更新等，实际计数会变化。

## 13. 6ND 的来源、用途与边界

### 13.1 从一个线性层推到整套主要矩阵

一个线性层有：

$$
N_W=d_{\text{in}}d_{\text{out}}
$$

个权重。对一个 token，前向约需 $2N_W$ FLOPs；包括通常的反向后约需 $6N_W$ FLOPs。

把主要稠密投影层加起来，对 $D$ 个 token：

$$
C\approx6ND
$$

这里的 $N$ 应当与该近似所覆盖的矩阵计算匹配。工程估算常用模型参数规模代替，但必须知道有哪些项没有被准确处理。

### 13.2 用它估算什么？

若按这个近似取 `N=1B=10^9`，累计训练 `D=100B=10^11` 个 token：

$$
C\approx6\times10^9\times10^{11}=6\times10^{20}\ \text{FLOPs}
$$

在其他条件近似相同时，模型规模翻倍或 token 总量翻倍，主要训练计算量分别约翻倍。

若要据此估算时间，还需要除以**实际持续达到的计算速度**，不能直接默认全程跑满厂家标称峰值。

### 13.3 为什么 N 不能无条件等于所有参数总和？

- 输入 embedding 主要是查表，不是每个 token 都乘完整个 embedding 矩阵。
- LM head 通常仍要进行大投影。它与 embedding 是否共享权重，会改变独立参数数目，却不会直接取消投影。
- Norm 等参数的运算模式与大矩阵乘不同。
- MoE 每个 token 只激活部分专家，不能把总参数简单当作所有 token 都参与的 dense 参数。
- 稀疏、冻结、重计算等也会改变实际运算。

所以，`6ND` 是有用的资源估算工具，不是适用于所有架构和执行方式的精确恒等式。

### 13.4 注意力还有与 T 平方有关的计算

`QKᵀ` 和 `P×Value` 不是权重矩阵投影，却同样要计算。

一个 head 的 `QKᵀ`：

```text
[T,d_h] @ [d_h,T]
```

约需 $2T^2d_h$ FLOPs。全部 batch 和 head：

$$
2BhT^2d_h=2BT^2d
$$

`P×Value` 约需同样的量，因此每层前向：

$$
F_{\text{attention core}}\approx4BT^2d
$$

粗略将反向也按三倍前向计入，并考虑 $L$ 层：

$$
C_{\text{attention core,step}}\approx12LBT^2d
$$

这里按完整稠密矩阵乘计数，忽略 softmax 等操作。能跳过因果上三角计算的实现会改变常数，具体 kernel 的计算也可能包含重算。

### 13.5 序列翻倍，哪一项翻几倍？

固定 $B,d,L$：

| 部分 | 随 $T$ 的主要变化 | $T$ 翻倍 |
| --- | --- | --- |
| Q/K/Value 和 FFN 等线性投影 | $O(T)$ | 约 2 倍 |
| 显式注意力分数元素数 | $O(T^2)$ | 约 4 倍 |
| 注意力核心两个乘法 | $O(T^2)$ | 约 4 倍 |
| logits 元素数 | $O(T)$ | 约 2 倍 |

这张表讨论的是**单步固定序列条数**。如果固定的是总训练 token 数 $D$，改变 $T$ 还会改变需要的训练步数，不要把两种比较方式混在一起。

## 14. 训练显存从哪里来

### 14.1 模型文件大小不等于训练峰值

模型文件通常主要包含参数；训练还需要：

```text
参数
+ 参数梯度
+ 优化器状态
+ 为反向保留的激活
+ 临时张量与库 workspace
+ CUDA 上下文、分配器缓存等其他占用
```

其中 logits 可以属于激活的一部分。核算时若单独列 logits，就不要又在“全部激活”里重复加一遍。

真正关心的是：**某个时刻，哪些存储同时还活着？**

### 14.2 Adam 为什么有两份额外状态？

Adam 会为每个参数保留梯度的一阶、二阶矩估计：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

粗略理解：`m` 平滑跟踪梯度方向，`v` 平滑跟踪梯度平方的大小。每个参数都对应一份 `m` 和一份 `v`，不是整个模型只需要两个标量。

### 14.3 16N byte 怎样得到？

按单设备未分片、参数与梯度为 FP32、Adam 的两份主要状态也为 FP32 的口径：

| 内容 | 元素数 | 每元素 byte | 总 byte |
| --- | --- | --- | --- |
| 参数 | $N$ | 4 | $4N$ |
| 梯度 | $N$ | 4 | $4N$ |
| Adam 的 $m$ | $N$ | 4 | $4N$ |
| Adam 的 $v$ | $N$ | 4 | $4N$ |
| 合计 |  |  | $16N$ |

对于 1B 参数：

$$
16\times10^9\ \text{byte}=16\ \text{GB}
$$

这是指定状态全部存在时的常用预算，不是所有 Adam 实现永远固定的开销。状态可能延迟到第一次 step 才建立；梯度设为 `None` 后也可能释放。优化器实现还可能使用额外临时空间。

如果是分片、offload、不同精度或其他优化器，必须分别统计。

### 14.4 约 1 亿参数模型的静态预算

第 11 节得到：

$$
N=100,313,856
$$

则：

$$
16N=1,605,021,696\ \text{byte}
$$

约为：

```text
1.605 GB，或约 1.495 GiB
```

这不能推出“训练只需 1.605 GB”，因为激活和临时空间还没有计入。

### 14.5 一个普通隐藏张量有多大？

令 `B=8,T=2048,d=768`，使用 FP32：

$$
M_{\text{hidden}}=8\times2048\times768\times4
=50,331,648\ \text{byte}
$$

即约 50.3 MB。

但一层内部有多个中间结果，反向也可能需要保留其中一些。不能只算一个 `[B,T,d]`，就声称已经算完全部激活。

### 14.6 显式注意力矩阵有多大？

再取 `h=12`，一个 FP32 的 `[B,h,T,T]` 张量：

$$
M_{\text{scores}}=8\times12\times2048^2\times4
=1,610,612,736\ \text{byte}
$$

约 1.61 GB。这只是一个这样的矩阵，不是整个模型的总激活。

若把 `T` 从 2,048 增到 4,096，在其他条件相同时，它会增大到约 6.44 GB。

FlashAttention 通过分块与减少 HBM 读写，避免按朴素方式保存完整巨大中间矩阵。它改善显存和执行效率，但不把标准稠密注意力的核心计算复杂度变成关于 $T$ 的线性函数。参见 [FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。

### 14.7 logits 有多大？

这次改用原提纲中的较大词表例子 `V=50000`，其他取 `B=8,T=2048`，FP32：

$$
M_{\text{logits}}=8\times2048\times50000\times4
=3,276,800,000\ \text{byte}
$$

约 3.28 GB。

这是显式保存完整 FP32 logits 的数据量。真实损失实现可能采用不同精度、融合或分块方式，其峰值不能只用这个例子断言。

### 14.8 峰值是同一时刻的总量，不是把每个阶段的峰值相加

训练中可能先后出现：

```text
加载模型时的峰值
前向结束附近的激活峰值
反向过程中的激活、梯度与临时量组合峰值
optimizer.step 建立状态或使用临时量时的峰值
```

整个过程的峰值是这些时刻总占用的最大值，不能把“前向最高值 + 反向最高值 + step 最高值”再相加。

### 14.9 allocated、reserved 和 nvidia-smi

| 指标 | 主要观察什么 |
| --- | --- |
| `torch.cuda.memory_allocated()` | 当前由张量占用的显存 |
| `torch.cuda.memory_reserved()` | PyTorch 缓存分配器管理的显存 |
| `torch.cuda.max_memory_allocated()` | 自上次重置以来的张量分配峰值 |
| `nvidia-smi` | 设备或进程的更广泛显存占用，包括上下文等 |

例如：

```text
allocated = 10 GB
reserved  = 12 GB
```

通常表示 12 GB 中有约 10 GB 正被张量使用，另有约 2 GB 是分配器管理但当前未用于活跃张量的部分。**不是总共 22 GB。**

`empty_cache()` 可以归还未使用缓存，不能把仍被活跃张量引用的数据直接释放。reserved 较高本身也不能证明存在张量泄漏。见 [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)。

## 15. 混合精度为何不保证总显存减半

### 15.1 先问哪部分改成了低精度

某张量从 FP32 变为 BF16，元素数不变时，数据大小减半。但训练系统有许多不同用途的张量，不一定全部同时改精度。

例如一种混合精度布局是：

| 内容 | 精度 | 每个参数对应 byte |
| --- | --- | --- |
| 用于计算的参数 | BF16 | 2 |
| 梯度 | BF16 | 2 |
| 主参数副本 master weights | FP32 | 4 |
| Adam 的 $m$ | FP32 | 4 |
| Adam 的 $v$ | FP32 | 4 |
| 合计 |  | 16 |

仍然可能是 `16N byte`，但激活等部分可能缩小，矩阵乘也可能更快。

这个表是一个明确的实现方案示例，不代表开启某个 PyTorch 开关后必然得到这套布局。

### 15.2 为什么可能保留 FP32 主参数？

较低精度对相近数值的区分能力有限。如果每次更新都很小，直接加到低精度参数上，更新可能被舍入掉。

FP32 主参数副本可以以较高精度累计更新，再转换成低精度计算所需的参数。是否采用主参数、怎样保存梯度和状态，取决于训练框架与优化器实现。

### 15.3 autocast 不等于把模型全部变成 BF16

例如：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)
```

autocast 会按操作的规则选择计算精度，并不保证所有输入、参数、输出与状态全部改为 BF16。某些需要较高稳定性的操作可能使用 FP32。

若模型参数本来一直保留 FP32，梯度和 Adam 状态也为 FP32，那么这些训练状态的主要预算仍可能接近 `16N`。

### 15.4 做显存比较时写清配置

一个可复核的记录至少应说明：参数 dtype、梯度 dtype、优化器状态 dtype、是否有 master weights、是否分片，以及激活采用什么计算方式。

“用了 BF16，所以 1B 训练状态一定从 16 GB 降到 8 GB”缺少必要条件。混合精度的具体用法见 [PyTorch AMP 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)。

## 16. 算术强度与 Roofline

### 16.1 算得快，还需要拿得到数据

GPU 执行运算时，需要在显存与计算单元附近的存储层级之间搬运数据。这里讨论的 HBM 带宽，不应与 CPU 到 GPU 的 PCIe 复制带宽混淆。

设一个工作负载需要：

- $F$ 次浮点运算。
- 从指定存储层级读写 $Q$ byte。
- 适用于该精度和运算口径的峰值计算能力为 $P_{peak}$。
- 内存带宽为 $BW$。

理想计算时间至少为：

$$
t_{\text{compute}}=\frac{F}{P_{peak}}
$$

理想搬运时间至少为：

$$
t_{\text{memory}}=\frac{Q}{BW}
$$

即使计算和搬运能很好地重叠，总时间也不会小于其中较大者：

$$
t\ge\max\left(\frac{F}{P_{peak}},\frac{Q}{BW}\right)
$$

现实还可能有启动、同步、调度等额外开销，因此这是一种理想边界分析。

### 16.2 算术强度的单位

定义：

$$
I=\frac{F}{Q}
$$

单位是 FLOP/byte，意思是每搬一个字节，做多少运算。

吞吐上界为：

$$
P\le\min(P_{peak},BW\cdot I)
$$

- 如果搬数据更慢，称为 memory-bound。
- 如果计算更慢，称为 compute-bound。

同一个操作在不同尺寸、精度、设备和数据复用方式下，可能处于不同区域。

### 16.3 一个假想硬件例子

假设某一指定计算口径下：

```text
峰值计算：100 TFLOP/s
内存带宽：1 TB/s
```

平衡点：

$$
I_* =\frac{100\times10^{12}}{10^{12}}=100\ \text{FLOP/byte}
$$

每字节只做很少计算的工作负载，更容易先被带宽限制；数据复用充分、每字节做很多计算的工作负载，更有机会接近计算上界。

这里采用假想数字帮助理解，不是对某块真实 GPU 的规格声明。

### 16.4 为什么逐元素 ReLU 常受带宽限制？

$$
y=\max(x,0)
$$

FP32 下，每元素读入 4 byte、写出 4 byte，只做一次很简单的比较/取最大值操作。

按原讲义把这类操作约记为一次操作的教学口径：

$$
I\approx\frac18
$$

严格 FLOP 统计对比较指令的处理可能不同，但不改变其“计算很少、搬运较多”的特点。

把加法等简单操作单独启动很多次，可能主要在反复读写中间结果。算子融合的一个意义，就是减少这些中间结果往返显存。

### 16.5 为什么大矩阵乘可能不同？

对于 FP32 的方阵乘：

$$
A_{s\times s}B_{s\times s}=C_{s\times s}
$$

计算量约为 $2s^3$。如果理想地把两个输入各读一次、输出写一次，数据量约为：

$$
Q=3\times4s^2=12s^2\ \text{byte}
$$

所以理想算术强度：

$$
I\approx\frac{2s^3}{12s^2}=\frac{s}{6}
$$

当 $s=1024$，约为 170.7 FLOP/byte。

矩阵元素参与多个输出的计算，分块实现可以复用已加载的数据。上述读写量是理想复用示意，真实流量还受缓存、分块和实现限制。

### 16.6 小矩阵乘也可能慢

如果矩阵很小，可能无法给 GPU 提供足够多并行工作；kernel 启动等固定成本也可能占很大比例。

因此，不能从“大矩阵乘有机会 compute-bound”推成“任何矩阵乘都能接近峰值”。标称 Tensor Core 峰值也不代表所有逐元素指令都能达到相同运算速度。

更完整的边界推导可参考 [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)。

## 17. MFU：模型利用了多少峰值算力

### 17.1 MFU 的问题是什么？

MFU，Model FLOPs Utilization，关注按模型计算量估算的实际训练运算速度，占参与训练硬件理论峰值的多少。

若采用 `6ND` 近似：

$$
\operatorname{MFU}\approx
\frac{6N\times\text{tokens/s}}{\text{参与训练设备的总峰值 FLOP/s}}
$$

分子与分母必须使用可比较的精度、设备数量和计算口径。

### 17.2 一个假想数值例子

假设：

```text
模型计算规模 N：1B
所有训练设备合计吞吐：20000 token/s
这些设备合计峰值：300 TFLOP/s
```

按近似计算：

$$
6\times10^9\times20000=120\times10^{12}\ \text{FLOP/s}
$$

因此：

$$
\operatorname{MFU}\approx\frac{120}{300}=40\%
$$

### 17.3 这个比例不代表什么？

MFU 不等于 `nvidia-smi` 显示的设备忙碌比例，也不直接代表模型质量。

比较 MFU 时，应注明：

- token/s 是单卡还是所有卡合计。
- 峰值对应哪种 dtype，是否使用普通 dense 口径。
- 分子是否计入注意力额外计算。
- 是否使用 checkpoint 重算；模型所需 FLOPs 的常用分子通常不把所有额外重算都算成有效模型工作。
- token 计数包含哪些位置。

不能把带结构化稀疏加速的理论峰值，直接用作普通稠密训练的分母。

## 18. 梯度累积：小批次怎样组成大批次

### 18.1 为什么要分 microbatch？

假设一次更新希望使用 8 条序列，但每次前向最多放下 2 条。

可以分四次：

```text
第1组2条：前向 → 反向 → 梯度留下
第2组2条：前向 → 反向 → 梯度继续相加
第3组2条：前向 → 反向 → 梯度继续相加
第4组2条：前向 → 反向 → 梯度继续相加
最后统一 optimizer.step 一次
```

四次过程中，参数保持不变；最后才更新。这与每组之后都 step 是不同的训练过程。

### 18.2 等大 microbatch 的平均损失

若四个 microbatch 含相同数量的有效训练单位，并且各自 loss 是相同口径的 mean：

$$
\mathcal L=\frac14(\mathcal L_1+\mathcal L_2+\mathcal L_3+\mathcal L_4)
$$

求导具有线性性质：

$$
\nabla\mathcal L=\frac14\sum_{i=1}^{4}\nabla\mathcal L_i
$$

所以可以这样写：

```python
optimizer.zero_grad(set_to_none=True)

# 假设恰好四组，且每组有效 token 数相同。
for inputs, targets in four_microbatches:
    prediction = model(inputs)
    loss = loss_fn(prediction, targets)  # 当前组的 mean loss
    (loss / 4).backward()

# 如需梯度裁剪，应在累积完成后、step 前按配置处理。
optimizer.step()
```

### 18.3 省显存的关键是每组及时 backward

每组前向之后立刻反向，通常可以释放这组不再需要的中间结果。下一组不必同时保留上一组全部激活。

下面这种写法可能失去预期节省：

```python
losses = []
for inputs, targets in four_microbatches:
    losses.append(loss_fn(model(inputs), targets))

sum(losses).backward()
```

因为这些 loss 可能让四份前向计算图及相应保存张量一起存活到最后。

梯度累积主要降低每次激活规模，不会把参数或 Adam 状态也按四分之一缩小。

### 18.4 有效 batch 为什么是 bap？

设每张设备每次处理 $b$ 条序列，累积 $a$ 次，数据并行有 $p$ 张设备：

$$
B_{\text{effective}}=bap
$$

例如：

$$
b=2,\ a=4,\ p=8\quad\Rightarrow\quad B_{\text{effective}}=64
$$

若每条都是 $T$ 个有效 token，这次更新的有效 token 数是 $bapT$。有 padding 或其他 mask 时，要按实际有效位置计数。

分布式框架可能默认对设备间梯度取平均；实现不等长全局 token 平均时，还需要结合这一归约规则正确缩放。

### 18.5 有效 token 数不同，为什么不能机械除以 a？

假设两个 microbatch 的有效 token 数是：

```text
n1=100，n2=10
```

每组内部先计算 token mean loss，分别得到 $\mathcal L_1,\mathcal L_2$。

全体 110 个 token 的 mean 应为：

$$
\mathcal L=\frac{100\mathcal L_1+10\mathcal L_2}{110}
$$

而不是：

$$
\frac{\mathcal L_1+\mathcal L_2}{2}
$$

后者把第二组的每个 token 赋予了过高的权重。

一般情况下，单设备上可以按：

$$
\mathcal L=\sum_i\frac{n_i}{\sum_j n_j}\mathcal L_i
$$

组织梯度累积。也可让每组计算有效 token loss 的和，再除以整个更新窗口的有效 token 总数。

### 18.6 “等价”指什么？

在参数固定、目标可按样本或 token 相加、归一化一致时，累加对应同一个平均目标的梯度。

不同浮点求和顺序、dropout 随机数和依赖 batch 的层可能让实际数值不逐位相同。学习率调度和优化器更新次数也应以预期的参数更新粒度组织。

FP16 配合 GradScaler 时还要保持一个累积窗口内的缩放一致，在累积完成后再做相应的 unscale、裁剪、step 和 update。见 [PyTorch gradient accumulation 示例](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)。

## 19. Activation checkpointing：少存一些、重新算一些

### 19.1 为什么可以少存？

普通训练在前向保存反向需要的中间张量。重计算可以保留某个区间的必要边界输入，在反向需要内部数据时，再执行该区间的部分前向。

例如：

```text
一个 block 的输入
   ↓
Q/K/Value、注意力、FFN 等内部计算
   ↓
一个 block 的输出
```

不必按普通方式把所有内部中间数据一直保存到反向；需要时从保留的边界重新生成。

### 19.2 用什么换什么？

| 改变 | 原因 |
| --- | --- |
| 激活峰值下降 | 少保存部分反向所需的中间数据 |
| 计算量增加 | 反向前后重做部分前向运算 |
| 参数量不变 | 权重矩阵没有因此变小 |
| Adam 状态不变 | 优化器仍需要相应历史统计 |

因此，checkpoint 不适合单独解决“模型状态本身就完全放不下”的问题。

### 19.3 为什么常见示意是 3F 变 4F？

若把前向计算量记为 $F$，普通训练主要矩阵乘近似：

$$
F+2F=3F
$$

如果再额外重做约一次前向：

$$
F+F+2F=4F
$$

相当于该极简模型下计算量多约三分之一。这不是通用性能承诺：选择性重算、区间划分、提前停止重算等都会改变开销，wall time 也不只由 FLOPs 决定。

### 19.4 PyTorch 接口与语义

```python
from torch.utils.checkpoint import checkpoint

# block 是一段可调用的模型计算，x 是它的输入。
y = checkpoint(block, x, use_reentrant=False)
```

重算应与原前向一致。默认机制会处理部分随机数状态，以尽可能保持 dropout 等行为一致，但存在设备和状态管理限制。

如果 block 在外部列表里记录数据、修改全局变量、更新某些状态，重复执行可能造成与普通前向不同的副作用。使用时应读 [PyTorch checkpoint 文档](https://docs.pytorch.org/docs/stable/checkpoint.html)，不要把它理解成任意函数都可以无条件执行两次。

## 20. 怎样正确测量训练速度

### 20.1 CUDA 为什么会让普通计时失真？

CUDA 通常异步执行。Python 提交一个矩阵乘后，可能在 GPU 尚未完成时就继续运行。

因此：

```python
start = time.perf_counter()
y = x @ w
elapsed = time.perf_counter() - start
```

可能主要测到提交任务的时间，而不是 GPU 完成这项工作的时间。

### 20.2 先明确测量对象

| 测量目标 | 计时范围应包含什么 |
| --- | --- |
| 单个算子 | 指定算子的执行；数据通常预先准备好 |
| 稳定训练 step | 清梯度、前向、loss、反向、优化器更新 |
| 端到端训练吞吐 | 视目标还需包含数据加载、传输、通信、日志等 |

排除某部分没有错，但必须说明。不能把排除数据加载的 kernel benchmark，直接叫作端到端训练速度。

### 20.3 warmup 为什么重要？

前几次运行可能包含：初始化、显存申请、编译、算法选择或缓存建立。

需要测稳定状态时，先运行若干次，再开始正式计时。输入形状或执行路径发生变化时，之前的 warmup 也未必覆盖新情况。

### 20.4 用同步测一组完整 step

以下针对 CUDA，假设 `train_step()` 已包含完整前向、反向和优化器更新；是否包含数据加载由调用者明确决定：

```python
import time
import torch

for _ in range(10):
    train_step()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
num_steps = 20

for _ in range(num_steps):
    train_step()

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

seconds_per_step = elapsed / num_steps
peak_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
print(seconds_per_step, peak_allocated_gb)
```

这是单组平均 step 时间。可以重复多组，记录中位数、波动范围和峰值显存，避免只挑一次最快的结果。

同步整个设备适合帮助理解和测量相应边界；多 stream 或更细粒度场景还要明确依赖关系。也可以用 CUDA events 记录 GPU 时间，但计时边界仍需与被测工作匹配。

Mac 的 MPS、CPU 等后端需要使用相应测量方法，不能直接照抄 CUDA 同步接口。相关机制见 [PyTorch CUDA asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)。

### 20.5 token/s 怎样算？

没有忽略位置、固定序列长度时，一张设备一次更新处理 $baT$ 个 token；数据并行合计可按 $bapT$ 计算。

$$
\text{tokens/s}=\frac{\text{计时区间内按约定统计的 token 总数}}{\text{区间秒数}}
$$

如果计时的是 microstep，不能又把累积次数乘进分子；如果计时的是完整 optimizer step，则应计入该窗口全部 microbatch。包含 padding 位置还是只计有效训练位置，也要注明。

### 20.6 item 为什么可能影响流水？

```python
python_number = loss.item()
```

要把 GPU 上的标量取回 Python，通常需要等待相关计算完成。把这类同步放在每个细粒度操作后面，会改变原本的异步执行方式。

日志需要数值并不意味着永远不能用 item；应按合适频率记录，且知道它是否被计入测量。

### 20.7 算子快三倍，为何训练未必快三倍？

假设原来一次训练：

```text
注意力：40 ms
其他：  60 ms
总计： 100 ms
```

注意力提速三倍后：

```text
注意力：13.3 ms
其他：  60.0 ms
总计：  73.3 ms
```

整体加速比：

$$
\frac{100}{73.3}\approx1.36
$$

一般写作 Amdahl 定律：若原耗时占比为 $f$ 的部分加速 $s$ 倍，其他不变：

$$
\operatorname{speedup}=\frac1{(1-f)+f/s}
$$

所以先用 profiler 找占时间最多的部分。一个只占总时间 5% 的环节，即使无限加速，也不能让整体快一倍。

## 21. 遇到 OOM 或训练慢时怎样定位

### 21.1 先判断是哪一类内存

| 现象或主要占用 | 优先检查 | 可能有效的方向 |
| --- | --- | --- |
| 模型状态本身就很大 | 参数、梯度、优化器状态精度与数量 | 更小模型，分片，按需求 offload |
| batch 稍大就 OOM | 激活、logits、临时量随 batch 的增长 | 减少 microbatch，配合梯度累积 |
| 序列变长后占用猛增 | 显式注意力矩阵与长序列激活 | 高效注意力，缩短上下文，重计算 |
| 输出层附近占用很大 | `[B,T,V]` logits 和损失实现 | 减小 microbatch，适用时用融合或分块损失 |
| 第一次 optimizer.step 才 OOM | Adam 延迟建立状态、step 临时量 | 核算状态与 workspace，考虑实现或分片 |
| 改布局或精度时突然出现峰值 | contiguous、cast、设备复制 | 排查同时存在的副本与不必要转换 |
| reserved 高但 allocated 没持续增长 | 缓存分配器行为 | 先区分缓存与活跃张量，再排查 |
| 每步 allocated 持续增长 | 被列表、日志、计算图等引用的张量 | 检查保留引用与不必要 retain_graph |

分片的基本思想是把原来每张设备都完整保存的某些训练状态，分摊到多张设备；FSDP、ZeRO 等方案的具体通信和状态划分在后续系统内容中展开。

### 21.2 为什么减小 batch 有时帮助不大？

若大部分显存是固定的参数和优化器状态，减小 batch 只减少一部分激活，并不会让这些固定状态一起变小。

反过来，若峰值主要来自大 batch 激活，梯度累积就可能很有用。

因此，必须先定位占用来源，再选优化。不能把一种办法当作所有 OOM 的通用答案。

### 21.3 几种配置变化的资源直觉

| 变化 | 独立参数量 | 普通隐藏激活 | 显式注意力分数 | 主要线性层计算 |
| --- | --- | --- | --- | --- |
| $B$ 翻倍 | 不变 | 约 2 倍 | 约 2 倍 | 约 2 倍 |
| $T$ 翻倍 | 固定架构不变 | 约 2 倍 | 约 4 倍 | 约 2 倍 |
| $d$ 翻倍，$d_{ff}$ 同比增加，固定 $h$ | 主体约 4 倍 | 约 2 倍 | 元素数不因 $d$ 改变 | 主体约 4 倍 |
| $L$ 翻倍 | block 部分约 2 倍 | 普通保存方式的相关部分约增加 | 各层相关保存量约增加 | block 部分约 2 倍 |

这里是分别观察各项的比例，不能直接等同于整个训练峰值的相同比例。宽度变化还会影响注意力核心乘法的计算；head 数若同时变化，分数矩阵也要另算。

### 21.4 训练慢时的检查顺序

1. 明确测的是算子、训练 step 还是端到端吞吐，并正确同步计时。
2. 检查数据加载与设备传输是否使 GPU 等待。
3. 用 profiler 观察耗时最大的算子、通信和同步。
4. 对大矩阵乘检查尺寸、dtype、布局；对简单操作检查反复搬运和 kernel 启动。
5. 只针对已确认的主要瓶颈优化，再用相同口径测量前后差异。

优化后的结果还应保持训练目标、有效 batch、序列长度和精度口径可比较。不能仅凭 step 时间下降就认定同等训练任务更快完成。

## 22. 术语表与自测解答

### 22.1 术语表

| 术语 | 本文中的含义 |
| --- | --- |
| Shape | 各个轴的元素数量 |
| Dtype | 元素的表示格式与精度 |
| Device | 张量所在的设备 |
| Storage | 保存底层元素的存储 |
| Stride | 下标沿某轴增加 1 时跨过的元素数 |
| View | 共享底层存储、用另一种元数据解释数据的张量 |
| Contiguous | 满足指定连续布局要求 |
| Parameter | 模型训练要调整的数字 |
| Activation | 一批输入经过模型产生的中间结果 |
| Gradient | 损失对某个变量的导数 |
| Autograd | 自动记录求导关系并执行梯度传播的机制 |
| Optimizer state | 优化器保存的历史统计等状态 |
| FLOPs | 浮点运算总量 |
| FLOP/s | 每秒浮点运算速度 |
| HBM bandwidth | GPU 显存与计算侧之间的数据搬运带宽 |
| Arithmetic intensity | 每搬一个字节完成的运算量 |
| Memory-bound | 数据搬运成为主要限制 |
| Compute-bound | 计算能力成为主要限制 |
| MFU | 按模型计算量估算的峰值算力利用比例 |
| Microbatch | 一次实际前向和反向处理的小批次 |
| Gradient accumulation | 多次反向累加梯度后再更新一次参数 |
| Activation checkpointing | 少保存部分激活，在反向需要时重新计算 |
| OOM | Out of memory，内存或显存不足 |
| Workspace | 库或算子执行时使用的临时工作空间 |

### 22.2 自测一：形状与存储

**问题：** `x` 为 `[[0,1,2],[3,4,5]]`。`x.reshape(3,2)` 与 `x.transpose(0,1)` 是否相同？

**解答：** 形状相同，但前者为 `[[0,1],[2,3],[4,5]]`，后者为 `[[0,3],[1,4],[2,5]]`。reshape 重组逻辑元素顺序，transpose 交换轴。

**问题：** 为什么两个 1 GB 张量对象不一定实际占 2 GB 数据存储？

**解答：** 它们可能是共享 storage 的视图，需要按底层实际存储判断，不能只数张量对象。

### 22.3 自测二：注意力

**问题：** `B=2,T=5,h=3,d_h=4`，Q 与 K 均为 `[2,3,5,4]`。`QKᵀ` 的分数形状是什么？

**解答：** `[2,3,5,5]`。最后一个 4 维特征轴收缩，保留 Query 位置和 Key 位置两个轴。

**问题：** 固定 $B,h$，`T` 从 2,048 增到 4,096，显式分数矩阵元素数怎样变化？

**解答：** 增为 4 倍，因为包含 $T^2$ 项。固定架构的模型参数量不因此变为 4 倍。

### 22.4 自测三：梯度与训练步骤

**问题：** $\hat y=wx$，$x=3,w=2,y=10$，$\mathcal L=\frac12(\hat y-y)^2$。梯度是多少？backward 后参数是多少？

**解答：** 梯度为 $(6-10)\times3=-12$。如果只调用 backward 而没有更新，`w` 仍为 2。使用学习率 0.1 的简单梯度下降后才得到 3.2。

**问题：** 为什么 eval 不等于 no_grad？

**解答：** eval 切换模块行为，no_grad 控制计算图记录；是两个不同机制。

### 22.5 自测四：参数量与显存

**问题：** 在第 11 节假设下，计算 `d=768,L=12,V=10000,d_ff=2048` 的参数量与 FP32 Adam 四组主要数组的预算。

**解答：**

```text
每个 block：7,079,424
全部 block：84,953,088
embedding + 独立 head：15,360,000
final RMSNorm：768
总参数：100,313,856
16N：1,605,021,696 byte ≈ 1.605 GB ≈ 1.495 GiB
```

**问题：** 能据此承诺训练峰值只有 1.605 GB 吗？

**解答：** 不能。还没计入激活、logits、临时计算空间和其他设备分配等；实际峰值还取决于同时存活的张量和具体实现。

### 22.6 自测五：FLOPs

**问题：** `[32,128] @ [128,256]` 前向 FLOPs 约为多少？通常的前向加反向约多少？

**解答：**

$$
F_{\text{forward}}\approx2\times32\times128\times256=2,097,152
$$

若输入梯度与权重梯度都需要计算，前向加反向约为：

$$
3\times2,097,152=6,291,456
$$

**问题：** 共享 embedding/head 后，为什么不能把减少后的独立参数量直接当成精确计算量变化？

**解答：** 同一份权重仍可能在输入查表和输出投影等位置使用，输出投影运算并未消失；运算次数和独立存储份数不同。

### 22.7 自测六：梯度累积

**问题：** 两个 microbatch 分别有 100 和 10 个有效 token，它们的 token mean loss 分别是 2 和 5。整体 token mean 是多少？

**解答：**

$$
\mathcal L=\frac{100\times2+10\times5}{110}=\frac{25}{11}\approx2.273
$$

不能直接得到 $(2+5)/2=3.5$，因为那是两个组的平均，而不是所有 token 的平均。

**问题：** `b=2,a=4,p=8`，每条 2,048 个有效 token，一次更新的有效序列数与 token 数是多少？

**解答：** 序列数 $2\times4\times8=64$；token 数 $64\times2048=131,072$。

### 22.8 自测七：性能与优化

**问题：** 注意力占原训练时间 40%，该部分提速 3 倍，整体提速多少？

**解答：**

$$
\operatorname{speedup}=\frac1{0.6+0.4/3}\approx1.36
$$

**问题：** 参数和 Adam 状态已占满设备，减小 microbatch 或开启激活重算一定有用吗？

**解答：** 它们主要减少激活，不能消除已经放不下的模型状态。应先核算固定状态，再考虑模型规模、分片或其他状态存放方案。

### 22.9 阅读后检查清单

- [ ] 能解释 `[B,T,d]` 每个轴的含义，并指出词表轴出现在哪里。
- [ ] 能手算 stride，并解释 transpose 为何可能不复制数据。
- [ ] 能说明 reshape 与 transpose 为何不等价。
- [ ] 能逐步写出多头注意力各张量的 shape。
- [ ] 能区分参数、激活、梯度和优化器状态。
- [ ] 能说出 backward 与 step 的区别。
- [ ] 能从矩阵大小推导 block 参数量。
- [ ] 能从点积推导 $2mkn$ 和通常的三倍训练近似。
- [ ] 能说明 $6ND$ 忽略或近似处理了哪些内容。
- [ ] 能从元素个数和 dtype 算 hidden、attention、logits 的大小。
- [ ] 能解释 FP32 Adam 的 $16N$ 预算及混合精度的不同布局。
- [ ] 能区分 allocated、reserved 和实际峰值。
- [ ] 能用搬运与计算解释 memory-bound 和 compute-bound。
- [ ] 能正确归一化等大和不等长 microbatch 的梯度累积。
- [ ] 能解释 checkpoint 用什么换什么。
- [ ] 能设计计时边界，并避免 CUDA 异步导致的错误结论。
- [ ] 遇到 OOM 时，能先判断是模型状态、激活、注意力还是 logits。

## 23. 参考资料

1. [Stanford CS336 官方课程网站](https://cs336.stanford.edu/)：课程与作业入口。
2. [官方第二讲 lecture_02.py](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)：张量、精度、计算量、算术强度和训练资源等主题。
3. [lvsir18：第二讲中文笔记](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes/blob/main/lectures/02-pytorch-resource-accounting.md)：本篇详细展开所参考的提纲。
4. [PyTorch：Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)：共享存储、视图、reshape 与 contiguous。
5. [PyTorch：Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：计算图、保存张量、梯度模式与 eval 的区别。
6. [PyTorch：CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)：异步执行、同步和 CUDA 显存管理。
7. [PyTorch：Automatic Mixed Precision examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)：混合精度、梯度累积与梯度缩放。
8. [PyTorch：torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)：激活重计算、接口选项和随机状态限制。
9. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)：Adam 的一阶、二阶矩状态。
10. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)：分块与 IO 感知的精确注意力。
11. [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/)：计算、内存带宽与算术强度的边界分析。
