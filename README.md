# CS336 Learning Notes

斯坦福 CS336（Spring 2026）中文学习笔记，用具体例子、公式拆解、张量形状追踪和手算过程理解语言模型。

这份笔记面向自学：先说明一个概念要解决什么问题，再解释每个符号、推导过程及其适用条件。目标是读完以后能自己算一遍、解释一遍，并知道如何在代码中使用。

## 阅读入口

| 讲次 | 主题 | 详细笔记 | 整理状态 |
| --- | --- | --- | --- |
| 第 01 讲 | 课程总览与 Tokenization | [从文本到语言模型：第一讲详细讲解](lectures/01-overview-tokenization-detailed.md) | 已整理 |
| 第 02 讲 | PyTorch、张量与资源核算 | [从张量到训练资源：第二讲详细讲解](lectures/02-pytorch-resource-accounting-detailed.md) | 已整理 |

这里的“已整理”表示已有学习文档，不代表完成了课程作业或性能实验。后续讲次随学习逐步补充。

## 建议阅读顺序

1. **先读第一讲，理解模型在处理什么。** 从文字、token ID 和 embedding 出发，理解下一个 token 预测、训练目标，以及 tokenizer 如何编码和解码。
2. **再读第二讲，理解这些计算怎样落到张量和硬件上。** 先掌握形状、存储与反向传播，再推导参数量、FLOPs 和显存，最后阅读梯度累积、重计算与性能测量。
3. **边读边手算。** 写出每个轴的含义，注明 dtype、设备与单位，再代入公式。第二讲文末提供自测和答案，可以先遮住答案独立完成。
4. **用代码和测量验证估算。** 教学算例帮助建立直觉；实际显存和速度还受算子实现、精度、硬件、数据长度和软件配置影响。

## 第一讲：从文本到语言模型

**核心问题：一段人类可读的文字，怎样变成模型可以学习和预测的数据？**

主要内容：

- token、词表、token ID 与 embedding 的区别。
- next-token prediction、交叉熵损失与 teacher forcing。
- Unicode、UTF-8、byte 与 byte-level BPE 的关系。
- `low` 和 `lower` 的 BPE 计数、合并、merge rank 与编码手算。
- 预分词、特殊 token、流式输入、词表大小与效率权衡。
- 为什么不同 tokenizer 的 perplexity 不能直接比较。

阅读：[第一讲详细笔记](lectures/01-overview-tokenization-detailed.md)。

## 第二讲：从张量到训练资源

**核心问题：一批 token 进入模型后，PyTorch 怎样表示和计算它们？训练需要保存什么、做多少运算，为什么会慢或爆显存？**

主要内容：

- **张量与存储：** 从 `[B,T,d]` 的轴含义开始，手算 storage、stride 和下标偏移，区分 `transpose`、`view`、`reshape` 与 `contiguous`。
- **注意力形状：** 跟踪拆分多头、Q/K/Value、`einsum`、`[B,h,T,T]` 分数矩阵与合并多头的全过程。
- **反向传播：** 用单参数回归例子解释梯度、计算图、激活保存，以及 `backward()` 和 `optimizer.step()` 的分工。
- **参数量：** 推导标准多头注意力的 `4d²`、SwiGLU 的 `3dd_ff`、RMSNorm、embedding 与输出层，完整核算约一亿参数的小模型。
- **计算量：** 从矩阵乘法逐步推导 `2mkn`、训练约为前向三倍、`6ND`，说明长上下文注意力等情况下的限制。
- **显存：** 分清参数、梯度、Adam 状态、激活、logits 和临时空间；解释 `16N byte`、混合精度、GB/GiB、allocated/reserved 与训练峰值。
- **硬件与优化：** 理解算术强度、Roofline、MFU、梯度累积、有效 token 归一化和 activation checkpointing。
- **实际测量：** CUDA 异步执行、同步、warmup、训练吞吐、峰值显存、Amdahl 定律，以及按占用来源定位 OOM。

文中使用的贯穿算例：`d=768`、`L=12`、`V=10000`、`d_ff=2048`，在标准 MHA、SwiGLU、无 bias、独立 embedding/LM head 等明确假设下，共 **100,313,856 个参数**。参数、梯度与两份 Adam 状态均为 FP32 时，这些状态合计约 **1.605 GB**，尚未包含激活和临时空间。

阅读：[第二讲详细笔记](lectures/02-pytorch-resource-accounting-detailed.md)。

## 怎样区分几种资源

| 项目 | 在回答什么问题 | 常见计量 |
| --- | --- | --- |
| 参数量 | 模型有多少可学习的数？ | 参数个数，M / B |
| 计算量 | 一次前向或训练总共做多少计算？ | FLOPs |
| 计算速度 | 每秒执行多少浮点运算？ | FLOP/s |
| 显存占用 | 计算时需要同时保存多少数据？ | byte、GB、GiB |
| 训练吞吐 | 实际每秒处理多少训练数据？ | token/s |

计算量、显存和运行时间相关，但不能互相直接替代。阅读时尤其要注意：公式算的是单层还是全模型、前向还是训练、单卡还是全部 GPU，以及使用了哪些简化假设。

## 资料来源

- [CS336 官方课程网站](https://cs336.stanford.edu/)
- [官方第一讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)
- [官方第二讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)
- [参考中文笔记：lvsir18 / CS336Spring-2026-Learning-Notes](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes)
- [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html)

各讲末尾还列出对应的官方文档、论文与延伸资料，便于对照核验。

## 笔记与示例说明

笔记根据学习对话整理，包含自编教学例子、公式推导与工程补充，不是官方讲义的逐字翻译。

- 教学文本的切分、token ID 与概率均为示例；具体模型以实际 tokenizer 为准。
- 参数量和资源公式会注明架构、精度及计数条件；近似值不等于真实训练峰值或性能保证。
- 代码用于说明概念，运行前应核对依赖、设备与输入；涉及 CUDA 的示例需要对应环境。本文不把未执行的示例描述为已验证的实验结果。

最近更新：2026-11-01，新增第二讲详解并完善阅读导航。
