# CS336 Learning Notes

斯坦福 CS336（Spring 2026）中文学习笔记，用具体例子、公式拆解和手算过程理解语言模型。

## 阅读入口

| 讲次 | 主题 | 笔记 |
| --- | --- | --- |
| 第 01 讲 | 课程总览与 Tokenization | [从文本到语言模型：第一讲详细讲解](lectures/01-overview-tokenization-detailed.md) |

第一讲从文字进入模型的完整流程出发，依次讲解 token、embedding、next-token prediction、训练损失、teacher forcing、Unicode / UTF-8、byte-level BPE、编码与解码、预分词、特殊 token、流式输入和效率权衡。

## 资料来源

- [CS336 官方课程网站](https://cs336.stanford.edu/)
- [官方第一讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)
- [参考中文笔记：lvsir18 / CS336Spring-2026-Learning-Notes](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes)

笔记根据学习对话整理，包含自编教学例子、推导与工程补充；具体模型的切分和 token ID 以其实际 tokenizer 为准。
