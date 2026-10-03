# 从 token 到下一个 token：完整 Transformer 数据流

[导读](README.md) · [F02 分词](../02-perception-language/f-nlp-transformers-llms.md#f02) · [F07 Transformer](../02-perception-language/f-nlp-transformers-llms.md#f07) · [原论文精读](../../readings/papers/transformer.md)

## 1. 先确定输入、目标与形状

以下以 decoder-only 教学模型为例。它不是 2017 年原论文完整的 encoder–decoder。设文本分词后为 `[开始, 我, 爱, 学习, 结束]`。输入取前四个 token，目标取后四个：位置“我”应预测“爱”，而不是预测自身。

批量 ID 张量为 $X\in\mathbb N^{B\times T}$，词嵌入表 $E\in\mathbb R^{V\times d}$。查表后得到 $H\in\mathbb R^{B\times T\times d}$。这里 $V$ 是词表大小，$d$ 是隐藏维度，$T$ 是当前上下文长度；词 ID 的数字大小没有语义距离。

BPE 反复合并训练语料中常见的相邻符号对。字节级初始化保证可表示任意 UTF-8 字节串，但任意生成 token 序列不一定构成合法 UTF-8；解码应明确非法序列如何显示。词表只在训练语料学习，验证和测试使用冻结的分词器。

## 2. 注意力矩阵到底在乘什么

线性投影得到 $Q=HW_Q,K=HW_K,V'=HW_V$。为了避免与词表大小 $V$ 混淆，这里把 value 张量记为 $V'$。单头维度为 $d_h$ 时：

$$S=QK^T/\sqrt{d_h},\quad A=\operatorname{softmax}(S+M),\quad O=AV'.$$

$S$ 的最后两个维度是 $T\times T$，第 $i$ 行表示当前位置向各位置分配注意力；softmax 沿 key 位置进行。若 Q、K 分量近似零均值、单位方差且独立，点积方差随 $d_h$ 增长，除以平方根让数值尺度较稳定；这不是对所有真实网络分布的严格保证。

考虑两个可见位置的分数是 $[0,\log3]$，则注意力权重是 $[1/4,3/4]$。若 value 分别为 2 和 6，输出为 5。注意力先根据 Q/K 决定权重，再混合 value；不能把“相似度矩阵”直接当作最终表示。

因果 mask 把 $j>i$ 的位置设为负无穷，在 softmax 后得到零权重。padding mask 处理补齐位置；二者解决不同问题。把 softmax 后的未来权重乘零却不重新归一化，不等价于在 softmax 前遮罩。

## 3. 多头、残差与前馈层

把投影结果拆为 $[B,h,T,d_h]$，要求 $d=h d_h$。不同头学习不同投影，每头独立进行注意力，之后拼接回 $[B,T,d]$ 并乘输出矩阵。不是把同一份注意力重复计算 $h$ 次。

以 pre-norm 块为例：

$$U=H+\operatorname{Attention}(\operatorname{LN}(H)),\qquad
H_{\mathrm{next}}=U+\operatorname{FFN}(\operatorname{LN}(U)).$$

FFN 在每个位置独立作用，例如 $\operatorname{GELU}(UW_1+b_1)W_2+b_2$。它不直接混合时间位置，但对注意力汇集的特征作非线性变换。LN 沿每个 token 的隐藏维度归一化，不能误沿 batch 维处理。

原始 Transformer 使用不同的归一化放置方式。现代 decoder 常见 RMSNorm、RoPE、SwiGLU 等选择，必须区分“Transformer 的必要机制”和“某一架构的具体配方”。

## 4. 位置信息和输出层

若没有位置机制且不考虑特殊 mask，注意力对输入排列具有相应的等变性，不知道谁先谁后。可学习绝对位置 embedding 可以直接加到 token 表示；RoPE 则旋转 Q/K 的成对坐标，使点积包含相对位置信息。它并不意味着模型能无条件理解任意长上下文。

最终隐藏表示乘词表投影 $W_o\in\mathbb R^{d\times V}$，得到 logits $[B,T,V]$。训练时每个位置与移位后的目标计算交叉熵。有效 token 的损失取平均；padding 和不应监督的 prompt 位置要排除，不能让“忽略位置”也进入分母。

## 5. 训练与生成为何不同

训练时整段真实上下文已知，可以并行计算所有位置，但因果 mask 保证每个位置只看过去。生成时未来 token 未知，只能把最后位置的分布转成一个选择，再把选择追加到上下文。

温度 $\tau$ 通过 $\operatorname{softmax}(z/\tau)$ 改变分布尖锐度；top-k 截断候选数量，top-p 保留累计概率达到阈值的候选。每种策略都改变输出分布，并不保证事实正确。greedy 的局部最优也不等于整句概率最大。

KV Cache 是生成加速机制：保留历史层的 K/V，后续只计算新 token 的投影。缓存模型必须得到与无缓存完整前缀计算一致的结果，并正确处理位置、mask 和最大长度；否则加速了错误答案。

## 6. 检测题与解析

1. **$B=2,T=4,d=12,h=3$ 时，注意力分数是什么形状？** 每头维度为4；Q/K为 `[2,3,4,4]`，分数为 `[2,3,4,4]`。后两个4分别是query和key长度，数值相同不表示轴含义相同。
2. **输入与目标未经任何移位，并在同一位置直接计算交叉熵，会发生什么？** 若输出当前位置可见自身，模型可能学习复制；损失很低但不具备正确的下一个token预测能力。注意，Transformers的合法用法 `labels=input_ids` 由损失函数内部完成移位；应检查整个数据到loss的链路，确保恰好移位一次。
3. **验证因果遮罩可以怎样做？** 固定前缀，只修改后缀，关闭dropout，检查前缀位置的输出是否不变。不要只看mask矩阵长得像三角形。

## 7. 阅读与实现

[CS336 2025 Assignment 1](https://cs336.stanford.edu/spring2025/)覆盖分词、模型和优化器；[D2L 注意力](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)连接公式与代码。随后阅读 [nanoGPT](../../readings/projects/nanogpt.md)和[Transformers](../../readings/projects/transformers.md)导读，比较不同位置的标签移位。[Mini-GPT](../../labs/minigpt/README.md)交付完整训练闭环，文内的小数字不是模型性能成绩。
