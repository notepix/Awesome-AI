# MiniGPT：从字节到自回归生成

本项目完整实现字节 BPE、训练窗口、Decoder Transformer、训练/续训、保存/加载、生成和验证集困惑度。先读 [环境与统一接口](../README.md)。

## 最短运行链

```powershell
python -m labs.run minigpt prepare
python -m labs.run minigpt train
python -m labs.run minigpt eval
python -m labs.run minigpt generate --prompt "The cat"
# 在原 checkpoint 上继续 20 个优化步骤
python -m labs.run minigpt train --resume --steps 20
```

默认 CPU、2 层、4 头、64 维、128 token 上下文、80 个优化步骤；适合查看流程，不能训练通用语言模型。`--steps` 表示本次新增步骤数；`--resume` 读取默认 `model.pt` 或 `--checkpoint` 指定的文件，并恢复参数、优化器、步骤和 CPU 随机数状态。

## 数据与算法

`prepare` 生成动物、地点与球颜色的英文短故事；先按文档划分训练集/验证集，只在训练文本上拟合 BPE。验证集中保留 library/yellow 内容，用于暴露小语料泛化限制。

BPE 从 256 个字节加 PAD/BOS/EOS 三个特殊符号开始，迭代合并训练集最频繁相邻 token。新 UTF-8 文本无需未知词符号；完整 encode→decode 可逆。任意生成字节序列未必形成合法 UTF-8，因此展示生成内容时会以替代字符呈现残缺字节。窗口包含 `block_size+1` 个 token，输入与目标错开一位。

实现采用学习式位置嵌入、Pre-LayerNorm、显式多头因果注意力与 GELU 前馈网络。交叉熵覆盖下一 token；验证集 token NLL 以 token 数加权，困惑度为 `exp(NLL)`。生成使用贪心解码，屏蔽 PAD/BOS，遇 EOS 停止，并只保留最近一个上下文窗口。

困惑度不截断NLL：正常时`perplexity_status=finite`；指数超出浮点范围时`perplexity=null`并标为`overflow`；NLL自身非有限时标为`non_finite_nll`。这些异常不能作为有效困惑度比较。

## 输出与后续接口

- `data.json` / `tokenizer.json`：自建数据和 BPE 合并顺序。
- `model.pt`：格式版本、网络配置、BPE、参数、优化器、训练步数和随机状态。
- `train_default_offline-core.json` / `eval_default_offline-core.json`：真实训练损失与验证集指标。

[后训练项目](../posttrain/README.md) 直接复用这份 tokenizer/config/checkpoint。改变 MiniGPT 模型配置后应重新训练；续训以 checkpoint 内的网络配置为准。当前精确续训边界为默认 CPU 配置；CUDA 并行算子的逐位复现不作承诺。

## 阅读与验证

实现位于 [`minigpt.py`](../ai_learning/minigpt.py)：`ByteBPE → windows → Attention/Block/TinyGPT → run`。读完可回答：为什么拟合 tokenizer 前先划分文档？为何因果掩码必须遮住未来？困惑度为什么必须注明分词器？

```powershell
python -m unittest discover -s labs/tests -v
```

测试覆盖未知 Unicode 往返、未来 token 不影响此前 logits、窗口位移和 checkpoint 加载；运行链负责证明训练、续训和生成实际可执行。代码不自动下载语料或模型。

## 回到讲义与精读

[Transformer 讲义](../../docs/08-walkthroughs/03-transformer.md) · [预训练与推理讲义](../../docs/08-walkthroughs/04-pretraining-inference.md) · [Transformer 论文精读](../../readings/papers/transformer.md) · [GPT-3 论文精读](../../readings/papers/gpt3.md) · [nanoGPT 工程精读](../../readings/projects/nanogpt.md)。
