# 后训练：SFT、LoRA 与 DPO

本项目把“训练目标”和“更新参数的方式”分开：SFT 使用回答部分的交叉熵；LoRA 使用同一 SFT 目标但仅更新低秩参数；DPO 根据 chosen/rejected 对优化相对冻结参考策略的偏好。

## CPU 完整运行链

先按 [公共环境](../README.md) 准备依赖，再运行前置预训练：

```powershell
python -m labs.run minigpt prepare
python -m labs.run minigpt train
python -m labs.run posttrain prepare
python -m labs.run posttrain train --method sft
python -m labs.run posttrain eval --method sft
python -m labs.run posttrain train --method lora
python -m labs.run posttrain eval --method lora
python -m labs.run posttrain train --method dpo
python -m labs.run posttrain eval --method dpo
python -m labs.run posttrain demo --method dpo
```

SFT/LoRA 默认从 `labs/outputs/minigpt/model.pt` 起步；DPO 默认从本项目的 `sft.pt` 起步。训练时 `--checkpoint` 表示初始化来源；eval/demo 时表示待评估的成品。`--output` 修改输出目录时，DPO 的默认 SFT 来源也随之改变，MiniGPT 来源仍由配置中的 `pretrain_checkpoint` 控制。

## 实现要点

数据由动物、球数量和颜色事实组成；正确颜色为 chosen，故意错误的 purple 为 rejected。数量 1–9 用于训练，10–12 用于验证。数据展示实现机制，不能用于声称模型更安全、更诚实或获得通用偏好对齐能力。

编码器保留完整回答和EOS，仅在必要时从左截断提示。`encode_preference_pair`先按较长回答预留空间，chosen/rejected使用完全相同的提示token，不能各自截断条件。提示和填充位置的labels为-100，损失只覆盖回答。DPO使用**回答序列log probability之和**，不是平均token概率，并一次性缓存初始SFT策略对训练偏好对的分数作为冻结参考。DPO策略与参考均采用eval模式禁用dropout；策略梯度仍正常计算。

LoRA 在注意力投影外增加 `B(Ax) * alpha/rank`；B 初始化为 0，基座全部冻结。小模型 checkpoint 保存完整参数和 LoRA 元数据，可直接重新加载。

## 本地 Qwen 扩展

自行准备 Qwen2.5-0.5B-Instruct 的完整本地目录，并安装公共说明中的可选依赖。示例路径仅为占位，不会触发下载。

```powershell
python -m labs.run posttrain prepare
python -m labs.run posttrain train --config labs/posttrain/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --method sft --steps 20
python -m labs.run posttrain eval --config labs/posttrain/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --method sft
python -m labs.run posttrain train --config labs/posttrain/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --method lora --steps 20
python -m labs.run posttrain train --config labs/posttrain/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --method dpo --steps 20
python -m labs.run posttrain eval --config labs/posttrain/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --method dpo
python -m labs.run posttrain demo --config labs/posttrain/local.json --model-path D:/models/Qwen2.5-0.5B-Instruct --method dpo
```

本地模型的 `sft` 与 `lora` 均以普通 LoRA 更新 q/v 投影，目标均为 SFT；它们是便于和 CPU 教学阶段对应的独立实验入口，并不代表两种不同算法。`dpo` 从 `local_sft.pt` 初始化，使用冻结的初始适配器参考分数；不会同时驻留第二个 0.5B 模型。默认 batch 1、上下文 128；输出只保存小型 adapter，基座仍从明确指定的本地目录加载，并核对目录、网络结构、rank 与 alpha 一致性。

## 评估与产物

每种方法输出独立 checkpoint 和阶段 JSON。指标包括回答 token NLL、合成偏好选择准确率和 chosen/rejected 序列分数差。不同回答长度会影响序列 log probability；必须结合数据定义解释偏好分数。

代码位于 [`posttrain.py`](../ai_learning/posttrain.py)，LoRA 与 checkpoint 共用 [`minigpt.py`](../ai_learning/minigpt.py)。公共测试覆盖回答掩码、DPO 在当前策略等于参考时的 `log(2)` 损失及梯度方向、LoRA 初始等价与基座无梯度。缺模型不计作通过。

## 回到讲义与精读

[后训练讲义](../../docs/08-walkthroughs/05-posttraining.md) · [LoRA 论文精读](../../readings/papers/lora.md) · [DPO 论文精读](../../readings/papers/dpo.md) · [PEFT 工程精读](../../readings/projects/peft.md) · [TRL 工程精读](../../readings/projects/trl.md)。
