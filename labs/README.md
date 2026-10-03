# 五个可运行的 AI 教学项目

这些项目把讲义中的算法变为可读、可运行的实现。数据均由仓库自行编写或生成，不抓取教材，不下载训练语料。默认 CPU 核心在**依赖已经安装**后可以断网运行。

| 项目 | 核心实现 | 运行顺序 |
|---|---|---|
| [MiniGPT](minigpt/README.md) | 可逆字节 BPE、因果 Decoder、预训练、续训、困惑度 | prepare → train → eval → generate |
| [后训练](posttrain/README.md) | 回答掩码 SFT、LoRA、冻结参考 DPO | 先 MiniGPT；prepare → train → eval → demo |
| [RAG](rag/README.md) | BM25、TF-IDF/SVD、混合检索、规则重排、引用、拒答 | prepare → index → eval → demo |
| [Agent](agent/README.md) | 严格工具接口、预算、超时、恢复、轨迹 | prepare → index → eval → demo |
| [多模态](multimodal/README.md) | 自生图文数据、双塔、多正例对比学习、双向检索 | prepare → train → eval → demo |

## 环境与入口

建议 Python 3.12；需要 numpy、PyTorch、scikit-learn、Pillow。根目录的 `pyproject.toml` 给出依赖范围。首次安装依赖需要联网或已有本地安装包；这不属于离线执行阶段。

```powershell
# 在仓库根目录执行；如果 .venv 已经准备好，则跳过创建步骤
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m unittest discover -s labs/tests -v
```

本目录内的示例以 `python` 代表上述环境中的解释器。可以激活环境，也可以把每行的 `python` 换成 `.venv/Scripts/python.exe`。从仓库根目录运行。

统一接口：

```text
python -m labs.run PROJECT STAGE [--config PATH] [--profile offline-core|local-model]
    [--output PATH] [--checkpoint PATH] [--steps N] [--prompt TEXT]
    [--method sft|lora|dpo] [--resume]
    [--model-path LOCAL_DIRECTORY] [--embedding-path LOCAL_DIRECTORY]
```

配置默认读取 `labs/项目名/config.json`；命令行只覆盖显式提供的选项。`local.json` 使用 `device=auto`；CPU 核心使用 `device=cpu`。每个项目只接受其 README 指定的阶段组合。

## 结果与验证边界

输出位于 `labs/outputs/项目名/`，已在本目录的 `.gitignore` 中排除。JSON 记录配置、种子、阶段、指标和验证类型；模型与索引独立保存。记录中的 `status=completed` 表示该阶段正常结束，**不是质量达标**。重复运行同一项目、方法和配置类型会覆盖其阶段记录；需要保留多组实验时使用不同的 `--output`。

- `offline_algorithm`：真实离线算法或小模型计算。
- `controller_fixture`：脚本驱动的 Agent 控制器测试，不代表 LLM 的规划能力。
- `local_model`：实际加载用户提供的本地预训练模型并执行。缺权重或缺依赖时退出码为 2，并打印 `NOT VALIDATED`；不会把跳过记为通过。

运行结果只描述自建小数据上的行为。小模型生成文本、合成偏好胜率和几何形状检索均不能证明通用能力。教材中的原理、这里的参考实现与正式基准测试是不同层次的证据。

## 可选本地模型

额外安装 `pip install -e ".[local-model]"`。用户自行准备完整模型目录，不要把模型权重提交到仓库。代码使用 `local_files_only=True`；自动模型加载明确关闭远程代码执行，固定 CLIP 类也不执行仓库自定义代码。

- Qwen2.5-0.5B-Instruct：后训练、RAG 生成、Agent。
- paraphrase-multilingual-MiniLM-L12-v2：RAG 神经稠密检索。
- openai/clip-vit-base-patch32：预训练图文检索，示例使用英文 caption。

8GB RTX 4060 / 16GB RAM 是扩展的目标预算，不能视为已测的峰值保证。使用 batch 1、短上下文、普通 LoRA；不要求 bitsandbytes、QLoRA 或 FlashAttention。只安装匹配本机的官方 PyTorch CUDA 版本；代码不修改驱动。出现显存不足时缩短上下文或改用 CPU，并保留实际失败记录。

## 实现索引

[`ai_learning/common.py`](ai_learning/common.py) 提供配置之外的薄公共层、设备选择、本地加载和结果记录；其余五个同名模块承载算法。`tests/test_algorithms.py` 覆盖因果性、回答掩码、DPO 梯度方向、LoRA 冻结、检索拒答、工具边界和多正例检索语义。无哈希值校验。

实际执行的环境、30 个阶段结果及质量限制见 [验证记录](VALIDATION.md)。复现完整 CPU 链使用 `python -m labs.validate`；该命令会重新生成默认输出目录中的数据、模型和阶段记录。

继续阅读：[系统讲义](../docs/08-walkthroughs/README.md) · [论文精读示例：Transformer](../readings/papers/transformer.md) · [论文精读示例：CLIP](../readings/papers/clip.md) · [工程精读索引](../readings/projects/README.md)。每个项目末尾还提供其对应章节与论文的直接链接。
