# 教学项目实际验证记录

完整流程记录时间：2026-10-03T00:39:22+0800；边界修复复测：2026-10-03T01:00:50+0800。本记录来自本机实际执行；不是静态估算或预期成绩。

## 环境与范围

- Python：3.12.14；平台：Windows-11-10.0.26200-SP0。
- 解释器：`D:/AI-learning/.venv/Scripts/python.exe`，全部项目使用 CPU、seed 42 与各自默认配置。
- 没有下载语料或模型权重，没有执行哈希值校验；依赖安装在执行前完成。

| 依赖 | 实际版本 |
|---|---|
| numpy | 2.5.3 |
| torch | 2.14.1+cpu |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| scipy | 1.18.1 |
| pytest | 9.1.1 |
| transformers | not installed |

## 执行结果

`python -m unittest discover -s labs/tests -v`：**26 项测试全部通过**（边界修复后的运行为1.612秒）。覆盖因果性、可逆BPE、窗口计数、短验证文档、CPU保存续训一致性、回答掩码、DPO梯度与成对提示截断、LoRA冻结与加载、检索拒答、工具schema/执行边界/预算/超时与精确结果判断，以及多模态划分与相关性定义。

`python -m labs.validate`：**30 个实际 CLI 阶段全部退出码 0**，逐阶段时间合计 76.3 秒。此时间只描述本次机器与依赖组合，不是通用性能保证。包含：

- MiniGPT prepare、80 步 train、eval、generate、20 步 resume、再次 eval。
- 后训练 prepare，SFT / LoRA / DPO 各自 train、eval、demo。
- RAG prepare、index、eval、可回答及不可回答 demo。
- Agent prepare、index、eval、demo。
- 多模态 prepare、train、index、eval、demo。

完整逐阶段配置、指标、输出与依赖版本保存在本地生成文件 `labs/outputs/validation.json`；重新执行验证命令可再生成。该目录不提交到仓库。

独立复核修正四项边界后，另行重跑 **9个受影响CLI阶段，全部退出码0**：MiniGPT eval；SFT、LoRA、DPO各自train与eval；Agent eval与demo。结果保存在 `labs/outputs/validation-boundary-fixes.json`，下表相关指标与复测一致。RAG与多模态实现未受这批修改影响，没有重复训练。

## 观察到的指标

| 项目 | 实际结果 | 解释边界 |
|---|---|---|
| MiniGPT，80 步 | 验证 NLL 5.8108；困惑度 333.87；109 个验证目标 token | 训练损失下降不能代表泛化改善 |
| MiniGPT，续训至 100 步 | 验证 NLL 6.0373；困惑度 418.75 | 验证困惑度反而上升，显示小语料过拟合/分布差异 |
| SFT | 回答 NLL 1.9156；合成偏好准确率 1.00 | 极简单的自建事实与故意错误回答，不是通用能力指标 |
| LoRA | 回答 NLL 3.9329；合成偏好准确率 1.00 | 与全参数更新的差异需结合参数量及数据难度解释 |
| DPO | 回答 NLL 3.1877；合成偏好准确率 1.00 | 偏好目标不保证回答 NLL 同时改善，不代表普适对齐 |
| RAG | Recall@3、答案关键词、正确引用、回答/拒答分类均为 1.0 | 仅 9 个可回答问题与 2 个不可回答问题；离线抽取式输出 |
| Agent | 7 / 7 控制器场景符合预期 | `controller_fixture`，不是 LLM 工具使用成功率 |
| 多模态图搜文 | R@1=0.4722，R@3=1.0000 | 36 张留出组合测试图，12 个英文 caption 候选 |
| 多模态文搜图 | R@1=1.0000，R@3=1.0000 | 仅 3 个测试 caption 类；相关性是相同语义 caption |

多模态随机初始化时图搜文 R@1=0，文搜图 R@1=1/3；训练后有改善，但这些结果不能外推到自然图片或任意新词。所有训练与评估数据均为仓库自建/自生成。

## 实际修复与未验证项

第一次完整运行发现：验证文本经 BPE 后总长度小于固定训练窗口，导致评估失败。现已把验证改成逐文档、可变长度窗口，覆盖短文档和尾段，并确保每个 target 只计一次；新增相应测试后完整链通过。首次失败记录保留于本地 `labs/outputs/validation-first-attempt.json`。

独立代码复核还发现并修复以下四项问题，各有回归测试：

- **DPO条件一致性**：先按一对回答的最大长度统一截断prompt，再编码chosen/rejected，防止两支使用不同条件。CPU tokenizer和本地chat模板接口均有测试；后者采用测试替身，不是加载真实Qwen。
- **Agent超时复用**：传给提供器的预算与单次等待预算一致；本地评估超时后停止进入同一模型，剩余案例标为未运行，不计成功。测试用受控阻塞提供器验证，未假装强制终止GPU内核。
- **答案假阳性**：按数值和时间解析结果，`142`不能通过期望`42`，`108:00`不能通过期望`08:00`。这仍是明确任务格式的检查，不是通用语义裁判。
- **困惑度溢出**：取消NLL=80的静默上限；真实指数超出浮点范围时输出`perplexity=null`和`perplexity_status=overflow`，非有限NLL另行标注，保持严格JSON可读。

缺模型负向验证实际得到退出码 **2**，提示 `NOT VALIDATED` 与准备本地模型目录，符合预期。这只验证失败路径，**不验证本地模型运行能力**。

未执行：Qwen / MiniLM / CLIP 的真实权重加载和推理，Qwen SFT/LoRA/DPO，本机 CUDA 与 8GB 显存峰值。相关接口已实现，仍需用户自行准备权重与匹配的 GPU PyTorch 环境后运行；当前没有将它们计为通过。
