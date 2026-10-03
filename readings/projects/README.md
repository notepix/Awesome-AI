# 核心源码导读

[返回精读入口](../README.md) · [论文精读](../papers/README.md) · [学习地图](../../docs/README.md)

这里把论文机制连接到实际官方源码：不只介绍库的用法，而是追踪一次输入经过哪些对象、张量或状态怎样改变、哪些地方会失败。六篇导读的源码阅读日为 **2026-10-03**，均属于静态源码分析；没有在本地部署上游框架、训练大模型或报告上游性能成绩。

## 推荐顺序与关系类型

| 顺序 | 导读 | 阅读范围和固定版本 | 与论文的关系 |
|---|---|---|---|
| 1 | [nanoGPT](nanogpt.md) | 本地保存的 2026-10-03 日历源码快照；历史弃用项目 | Transformer／GPT 风格的精简机制实现，不是 GPT-3 规模复现 |
| 2 | [Transformers](transformers.md) | v4.57.1；以 Qwen2 decoder、loss、cache 为入口 | 架构实现和框架接口对照；Qwen2 不等同原始 LLaMA |
| 3 | [PEFT](peft.md) | v0.17.1；LoRA 注入、forward、保存和合并 | LoRA 的算法实现与工程适配 |
| 4 | [TRL](trl.md) | v0.24.0；SFT、DPO、GRPO 数据流及损失 | 训练目标实现对照；多种配置不等于原论文同一配方 |
| 5A | [vLLM](vllm.md) | v0.11.0；V1 engine、scheduler、KV manager | PagedAttention 系统演化及推理控制流，区别于 FlashAttention 计算内核 |
| 5B | [LangGraph](langgraph.md) | 0.6.7；StateGraph、reducer、Pregel loop、checkpoint | ReAct／RAG 的流程组织对照，不是原论文作者实现 |

先完成前两篇，再按训练方向进入 PEFT→TRL，或按服务方向进入 vLLM；应用方向可在理解 [O05–O08](../../docs/04-systems-agents-robotics/o-llm-posttraining-rag-agents.md) 后学习 LangGraph。无需按表格顺序读完全部论文后才开始源码。

## 每篇怎么读

1. 先读版本边界与先修，只选择正文定义的具体路径；不要把历史版本的函数签名直接拼接进新版环境。
2. 根据源码证据表打开真实文件，沿入口追到输出；图示中的执行器或外部模型若不在已读范围内，继续保留为边界。
3. 手工完成张量形状或状态轨迹，检查标签移位、mask、缓存位置、归约分母和状态合并规则。
4. 回答检测题，再进入对应实践。静态例子是预期行为解释，实践输出才是实际运行证据，二者分别记录。

完成一篇的标准不是记住函数名，而是能解释“输入是什么、谁负责变化、何时变化、为何这样设计、失败后去哪里定位”。所有版本均为阅读对象，不代表 2026-10-03 的最新发布。源码位置链接绑定发布标签；nanoGPT 的可回读内容绑定仓库内的[日历快照](source-snapshots/nanogpt/README.md)。

## 实践连接

- [MiniGPT](../../labs/minigpt/README.md)：用小模型串起移位、注意力、训练与生成，对照 nanoGPT／Transformers。
- [后训练](../../labs/posttrain/README.md)：分开检查 LoRA 参数更新与 SFT／DPO／GRPO 目标，避免把参数方法和目标方法混为一谈。
- [RAG](../../labs/rag/README.md) 与 [Agent](../../labs/agent/README.md)：验证证据、状态、停止条件和失败路径，对照 LangGraph 的控制机制。
- [多模态](../../labs/multimodal/README.md)：将视觉条件接入语言模型时，复用模型侧 shape／mask 阅读方法；六篇源码导读没有宣称覆盖某一 VLM 全部实现。
