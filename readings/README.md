# 论文精读与项目源码导读

这里把知识讲义中的概念连接到原论文、真实代码与本地实验。已提供 **24篇论文精读、6篇项目导读**；论文正文包含推导、可手算例子、实验条件与带答案的问题，项目正文沿实际数据与调用链阅读。先修关系使用原有知识单元ID，不另建一套冲突编号。

入口：[全部论文索引](papers/README.md) · [项目源码索引](projects/README.md) · [知识讲义](../docs/README.md) · [本地实验](../labs/README.md) · [机器可读目录](catalog.json)

## 怎样读一篇论文

1. 从标题下的先修链接回补概念，先能说清输入、输出和要解决的问题。
2. 按正文算一遍小例子，核对形状、归一化轴与损失符号。例子为独立教学构造，不是作者实验样本。
3. 打开文首指定版本的原论文，阅读方法及所标出的图表；把指标、数据、预算和评估协议一起记下。
4. 进入相应项目导读，找到公式对应的张量、配置与训练/推理边界。
5. 完成本地实验或纸面练习，最后回答三道自测题。只看过代码或算过例子，应记录为静态理解或教学验证。

论文作者的性能结果、讲义的数值例子、本地实际运行结果属于不同证据。正文不会把原论文数字写成本仓库复现；本地运行应以实验生成的报告为准。全量训练需要的数据、硬件与历史环境没有因一篇导读而自动准备齐全。

## 按领域选择

|领域|论文精读|关联项目与实验|
|---|---|---|
|优化、树模型与深层网络|[Adam](papers/adam.md)、[XGBoost](papers/xgboost.md)、[ResNet](papers/resnet.md)|[nanoGPT](projects/nanogpt.md)、[minigpt](../labs/minigpt/README.md)|
|语言建模与预训练|[Transformer](papers/transformer.md)、[BERT](papers/bert.md)、[GPT-3](papers/gpt3.md)、[T5](papers/t5.md)、[Chinchilla](papers/chinchilla.md)、[LLaMA](papers/llama.md)|[nanoGPT](projects/nanogpt.md)、[Transformers](projects/transformers.md)|
|适配与后训练|[LoRA](papers/lora.md)、[InstructGPT](papers/instructgpt.md)、[DPO](papers/dpo.md)、[DeepSeekMath](papers/deepseekmath.md)|[PEFT](projects/peft.md)、[TRL](projects/trl.md)、[posttrain](../labs/posttrain/README.md)|
|检索与智能体|[DPR](papers/dpr.md)、[RAG](papers/rag.md)、[ReAct](papers/react.md)|[LangGraph](projects/langgraph.md)、[rag](../labs/rag/README.md)、[agent](../labs/agent/README.md)|
|概率生成与扩散|[VAE](papers/vae.md)、[DDPM](papers/ddpm.md)|[生成模型讲义](../docs/02-perception-language/g-generative-models.md)|
|多模态|[CLIP](papers/clip.md)、[LLaVA](papers/llava.md)|[Transformers](projects/transformers.md)、[multimodal](../labs/multimodal/README.md)|
|强化学习与图学习|[DQN](papers/dqn.md)、[GCN](papers/gcn.md)|[强化学习讲义](../docs/03-decision-specialties/i-reinforcement-learning.md)、[图学习讲义](../docs/03-decision-specialties/j-graph-neural-networks.md)|
|注意力计算与推理服务|[FlashAttention](papers/flashattention.md)、[PagedAttention](papers/pagedattention.md)|[vLLM](projects/vllm.md)|

## 三条连续阅读路线

**从零理解大模型：** Adam → ResNet → Transformer → BERT/GPT-3/T5 对照 → Chinchilla → LLaMA → nanoGPT/Transformers → minigpt。前三篇解决优化与基本构件，中间比较训练目标，后面连接资源分配、架构与训练程序。

**把模型变成可用助手：** LoRA → InstructGPT → DPO → DeepSeekMath → PEFT/TRL → posttrain；随后 DPR → RAG → ReAct → LangGraph → rag/agent。后训练修改参数行为，检索提供外部证据，工具流程根据观察采取后续动作，三者各有职责。

**多模态与系统：** CLIP → LLaVA → multimodal；Transformer → FlashAttention → PagedAttention → vLLM。前一条连接视觉与语言表示，后一条依次处理注意力数学、计算调度与跨请求缓存。

## 版本与公开程度

论文文首固定阅读版本，图表号仅对该版本负责。[来源核验记录](assets/source-verification.md) 列出原文入口与本次核验范围。源码会演进，应先看各项目导读的版本说明；nanoGPT保留为经典精读，官方已指向nanochat作为后续项目，不能把它当持续维护的新训练栈。

本地实验统一从 [labs入口](../labs/README.md) 查看支持的阶段；CLI形态为 `python -m labs.run <project> <stage> --config labs/<project>/config.json`，具体合法stage和依赖以对应实验README为准。这里列出的是学习关联，不表示每篇论文都有同等规模的可运行复现。

## 新增材料的标准

[论文模板](templates/paper-study.md) 和 [项目模板](templates/project-study.md) 约束后续质量。新增正文须提供独立推导或算法轨迹、完整小例子、原始实验定位、失败边界与有答案自测，不能仅填摘要和链接。同步更新catalog，并在知识单元、论文、项目、实验之间建立精确关联。

[返回仓库首页](../README.md)
