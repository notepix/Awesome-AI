# 04 · 系统、Agent 与机器人

[← 返回学习地图](../README.md)

本层关注模型进入真实系统后的能力：LLM 后训练、RAG 与工具调用，生命周期与安全评测，以及感知—规划—控制闭环中的具身智能。这里的重点是系统边界和可验证性，而不只是模型演示。

## 推荐顺序

1. LLM/Agent 方向：先学 [O. LLM 后训练、RAG 与 Agent](o-llm-posttraining-rag-agents.md)，再学 [P. MLOps、安全与评测](p-mlops-safety-evaluation.md)。
2. 机器人方向：先完成 E/H/I 与经典规划，再学 [Q. 具身智能与机器人](q-embodied-ai-robotics.md)。
3. 在集成或部署前补齐 P，把数据、模型、工具和执行器纳入统一评测与监控。

## 章节

- [O. LLM 后训练、RAG 与 Agent](o-llm-posttraining-rag-agents.md)
- [P. MLOps、安全与评测](p-mlops-safety-evaluation.md)
- [Q. 具身智能与机器人](q-embodied-ai-robotics.md)

## 完成标准

能画出端到端数据流、信任边界和失败恢复路径；能让检索器、模型、工具或策略模块独立替换与回归测试；能区分高层语义规划与确定性的约束、控制和安全检查，并用日志与指标复现失败。
