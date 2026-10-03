# 各领域怎样接回共同主干

[导读](README.md) · [课程教材索引](../07-resources/t-source-index.md) · [论文精读目录](../../readings/README.md)

本讲义保留大模型以外的主要方向。选方向时先问：输入输出是什么，允许观察什么，干预或行动会不会改变后续数据，任务是否存在时间、空间或图结构。下面每节给出问题、方法与一个有答案的诊断题，具体数学回到已有单元。

## 1. 经典 AI：搜索、约束、逻辑与规划

[B07–B12](../01-foundations/b-plus-classical-ai.md#b07)把问题表示为状态、动作、转移和目标。搜索寻找动作序列，CSP寻找满足约束的赋值，逻辑推理判断结论是否被知识库支持。它们可以作为Agent的确定性子模块，而不是被“让模型想一下”完全替代。

**例子与答案**：迷宫中每步成本为1，BFS寻找最少步数路径；若不同格子代价不同，BFS不再保证最低总成本。A*的启发式和图搜索实现需要满足相应条件，才能使用最优性结论。资料：[CS188在线教材](https://inst.eecs.berkeley.edu/~cs188/textbook/)的search、CSP与games章节。

## 2. 概率模型：把结构和不确定性显式写出来

[C13 图模型](../01-foundations/c-machine-learning.md#c13)与[C14–C15 概率推断](../01-foundations/c-plus-probabilistic-black-box.md#c14)连接图模型、GP、采样与变分推断。图模型中一条边表达概率依赖，不一定表达因果关系。GP对函数建先验；MCMC使用相关样本近似后验期望，不能把迭代数等同独立样本数。

**例子与答案**：两条MCMC链都跑一万步但一直停在不同模式，不能因为样本数多就认为收敛；应检查轨迹、混合和有效样本量。资料：[CS228公开笔记](https://ermongroup.github.io/cs228-notes/)、[Probabilistic Machine Learning](https://probml.github.io/pml-book/book1.html)。这条先修支线帮助理解VAE和Bayesian uncertainty。

## 3. 视觉与语音：先理解测量过程

[E视觉](../02-perception-language/e-computer-vision.md)、[K语音](../03-decision-specialties/k-speech-audio.md)共同依赖采样、局部结构和表示学习。图像分辨率、颜色空间、摄像机几何影响“看到什么”；音频采样率、时间窗和频谱变换影响“听到什么”。

**例子与答案**：提高音频采样率不会凭空恢复录音时已经丢失的高频信息；调整图像尺寸也可能改变小物体可见性。评估应分别报告感知输入条件。视觉读[CS231n](https://cs231n.stanford.edu/)和[Szeliski教材](https://szeliski.org/Book/)；语音读[李宏毅DLHLP](https://speech.ee.ntu.edu.tw/~hylee/dlhlp/2020-spring.php)及[SLP3](https://web.stanford.edu/~jurafsky/slp3/)语音章节。

## 4. 强化学习：行动改变后续数据

[I强化学习](../03-decision-specialties/i-reinforcement-learning.md)从MDP、回报和Bellman方程出发。监督学习通常直接拥有标签，RL需要通过行动获得回报和新状态，探索行为还会改变收集到的数据分布。

**例子与答案**：一次轨迹成功不能说明策略稳健；不同随机种子、初始状态和环境扰动可能导致相反结果。评估时关闭训练探索，并分清环境终止和时间截断。学习顺序为老虎机 → MDP → DP/MC/TD → 函数逼近 → 策略梯度；[DQN精读](../../readings/papers/dqn.md)不替代后训练需要的PPO先修。资料：[CS285](https://rail.eecs.berkeley.edu/deeprlcourse/)、[Sutton与Barto教材入口](http://incompleteideas.net/book/the-book-2nd.html)。

## 5. 图学习：邻居不是普通特征列

[J图学习](../03-decision-specialties/j-graph-neural-networks.md)把关系结构纳入表示。消息传递先从邻居收集信息，再用与节点排列无关的聚合更新自身。堆叠层数增加感受范围，但也可能导致过平滑或信息瓶颈。

**例子与答案**：把节点ID重新编号后，图级预测应不变，节点级预测应随编号对应重排。若测试边提前进入训练图，可能发生结构泄漏。资料：[CS224W](https://web.stanford.edu/class/cs224w/)、[Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/)、[GCN精读](../../readings/papers/gcn.md)。

## 6. 时间序列、因果与推荐：评测协议决定问题

[L时间序列](../03-decision-specialties/l-time-series.md)要区分预测时可用信息和事后统计。滚动验证应在每个预测时点只用过去；使用全序列均值标准化可能泄漏未来。

[M因果推断](../03-decision-specialties/m-causal-inference.md)区分观察条件 $P(Y\mid X)$ 与干预后的 $P(Y\mid do(X))$。天气同时影响冰淇淋销量和溺水风险时，销量可预测风险，并不意味着干预销量会改变溺水概率。混杂、选择偏差和可识别性假设必须先说明，再谈估计器。

[N推荐检索](../03-decision-specialties/n-recommendation-search-retrieval.md)的数据还受到曝光策略影响。用户没点击可能因为不喜欢，也可能根本没看到；随机负采样的离线高分不能直接推出线上收益。

**检测与答案**：三个方向都不应只随机拆行。时间序列考虑时间边界，因果考虑分配机制与混杂，推荐考虑用户/物品和曝光边界。进一步阅读：[Forecasting: Principles and Practice](https://otexts.com/fpp3/)、[Causal Inference: What If](https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf)、[Stanford IR教材](https://nlp.stanford.edu/IR-book/)。

## 7. 机器人：策略输出之后仍有物理系统

[Q机器人](../04-systems-agents-robotics/q-embodied-ai-robotics.md)连接坐标变换、感知、状态估计、运动学、规划与控制。语言模型输出“抓取杯子”不是电机指令；学习策略输出候选动作也不意味着满足碰撞、速度、接触和可达性约束。

**例子与答案**：两臂末端目标都可达，联合轨迹仍可能相撞；单臂IK成功不足以证明双臂任务安全。先明确观测、动作接口、控制周期和异常停机，再比较模仿学习或RL。资料：[MIT Underactuated Robotics](https://underactuated.csail.mit.edu/)与[Robotic Manipulation](https://manipulation.csail.mit.edu/)。本轮不运行机器人仿真或把静态讲解写成碰撞验证结果。

## 8. 可信 AI 贯穿全部方向

公平性、隐私、鲁棒性和安全性通常对应不同目标，不能用一个总体准确率替代。不同公平性指标可能冲突，需要说明目标人群、风险和权衡。学习来源：[Fairness and Machine Learning](https://fairmlbook.org/)。在每个实验里保留数据来源、分组误差、失败条件和用途限制，比孤立列出几个原则更可执行。
