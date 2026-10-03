# 后训练：SFT、LoRA、奖励模型与偏好优化

[导读](README.md) · [O 后训练](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md) · [I08 策略梯度](../03-decision-specialties/i-reinforcement-learning.md#i08)

## 1. 先分清目标、参数化与优化算法

SFT规定从示范回答学习的目标；LoRA规定允许更新哪些参数及如何表示增量；DPO规定从偏好对学习的目标。它们不是互斥的三个模型类型。可以用LoRA参数化训练SFT，也可以用LoRA训练DPO。

一条对话有system、user和assistant角色，chat template把结构转成具体token序列。模板错误会让训练与推理输入分布不一致。对普通instruction SFT，损失只监督assistant回答：

$$L_{SFT}=-\frac{\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t})}{\sum_t m_t},\quad m_t\in\{0,1\}.$$

例如序列包含3个prompt token和2个回答token，两个回答的预测概率是0.8和0.5，平均损失约为 $-(\log0.8+\log0.5)/2=0.458$。prompt仍参与前向计算并影响回答，只是不直接作为被监督的目标位置。若mask全为0，该样本没有学习信号，应拒绝或明确跳过，不能除零。

## 2. LoRA 的矩阵、初始化与保存

基础线性层 $W\in\mathbb R^{d_o\times d_i}$ 冻结，增量为

$$W'=W+\frac\alpha r BA,\quad A\in\mathbb R^{r\times d_i},\ B\in\mathbb R^{d_o\times r}.$$

取 $d_i=d_o=4,r=1$，全矩阵16个参数，低秩增量只需8个参数。令 $A=[1,0,0,0]$，$B=[1,2,0,0]^T$，$\alpha/r=1$，输入 $x=[3,5,7,9]^T$，增量输出 $BAx=[3,6,0,0]^T$。低秩不是把输入截成前r维，而是学习两个投影。

常用一侧随机、一侧置零，让初始模型等于基座；两侧都置零会使乘积的两个梯度都为零。只保存adapter时，恢复还依赖原基座、tokenizer和配置。合并后模型与未合并模型应在容差内一致，量化配置下不能无条件假定完全等价。

## 3. 从偏好对到奖励模型

标注者给同一prompt的两个回答排序。Bradley–Terry模型用奖励差表示偏好概率：

$$P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l)).$$

若奖励差是 $\log3$，则预测胜出概率是0.75。奖励只由相对差决定，给两个回答奖励同时加常数不会改变偏好概率。标注还可能混合事实性、语气、长度和格式，奖励模型不等于客观真理函数。

KL正则化的策略目标在高奖励与不过度偏离参考策略间折中。参考策略通常是固定的已有模型。KL过强可能学不动，过弱可能让策略利用奖励模型的盲点，因此要同时观察独立任务指标和奖励，而不是只优化奖励曲线。

对固定prompt，最大化 $J(\pi)=\sum_y\pi(y)r(y)-\beta\sum_y\pi(y)\log[\pi(y)/\pi_{ref}(y)]$，并要求概率和为1。对每个 $\pi(y)$ 求导，再用拉格朗日乘子吸收归一化常数，得到 $r(y)-\beta(\log[\pi(y)/\pi_{ref}(y)]+1)+\lambda=0$。因此最优分布为 $\pi^*(y)=\pi_{ref}(y)\exp[r(y)/\beta]/Z$。这里假设参考概率覆盖所考虑的回答、配分函数有限；这是一条分布优化关系，不保证任意神经网络训练都达到该最优值。

## 4. DPO 为什么需要参考模型

从KL正则化最优策略形式，可得到奖励与 $\log(\pi_\theta/\pi_{ref})$ 的关系；同prompt下分区常数在回答差中抵消，代入偏好模型得到：

$$L_{DPO}=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right]\right).$$

令policy对胜者/败者的序列logprob为-2/-3，reference为-2.5/-2.8，括号内差为0.7。$\beta=0.1$时，损失约为 $-\log\sigma(0.07)=0.659$。这是手算目标，不是模型胜率。序列logprob通常求回答token的和，不能未经说明替换成平均值；长度因素会因此影响目标。

reference必须固定且关闭dropout；可以预计算固定数据的reference logprob，避免两份大模型同时驻留。precompute只适用于数据、模板、分词和reference均不再变化的情况。

## 5. PPO、GRPO 的先修桥接

把回答看成一条轨迹：状态是prompt和已生成的前缀，动作是下一个token，终止于EOS或长度上限。轨迹概率为 $p_\theta(\tau)=\prod_t\pi_\theta(a_t\mid s_t)$，环境转移部分不依赖策略参数。对 $J=\mathbb E[R(\tau)]$ 使用对数导数恒等式得到

$$\nabla_\theta J=\mathbb E\left[R(\tau)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].$$

这是REINFORCE的起点。减去与当前动作无关的baseline不改变期望梯度，因为 $\sum_a\pi(a\mid s)\nabla\log\pi(a\mid s)=\nabla1=0$。优势 $A=Q-V$ 衡量动作比该状态通常表现好多少；训练时优势一般视为固定目标，不通过它反传到actor。

Actor–Critic学习价值函数。TD残差为 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，终止状态的下一价值置零。广义优势估计为

$$\hat A_t^{GAE}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.$$

$\gamma$控制远期奖励，$\lambda$控制多步估计；增大 $\lambda$ 通常减少对单步价值估计的依赖，但方差可能增大。教学例子取两步奖励 $(0,1)$、价值 $(0.4,0.5,0)$，$\gamma=1,\lambda=0.5$，则残差为 $(0.1,0.5)$，优势为 $(0.35,0.5)$；不能把优势直接当成奖励本身。

PPO先用旧策略采样，再优化裁剪替代目标。记 $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{old}(a_t\mid s_t)$：

$$L^{clip}=\mathbb E_t\left[\min(\rho_t\hat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)\right].$$

例如优势为2，$\epsilon=0.2$，概率比为1.4，未裁剪项是2.8，裁剪项是2.4，目标取2.4；继续把这个正优势动作的概率抬高，不再从该裁剪项获得增益。负优势时要保留乘号与min的完整形式，不能直接照搬正优势直觉。clip不是严格的全局KL约束；LLM中的参考策略 $\pi_{ref}$ 与每轮采样策略 $\pi_{old}$ 也有不同职责：前者限制偏离初始模型，后者定义更新时的重要性比。

GRPO对同一个prompt采样一组回答，用组内奖励构造 $\hat A_i=(r_i-\bar r)/(s_r+\varepsilon)$，不再为baseline单独训练critic。取奖励 $(0,1,2)$ 并用总体标准差，优势约为 $(-1.225,0,1.225)$；这是教学归一化，实际框架可能采用不同标准差约定。序列级奖励会把同一组优势分给该回答的有效token，再使用概率比裁剪及KL项。奖励组若全相同，归一化要处理零方差，且几乎没有区分回答的信号。组内标准化、token/序列归一化、KL估计和采样策略都会改变训练行为；论文公式与框架变体应逐项对照。

[InstructGPT](../../readings/papers/instructgpt.md)、[DPO](../../readings/papers/dpo.md)与[DeepSeekMath](../../readings/papers/deepseekmath.md)分别承担不同历史和机制角色，不能把它们统称为“用了人类反馈所以相同”。

## 6. 检测题与解析

1. **SFT只计算回答loss，prompt是否不影响训练？** 影响；回答位置会通过注意力依赖prompt的表示和相关参数。
2. **DPO训练时同步更新reference，有什么问题？** 改变目标中的固定参照，破坏上述推导和学习信号；不能称为同一个DPO设定。
3. **合成偏好胜率上升说明模型更符合人类价值吗？** 不能。它仅证明模型在特定标签规则和划分下改变了偏好，需要独立数据和明确的评价维度。

## 7. 来源与完整实现

[RLHF Book](https://rlhfbook.com/)的instruction tuning、reward modeling、policy gradients与direct preference optimization章节；[CS336](https://cs336.stanford.edu/spring2025/)后训练讲次。库级行为参见[PEFT](../../readings/projects/peft.md)与[TRL](../../readings/projects/trl.md)；完整离线链路参见[后训练实验室](../../labs/posttrain/README.md)。教材草稿与库接口按所注明版本理解。
