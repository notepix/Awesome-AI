# I. 强化学习与 Deep RL


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](../02-perception-language/h-multimodal-vlm.md) · [下一章 →](j-graph-neural-networks.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/05-posttraining.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="i01"></a>

### I01 多臂老虎机与探索—利用【稳定】
- **先修**：期望、均值、独立同分布采样。
- **定义与解析**：老虎机没有状态转移；每轮选臂并观察该臂奖励。它隔离了 RL 最核心的探索成本。
- **公式/机制**：遗憾 $R_T=T\mu^*-\sum_{t=1}^T\mu_{a_t}$；ε-greedy 以 ε 随机探索，否则选当前均值最大臂。
- **资料**：Sutton & Barto 2e [§2.1–2.4](http://incompleteideas.net/book/RLbook2020.pdf)；逐步实现见 Gymnasium [Bandit 教程入口](https://gymnasium.farama.org/tutorials/training_agents/)。
- **最小代码（可执行，NumPy/CPU）**：
```python
import numpy as np
def run(seed, T=2000):
    g=np.random.default_rng(seed); mu=np.array([.1,.2,.35]); n=np.zeros(3); q=np.zeros(3)
    total=0.
    for t in range(T):
        a=g.integers(3) if g.random()<.1 else int(q.argmax())
        r=float(g.random()<mu[a]); n[a]+=1; q[a]+=(r-q[a])/n[a]; total+=r
    return T*mu.max()-total
print(np.mean([run(s) for s in range(5)]))
```
- **检测/实验**：把 ε 改为 0、0.01、0.1、0.5，先预测五种子平均遗憾排序；验收要求报告均值与标准差。
- **常见坑**：只跑一个种子；用最终训练奖励冒充独立评估；误把非平稳奖励仍当作样本均值问题。

<a id="i02"></a>

### I02 MDP、轨迹与环境接口【稳定】
- **先修**：条件概率、马尔可夫性、有限状态机。
- **定义与解析**：MDP 用 $\left(\mathcal S,\mathcal A,P,R,\gamma\right)$ 描述“当前状态和动作足以决定下一步分布”的序贯决策；轨迹是交互样本而非固定标签集。
- **公式/机制**：$P(s',r\mid s,a)$，轨迹 $\tau=(s_0,a_0,r_1,\ldots)$；终止 `terminated` 与时间截断 `truncated` 语义不同。
- **资料**：Sutton & Barto [§3.1–3.4](http://incompleteideas.net/book/RLbook2020.pdf)；Gymnasium [Basic Usage: reset/step](https://gymnasium.farama.org/main/introduction/basic_usage/)。
- **最小代码（可执行模拟）**：
```python
import numpy as np
P=np.array([[[.8,.2],[.1,.9]], [[.6,.4],[.3,.7]]]) # [s,a,s']
r=np.array([[0.,1.],[0.,2.]])
g=np.random.default_rng(0); s=0; traj=[]
for _ in range(5):
    a=int(g.integers(2)); ns=int(g.choice(2,p=P[s,a]))
    traj.append((s,a,r[s,a],ns)); s=ns
print(traj)
assert all(abs(P[s,a].sum()-1)<1e-9 for s in range(2) for a in range(2))
```
- **检测/实验**：若观察缺少速度，位置控制是否仍是 MDP？给出补历史或 belief state 的办法。
- **常见坑**：把观测当真实状态；忽略时间上限 bootstrap；训练与评估环境 wrapper 不一致。

<a id="i03"></a>

### I03 回报、策略与价值函数【稳定】
- **先修**：I02、几何级数、条件期望。
- **定义与解析**：策略给动作分布；状态价值和动作价值把未来随机回报压缩为期望，不能解释为必然结果。
- **公式/机制**：$G_t=\sum_{k\ge0}\gamma^kR_{t+k+1}$，$V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]$，$Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]$。
- **资料**：Sutton & Barto [§3.5–3.6](http://incompleteideas.net/book/RLbook2020.pdf)，重点看式 (3.8)–(3.14)。
- **最小代码（可执行）**：
```python
import numpy as np
r=np.array([1.,0.,2.,3.]); gamma=.9
G=np.empty_like(r); carry=0.
for t in range(len(r)-1,-1,-1):
    carry=r[t]+gamma*carry; G[t]=carry
print(G)
assert np.allclose(G,[1+2*.9**2+3*.9**3, 2*.9+3*.9**2, 4.7, 3.])
```
- **检测/实验**：手算同一奖励序列在 $\gamma=0,0.5,1$ 时的 $G_0$，解释 $\gamma$ 同时改变偏好与数值尺度。
- **常见坑**：奖励与回报混用；继续任务直接令 γ=1；比较不同 γ 的原始 value 大小。

<a id="i04"></a>

### I04 Bellman 方程与动态规划【稳定】

<!-- readings:start -->
**进一步精读：** [DQN：经验回放与目标网络](../../readings/papers/dqn.md)
<!-- readings:end -->
- **先修**：I03、全概率公式、矩阵迭代。
- **定义与解析**：Bellman 方程把长时程价值拆成一步奖励加后继价值；已知完整模型时可做策略评估、策略迭代或价值迭代。
- **公式/机制**：$V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)\left[R+\gamma V^*(s')\right]$；最优算子在 $\gamma<1$ 时为压缩映射。
- **资料**：Sutton & Barto [Ch.4，尤其 §4.4](http://incompleteideas.net/book/RLbook2020.pdf)，算法框见 p.83。
- **最小代码（可执行）**：
```python
import numpy as np
P=np.array([[[1,0],[0,1]],[[.5,.5],[0,1.]]],float); R=np.array([[0,1],[0,2.]])
V=np.zeros(2); gamma=.9
for _ in range(100):
    Q=R+gamma*np.einsum('ast,t->as',P,V)
    new=Q.max(1)
    if np.max(abs(new-V))<1e-10: break
    V=new
print(V,Q.argmax(1))
assert np.all(np.isfinite(V))
```
- **检测/实验**：区分“对固定策略求期望”与“对动作取最大”；将最大误写到求和内会发生什么？
- **常见坑**：奖励张量索引错位；终止状态仍 bootstrap；用动态规划却声称 model-free。

<a id="i05"></a>

### I05 Monte Carlo 估计与重要性采样【稳定】
- **先修**：I03、样本均值、大数定律。
- **定义与解析**：MC 等完整 episode 后用实际回报估计价值，无 bootstrap、偏差低但方差可大；off-policy 时用似然比修正分布。
- **公式/机制**：$V(s)\leftarrow V(s)+\alpha\left(G-V(s)\right)$；$\rho_{t:T-1}=\prod_k\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}$。
- **资料**：Sutton & Barto [Ch.5，§5.1 与 §5.5](http://incompleteideas.net/book/RLbook2020.pdf)。
- **最小代码（可执行）**：
```python
import numpy as np
def estimate(seed,n=5000):
    g=np.random.default_rng(seed); returns=[]
    for _ in range(n):
        rewards=g.normal([1,2,3],1); returns.append(rewards@[1,.9,.81])
    return np.mean(returns)
vals=[estimate(s) for s in range(5)]
print(np.mean(vals),np.std(vals))
assert abs(np.mean(vals)-(1+1.8+2.43))<.1
```
- **检测/实验**：把 episode 长度从 3 增到 100，观察估计方差；说明普通与加权重要性采样的偏差—方差取舍。
- **常见坑**：把每步相关样本当 IID；行为策略对目标动作概率为零仍做修正；只报最后一条 episode。

<a id="i06"></a>

### I06 TD、SARSA 与 Q-learning【稳定】

<!-- readings:start -->
**进一步精读：** [DQN：经验回放与目标网络](../../readings/papers/dqn.md)
<!-- readings:end -->
- **先修**：I03–I05、随机逼近。
- **定义与解析**：TD 用下一状态估计 bootstrap；SARSA 学行为策略价值，Q-learning 用最大动作目标学习 off-policy 最优价值。
- **公式/机制**：$Q(s,a)\leftarrow Q(s,a)+\alpha\left[r+\gamma Q(s',a')-Q(s,a)\right]$；Q-learning 将 $Q(s',a')$ 换成 $\max_{a'}Q(s',a')$。
- **资料**：Sutton & Barto [§6.1–6.5](http://incompleteideas.net/book/RLbook2020.pdf)；Gymnasium [Blackjack Q-learning](https://gymnasium.farama.org/main/introduction/train_agent/)。
- **最小代码（可执行；训练/评估分离且五种子）**：
```python
import numpy as np
def train(seed):
    g=np.random.default_rng(seed); Q=np.zeros((6,2))
    for ep in range(800):
        s=0
        while s<5:
            a=g.integers(2) if g.random()<max(.05,1-ep/500) else Q[s].argmax()
            ns=max(0,min(5,s+(-1,1)[a])); r=ns==5
            Q[s,a]+=.2*(r+.95*(ns<5)*Q[ns].max()-Q[s,a]); s=ns
    return Q
def evaluate(Q): return all(Q[s].argmax()==1 for s in range(5))
assert sum(evaluate(train(s)) for s in range(5))>=4
```
- **检测/实验**：Cliff Walking 中为什么 SARSA 可能比 Q-learning 走得更安全？验收需冻结 Q 后单独评估。
- **常见坑**：评估仍用 ε-greedy；将截断一律视为终止；看训练移动平均而没有独立 episode。

<a id="i07"></a>

### I07 函数逼近、经验回放与 DQN【稳定基础】

<!-- readings:start -->
**进一步精读：** [DQN：经验回放与目标网络](../../readings/papers/dqn.md)
<!-- readings:end -->
- **先修**：I06、MLP、反向传播、目标网络。
- **定义与解析**：DQN 用神经网络近似离散动作 Q；回放打散相关性，延迟目标网络缓和“追逐移动目标”。
- **公式/机制**：$y=r+\gamma(1-d)\max_{a'}Q_{\bar\theta}(s',a')$，最小化 $\operatorname{Huber}\!\left(Q_\theta(s,a)-y\right)$。
- **资料**：Mnih et al. [Nature 2015，Methods 与 Extended Data](https://www.nature.com/articles/nature14236)；PyTorch [CartPole DQN 教程，Replay Memory/Training](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)。
- **最小代码（可执行的一步损失）**：
```python
import torch
torch.manual_seed(0); B,S,A=8,4,2
q=torch.nn.Linear(S,A); target=torch.nn.Linear(S,A); target.load_state_dict(q.state_dict())
s=torch.randn(B,S); ns=torch.randn(B,S); a=torch.randint(A,(B,)); r=torch.randn(B); done=torch.rand(B)>.7
pred=q(s).gather(1,a[:,None]).squeeze(1)
with torch.no_grad(): y=r+.99*(~done)*target(ns).max(1).values
loss=torch.nn.functional.smooth_l1_loss(pred,y); loss.backward()
assert torch.isfinite(loss)
```
- **检测/实验**：去掉 target detach、回放或目标网络分别预测故障；完整项目至少 5 个训练种子和冻结策略评估均值/置信区间。
- **常见坑**：对终止状态 bootstrap；训练网络同时生成有梯度 target；以最好种子代表算法。

<a id="i08"></a>

### I08 策略梯度与 REINFORCE【稳定】
- **先修**：I03、概率分布、log-derivative trick、自动微分。
- **定义与解析**：策略梯度直接提高高回报动作的对数概率；适合随机或连续策略，但 MC 回报使方差较大。
- **公式/机制**：$\nabla J(\theta)=\mathbb E\!\left[\nabla_\theta\log\pi_\theta(a\mid s)G_t\right]$；减去与动作无关 baseline 不改变期望。
- **资料**：Sutton & Barto [§13.1–13.3](http://incompleteideas.net/book/RLbook2020.pdf)；OpenAI Spinning Up [VPG Key Equations](https://spinningup.openai.com/en/latest/algorithms/vpg.html#key-equations)。
- **最小代码（可执行的一步更新）**：
```python
import torch
torch.manual_seed(0); logits=torch.nn.Parameter(torch.zeros(3)); opt=torch.optim.SGD([logits],.2)
actions=torch.tensor([0,2,2,1]); returns=torch.tensor([0.,2.,3.,1.])
dist=torch.distributions.Categorical(logits=logits)
baseline=returns.mean()
loss=-(dist.log_prob(actions)*(returns-baseline)).mean()
opt.zero_grad(); loss.backward(); opt.step()
print(logits.detach()); assert logits[2]>logits[0]
```
- **检测/实验**：证明常数 baseline 不改变期望梯度；比较有无 baseline 的五种子梯度方差。
- **常见坑**：对采样动作反传；最大化目标却忘记负号；用同批数据反复更新却称严格 on-policy。

<a id="i09"></a>

### I09 Actor–Critic 与 GAE【稳定】

<!-- readings:start -->
**进一步精读：** [DeepSeekMath：数学训练与GRPO](../../readings/papers/deepseekmath.md)
<!-- readings:end -->
- **先修**：I06、I08、价值函数拟合。
- **定义与解析**：actor 更新策略，critic 估计价值提供低方差 advantage；GAE 用 λ 连续调节 TD 偏差与 MC 方差。
- **公式/机制**：$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$，$\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}$。
- **资料**：Schulman et al. [GAE §2–3，式 (11)–(16)](https://arxiv.org/abs/1506.02438)；Spinning Up [PPO pseudocode](https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode)。
- **最小代码（可执行 GAE）**：
```python
import torch
r=torch.tensor([1.,0.,2.]); v=torch.tensor([.4,.5,.7,0.]); done=torch.tensor([0.,0.,1.])
gamma,lam=.99,.95; adv=torch.zeros(3); carry=0.
for t in range(2,-1,-1):
    delta=r[t]+gamma*(1-done[t])*v[t+1]-v[t]
    carry=delta+gamma*lam*(1-done[t])*carry; adv[t]=carry
ret=adv+v[:-1]
print(adv,ret); assert torch.isfinite(adv).all()
```
- **检测/实验**：λ=0 与 λ=1 分别接近什么？比较 advantage 标准化前后的尺度而非宣称其必然提升。
- **常见坑**：跨 episode 传播 GAE；value target 未 detach；把 advantage 与 return 混作同一监督量。

<a id="i10"></a>

### I10 PPO 与受限策略更新【稳定工程基线】

<!-- readings:start -->
**进一步精读：** [InstructGPT：人类反馈与策略优化](../../readings/papers/instructgpt.md) · [DeepSeekMath：数学训练与GRPO](../../readings/papers/deepseekmath.md) · [TRL：监督偏好与在线策略训练](../../readings/projects/trl.md)
<!-- readings:end -->
- **先修**：I08–I09、重要性比率、KL 散度。
- **定义与解析**：PPO-Clip 用概率比截断减少一次更新离旧策略过远的激励；截断不是严格 KL 约束或单调改进保证。
- **公式/机制**：$L=\mathbb E\!\left[\min\!\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]$，$r_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$。
- **资料**：Schulman et al. [PPO §2，式 (7)](https://arxiv.org/abs/1707.06347)；Spinning Up [PPO Key Equations/Pseudocode](https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations)。
- **最小代码（可执行代理目标）**：
```python
import torch
old=torch.tensor([-.7,-1.2,-.4]); new=torch.tensor([-.5,-1.5,-.35],requires_grad=True)
adv=torch.tensor([1.,-1.,.5]); ratio=(new-old).exp(); eps=.2
raw=ratio*adv; clipped=ratio.clamp(1-eps,1+eps)*adv
loss=-torch.minimum(raw,clipped).mean()
loss.backward(); print(ratio.detach(),new.grad)
assert torch.isfinite(loss)
```
- **检测/实验**：分别画正、负 advantage 下目标随 ratio 的曲线；训练至少 5 种子，评估时冻结参数、动作使用明确 deterministic/stochastic 约定。
- **常见坑**：old log-prob 随更新变化；不监测 KL、clip fraction、value loss；只复用同批数据却不打乱 minibatch。

<a id="i11"></a>

### I11 连续控制与 Soft Actor-Critic【稳定基线】
- **先修**：I09、连续分布、重参数化、双 Q。
- **定义与解析**：SAC 是 off-policy 最大熵 actor–critic；奖励与策略熵共同优化，使连续控制保留探索并复用回放数据。
- **公式/机制**：$J_\pi=\mathbb E\!\left[\alpha\log\pi(a\mid s)-\min_iQ_i(s,a)\right]$；target 含 $Q-\alpha\log\pi$。
- **资料**：Haarnoja et al. [SAC §4，Algorithm 1](https://arxiv.org/abs/1801.01290)；Spinning Up [SAC Key Equations](https://spinningup.openai.com/en/latest/algorithms/sac.html#key-equations)。
- **最小代码（可执行 actor loss 模拟）**：
```python
import torch
torch.manual_seed(0); mu=torch.zeros(6,1,requires_grad=True); log_std=torch.full_like(mu,-.5)
dist=torch.distributions.Normal(mu,log_std.exp()); u=dist.rsample(); a=torch.tanh(u)
logp=dist.log_prob(u)-torch.log(1-a.square()+1e-6)
q=-(a-.4).square(); alpha=.2
loss=(alpha*logp-q).mean(); loss.backward()
print(loss.item(),mu.grad.mean().item()); assert torch.isfinite(loss)
```
- **检测/实验**：调 α 观察动作熵与回报；说明 tanh 后为何需要 log-prob Jacobian 修正。
- **常见坑**：省略双 Q 的最小值；动作缩放与环境边界不一致；把训练采样策略直接当确定性部署策略。

<a id="i12"></a>

### I12 离线强化学习与分布外动作【演进中】
- **先修**：I06–I11、分布偏移、行为策略。
- **定义与解析**：offline RL 只能使用固定数据集；核心难点是策略选择数据支持外动作时，Q 误差会被最大化与 bootstrap 放大。
- **公式/机制**：CQL 在 Bellman loss 外加入 $\alpha\left[\log\sum_a e^{Q(s,a)}-\mathbb E_{a\sim D}Q(s,a)\right]$，压低未被数据支持的动作。
- **资料**：Fu et al. [D4RL §2–4](https://arxiv.org/abs/2004.07219)；Kumar et al. [CQL §3，式 (1)](https://arxiv.org/abs/2006.04779)。
- **最小代码（可执行的离散保守选择演示）**：
```python
import numpy as np
q=np.array([[1.,9.,2.],[3.,8.,1.]])          # 未见动作可能虚高
counts=np.array([[20,0,8],[0,15,4]])
naive=q.argmax(1)
supported=np.where(counts>0,q,-np.inf).argmax(1)
print(naive,supported)
assert np.all(counts[np.arange(2),supported]>0)
assert np.any(counts[np.arange(2),naive]==0)
```
- **检测/实验**：构造 random/medium/expert 三种数据覆盖，比较 BC、普通 Q-learning、保守选择；不能在线调参后仍称纯离线评估。
- **常见坑**：把 replay buffer 训练等同 offline RL；测试环境反馈渗入调参；只报 D4RL normalized score 而不说明版本与归一化。

<a id="i13"></a>

### I13 模仿学习：BC、DAgger 与 GAIL【稳定基础，扩展活跃】
- **先修**：监督学习、I02、I08。
- **定义与解析**：BC 对专家状态动作做监督学习，但自身错误改变后续状态分布；DAgger 在学习者访问的状态上请求专家并聚合数据，GAIL 匹配占用分布。
- **公式/机制**：BC 最小化 $-\sum\log\pi(a_E\mid s_E)$；DAgger 循环 $D\leftarrow D\cup\{(s,\pi_E(s))\}$。
- **资料**：Ross et al. [DAgger Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a.html)；Ho & Ermon [GAIL §3–4](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html)。
- **最小代码（可执行 BC）**：
```python
import torch
torch.manual_seed(0); x=torch.linspace(-1,1,80)[:,None]; y=(x[:,0]>0).long()
pi=torch.nn.Linear(1,2); opt=torch.optim.Adam(pi.parameters(),.05)
for _ in range(100):
    loss=torch.nn.functional.cross_entropy(pi(x),y)
    opt.zero_grad(); loss.backward(); opt.step()
acc=(pi(x).argmax(1)==y).float().mean()
print(acc.item()); assert acc>.95
```
- **检测/实验**：在链式环境逐步注入 1% 动作错误，测成功率随 horizon 的下降；DAgger 需要在线专家，不能假装免费标签。
- **常见坑**：随机切分同一轨迹帧导致泄漏；只测动作 MSE 不测 rollout 成功；专家动作多模态却用单峰回归。

<a id="i14"></a>

### I14 多智能体强化学习与 CTDE【演进中】
- **先修**：I02–I11、博弈论基本概念、联合动作空间。
- **定义与解析**：MARL 中其他学习者使单体看来环境非平稳；集中训练、分散执行（CTDE）可在训练时用全局信息，执行时每个体只用本地观测。
- **公式/机制**：合作 Dec-POMDP 的 $Q_{\mathrm{tot}}(s,\mathbf a)$；QMIX 约束 $\partial Q_{\mathrm{tot}}/\partial Q_i\ge 0$，以保证局部 argmax 与联合 argmax 一致。
- **资料**：PettingZoo [AEC API，Agent iteration](https://pettingzoo.farama.org/api/aec/)；Rashid et al. [QMIX §3，式 (4)](https://proceedings.mlr.press/v80/rashid18a.html)。
- **最小代码（可执行协调博弈）**：
```python
import numpy as np
R=np.array([[2.,0.],[0.,1.]])                 # 联合动作奖励
def value(p,q):
    pa=np.array([p,1-p]); pb=np.array([q,1-q])
    return float(pa@R@pb)
grid=np.linspace(0,1,11)
best=max((value(p,q),p,q) for p in grid for q in grid)
print(best); assert best[0]==2.
```
- **检测/实验**：独立 Q-learning 与共享全局 critic 各跑 5 种子，冻结所有体后联合评估；说明“共享奖励”不等于“共享观测”。
- **常见坑**：异步轮次错配；训练偷看全局状态后执行也依赖它；只报告个体指标、不报告团队成功和最差个体。
