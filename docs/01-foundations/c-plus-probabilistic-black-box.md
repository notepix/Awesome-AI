# C+. 概率函数模型与无梯度优化

这一小节补足常规“监督学习→深度学习”课程容易略过、但在小数据不确定性、科学计算与不可微目标中很重要的三类工具。


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](c-machine-learning.md) · [下一章 →](d-deep-learning.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/08-domain-bridges.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="c14"></a>

### C14 `[选]` Gaussian Process、核先验与 Bayesian Optimization

**先修**：A05 正定矩阵，A09–A13 概率与 Bayesian 推断，C02 线性回归，C09 核方法。

**定义与解析**：Gaussian Process（GP）不是“高斯形状的一条曲线”，而是对函数的概率分布：任意有限个函数值都服从联合高斯分布，记作 $f\sim\mathcal{GP}(m,k)$。核 $k(x,x')$ 编码“两个输入的函数值应有多相似”；观测数据通过高斯条件分布把函数先验更新成后验。它不仅给预测均值，还给与数据距离、噪声和核假设相关的预测方差。Bayesian Optimization 再把这个后验当作昂贵黑盒目标的代理，以 acquisition function 权衡探索与利用。

**公式 / 机制**：令 $K_{ij}=k(x_i,x_j)$、观测噪声方差为 $\sigma_n^2$。测试点 $x_*$ 的后验均值和方差为
$$\mu_*=k_*^\top(K+\sigma_n^2I)^{-1}y,$$
$$\sigma_*^2=k(x_*,x_*)-k_*^\top(K+\sigma_n^2I)^{-1}k_*.$$
实际实现用 Cholesky 分解求解线性方程，不显式求逆。精确 GP 通常有 $O(n^3)$ 训练时间与 $O(n^2)$ 存储，因此不是大样本默认方案。

**精确资料**：Rasmussen & Williams 的开放教材 [*Gaussian Processes for Machine Learning* 章节页](https://gaussianprocess.org/gpml/chapters/)：Ch. 2 §2.2–2.3 学回归后验，Ch. 4 §4.1–4.2 学 covariance function；实现可对照 [scikit-learn Gaussian processes 用户指南](https://scikit-learn.org/stable/modules/gaussian_process.html)。

**代码（NumPy，CPU）**：

```python
import numpy as np
x = np.array([-1.0, 0.0, 1.0])[:, None]
y = np.sin(3 * x[:, 0])
xs = np.linspace(-2, 2, 80)[:, None]
rbf = lambda a, b: np.exp(-0.5 * (a - b.T) ** 2 / 0.7**2)
K = rbf(x, x) + 0.05**2 * np.eye(len(x))
L = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
Ks = rbf(x, xs)
mean = Ks.T @ alpha
v = np.linalg.solve(L, Ks)
var = np.diag(rbf(xs, xs)) - np.sum(v * v, axis=0)
print(mean[[0, 40, -1]], np.sqrt(np.maximum(var[[0, 40, -1]], 0)))
```

**检测题 / 小实验**：把长度尺度 `0.7` 分别改为 `0.15` 与 `3.0`，画出均值和 $\mu\pm2\sigma$；解释哪一个先验更偏好快速变化的函数。再在方差最大的测试点加入一次观测，验证该区域的后验方差下降。

**常见误区**：把后验方差当作真实世界全部风险；只调核而不看边缘似然和残差；显式计算矩阵逆；把“置信高”误作“模型假设正确”。

---

<a id="c15"></a>

### C15 `[选]` MCMC、变分推断与近似 Bayesian 计算

<!-- readings:start -->
**进一步精读：** [VAE：变分下界与可微采样](../../readings/papers/vae.md)
<!-- readings:end -->

**先修**：A09–A15，C13 概率图模型。

**后续联系**：完成本单元后可在 G03 从生成模型角度重看 ELBO。

**定义与解析**：Bayesian 推断的目标是后验 $p(z\mid x)=p(x,z)/p(x)$，但证据 $p(x)=\int p(x,z)dz$ 往往算不出。MCMC 构造以目标后验为平稳分布的 Markov chain，用相关样本近似期望；变分推断（VI）选一个易处理的分布族 $q_\phi(z)$，把推断转化为优化。MCMC 渐近上更忠于目标但可能混合慢；VI 通常更快、更易扩展，却受分布族与优化偏差影响。

**公式 / 机制**：Metropolis–Hastings 从 proposal $q(z'\mid z)$ 提议新状态，以
$$a=\min\!\left(1,\frac{p(z'\mid x)q(z\mid z')}{p(z\mid x)q(z'\mid z)}\right)$$
接受。VI 则最大化
$$\operatorname{ELBO}=\mathbb E_q[\log p(x,z)]-\mathbb E_q[\log q_\phi(z)]
=\log p(x)-D_{KL}(q_\phi\Vert p(\cdot\mid x)).$$

**精确资料**：[Stanford CS228 课程页](https://cs.stanford.edu/~ermon/cs228/index.html) 的 Week 5 “Sampling”（Koller & Friedman Ch. 12）和 Week 9 “Exponential families; variational inference”（Wainwright & Jordan §3）；再读 [D2L §18.4 Markov Chain Monte Carlo](https://d2l.ai/chapter_gaussian-processes/mcmc.html) 的 Metropolis–Hastings 推导与代码。

**代码（随机游走 Metropolis，CPU）**：

```python
import numpy as np
rng = np.random.default_rng(0)
logp = lambda z: -0.5 * z**2                 # N(0, 1)，常数省略
z, samples, accepted = 8.0, [], 0
for t in range(12_000):
    proposal = z + rng.normal(scale=1.2)
    if np.log(rng.random()) < logp(proposal) - logp(z):
        z, accepted = proposal, accepted + 1
    if t >= 2_000:                            # 丢弃 burn-in
        samples.append(z)
samples = np.asarray(samples)
print(samples.mean(), samples.std(), accepted / 12_000)
```

**检测题 / 小实验**：把 proposal 标准差改为 `0.02`、`1.2`、`20`，比较接受率、自相关与有效样本量；说明“接受率最高”为什么不等于“估计最好”。用四条不同初值的链检查它们是否混到同一分布区域。

**常见误区**：把迭代次数当作独立样本数；只看接受率不看 trace/autocorrelation；漏掉 burn-in 与多链诊断；认为 ELBO 更大就保证近似覆盖全部后验模式；把近似推断输出当成经过校准的真概率。

---

<a id="c16"></a>

### C16 `[选]` 进化算法、随机搜索与黑盒优化

**先修**：A17–A18 优化，B06 受控实验，C14 Bayesian Optimization。

**后续联系**：完成后可与 I01 的探索—利用并列比较。

**定义与解析**：黑盒优化只要求能查询 $f(x)$，不要求目标可微、连续或有解析式。随机搜索建立重要基线；进化算法维护一组候选，反复执行选择、变异和重组，让高质量候选更可能产生下一代。它们适合离散结构、仿真器、编译器参数或梯度不可靠的目标，但样本效率常低于可用正确梯度时的一阶方法。

**公式 / 机制**：最简单的 evolution strategy 可从 $x_i=\mu+\sigma\epsilon_i$、$\epsilon_i\sim\mathcal N(0,I)$ 采样，按 fitness 选出前 $k$ 个并更新
$$\mu\leftarrow\frac1k\sum_{i\in\text{elite}}x_i.$$
“population → evaluate → select → vary”是机制骨架，不同算法在分布更新、协方差、自适应步长和约束处理上不同。

**精确资料**：[Nevergrad 官方 Optimization 文档](https://github.com/facebookresearch/nevergrad/blob/main/docs/optimization.md) 的 “Ask and tell interface” 与 evolutionary algorithm 示例；理论脉络可读 [ACM Computing Surveys: Quality-Diversity Optimization](https://dl.acm.org/doi/10.1145/3462622) §2（进化搜索基础）和 §3（quality-diversity 容器/选择）。

**代码（极简 evolution strategy，CPU）**：

```python
import numpy as np
rng = np.random.default_rng(7)
f = lambda x: (x[..., 0] - 2) ** 2 + 3 * (x[..., 1] + 1) ** 2
mean, sigma = np.zeros(2), 2.0
for generation in range(60):
    population = mean + sigma * rng.normal(size=(40, 2))
    scores = f(population)
    elite = population[np.argsort(scores)[:8]]
    mean = elite.mean(axis=0)
    sigma *= 0.96
print(mean, f(mean))
```

**检测题 / 小实验**：与同样 2,400 次函数查询的 uniform random search 比较；在 20 个随机种子上报告中位数与四分位数，而不是挑最好的一次。再把 `sigma` 固定，观察收敛速度和最终误差。

**常见误区**：不计算 query budget 就宣称优于梯度法；只报单个幸运种子；过早缩小变异尺度；把 genetic/evolutionary 算法当作天然全局最优保证；在目标可微且梯度廉价时仍盲目使用昂贵黑盒搜索。

---
