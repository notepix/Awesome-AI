# A. 数学、统计与优化


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [下一章 →](b-programming-data-experiments.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/01-probability-learning.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="a01"></a>

### A01 `[核]` 集合、函数、关系与逻辑记号

**先修：** 无。

**定义与解析：** 集合（set）描述对象的范围；函数（function）是从定义域到陪域、每个输入恰有一个输出的映射；关系（relation）是笛卡尔积的子集。命题逻辑中的否定、合取、析取、蕴含和量词，是读懂定理假设、算法不变量与概率事件的语法。

**公式/机制：** `f:X→Y`；`A⊆B ⇔ ∀x(x∈A⇒x∈B)`；逆否命题 `P⇒Q ≡ ¬Q⇒¬P`。证明“充要条件”必须分别证明两个方向。

**资料定位：** MIT *Mathematics for Computer Science*，[第 1 章 Propositions、第 4 章 Mathematical Data Types](https://courses.csail.mit.edu/6.042/spring18/mcs.pdf)。

```python
U = set(range(6))
even = {x for x in U if x % 2 == 0}
f = lambda x: x * x
image = {f(x) for x in even}
assert even <= U and image == {0, 4, 16}
print(even, image)
```

**检测题/小实验：** 写出“可微必连续”的逆命题、否命题与逆否命题，并判断哪些等价；用代码检查两个有限集合上的 De Morgan 律。

**常见坑：** 混淆“元素”和“子集”、定义域和实际观测样本；把 `P⇒Q` 错读成 `Q⇒P`；用若干例子代替一般性证明。

<a id="a02"></a>

### A02 `[核]` 标量、向量、矩阵与张量

<!-- readings:start -->
**进一步精读：** [Transformer：注意力与编解码器](../../readings/papers/transformer.md)
<!-- readings:end -->

**先修：** A01。

**定义与解析：** 标量（scalar）是单个数，向量（vector）是一阶数组，矩阵（matrix）表示二维线性运算，张量（tensor）在工程语境中通常指多轴数值数组。形状（shape）既是数据布局，也是运算是否合法的类型约束。

**公式/机制：** `x∈R^d`，`A∈R^{m×d}`，批量线性映射 `Y=XA^T+b`；矩阵乘法收缩相邻维度，而 Hadamard 积 `A⊙B` 逐元素计算。

**资料定位：** Goodfellow 等 *Deep Learning*，[2.1 Scalars, Vectors, Matrices and Tensors；2.2 Multiplying Matrices and Vectors](https://www.deeplearningbook.org/contents/linear_algebra.html)。

```python
import numpy as np
x = np.array([1., 2., 3.])          # (3,)
X = np.stack([x, 2 * x])            # (2, 3)
W = np.array([[1., 0., -1.], [0., 1., 1.]])
b = np.array([0.5, -0.5])
Y = X @ W.T + b                      # (2, 2)
print(X.shape, W.shape, Y)
```

**检测题/小实验：** 不运行代码，先写出 `(32,10)@(10,4)` 和 `(32,10)*(10,)` 的输出形状；把最后一式改成显式循环并核对结果。

**常见坑：** 把 `*` 当矩阵乘、忽略一维数组没有“行/列”方向、未经检查就依赖广播，或把 batch 轴与 feature 轴互换。

<a id="a03"></a>

### A03 `[核]` 向量空间、基、秩、线性映射与子空间

**先修：** A02。

**定义与解析：** 向量空间（vector space）对加法和数乘封闭；基（basis）是线性无关且张成全空间的向量组。矩阵是选定基后的线性映射；秩（rank）是像空间维数，零空间（null space）刻画被映到零的方向。

**公式/机制：** `rank(A)+nullity(A)=n`；若列满秩，`Ax=b` 至多一个解；换基改变坐标而不改变抽象向量。线性无关等价于 `Σ_i α_i v_i=0` 只有零系数解。

**资料定位：** *Deep Learning*，[2.3 Identity and Inverse Matrices；2.4 Linear Dependence and Span](https://www.deeplearningbook.org/contents/linear_algebra.html)。

```python
import numpy as np
A = np.array([[1., 2., 3.], [2., 4., 6.]])
rank = np.linalg.matrix_rank(A)
_, s, vt = np.linalg.svd(A)
null_vec = vt[-1]
print("rank=", rank, "nullity=", A.shape[1] - rank)
print("A v ≈", A @ null_vec, "singular values=", s)
```

**检测题/小实验：** 构造一个 `3×4`、秩为 2 的矩阵，数值求零空间并验证秩—零度定理；解释为何相关特征会使线性回归参数不唯一。

**常见坑：** 把“矩阵可逆”等同于“任何矩阵有逆”；把小但非零奇异值机械判零；混淆行空间、列空间和输入空间。

<a id="a04"></a>

### A04 `[核]` 内积、范数、距离、正交与投影

**先修：** A02、A03。

**定义与解析：** 内积（inner product）给出长度和角度；范数（norm）度量大小；距离（metric）还需满足非负、对称与三角不等式。正交投影把向量分解为子空间内分量与正交残差，是最小二乘、PCA 和注意力相似度的共同几何语言。

**公式/机制：** `||x||_2=sqrt(x^Tx)`，`cos(x,y)=x^Ty/(||x||||y||)`；正交列矩阵 `Q` 上的投影为 `P=QQ^T`，且 `P^2=P=P^T`。

**资料定位：** *Deep Learning*，[2.5 Norms；2.6 Special Kinds of Matrices and Vectors](https://www.deeplearningbook.org/contents/linear_algebra.html)。

```python
import numpy as np
x = np.array([2., 1.])
q = np.array([1., 1.]); q /= np.linalg.norm(q)
proj = q * (q @ x)
residual = x - proj
cosine = (x @ q) / np.linalg.norm(x)
print("projection", proj, "residual", residual)
print("orthogonal?", np.isclose(q @ residual, 0), "cos", cosine)
```

**检测题/小实验：** 比较同一组二维点在 `L1`、`L2` 和余弦距离下的最近邻；证明投影残差与子空间正交。

**常见坑：** 未标准化就比较余弦与点积；认为任意“相异度”都是度量；用非正交基时仍套用 `QQ^T`。

<a id="a05"></a>

### A05 `[核]` 特征分解、SVD、正定矩阵与二次型

<!-- readings:start -->
**进一步精读：** [LoRA：低秩任务更新](../../readings/papers/lora.md) · [GCN：图归一化与节点分类](../../readings/papers/gcn.md)
<!-- readings:end -->

**先修：** A03、A04。

**定义与解析：** 特征分解（eigendecomposition）寻找线性映射不改变方向的向量；SVD 适用于任意矩阵，将映射拆为旋转—缩放—旋转。正半定（PSD）矩阵满足所有二次型非负，常见于协方差、Hessian 与核矩阵。

**公式/机制：** 对称矩阵 `A=QΛQ^T`；任意 `X=UΣV^T`；`X^TX=VΣ²V^T`；`A≽0 ⇔ x^TAx≥0`。截断 SVD 给 Frobenius 范数下最佳低秩近似。

**资料定位：** *Deep Learning*，[2.7 Eigendecomposition；2.8 Singular Value Decomposition；2.11 The Determinant](https://www.deeplearningbook.org/contents/linear_algebra.html)。

```python
import numpy as np
X = np.array([[3., 1.], [1., 3.], [2., 2.]])
U, s, Vt = np.linalg.svd(X, full_matrices=False)
X_rank1 = s[0] * np.outer(U[:, 0], Vt[0])
C = X.T @ X
eig = np.linalg.eigvalsh(C)
print("singular values", s, "PSD eigenvalues", eig)
print("rank-1 error", np.linalg.norm(X - X_rank1))
```

**检测题/小实验：** 随机生成矩阵，验证 `X^TX` 的特征值等于奇异值平方；随截断秩增加绘制重构误差。

**常见坑：** 对非对称矩阵假设正交特征向量；直接求逆而非 `solve/lstsq`；把特征值与奇异值当成同一概念。

<a id="a06"></a>

### A06 `[核]` 单变量与多变量微分、偏导和梯度

**先修：** A01、A02。

**定义与解析：** 导数（derivative）是局部线性变化率；偏导固定其他变量；梯度（gradient）收集标量函数对各坐标的偏导，并指向欧氏几何下最陡上升方向。方向导数把变化限定到方向 `v`。

**公式/机制：** `f(x+Δ)≈f(x)+∇f(x)^TΔ`；`D_v f=∇f^Tv`。对 `f(x)=||Ax-b||²/2`，有 `∇f=A^T(Ax-b)`。

**资料定位：** *Mathematics for Machine Learning*，[第 5 章 Vector Calculus，5.1–5.3](https://mml-book.github.io/book/mml-book.pdf)；*Deep Learning*，[4.3 Gradient-Based Optimization](https://www.deeplearningbook.org/contents/numerical.html)。

```python
import numpy as np
A = np.array([[1., 2.], [3., -1.]])
b = np.array([1., 0.]); x = np.array([.4, -.2])
f = lambda z: 0.5 * np.sum((A @ z - b) ** 2)
grad = A.T @ (A @ x - b)
eps = 1e-6
fd = np.array([(f(x + eps*np.eye(2)[i]) - f(x - eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
print(grad, fd, np.linalg.norm(grad - fd))
```

**检测题/小实验：** 推导 `x^TAx` 在 `A` 非对称时的梯度；对不同 `eps` 做有限差分，观察截断误差与浮点误差的折中。

**常见坑：** 忽略向量形状；把梯度当普通“分数”；有限差分步长越小越好；漏掉均值损失中的样本数因子。

<a id="a07"></a>

### A07 `[核]` Jacobian、Hessian、链式法则与矩阵微积分

**先修：** A02、A06。

**定义与解析：** Jacobian 是向量函数的一阶导数矩阵；Hessian 是标量函数的二阶偏导矩阵。链式法则把复合计算图上的局部导数相乘；反向模式自动微分高效计算“一个标量输出对很多参数”的梯度。

**公式/机制：** 若 `z=g(x), y=f(z)`，则 `J_{f∘g}=J_f J_g`；Hessian `H_ij=∂²f/∂x_i∂x_j`。反传计算 vector–Jacobian product，而非显式构造每层完整 Jacobian。

**资料定位：** *Mathematics for Machine Learning*，[5.4 Gradients of Matrices、5.5 Useful Identities for Computing Gradients](https://mml-book.github.io/book/mml-book.pdf)；PyTorch，[Automatic Differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

```python
import torch
x = torch.tensor([1., 2.], requires_grad=True)
A = torch.tensor([[2., 0.], [1., 3.]])
z = torch.tanh(A @ x)
y = (z ** 2).sum()
y.backward()
auto = x.grad.clone()
J = torch.diag(1 - z**2) @ A
manual = J.T @ (2 * z)
print(auto, manual)
```

**检测题/小实验：** 手算两层标量网络的前向值和梯度，再与 autograd 对比；解释反向模式为何适合训练、前向模式为何适合少输入多输出。

**常见坑：** Jacobian 乘法次序颠倒；原地修改破坏计算图；忘记梯度会累加；把 Hessian 的正定性误当成全局凸性。

<a id="a08"></a>

### A08 `[核]` Taylor 展开、局部近似与曲率

<!-- readings:start -->
**进一步精读：** [XGBoost：二阶树提升与系统设计](../../readings/papers/xgboost.md)
<!-- readings:end -->

**先修：** A07。

**定义与解析：** Taylor 展开用某点处的导数近似邻域内函数；一阶项给局部斜率，二阶 Hessian 给曲率。优化中的学习率、Newton 法和“平坦/尖锐”方向都可由局部二次模型理解。

**公式/机制：** `f(x+δ)≈f(x)+g^Tδ+δ^THδ/2`；一维 Newton 更新 `x←x-f'(x)/f''(x)`。余项意味着近似只在足够小邻域可靠。

**资料定位：** *Deep Learning*，[4.3.1 Beyond the Gradient: Jacobian and Hessian Matrices](https://www.deeplearningbook.org/contents/numerical.html)；*Mathematics for Machine Learning*，[5.2.6 Taylor Series](https://mml-book.github.io/book/mml-book.pdf)。

```python
import numpy as np
x0 = 0.4
f = np.sin
g, h = np.cos(x0), -np.sin(x0)
for delta in [1e-1, 5e-1, 1.0]:
    first = f(x0) + g * delta
    second = first + 0.5 * h * delta**2
    truth = f(x0 + delta)
    print(delta, abs(first-truth), abs(second-truth))
```

**检测题/小实验：** 对 `log(1+x)` 比较一阶、二阶近似误差随 `|x|` 的变化；为一个非凸函数找出 Hessian 为正但非全局最优的点。

**常见坑：** 把局部近似当全局等式；认为 Hessian 半正定必为严格极小值；直接显式求逆 Hessian 而忽略数值代价。

<a id="a09"></a>

### A09 `[核]` 概率公理、条件概率与 Bayes 公式

**先修：** A01。

**定义与解析：** 概率空间由样本空间、事件集合和概率测度组成。条件概率是在已知事件后重新归一化；Bayes 公式把“原因给结果”的似然转换为“结果给原因”的后验，是分类、诊断和 Bayesian 学习的基础。

**公式/机制：** `P(A|B)=P(A∩B)/P(B)`；`P(A|B)=P(B|A)P(A)/P(B)`；全概率公式 `P(B)=Σ_i P(B|A_i)P(A_i)`。

**资料定位：** *Deep Learning*，[3.3 Probability Distributions；3.5 Conditional Probability；3.11 Bayes' Rule](https://www.deeplearningbook.org/contents/prob.html)。

```python
prevalence = 0.01
sensitivity = 0.95
specificity = 0.90
p_positive = sensitivity*prevalence + (1-specificity)*(1-prevalence)
posterior = sensitivity * prevalence / p_positive
print("P(disease | positive) =", round(posterior, 4))
odds_prior = prevalence / (1 - prevalence)
odds_post = odds_prior * sensitivity / (1 - specificity)
print("odds check =", odds_post / (1 + odds_post))
```

**检测题/小实验：** 改变患病率并画出阳性后验；解释为什么 95% 灵敏度的检测在低基率下仍可能有大量假阳性。

**常见坑：** 混淆 `P(A|B)` 与 `P(B|A)`；忽略基率；把互斥误当独立；条件事件概率为零时仍套公式。

<a id="a10"></a>

### A10 `[核]` 随机变量、常见分布、期望与方差

<!-- readings:start -->
**进一步精读：** [Adam：自适应梯度与偏差校正](../../readings/papers/adam.md)
<!-- readings:end -->

**先修：** A09。

**定义与解析：** 随机变量（random variable）把随机结果映射为数值；离散变量用概率质量函数（PMF），连续变量用概率密度函数（PDF）。期望是按概率加权的长期平均，方差度量围绕均值的二阶波动；分布而非单次样本才是模型对象。

**公式/机制：** `E[X]=Σ_x xp(x)` 或 `∫xp(x)dx`；`Var(X)=E[(X-E[X])²]=E[X²]-E[X]²`。Bernoulli、Categorical、Gaussian、Exponential 是常用构件。

**资料定位：** *Deep Learning*，[3.2 Random Variables；3.8 Expectation, Variance and Covariance；3.9 Common Probability Distributions](https://www.deeplearningbook.org/contents/prob.html)。

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(loc=2.0, scale=3.0, size=20_000)
print("sample mean/var", x.mean(), x.var())
values = np.array([0., 1., 2.])
pmf = np.array([0.2, 0.5, 0.3])
mu = values @ pmf
var = ((values - mu) ** 2) @ pmf
print("discrete mean/var", mu, var)
```

**检测题/小实验：** 推导 Bernoulli(`p`) 的均值和方差；分别从 Gaussian 与 Laplace 分布抽样，比较相同方差下极端值比例。

**常见坑：** 把密度值当区间概率；样本方差与总体方差分母混淆；只报均值而忽略分布形状和尾部。

<a id="a11"></a>

### A11 `[核]` 联合、边缘、条件分布、独立性与协方差

<!-- readings:start -->
**进一步精读：** [DDPM：逐步去噪生成](../../readings/papers/ddpm.md)
<!-- readings:end -->

**先修：** A10。

**定义与解析：** 联合分布（joint distribution）描述变量共同变化；边缘化消去不关心的变量；条件分布在给定信息后更新不确定性。独立意味着联合分解为边缘乘积，零协方差通常只说明无线性相关，并不保证独立。

**公式/机制：** `p(x)=Σ_y p(x,y)`；`p(x,y)=p(x|y)p(y)`；独立时 `p(x,y)=p(x)p(y)`；`Cov(X,Y)=E[(X-μ_X)(Y-μ_Y)]`。

**资料定位：** *Deep Learning*，[3.4 Marginal Probability；3.6 The Chain Rule；3.7 Independence and Conditional Independence；3.8 Expectation, Variance and Covariance](https://www.deeplearningbook.org/contents/prob.html)。

```python
import numpy as np
rng = np.random.default_rng(1)
z = rng.normal(size=10_000)
x = z + rng.normal(scale=.5, size=z.size)
y = z + rng.normal(scale=.5, size=z.size)
print("marginal corr", np.corrcoef(x, y)[0, 1])
rx, ry = x - z, y - z                  # 给定共同原因后的残差
print("conditional-residual corr", np.corrcoef(rx, ry)[0, 1])
```

**检测题/小实验：** 构造 `Y=X²`、对称分布的 `X`，验证 `Cov(X,Y)=0` 但不独立；用一张三变量表手算边缘与条件概率。

**常见坑：** 将相关解释为因果；把不相关等同独立；漏掉共同原因；从有限样本的小相关系数断言条件独立。

<a id="a12"></a>

### A12 `[核]` 大数定律、中心极限定理与集中现象

**先修：** A10、A11。

**定义与解析：** 大数定律（LLN）说明独立同分布样本均值趋近总体期望；中心极限定理（CLT）说明适当标准化的样本均值趋近 Gaussian。集中不等式给有限样本偏离概率上界，是理解 mini-batch 噪声和统计置信度的基础。

**公式/机制：** `X̄_n→E[X]`；`√n(X̄_n-μ)/σ ⇒ N(0,1)`；Chebyshev：`P(|X-μ|≥t)≤σ²/t²`。CLT 不是“原分布会变成正态”。

**资料定位：** MIT OpenCourseWare 18.05，[Reading 6b: Central Limit Theorem and the Law of Large Numbers](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/resources/mit18_05_s22_class06-prep-b_pdf/)。

```python
import numpy as np
rng = np.random.default_rng(2)
for n in [2, 10, 100]:
    means = rng.exponential(size=(5000, n)).mean(axis=1)
    z = np.sqrt(n) * (means - 1.0)       # Exponential(1): μ=σ=1
    print(n, "mean/std/skew", z.mean(), z.std(),
          np.mean(((z-z.mean())/z.std())**3))
```

**检测题/小实验：** 以 Bernoulli 和重尾分布重复实验，观察均值误差如何随 `n` 缩小；说明 CLT 需要哪些条件、何时收敛很慢。

**常见坑：** 认为小样本也近似正态；忽略独立性和有限方差条件；把概率收敛误作每条样本路径单调收敛。

<a id="a13"></a>

### A13 `[核]` 参数估计、MLE、MAP 与 Bayesian 推断

<!-- readings:start -->
**进一步精读：** [VAE：变分下界与可微采样](../../readings/papers/vae.md)
<!-- readings:end -->

**先修：** A06、A09–A11。

**定义与解析：** 估计量（estimator）是由数据计算参数的规则。最大似然（MLE）选择使观测数据最可能的参数；MAP 最大化似然乘先验；完整 Bayesian 推断保留后验分布并对参数积分，而非只给点估计。

**公式/机制：** `θ_MLE=argmax_θ Σ_i log p(x_i|θ)`；`p(θ|D)∝p(D|θ)p(θ)`；`θ_MAP=argmax[log p(D|θ)+log p(θ)]`。Beta–Bernoulli 后验仍为 Beta。

**资料定位：** *Deep Learning*，[5.4 Estimators, Bias and Variance；5.5 Maximum Likelihood Estimation；5.6 Bayesian Statistics](https://www.deeplearningbook.org/contents/ml.html)。

```python
import numpy as np
x = np.array([1, 1, 0, 1, 0])
mle = x.mean()
alpha, beta = 2., 2.                 # Beta 先验
post_a, post_b = alpha+x.sum(), beta+len(x)-x.sum()
map_est = (post_a - 1) / (post_a + post_b - 2)
post_mean = post_a / (post_a + post_b)
print("MLE", mle, "MAP", map_est, "posterior mean", post_mean)
```

**检测题/小实验：** 改变样本量和 Beta 先验强度，画 MLE、MAP、后验均值；解释为何数据增多后合理先验影响减弱。

**常见坑：** 把似然 `p(D|θ)` 当后验 `p(θ|D)`；忽略 log-likelihood；把 MAP 称作完整 Bayesian 推断；先验由测试集调出。

<a id="a14"></a>

### A14 `[核]` 置信区间、假设检验、Bootstrap 与显著性

**先修：** A12、A13。

**定义与解析：** 置信区间（confidence interval）是重复抽样程序的覆盖率陈述，不是固定参数“落在本次区间的概率”。假设检验用零假设下统计量的尾部概率衡量数据极端程度；Bootstrap 通过重采样近似估计量的抽样分布。

**公式/机制：** 大样本均值区间 `X̄±z_{1-α/2}s/√n`；`p-value=P(T≥T_obs|H0)`。显著不等于效应大，未显著也不等于两者相同。

**资料定位：** NIST/SEMATECH *e-Handbook*，[7.1.4 What are confidence intervals?](https://www.itl.nist.gov/div898/handbook/prc/section1/prc14.htm)；MIT OpenCourseWare 18.05，[Reading 24: Bootstrap confidence intervals](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/resources/mit18_05_s22_class24-prep_pdf/)。

```python
import numpy as np
rng = np.random.default_rng(3)
x = np.array([1.2, .8, 1.5, .9, 1.1, 2.0, .7])
idx = rng.integers(0, len(x), size=(5000, len(x)))
boot_means = x[idx].mean(axis=1)
lo, hi = np.quantile(boot_means, [.025, .975])
print("mean", x.mean(), "bootstrap 95% CI", (lo, hi))
print("P(bootstrap mean <= 1)", np.mean(boot_means <= 1.0))
```

**检测题/小实验：** 模拟 1000 次样本并检查 95% 区间的覆盖率；同时报效应量、区间和 p-value，比较各自回答的问题。

**常见坑：** 把 p-value 当零假设为真的概率；反复试验后只报告显著者；对强依赖序列进行普通 i.i.d. Bootstrap。

<a id="a15"></a>

### A15 `[核]` 熵、交叉熵、KL 散度、互信息与编码长度

<!-- readings:start -->
**进一步精读：** [DPO：从偏好直接优化策略](../../readings/papers/dpo.md)
<!-- readings:end -->

**先修：** A09–A11。

**定义与解析：** 熵（entropy）是不确定性的平均编码长度；交叉熵评价用分布 `q` 编码真实分布 `p` 的代价；KL 散度是额外编码代价。互信息衡量知道一个变量后另一个变量不确定性减少多少。

**公式/机制：** `H(p)=-Σp log p`；`H(p,q)=H(p)+D_KL(p||q)`；`D_KL=Σp log(p/q)≥0`；`I(X;Y)=D_KL(p(x,y)||p(x)p(y))`。

**资料定位：** *Deep Learning*，[3.13 Information Theory](https://www.deeplearningbook.org/contents/prob.html)。

```python
import numpy as np
p = np.array([0.8, 0.2])
q = np.array([0.6, 0.4])
H = -(p * np.log2(p)).sum()
CE = -(p * np.log2(q)).sum()
KL = (p * np.log2(p / q)).sum()
print("H, CE, KL", H, CE, KL)
assert np.allclose(CE, H + KL)
print("KL reverse", (q * np.log2(q / p)).sum())
```

**检测题/小实验：** 对固定 `p` 扫描二分类 `q`，找交叉熵最小点；用数值例子证明 KL 不对称，并说明它为何不是距离。

**常见坑：** 混用自然对数和二进制对数却比较绝对值；处理 `p=0`、`q=0` 不当；把低熵等同高质量预测。

<a id="a16"></a>

### A16 `[核]` 浮点数、数值稳定性、条件数与数值线性代数

<!-- readings:start -->
**进一步精读：** [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md)
<!-- readings:end -->

**先修：** A02、A05、A06。

**定义与解析：** 浮点数只有有限精度和动态范围；舍入、上溢、下溢和灾难性消减会让数学等价式产生不同数值结果。条件数描述输入微扰被问题本身放大的程度，稳定算法则避免额外放大误差。

**公式/机制：** `κ(A)=||A||||A^{-1}||`；稳定 log-sum-exp：`logΣe^{x_i}=m+logΣe^{x_i-m}`，其中 `m=max x_i`。解线性方程优先 `solve/lstsq`，避免显式逆。

**资料定位：** *Deep Learning*，[4.1 Overflow and Underflow；4.2 Poor Conditioning](https://www.deeplearningbook.org/contents/numerical.html)；NumPy，[Linear algebra routines](https://numpy.org/doc/stable/reference/routines.linalg.html)。

```python
import numpy as np
x = np.array([1000., 1001., 999.])
naive = np.log(np.exp(x).sum())       # overflow
m = x.max()
stable = m + np.log(np.exp(x - m).sum())
A = np.array([[1., 1.], [1., 1.000001]])
b = np.array([2., 2.000001])
print("naive/stable", naive, stable)
print("condition/solution", np.linalg.cond(A), np.linalg.solve(A, b))
```

**检测题/小实验：** 比较 `float16/32/64` 累加百万个小数的误差；扰动病态矩阵的 `b`，观察解的变化。

**常见坑：** 用完全相等比较浮点数；以为更高精度能修复病态问题；手写不稳定 softmax；显式计算逆矩阵。

<a id="a17"></a>

### A17 `[核]` 梯度下降、凸性、约束优化与对偶思想

<!-- readings:start -->
**进一步精读：** [Chinchilla：计算最优的模型数据分配](../../readings/papers/chinchilla.md)
<!-- readings:end -->

**先修：** A07、A08。

**定义与解析：** 优化寻找目标函数的最小点。凸函数的局部极小即全局极小；梯度下降沿负梯度迭代；约束问题可用投影、Lagrange 乘子或对偶处理。非凸深度网络通常只能寻求良好驻点。

**公式/机制：** `x_{t+1}=x_t-η∇f(x_t)`；凸性 `f(λx+(1-λ)y)≤λf(x)+(1-λ)f(y)`；Lagrangian `L(x,λ)=f(x)+λ^Tg(x)`。

**资料定位：** *Deep Learning*，[4.3 Gradient-Based Optimization；4.4 Constrained Optimization](https://www.deeplearningbook.org/contents/numerical.html)；*Mathematics for Machine Learning*，[第 7 章 Continuous Optimization](https://mml-book.github.io/book/mml-book.pdf)。

```python
import numpy as np
Q = np.diag([1., 8.]); c = np.array([-2., 1.])
x = np.array([2., 2.]); lr = 0.12
for _ in range(40):
    grad = Q @ x + c
    x -= lr * grad
    x = np.clip(x, -1., 1.)           # 投影到盒约束
objective = 0.5*x@Q@x + c@x
print("x", x, "f", objective, "grad", Q@x+c)
```

**检测题/小实验：** 对同一二次函数尝试稳定、过大和过小学习率；画轨迹并由最大特征值解释稳定上界。

**常见坑：** 梯度为零就宣称全局最优；不缩放特征却统一学习率；约束后只截断一次；把训练损失最小当泛化最好。

<a id="a18"></a>

### A18 `[核]` 随机优化、经验风险、正则化与泛化

<!-- readings:start -->
**进一步精读：** [Adam：自适应梯度与偏差校正](../../readings/papers/adam.md)
<!-- readings:end -->

**先修：** A12、A13、A17。

**定义与解析：** 经验风险最小化（ERM）用有限样本平均损失近似总体风险。SGD 用 mini-batch 梯度作无偏或近似无偏估计；正则化限制有效容量以改善未见数据表现。泛化差距必须用未参与选择的数据估计。

**公式/机制：** `R(θ)=E_{(x,y)∼P}ℓ(fθ(x),y)`，`R̂_n=Σℓ_i/n`；`θ←θ-η(1/|B|)Σ_{i∈B}∇ℓ_i`；正则目标 `R̂+λΩ(θ)`。

**资料定位：** *Deep Learning*，[5.2 Capacity, Overfitting and Underfitting；5.9 Stochastic Gradient Descent](https://www.deeplearningbook.org/contents/ml.html)；[8.1 How Learning Differs from Pure Optimization](https://www.deeplearningbook.org/contents/optimization.html)。

```python
import numpy as np
rng = np.random.default_rng(4)
xtr = np.linspace(-1, 1, 18); ytr = np.sin(3*xtr)+rng.normal(0,.2,18)
xte = np.linspace(-1, 1, 200); yte = np.sin(3*xte)
Xtr = np.vander(xtr, 13, increasing=True)
Xte = np.vander(xte, 13, increasing=True)
for lam in [0., 1e-3, 1e-1]:
    w = np.linalg.solve(Xtr.T@Xtr + lam*np.eye(13), Xtr.T@ytr)
    print(lam, "train/test", np.mean((Xtr@w-ytr)**2), np.mean((Xte@w-yte)**2))
```

**检测题/小实验：** 改变多项式阶数、样本数和 `λ`，画训练/测试误差；解释正则化为何可能提高训练误差却降低测试误差。

**常见坑：** 在测试集调超参数；把 SGD 噪声一概视为缺陷；混淆 L2 penalty 与某些优化器中的 decoupled weight decay。
