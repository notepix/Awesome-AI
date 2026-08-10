# B. 编程、数据与实验基础


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](a-math-statistics-optimization.md) · [下一章 →](b-plus-classical-ai.md)

---

## 知识单元

### B01 `[核]` Python、NumPy、广播与向量化

**先修：** A02。

**定义与解析：** 向量化（vectorization）把逐元素 Python 循环改写成数组运算，让底层编译代码批量执行。广播（broadcasting）从尾轴对齐：维度相等或其中一个为 1 才兼容；它通常不复制数据，但产生的大型中间结果仍会消耗内存。

**公式/机制：** 若 `X∈R^{n×d}`、`μ∈R^d`，`X-μ` 将 `μ` 视作每行共享；批量两两差可写成 `X[:,None,:]-C[None,:,:]`，输出形状为 `(n,k,d)`。

**资料定位：** NumPy 用户指南，[Broadcasting—General broadcasting rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules)；[NumPy quickstart—Less basic operations](https://numpy.org/doc/stable/user/quickstart.html#less-basic-operations)。

```python
import numpy as np
X = np.array([[1., 2.], [3., 4.], [5., 8.]])
C = np.array([[0., 0.], [4., 4.]])
diff = X[:, None, :] - C[None, :, :]
dist2 = (diff ** 2).sum(axis=-1)
nearest = dist2.argmin(axis=1)
loop = np.array([[((x-c)**2).sum() for c in C] for x in X])
print(dist2, nearest, np.allclose(dist2, loop))
```

**检测题/小实验：** 只用广播实现 100 个点与 5 个中心的距离矩阵；估算 `100000×1000×128` 中间数组的内存并提出分块方案。

**常见坑：** 轴对齐错误但结果仍可运行；用 `tile` 制造无谓副本；为追求一行代码生成超大中间张量；忘记整数数组除法与 dtype。

### B02 `[核]` 数据结构、算法复杂度与内存复杂度

**先修：** A01。

**定义与解析：** 时间复杂度描述输入规模增长时操作次数的量级；空间复杂度描述额外内存。数组支持常数时间索引，哈希表平均常数时间查找，堆适合动态取最值，图结构表示依赖关系。复杂度必须与常数、缓存和实际规模共同判断。

**公式/机制：** Big-O 给渐近上界；顺序扫描 `O(n)`，排序 `O(n log n)`，稠密矩阵乘法朴素约 `O(n³)`。训练还应估算参数、激活、梯度和优化器状态占用。

**资料定位：** Python 教程，[5. Data Structures：Lists、Sets、Dictionaries](https://docs.python.org/3/tutorial/datastructures.html)；Python Wiki，[TimeComplexity](https://wiki.python.org/moin/TimeComplexity)。

```python
import timeit
n = 20_000
items = list(range(n))
lookup = set(items)
t_list = timeit.timeit(lambda: n-1 in items, number=200)
t_set = timeit.timeit(lambda: n-1 in lookup, number=200)
print("list/set seconds", t_list, t_set)
top5 = sorted(items, reverse=True)[:5]
print("top5", top5)
```

**检测题/小实验：** 为 kNN 推理、全量 self-attention、mini-batch SGD 分别写时间/空间复杂度；用 `heapq.nlargest` 替换完整排序并比较。

**常见坑：** 只看 Big-O 不看数据布局；忽略中间张量；把哈希查找的平均复杂度当绝对保证；过早微优化而不先 profile。

### B03 `[核]` Git、环境、依赖、测试与调试

**先修：** B01。

**定义与解析：** 版本控制记录可审计的变更历史；隔离环境固定解释器与依赖边界；测试把预期行为变为可重复检查。最小可复现实验应含源码、环境声明、配置、随机种子、运行命令和产出，而不是只保存 notebook 输出。

**公式/机制：** 一次提交应代表一个可解释变更；单元测试遵循 arrange–act–assert；失败调试优先缩小到最小复现，再检查输入、形状、范围、梯度和状态。

**资料定位：** *Pro Git*，[2.2 Recording Changes to the Repository](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository)；pytest，[Get Started—Create your first test](https://docs.pytest.org/en/stable/getting-started.html#create-your-first-test)。

```python
def standardize(x):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / var**0.5 for v in x]

z = standardize([1., 2., 3.])
assert abs(sum(z)) < 1e-12
assert abs(sum(v*v for v in z)/len(z) - 1) < 1e-12
print("tests passed", z)
```

**检测题/小实验：** 建一个只含函数和测试的小仓库；在新环境按 README 从零运行。故意制造 off-by-one、NaN 和形状错误，为每个错误添加回归测试。

**常见坑：** 提交数据、密钥或巨大模型；只写“能运行”的测试；环境文件不锁关键版本；调试时同时改多个变量。

### B04 `[核]` 张量、自动微分、GPU 与计算图工具

**先修：** A07、B01。

**定义与解析：** 张量库把数组运算、设备和自动微分统一起来。叶张量设置 `requires_grad=True` 后，运算构成动态计算图；对标量调用 `backward()` 反向累积梯度。设备迁移必须让参数和输入位于同一设备，本节示例刻意保持 CPU 友好。

**公式/机制：** 反向传播递归应用链式法则；默认计算 vector–Jacobian product。梯度存于 `.grad` 且会累加，因此每轮优化前需清零。

**资料定位：** PyTorch 基础教程，[Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)；[Automatic Differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

```python
import torch
torch.manual_seed(0)
X = torch.randn(8, 3)
w = torch.randn(3, 1, requires_grad=True)
y = X @ w
loss = y.square().mean()
loss.backward()
print("loss", loss.item(), "grad shape", w.grad.shape)
with torch.no_grad():
    w -= 0.1 * w.grad
w.grad = None
```

**检测题/小实验：** 连续调用两次 `backward()` 观察梯度累加；分别用 autograd 和有限差分检查一个小网络参数梯度。

**常见坑：** 在计算图中误用 `.detach()` 或 `.item()`；参数和输入设备不同；原地运算破坏反传；把张量形状正确误当语义正确。

### B05 `[核]` 数据清洗、预处理、划分与数据泄漏

**先修：** A13、B01。

**定义与解析：** 训练集拟合参数，验证集选择模型，测试集只做最终一次无偏估计。任何使用验证/测试标签或整体统计量的预处理都会泄漏（data leakage）。时间、用户、设备或群组相关数据必须按真实部署边界划分。

**公式/机制：** 标准化参数 `μ_train,σ_train` 只能从训练集估计，再同样作用于其他划分。Pipeline 将 `fit` 的边界绑定到交叉验证折内，降低泄漏风险。

**资料定位：** scikit-learn，[12.2 Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)；[8.1 Pipeline and composite estimators](https://scikit-learn.org/stable/modules/compose.html#pipeline)。

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=300, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
model.fit(Xtr, ytr)
print("held-out accuracy", model.score(Xte, yte))
```

**检测题/小实验：** 构造一个“先全数据标准化再交叉验证”的错误流程，与 Pipeline 比较；为同一用户多条记录改用 GroupKFold。

**常见坑：** 特征选择在划分前完成；测试集被反复查看；时间序列随机打乱；重复样本跨集合；用预测时不可获得的未来变量。

### B06 `[核]` 指标、基线、受控实验、复现与误差分析

**先修：** A14、B03、B05。

**定义与解析：** 指标必须匹配任务成本与类别分布。基线界定问题难度；受控实验一次改变一个因素；复现要求保存数据版本、种子和配置。总体分数之后还要按类别、群体和失败模式切片。

**公式/机制：** `Precision=TP/(TP+FP)`，`Recall=TP/(TP+FN)`，`F1=2PR/(P+R)`；ROC-AUC 衡量排序，PR-AUC 在稀有正例下更直观；校准评价概率而非硬分类。

**资料定位：** scikit-learn，[3.4 Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)；PyTorch，[Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
y = np.array([0]*95 + [1]*5)
pred_majority = np.zeros_like(y)
pred_candidate = np.array([0]*93 + [1]*2 + [0, 1, 1, 1, 1])
for name, pred in [("majority", pred_majority), ("candidate", pred_candidate)]:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    print(name, "accuracy", accuracy_score(y,pred), "P/R/F1", p, r, f)
```

**检测题/小实验：** 为癌症筛查、垃圾邮件和图像检索分别选主指标并说明错误成本；对五个随机种子报告均值、标准差和每次结果。

**常见坑：** 只报 accuracy；从多个指标中挑最好看的；没有朴素基线；单随机种子下宣称提升；把测试集误差分析变成继续调参。
