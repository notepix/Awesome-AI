# C. 传统机器学习


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](b-plus-classical-ai.md) · [下一章 →](c-plus-probabilistic-black-box.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/01-probability-learning.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="c01"></a>

### C01 `[核]` 监督学习问题、假设空间与经验风险最小化

**先修：** A18、B05、B06。

**定义与解析：** 监督学习从输入—标签样本学习映射。任务由数据分布、假设空间（hypothesis class）、损失和评价协议共同定义；学习算法只是从假设空间选择模型的规则。ERM 最小化训练平均损失，泛化则关心同分布新样本上的总体风险。

**公式/机制：** `ĥ=argmin_{h∈H}(1/n)Σ_i ℓ(h(x_i),y_i)`；归纳偏置来自模型族、特征、正则和优化。分类输出分数、概率与决策三层对象，不应混为一谈。

**资料定位：** *Deep Learning*，[5.1 Learning Algorithms；5.2 Capacity, Overfitting and Underfitting](https://www.deeplearningbook.org/contents/ml.html)。

```python
import numpy as np
rng = np.random.default_rng(5)
X = rng.normal(size=(80, 1)); y = 2*X[:, 0] + rng.normal(scale=.5, size=80)
split = 60
X1 = np.c_[np.ones(len(X)), X]
w = np.linalg.lstsq(X1[:split], y[:split], rcond=None)[0]
mean_baseline = np.full(20, y[:split].mean())
pred = X1[split:] @ w
print("baseline/model MSE", np.mean((mean_baseline-y[split:])**2), np.mean((pred-y[split:])**2))
```

**检测题/小实验：** 为“预测明天是否流失”明确输入、标签时点、损失、主指标和部署分布；给出常数、规则和简单线性三层基线。

**常见坑：** 先选模型再定义问题；标签包含未来信息；把训练目标当业务指标；没有说明样本独立单位与部署分布。

<a id="c02"></a>

### C02 `[核]` 线性回归、最小二乘、Ridge 与 Lasso

**先修：** A05、A13、A17、C01。

**定义与解析：** 线性回归对特征的线性组合建模条件均值。最小二乘对应同方差 Gaussian 噪声下的 MLE；Ridge 用 L2 收缩相关特征，Lasso 用 L1 产生稀疏系数。线性指对参数线性，特征本身可经非线性变换。

**公式/机制：** OLS：`min_w ||Xw-y||²`；满秩时 `w=(X^TX)^{-1}X^Ty`，实际用 QR/SVD；Ridge：加 `λ||w||²`；Lasso：加 `λ||w||_1`。

**资料定位：** scikit-learn，[1.1.1 Ordinary Least Squares、1.1.2 Ridge、1.1.3 Lasso](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)。

```python
import numpy as np
rng = np.random.default_rng(6)
X = rng.normal(size=(40, 3)); X[:, 2] = X[:, 0] + .01*rng.normal(size=40)
y = X @ np.array([2., -1., 2.]) + rng.normal(scale=.3, size=40)
for lam in [0., .1, 10.]:
    A = X.T@X + lam*np.eye(X.shape[1])
    w = np.linalg.solve(A, X.T@y)
    mse = np.mean((X@w-y)**2)
    print(lam, "coef", np.round(w, 2), "mse", round(mse, 3))
```

**检测题/小实验：** 增强两列共线性并重复抽样，比较 OLS/Ridge 系数方差；说明为何不应通过显式矩阵逆求解。

**常见坑：** 系数相关就解释为因果；正则化前不缩放；对截距也无意施加惩罚；以训练 `R²` 作为唯一依据。

<a id="c03"></a>

### C03 `[核]` Logistic、Softmax、交叉熵与概率校准

**先修：** A15、A17、C01。

**定义与解析：** Logistic 回归把线性 log-odds 映射为二分类概率；Softmax 是多类推广。最小化负对数似然等价于交叉熵训练。排序、分类正确率与概率校准是不同性质：高准确率模型仍可能过度自信。

**公式/机制：** `p(y=1|x)=σ(w^Tx)`，`log[p/(1-p)]=w^Tx`；`p_k=e^{z_k}/Σ_je^{z_j}`；二分类梯度 `X^T(p-y)/n`。

**资料定位：** scikit-learn，[1.1.11 Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)；[1.16 Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)。

```python
import numpy as np
rng = np.random.default_rng(7)
X = rng.normal(size=(200, 2)); y = (X[:, 0]-X[:, 1] > 0).astype(float)
X = np.c_[np.ones(len(X)), X]; w = np.zeros(3)
for _ in range(300):
    z = np.clip(X @ w, -30, 30)
    p = 1 / (1 + np.exp(-z))
    w -= 0.3 * X.T @ (p-y) / len(y)
eps = 1e-9
loss = -np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))
print("w", w, "log-loss", loss, "accuracy", np.mean((p>.5)==y))
```

**检测题/小实验：** 把所有 logit 乘 5，比较 accuracy、log-loss 和可靠性图；推导 softmax 交叉熵对 logit 的梯度 `p-y_onehot`。

**常见坑：** 对 softmax 前先取整；用不稳定的 `exp`；把 0.5 当所有成本场景的最佳阈值；在校准集上再报告最终性能。

<a id="c04"></a>

### C04 `[核]` 特征缩放、缺失值、类别编码与特征工程

**先修：** B05、C01。

**定义与解析：** 预处理把原始字段变成模型可用表示。缩放影响距离、正则与梯度；缺失本身可能有信息但其机制需分析；类别变量通常用 one-hot、ordinal 或目标编码。所有“学习型”变换必须仅在训练折拟合。

**公式/机制：** 标准化 `z=(x-μ_train)/σ_train`；one-hot 不暗示类别顺序；缺失机制常区分 MCAR、MAR、MNAR。交互项与基函数能让线性模型表示非线性关系。

**资料定位：** scikit-learn，[8.3 Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)；[Column Transformer with Heterogeneous Data Sources](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html)。

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
X = np.array([[20., "A"], [np.nan, "B"], [40., "A"]], dtype=object)
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(), StandardScaler()), [0]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), [1])])
Z = prep.fit_transform(X)
print(Z.toarray() if hasattr(Z, "toarray") else Z)
```

**检测题/小实验：** 为有序等级和无序城市分别选择编码；在交叉验证外、内进行均值填补，比较泄漏造成的差异。

**常见坑：** 用整数编码无序类别；全数据拟合 scaler/imputer；线上出现新类别即报错；机械填补而不保留缺失指示。

<a id="c05"></a>

### C05 `[核]` kNN、距离学习与原型方法

**先修：** A04、C04。

**定义与解析：** k 近邻（k-nearest neighbors）不显式拟合参数，而在预测时找距离最近的训练样本并投票/平均。小 `k` 方差大，大 `k` 偏差大；距离、缩放与维度决定“邻近”的含义。

**公式/机制：** 分类 `ŷ=mode{y_i:i∈N_k(x)}`；距离加权可用 `w_i=1/(d_i+ε)`。朴素查询约 `O(nd)`；高维中距离趋同，索引结构也会退化。

**资料定位：** scikit-learn，[1.6.1.1 Nearest Neighbors Classification](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)。

```python
import numpy as np
X = np.array([[0.,0.], [0.,1.], [3.,3.], [3.,4.]])
y = np.array([0, 0, 1, 1])
Q = np.array([[.2,.7], [2.8,3.5]])
dist2 = ((Q[:,None,:] - X[None,:,:])**2).sum(-1)
k = 3
idx = np.argpartition(dist2, k-1, axis=1)[:, :k]
pred = np.array([np.bincount(y[row]).argmax() for row in idx])
print(idx, pred)
```

**检测题/小实验：** 加入一个数值尺度大 100 倍的无关特征，比较缩放前后预测；画验证误差随 `k` 的曲线。

**常见坑：** 忽略缩放；用测试集选 `k`；偶数 `k` 的平票未定义；高维稀疏数据盲用欧氏距离。

<a id="c06"></a>

### C06 `[核]` Naive Bayes、LDA 与 QDA

**先修：** A11、A13、C01。

**定义与解析：** 三者都是生成式分类器，先建模 `p(x|y)p(y)` 再用 Bayes 规则分类。Naive Bayes 假设给定类别后特征独立；LDA 假设各类 Gaussian 且共享协方差，得到线性边界；QDA 允许各类协方差不同，得到二次边界。

**公式/机制：** `ŷ=argmax_k [log p(y=k)+log p(x|y=k)]`；Gaussian 判别项含 Mahalanobis 距离。共享 `Σ` 时二次项抵消；类独立 `Σ_k` 则不抵消。

**资料定位：** scikit-learn，[1.9 Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)；[1.2 LDA and QDA—Mathematical formulation](https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-the-lda-and-qda-classifiers)。

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
X, y = make_classification(n_samples=300, n_features=4, random_state=1)
models = [GaussianNB(), LinearDiscriminantAnalysis(),
          QuadraticDiscriminantAnalysis(reg_param=.05)]
for m in models:
    print(type(m).__name__, cross_val_score(m, X, y, cv=5).mean())
```

**检测题/小实验：** 模拟共享协方差和不同协方差两组数据，比较 LDA/QDA；检查小样本高维时 QDA 的协方差估计。

**常见坑：** 把“朴素独立”理解为边缘独立；忽略先验类概率；高维小样本协方差奇异；凭训练准确率选 LDA/QDA。

<a id="c07"></a>

### C07 `[核]` 决策树、划分准则、剪枝与可解释性

<!-- readings:start -->
**进一步精读：** [XGBoost：二阶树提升与系统设计](../../readings/papers/xgboost.md)
<!-- readings:end -->

**先修：** A15、C01。

**定义与解析：** 决策树递归选择特征阈值，使子节点标签更纯；叶节点输出均值或类别分布。树能表示非线性交互、无需缩放，但轴对齐切分不平滑且深树方差高。剪枝通过限制容量改善泛化。

**公式/机制：** Gini `1-Σ_k p_k²`，entropy `-Σ_kp_k log p_k`；选择最大杂质下降 `I(parent)-Σ_j(n_j/n)I(child_j)`。回归常最小化平方误差。

**资料定位：** scikit-learn，[1.10.7 Mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)；[1.10.9 Minimal Cost-Complexity Pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)。

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
for depth in [1, 2, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    print(depth, tree.score(Xtr,ytr), tree.score(Xte,yte), tree.get_n_leaves())
print(export_text(tree, max_depth=2))
```

**检测题/小实验：** 手算一个二分类候选切分前后的 Gini；画深度与训练/验证误差，尝试 `ccp_alpha` 剪枝。

**常见坑：** 将单棵深树的 feature importance 当因果解释；忽略类别不均衡；用测试集决定深度；认为树完全无需数据清洗。

<a id="c08"></a>

### C08 `[核]` Bagging、随机森林、Boosting 与 GBDT

<!-- readings:start -->
**进一步精读：** [XGBoost：二阶树提升与系统设计](../../readings/papers/xgboost.md)
<!-- readings:end -->

**先修：** A18、C07。

**定义与解析：** 集成学习组合多个弱或不稳定模型。Bagging 并行训练 bootstrap 样本上的模型以降方差；随机森林再随机采样特征以降低树间相关；Boosting 顺序拟合前一轮残差/负梯度，主要降偏差但更易受噪声影响。

**公式/机制：** Bagging `f̄(x)=B^{-1}Σ_b f_b(x)`；若单树方差 `σ²`、相关系数 `ρ`，平均方差近似 `ρσ²+(1-ρ)σ²/B`；Gradient Boosting 更新 `F_m=F_{m-1}+ηh_m`。

**资料定位：** scikit-learn，[1.11.2 Forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest)；[1.11.4 Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)。

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
X, y = make_moons(n_samples=400, noise=.3, random_state=0)
models = [DecisionTreeClassifier(random_state=0),
          RandomForestClassifier(n_estimators=100, random_state=0),
          HistGradientBoostingClassifier(random_state=0)]
for m in models:
    print(type(m).__name__, cross_val_score(m, X, y, cv=5).mean())
```

**检测题/小实验：** 改变森林树数、单树深度和 Boosting 学习率，记录均值与训练时间；比较 permutation importance 与 impurity importance。

**常见坑：** 认为更多树必然过拟合；把 Boosting 的学习率和树数分开调；使用有偏的 impurity importance 解释高基数特征。

<a id="c09"></a>

### C09 `[核]` 间隔、SVM 与核方法

**先修：** A04、A17、C01。

**定义与解析：** 支持向量机（SVM）寻找分隔两类且几何间隔最大的超平面；软间隔允许违例。核技巧只通过样本内积计算隐式高维特征，使线性间隔变成输入空间的非线性边界。

**公式/机制：** 软间隔原问题 `min_{w,b,ξ} ||w||²/2+CΣξ_i`，约束 `y_i(w^Tx_i+b)≥1-ξ_i`；hinge loss `max(0,1-yf(x))`；RBF 核 `exp(-γ||x-x'||²)`。

**资料定位：** scikit-learn，[1.4.7 Mathematical formulation](https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)；[1.4.6 Kernel functions](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)。

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
X, y = make_moons(n_samples=250, noise=.2, random_state=1)
for kernel in ["linear", "rbf"]:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1, gamma="scale"))
    model.fit(X, y)
    svc = model[-1]
    print(kernel, "train acc/support", model.score(X,y), svc.n_support_)
```

**检测题/小实验：** 在 moons 数据上网格搜索 `C,γ` 并画边界；说明 `C` 增大如何改变间隔违例，及为何必须在 Pipeline 内缩放。

**常见坑：** 将核理解为显式生成新样本；不缩放特征；大样本直接使用核 SVM；在训练集可分就认为泛化良好。

<a id="c10"></a>

### C10 `[核]` k-means、GMM、EM 与层次聚类

**先修：** A04、A13。

**定义与解析：** k-means 以平方距离最小化簇内离差，隐含球形、相近尺度簇假设。GMM 用多个 Gaussian 的加权和建模软簇；EM 在隐变量后验期望（E 步）与参数最大化（M 步）间迭代。层次聚类生成多尺度树状结构。

**公式/机制：** k-means 目标 `Σ_i||x_i-μ_{z_i}||²`；E 步责任度 `r_ik∝π_kN(x_i|μ_k,Σ_k)`；M 步用责任度加权更新参数。两者都易陷局部最优。

**资料定位：** scikit-learn，[2.3.2 K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)；[2.1 Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html)。

```python
import numpy as np
rng = np.random.default_rng(8)
X = np.r_[rng.normal([-2,0], .5, (60,2)), rng.normal([2,0], .7, (60,2))]
centers = X[rng.choice(len(X), 2, replace=False)]
for _ in range(20):
    label = ((X[:,None,:]-centers[None,:,:])**2).sum(-1).argmin(1)
    new = np.stack([X[label==k].mean(0) for k in range(2)])
    if np.allclose(new, centers): break
    centers = new
print("centers", centers, "inertia", ((X-centers[label])**2).sum())
```

**检测题/小实验：** 对非球形 moons、不同方差和不同密度数据比较 k-means、GMM 与层次聚类；重复不同初始化并报告目标分布。

**常见坑：** 把簇编号当有序标签；用轮廓系数机械决定真实类别数；忽略缩放与离群点；空簇和局部最优未处理。

<a id="c11"></a>

### C11 `[核]` PCA、降维、流形学习与可视化

**先修：** A05、C10。

**定义与解析：** PCA 寻找最大方差的正交方向，等价于中心化数据的最佳低秩线性重构。流形方法试图保留局部邻域或全局距离；用于可视化时，二维图是算法产生的投影，不能直接证明天然簇存在。

**公式/机制：** 中心化 `X_c=UΣV^T` 后，前 `r` 个主方向为 `V_r`，投影 `Z=X_cV_r`，解释方差比为 `σ_i²/Σ_jσ_j²`。PCA 对尺度和离群点敏感。

**资料定位：** scikit-learn，[2.5.1 PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)；[2.2 Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)。

```python
import numpy as np
rng = np.random.default_rng(9)
t = rng.normal(size=200)
X = np.c_[t, 2*t + rng.normal(scale=.2, size=200), rng.normal(scale=.1, size=200)]
Xc = X - X.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = Xc @ Vt[:2].T
Xhat = Z @ Vt[:2] + X.mean(0)
ratio = s**2 / (s**2).sum()
print("variance ratio", ratio, "reconstruction MSE", np.mean((X-Xhat)**2))
```

**检测题/小实验：** 比较未缩放与标准化后的 PCA；随维数画累计解释方差和重构误差；用不同随机种子运行 t-SNE/UMAP 比较稳定性。

**常见坑：** PCA 前忘记中心化；把主成分当原始特征因果；把 t-SNE 簇间距离作定量结论；在全数据拟合降维后交叉验证。

<a id="c12"></a>

### C12 `[核]` 交叉验证、调参、类别不均衡与阈值选择

**先修：** B06、C01–C03。

**定义与解析：** 交叉验证（CV）复用训练数据估计模型选择性能；分层、分组和时间顺序应匹配采样结构。超参数选择与最终评估必须隔离；类别不均衡下可调整损失、采样或阈值，但要依据真实成本和校准概率。

**公式/机制：** k 折分数为各验证折指标均值；嵌套 CV 的外层估计选择流程，内层调参。阈值决策最小化 `C_FP P(FP)+C_FN P(FN)`，不必固定 0.5。

**资料定位：** scikit-learn，[3.1 Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)；[3.3 Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)。

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
y = np.array([0,0,0,0,0,1,1,1])
p = np.array([.05,.1,.2,.35,.6,.3,.55,.8])
precision, recall, thresholds = precision_recall_curve(y, p)
f1 = 2*precision*recall / np.maximum(precision+recall, 1e-12)
i = f1[:-1].argmax()
threshold = thresholds[i]
print("chosen on validation", threshold, "P/R/F1", precision[i], recall[i], f1[i])
print("predictions", (p >= threshold).astype(int))
```

**检测题/小实验：** 对普通 KFold、StratifiedKFold、GroupKFold、TimeSeriesSplit 各举一个适用场景；实现嵌套 CV 比较乐观偏差。

**常见坑：** 在全数据调参后仍把 CV 均值当无偏测试结果；忽略群组重复；先过采样再划分；按测试集挑阈值。

<a id="c13"></a>

### C13 `[选]` 概率图模型、隐变量推断与 HMM

**先修：** A09–A13、C10。

**定义与解析：** 概率图模型用图表达联合分布的因子分解和条件独立。隐马尔可夫模型（HMM）假设离散隐状态满足一阶 Markov 性、观测在给定当前状态后条件独立；前向—后向用于边缘推断，Viterbi 求最可能状态路径，Baum–Welch 是 HMM 的 EM。

**公式/机制：** `p(z_{1:T},x_{1:T})=p(z_1)∏_{t=2}^Tp(z_t|z_{t-1})∏_{t=1}^Tp(x_t|z_t)`；前向量 `α_t(j)=p(x_{1:t},z_t=j)` 递推为发射概率乘转移加权和。

**资料定位：** *Deep Learning*，[第 16 章 Structured Probabilistic Models，16.1–16.4](https://www.deeplearningbook.org/contents/structured_prob.html)；hmmlearn，[Tutorial—Building HMM and generating samples](https://hmmlearn.readthedocs.io/en/stable/tutorial.html#building-hmm-and-generating-samples)。

```python
import numpy as np
pi = np.array([.6, .4])
A = np.array([[.7,.3], [.2,.8]])
B = np.array([[.9,.1], [.3,.7]])       # state × observation
obs = [0, 1, 1, 0]
alpha = pi * B[:, obs[0]]
for o in obs[1:]:
    alpha = (alpha @ A) * B[:, o]
likelihood = alpha.sum()
print("sequence likelihood", likelihood, "last-state posterior", alpha/likelihood)
```

**检测题/小实验：** 手算长度 3 序列的前向概率并与代码核对；加入每步归一化或 log-space，比较长序列下的下溢。

**常见坑：** 混淆“最可能路径”与每时刻边缘最可能状态；直接连乘导致下溢；状态编号被误作有语义标签；忽略模型不可辨识性。
