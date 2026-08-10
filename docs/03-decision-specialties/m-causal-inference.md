# M. 因果推断


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](l-time-series.md) · [下一章 →](n-recommendation-search-retrieval.md)

---

## 知识单元

### M01 结构因果模型、干预与反事实【稳定理论】
- **先修**：概率图、回归、条件概率。
- **定义与解析**：SCM 用结构方程和外生变量表达生成机制；观察 $P(Y\mid X=x)$ 与主动干预 $P(Y\mid\operatorname{do}(X=x))$ 通常不同，预测相关性不能自动回答政策问题。
- **公式/机制**：$X=f_X(\operatorname{PA}_X,U_X)$；$\operatorname{do}(X=x)$ 用常数 $x$ 替换 $X$ 的方程并切断其入边。
- **资料**：Pearl [Causal inference in statistics: An overview，§2–3](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf)；DoWhy [Model causal mechanisms/identify-estimate-refute](https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/index.html)。
- **最小代码（可执行观察与干预差异）**：
```python
import numpy as np
g=np.random.default_rng(0); n=20000; u=g.normal(size=n)
x=u+g.normal(size=n); y=2*x+3*u+g.normal(size=n)
obs=np.cov(x,y,bias=True)[0,1]/np.var(x)
u2=g.normal(size=n); y0=2*0+3*u2+g.normal(size=n); y1=2*1+3*u2+g.normal(size=n)
causal=(y1-y0).mean()
print(obs,causal); assert abs(causal-2)<.1 and obs>3
```
- **检测/实验**：画出 U→X、U→Y、X→Y 的 DAG；解释为什么更准的 (E[Y|X]) 仍可能给错干预结论。
- **常见坑**：把 `do` 当条件筛选；DAG 方向只由相关数据决定；没有领域假设却宣称识别反事实。

### M02 潜在结果、随机试验与 ATE【稳定理论】
- **先修**：M01、抽样、置信区间、假设检验。
- **定义与解析**：个体同时有 (Y(1),Y(0))，但只能观察其中一个；随机化使处理与潜在结果独立，从而差均值无偏估计 ATE。
- **公式/机制**：$\operatorname{ATE}=\mathbb E[Y(1)-Y(0)]$；一致性 $Y=TY(1)+(1-T)Y(0)$，随机化 $T\perp(Y(0),Y(1))$。
- **资料**：Hernán & Robins [What If Ch.1–2](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)；Rubin [1974，§2–3](https://doi.org/10.1037/h0037350)。
- **最小代码（可执行 RCT）**：
```python
import numpy as np
g=np.random.default_rng(1); n=4000; base=g.normal(size=n); tau=1+.2*base
t=g.integers(0,2,size=n); y=base+t*tau+g.normal(size=n)
ate=y[t==1].mean()-y[t==0].mean()
se=np.sqrt(y[t==1].var()/sum(t==1)+y[t==0].var()/sum(t==0))
print(ate,(ate-1.96*se,ate+1.96*se))
assert abs(ate-tau.mean())<3*se
```
- **检测/实验**：区分 ATE、ATT、个体效应；检查随机化前后协变量平衡但不要以“不显著”作为唯一判断。
- **常见坑**：观察不到个体反事实却汇报个体真实效应；随机分配后按处理依从性直接分组；多次窥视结果再停止试验。

### M03 混杂、后门准则与调整【稳定理论】
- **先修**：M01–M02、DAG、线性回归。
- **定义与解析**：混杂变量同时影响处理和结果，打开非因果后门路径；调整集需阻断所有后门路径且不能包含处理后变量或 collider。
- **公式/机制**：若 $Z$ 满足后门准则，$P(y\mid\operatorname{do}(x))=\sum_zP(y\mid x,z)P(z)$。
- **资料**：Hernán & Robins [What If Ch.7–8](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)；Pearl [overview §3.3 Back-door criterion](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf)。
- **最小代码（可执行回归调整）**：
```python
import numpy as np
g=np.random.default_rng(2); n=5000; z=g.normal(size=n)
x=1.5*z+g.normal(size=n); y=2*x+4*z+g.normal(size=n)
naive=np.linalg.lstsq(np.c_[np.ones(n),x],y,rcond=None)[0][1]
adj=np.linalg.lstsq(np.c_[np.ones(n),x,z],y,rcond=None)[0][1]
print(naive,adj)
assert abs(adj-2)<.1 and abs(naive-2)>1
```
- **检测/实验**：分别调整混杂、mediator、collider，模拟估计偏差；要求先画 DAG 再选特征。
- **常见坑**：“控制变量越多越好”；从结果发生后生成的特征做调整；仅凭相关系数识别混杂。

### M04 倾向得分、重加权与双重稳健【稳定方法】
- **先修**：M02–M03、逻辑回归、positivity。
- **定义与解析**：倾向得分 $e(X)=P(T=1\mid X)$ 将可观测混杂压缩为处理概率；IPW 构造伪总体，双重稳健估计结合处理与结果模型。
- **公式/机制**：$\widehat{\operatorname{ATE}}_{\mathrm{IPW}}=n^{-1}\sum_i\left[T_iY_i/e_i-(1-T_i)Y_i/(1-e_i)\right]$；需一致性、无未测混杂、positivity。
- **资料**：Rosenbaum & Rubin [1983，Theorem 1–3](https://doi.org/10.1093/biomet/70.1.41)；Hernán & Robins [What If Ch.12–13](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)。
- **最小代码（可执行 IPW，已知倾向用于教学）**：
```python
import numpy as np
g=np.random.default_rng(3); n=20000; x=g.normal(size=n)
e=1/(1+np.exp(-x)); t=g.random(n)<e
y=1.5*t+2*x+g.normal(size=n)
ec=np.clip(e,.02,.98)
ate=np.mean(t*y/ec-(~t)*y/(1-ec))
ess=(np.sum(np.where(t,1/ec,1/(1-ec)))**2)/np.sum(np.where(t,1/ec,1/(1-ec))**2)
print(ate,ess); assert abs(ate-1.5)<.15
```
- **检测/实验**：画 propensity 重叠与权重直方图；改变 clipping，报告偏差、方差和有效样本量。
- **常见坑**：倾向模型 AUC 越高越好；没有共同支持仍外推；只报加权后点估计、不报权重极值和 balance。

### M05 自然实验：工具变量与双重差分【稳定设计，假设强】
- **先修**：M01–M04、回归、面板数据。
- **定义与解析**：IV 用只通过处理影响结果的外生变量识别局部效应；DiD 用处理组和对照组前后变化差识别政策效应，核心是无处理时平行趋势。
- **公式/机制**：$\widehat\tau_{\mathrm{DID}}=(\bar Y_{T,\mathrm{post}}-\bar Y_{T,\mathrm{pre}})-(\bar Y_{C,\mathrm{post}}-\bar Y_{C,\mathrm{pre}})$；Wald IV 为 $\Delta Y/\Delta X$。
- **资料**：Angrist, Imbens & Rubin [IV assumptions §2](https://doi.org/10.1080/01621459.1996.10476902)；Card & Krueger [1994，§III，difference-in-differences](https://doi.org/10.2307/2118030)。
- **最小代码（可执行 DiD）**：
```python
import numpy as np
control=np.array([10.,12.])                 # pre, post: 共同趋势 +2
treated=np.array([11.,16.])                 # 共同趋势 +2，政策效应 +3
did=(treated[1]-treated[0])-(control[1]-control[0])
print(did); assert did==3
placebo_t=np.array([8.,10.,11.]); placebo_c=np.array([7.,9.,10.])
pre_did=(placebo_t[1]-placebo_t[0])-(placebo_c[1]-placebo_c[0])
assert pre_did==0
```
- **检测/实验**：至少画多期 pre-trend 并做伪政策时间；说明“pre-trend 不显著”不能证明平行趋势。
- **常见坑**：IV 直接效应违反排除限制；弱工具导致不稳定；DiD 在预期政策或组别构成变化时失效。
