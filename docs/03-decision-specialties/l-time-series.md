# L. 时间序列


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](k-speech-audio.md) · [下一章 →](m-causal-inference.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/08-domain-bridges.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="l01"></a>

### L01 时间索引、窗口化与无泄漏切分【稳定】
- **先修**：监督学习、时间戳、数组切片。
- **定义与解析**：时间序列样本有顺序依赖；滑窗将过去 (L) 步映射到未来 (H) 步，验证集必须位于训练期之后并给特征计算留出边界。
- **公式/机制**：$X_t=[y_{t-L+1},\dots,y_t]\mapsto Y_t=[y_{t+1},\dots,y_{t+H}]$；rolling-origin 逐步前推预测起点。
- **资料**：Hyndman & Athanasopoulos [FPP3 §5.10 Time series cross-validation](https://otexts.com/fpp3/tscv.html)；scikit-learn [`TimeSeriesSplit` 与 gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)。
- **最小代码（可执行）**：
```python
import numpy as np
y=np.arange(12,dtype=float); L,H=4,2
X=np.stack([y[i:i+L] for i in range(len(y)-L-H+1)])
Y=np.stack([y[i+L:i+L+H] for i in range(len(y)-L-H+1)])
cut=5; Xtr,Ytr=X[:cut],Y[:cut]; Xte,Yte=X[cut:],Y[cut:]
assert Ytr.max()<Yte.max()
print(Xtr.shape,Yte.shape)
```
- **检测/实验**：标准化器若用全序列拟合，泄漏了什么？加入 gap 后样本数如何变化？
- **常见坑**：随机打乱窗口；同一原始点同时出现在训练标签与测试输入；节假日特征使用事后信息。

<a id="l02"></a>

### L02 朴素、季节朴素与指数平滑基线【稳定】
- **先修**：L01、均值、趋势、季节性。
- **定义与解析**：朴素预测复制最后值，季节朴素复制上一周期同位置；简单基线常比未调好的深网更可信，也是 MASE 的尺度基准。
- **公式/机制**：$\hat y_{T+h}=y_T$ 或 $\hat y_{T+h}=y_{T+h-m(k+1)}$；简单指数平滑 $\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}$。
- **资料**：FPP3 [§3.1 Simple forecasting methods](https://otexts.com/fpp3/simple-methods.html)；[§8.1 Simple exponential smoothing](https://otexts.com/fpp3/ses.html)。
- **最小代码（可执行季节基线）**：
```python
import numpy as np
t=np.arange(48); y=10+np.sin(2*np.pi*t/12)+.05*t
train,test=y[:36],y[36:]; m=12
pred=train[-m:]
mae=np.mean(abs(test-pred))
mean_mae=np.mean(abs(test-train.mean()))
print(mae,mean_mae); assert mae<mean_mae
```
- **检测/实验**：对有趋势的季节序列比较 naive、seasonal-naive、drift；验收要求后续模型必须优于明确基线。
- **常见坑**：季节周期凭感觉设定；只在一个预测起点测试；把插值后的测试真值用于特征。

<a id="l03"></a>

### L03 自回归、平稳性与滚动预测【稳定】
- **先修**：L01–L02、线性回归、相关与残差。
- **定义与解析**：AR 模型用过去值线性预测当前值；平稳性要求统计规律不随时间漂移，差分可去趋势但也可能丢失长期信息。
- **公式/机制**：$\operatorname{AR}(p): y_t=c+\sum_{i=1}^p\phi_i y_{t-i}+\epsilon_t$；$\operatorname{AR}(1)$ 的平稳条件为 $|\phi|<1$。
- **资料**：FPP3 [§9.3 Autoregressive models](https://otexts.com/fpp3/AR.html)；[§9.1 Stationarity and differencing](https://otexts.com/fpp3/stationarity.html)。
- **最小代码（可执行 AR(1) 拟合）**：
```python
import numpy as np
g=np.random.default_rng(0); y=np.zeros(300)
for t in range(1,len(y)): y[t]=.8*y[t-1]+g.normal(scale=.3)
X=np.c_[np.ones(249),y[:249]]; target=y[1:250]
b=np.linalg.lstsq(X,target,rcond=None)[0]
pred=b[0]+b[1]*y[249:-1]
mae=np.mean(abs(pred-y[250:]))
print(b,mae); assert abs(b[1]-.8)<.15
```
- **检测/实验**：分别模拟 φ=.8 与 1.0，比较方差、ACF 和滚动误差；拟合残差仍有自相关说明什么？
- **常见坑**：在全序列选 $p$；递归多步预测不传播不确定性；非平稳数据上把高 $R^2$ 当有效预测。

<a id="l04"></a>

### L04 深度时序：RNN、TCN 与 Transformer【成熟组件，选型演进中】
- **先修**：DL、L01–L03、卷积/注意力。
- **定义与解析**：RNN 递归传状态，TCN 用因果空洞卷积扩感受野，Transformer 用注意力建长依赖；结构更复杂不保证胜过树模型或季节基线。
- **公式/机制**：TCN 第 $l$ 层 dilation $d_l$ 仅访问过去；自注意力必须用 causal mask 防止看到未来。
- **资料**：Bai et al. [TCN §3，Fig.1](https://arxiv.org/abs/1803.01271)；Lim et al. [Temporal Fusion Transformer §3](https://arxiv.org/abs/1912.09363)。
- **最小代码（可执行因果空洞卷积）**：
```python
import torch
torch.manual_seed(0); x=torch.randn(2,1,20)
conv=torch.nn.Conv1d(1,4,kernel_size=3,dilation=2,padding=4)
y=conv(x)[:,:,:x.shape[-1]]                 # 去掉右侧，保持因果长度
x2=x.clone(); x2[:,:,15:]+=100
y2=conv(x2)[:,:,:x.shape[-1]]
assert torch.allclose(y[:,:,:15],y2[:,:,:15])
print(y.shape)
```
- **检测/实验**：修改未来输入不应改变过去输出；比较相同参数量的 MLP/TCN，报告多起点误差而非单次 split。
- **常见坑**：对称 padding 泄漏未来；位置/时间特征错位；用测试集挑 lookback、层数和 early stopping。

<a id="l05"></a>

### L05 概率预测、区间校准与异常检测【稳定原则，模型演进中】
- **先修**：L01–L04、分位数、概率分布、校准。
- **定义与解析**：点预测不描述风险；概率预测输出分布或分位数。异常分数只有在阈值、告警成本和时间容忍窗口定义后才有业务含义。
- **公式/机制**：pinball loss 为 $L_q(y,\hat y)=\max\!\left(q(y-\hat y),(q-1)(y-\hat y)\right)$；区间覆盖率应接近名义覆盖率且宽度尽量小。
- **资料**：Salinas et al. [DeepAR §2.1–2.3](https://arxiv.org/abs/1704.04110)；FPP3 [§5.5 Distributional forecasts and intervals](https://otexts.com/fpp3/prediction-intervals.html)。
- **最小代码（可执行）**：
```python
import numpy as np
g=np.random.default_rng(0); y=g.normal(size=1000)
lo,hi=np.full(1000,-1.645),np.full(1000,1.645)
coverage=np.mean((y>=lo)&(y<=hi)); width=np.mean(hi-lo)
def pinball(y,p,q):
    e=y-p; return np.mean(np.maximum(q*e,(q-1)*e))
print(coverage,width,pinball(y,hi,.95))
assert .86<coverage<.94
```
- **检测/实验**：按季节/负载分组检查覆盖率；异常检测同时报告事件级 precision/recall、检测延迟和每日误报数。
- **常见坑**：只追求宽区间带来的高覆盖；用异常标签调阈值后仍在同批数据报告；点级指标惩罚持续事件方式不合理。
