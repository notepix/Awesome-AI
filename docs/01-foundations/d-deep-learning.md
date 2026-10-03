# D. 深度学习共同主干


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](c-plus-probabilistic-black-box.md) · [下一章 →](../02-perception-language/e-computer-vision.md)

---

## 配套系统讲解

[串联本章的推导、例子与练习解析](../08-walkthroughs/02-training-experiments.md) · [论文与源码精读](../../readings/README.md) · [完整实践代码](../../labs/README.md)

原有知识单元保留稳定编号；概念卡用于定位，系统讲解用于连接完整过程。

## 知识单元

<a id="d01"></a>

### D01 `[核]` 感知机、神经元、MLP 与激活函数

**先修：** B04、C02、C03。

**定义与解析：** 神经元先做仿射变换再过非线性；多层感知机（MLP）通过层级复合学习非线性函数。没有激活时，多层线性层仍等价于一层。ReLU 计算简单且缓解饱和，sigmoid/tanh 常用于概率输出或门控。

**公式/机制：** `h^{(l)}=φ(W^{(l)}h^{(l-1)}+b^{(l)})`，`ŷ=g(W^{(L)}h^{(L-1)}+b)`；宽度提供并行特征，深度提供函数复合。

**资料定位：** *Deep Learning*，[第 6 章 Deep Feedforward Networks，6.1 Example: Learning XOR；6.3 Hidden Units](https://www.deeplearningbook.org/contents/mlp.html)。

```python
import torch
from torch import nn
torch.manual_seed(0)
X = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
logits = model(X)
print(model)
print("input/hidden-output shapes", X.shape, logits.shape)
print("probabilities", logits.sigmoid().detach().squeeze())
```

**检测题/小实验：** 证明两层无激活线性网络可合并；训练 XOR，分别替换 ReLU、tanh、sigmoid 并比较收敛与隐藏表示。

**常见坑：** 把“神经元像生物神经元”当机制解释；分类输出层重复 sigmoid；张量 batch/feature 轴弄反；盲目加深而无基线。

<a id="d02"></a>

### D02 `[核]` 计算图、反向传播与自动微分

<!-- readings:start -->
**进一步精读：** [ResNet：残差学习与深层优化](../../readings/papers/resnet.md)
<!-- readings:end -->

**先修：** A07、B04、D01。

**定义与解析：** 计算图把复杂函数拆为可微基本操作；反向传播（backpropagation）从标量损失逆拓扑遍历，复用中间梯度。自动微分是对程序应用链式法则，不是符号化简，也不是有限差分。

**公式/机制：** 若 `u=f(x), v=g(u), L=h(v)`，则 `∂L/∂x=(∂u/∂x)^T(∂v/∂u)^T∂L/∂v`。反向模式时间通常与一次前向同阶，但需保存中间激活。

**资料定位：** *Deep Learning*，[6.5 Back-Propagation and Other Differentiation Algorithms](https://www.deeplearningbook.org/contents/mlp.html)；PyTorch，[Automatic Differentiation with `torch.autograd`](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

```python
import torch
x = torch.tensor(2., requires_grad=True)
a = x * x
b = torch.sin(a)
loss = b + 0.1 * a
loss.backward()
manual = 2*x.detach()*torch.cos(x.detach()**2) + 0.2*x.detach()
print("autograd/manual", x.grad.item(), manual.item())
x.grad = None
```

**检测题/小实验：** 画出 `L=(wx+b-y)²` 的计算图并手算每条边的局部导数；用 central difference 做参数级 gradient check。

**常见坑：** 每步未清梯度；跨 iteration 保留图导致内存增长；不可微点误认为无法训练；对非标量调用 backward 不给上游向量。

<a id="d03"></a>

### D03 `[核]` 输出分布、任务损失与复合目标

**先修：** A13、A15、D01。

**定义与解析：** 输出层和损失共同声明条件分布：Gaussian 均值配 MSE，Bernoulli logit 配 binary cross-entropy，Categorical logits 配 cross-entropy。损失是训练代理目标；多任务/复合目标还需处理量纲、权重和冲突梯度。

**公式/机制：** NLL `L=-Σ_i log p_θ(y_i|x_i)`；MSE 对应固定方差 Gaussian；交叉熵直接接 logits 可用 log-sum-exp 保持稳定；复合损失 `L=Σ_jλ_jL_j`。

**资料定位：** *Deep Learning*，[6.2 Gradient-Based Learning，Cost Functions](https://www.deeplearningbook.org/contents/mlp.html)；PyTorch，[Loss functions](https://docs.pytorch.org/docs/stable/nn.html#loss-functions)。

```python
import torch
from torch.nn import functional as F
logits = torch.tensor([[2., .1, -1.], [.2, .3, 1.2]], requires_grad=True)
target = torch.tensor([0, 2])
ce = F.cross_entropy(logits, target)
probs = logits.softmax(-1)
manual = -torch.log(probs[torch.arange(2), target]).mean()
aux = 0.01 * logits.square().mean()
(ce + aux).backward()
print(ce.item(), manual.item(), logits.grad)
```

**检测题/小实验：** 为多标签分类、单标签多类、回归带异方差分别选择输出与损失；改变复合权重并比较各项梯度范数。

**常见坑：** CrossEntropyLoss 前先 softmax；类别索引与 one-hot 接口混淆；损失下降就代表业务指标改善；复合项尺度差几个数量级。

<a id="d04"></a>

### D04 `[核]` 参数初始化、信号传播与梯度稳定性

**先修：** A05、D02。

**定义与解析：** 初始化要打破神经元对称性，并让前向激活与反向梯度在深度上传播时不过度放大或衰减。Xavier 适配近线性/tanh，He/Kaiming 适配 ReLU；具体增益取决于激活和 fan-in/fan-out。

**公式/机制：** Xavier 常取 `Var(W)=2/(fan_in+fan_out)`；ReLU 的 Kaiming 初始化常取 `Var(W)=2/fan_in`。饱和激活、过大谱半径与不当偏置都会破坏传播。

**资料定位：** *Deep Learning*，[8.4 Parameter Initialization Strategies](https://www.deeplearningbook.org/contents/optimization.html)；PyTorch，[`torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init.html)。

```python
import torch
from torch import nn
torch.manual_seed(0)
model = nn.Sequential(*sum(([nn.Linear(128,128), nn.ReLU()] for _ in range(8)), []))
for m in model:
    if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)
x = torch.randn(256, 128)
for i, layer in enumerate(model):
    x = layer(x)
    if isinstance(layer, nn.ReLU): print(i, round(x.std().item(), 3))
```

**检测题/小实验：** 对 20 层 ReLU MLP 比较全零、标准正态、Xavier、Kaiming 初始化的激活/梯度标准差。

**常见坑：** 所有权重初始化为零；忽略激活函数选择 gain；只看参数分布不看逐层激活与梯度；把随机种子当初始化策略。

<a id="d05"></a>

### D05 `[核]` SGD、Momentum、Adam 与学习率调度

<!-- readings:start -->
**进一步精读：** [Adam：自适应梯度与偏差校正](../../readings/papers/adam.md)
<!-- readings:end -->

**先修：** A17、A18、D02。

**定义与解析：** SGD 用 mini-batch 梯度更新；Momentum 累积低频一致方向、抑制高频震荡；Adam 按一、二阶矩自适应缩放。学习率通常是最重要超参数，调度器定义训练过程中步长变化。

**公式/机制：** Momentum：`v_t=βv_{t-1}+g_t, θ←θ-ηv_t`；Adam：对 `m_t,v_t` 做偏差校正后 `θ←θ-ηm̂/(√v̂+ε)`。AdamW 将 weight decay 与损失梯度解耦。

**资料定位：** *Deep Learning*，[8.3 Basic Algorithms；8.8 Strategies for Choosing the Learning Rate](https://www.deeplearningbook.org/contents/optimization.html)；PyTorch，[Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

```python
import torch
p = torch.tensor([4., -3.], requires_grad=True)
opt = torch.optim.AdamW([p], lr=.2, weight_decay=.01)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=.5)
for step in range(30):
    loss = ((p - torch.tensor([1., 2.]))**2).sum()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step in [0, 9, 19, 29]:
        print(step, loss.item(), p.detach().tolist(), sched.get_last_lr()[0])
```

**检测题/小实验：** 在条件数很大的二次碗上画 SGD、Momentum、Adam 轨迹；做 learning-rate range test，再比较 constant/cosine schedule。

**常见坑：** 同时改优化器和学习率却归因于优化器；scheduler 调用时机错误；忘记 `zero_grad`；把 Adam 的 L2 penalty 当 AdamW。

<a id="d06"></a>

### D06 `[核]` Weight Decay、Dropout、早停与数据增强

**先修：** A18、D03、D05。

**定义与解析：** 正则化旨在降低泛化误差。Weight decay 收缩参数；Dropout 训练时随机屏蔽单元；早停用验证性能限制有效训练时长；数据增强编码任务不变性并扩大观测支持。它们改变机制不同，不应无脑叠加。

**公式/机制：** L2 目标 `L+λ||w||²/2`；Dropout `h̃=m⊙h/(1-p), m∼Bernoulli(1-p)`，保持期望；早停保存最佳验证检查点而非最后一步。

**资料定位：** *Deep Learning*，[7.1 Parameter Norm Penalties；7.8 Early Stopping；7.12 Dropout](https://www.deeplearningbook.org/contents/regularization.html)。

```python
import torch
from torch import nn
torch.manual_seed(0)
drop = nn.Dropout(p=.5)
x = torch.ones(1000)
drop.train(); samples = torch.stack([drop(x) for _ in range(50)])
print("train mean/zero fraction", samples.mean().item(), (samples==0).float().mean().item())
drop.eval(); y = drop(x)
print("eval unchanged", torch.equal(x, y))
w = torch.tensor([3., 4.])
print("L2 penalty", 0.5 * w.square().sum())
```

**检测题/小实验：** 在小数据 MLP 上分别消融 weight decay、dropout、早停；记录训练误差、验证误差和最佳 epoch。

**常见坑：** 验证/推理时忘记 `eval()`；增强破坏标签语义；用测试集早停；把更多正则化默认当更好。

<a id="d07"></a>

### D07 `[核]` BatchNorm、LayerNorm 与 RMSNorm

<!-- readings:start -->
**进一步精读：** [LLaMA：现代自回归模型的设计](../../readings/papers/llama.md)
<!-- readings:end -->

**先修：** A11、D02、D04。

**定义与解析：** 归一化稳定特征尺度并改善优化。BatchNorm 按 batch 与空间轴统计每通道均值方差，训练和推理行为不同；LayerNorm 对每个样本的特征轴归一化；RMSNorm 只按均方根缩放、不减均值，常用于 Transformer。

**公式/机制：** `x̂=(x-μ)/√(σ²+ε), y=γx̂+β`；BN 推理使用运行统计，LN 使用当前样本统计；残差网络中归一化位置会改变优化性质。

**资料定位：** 原始 BatchNorm 论文，[Ioffe & Szegedy 2015](https://proceedings.mlr.press/v37/ioffe15.html)；PyTorch，[`BatchNorm1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) 与 [`LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)。

```python
import torch
from torch import nn
torch.manual_seed(0)
x = torch.randn(4, 6) * 3 + 5
bn, ln = nn.BatchNorm1d(6), nn.LayerNorm(6)
y_bn, y_ln = bn(x), ln(x)
print("BN feature means", y_bn.mean(0).round(decimals=4))
print("LN sample means", y_ln.mean(1).round(decimals=4))
bn.eval()
print("BN eval uses running stats", bn(x).mean().item())
```

**检测题/小实验：** 将 batch size 从 64 降到 2，观察 BN 稳定性；手写 LN 并与 PyTorch 结果核对轴和 `eps`。

**常见坑：** 混淆归一化轴；BN 推理仍处于 train；认为归一化可替代输入预处理；忽略小 batch 和分布漂移。

<a id="d08"></a>

### D08 `[核]` 训练循环、微型过拟合、检查点与系统调试

<!-- readings:start -->
**进一步精读：** [nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md)
<!-- readings:end -->

**先修：** B06、D02–D07。

**定义与解析：** 可靠训练循环明确 train/eval 模式、数据、损失、反传、更新、日志和检查点边界。微型过拟合（overfit a tiny batch）是首要冒烟测试：模型若不能记住几十个样本，应先查实现而非加算力。

**公式/机制：** 每步顺序通常为 `zero_grad → forward → loss → backward → optional clip → step`；验证阶段使用 `eval()` 与 `no_grad()`；最佳检查点由预先指定验证指标选择。

**资料定位：** PyTorch，[Quickstart—Optimizing the Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html#optimizing-the-model-parameters)；*Deep Learning*，[第 11 章 Practical Methodology，11.2–11.5](https://www.deeplearningbook.org/contents/guidelines.html)。

```python
import torch
from torch import nn
torch.manual_seed(1)
X = torch.randn(16, 4); y = (X[:, 0] > 0).long()
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))
opt = torch.optim.Adam(model.parameters(), lr=.05)
for step in range(120):
    model.train(); logits = model(X); loss = nn.functional.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad(): acc = (model(X).argmax(1)==y).float().mean()
print("tiny-batch loss/accuracy", loss.item(), acc.item())
```

**检测题/小实验：** 故意打乱标签对齐、冻结参数或删除 `zero_grad`，根据曲线定位；实现保存/恢复后下一步损失一致的检查点测试。

**常见坑：** 验证时仍启用 dropout/BN 更新；平均 batch loss 未按样本数加权；只保存权重不保存优化器和配置；训练失败先调大模型。

<a id="d09"></a>

### D09 `[核]` Embedding、表征学习与度量学习

<!-- readings:start -->
**进一步精读：** [DPR：稠密段落检索](../../readings/papers/dpr.md) · [CLIP：图文对比对齐](../../readings/papers/clip.md)
<!-- readings:end -->

**先修：** A04、D01、D03。

**定义与解析：** 表征学习把原始对象映射到可供下游任务使用的向量。Embedding 是可学习查表或编码器输出；度量学习让相似样本更近、不同样本更远。表示好坏必须由目标任务和不变性定义，二维可视化只是诊断。

**公式/机制：** 查表 `e_i=E[i]`；triplet loss `max(0,d(a,p)-d(a,n)+m)`；余弦相似度常配 L2 归一化。负样本选择决定训练信号难度，也可能引入假负例。

**资料定位：** *Deep Learning*，[第 15 章 Representation Learning，15.1–15.4](https://www.deeplearningbook.org/contents/representation.html)；PyTorch，[`TripletMarginLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html)。

```python
import torch
from torch import nn
torch.manual_seed(0)
table = nn.Embedding(6, 3)
anchor = table(torch.tensor([0, 1]))
positive = table(torch.tensor([0, 1])) + .05
negative = table(torch.tensor([4, 5]))
loss = nn.TripletMarginLoss(margin=.5)(anchor, positive, negative)
loss.backward()
active_rows = (table.weight.grad.norm(dim=1) > 0).nonzero().squeeze()
print("loss", loss.item(), "gradient rows", active_rows)
```

**检测题/小实验：** 训练一个小型类别 embedding，比较随机负例与 hard negative；用 Recall@k 而非仅 t-SNE 图评价检索。

**常见坑：** 把向量距离天然解释为语义；未归一化却比较余弦/点积；负样本实际同类；只看漂亮降维图。

<a id="d10"></a>

### D10 `[核]` 残差、门控、递归与状态传递结构

<!-- readings:start -->
**进一步精读：** [ResNet：残差学习与深层优化](../../readings/papers/resnet.md) · [Transformer：注意力与编解码器](../../readings/papers/transformer.md)
<!-- readings:end -->

**先修：** D01、D02、D04。

**定义与解析：** 残差连接学习输入上的修正，使深层网络保留恒等路径；门控用数据依赖系数控制信息保留与更新；递归网络在时间上共享参数并传递状态。三者都在缓解深长计算路径中的信息与梯度损失。

**公式/机制：** Residual：`y=x+F(x)`；门控：`h'=g⊙candidate+(1-g)⊙h`；RNN：`h_t=φ(W_xx_t+W_hh_{t-1})`。LSTM/GRU 用加性状态路径与门控制长期依赖。

**资料定位：** 原始 ResNet 论文，[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)；*Deep Learning*，[第 10 章 Sequence Modeling: Recurrent and Recursive Nets，10.2、10.10](https://www.deeplearningbook.org/contents/rnn.html)。

```python
import torch
from torch import nn
torch.manual_seed(0)
x = torch.randn(2, 5, 8)              # batch, time, feature
block = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
residual = x + block(x)
gru = nn.GRU(input_size=8, hidden_size=8, batch_first=True)
sequence, h_last = gru(residual)
gate = torch.sigmoid(torch.tensor([-3., 0., 3.]))
old, new = torch.ones(3), torch.zeros(3)
mixed = gate*new + (1-gate)*old
print(residual.shape, sequence.shape, h_last.shape, mixed)
```

**检测题/小实验：** 比较 20 层 MLP 有无残差时输入层梯度；手算一个标量 GRU 门值极端为 0/1 时的信息流。

**常见坑：** 残差两支形状不匹配；认为残差必然无损；RNN 隐状态未按序列边界重置；把门值当硬开关。

<a id="d11"></a>

### D11 `[核]` 迁移学习、微调、冻结策略与参数高效适配

<!-- readings:start -->
**进一步精读：** [BERT：双向预训练与迁移](../../readings/papers/bert.md) · [LoRA：低秩任务更新](../../readings/papers/lora.md) · [Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md) · [PEFT：参数高效适配的注入保存与合并](../../readings/projects/peft.md)
<!-- readings:end -->

**先修：** D06、D08、D09。

**定义与解析：** 迁移学习复用源任务学到的参数或表示。线性探针只训练新头；部分/全量微调逐步释放骨干；参数高效微调（PEFT）仅训练 adapter、prompt 或低秩增量。策略取决于数据量、域差、算力和遗忘风险。

**公式/机制：** LoRA 令 `W'=W+BA`，其中秩 `r≪min(d_in,d_out)`，冻结 `W` 仅训练 `A,B`；判别学习率常让头部学习率大于底层。

**资料定位：** PyTorch，[Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)；LoRA 原始论文，[Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。

```python
import torch
from torch import nn
backbone = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 8))
for p in backbone.parameters(): p.requires_grad = False
head = nn.Linear(8, 3)
model = nn.Sequential(backbone, head)
X = torch.randn(12, 10); y = torch.randint(0, 3, (12,))
loss = nn.functional.cross_entropy(model(X), y)
loss.backward()
print("backbone grad", backbone[0].weight.grad, "head grad", head.weight.grad.norm())
```

**检测题/小实验：** 在小数据上比较线性探针、最后一层解冻、全量微调；记录参数量、训练时间和验证性能，而非只比准确率。

**常见坑：** 冻结参数却让 BN 运行统计继续变化；新头未重置；源域差异很大仍盲目冻结；PEFT 参数少就误认为显存一定很少。

<a id="d12"></a>

### D12 `[核]` 自监督、对比学习与掩码建模

<!-- readings:start -->
**进一步精读：** [CLIP：图文对比对齐](../../readings/papers/clip.md)
<!-- readings:end -->

**先修：** A15、D09、D11。

**定义与解析：** 自监督学习从数据本身构造监督信号。对比学习拉近同一实例的两个视图、推远其他实例；掩码建模从上下文恢复被遮蔽内容。增强或遮蔽策略定义模型被鼓励忽略与保留的信息。

**公式/机制：** InfoNCE 对正对 `(i,i)` 最小化 `-log exp(sim(z_i,z'_i)/τ)/Σ_jexp(sim(z_i,z'_j)/τ)`；温度 `τ` 控制分布尖锐度。掩码目标只在选定位置计损失。

**资料定位：** SimCLR 原始论文，[A Simple Framework for Contrastive Learning of Visual Representations](https://proceedings.mlr.press/v119/chen20j.html)；MAE 原始论文，[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)。

```python
import torch
from torch.nn import functional as F
torch.manual_seed(0)
z1 = F.normalize(torch.randn(6, 8), dim=1)
z2 = F.normalize(z1 + .15*torch.randn(6, 8), dim=1)
temperature = .2
logits = z1 @ z2.T / temperature
labels = torch.arange(len(z1))
loss = F.cross_entropy(logits, labels)
retrieval = (logits.argmax(1) == labels).float().mean()
print("InfoNCE/retrieval@1", loss.item(), retrieval.item())
```

**检测题/小实验：** 扫描增强强度和温度，比较线性探针与最近邻性能；构造假负例并观察 InfoNCE 梯度方向。

**常见坑：** 增强破坏语义；batch 太小却无 memory bank；把预训练损失低等同迁移好；线性探针协议不一致。

<a id="d13"></a>

### D13 `[核]` Attention、Multi-Head Attention 与 Transformer 公共结构

<!-- readings:start -->
**进一步精读：** [Transformer：注意力与编解码器](../../readings/papers/transformer.md) · [LLaMA：现代自回归模型的设计](../../readings/papers/llama.md) · [nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md)
<!-- readings:end -->

**先修：** A15、D02、D09。

**定义与解析：** Attention 让查询（query）按与键（key）的匹配权重聚合值（value）；多头机制在多个投影子空间并行建模关系。Transformer 块通常由 attention、前馈网络、残差与 LayerNorm 构成，Mask 控制可见边。

**公式/机制：** `Attention(Q,K,V)=softmax(QK^T/√d_k)V`；缩放防止高维点积使 softmax 饱和；self-attention 的标准时间/显存随序列长度二次增长。

**资料定位：** 原始论文，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，第 3.2 节；PyTorch，[Transformer building blocks](https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html)。

```python
import math, torch
torch.manual_seed(0)
x = torch.randn(1, 4, 6)
Q, K, V = x, x, x
scores = Q @ K.transpose(-2, -1) / math.sqrt(x.size(-1))
causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal, float("-inf"))
weights = scores.softmax(-1)
out = weights @ V
print("weights", weights[0])
print("row sums/output shape", weights.sum(-1), out.shape)
```

**检测题/小实验：** 手算两 token、单头 attention；删除 `√d_k` 比较 logits/entropy；检查 causal mask 上三角权重是否严格为零。

**常见坑：** Mask 方向反了；padding 与 causal mask 混淆；softmax 轴错误；认为 attention 权重天然等于因果解释。

<a id="d14"></a>

### D14 `[核]` 表达能力、优化偏置、泛化与 Scaling Law

<!-- readings:start -->
**进一步精读：** [Chinchilla：计算最优的模型数据分配](../../readings/papers/chinchilla.md)
<!-- readings:end -->

**先修：** A18、D01、D05。

**定义与解析：** 通用逼近定理说明足够宽的网络可在一定条件下逼近连续函数，但不保证参数高效、可学或能泛化。过参数化网络的实际解还由初始化、优化路径和数据决定，这称为隐式偏置。Scaling Law 是特定分布和训练制度下损失随模型、数据、算力的经验规律。

**公式/机制：** 典型经验式 `L(N,D,C)≈L_∞+aN^{-α}+bD^{-β}`；容量增大可同时降低训练误差并在合适正则/数据下改善测试误差。外推前必须检查数据、token 和算力定义。

**资料定位：** *Deep Learning*，[6.4.1 Universal Approximation Properties and Depth](https://www.deeplearningbook.org/contents/mlp.html)；Kaplan 等，[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)。

```python
import torch
from torch import nn
torch.manual_seed(0)
x = torch.linspace(-1, 1, 64).unsqueeze(1); y = torch.sin(5*x)
for width in [2, 8, 32]:
    net = nn.Sequential(nn.Linear(1,width), nn.Tanh(), nn.Linear(width,1))
    opt = torch.optim.Adam(net.parameters(), lr=.03)
    for _ in range(250):
        loss = (net(x)-y).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    params = sum(p.numel() for p in net.parameters())
    print(width, params, loss.item())
```

**检测题/小实验：** 多随机种子比较宽度、参数量、训练/验证误差；在 log-log 坐标拟合经验幂律并检查残差，禁止用三个点宣称普适定律。

**常见坑：** 用存在性定理推断 SGD 必能找到解；将训练集记忆等同泛化；从单一模型族外推 scaling；忽略数据质量和计算预算。

<a id="d15"></a>

### D15 `[核]` 不确定性、校准、OOD 与对抗鲁棒性

**先修：** C03、D03、D08。

**定义与解析：** Aleatoric uncertainty 来自观测噪声，epistemic uncertainty 来自知识不足。校准要求“置信度约 0.8 的样本约 80% 正确”；OOD 检测识别训练分布外输入；对抗鲁棒性研究微小、刻意扰动下的行为。三者相关但不能互相替代。

**公式/机制：** ECE 将置信度分箱后计算 `Σ_b(n_b/n)|acc_b-conf_b|`；温度缩放用 `softmax(z/T)` 在校准集优化 `T`；FGSM `x'=x+ε sign(∇_xL)`。

**资料定位：** Guo 等，[On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)；Hendrycks & Gimpel，[A Baseline for Detecting Misclassified and Out-of-Distribution Examples](https://arxiv.org/abs/1610.02136)；Goodfellow 等，[Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)。

```python
import numpy as np
conf = np.array([.55,.62,.71,.78,.83,.88,.93,.97])
correct = np.array([1,0,1,0,1,1,0,1])
edges = np.linspace(0, 1, 6)
ece = 0.0
for lo, hi in zip(edges[:-1], edges[1:]):
    mask = (conf > lo) & (conf <= hi)
    if mask.any():
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
        print((lo,hi), "acc/conf", correct[mask].mean(), conf[mask].mean())
print("ECE", ece)
```

**检测题/小实验：** 制作 reliability diagram，并在独立校准集做温度缩放；比较干净、常见扰动、OOD、FGSM 四组性能。

**常见坑：** 以最大 softmax 概率当可靠不确定性；在测试集校准；只测一种攻击；把 OOD 检出率高误作模型在 OOD 上预测正确。

<a id="d16"></a>

### D16 `[选]` 混合精度、分布式训练、性能剖析、剪枝与量化

<!-- readings:start -->
**进一步精读：** [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md) · [vLLM：缓存调度与模型服务](../../readings/projects/vllm.md)
<!-- readings:end -->

**先修：** A16、B04、D08。

**定义与解析：** 系统优化的目标是在精度约束下减少时间、显存、通信或部署成本。混合精度让部分算子用低精度并对脆弱操作保留高精度；数据并行复制模型、切分 batch 并同步梯度；剪枝移除结构，量化减少位宽。应先 profile 再优化瓶颈。

**公式/机制：** 参数内存约 `参数量×每元素字节数`，训练还含梯度、激活和优化器状态；同步数据并行近似 `g=(1/R)Σ_rg_r`。理论 FLOPs 降低不保证墙钟时间下降。

**资料定位：** PyTorch，[Automatic Mixed Precision recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html)；[Getting Started with Distributed Data Parallel](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)。

```python
import torch
from torch import nn
model = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 10))
n = sum(p.numel() for p in model.parameters())
bytes_fp32 = sum(p.numel()*p.element_size() for p in model.parameters())
model16 = model.half()
bytes_fp16 = sum(p.numel()*p.element_size() for p in model16.parameters())
print("parameters", n)
print("fp32/fp16 MiB", bytes_fp32/2**20, bytes_fp16/2**20)
print("stored dtype", next(model16.parameters()).dtype)
```

**检测题/小实验：** 用 profiler 分解数据加载、前向、反向和优化器耗时；比较 FP32/混合精度的吞吐、峰值内存与数值误差；量化后同时测体积和延迟。

**常见坑：** 只报理论 FLOPs；把显存减半等同速度翻倍；不同硬件上比较吞吐却不说明环境；压缩后不重新评测分布切片。
