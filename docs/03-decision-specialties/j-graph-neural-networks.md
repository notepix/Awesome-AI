# J. 图神经网络


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](i-reinforcement-learning.md) · [下一章 →](k-speech-audio.md)

---

## 知识单元

### J01 图表示与消息传递【稳定】
- **先修**：线性代数、MLP、图的邻接表/矩阵。
- **定义与解析**：GNN 通过邻居消息聚合更新节点表示；共享参数和置换等变性使同一规则适用于不同大小、不同编号的图。
- **公式/机制**：$m_v=\operatorname{AGG}_{u\in\mathcal N(v)}M(h_v,h_u,e_{uv})$，$h'_v=U(h_v,m_v)$。
- **资料**：Gilmer et al. [MPNN §2，Algorithm 1](https://proceedings.mlr.press/v70/gilmer17a.html)；PyG [MessagePassing 基类说明](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html)。
- **最小代码（可执行，PyTorch）**：
```python
import torch
X=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
A=torch.tensor([[0.,1.,1.],[1.,0.,0.],[1.,0.,0.]])
deg=A.sum(1,keepdim=True).clamp_min(1)
msg=A@X/deg
H=torch.relu(X+msg)
print(H); assert H.shape==X.shape
```
- **检测/实验**：同时置换 A 的行列与 X 的行，验证输出按同一置换变化；只置换 X 为什么不成立？
- **常见坑**：混淆置换等变与不变；漏加自环；孤立节点除零。

### J02 图卷积网络 GCN【稳定】
- **先修**：J01、矩阵归一化、半监督分类。
- **定义与解析**：GCN 将自身与邻居特征按度归一后线性变换；谱图卷积的一阶近似可写成简单消息传递。
- **公式/机制**：$H^{(l+1)}=\sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right)$，$\tilde A=A+I$。
- **资料**：Kipf & Welling [ICLR 2017 §2.2，式 (2)](https://openreview.net/forum?id=SJU4ayYgl)；PyG [GCNConv 数学定义](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html)。
- **最小代码（可执行）**：
```python
import torch
torch.manual_seed(0); A=torch.tensor([[0.,1.,1.],[1.,0.,0.],[1.,0.,0.]])
At=A+torch.eye(3); d=At.sum(1); N=d.rsqrt()[:,None]*At*d.rsqrt()[None,:]
X=torch.randn(3,4); W=torch.randn(4,2)
H=torch.relu(N@X@W)
assert H.shape==(3,2)
assert torch.allclose(N,N.T)
```
- **检测/实验**：比较无归一、行归一、对称归一在星形图上的中心节点尺度。
- **常见坑**：稠密 A 在大图爆内存；训练/测试边泄漏；层数加深导致过平滑而非必然更强。

### J03 图注意力 GAT【稳定】
- **先修**：J01、softmax、自注意力。
- **定义与解析**：GAT 为每条邻边学习数据相关权重，仍需用邻接 mask 限制可见节点；权重可辅助诊断但不是因果解释。
- **公式/机制**：$\alpha_{ij}=\operatorname{softmax}_{j\in\mathcal N(i)}\operatorname{LeakyReLU}\!\left(a^T[Wh_i\Vert Wh_j]\right)$，$h'_i=\sigma\!\left(\sum_j\alpha_{ij}Wh_j\right)$。
- **资料**：Veličković et al. [ICLR 2018 §2，式 (1)–(4)](https://openreview.net/forum?id=rJXMpikCZ)。
- **最小代码（可执行的缩放点积变体）**：
```python
import torch, math
torch.manual_seed(0); X=torch.randn(4,3); A=torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]],dtype=torch.bool)
Q=X; K=X; V=X
score=Q@K.T/math.sqrt(X.shape[1])
score=score.masked_fill(~A,-torch.inf)
alpha=score.softmax(1); H=alpha@V
assert torch.allclose(alpha.sum(1),torch.ones(4))
```
- **检测/实验**：删除 mask 后输出代表图模型还是全连接注意力？检查孤立节点整行 `-inf` 的 NaN。
- **常见坑**：softmax 维度错误；注意力分数未对非边屏蔽；直接把高 α 解释为边的重要因果效应。

### J04 图级读出、批处理与不变性【稳定】
- **先修**：J01–J03、集合函数、分类评估。
- **定义与解析**：节点任务输出每节点表示；图任务需用 sum/mean/max 或可学习 readout 得到与节点顺序无关的图表示。
- **公式/机制**：$h_G=\rho\!\left(\sum_{v\in G}\phi(h_v)\right)$；sum 保留规模信息，mean 消除规模但可能混淆不同多重集。
- **资料**：Zaheer et al. [Deep Sets §2，Theorem 2](https://papers.neurips.cc/paper/6931-deep-sets)；PyG [mini-batches](https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html)。
- **最小代码（可执行）**：
```python
import torch
H=torch.tensor([[1.,0.],[0.,2.],[3.,0.],[1.,1.]])
batch=torch.tensor([0,0,1,1]); out=torch.zeros(2,2)
out.index_add_(0,batch,H)
count=torch.bincount(batch)[:,None]
mean=out/count
assert torch.allclose(mean,torch.tensor([[.5,1.],[2.,.5]]))
print(out,mean)
```
- **检测/实验**：构造均值相同、节点数不同的两张图，比较 sum 与 mean 是否可分。
- **常见坑**：批次图之间误连边；将 padding 节点纳入池化；随机按节点切分导致同图泄漏。

### J05 链接预测、负采样与归纳泛化【稳定基础】
- **先修**：J01–J04、二分类、采样偏差。
- **定义与解析**：链接预测对节点对打分；全体非边数量巨大，常用负采样近似，但采样分布决定训练目标与离线指标。
- **公式/机制**：内积解码 $p((u,v)\in E)=\sigma(z_u^Tz_v)$；BCE 同时训练正边与采样负边。
- **资料**：Kipf & Welling [Variational Graph Auto-Encoders §2](https://arxiv.org/abs/1611.07308)；Hamilton et al. [GraphSAGE §2.1](https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs)。
- **最小代码（可执行）**：
```python
import torch
torch.manual_seed(0); Z=torch.randn(5,3,requires_grad=True)
pos=torch.tensor([[0,1],[1,2],[3,4]]); neg=torch.tensor([[0,4],[1,4],[0,3]])
def score(e): return (Z[e[:,0]]*Z[e[:,1]]).sum(1)
logits=torch.cat([score(pos),score(neg)])
labels=torch.cat([torch.ones(3),torch.zeros(3)])
loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
loss.backward(); assert torch.isfinite(loss)
```
- **检测/实验**：随机边切分与按时间/节点归纳切分各测 AUC、Hits@K；为何随机负样本可能过于容易？
- **常见坑**：负样本其实是未观测正边；消息传递图含测试边；只报 ROC-AUC、不报候选集上的排名指标。
