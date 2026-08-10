# N. 推荐、搜索与检索


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](m-causal-inference.md) · [下一章 →](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md)

---

## 知识单元

### N01 候选、排序与离线检索指标【稳定】
- **先修**：二分类指标、对数、集合运算。
- **定义与解析**：检索先从语料生成候选，再按相关性排序；Precision/Recall 衡量集合，MRR/NDCG 对名次敏感，指标必须绑定同一候选库和 relevance 定义。
- **公式/机制**：$\mathrm{DCG}@k=\sum_{i=1}^k\frac{2^{rel_i}-1}{\log_2(i+1)}$，$\mathrm{NDCG}=\mathrm{DCG}/\mathrm{IDCG}$，$\mathrm{MRR}=|Q|^{-1}\sum_q1/\operatorname{rank}_q$。
- **资料**：Stanford IR Book [Ch.8 Evaluation of ranked retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)，重点 §8.4–8.5。
- **最小代码（可执行 NDCG）**：
```python
import numpy as np
rel=np.array([3,0,2,1,0]); k=4
def dcg(x):
    x=np.asarray(x)[:k]
    return np.sum((2**x-1)/np.log2(np.arange(2,len(x)+2)))
ndcg=dcg(rel)/dcg(np.sort(rel)[::-1])
rr=1/(np.flatnonzero(rel>0)[0]+1)
print(ndcg,rr); assert 0<=ndcg<=1
```
- **检测/实验**：交换第 1 与第 10 名相关文档，比较 Recall@10 与 NDCG@10；说明何时二者结论不同。
- **常见坑**：不同候选库直接比指标；没有相关文档的 query 处理不统一；对二值/分级 relevance 混用 DCG。

### N02 隐式反馈、时间切分与曝光偏差【稳定问题，纠偏演进中】
- **先修**：N01、推荐系统用户—物品矩阵、缺失非随机。
- **定义与解析**：点击、停留和购买是正向行为但未曝光不等于不喜欢；时间留一切分更接近“用过去推荐未来”，候选负例需与线上曝光机制对齐。
- **公式/机制**：观察 $O_{ui}=1$ 才可能得到反馈；经验风险实际按 $P(O=1\mid u,i)$ 加权，导致热门与位置偏差。
- **资料**：TensorFlow Recommenders [Basic retrieval: data split/model/evaluation](https://www.tensorflow.org/recommenders/examples/basic_retrieval)；Schnabel et al. [Recommendations as Treatments，§2–3](https://proceedings.mlr.press/v48/schnabel16.html)。
- **最小代码（可执行时间留一）**：
```python
events=[('u1',1,'a'),('u1',3,'b'),('u2',2,'a'),('u2',5,'c'),('u1',6,'c')]
by_user={}
for u,t,i in sorted(events,key=lambda z:z[1]): by_user.setdefault(u,[]).append((t,i))
train=[]; test=[]
for u,seq in by_user.items():
    train += [(u,t,i) for t,i in seq[:-1]]
    test.append((u,*seq[-1]))
assert all(max([t for u2,t,i in train if u2==u],default=-1)<t for u,t,i in test)
print(train,test)
```
- **检测/实验**：随机切分与时间留一分别测热门基线和个性化模型；记录全量候选与采样 100 负例两种排名。
- **常见坑**：把未点击全当负例；同一会话拆到训练和测试；用测试期物品流行度构造训练特征。

### N03 协同过滤、矩阵分解与 BPR【稳定】
- **先修**：N02、embedding、SGD、正则化。
- **定义与解析**：矩阵分解以用户、物品向量内积表示偏好；BPR 不拟合绝对分数，而让已观察物品 (i) 排在未观察物品 (j) 前。
- **公式/机制**：$\hat y_{ui}=p_u^Tq_i$，$L=-\sum_{(u,i,j)}\log\sigma(\hat y_{ui}-\hat y_{uj})+\lambda\|\Theta\|_2^2$。
- **资料**：Rendle et al. [BPR §2.2–2.3 与 LearnBPR Algorithm 1](https://arxiv.org/abs/1205.2618)。
- **最小代码（可执行 BPR 更新）**：
```python
import torch
torch.manual_seed(0); U=torch.nn.Embedding(3,4); I=torch.nn.Embedding(6,4)
u=torch.tensor([0,1,2]); pos=torch.tensor([1,2,3]); neg=torch.tensor([5,4,0])
opt=torch.optim.Adam([*U.parameters(),*I.parameters()],.05)
for _ in range(50):
    diff=(U(u)*(I(pos)-I(neg))).sum(1)
    loss=-torch.nn.functional.logsigmoid(diff).mean()
    opt.zero_grad(); loss.backward(); opt.step()
assert diff.mean()>1
```
- **检测/实验**：比较均匀负采样与按流行度采样的 Recall@K、长尾覆盖；解释训练负例为何不是“真实不喜欢”。
- **常见坑**：测试正例参与负采样；只评 sampled ranking；冷启动用户/物品没有旁路特征。

### N04 双塔与稠密检索【稳定架构，训练技巧演进中】
- **先修**：embedding、对比学习、N01–N03。
- **定义与解析**：双塔分别编码 query/user 与 document/item，向量可离线建索引；交叉编码器更精细但不能低成本遍历全库。
- **公式/机制**：$s(q,d)=E_q(q)^TE_d(d)$，in-batch negatives 下用行方向交叉熵训练正确配对。
- **资料**：Karpukhin et al. [DPR §3，式 (1)](https://aclanthology.org/2020.emnlp-main.550/)；TensorFlow Recommenders [Retrieval task/candidate corpus](https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval)。
- **最小代码（可执行）**：
```python
import torch
torch.manual_seed(0); z=torch.randn(16,12)
q=torch.nn.functional.normalize(z+.1*torch.randn_like(z),dim=1)
d=torch.nn.functional.normalize(z+.1*torch.randn_like(z),dim=1)
logits=q@d.T/.07; target=torch.arange(16)
loss=torch.nn.functional.cross_entropy(logits,target)
r1=(logits.argmax(1)==target).float().mean()
print(loss.item(),r1.item()); assert r1>.8
```
- **检测/实验**：加入随机、同主题 hard negatives，比较 Recall@1/10；检查同 batch 的“假负例”。
- **常见坑**：query/document 归一化与索引侧不一致；只评训练 batch 内检索；语料更新后不重建 embedding。

### N05 Learning to Rank 与重排【稳定基础】
- **先修**：N01、N04、二元交叉熵、分组数据。
- **定义与解析**：pointwise 预测单文档相关性，pairwise 学偏好次序，listwise 直接处理整列；重排只能改善召回候选中已有文档。
- **公式/机制**：RankNet 对 $s_i-s_j$ 最小化 $-\left[y\log\sigma(s_i-s_j)+(1-y)\log\!\left(1-\sigma(s_i-s_j)\right)\right]$。
- **资料**：Burges et al. [RankNet §2，式 (1)–(7)](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)；LightGBM [LGBMRanker/lambdarank 参数](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html)。
- **最小代码（可执行 pairwise loss）**：
```python
import torch
torch.manual_seed(0); X=torch.tensor([[2.,0.],[1.,1.],[0.,2.]])
ranker=torch.nn.Linear(2,1); pairs=torch.tensor([[0,1],[1,2],[0,2]])
opt=torch.optim.SGD(ranker.parameters(),.2)
for _ in range(80):
    s=ranker(X).squeeze(); diff=s[pairs[:,0]]-s[pairs[:,1]]
    loss=torch.nn.functional.softplus(-diff).mean()
    opt.zero_grad(); loss.backward(); opt.step()
assert torch.all(ranker(X)[:-1]>ranker(X)[1:])
```
- **检测/实验**：固定候选召回，比较 pointwise 与 pairwise NDCG；把候选 Recall 上限一并报告。
- **常见坑**：跨 query 构造 pair；以分类 AUC 代替 query-group NDCG；重排离线增益掩盖召回退化。

### N06 ANN、混合检索与线上实验【成熟系统，索引演进中】
- **先修**：N01、N04–N05、向量距离、哈希/倒排索引。
- **定义与解析**：ANN 用近似换吞吐和内存；混合检索融合 BM25 与 dense 结果，再重排。索引召回、端到端质量和 p95 延迟必须分层测。
- **公式/机制**：余弦/内积最近邻；RRF 分数 $\sum_r1/(k+\operatorname{rank}_r(d))$ 对不同打分尺度较稳健。
- **资料**：Johnson et al. [FAISS §3–5](https://arxiv.org/abs/1702.08734)；FAISS [Index factory/CPU indexes wiki](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)；Cormack et al. [RRF §2](https://doi.org/10.1145/1571941.1572114)。
- **最小代码（可执行穷举基准，非 ANN）**：
```python
import numpy as np
g=np.random.default_rng(0); db=g.normal(size=(1000,32)); q=g.normal(size=32)
db/=np.linalg.norm(db,axis=1,keepdims=True); q/=np.linalg.norm(q)
exact=np.argsort(-(db@q))[:10]
rank_a={d:r for r,d in enumerate(exact,1)}; rank_b={d:r for r,d in enumerate(exact[::-1],1)}
rrf={d:1/(60+rank_a[d])+1/(60+rank_b[d]) for d in exact}
fused=sorted(rrf,key=rrf.get,reverse=True)
assert len(fused)==10
```
- **检测/实验**：对不同 HNSW/IVF 参数画 recall@10—p95—内存曲线；线上 A/B 预注册 CTR、转化、延迟与护栏指标。
- **常见坑**：只测 ANN QPS、不测 exact recall；索引版本与 embedding 版本错配；以点击提升自动代表满意度提升。
