# O. LLM 后训练、RAG 与 Agent


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](../03-decision-specialties/n-recommendation-search-retrieval.md) · [下一章 →](p-mlops-safety-evaluation.md)

---

## 知识单元

### O01 指令数据与监督微调 SFT【稳定流程】
- **先修**：Transformer、语言模型交叉熵、tokenization。
- **定义与解析**：SFT 用“指令/上下文→期望回答”继续训练预训练模型；通常只对 assistant token 计 loss，数据质量、混合比例和格式一致性比盲目增量更重要。
- **公式/机制**：$L_{\mathrm{SFT}}=-\sum_{t\in\mathrm{assistant}}\log p_\theta(y_t\mid x,y_{<t})$，prompt token label 设为 ignore index。
- **资料**：Ouyang et al. [InstructGPT §3.2，Fig.2](https://arxiv.org/abs/2203.02155)；Hugging Face TRL [`SFTTrainer` data formats/loss masking](https://huggingface.co/docs/trl/sft_trainer)。
- **最小代码（可执行 mask loss）**：
```python
import torch
torch.manual_seed(0); B,T,V=2,6,9
logits=torch.randn(B,T,V,requires_grad=True); labels=torch.randint(V,(B,T))
labels[:,:3]=-100                              # prompt 不计损失
loss=torch.nn.functional.cross_entropy(logits.view(-1,V),labels.view(-1),ignore_index=-100)
loss.backward()
assert logits.grad[:,:3].abs().sum()==0
print(loss.item())
```
- **检测/实验**：同一小模型比较全序列 loss 与 response-only loss；检查聊天模板、EOS、截断后监督 token 数。
- **常见坑**：训练/推理 chat template 不同；把用户文本也当回答监督；仅看训练 loss、不做人评与保留能力回归。

### O02 参数高效微调 LoRA【稳定基础，变体活跃】
- **先修**：O01、矩阵秩、线性层。
- **定义与解析**：LoRA 冻结原权重，用低秩 (BA) 表示任务更新；降低可训练参数和优化器内存，但不保证所有任务与秩都等价于全量微调。
- **公式/机制**：$W'=W+(\alpha/r)BA$，$A\in\mathbb R^{r\times d_{\mathrm{in}}},\ B\in\mathbb R^{d_{\mathrm{out}}\times r}$。
- **资料**：Hu et al. [LoRA §4.1，式 (3)](https://openreview.net/forum?id=nZeVKeeFYf9)；Hugging Face PEFT [LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)。
- **最小代码（可执行）**：
```python
import torch
torch.manual_seed(0); din,dout,r=8,5,2
W=torch.randn(dout,din); A=torch.randn(r,din,requires_grad=True); B=torch.zeros(dout,r,requires_grad=True)
x=torch.randn(4,din); alpha=4
y=x@(W+(alpha/r)*(B@A)).T
loss=y.square().mean(); loss.backward()
assert W.grad is None and A.grad is not None and B.grad is not None
print(A.numel()+B.numel(),W.numel())
```
- **检测/实验**：比较 r=1/2/8 的可训练参数、验证指标与合并前后输出误差。
- **常见坑**：target modules 选错；保存 adapter 却漏记 base model revision；量化、merge 和 dtype 导致输出漂移。

### O03 偏好数据与奖励模型【稳定框架，标注科学活跃】
- **先修**：O01、成对排序、采样偏差。
- **定义与解析**：奖励模型把 prompt-response 映射为标量，并从 chosen/rejected 对学习偏好；它拟合的是标注协议与人群，不是真实、普遍的“人类价值函数”。
- **公式/机制**：Bradley–Terry：$P(y_w\succ y_l\mid x)=\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)$。
- **资料**：Ouyang et al. [§3.3 Reward modeling，式 (1)](https://arxiv.org/abs/2203.02155)；Bradley & Terry [1952 原始模型](https://doi.org/10.2307/2334029)。
- **最小代码（可执行 pairwise RM）**：
```python
import torch
torch.manual_seed(0); chosen=torch.randn(20,4)+.5; rejected=torch.randn(20,4)-.5
rm=torch.nn.Linear(4,1); opt=torch.optim.Adam(rm.parameters(),.05)
for _ in range(100):
    margin=rm(chosen).squeeze()-rm(rejected).squeeze()
    loss=-torch.nn.functional.logsigmoid(margin).mean()
    opt.zero_grad(); loss.backward(); opt.step()
acc=(margin>0).float().mean()
assert acc>.8
```
- **检测/实验**：按标注者、主题、答案长度分层看一致率；加入长度相同的对照检查 reward 是否学到长度捷径。
- **常见坑**：同 prompt 的回答跨 split；chosen 总是更长；RM 分数跨模型版本直接比较。

### O04 RLHF、KL 约束与 DPO【成熟主线，快速演进】
- **先修**：O03、策略梯度、参考模型、log-prob。
- **定义与解析**：经典 RLHF 用奖励模型加 KL 约束优化策略；DPO 将同一偏好目标化为 chosen/rejected 的分类损失，无显式 reward model 与在线 RL，但仍依赖参考模型和数据分布。
- **公式/机制**：DPO：$L=-\log\sigma\!\left(\beta\left[(\log\pi_w-\log\pi_{\mathrm{ref},w})-(\log\pi_l-\log\pi_{\mathrm{ref},l})\right]\right)$。
- **资料**：Rafailov et al. [DPO §4，式 (7)](https://arxiv.org/abs/2305.18290)；Ouyang et al. [§3.4 RLHF](https://arxiv.org/abs/2203.02155)。
- **最小代码（可执行 DPO loss）**：
```python
import torch
pi_w=torch.tensor([-2.,-1.],requires_grad=True); pi_l=torch.tensor([-2.2,-.8],requires_grad=True)
ref_w=torch.tensor([-2.1,-1.1]); ref_l=torch.tensor([-2.0,-.7]); beta=.1
margin=(pi_w-ref_w)-(pi_l-ref_l)
loss=-torch.nn.functional.logsigmoid(beta*margin).mean()
loss.backward()
print(loss.item(),pi_w.grad,pi_l.grad)
assert (pi_w.grad<0).all() and (pi_l.grad>0).all()
```
- **检测/实验**：扫描 β，报告偏好 win-rate、KL、通用能力和安全集，不以单一 judge 分数验收。
- **常见坑**：sequence log-prob 是否按长度归一不明确；reference/template 不匹配；“不用 RL”误解为“不做分布约束”。

### O05 RAG：切分、索引与检索【稳定架构，配方演进中】
- **先修**：N01、N04–N06、文本切分、LLM 上下文窗口。
- **定义与解析**：RAG 将外部资料检索结果放入生成上下文；切分决定证据粒度，检索失败不能靠生成器可靠补救。
- **公式/机制**：RAG-sequence 近似 $p(y\mid x)=\sum_{z\in\operatorname{top}\text{-}k}p_\eta(z\mid x)p_\theta(y\mid x,z)$。
- **资料**：Lewis et al. [RAG §2，式 (1)–(2)](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)；DPR [§3](https://aclanthology.org/2020.emnlp-main.550/)。
- **最小代码（可执行 TF-IDF 检索）**：
```python
from sklearn.feature_extraction.text import TfidfVectorizer
docs=['Bellman 方程用于序贯决策','GCN 聚合图邻居','RAG 先检索再生成']
query='什么方法会先检索资料再回答'
v=TfidfVectorizer(analyzer='char',ngram_range=(2,3))
D=v.fit_transform(docs); q=v.transform([query]); score=(D@q.T).toarray().ravel()
top=score.argsort()[::-1][:2]
print(top,[docs[i] for i in top])
assert top[0]==2
```
- **检测/实验**：扫描 chunk 长度/重叠/k，分别报告 context recall、MRR、答案正确率与延迟。
- **常见坑**：文档解析顺序错乱；query 与 corpus encoder/version 不一致；仅测最终回答，不定位检索还是生成故障。

### O06 Grounded RAG、引用与端到端评估【演进中】
- **先修**：O05、N01、事实核验、评测设计。
- **定义与解析**：grounded answer 的每个可验证主张应由提供的证据蕴含并能定位来源；“含引用”不等于引用支持该句。
- **公式/机制**：端到端拆为 context relevance/recall、faithfulness、answer correctness；自动 judge 需用人工样本校准一致率。
- **资料**：Es et al. [RAGAS EACL 2024 Demo，§2–3](https://aclanthology.org/2024.eacl-demo.16/)；Lewis et al. [RAG §4.5 Factuality](https://papers.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)。
- **最小代码（可执行的词项覆盖诊断；不是语义蕴含器）**：
```python
import re
evidence='RAG combines a retriever with a generator.'
claim='RAG uses a retriever and a generator.'
tok=lambda s:set(re.findall(r'[a-z]+',s.lower()))-{'a','the','and','with','uses'}
coverage=len(tok(claim)&tok(evidence))/max(1,len(tok(claim)))
print(coverage)
assert 0<=coverage<=1
# 此规则只作可执行冒烟测试，不能替代人工或 NLI 忠实度评估
```
- **检测/实验**：建 50 条含“正确、错引、无证据、证据冲突”的金标集，校准自动 judge；报告各类混淆矩阵。
- **常见坑**：judge 与生成模型同源造成偏好偏差；证据在上下文但不支持结论；网页更新后引用无法复现。

### O07 工具、资源与 Agent 协议【协议稳定化中】
- **先修**：JSON Schema、RPC、鉴权、O05。
- **定义与解析**：协议规定消息、能力发现、参数 schema、结果和权限边界；它不规定模型何时调用何工具。MCP/JSON-RPC 是接口协议，ReAct/规划器才是策略。
- **公式/机制**：`initialize→capability negotiation→tools/list→tools/call`；host 负责 consent/隔离，server 暴露 tools/resources/prompts。
- **资料**：MCP 规范 [2025-11-25 Schema/Capabilities](https://modelcontextprotocol.io/specification/2025-11-25/schema)；[2025-06-18 Architecture: host/client/server](https://modelcontextprotocol.io/specification/2025-06-18/architecture)；[Tools safety](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)。
- **最小代码（可执行安全模拟；不发网络请求）**：
```python
TOOLS={'add':lambda a,b:a+b}
request={'name':'add','arguments':{'a':2,'b':3}}
assert request['name'] in TOOLS
args=request['arguments']
assert set(args)=={'a','b'} and all(isinstance(v,(int,float)) for v in args.values())
result={'ok':True,'value':TOOLS[request['name']](**args)}
print(result); assert result['value']==5
# 真实协议还必须做鉴权、用户确认、超时和审计
```
- **检测/实验**：给工具增加删除副作用，设计 dry-run、最小权限、幂等键和用户确认；说明协议兼容不保证策略正确。
- **常见坑**：信任 tool description/annotation；把任意字符串送 `eval`/shell；服务端看到超出最小需要的完整上下文。

### O08 Agent 策略：ReAct、规划、记忆与多 Agent【研究前沿】
- **先修**：O05–O07、状态机、错误恢复、评测。
- **定义与解析**：Agent 策略决定“观察—思考—选工具—验证—停止”；工作记忆保存当前状态，长期记忆需检索与过期策略。多 Agent 是多个策略主体，不是多开几个相同 prompt。
- **公式/机制**：策略 $\pi(a_t\mid h_t)$ 作用在协议允许的动作集合；停止条件、预算、重试和补偿事务属于控制器，而非 LLM 自由文本。
- **资料**：Yao et al. [ReAct §2–3，Fig.1](https://openreview.net/forum?id=WE_vluYUL-X)；MCP [Architecture 的安全隔离原则](https://modelcontextprotocol.io/specification/2025-06-18/architecture)。
- **最小代码（可执行确定性策略模拟；不调用 LLM）**：
```python
def search(q): return {'capital of france':'Paris'}.get(q.lower(),'UNKNOWN')
def agent(question,budget=2):
    trace=[]
    for _ in range(budget):
        obs=search(question); trace.append(('search',obs))
        if obs!='UNKNOWN': return {'answer':obs,'trace':trace,'status':'ok'}
    return {'answer':None,'trace':trace,'status':'insufficient_evidence'}
r=agent('capital of France')
print(r); assert r['answer']=='Paris'
```
- **检测/实验**：注入超时、矛盾结果、循环调用和预算耗尽；比较单 Agent 与角色化多 Agent 的成功率、成本和新增故障面。
- **常见坑**：协议层日志当作推理质量；无限重试；把未验证 observation 写入长期记忆；多 Agent 投票制造相关错误而非独立证据。
