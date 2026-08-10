# F. NLP、Transformer 与 LLM


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](e-computer-vision.md) · [下一章 →](g-generative-models.md)

---

## 知识单元

### F01 语言层次、Unicode、规范化与语料 `[核心·成熟]`

- **先修**：B01、B05–B06、A10。
- **定义与解析**：自然语言有字符、形态、词法、句法、语义、语用等层次；模型看到的是编码后的符号序列。Unicode code point、UTF-8 code unit/byte、用户感知字素不是同一对象。语料是带来源、许可、时间与采样偏差的数据集，而非“自然语言本身”。
- **公式/机制**：NFC/NFD 保持规范等价，NFKC/NFKD 还折叠兼容字符，可能抹去语义区分。语料经验分布 `p_data(x)=count(x)/N` 受抓取、过滤、去重和混合权重共同决定；下游偏差不能仅靠扩大参数消除。
- **资料定位**：[Unicode UAX #15 v17.0，§1.1 Canonical and Compatibility Equivalence、§1.2 Normalization Forms、§1.3 Process](https://unicode.org/reports/tr15/)；[T5/JMLR，§2.2 Colossal Clean Crawled Corpus](https://jmlr.org/papers/v21/20-074.html)。
- **最小代码（可运行，标准库）**：

```python
import unicodedata as ud
a, b = "é", "e\u0301"
print(a == b, len(a), len(b), a.encode(), b.encode())
for form in ("NFC", "NFD", "NFKC"):
    x, y = ud.normalize(form, a), ud.normalize(form, b)
    print(form, x == y, [hex(ord(c)) for c in x])
print(ud.normalize("NFKC", "① ℌ"))
```

- **检测题/小实验**：为什么 `len(s)` 不一定是屏幕字符数？比较 NFC 与 NFKC 处理数学字母、圈号数字的结果，并判断你的任务能否接受信息折叠。
- **常见坑**：先按 byte 截断再解码；把小写化/繁简转换当无损规范化；训练/推理清洗不同；忽略来源许可、隐私、时间污染和语言覆盖。

### F02 分词、子词、BPE、WordPiece 与词表 `[核心·成熟]`

- **先修**：F01、A10、B01。
- **定义与解析**：tokenizer 把字符串确定性映射为 ID 序列并尽量可逆。子词在词级 OOV 与字符级长序列间折中；BPE 反复合并高频相邻符号，WordPiece 用似然/打分选合并，SentencePiece 可直接在原始字符串上训练。token 不是词，也不跨模型通用。
- **公式/机制**：词表 `V` 与模型 embedding/output 维度耦合；序列计算成本随 token 数增长。BPE 每轮选 `argmax_(a,b) count(a,b)` 并替换；实际系统还要规定预分词、字节回退、特殊 token 与规范化。
- **资料定位**：[BPE for NMT，ACL 2016 §3](https://aclanthology.org/P16-1162/)；[SentencePiece，EMNLP 2018 §2 system overview、§3](https://aclanthology.org/D18-2012/)；[BERT §3.4，WordPiece vocabulary](https://aclanthology.org/N19-1423/)。
- **最小代码（可运行，玩具 BPE）**：

```python
symbols = list("lower") + ["</w>"]
for a, b in [("l", "o"), ("lo", "w"), ("e", "r")]:
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i+1]) == (a, b):
            out.append(a + b); i += 2
        else:
            out.append(symbols[i]); i += 1
    symbols = out
    print(symbols)
```

- **检测题/小实验**：同一中英混合句分别按字符、空格词、子词计长度；词表增大时 embedding 参数、平均序列长度和稀有 token 学习次数如何变化？
- **常见坑**：只保存词表不保存 tokenizer 配置/合并表；新增特殊 token 后不扩 embedding；在 token 化后去重；用“token 数”直接比较不同 tokenizer 的数据量。

### F03 BoW、TF-IDF 与传统文本分类 `[核心·成熟]`

- **先修**：A02、A10、C03、B05–B06、F01–F02。
- **定义与解析**：Bag-of-Words 用词项计数向量表示文档，忽略大部分顺序；n-gram 补局部顺序。TF-IDF 降低跨文档普遍词的权重，配线性分类器仍是强、便宜、可解释的基线。
- **公式/机制**：常见 `tfidf(t,d)=tf(t,d)[log((1+n)/(1+df(t)))+1]`，再做 L2 归一化；逻辑回归为 `p(y|x)=softmax(Wx+b)`。公式变体很多，复现实验须固定 tokenizer、idf 平滑和归一化。
- **资料定位**：[scikit-learn User Guide 8.2.3.5，Tf–idf term weighting（含实现公式与参数）](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)；[CS229 公共讲义 Notes 1，Logistic Regression](https://cs229.stanford.edu/notes2022fall/lecture1.pdf)。
- **最小代码（可运行，scikit-learn）**：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
docs = ["good fast", "great good", "bad slow", "awful bad"]
y = [1, 1, 0, 0]
vec = TfidfVectorizer(ngram_range=(1, 2))
X = vec.fit_transform(docs)
clf = LogisticRegression(random_state=0).fit(X, y)
print(clf.predict(vec.transform(["good", "bad slow"])), X.shape)
```

- **检测题/小实验**：把词序反转，unigram 特征是否变化？只在训练折 `fit` 与全数据 `fit` TF-IDF，对验证分数有何潜在差别？
- **常见坑**：验证/测试参与词表与 IDF 拟合；稀疏矩阵无意转 dense；只用 accuracy 处理不均衡；拿神经模型和未调参、未用 n-gram 的弱基线比较。

### F04 分布式语义、Word2Vec 与静态词向量 `[核心·成熟]`

- **先修**：A04、A09–A10、A15、D09、F03。
- **定义与解析**：分布式假设认为上下文相似的词具有相似表示。Skip-gram 由中心词预测上下文，CBOW 反向；负采样把大词表 softmax 近似为真假词对二分类。静态词向量每个词一个向量，不能随语境消歧。
- **公式/机制**：负采样目标 `log σ(v'_o·v_c)+Σ_(i=1)^k E_(w_i~P_n) log σ(-v'_(w_i)·v_c)`；两套 embedding（中心/上下文）角色不同。余弦相似只反映训练分布几何，不自动等于语义真值。
- **资料定位**：[Mikolov et al., NeurIPS 2013，§2 Skip-gram、§2.2 Negative Sampling](https://proceedings.neurips.cc/paper/2013/hash/9aa42b31882ec039965f3c4923ce901b-Abstract.html)。
- **最小代码（可运行，PyTorch 单个负采样步）**：

```python
import torch
torch.manual_seed(0)
vin = torch.nn.Embedding(8, 4); vout = torch.nn.Embedding(8, 4)
center = torch.tensor([1]); pos = torch.tensor([2]); neg = torch.tensor([3,4,5])
vc = vin(center)
pos_score = (vc * vout(pos)).sum()
neg_score = (vc * vout(neg)).sum(1)
loss = -torch.nn.functional.logsigmoid(pos_score)
loss -= torch.nn.functional.logsigmoid(-neg_score).sum()
loss.backward()
print(loss.item(), vin.weight.grad[1])
```

- **检测题/小实验**：把负样本数从 1 增到 20，损失尺度和计算量如何变？“bank”两个词义为何会挤在一个向量里？
- **常见坑**：把类比偶然性当逻辑推理；未处理高频词采样；混淆输入/输出向量；用含社会偏差的向量而不做审计。

### F05 RNN、LSTM 与 GRU `[核心·成熟]`

- **先修**：A07、D02、D04、D10、F04。
- **定义与解析**：RNN 用共享转移函数逐步更新状态；LSTM/GRU 用门控为信息提供更直接的跨步路径，缓解但不消除长依赖与梯度问题。递归是顺序瓶颈，却适合流式、小状态任务。
- **公式/机制**：基本 RNN `h_t=φ(W_xx_t+W_hh_(t-1)+b)`；LSTM 用输入/遗忘/输出门更新 `c_t=f_t⊙c_(t-1)+i_t⊙g_t`；BPTT 沿时间展开，共享参数梯度为各步贡献之和。
- **资料定位**：[D2L 10.1 Long Short-Term Memory，门公式与 Fig.10.1.3](https://d2l.ai/chapter_recurrent-modern/lstm.html)；[D2L 10.2 GRU，Reset/Update Gates](https://d2l.ai/chapter_recurrent-modern/gru.html)。
- **最小代码（可运行，PyTorch）**：

```python
import torch
torch.manual_seed(0)
cell = torch.nn.GRUCell(3, 5)
x = torch.randn(2, 4, 3, requires_grad=True)
h = torch.zeros(2, 5)
states = []
for t in range(x.size(1)):
    h = cell(x[:, t], h); states.append(h)
loss = states[-1].square().mean(); loss.backward()
print(torch.stack(states).shape, x.grad[:, 0].norm().item())
```

- **检测题/小实验**：把序列长度从 4 增到 100，比较首步输入梯度；双向 RNN 为什么不能无延迟用于严格在线生成？
- **常见坑**：padding 步仍更新状态；hidden/state 形状混乱；训练时未 detach 跨 batch 状态；认为门控能可靠记住任意长度信息。

### F06 Seq2Seq、编码器—解码器与注意力 `[核心·成熟]`

- **先修**：A04、D13、F05。
- **定义与解析**：Seq2Seq 把变长输入编码为状态，再自回归解码输出；固定单向量瓶颈促成注意力：每个解码步按相关性汇聚全部编码状态。teacher forcing 加快训练，但推理时模型消费自身输出，形成暴露偏差。
- **公式/机制**：`e_ti=score(s_(t-1),h_i)`，`α_t=softmax(e_t)`，`c_t=Σ_i α_ti h_i`，再预测 `p(y_t|y_<t,c_t)`；padding 位置必须 mask。注意力权重是信息混合系数，不自动是可信解释。
- **资料定位**：[Bahdanau et al.，ICLR 2015 §3 Learning to Align and Translate、Eq.(4)–(6)](https://arxiv.org/abs/1409.0473)；[D2L 11.4 Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html)。
- **最小代码（可运行，加性注意力形状）**：

```python
import torch
torch.manual_seed(0)
keys = torch.randn(2, 5, 4)             # encoder states
query = torch.randn(2, 1, 4)            # decoder state
Wk, Wq, v = torch.nn.Linear(4, 6), torch.nn.Linear(4, 6), torch.nn.Linear(6, 1)
scores = v(torch.tanh(Wk(keys) + Wq(query))).squeeze(-1)
scores[:, -1] = -torch.inf              # 假设末位 padding
alpha = scores.softmax(-1)
context = (alpha[..., None] * keys).sum(1)
print(alpha, context.shape, alpha.sum(1))
```

- **检测题/小实验**：全 mask 一行会发生什么数值问题？训练时 100% teacher forcing 而推理逐步生成，输入分布发生了什么变化？
- **常见坑**：softmax 维度错；mask 在 softmax 后才乘零且不重归一；将注意力热图直接解释为因果；目标序列未右移导致偷看当前 token。

### F07 Transformer、位置编码、Mask 与 KV Cache `[核心·成熟]`

- **先修**：A02、A04、A16、D07、D13、F06。
- **定义与解析**：Transformer 以多头注意力和逐位置前馈层替代循环；自注意力本身对 token 排列等变，必须注入位置。因果 mask 禁止看未来；KV cache 在自回归推理中复用旧 token 的 key/value，但不会免掉新 query 与全部历史的注意力计算。
- **公式/机制**：`Attention(Q,K,V)=softmax(QK^T/√d_k+M)V`；多头先投影再拼接。正弦位置编码固定，learned absolute 学表，RoPE 在 Q/K 子空间施加随位置变化的旋转，使点积含相对位置信息。
- **资料定位**：[Transformer，NeurIPS 2017 §3.2 Attention、§3.4 Embeddings、§3.5 Positional Encoding](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)；[RoPE §3 Proposed Method](https://arxiv.org/abs/2104.09864)。
- **最小代码（可运行，因果自注意力）**：

```python
import torch, math
torch.manual_seed(0)
x = torch.randn(1, 4, 8)
q, k, v = (torch.nn.Linear(8, 8, bias=False)(x) for _ in range(3))
score = q @ k.transpose(-2, -1) / math.sqrt(8)
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
score = score.masked_fill(mask, -torch.inf)
a = score.softmax(-1)
y = a @ v
print(a[0], y.shape)
```

- **检测题/小实验**：验证第 0 行只能关注自己；把输入 token 同步置换且不加位置编码，输出应如何置换？KV cache 为何主要省去旧 token 的 K/V 投影？
- **常见坑**：把 padding mask、causal mask、loss mask 混用；缩放除以 `√d_model` 而非头维；cache 位置索引错位；把 RoPE 外推当成训练长度外必然可靠。

### F08 因果、掩码与 Encoder–Decoder 语言建模目标 `[核心·成熟]`

- **先修**：A09、A15、D03、F02、F07。
- **定义与解析**：因果 LM 预测下一个 token；掩码 LM 从双向上下文恢复被遮 token；encoder–decoder 去噪把受损输入映射回目标跨度/文本。目标决定可见信息和训练信号，不等同于具体模型品牌。
- **公式/机制**：CLM 为 $L=-\sum_t\log p(x_t\mid x_{<t})$；MLM 为 $L=-\sum_{t\in M}\log p(x_t\mid x_{\setminus M})$；seq2seq 为 $L=-\sum_t\log p(y_t\mid y_{<t},\tilde{x})$。只对指定位置计 loss；标签移位和 mask 是防止信息泄漏的核心。
- **资料定位**：[BERT §3.1，Masked LM 与 Next Sentence Prediction](https://aclanthology.org/N19-1423/)；[GPT-1 §3.1 Unsupervised pre-training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)；[T5 §3.1–3.3，text-to-text 与 unsupervised objectives](https://jmlr.org/papers/v21/20-074.html)。
- **最小代码（可运行，三种 loss mask）**：

```python
import torch
torch.manual_seed(0)
B, T, V = 2, 5, 11
logits = torch.randn(B, T, V)
ids = torch.randint(0, V, (B, T))
clm = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V), ids[:, 1:].reshape(-1))
masked = torch.tensor([[0,1,0,1,0], [0,0,1,0,0]], dtype=torch.bool)
mlm = torch.nn.functional.cross_entropy(logits[masked], ids[masked])
valid = ids.ne(0); seq2seq = torch.nn.functional.cross_entropy(logits[valid], ids[valid])
print(clm.item(), mlm.item(), seq2seq.item())
```

- **检测题/小实验**：若 CLM 的输入和标签同位置对齐且模型有残差，会出现什么捷径？MLM 为何不天然适合逐 token 左到右生成？
- **常见坑**：padding 也计入 loss；把 `[MASK]` 留在下游真实输入；错误 shift 造成当前 token 泄漏；将 NSP、sentence order 等辅助目标视作 BERT 必不可少定义。

### F09 BERT、GPT 与 T5 `[核心·成熟]`

- **先修**：F07–F08、D11–D12。
- **定义与解析**：BERT 是双向 Transformer encoder 的掩码预训练范式，适合理解/编码；GPT 是 decoder-only 因果语言模型，统一为续写；T5 是 encoder–decoder，把任务写成 text-to-text。三者边界来自可见性、结构和目标，而非参数规模。
- **公式/机制**：BERT 输出上下文化 token 表征；GPT 分解联合概率 `Π_t p(x_t|x_<t)`；T5 encoder 全局读输入、decoder 因果生成并 cross-attend encoder memory。微调可更新全模型、头部或参数高效适配器。
- **资料定位**：[BERT，NAACL 2019 §3 architecture/input/pretraining、§4 fine-tuning](https://aclanthology.org/N19-1423/)；[GPT-1 §3 Framework](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)；[T5/JMLR §3 systematic study、Fig.1](https://jmlr.org/papers/v21/20-074.html)。
- **最小代码（可运行，三种可见性）**：

```python
import torch
torch.manual_seed(0)
layer = torch.nn.TransformerEncoderLayer(8, 2, batch_first=True)
encoder = torch.nn.TransformerEncoder(layer, 1)
x = torch.randn(1, 4, 8)
bert = encoder(x)                         # 双向 encoder
causal = torch.triu(torch.full((4, 4), -torch.inf), diagonal=1)
gpt = encoder(x, mask=causal)             # 用同层示意 decoder 自注意力可见性
dec = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(8, 2, batch_first=True), 1)
t5 = dec(x[:, :2], encoder(x), tgt_mask=causal[:2, :2])
print(bert.shape, gpt.shape, t5.shape)
```

- **检测题/小实验**：分类任务为何常取 BERT 的 pooled/特殊 token 表征，而抽取任务需逐 token 输出？T5 decoder 若无 cross-attention 会退化成什么？
- **常见坑**：把“GPT”泛指所有 LLM；用 encoder 的双向 mask 做生成训练；只按架构名推断数据/能力；把预训练目标成绩当下游可靠性保证。

### F10 预训练数据、去重、Scaling 与配比 `[核心·较成熟；配方快速演进]`

- **先修**：A13、A18、B05–B06、D14、F01–F09。
- **定义与解析**：预训练系统由数据来源、许可/治理、过滤、去重、采样配比、tokenizer、训练预算共同定义。scaling law 是给定范围内损失随模型/数据/计算的经验幂律拟合；compute-optimal 配比取决于架构、数据质量和训练制度，不是自然常数。
- **公式/机制**：常见拟合 `L(N,D)=E+A/N^α+B/D^β`；固定计算近似 `C≈kND` 时在参数量 `N` 与 token 数 `D` 间权衡。近重复去重既降低记忆/污染，也会改变长尾与语言分布。
- **资料定位**：[OpenAI Scaling Laws §3、Fig.1](https://arxiv.org/abs/2001.08361)；[Chinchilla，NeurIPS 2022 §2–3 compute-optimal scaling](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html)；[Llama 3 report §3 Pre-training：data、scaling、training](https://arxiv.org/abs/2407.21783)。
- **最小代码（可运行，玩具预算搜索）**：

```python
import numpy as np
C = 1e8
N = np.logspace(2, 6, 200)
D = C / N
E, A, B, alpha, beta = 1.0, 20.0, 30.0, .4, .3  # 仅玩具拟合
loss = E + A / N**alpha + B / D**beta
i = loss.argmin()
print(f"N={N[i]:.0f}, D={D[i]:.0f}, loss={loss[i]:.3f}")
```

- **检测题/小实验**：改变 `alpha/beta`，最优配比如何移动？去重为何可能同时降低 benchmark 分数（去掉污染）又提高真实泛化？
- **常见坑**：跨论文直接套指数；把 token 数等同信息量；忽略测试污染和训练数据时间边界；模型报告未公开完整数据时仍声称已复现。结论边界截至 **2026-08-11**。

### F11 Greedy、Beam、采样与约束解码 `[核心·成熟]`

- **先修**：A09–A10、A15、F08–F09。
- **定义与解析**：greedy 每步取最大概率；beam 保留若干高累积分序列；随机采样从截断/重标定分布取样；约束解码限制合法 token 或结构。解码改变输出分布，却不能补回模型没学到的事实。
- **公式/机制**：温度 `p_i∝exp(z_i/T)`；top-k 仅保留 k 个，top-p 保留累计概率至少 p 的最小集合；beam 常按 `Σ_t log p(y_t|y_<t)` 并做长度归一。低温不等于校准，高 beam 可能放大长度/重复偏置。
- **资料定位**：[Hugging Face Transformers 官方 Generation strategies：greedy、sampling、beam、speculative decoding](https://huggingface.co/docs/transformers/main/en/generation_strategies)；[Holtzman et al., ICLR 2020 §3 Nucleus Sampling](https://openreview.net/forum?id=rygGQyrFvH)。
- **最小代码（可运行，top-p 单步）**：

```python
import torch
torch.manual_seed(0)
logits = torch.tensor([2.0, 1.2, .5, .1, -1.]) / .8
order = logits.argsort(descending=True)
prob = logits[order].softmax(0)
keep = prob.cumsum(0) - prob <= .90       # 保留越过阈值的那个 token
candidates, p = order[keep], prob[keep]
p = p / p.sum()
sample = candidates[torch.multinomial(p, 1)]
print(candidates.tolist(), p.tolist(), sample.item())
```

- **检测题/小实验**：温度趋近 0/无穷时分布怎样？构造一个 greedy 首步最优却整句概率低于另一序列的二步例子。
- **常见坑**：softmax 后再除温度；top-p 集合未重归一；比较采样方法却不固定随机种子/预算；结构约束只保证语法，不保证语义与安全。

### F12 SFT、指令数据、Chat Template 与 PEFT/LoRA `[核心·较成熟；工具接口会变]`

- **先修**：D11、F02、F08–F11。
- **定义与解析**：SFT 用示范 `(instruction,response)` 的 token 级交叉熵教模型遵循交互格式；chat template 将角色消息序列化为模型训练时的特殊 token 协议。PEFT 只训练小量参数；LoRA 把线性层增量限制为低秩。SFT 不是偏好优化，也不能保证事实性。
- **公式/机制**：`W'=W+(α/r)BA`，`A∈R^(r×d_in),B∈R^(d_out×r)`，冻结 `W`；常把 prompt token loss 置 `-100`，只监督 assistant response。合并 LoRA 便于推理，但合并/量化顺序影响误差。
- **资料定位**：[LoRA，ICLR 2022 §4.1、Eq.(3)](https://openreview.net/forum?id=nZeVKeeFYf9) 与[微软官方实现](https://github.com/microsoft/LoRA)；[Transformers 官方 Chat templates，Using apply_chat_template](https://huggingface.co/docs/transformers/main/en/chat_templating)；[TRL 官方 SFTTrainer，Quick start、Expected dataset type](https://huggingface.co/docs/trl/main/en/sft_trainer)。
- **最小代码（可运行，LoRA 线性层）**：

```python
import torch
torch.manual_seed(0)
d_in, d_out, r, alpha = 6, 4, 2, 4
W = torch.randn(d_out, d_in)              # 视为冻结基座
A = torch.nn.Parameter(torch.randn(r, d_in) * .01)
B = torch.nn.Parameter(torch.zeros(d_out, r))
x = torch.randn(3, d_in)
y = x @ (W + (alpha / r) * (B @ A)).T
loss = y.square().mean(); loss.backward()
print(y.shape, A.grad.norm().item(), B.grad.norm().item())
```

- **检测题/小实验**：为什么 `B=0` 初始化时首步 `A.grad` 可能为 0 而 `B.grad` 非 0？同一消息用两个 chat template 序列化，token loss 能直接比较吗？
- **常见坑**：训练/推理模板不一致；特殊 token 重复；prompt、padding 也计 loss；把“可训练参数少”误解为显存一定极低（激活仍在）；`main` 文档接口可能漂移，工程复现需固定版本。边界截至 **2026-08-11**。

### F13 分类、标注、抽取、翻译、摘要与 QA 评测 `[核心·成熟；开放生成评测仍不完备]`

- **先修**：B06、C03、C12、F03、F09–F12。
- **定义与解析**：分类给序列标签，序列标注给 token 标签，抽取预测 span，翻译/摘要生成文本，QA 可抽取或生成。指标必须对应错误成本：macro-F1 关注小类，span EM/F1 关注边界，BLEU/ROUGE 测表面重叠；任何单指标都不等于语义正确或有用。
- **公式/机制**：`precision=TP/(TP+FP)`、`recall=TP/(TP+FN)`、`F1=2PR/(P+R)`；macro 先按类求再平均，micro 汇总计数。BLEU 组合裁剪 n-gram precision 与 brevity penalty；抽取 QA 常经规范化后算 EM/token F1。
- **资料定位**：[BLEU，ACL 2002 §2 The Baseline BLEU Metric](https://aclanthology.org/P02-1040/)；[ROUGE，ACL Workshop 2004 §2](https://aclanthology.org/W04-1013/)；[SQuAD，EMNLP 2016 §4 Evaluation](https://aclanthology.org/D16-1264/)。
- **最小代码（可运行，macro/micro F1）**：

```python
from sklearn.metrics import f1_score, confusion_matrix
y = [0, 0, 0, 0, 1, 2]
pred = [0, 0, 0, 0, 0, 0]
for avg in ("micro", "macro", None):
    print(avg, f1_score(y, pred, average=avg, zero_division=0))
print(confusion_matrix(y, pred))
```

- **检测题/小实验**：解释代码中 micro 与 macro 的差距；两个语义等价译文可能 BLEU 低，怎样用人工盲评和任务成功率补充？
- **常见坑**：用测试集选阈值；tokenizer 不同仍直接比 token F1；抽取 span 的字符/token offset 错位；只报平均分不做按语言、长度、类别、时间切片误差分析。

### F14 长上下文、高效 Attention、KV Cache 与 LLM 推理 `[前沿；系统快速演进]`

- **先修**：A16、D16、F07、F10–F12。
- **定义与解析**：长上下文要同时解决位置外推、注意力计算/显存与有效检索；FlashAttention 用 IO-aware 分块精确计算注意力，结果不是稀疏近似；MQA/GQA 让多个 query 头共享较少 KV 头以压 cache。声称的最大窗口不等于所有位置都能可靠利用。
- **公式/机制**：标准 prefill 注意力算量约 `O(L²d)`；每层 KV cache 字节约 `2·B·L·H_kv·d_head·bytes`。分块在线 softmax 避免物化完整 `L×L` 矩阵；decode 每步仍读取历史 KV，带宽常成瓶颈。
- **资料定位**：[FlashAttention §3，Tiling 与 recomputation](https://arxiv.org/abs/2205.14135)；[Multi-Query Attention §2](https://arxiv.org/abs/1911.02150)；[LongRoPE，ICML 2024 poster/论文入口，位置插值搜索与渐进扩展](https://icml.cc/virtual/2024/poster/34166)。
- **最小代码（可运行，增量 KV cache）**：

```python
import torch, math
torch.manual_seed(0)
Wk, Wv, Wq = (torch.nn.Linear(8, 8, bias=False) for _ in range(3))
K = torch.empty(1, 0, 8); V = torch.empty(1, 0, 8)
for _ in range(4):
    x = torch.randn(1, 1, 8)
    K = torch.cat([K, Wk(x)], 1); V = torch.cat([V, Wv(x)], 1)
    q = Wq(x)
    y = (q @ K.transpose(-2, -1) / math.sqrt(8)).softmax(-1) @ V
print(K.shape, V.shape, y.shape, "elements=", K.numel() + V.numel())
```

- **检测题/小实验**：把长度翻倍，prefill 注意力矩阵和 KV cache 各放大几倍？为何“needle in a haystack”通过仍不能证明长文综合推理可靠？
- **常见坑**：把 FlashAttention 说成线性时间；只报可输入长度不报质量/延迟/显存；混淆 prefill 与 decode；cache 截断后位置/attention mask 错位。论文、硬件与框架结论均以 **2026-08-11** 为边界，部署需按固定版本实测。
