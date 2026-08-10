# H. VLM 与多模态


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](g-generative-models.md) · [下一章 →](../03-decision-specialties/i-reinforcement-learning.md)

---

## 知识单元

### H01 多模态表示、对齐、融合与 Cross-Attention `[核心·较成熟]`

- **先修**：D09、D12–D13、E08、F07。
- **定义与解析**：多模态模型把图像、文本、音频等映射到可比较或可交互的表示。对齐让对应样本接近；早融合在浅层混合 token，晚融合组合独立决策，cross-attention 让一个模态的 query 从另一模态的 key/value 读取信息。
- **公式/机制**：跨注意力 `softmax(Q_text K_image^T/√d)V_image`；对比目标学全局对齐，token/区域级交互学细粒度条件关系。对齐数据中的共现并非因果对应，单一共享空间也可能丢掉模态私有信息。
- **资料定位**：[Flamingo，NeurIPS 2022 §3 Model：Perceiver Resampler、GATED XATTN-DENSE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)；[ViLBERT，NeurIPS 2019 §3 Model：co-attentional transformer layers](https://proceedings.neurips.cc/paper/2019/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract.html)。
- **最小代码（可运行，Cross-Attention）**：

```python
import torch
torch.manual_seed(0)
text = torch.randn(2, 3, 8)              # query: 3 text tokens
image = torch.randn(2, 5, 8)             # key/value: 5 visual tokens
xattn = torch.nn.MultiheadAttention(8, 2, batch_first=True)
fused, weights = xattn(text, image, image)
print(fused.shape, weights.shape)
print(weights[0].sum(-1))                 # 每个 text query 对图像 token 归一
```

- **检测题/小实验**：交换 query 与 key/value 后输出长度为什么变化？对比学习学到图文全局接近，是否足以定位句中每个名词？
- **常见坑**：不同模态 padding mask 未传；把 embedding 维度相同当已对齐；把 attention 权重当可靠定位/解释；训练集共现捷径误当组合理解。

### H02 CLIP 图文对比预训练 `[核心·成熟]`

- **先修**：A04、A15、D12、E08、F09、H01。
- **定义与解析**：CLIP 用成对图像—文本批次训练双编码器，使正确配对相似度高、批内错配低；推理可把类别写成文本提示做零样本分类，也可做双向检索。它学到开放词汇表征，但不是生成式 VLM。
- **公式/机制**：归一化向量后 `s_ij=(v_i·t_j)/τ`，对行做 image→text、对列做 text→image 交叉熵并平均。温度控制分布尖锐度；批内负样本可能包含语义等价“假负例”。
- **资料定位**：[CLIP，ICML 2021 §2.3 Selecting an Efficient Pre-Training Method、§2.4 Choosing and Scaling a Model、Fig.1](https://proceedings.mlr.press/v139/radford21a.html)。
- **最小代码（可运行，对称 InfoNCE）**：

```python
import torch
torch.manual_seed(0)
v = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
t = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
logits = v @ t.T / .07
labels = torch.arange(4)
li = torch.nn.functional.cross_entropy(logits, labels)
lt = torch.nn.functional.cross_entropy(logits.T, labels)
loss = (li + lt) / 2
print(logits.shape, loss.item(), logits.argmax(1))
```

- **检测题/小实验**：将 batch 从 4 增到 64，批内负例数怎样变化？zero-shot 类别名更换为多模板平均为何可能改变分数？
- **常见坑**：图/文 encoder 输出未 L2 归一；温度方向/可学习参数写反；用训练配对作检索测试；把 CLIP 相似度当校准概率或细粒度事实验证。

### H03 Caption、VQA、视觉定位与文档理解 `[分支·较成熟]`

- **先修**：E06–E08、F09–F13、H01–H02。
- **定义与解析**：caption 是图到文本；VQA 是图像与问题到答案；grounding 把语言短语对到框/掩码；文档理解还需 OCR、二维布局、阅读顺序和表格结构。相同 VLM 骨干可共享，但标注、输出空间与指标不能混用。
- **公式/机制**：caption/生成式 VQA 用条件 NLL；分类式 VQA 对候选答案作 soft target；grounding 联合分类与框/掩码损失；文档模型把词 token embedding 与 2D box embedding 相加/融合。答案语言先验可能绕过视觉。
- **资料定位**：[Show and Tell，CVPR 2015 §2 Model](https://openaccess.thecvf.com/content_cvpr_2015/html/Vinyals_Show_and_Tell_2015_CVPR_paper.html)；[VQA，ICCV 2015 §3 Dataset、§4 Evaluation](https://openaccess.thecvf.com/content_iccv_2015/html/Antol_VQA_Visual_Question_ICCV_2015_paper.html)；[GLIP，CVPR 2022 §3 grounding formulation](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Grounded_Language-Image_Pre-Training_CVPR_2022_paper.html)；[LayoutLM §2](https://arxiv.org/abs/1912.13318)。
- **最小代码（可运行，多任务 loss 形状）**：

```python
import torch
torch.manual_seed(0)
caption_logits = torch.randn(2, 5, 20, requires_grad=True)
caption_target = torch.randint(0, 20, (2, 5))
answer_logits = torch.randn(2, 6, requires_grad=True)
answer = torch.tensor([1, 4])
box_pred = torch.sigmoid(torch.randn(2, 4, requires_grad=True))
box_true = torch.tensor([[.1,.2,.7,.8], [.2,.1,.5,.6]])
loss = (torch.nn.functional.cross_entropy(caption_logits.flatten(0,1), caption_target.flatten())
        + torch.nn.functional.cross_entropy(answer_logits, answer)
        + 5 * torch.nn.functional.l1_loss(box_pred, box_true))
loss.backward(); print(loss.item())
```

- **检测题/小实验**：遮住图像只给问题，若 VQA 仍很高说明什么？OCR 文本正确但 box 坐标全错，对文档阅读顺序会有何影响？
- **常见坑**：把 closed-vocab VQA accuracy 当开放回答能力；caption 指标代替事实核查；框坐标未按缩放同步；OCR 错误与推理错误不分层归因。

### H04 视觉编码器—连接器—LLM 的 VLM 架构 `[核心·较成熟；具体配方演进]`

- **先修**：D11–D13、E08、F09、H01–H03。
- **定义与解析**：常见生成式 VLM 由视觉编码器提取 token，连接器把视觉维度/长度适配到 LLM embedding，再由 LLM 条件生成。连接器可为线性/MLP、query transformer 或 resampler；冻结还是联训决定成本与适配能力。
- **公式/机制**：线性连接 `Z_v=X_vW` 后与文本 embedding 拼接；Q-Former 用少量 learned queries cross-attend 冻结视觉特征，再连接冻结 LLM。视觉 token 只是条件上下文，并不天然可逆或逐像素保真。
- **资料定位**：[BLIP-2，ICML 2023 §3，Q-Former 与 two-stage pre-training](https://proceedings.mlr.press/v202/li23q.html)；[LLaVA §3 Approach：feature alignment 与 visual instruction tuning](https://arxiv.org/abs/2304.08485) 及[作者官方代码](https://github.com/haotian-liu/LLaVA)。
- **最小代码（可运行，连接器与拼接）**：

```python
import torch
torch.manual_seed(0)
vision = torch.randn(2, 4, 12)           # 4 visual tokens
text_ids = torch.randint(0, 30, (2, 5))
connector = torch.nn.Linear(12, 8)
embed = torch.nn.Embedding(30, 8)
visual_tokens = connector(vision)
sequence = torch.cat([visual_tokens, embed(text_ids)], dim=1)
layer = torch.nn.TransformerEncoderLayer(8, 2, batch_first=True)
mask = torch.triu(torch.full((9, 9), -torch.inf), diagonal=1)
out = layer(sequence, src_mask=mask)
print(visual_tokens.shape, sequence.shape, out.shape)
```

- **检测题/小实验**：视觉 token 从 4 增到 576 对上下文长度/attention 成本有何影响？冻结视觉 encoder 时 connector 能否恢复 encoder 已丢弃的信息？
- **常见坑**：忽略图像 resize/crop 与 encoder 预处理；视觉 token 的位置/mask 错；把线性连接成功当“模态鸿沟已解决”；由端到端答案猜测未公开内部架构。

### H05 多模态指令微调与数据混合 `[核心·前沿工程]`

- **先修**：B05–B06、F12、H03–H04。
- **定义与解析**：多模态 SFT 用图像/视频/音频与多轮指令—回答示范，让预训练模型适应交互和任务。训练常分“连接器对齐→指令微调”，并混合纯文本以减轻语言能力遗忘。合成指令可扩规模，也会继承生成器错误与风格。
- **公式/机制**：`L=Σ_d λ_d E_(x,y~D_d)[-Σ_(t∈assistant)log p(y_t|context)]`；视觉/用户/padding token 通常不计生成 loss。混合权重 `λ_d` 决定有效分布，不能只报总样本数。
- **资料定位**：[LLaVA §3.1–3.2，两阶段训练与 visual instruction data generation](https://arxiv.org/abs/2304.08485)；[Llama 3 report §7 Multimodal，image encoder/adaptor/training recipe](https://arxiv.org/abs/2407.21783)（模型报告，结果为作者自报）。
- **最小代码（可运行，assistant-only loss）**：

```python
import torch
torch.manual_seed(0)
B, T, V = 2, 9, 17
logits = torch.randn(B, T, V, requires_grad=True)
labels = torch.randint(0, V, (B, T))
labels[:, :6] = -100                     # 图像 token + 用户提示不监督
labels[1, -1] = -100                     # padding
loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), labels.flatten())
loss.backward()
print(loss.item(), (labels != -100).sum().item())
```

- **检测题/小实验**：若把用户问题也计 loss，优化目标发生什么变化？数据集 A 有百万短回答、B 有万条长推理，只按样本均匀与按 token 均匀有何区别？
- **常见坑**：图像与对话错配；模板/特殊 token 不一致；混合后某模态被大量短样本淹没；把作者报告的 benchmark 提升写成已独立验证。结论边界截至 **2026-08-11**。

### H06 跨模态检索、组合泛化、幻觉与 VLM 评测 `[核心·评测前沿]`

- **先修**：B06、C12、F13、H02–H05。
- **定义与解析**：跨模态检索按图文相似度排名；组合泛化测试已见概念的新关系/次序；视觉幻觉指回答声称图中不存在或不受图证据支持的内容。评测要拆成感知、OCR/定位、知识、推理、校准与拒答，不能用总分掩盖短板。
- **公式/机制**：Recall@K 是至少一个真匹配进入前 K 的 query 比例；组合测试需最小对照（如主客体互换）；POPE 用正/负对象询问统计 accuracy/precision/recall/F1。benchmark 可能受污染、提示敏感、判分器偏差影响。
- **资料定位**：[Winoground，CVPR 2022 §3 Task、§4 Metric](https://openaccess.thecvf.com/content/CVPR2022/html/Thrush_Winoground_Probing_Vision_and_Language_Models_for_Visio-Linguistic_Compositionality_CVPR_2022_paper.html)；[POPE，EMNLP 2023 §3–4](https://aclanthology.org/2023.emnlp-main.20/)；[MMMU，CVPR 2024 §3 Dataset、§4 Experiments](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.html)。
- **最小代码（可运行，图到文 Recall@K）**：

```python
import torch
torch.manual_seed(0)
img = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
txt = img + .20 * torch.randn(5, 8)
txt = torch.nn.functional.normalize(txt, dim=1)
sim = img @ txt.T
ranking = sim.argsort(dim=1, descending=True)
truth = torch.arange(5)[:, None]
for k in (1, 3, 5):
    print(k, (ranking[:, :k] == truth).any(1).float().mean().item())
```

- **检测题/小实验**：Recall@5=100% 是否说明第一名可靠？把“狗追人/人追狗”作成对 caption，普通全局相似度为何可能都高？
- **常见坑**：一图多真 caption 却只认一个；用 LLM judge 不校验一致性/位置偏差；问答准确就忽略幻觉率；闭源模型版本变化仍把跨日期分数排成静态榜单。边界截至 **2026-08-11**。

### H07 统一多模态 Token 与图像、视频、音频生成 `[前沿·快速变化]`

- **先修**：F07–F11、G02、G07–G08、H01–H05。
- **定义与解析**：统一多模态模型把文本与离散图像/音频/视频 token 放进一个序列，或以共享 Transformer 配多个连续生成头；“统一”不必意味着同一 tokenizer/损失。自回归 token、扩散/flow latent、声码器可组合。跨模态输出能力必须按每种模态单独验证。
- **公式/机制**：离散早融合可用 `L=-Σ_t m_t log p(s_t|s_<t)`，不同 token 区间/模态嵌入区分来源；连续生成头则预测噪声/velocity。视频 token 随空间×时间暴涨，音频还受采样率、码率和流式延迟约束。
- **资料定位**：[Chameleon §2 Model Architecture、§3 Training，mixed-modal early fusion](https://arxiv.org/abs/2405.09818) 与[官方代码](https://github.com/facebookresearch/chameleon)；[Video Diffusion Models，NeurIPS 2022 §3](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html)；[AudioLM §2–3，semantic/acoustic token hierarchy](https://arxiv.org/abs/2209.03143)；[Qwen2.5-Omni §2 Thinker–Talker](https://arxiv.org/abs/2503.20215)（2025 技术报告，能力数字为作者自报）。
- **最小代码（可运行，统一离散序列 loss）**：

```python
import torch
torch.manual_seed(0)
# 0..9 text, 10..17 image, 18..23 audio；这里只模拟统一 ID 空间
seq = torch.tensor([[1, 4, 10, 12, 15, 2, 18, 20]])
embed = torch.nn.Embedding(24, 12)
layer = torch.nn.TransformerEncoderLayer(12, 3, batch_first=True)
mask = torch.triu(torch.full((7, 7), -torch.inf), diagonal=1)
h = layer(embed(seq[:, :-1]), src_mask=mask)
logits = torch.nn.Linear(12, 24)(h)
loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), seq[:, 1:].flatten())
loss.backward(); print(logits.shape, loss.item())
```

- **检测题/小实验**：图像码本扩大如何影响 softmax 参数和序列长度？一个模型能输入视频、输出语音，是否说明内部必然使用统一离散 token？
- **常见坑**：把产品名“omni”当机制定义；忽略 tokenizer/codec 重构上限；文本、图像、音频指标混成一个总分；把视频连贯演示写成已证明 3D/物理世界模型。所有前沿判断仅覆盖公开一手资料至 **2026-08-11**，技术报告不等于同行评审或独立复现。
