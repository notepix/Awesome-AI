# G. 生成模型


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](f-nlp-transformers-llms.md) · [下一章 →](h-multimodal-vlm.md)

---

## 知识单元

### G01 显式似然、隐式、潜变量、能量与 Score 范式 `[核心·成熟框架]`

- **先修**：A09–A10、A13、A15、D03、D09。
- **定义与解析**：生成模型学习数据分布或可采样过程。显式模型能计算/下界化似然；隐式模型只定义采样器；潜变量模型引入未观测 `z`；能量模型用未归一化 `exp(-E(x))`；score 模型学习 `∇_x log p_t(x)`。这些轴可交叉，不能把模型硬塞进互斥五类。
- **公式/机制**：潜变量 `p(x)=∫p(x,z)dz`；能量模型 `p(x)=exp(-E(x))/Z`，难点常在配分函数 `Z`；score 不含 `Z` 因为对 `x` 求梯度会消去常数。likelihood 高、样本好看、下游有用是不同目标。
- **资料定位**：[Goodfellow 等《Deep Learning》官方 Ch.20 Deep Generative Models，§20.9–20.14 与 §20.14 Evaluation](https://www.deeplearningbook.org/contents/generative_models.html)；[Score-SDE，ICLR 2021 §2 Background](https://openreview.net/forum?id=PxTIG12RRHS)。
- **最小代码（可运行，离散能量归一化）**：

```python
import torch
support = torch.arange(5.0)
p = torch.softmax(-(support - 2.0).square(), dim=0)
sample = torch.multinomial(p, 5, replacement=True)
q = torch.tensor([1.0], requires_grad=True)
log_unnormalized = -(q - 2.0).square()
score = torch.autograd.grad(log_unnormalized, q)[0]
print(p, p.sum(), sample, score)
```

- **检测题/小实验**：给 EBM 的所有能量加常数，概率和 score 是否变化？为什么只会采样但不能算密度的模型仍可有用？
- **常见坑**：把“生成”限定为图像；把 ELBO 当精确 log-likelihood；用不可比较的似然/感知指标排名所有范式；把 score 误作分类分数。

### G02 自回归生成与密度分解 `[核心·成熟]`

- **先修**：A09、A15、F07–F08、G01。
- **定义与解析**：自回归模型按既定顺序把联合分布分解为条件分布，训练可并行给所有位置算 loss，采样通常逐步。顺序不改变链式法则正确性，却改变条件建模难度与生成延迟。
- **公式/机制**：`p(x_1:T)=Π_t p(x_t|x_<t)`，`NLL=-Σ_t log p(x_t|x_<t)`；图像可按像素/通道排序，文本按 token 排序。teacher forcing 的每个前缀来自真实数据，推理前缀来自模型。
- **资料定位**：[PixelRNN，ICML 2016 §2 Generating an Image Pixel by Pixel、Eq.(1)](https://proceedings.mlr.press/v48/oord16.html)；[Transformer decoder masking，原论文 §3.1–3.2](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)。
- **最小代码（可运行，二值序列 NLL）**：

```python
import torch
torch.manual_seed(0)
x = torch.tensor([[1., 0., 1., 1., 0.]])
prefix = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
model = torch.nn.Linear(1, 1)
logits = model(prefix[..., None]).squeeze(-1)
nll = torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction="sum")
nll.backward()
print(logits.shape, nll.item(), model.weight.grad)
```

- **检测题/小实验**：为什么训练五个位置可一次算完、严格采样却要五步？换一种变量顺序，理论联合分布表达能力与优化难度分别怎样？
- **常见坑**：目标未右移；生成时忘记停止条件；以 token 平均 NLL 直接比较不同 tokenization；把暴露偏差等同于“链式法则有错”。

### G03 潜变量、变分推断与 ELBO `[核心·成熟]`

- **先修**：A09–A15、A07、G01。
- **定义与解析**：潜变量 `z` 表示未直接观察的生成因素。真实后验 `p(z|x)` 往往难算，变分推断用可处理的 `q_φ(z|x)` 逼近；ELBO 同时是 log-likelihood 下界与后验逼近目标。
- **公式/机制**：`log p(x)=ELBO+KL(q(z|x)||p(z|x))`，`ELBO=E_q[log p(x,z)-log q(z|x)] = E_q log p(x|z)-KL(q(z|x)||p(z))`。下界松紧取决于变分族与优化，不只取决于生成器。
- **资料定位**：[Auto-Encoding Variational Bayes，§2.1 Problem Scenario、§2.2 Variational Bound、Eq.(1)–(3)](https://arxiv.org/abs/1312.6114)。
- **最小代码（可运行，离散潜变量精确核对）**：

```python
import torch
pxz = torch.tensor([0.06, 0.24])          # 固定 x、两个 z 的 joint
q = torch.tensor([0.4, 0.6])
log_px = pxz.sum().log()
elbo = (q * (pxz.log() - q.log())).sum()
posterior = pxz / pxz.sum()
gap = (q * (q.log() - posterior.log())).sum()
print(log_px.item(), elbo.item(), gap.item(), (log_px-elbo).item())
```

- **检测题/小实验**：令 `q=posterior`，gap 应为多少？ELBO 上升时 log-likelihood 是否必然同幅上升？
- **常见坑**：漏掉 KL 方向；把单样本 Monte Carlo ELBO 当精确值；混淆先验与聚合后验；下界更高就断言样本更好。

### G04 VAE、层次潜变量与解耦 `[核心·成熟；解耦主张需谨慎]`

- **先修**：D02–D05、G03。
- **定义与解析**：VAE 用 encoder 参数化 `q_φ(z|x)`、decoder 参数化 `p_θ(x|z)`，通过重参数化反传。层次 VAE 用多层潜变量表达多尺度结构。β-VAE 加大 KL 权重鼓励受限表示，但“无监督自动发现真实独立因素”没有一般保证。
- **公式/机制**：高斯重参数 `z=μ+σ⊙ε, ε~N(0,I)`；loss `=-E_q log p(x|z)+β KL(q||p)`。β 大会增加压缩并可能造成 posterior collapse；重构分布应匹配数据类型。
- **资料定位**：[AEVB §2.3、Eq.(4)–(10)](https://arxiv.org/abs/1312.6114)；[β-VAE，ICLR 2017 §2–3](https://openreview.net/forum?id=Sy2fzU9gl)；[Locatello et al., ICML 2019 §3 impossibility result](https://proceedings.mlr.press/v97/locatello19a.html)。
- **最小代码（可运行，VAE 单步机制）**：

```python
import torch
torch.manual_seed(0)
x = torch.randn(4, 6)
enc = torch.nn.Linear(6, 4); dec = torch.nn.Linear(2, 6)
mu, logvar = enc(x).chunk(2, dim=-1)
z = mu + torch.exp(.5 * logvar) * torch.randn_like(mu)
recon = dec(z)
rec = torch.nn.functional.mse_loss(recon, x, reduction="sum") / len(x)
kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / len(x)
loss = rec + kl; loss.backward()
print(rec.item(), kl.item(), z.shape)
```

- **检测题/小实验**：把 β 从 0 改到 10，预测重构/KL 的长期趋势；为何 decoder 很强时可能忽略 z？
- **常见坑**：`logvar` 当 `std`；KL 的 batch/维度 reduction 不一致；Bernoulli/Gaussian likelihood 选错；把漂亮 latent traversal 当无监督可辨识性证明。

### G05 Normalizing Flow `[分支·成熟]`

- **先修**：A05、A07、A13、A16、G01。
- **定义与解析**：normalizing flow 用可逆变换把简单基分布映射到数据分布，可精确算密度和采样。代价是变换须可逆且 Jacobian 行列式可高效计算，结构自由度受限。
- **公式/机制**：若 `x=f(z)`，则 `log p_X(x)=log p_Z(f^{-1}(x))+log|det J_(f^{-1})(x)|`。仿射耦合层保留一部分变量，用其预测另一部分的 scale/shift，使 Jacobian 为三角矩阵。
- **资料定位**：[Real NVP，ICLR 2017 §2.1 Change of Variables、§4.1 Coupling Layers](https://openreview.net/forum?id=HkpbnH9lx)；[Glow，NeurIPS 2018 §3，actnorm、invertible 1×1 conv、affine coupling](https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html)。
- **最小代码（可运行，二维仿射 flow）**：

```python
import torch, math
z = torch.tensor([[.2, -.4], [1., .5]])
s, t = .3 * z[:, :1], 2 * z[:, :1]
x = torch.cat([z[:, :1], z[:, 1:] * torch.exp(s) + t], 1)
z_inv = torch.cat([x[:, :1], (x[:, 1:] - 2*x[:, :1]) * torch.exp(-.3*x[:, :1])], 1)
base_logp = -.5 * (z.square() + math.log(2*math.pi)).sum(1)
logp_x = base_logp - s.squeeze(1)
print(x, torch.allclose(z, z_inv), logp_x)
```

- **检测题/小实验**：为什么前一维不变仍能经多层/置换后变换所有维？删掉 log-determinant 后密度为什么错？
- **常见坑**：正反方向 log-det 符号错；可逆不等于数值稳定；离散图像直接套连续密度忘记 dequantization；以 bits/dim 单指标代表感知质量。

### G06 GAN `[核心·成熟；训练稳定性仍任务相关]`

- **先修**：A17–A18、D03–D08、G01。
- **定义与解析**：GAN 让生成器把噪声映射为样本，判别器区分真实/生成；两者进行极小极大博弈。GAN 可一次前向采样且常锐利，但不提供可处理的显式似然，可能 mode collapse。
- **公式/机制**：原目标 `min_G max_D E_data log D(x)+E_z log(1-D(G(z)))`；实践常用 non-saturating generator loss `-E_z log D(G(z))`。理想判别器与分布散度的结论依赖无限容量和优化到位。
- **资料定位**：[Generative Adversarial Nets，NeurIPS 2014 §2、§4.1 theoretical results](https://proceedings.neurips.cc/paper/2014/hash/f033ed80deb0234979a61f95710dbe25-Abstract.html)；[WGAN，ICML 2017 §2–3](https://proceedings.mlr.press/v70/arjovsky17a.html)。
- **最小代码（可运行，一次交替更新的 loss）**：

```python
import torch
torch.manual_seed(0)
G, D = torch.nn.Linear(2, 1), torch.nn.Linear(1, 1)
real, z = torch.randn(8, 1) + 2, torch.randn(8, 2)
fake = G(z)
d_loss = (torch.nn.functional.softplus(-D(real)).mean()
          + torch.nn.functional.softplus(D(fake.detach())).mean())
d_loss.backward()
for p in D.parameters(): p.grad = None
g_loss = torch.nn.functional.softplus(-D(fake)).mean()
g_loss.backward()
print(d_loss.item(), g_loss.item(), G.weight.grad.norm().item())
```

- **检测题/小实验**：为什么更新 D 时要 `fake.detach()`？若 D 轻易完美，原始饱和型 G loss 的梯度会怎样？
- **常见坑**：同一反向图错误更新双方；只看生成样本网格不测覆盖；把训练震荡都称“博弈正常”；比较 FID 时样本数、预处理和特征实现不同。

### G07 Score Matching、扩散、DDPM 与 SDE `[核心·较成熟；采样研究活跃]`

- **先修**：A07、A10、A15、D03、G01。
- **定义与解析**：扩散模型逐步给数据加噪，再学习反向去噪；DDPM 常预测加入的噪声，等价联系到不同噪声尺度的 score。连续极限以 SDE 描述前向扰动，反向时间 SDE/概率流 ODE 用学习到的 score 生成。
- **公式/机制**：闭式前向 `x_t=√ᾱ_t x_0+√(1-ᾱ_t)ε`；简化 loss `E||ε-ε_θ(x_t,t)||²`。score 与噪声预测成比例 `s_θ(x_t,t)≈-ε_θ/√(1-ᾱ_t)`；完整采样还需时间表与反向更新器。
- **资料定位**：[DDPM，NeurIPS 2020 §2 Background、§3 Diffusion Models and Denoising Autoencoders](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html)；[Score-SDE，ICLR 2021 §3–4](https://openreview.net/forum?id=PxTIG12RRHS)。
- **最小代码（可运行，噪声预测 loss）**：

```python
import torch
torch.manual_seed(0)
x0 = torch.randn(4, 3)
alpha_bar = torch.tensor(.7)
eps = torch.randn_like(x0)
xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
net = torch.nn.Linear(3, 3)
eps_hat = net(xt)
loss = torch.nn.functional.mse_loss(eps_hat, eps)
loss.backward()
print(xt.shape, loss.item(), net.weight.grad.norm().item())
```

- **检测题/小实验**：`ᾱ_t→1` 与 `→0` 时 `x_t` 各像什么？只训练一次噪声预测网络为何还不能直接一步获得正确样本？
- **常见坑**：混淆 `α_t` 与累积 `ᾱ_t`；训练/采样 scheduler 不匹配；说 DDPM loss 就是无条件精确 NLL；只比步数不比函数评估次数和质量。

### G08 条件/潜空间扩散、Guidance 与生成评测 `[核心·较成熟；模型配方快速变化]`

- **先修**：C12、E08、F07、G04、G07。
- **定义与解析**：条件扩散把类别/文本等条件注入去噪器；latent diffusion 先在压缩潜空间扩散再解码，降低空间计算但继承自编码器失真。classifier-free guidance（CFG）在条件/无条件预测间外推，增强条件一致性通常牺牲多样性并可能过饱和。
- **公式/机制**：`ε_cfg=ε_uncond+w(ε_cond-ε_uncond)`；`w=1` 为条件预测，`w>1` 外推。FID 比较特征高斯的均值/协方差，仅是特定特征空间的分布距离；提示一致性、覆盖、人工偏好、安全需另测。
- **资料定位**：[Latent Diffusion，CVPR 2022 §3.1 Perceptual Compression、§3.3 Conditioning Mechanisms](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html)；[Classifier-Free Guidance §2–3](https://arxiv.org/abs/2207.12598)；[FID 原始出处，TTUR/NeurIPS 2017 §4](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)。
- **最小代码（可运行，CFG 机制）**：

```python
import torch
eps_u = torch.tensor([.2, -.1, .4])
eps_c = torch.tensor([.0,  .3, .5])
for w in (0., 1., 3., 7.5):
    eps = eps_u + w * (eps_c - eps_u)
    step = torch.tensor([1., 0., -1.]) - .1 * eps
    print(w, eps.tolist(), step.tolist())
```

- **检测题/小实验**：`w=0/1/>1` 分别代表什么？同一生成器 FID 更低时，文字拼写和事实一致性是否必然更好？
- **常见坑**：把 CFG 与 classifier guidance 混同；在像素/latent 的噪声尺度间错配；只挑样图或只报 FID；由产品演示推断未公开的数据、损失或采样机制。边界截至 **2026-08-11**。
