# R. 2024–2026 前沿技术地图

这一章不是“模型排行榜”。它抽取截至 2026-08-11 仍有解释力的技术主线，并明确哪些结论只是特定论文或产品在特定评测上的结果。先完成对应先修单元，再阅读本章。


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](../04-systems-agents-robotics/q-embodied-ai-robotics.md) · [下一章 →](../06-projects/s-projects-assessment.md)

---

## 知识单元

### R01 `[前沿]` 稀疏 Mixture of Experts（MoE）与路由

**先修**：D09 表征、D13 Transformer、D16 训练系统、F07 Transformer 细节。

**定义与解析**：MoE 为每个 token 准备多个前馈“专家”，路由器只激活其中 top-k 个。这样可以扩大**总参数量**，而不让每个 token 都经过全部参数。总参数量、激活参数量、显存占用、通信量和实际 FLOPs 是五个不同概念。MoE 的难点不是公式，而是专家负载不均、token 丢弃、跨设备通信和训练稳定性。

**公式/机制**：对 token 表示 $x$，路由权重 $g=\operatorname{softmax}(W_rx)$，选择集合 $S=\operatorname{TopK}(g)$：

$$
y=\sum_{i\in S}\frac{g_i}{\sum_{j\in S}g_j}E_i(x).
$$

实践中还会加入负载均衡目标或无辅助损失的平衡策略。

**资料定位**：先读 [Switch Transformer §2](https://arxiv.org/abs/2101.03961)，再读 [Mixtral §2 Architecture](https://arxiv.org/abs/2401.04088)。工程前沿案例见 [DeepSeek-V3 §2](https://arxiv.org/abs/2412.19437)；其中的实现配方不是普适最优结论。

```python
# 代码：可运行。为便于观察，这里仍计算全部专家，再用 mask 模拟 top-k。
import torch
from torch import nn
torch.manual_seed(0)
B, D, E, K = 8, 4, 3, 2
x = torch.randn(B, D)
router = nn.Linear(D, E)
experts = nn.ModuleList([nn.Linear(D, D) for _ in range(E)])
gate = router(x).softmax(-1)                    # [B, E]
top = gate.topk(K, dim=-1).indices
mask = torch.zeros_like(gate).scatter_(1, top, 1.0)
weight = gate * mask
weight = weight / weight.sum(-1, keepdim=True)
expert_y = torch.stack([expert(x) for expert in experts], dim=1)
y = torch.einsum("be,bed->bd", weight, expert_y)
load = mask.mean(0)
assert y.shape == x.shape and torch.isfinite(y).all()
print("expert load:", load.tolist())
```

**检测题/小实验**：把路由器某个 bias 设得很大，观察 `load`；解释为什么“671B 总参数、每 token 激活较少参数”不能直接推出速度一定快。尝试加入负载熵或方差惩罚。

**常见坑**：把 MoE 当成模型集成；忽略路由通信；只报总参数；把某一篇论文的专家数、top-k 和平衡策略当作固定标准。

### R02 `[前沿]` RoPE、位置外推与长上下文

**先修**：A04 内积、F07 位置编码、F14 长上下文。

**定义与解析**：Rotary Position Embedding（RoPE）按位置旋转 Query/Key 的二维通道对，使注意力内积显式依赖相对位置。位置插值、YaRN、LongRoPE 等方法可以扩展可输入的窗口，但“能放入 N 个 token”不等于模型能正确检索、组合并推理 N 个 token。

**公式/机制**：对二维通道对 $(x_{2i},x_{2i+1})$，位置 $m$ 施加旋转 $R(m\theta_i)$。由于 $R(m)^\top R(n)=R(n-m)$，Query/Key 内积包含相对位置信息。

**资料定位**：[RoFormer §3](https://arxiv.org/abs/2104.09864)；扩窗方法读 [YaRN](https://openreview.net/forum?id=wHBfxhZu1u) 和 [LongRoPE](https://icml.cc/virtual/2024/poster/34166)。分布式超长注意力读 [Ring Attention](https://openreview.net/forum?id=WsRHpHH4s0)。

```python
# 代码：可运行。最小 RoPE；旋转应保持每个 token 的向量范数。
import torch
def rope(x):                                    # x: [T, D], D 为偶数
    T, D = x.shape
    freq = 10000.0 ** (-torch.arange(0, D, 2).float() / D)
    angle = torch.arange(T).float()[:, None] * freq[None, :]
    even, odd = x[:, 0::2], x[:, 1::2]
    y = torch.empty_like(x)
    y[:, 0::2] = even * angle.cos() - odd * angle.sin()
    y[:, 1::2] = even * angle.sin() + odd * angle.cos()
    return y
torch.manual_seed(0)
x = torch.randn(12, 8)
y = rope(x)
assert torch.allclose(x.norm(dim=1), y.norm(dim=1), atol=1e-5)
```

**检测题/小实验**：比较短上下文准确率、长文“针尖检索”、跨段组合推理三种评测。为什么只测 needle-in-a-haystack 仍不能证明长程推理？

**常见坑**：把训练窗口、API 窗口和有效上下文混为一谈；扩窗后不复测短文本能力；忽略位置分布外推与显存/时延。

### R03 `[前沿]` KV Cache、GQA、FlashAttention 与长序列系统

**先修**：B02 复杂度、D13 Attention、D16 性能、F14 推理。

**定义与解析**：自回归生成每步只新增一个 Query；历史 token 的 Key/Value 可以缓存。Multi-Query Attention（MQA）让多个 Query 头共享一组 KV，Grouped-Query Attention（GQA）在质量与缓存大小之间折中。FlashAttention 通过分块和 GPU 内存层级减少 HBM 读写，是**精确注意力算法**，不是稀疏近似。Ring Attention 则解决跨设备分块通信。

**公式/机制**：普通注意力为 $\operatorname{softmax}(QK^\top/\sqrt d)V$，显式分数矩阵为 $O(T^2)$；KV cache 避免每一步重算历史 K/V，但缓存仍随上下文长度线性增长。

**资料定位**：[GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention-2](https://arxiv.org/abs/2307.08691)、[Ring Attention](https://openreview.net/forum?id=WsRHpHH4s0)。

```python
# 代码：可运行。验证最后一个 token 使用缓存 K/V 与完整因果注意力一致。
import math, torch
torch.manual_seed(0)
T, D = 6, 4
x = torch.randn(T, D)
Wq, Wk, Wv = (torch.randn(D, D) for _ in range(3))
q, k, v = x @ Wq, x @ Wk, x @ Wv
score = q @ k.T / math.sqrt(D)
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
full = score.masked_fill(causal, float("-inf")).softmax(-1) @ v
cached_last = (q[-1:] @ k.T / math.sqrt(D)).softmax(-1) @ v
assert torch.allclose(full[-1], cached_last[0], atol=1e-6)
print("cached K/V shape:", tuple(k.shape), tuple(v.shape))
```

**检测题/小实验**：推导 MHA 与 GQA 的 KV 缓存元素数；区分“算术 FLOPs 少”“HBM 访问少”“端到端 latency 低”。

**常见坑**：把 FlashAttention 当成近似注意力；只测单请求吞吐；忽略 batch、序列长度、首 token 延迟和生成 token 延迟。

### R04 `[前沿]` State Space Model 与 Mamba 路线

**先修**：A05 线性系统、D10 递归状态、F05 RNN、F07 Transformer。

**定义与解析**：State Space Model（SSM）用隐藏状态递推压缩历史。Mamba 的关键是让状态更新参数依赖当前输入，即“选择性”地保留或忘记信息，并配合硬件感知的扫描算法获得序列长度近似线性的计算。它是 Transformer 的重要替代/补充路线，不是已普遍取代 Attention 的定论。

**公式/机制**：离散状态更新可写成 $h_t=A(x_t)h_{t-1}+B(x_t)x_t,\; y_t=C(x_t)h_t$。真正 Mamba 还涉及连续到离散参数化、selective scan 和特定门控结构。

**资料定位**：[Mamba 原论文](https://arxiv.org/abs/2312.00752) 与[官方实现/后续论文索引](https://github.com/state-spaces/mamba)。学习时先掌握论文 §2–3，不要从 CUDA kernel 反推数学。

```python
# 代码：机制示意。它展示输入依赖的选择性递推，不是完整 Mamba。
import torch
from torch import nn
torch.manual_seed(0)
T, D = 10, 6
x = torch.randn(T, D)
keep, write = nn.Linear(D, D), nn.Linear(D, D)
h, ys = torch.zeros(D), []
for xt in x:
    a = torch.sigmoid(keep(xt))                 # 本 token 决定保留多少旧状态
    candidate = torch.tanh(write(xt))
    h = a * h + (1 - a) * candidate
    ys.append(h)
y = torch.stack(ys)
assert y.shape == (T, D)
```

**检测题/小实验**：构造“很久前出现一次触发 token”的序列，比较固定衰减递推和输入依赖递推的记忆；说明线性时间不自动等于更低实际延迟。

**常见坑**：把所有 SSM 都叫 Mamba；把理论复杂度直接等同于 GPU 实测；只看超长序列而不测短序列和检索型任务。

### R05 `[前沿]` 推理模型后训练：DPO、GRPO 与可验证奖励

**先修**：F12 SFT、I08 Policy Gradient、I09 Actor–Critic、I10 PPO、O03 奖励模型、O04 RLHF/DPO。

**定义与解析**：现代推理后训练通常从预训练/SFT 策略出发，再用偏好或可验证结果优化。DPO 直接在偏好对上优化策略相对参考模型的对数概率差；GRPO 对同一问题采样一组答案，以组内相对奖励形成优势，避免单独训练价值网络。DeepSeekMath 提出 GRPO；DeepSeek-R1 是大规模推理 RL 的后续案例，而不是 GRPO 的起点。

**公式/机制**：对同一提示的一组奖励 $r_1,\ldots,r_G$，组相对优势常写为 $A_i=(r_i-\bar r)/(s_r+\epsilon)$，再放入带概率比率和 KL 约束的策略目标。

**资料定位**：[InstructGPT Fig.2/§3](https://proceedings.neurips.cc/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html)、[DPO §4](https://arxiv.org/abs/2305.18290)、[DeepSeekMath §4/Algorithm 1](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1 §2–3](https://arxiv.org/abs/2501.12948)。

```python
# 代码：可运行。GRPO/PPO 风格的组相对优势与 clipped surrogate。
import torch
rewards = torch.tensor([[1.0, 0.0, 0.5, 1.0], [0.0, 0.2, 0.8, 1.0]])
adv = (rewards - rewards.mean(1, keepdim=True)) / (
    rewards.std(1, keepdim=True, unbiased=False) + 1e-6)
old_logp = torch.full_like(rewards, -1.0)
new_logp = old_logp + torch.tensor([[.1, -.2, .05, .2], [0., .1, -.1, .3]])
ratio = (new_logp - old_logp).exp()
clipped = ratio.clamp(0.8, 1.2)
loss = -torch.minimum(ratio * adv, clipped * adv).mean()
assert torch.allclose(adv.mean(1), torch.zeros(2), atol=1e-5)
print(float(loss))
```

**检测题/小实验**：当一组答案奖励完全相同时，优势会怎样？设计一个能被格式投机但不能被真实推理解决的奖励，说明 reward hacking。

**常见坑**：把“奖励可验证”误写成推理过程必然正确；只看最终准确率；忽略 KL、采样分布、奖励尺度、数据污染和训练稳定性。

### R06 `[前沿]` Test-time Compute、Self-consistency 与 Verifier Search

**先修**：F11 解码、F13 评测、R05 后训练。

**定义与解析**：测试时计算（test-time compute）通过采样更多候选、延长推理、搜索或使用 verifier 选择答案来换取质量。Self-consistency 对多条推理路径的最终答案投票；best-of-N 用结果或过程验证器排序；自适应策略为不同难度问题分配不同预算。收益依赖基座模型、问题难度、候选多样性和验证器可靠性，不是“想得越久必然越好”。

**机制**：若 $N$ 次候选可近似看作独立、每次正确率为 $p$，奇数 $N$ 的多数投票正确率为 $\sum_{j>(N/2)}\binom Njp^j(1-p)^{N-j}$。真实候选通常相关，因此必须实测相关性、验证器误判率、成本与延迟，不能只套独立公式。

**资料定位**：[Chain-of-Thought](https://proceedings.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)、[Self-consistency](https://openreview.net/forum?id=1PL1NIMMrw)、[过程监督/PRM](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html)、[ICLR 2025 Test-time Compute Scaling](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html)。

```python
# 代码：可运行。一个有噪声的“推理器”，比较单样本和多数投票。
from collections import Counter
import numpy as np
rng = np.random.default_rng(0)
truth = 17 + 8
def noisy_solver():
    return truth if rng.random() < .65 else truth + rng.choice([-2, -1, 1, 2])
single = noisy_solver()
samples = [noisy_solver() for _ in range(21)]
vote = Counter(samples).most_common(1)[0][0]
print({"single": single, "vote": vote, "truth": truth})
assert vote == truth
```

**检测题/小实验**：把单次正确率从 0.65 改成 0.4，重复 1000 次，观察多数投票为何可能失效；再加入一个有系统偏差的 verifier。

**常见坑**：只报告最优样例；忽略成本和延迟；验证器与候选共享同一偏差；把不可见的内部过程当成可靠解释。

### R07 `[前沿]` 统一多模态 Token、Early Fusion 与 Omni 模型

**先修**：F02 Token、F07 Transformer、H01–H05 多模态架构。

**定义与解析**：多模态模型主要有三条接口路线：双编码器对齐到共享空间；用连接器/交叉注意力把视觉或音频特征注入 LLM；把文本、图像、视频、音频离散或连续表示放进统一序列做 early fusion。统一 token 接口便于任意交错输入输出，但不同采样率、长度和信息密度会造成序列爆炸、模态竞争与实时流式难题。

**机制**：一张 $H\times W$ 图像按 $P\times P$ patch 化会产生 $HW/P^2$ 个视觉 token；全注意力的分数矩阵随总序列长度平方增长。双编码器以对比损失对齐全局表示，连接器路线把视觉特征映射到语言模型维度，early fusion 则联合建模混合 token 序列。

**资料定位**：[Chameleon](https://arxiv.org/abs/2405.09818) 与其[官方代码](https://github.com/facebookresearch/chameleon)展示 early-fusion 离散图像 token；[ImageBind](https://arxiv.org/abs/2305.05665)展示共享嵌入；[Qwen2.5-Omni §2](https://arxiv.org/abs/2503.20215)是 2025 年流式音视频到文本/语音案例，属于快速变化前沿。

```python
# 代码：机制示意。把图像 patch 与文本 token 加上模态类型后拼接。
import torch
from torch import nn
torch.manual_seed(0)
D = 8
image_patch = torch.randn(4, D)
text_token = torch.randn(5, D)
type_embed = nn.Embedding(2, D)
seq = torch.cat([
    image_patch + type_embed(torch.tensor(0)),
    text_token + type_embed(torch.tensor(1))
], dim=0)
causal_mask = torch.triu(torch.ones(len(seq), len(seq), dtype=torch.bool), 1)
assert seq.shape == (9, D) and causal_mask.shape == (9, 9)
```

**检测题/小实验**：图像从 16×16 patch 改为 8×8 patch 后 token 数和注意力矩阵大小怎样变化？为实时语音设计 chunk、缓存和中断策略。

**常见坑**：把统一接口等同于统一理解；忽略时间同步和模态缺失；仅凭聊天样例判断 grounding；混淆输入多模态与输出多模态。

### R08 `[前沿]` Diffusion Transformer、Flow Matching 与视频生成

**先修**：G03–G08 生成模型、D13 Transformer、E09 视频。

**定义与解析**：Diffusion Transformer（DiT）用 Transformer 替换扩散模型常见的 U-Net，在潜空间 patch 上预测噪声或速度。Flow Matching 直接回归把基础分布运输到数据分布的时间依赖向量场；扩散路径只是可选概率路径家族之一。视频模型还必须处理时间一致性、3D 压缩、空间—时间注意力和数据配对。

**公式/机制**：在线性条件路径 $x_t=(1-t)x_0+tx_1$ 下，目标速度 $u_t=x_1-x_0$。训练 $v_\theta(x_t,t)$ 最小化 $\mathbb E\|v_\theta-u_t\|^2$，采样时求解 $dx/dt=v_\theta(x,t)$。

**资料定位**：[Flow Matching §3](https://iclr.cc/virtual/2023/poster/11309)、[DiT](https://arxiv.org/abs/2212.09748)、[Video Diffusion Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html)、[Movie Gen 技术报告](https://arxiv.org/abs/2410.13720)。

```python
# 代码：可运行。1D 条件 flow matching 的最小速度回归。
import torch
from torch import nn
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
opt = torch.optim.Adam(net.parameters(), lr=2e-2)
def batch(n=256):
    x0 = torch.randn(n, 1)
    x1 = torch.sign(torch.randn(n, 1)) * 2 + .2 * torch.randn(n, 1)
    t = torch.rand(n, 1)
    return torch.cat([(1-t)*x0+t*x1, t], 1), x1-x0
z, u = batch(); start = ((net(z)-u)**2).mean().item()
for _ in range(300):
    z, u = batch(); loss = ((net(z)-u)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
z, u = batch(); end = ((net(z)-u)**2).mean().item()
assert end < start
```

**检测题/小实验**：用 Euler 1、5、20 步从噪声积分，比较样本分布；解释训练损失下降为何不保证视频物理一致性。

**常见坑**：把 flow matching 说成“另一种去噪扩散”；只展示精选视频；把视觉逼真等同于世界状态和动作可控。

### R09 `[前沿]` 世界模型、VLA 与具身基础模型

**先修**：I02–I11 强化学习主干、H04 VLM、Q05–Q07 机器人策略/VLA/世界模型。

**定义与解析**：世界模型学习 $p(s_{t+1},r_t\mid s_t,a_t)$ 或其潜变量版本，用于 imagined rollout、规划或策略训练。Vision-Language-Action（VLA）模型把视觉、语言和机器人观测映射为离散动作 token 或连续动作块。生成逼真视频、预测可控动力学和产生安全机器人动作是三个不同目标。

**机制**：模型式控制在潜状态中反复执行“编码观测 → 预测动作条件转移与奖励 → imagined rollout → 更新 actor/critic 或搜索动作”。VLA 则学习 $\pi(a_{t:t+H}\mid o_{\le t},\text{instruction})$；部署仍需确定性的可达性、碰撞、同步与安全监控。

**资料定位**：[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2.pdf)展示潜空间 imagined rollout；[RT-2](https://arxiv.org/abs/2307.15818)把动作表示为 token；[OpenVLA](https://arxiv.org/abs/2406.09246)提供开放权重/代码路线；[π0](https://arxiv.org/abs/2410.24164)使用 flow matching 生成连续动作块。

```python
# 代码：可运行。学习一维动作条件动力学，再做一步模型预测控制。
import numpy as np
rng = np.random.default_rng(0)
s = rng.uniform(-2, 2, 200)
a = rng.choice([-1.0, 1.0], 200)
next_s = .9*s + .6*a + rng.normal(0, .03, 200)
X = np.c_[s, a, np.ones_like(s)]
w, *_ = np.linalg.lstsq(X, next_s, rcond=None)
state, goal = -1.2, 1.0
candidates = np.array([-1.0, 1.0])
pred = np.c_[np.full(2, state), candidates, np.ones(2)] @ w
action = candidates[np.argmin((pred-goal)**2)]
assert action == 1.0
```

**检测题/小实验**：让训练数据只覆盖 $s\in[-2,2]$，从 $s=10$ 规划，观察模型误差；为双臂任务列出 VLA 之外仍需的锁、碰撞、同步和接触约束。

**常见坑**：平台有双臂就宣称解决多臂协同；VLM 输出动作名字就视为可执行轨迹；忽略控制频率、闭环反馈、OOD 与安全停止。

### R10 `[前沿]` Agent 协议、状态机与可靠性评测

**先修**：O01–O08、P06–P07。

**定义与解析**：Agent 策略决定何时规划、调用什么工具、如何根据 Observation 更新状态；协议只规定上下文和工具如何被发现与交换。MCP 等协议不能替代规划、权限、回滚和评测。对随机 Agent，单次成功率不足以描述可靠性；应报告重复运行、任务完成、策略遵守、副作用和恢复能力。

**机制**：一个最小闭环是 $\text{state}\rightarrow\text{plan/action}\rightarrow\text{tool}\rightarrow\text{observation}\rightarrow\text{state}'$，并在每次状态变化前后验证参数、权限和后置条件。若单次成功率为 $p$，独立近似下连续 $k$ 次都成功仅为 $p^k$，所以长任务尤其需要分段验收与恢复。

**资料定位**：[ReAct](https://openreview.net/pdf?id=WE_vluYUL-X)看 Thought–Action–Observation；[MCP 固定版本 Tools 规范](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)看协议与权限边界；评测读 [AgentBench](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html)、[SWE-bench](https://www.swebench.com/original.html)和 [τ-bench](https://arxiv.org/abs/2406.12045)。最新协议请另查[当前 MCP specification](https://modelcontextprotocol.io/specification)。

```python
# 代码：可运行。最小工具注册表：schema 验证与 allowlist 属于宿主责任。
TOOLS = {"add": lambda a, b: a + b, "delete": lambda path: "blocked demo"}
SCHEMA = {"add": {"a": int, "b": int}, "delete": {"path": str}}
def call_tool(name, args, allowed=("add",)):
    if name not in allowed:
        raise PermissionError(f"tool {name} is not allowed")
    expected = SCHEMA[name]
    if set(args) != set(expected) or any(not isinstance(args[k], t)
                                         for k, t in expected.items()):
        raise TypeError("schema mismatch")
    return TOOLS[name](**args)
assert call_tool("add", {"a": 2, "b": 3}) == 5
```

**检测题/小实验**：构造三类失败：错误参数、工具超时、工具成功但后置条件不满足；实现有限重试和幂等键。若单次成功率为 $p$，独立近似下连续 $k$ 次都成功为何约为 $p^k$？

**常见坑**：把协议等同于智能；工具输出不验证；无限循环；无最小权限；只看 demo，不做 execution-based evaluation。

### R11 `[前沿]` 如何阅读“当前 GPT/基础模型”产品信息

**先修**：F09 GPT、F10 规模化、F14 推理、P07 评测。

**定义与解析**：产品模型页通常公开输入输出模态、上下文窗口、工具支持、API 参数、价格或退役状态；它不一定公开参数量、训练数据、损失细节或内部推理机制。模型名、上下文长度和排行榜成绩不能反推出架构。百科中的“当前”必须附检索日期，稳定原理应由论文支撑。

**机制**：把陈述分成四层：官方产品事实、公开论文证据、你自己的受控测量、未知项。只有同版本、同提示、同数据切分和同预算的任务评测才能支持横向比较；缺失的参数量、训练数据或内部机制保持“未知”，不从营销名称反推。

**资料定位**：截至本讲义核验日，OpenAI 产品事实只以[官方 OpenAI Models 页面](https://developers.openai.com/api/docs/models)和[Model guidance](https://developers.openai.com/api/docs/guides/latest-model)为准。GPT 的公开学术脉络仍分别阅读 [GPT-1 §3](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)与 [GPT-3 §2–3](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)。

```python
# 代码：可运行。示范基于“任务证据”选择，而不是把模型名当能力证明。
models = {
    "A": {"vision": True,  "tool": True,  "latency_ms": 900, "eval": .86},
    "B": {"vision": False, "tool": True,  "latency_ms": 250, "eval": .80},
}
task = {"vision": False, "tool": True, "max_latency_ms": 400}
eligible = [name for name, m in models.items()
            if (not task["vision"] or m["vision"])
            and (not task["tool"] or m["tool"])
            and m["latency_ms"] <= task["max_latency_ms"]]
assert eligible == ["B"]
```

**检测题/小实验**：做一张“官方声明 / 论文证据 / 自己评测 / 未知”四列表，把一个模型的事实放入正确列；为你的真实任务建立质量、延迟、成本、安全四维 eval。

**常见坑**：引用搜索摘要而不打开官方页；用旧价格和退役模型；把 reasoning effort、产品模式或上下文窗口当作公开训练算法；跨版本比较时更换提示和评测集。

### R12 `[前沿]` 合成数据、知识蒸馏与自举训练

**先修**：D03 损失、D11 微调、F10 数据、R05 推理后训练。

**定义与解析**：知识蒸馏让学生拟合教师的软分布或生成数据；Self-Instruct 类方法由模型生成指令并过滤；推理蒸馏把高质量推理轨迹或答案迁移到较小模型。合成数据可以扩大覆盖面，但会继承教师偏差、产生重复和错误强化，因此过滤、真实数据锚点和独立评测比“生成数量”重要。

**公式/机制**：温度 $T$ 下，教师 $p_T=\operatorname{softmax}(z_T/T)$，学生以 $T^2\operatorname{KL}(p_T\|p_S)$ 学习软目标，并可与真实标签交叉熵混合。

**资料定位**：[Distilling the Knowledge](https://arxiv.org/abs/1503.02531)、[Self-Instruct](https://aclanthology.org/2023.acl-long.754/)、[DeepSeek-R1 §3 Distillation](https://arxiv.org/abs/2501.12948)。

```python
# 代码：可运行。最小软标签蒸馏损失。
import torch
import torch.nn.functional as F
teacher_logits = torch.tensor([[4.0, 1.0, -1.0], [0.5, 2.0, 1.0]])
student_logits = torch.tensor([[2.0, .5, 0.0], [0.2, 1.0, .7]], requires_grad=True)
T = 2.0
p_t = F.softmax(teacher_logits / T, dim=-1)
log_p_s = F.log_softmax(student_logits / T, dim=-1)
distill = F.kl_div(log_p_s, p_t, reduction="batchmean") * T**2
hard = F.cross_entropy(student_logits, torch.tensor([0, 1]))
loss = .7 * distill + .3 * hard
loss.backward()
assert torch.isfinite(student_logits.grad).all()
```

**检测题/小实验**：改变温度并观察教师分布熵；将 10% 合成标签系统性翻转，比较有无真实数据锚点的学生。设计去重、难度和可验证性过滤器。

**常见坑**：把教师答案当真值；训练与评测使用同一教师；只做表面去重；合成数据越多越好；蒸馏后不检查能力边界与安全退化。

---
