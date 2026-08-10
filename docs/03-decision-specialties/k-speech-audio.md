# K. 语音与音频


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](j-graph-neural-networks.md) · [下一章 →](l-time-series.md)

---

## 知识单元

### K01 波形、采样、频谱与混叠【稳定】
- **先修**：正弦、复数、傅里叶变换、NumPy。
- **定义与解析**：数字音频是按采样率 $f_s$ 离散化的振幅序列；超过 Nyquist 频率 $f_s/2$ 的成分会折叠成低频，后续模型无法凭空恢复。
- **公式/机制**：$x[n]=x(n/f_s)$，DFT 为 $X[k]=\sum_nx[n]e^{-j2\pi kn/N}$；采样前需抗混叠滤波。
- **资料**：SciPy [FFT 教程：Discrete Fourier transforms](https://docs.scipy.org/doc/scipy/tutorial/fft.html#discrete-fourier-transforms)；`scipy.signal` [STFT API 与 COLA/NOLA 注释](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html)。
- **最小代码（可执行）**：
```python
import numpy as np
fs=8000; t=np.arange(fs)/fs
x=np.sin(2*np.pi*1000*t)+.5*np.sin(2*np.pi*5500*t)
freq=np.fft.rfftfreq(len(x),1/fs)
peak=freq[np.argsort(abs(np.fft.rfft(x)))[-2:]]
print(np.sort(peak))                         # 5500 Hz 混叠到 2500 Hz
assert set(np.round(peak).astype(int))=={1000,2500}
```
- **检测/实验**：用 8 kHz 与 16 kHz 采样同一 5.5 kHz 正弦，解释两个频谱峰不同的原因。
- **常见坑**：把采样率当 bit rate；忘记双边频谱对实信号对称；音频归一化后仍发生播放端 clipping。

### K02 分帧、STFT、Mel 频谱与 MFCC【稳定】
- **先修**：K01、窗函数、对数尺度。
- **定义与解析**：STFT 假设短窗内近似平稳；Mel 滤波组把线性频率压缩为感知尺度，log-mel 保留时频结构，MFCC 再用 DCT 压缩谱包络。
- **公式/机制**：$X(m,k)=\sum_nx[n]w[n-mH]e^{-j2\pi kn/N}$，$m(f)=2595\log_{10}(1+f/700)$。
- **资料**：librosa [`stft` 参数中心化/padding](https://librosa.org/doc/latest/generated/librosa.stft.html)；[`melspectrogram`](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html) 与 [`mfcc`](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html)。
- **最小代码（可执行的简化 Mel 滤波组）**：
```python
import numpy as np
fs=16000; x=np.random.default_rng(0).normal(size=400)
p=abs(np.fft.rfft(x*np.hanning(len(x))))**2
f=np.fft.rfftfreq(len(x),1/fs); mel=2595*np.log10(1+f/700)
centers=np.linspace(mel.min(),mel.max(),12); width=centers[1]-centers[0]
bank=np.maximum(0,1-abs(mel[None,:]-centers[:,None])/width)
feat=np.log(bank@p+1e-6)
assert feat.shape==(12,) and np.isfinite(feat).all()
```
- **检测/实验**：改变 hop length 但保持窗长，预测时间分辨率、帧数和计算量；比较 log 前后动态范围。
- **常见坑**：训练/推理采样率不一致；对功率谱和幅度谱使用同一 dB 系数；把 MFCC 当可逆波形表示。

### K03 CTC 与端到端语音识别【稳定基础】
- **先修**：K02、序列概率、动态规划、softmax。
- **定义与解析**：CTC 在不知道帧—字符对齐时，对所有折叠后等于目标文本的路径求和；blank 和重复规则允许输入帧数大于输出长度。
- **公式/机制**：$p(y\mid x)=\sum_{\pi:B(\pi)=y}\prod_t p(\pi_t\mid x)$；前向—后向算法避免枚举路径。
- **资料**：Graves et al. [ICML 2006 §2，式 (1)–(7)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)；PyTorch [`CTCLoss` shape 与 zero_infinity](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)。
- **最小代码（可执行的 CTC 折叠）**：
```python
blank=0
def collapse(path):
    out=[]; prev=None
    for token in path:
        if token!=blank and token!=prev: out.append(token)
        prev=token
    return out
assert collapse([0,1,1,0,2,2,0])==[1,2]
assert collapse([1,0,1])==[1,1]
print(collapse([0,3,3,0,4]))
```
- **检测/实验**：解释路径 `[a,a]`、`[a,blank,a]` 的折叠差异；检查 target length 超过 input length 时 loss。
- **常见坑**：blank id 与词表 id 冲突；输入维度 T/N/C 排错；用字符准确率代替标准 WER/CER 且不统一文本归一化。

### K04 自监督语音表示：wav2vec 2.0【演进中】
- **先修**：K02、Transformer、对比学习、掩码建模。
- **定义与解析**：wav2vec 2.0 在原始波形编码后遮蔽潜表示，通过从量化候选中识别真实目标预训练，再用少量转写微调；表示强不等于自动适配任意口音和语言。
- **公式/机制**：对比损失 $-\log\frac{e^{\operatorname{sim}(c_t,q_t)/\kappa}}{\sum_{\tilde q}e^{\operatorname{sim}(c_t,\tilde q)/\kappa}}$，另加码本多样性项。
- **资料**：Baevski et al. [NeurIPS 2020 §2，式 (3)–(5)](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)；fairseq [wav2vec 2.0 README](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)。
- **最小代码（可执行 InfoNCE 示意）**：
```python
import torch
torch.manual_seed(0); q=torch.randn(8,16); c=q+.1*torch.randn_like(q)
q=torch.nn.functional.normalize(q,dim=1); c=torch.nn.functional.normalize(c,dim=1)
logits=c@q.T/.1; target=torch.arange(8)
loss=torch.nn.functional.cross_entropy(logits,target)
r1=(logits.argmax(1)==target).float().mean()
print(loss.item(),r1.item()); assert r1>.75
```
- **检测/实验**：增大温度与负样本数，观察 loss 和 Recall@1；解释预训练验证损失低为何不保证下游 WER 低。
- **常见坑**：预训练集与测试说话人重叠；微调忘记 attention mask；把冻结特征抽取与端到端微调结果直接比较。

### K05 TTS、声码器与音频评估【成熟组件，生成前沿活跃】
- **先修**：K01–K04、seq2seq、生成模型。
- **定义与解析**：典型 TTS 先由文本生成时长/基频/谱表示，再由声码器生成波形；端到端系统仍应分别检查可懂度、音色、韵律、实时率与安全水印。
- **公式/机制**：自回归 $p(x)=\prod_t p(x_t\mid x_{<t},c)$；并行/扩散声码器改用并行变换或逐步去噪。
- **资料**：Tacotron 2 [§2，Fig.1](https://arxiv.org/abs/1712.05884)；WaveNet [§2.1–2.2](https://arxiv.org/abs/1609.03499)；ITU-T [P.808 众包主观语音质量](https://www.itu.int/rec/T-REC-P.808/en)。
- **最小代码（可执行教学合成；不是 TTS/声码器）**：
```python
import numpy as np
fs=16000; notes=[220,330,440]; dur=.12
parts=[]
for f in notes:
    t=np.arange(int(fs*dur))/fs
    env=np.minimum(1,t/.01)*np.minimum(1,(dur-t)/.02)
    parts.append(.2*env*np.sin(2*np.pi*f*t))
wave=np.concatenate(parts)
print(len(wave)/fs,abs(wave).max())
assert abs(wave).max()<=1
```
- **检测/实验**：同一文本至少测 MOS/偏好、ASR-WER、说话人相似度与 real-time factor；上面代码只验证采样和包络，不能证明 TTS 能力。
- **常见坑**：只挑选最好音频；训练语音未经授权克隆；忽略静音、响度归一、长文本和流式延迟。
