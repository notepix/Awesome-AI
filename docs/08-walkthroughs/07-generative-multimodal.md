# 生成与多模态：从分布、去噪到图文对齐

[导读](README.md) · [G 生成模型](../02-perception-language/g-generative-models.md) · [H 多模态](../02-perception-language/h-multimodal-vlm.md)

## 1. 生成模型首先回答概率问题

分类模型常学 $p(y\mid x)$；生成模型可以学 $p(x)$ 或带条件的 $p(x\mid c)$。观察到一幅清晰图像并不能说明模型覆盖了数据分布，生成模型还可能只重复少数训练样本。因此要区分样本质量、覆盖度、多样性和记忆。

自回归把联合分布分解为条件概率乘积，潜变量模型引入未观察的 $z$，扩散模型定义逐步加噪与反向生成过程。它们比较的是建模和计算方式，不是简单的“谁画得更好”。

## 2. VAE 的 ELBO 从哪里来

设生成模型 $p_\theta(x,z)=p(z)p_\theta(x\mid z)$，近似后验为 $q_\phi(z\mid x)$。插入这个分布并利用Jensen不等式，可得到

$$\log p_\theta(x)\geq
\mathbb E_q[\log p_\theta(x\mid z)]-D_{KL}(q_\phi(z\mid x)\Vert p(z))=\operatorname{ELBO}.$$

也可由恒等式理解差距：$\log p_\theta(x)-\operatorname{ELBO}=D_{KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))\geq0$。第一项鼓励从潜变量重构数据，第二项约束后验与先验的差异；提高ELBO同时涉及模型拟合和推断近似。

对对角Gaussian后验，令 $z=\mu+\sigma\odot\epsilon$、$\epsilon\sim\mathcal N(0,I)$，把随机性移到与参数无关的噪声，梯度可以穿过 $\mu,\sigma$。一维后验 $\mathcal N(1,1)$ 相对标准Gaussian的KL为 $\frac12(1^2+1-\log1-1)=0.5$。这是分布差异，不能把它当成重构像素误差。

重构项具体是Bernoulli、Gaussian还是其他似然，需要与数据假设匹配。过强的解码器可能忽略 $z$，形成posterior collapse；好看的重构图不证明潜变量有用。

## 3. DDPM 的训练不是直接从纯噪声生成

前向过程可写为 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$。每次训练抽取一个时间步和噪声，网络根据 $x_t,t$ 预测噪声。常用简化目标为

$$L=\mathbb E_{x_0,t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2.$$

令 $\bar\alpha_t=0.25,x_0=2,\epsilon=-1$，得到 $x_t=1-\sqrt{0.75}\approx0.134$。已知噪声时可按代数关系恢复 $x_0$；真实生成时没有原始 $x_0$，必须从噪声出发，反复使用网络预测构建反向更新。只计算一次MSE反传，不等于实现完整扩散采样器。

时间条件不能省略，同一个带噪数值在不同噪声级别可能对应不同任务。噪声表、反向方差和采样算法都会影响结果；减少采样步数时也应比较质量变化和计算预算。

CFG把有条件和无条件预测组合，例如 $\epsilon_u+s(\epsilon_c-\epsilon_u)$。增大 $s$ 往往强化条件，却可能降低多样性或引入失真，它不是无代价改善。

## 4. CLIP：配对目标如何形成共享空间

图像编码器和文本编码器分别输出归一化向量 $u_i,v_i\in\mathbb R^d$。相似度 $S_{ij}=u_i^Tv_j/\tau$ 构成 $B\times B$ 矩阵，正确配对通常在对角线上。图搜文损失是对每行做交叉熵，文搜图对每列做交叉熵，再取平均。

两个样本的logits若为 $[2,0]$，正确配对概率约为0.881，对应损失约0.127。降低温度会使分布更尖，但错误配对也会受到更强惩罚。重复描述或语义相同图像可能成为false negative，因此batch构成是学习目标的一部分。

检索是比较向量；零样本分类把类别写成文本提示再比较；生成式VLM则需要输出语言token。这三件事不相同。合成数据训练的小双塔若从未见过某个类别词，也不能被直接称为开放词汇模型。

## 5. 从 CLIP 到生成式 VLM

一个常见结构是视觉encoder → connector → LLM。图像切为patch或其他视觉token，connector把视觉表示映射到语言模型可接收的隐藏空间，再与文本共同处理。训练可能分为对齐连接器和多模态指令微调，但具体冻结策略由方法决定。

有视觉token并不保证回答真正依赖图像。模型可能依靠语言先验猜答案，需要图像替换、问题扰动和细粒度grounding测试。图文匹配准确也不能证明计数、空间关系或文档细节正确。

## 6. 检测题与解析

1. **VAE中重构项更好而总ELBO更差可能吗？** 可以，KL项的增加可能超过重构改善；必须同时看两项和它们的单位、权重。
2. **DDPM训练抽一个t，生成也只需一次预测吗？** 不必然。标准反向过程需要多个步骤；一步模型需要额外方法和相应训练，不能从随机t训练直接推出。
3. **图文检索测试把同一形状的几乎相同图片随机分两组有什么问题？** 可能只测到近重复识别。应按场景或属性组合划分，说明测试是已见组合的新外观，还是未见组合泛化。

## 7. 精读与实践

先读[VAE](../../readings/papers/vae.md)、[DDPM](../../readings/papers/ddpm.md)，再读[CLIP](../../readings/papers/clip.md)、[LLaVA](../../readings/papers/llava.md)。[CMU 11-777课程](https://multicomp.cs.cmu.edu/mmml-course/fall2023/)按表示、对齐、融合与生成组织多模态问题。[图文检索实验](../../labs/multimodal/README.md)提供完整小数据双塔流程，预训练CLIP扩展使用英文描述并单独记录结果。
