# 知识点与术语索引

[学习目录](../README.md) · [系统推导](../08-walkthroughs/README.md) · [论文与源码](../../readings/README.md) · [实践](../../labs/README.md)

> 由章节标题和精读清单自动生成：`python scripts/sync_navigation.py`。标题是检索入口，编号是稳定跳转接口；精读不是该单元的必修前置。

## A. 数学、统计与优化

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| A01 | [集合、函数、关系与逻辑记号](../01-foundations/a-math-statistics-optimization.md#a01) | — |
| A02 | [标量、向量、矩阵与张量](../01-foundations/a-math-statistics-optimization.md#a02) | [Transformer：注意力与编解码器](../../readings/papers/transformer.md) |
| A03 | [向量空间、基、秩、线性映射与子空间](../01-foundations/a-math-statistics-optimization.md#a03) | — |
| A04 | [内积、范数、距离、正交与投影](../01-foundations/a-math-statistics-optimization.md#a04) | — |
| A05 | [特征分解、SVD、正定矩阵与二次型](../01-foundations/a-math-statistics-optimization.md#a05) | [LoRA：低秩任务更新](../../readings/papers/lora.md)、[GCN：图归一化与节点分类](../../readings/papers/gcn.md) |
| A06 | [单变量与多变量微分、偏导和梯度](../01-foundations/a-math-statistics-optimization.md#a06) | — |
| A07 | [Jacobian、Hessian、链式法则与矩阵微积分](../01-foundations/a-math-statistics-optimization.md#a07) | — |
| A08 | [Taylor 展开、局部近似与曲率](../01-foundations/a-math-statistics-optimization.md#a08) | [XGBoost：二阶树提升与系统设计](../../readings/papers/xgboost.md) |
| A09 | [概率公理、条件概率与 Bayes 公式](../01-foundations/a-math-statistics-optimization.md#a09) | — |
| A10 | [随机变量、常见分布、期望与方差](../01-foundations/a-math-statistics-optimization.md#a10) | [Adam：自适应梯度与偏差校正](../../readings/papers/adam.md) |
| A11 | [联合、边缘、条件分布、独立性与协方差](../01-foundations/a-math-statistics-optimization.md#a11) | [DDPM：逐步去噪生成](../../readings/papers/ddpm.md) |
| A12 | [大数定律、中心极限定理与集中现象](../01-foundations/a-math-statistics-optimization.md#a12) | — |
| A13 | [参数估计、MLE、MAP 与 Bayesian 推断](../01-foundations/a-math-statistics-optimization.md#a13) | [VAE：变分下界与可微采样](../../readings/papers/vae.md) |
| A14 | [置信区间、假设检验、Bootstrap 与显著性](../01-foundations/a-math-statistics-optimization.md#a14) | — |
| A15 | [熵、交叉熵、KL 散度、互信息与编码长度](../01-foundations/a-math-statistics-optimization.md#a15) | [DPO：从偏好直接优化策略](../../readings/papers/dpo.md) |
| A16 | [浮点数、数值稳定性、条件数与数值线性代数](../01-foundations/a-math-statistics-optimization.md#a16) | [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md) |
| A17 | [梯度下降、凸性、约束优化与对偶思想](../01-foundations/a-math-statistics-optimization.md#a17) | [Chinchilla：计算最优的模型数据分配](../../readings/papers/chinchilla.md) |
| A18 | [随机优化、经验风险、正则化与泛化](../01-foundations/a-math-statistics-optimization.md#a18) | [Adam：自适应梯度与偏差校正](../../readings/papers/adam.md) |

## B. 编程、数据与实验基础

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| B01 | [Python、NumPy、广播与向量化](../01-foundations/b-programming-data-experiments.md#b01) | — |
| B02 | [数据结构、算法复杂度与内存复杂度](../01-foundations/b-programming-data-experiments.md#b02) | — |
| B03 | [Git、环境、依赖、测试与调试](../01-foundations/b-programming-data-experiments.md#b03) | — |
| B04 | [张量、自动微分、GPU 与计算图工具](../01-foundations/b-programming-data-experiments.md#b04) | [nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md)、[Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md) |
| B05 | [数据清洗、预处理、划分与数据泄漏](../01-foundations/b-programming-data-experiments.md#b05) | — |
| B06 | [指标、基线、受控实验、复现与误差分析](../01-foundations/b-programming-data-experiments.md#b06) | — |

## B+. 经典人工智能：搜索、约束、逻辑与规划

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| B07 | [状态空间、图搜索与问题建模](../01-foundations/b-plus-classical-ai.md#b07) | — |
| B08 | [启发式搜索与 A*](../01-foundations/b-plus-classical-ai.md#b08) | — |
| B09 | [约束满足问题（CSP）](../01-foundations/b-plus-classical-ai.md#b09) | — |
| B10 | [对抗搜索、Minimax 与 Alpha–Beta](../01-foundations/b-plus-classical-ai.md#b10) | — |
| B11 | [命题逻辑、蕴含与规则推理](../01-foundations/b-plus-classical-ai.md#b11) | — |
| B12 | [自动规划、STRIPS 与执行监控](../01-foundations/b-plus-classical-ai.md#b12) | [ReAct：推理行动与外部观察](../../readings/papers/react.md)、[LangGraph：有状态工具流程与恢复](../../readings/projects/langgraph.md) |

## C. 传统机器学习

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| C01 | [监督学习问题、假设空间与经验风险最小化](../01-foundations/c-machine-learning.md#c01) | — |
| C02 | [线性回归、最小二乘、Ridge 与 Lasso](../01-foundations/c-machine-learning.md#c02) | — |
| C03 | [Logistic、Softmax、交叉熵与概率校准](../01-foundations/c-machine-learning.md#c03) | — |
| C04 | [特征缩放、缺失值、类别编码与特征工程](../01-foundations/c-machine-learning.md#c04) | — |
| C05 | [kNN、距离学习与原型方法](../01-foundations/c-machine-learning.md#c05) | — |
| C06 | [Naive Bayes、LDA 与 QDA](../01-foundations/c-machine-learning.md#c06) | — |
| C07 | [决策树、划分准则、剪枝与可解释性](../01-foundations/c-machine-learning.md#c07) | [XGBoost：二阶树提升与系统设计](../../readings/papers/xgboost.md) |
| C08 | [Bagging、随机森林、Boosting 与 GBDT](../01-foundations/c-machine-learning.md#c08) | [XGBoost：二阶树提升与系统设计](../../readings/papers/xgboost.md) |
| C09 | [间隔、SVM 与核方法](../01-foundations/c-machine-learning.md#c09) | — |
| C10 | [k-means、GMM、EM 与层次聚类](../01-foundations/c-machine-learning.md#c10) | — |
| C11 | [PCA、降维、流形学习与可视化](../01-foundations/c-machine-learning.md#c11) | — |
| C12 | [交叉验证、调参、类别不均衡与阈值选择](../01-foundations/c-machine-learning.md#c12) | — |
| C13 | [概率图模型、隐变量推断与 HMM](../01-foundations/c-machine-learning.md#c13) | — |

## C+. 概率函数模型与无梯度优化

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| C14 | [Gaussian Process、核先验与 Bayesian Optimization](../01-foundations/c-plus-probabilistic-black-box.md#c14) | — |
| C15 | [MCMC、变分推断与近似 Bayesian 计算](../01-foundations/c-plus-probabilistic-black-box.md#c15) | [VAE：变分下界与可微采样](../../readings/papers/vae.md) |
| C16 | [进化算法、随机搜索与黑盒优化](../01-foundations/c-plus-probabilistic-black-box.md#c16) | — |

## D. 深度学习共同主干

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| D01 | [感知机、神经元、MLP 与激活函数](../01-foundations/d-deep-learning.md#d01) | — |
| D02 | [计算图、反向传播与自动微分](../01-foundations/d-deep-learning.md#d02) | [ResNet：残差学习与深层优化](../../readings/papers/resnet.md) |
| D03 | [输出分布、任务损失与复合目标](../01-foundations/d-deep-learning.md#d03) | — |
| D04 | [参数初始化、信号传播与梯度稳定性](../01-foundations/d-deep-learning.md#d04) | — |
| D05 | [SGD、Momentum、Adam 与学习率调度](../01-foundations/d-deep-learning.md#d05) | [Adam：自适应梯度与偏差校正](../../readings/papers/adam.md) |
| D06 | [Weight Decay、Dropout、早停与数据增强](../01-foundations/d-deep-learning.md#d06) | — |
| D07 | [BatchNorm、LayerNorm 与 RMSNorm](../01-foundations/d-deep-learning.md#d07) | [LLaMA：现代自回归模型的设计](../../readings/papers/llama.md) |
| D08 | [训练循环、微型过拟合、检查点与系统调试](../01-foundations/d-deep-learning.md#d08) | [nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md) |
| D09 | [Embedding、表征学习与度量学习](../01-foundations/d-deep-learning.md#d09) | [DPR：稠密段落检索](../../readings/papers/dpr.md)、[CLIP：图文对比对齐](../../readings/papers/clip.md) |
| D10 | [残差、门控、递归与状态传递结构](../01-foundations/d-deep-learning.md#d10) | [ResNet：残差学习与深层优化](../../readings/papers/resnet.md)、[Transformer：注意力与编解码器](../../readings/papers/transformer.md) |
| D11 | [迁移学习、微调、冻结策略与参数高效适配](../01-foundations/d-deep-learning.md#d11) | [BERT：双向预训练与迁移](../../readings/papers/bert.md)、[LoRA：低秩任务更新](../../readings/papers/lora.md)、[Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md)、[PEFT：参数高效适配的注入保存与合并](../../readings/projects/peft.md) |
| D12 | [自监督、对比学习与掩码建模](../01-foundations/d-deep-learning.md#d12) | [CLIP：图文对比对齐](../../readings/papers/clip.md) |
| D13 | [Attention、Multi-Head Attention 与 Transformer 公共结构](../01-foundations/d-deep-learning.md#d13) | [Transformer：注意力与编解码器](../../readings/papers/transformer.md)、[LLaMA：现代自回归模型的设计](../../readings/papers/llama.md)、[nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md) |
| D14 | [表达能力、优化偏置、泛化与 Scaling Law](../01-foundations/d-deep-learning.md#d14) | [Chinchilla：计算最优的模型数据分配](../../readings/papers/chinchilla.md) |
| D15 | [不确定性、校准、OOD 与对抗鲁棒性](../01-foundations/d-deep-learning.md#d15) | — |
| D16 | [混合精度、分布式训练、性能剖析、剪枝与量化](../01-foundations/d-deep-learning.md#d16) | [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md)、[vLLM：缓存调度与模型服务](../../readings/projects/vllm.md) |

## E. 计算机视觉

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| E01 | [图像表示、成像、颜色、采样与增强 [核心·成熟]](../02-perception-language/e-computer-vision.md#e01) | — |
| E02 | [滤波、边缘、局部特征、多视几何与传统视觉 [分支·成熟]](../02-perception-language/e-computer-vision.md#e02) | — |
| E03 | [CNN、卷积、等变性与感受野 [核心·成熟]](../02-perception-language/e-computer-vision.md#e03) | [ResNet：残差学习与深层优化](../../readings/papers/resnet.md)、[DQN：经验回放与目标网络](../../readings/papers/dqn.md) |
| E04 | [ResNet、EfficientNet、ConvNeXt 与现代骨干 [核心·成熟]](../02-perception-language/e-computer-vision.md#e04) | [ResNet：残差学习与深层优化](../../readings/papers/resnet.md) |
| E05 | [图像分类、定位、归因与可解释性 [核心·较成熟]](../02-perception-language/e-computer-vision.md#e05) | — |
| E06 | [目标检测与实例分割 [分支·成熟]](../02-perception-language/e-computer-vision.md#e06) | — |
| E07 | [语义分割、深度估计、光流与稠密预测 [分支·成熟]](../02-perception-language/e-computer-vision.md#e07) | — |
| E08 | [ViT、视觉自监督与视觉基础模型 [核心·较成熟；基础模型持续演进]](../02-perception-language/e-computer-vision.md#e08) | — |
| E09 | [视频理解、3D 视觉、点云与 NeRF [分支·较成熟；世界建模解释仍前沿]](../02-perception-language/e-computer-vision.md#e09) | — |

## F. NLP、Transformer 与 LLM

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| F01 | [语言层次、Unicode、规范化与语料 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f01) | — |
| F02 | [分词、子词、BPE、WordPiece 与词表 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f02) | — |
| F03 | [BoW、TF-IDF 与传统文本分类 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f03) | — |
| F04 | [分布式语义、Word2Vec 与静态词向量 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f04) | — |
| F05 | [RNN、LSTM 与 GRU [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f05) | — |
| F06 | [Seq2Seq、编码器—解码器与注意力 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f06) | [Transformer：注意力与编解码器](../../readings/papers/transformer.md)、[T5：统一文本接口与跨度去噪](../../readings/papers/t5.md)、[RAG：检索文档的概率边际化](../../readings/papers/rag.md) |
| F07 | [Transformer、位置编码、Mask 与 KV Cache [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f07) | [Transformer：注意力与编解码器](../../readings/papers/transformer.md)、[LLaMA：现代自回归模型的设计](../../readings/papers/llama.md)、[FlashAttention：分块精确注意力](../../readings/papers/flashattention.md)、[nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md)、[Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md) |
| F08 | [因果、掩码与 Encoder–Decoder 语言建模目标 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f08) | [BERT：双向预训练与迁移](../../readings/papers/bert.md)、[GPT-3：上下文学习与规模](../../readings/papers/gpt3.md)、[T5：统一文本接口与跨度去噪](../../readings/papers/t5.md)、[nanoGPT：经典最小GPT训练器](../../readings/projects/nanogpt.md) |
| F09 | [BERT、GPT 与 T5 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f09) | [BERT：双向预训练与迁移](../../readings/papers/bert.md)、[GPT-3：上下文学习与规模](../../readings/papers/gpt3.md)、[T5：统一文本接口与跨度去噪](../../readings/papers/t5.md)、[DPR：稠密段落检索](../../readings/papers/dpr.md)、[Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md) |
| F10 | [预训练数据、去重、Scaling 与配比 [核心·较成熟；配方快速演进]](../02-perception-language/f-nlp-transformers-llms.md#f10) | [GPT-3：上下文学习与规模](../../readings/papers/gpt3.md)、[Chinchilla：计算最优的模型数据分配](../../readings/papers/chinchilla.md)、[LLaMA：现代自回归模型的设计](../../readings/papers/llama.md) |
| F11 | [Greedy、Beam、采样与约束解码 [核心·成熟]](../02-perception-language/f-nlp-transformers-llms.md#f11) | [Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md)、[vLLM：缓存调度与模型服务](../../readings/projects/vllm.md) |
| F12 | [SFT、指令数据、Chat Template 与 PEFT/LoRA [核心·较成熟；工具接口会变]](../02-perception-language/f-nlp-transformers-llms.md#f12) | [LoRA：低秩任务更新](../../readings/papers/lora.md)、[PEFT：参数高效适配的注入保存与合并](../../readings/projects/peft.md) |
| F13 | [分类、标注、抽取、翻译、摘要与 QA 评测 [核心·成熟；开放生成评测仍不完备]](../02-perception-language/f-nlp-transformers-llms.md#f13) | [GPT-3：上下文学习与规模](../../readings/papers/gpt3.md) |
| F14 | [长上下文、高效 Attention、KV Cache 与 LLM 推理 [前沿；系统快速演进]](../02-perception-language/f-nlp-transformers-llms.md#f14) | [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md)、[PagedAttention：KV缓存分页与共享](../../readings/papers/pagedattention.md)、[vLLM：缓存调度与模型服务](../../readings/projects/vllm.md) |

## G. 生成模型

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| G01 | [显式似然、隐式、潜变量、能量与 Score 范式 [核心·成熟框架]](../02-perception-language/g-generative-models.md#g01) | — |
| G02 | [自回归生成与密度分解 [核心·成熟]](../02-perception-language/g-generative-models.md#g02) | — |
| G03 | [潜变量、变分推断与 ELBO [核心·成熟]](../02-perception-language/g-generative-models.md#g03) | [VAE：变分下界与可微采样](../../readings/papers/vae.md)、[DDPM：逐步去噪生成](../../readings/papers/ddpm.md) |
| G04 | [VAE、层次潜变量与解耦 [核心·成熟；解耦主张需谨慎]](../02-perception-language/g-generative-models.md#g04) | [VAE：变分下界与可微采样](../../readings/papers/vae.md) |
| G05 | [Normalizing Flow [分支·成熟]](../02-perception-language/g-generative-models.md#g05) | — |
| G06 | [GAN [核心·成熟；训练稳定性仍任务相关]](../02-perception-language/g-generative-models.md#g06) | — |
| G07 | [Score Matching、扩散、DDPM 与 SDE [核心·较成熟；采样研究活跃]](../02-perception-language/g-generative-models.md#g07) | [DDPM：逐步去噪生成](../../readings/papers/ddpm.md) |
| G08 | [条件/潜空间扩散、Guidance 与生成评测 [核心·较成熟；模型配方快速变化]](../02-perception-language/g-generative-models.md#g08) | — |

## H. VLM 与多模态

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| H01 | [多模态表示、对齐、融合与 Cross-Attention [核心·较成熟]](../02-perception-language/h-multimodal-vlm.md#h01) | [CLIP：图文对比对齐](../../readings/papers/clip.md) |
| H02 | [CLIP 图文对比预训练 [核心·成熟]](../02-perception-language/h-multimodal-vlm.md#h02) | [CLIP：图文对比对齐](../../readings/papers/clip.md)、[LLaVA：视觉指令微调](../../readings/papers/llava.md) |
| H03 | [Caption、VQA、视觉定位与文档理解 [分支·较成熟]](../02-perception-language/h-multimodal-vlm.md#h03) | — |
| H04 | [视觉编码器—连接器—LLM 的 VLM 架构 [核心·较成熟；具体配方演进]](../02-perception-language/h-multimodal-vlm.md#h04) | [LLaVA：视觉指令微调](../../readings/papers/llava.md)、[Transformers：模型定义与训练生成接口](../../readings/projects/transformers.md) |
| H05 | [多模态指令微调与数据混合 [核心·前沿工程]](../02-perception-language/h-multimodal-vlm.md#h05) | [LLaVA：视觉指令微调](../../readings/papers/llava.md) |
| H06 | [跨模态检索、组合泛化、幻觉与 VLM 评测 [核心·评测前沿]](../02-perception-language/h-multimodal-vlm.md#h06) | [LLaVA：视觉指令微调](../../readings/papers/llava.md) |
| H07 | [统一多模态 Token 与图像、视频、音频生成 [前沿·快速变化]](../02-perception-language/h-multimodal-vlm.md#h07) | — |

## I. 强化学习与 Deep RL

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| I01 | [多臂老虎机与探索—利用【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i01) | — |
| I02 | [MDP、轨迹与环境接口【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i02) | — |
| I03 | [回报、策略与价值函数【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i03) | — |
| I04 | [Bellman 方程与动态规划【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i04) | [DQN：经验回放与目标网络](../../readings/papers/dqn.md) |
| I05 | [Monte Carlo 估计与重要性采样【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i05) | — |
| I06 | [TD、SARSA 与 Q-learning【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i06) | [DQN：经验回放与目标网络](../../readings/papers/dqn.md) |
| I07 | [函数逼近、经验回放与 DQN【稳定基础】](../03-decision-specialties/i-reinforcement-learning.md#i07) | [DQN：经验回放与目标网络](../../readings/papers/dqn.md) |
| I08 | [策略梯度与 REINFORCE【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i08) | — |
| I09 | [Actor–Critic 与 GAE【稳定】](../03-decision-specialties/i-reinforcement-learning.md#i09) | [DeepSeekMath：数学训练与GRPO](../../readings/papers/deepseekmath.md) |
| I10 | [PPO 与受限策略更新【稳定工程基线】](../03-decision-specialties/i-reinforcement-learning.md#i10) | [InstructGPT：人类反馈与策略优化](../../readings/papers/instructgpt.md)、[DeepSeekMath：数学训练与GRPO](../../readings/papers/deepseekmath.md)、[TRL：监督偏好与在线策略训练](../../readings/projects/trl.md) |
| I11 | [连续控制与 Soft Actor-Critic【稳定基线】](../03-decision-specialties/i-reinforcement-learning.md#i11) | — |
| I12 | [离线强化学习与分布外动作【演进中】](../03-decision-specialties/i-reinforcement-learning.md#i12) | — |
| I13 | [模仿学习：BC、DAgger 与 GAIL【稳定基础，扩展活跃】](../03-decision-specialties/i-reinforcement-learning.md#i13) | — |
| I14 | [多智能体强化学习与 CTDE【演进中】](../03-decision-specialties/i-reinforcement-learning.md#i14) | — |

## J. 图神经网络

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| J01 | [图表示与消息传递【稳定】](../03-decision-specialties/j-graph-neural-networks.md#j01) | [GCN：图归一化与节点分类](../../readings/papers/gcn.md) |
| J02 | [图卷积网络 GCN【稳定】](../03-decision-specialties/j-graph-neural-networks.md#j02) | [GCN：图归一化与节点分类](../../readings/papers/gcn.md) |
| J03 | [图注意力 GAT【稳定】](../03-decision-specialties/j-graph-neural-networks.md#j03) | — |
| J04 | [图级读出、批处理与不变性【稳定】](../03-decision-specialties/j-graph-neural-networks.md#j04) | — |
| J05 | [链接预测、负采样与归纳泛化【稳定基础】](../03-decision-specialties/j-graph-neural-networks.md#j05) | — |

## K. 语音与音频

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| K01 | [波形、采样、频谱与混叠【稳定】](../03-decision-specialties/k-speech-audio.md#k01) | — |
| K02 | [分帧、STFT、Mel 频谱与 MFCC【稳定】](../03-decision-specialties/k-speech-audio.md#k02) | — |
| K03 | [CTC 与端到端语音识别【稳定基础】](../03-decision-specialties/k-speech-audio.md#k03) | — |
| K04 | [自监督语音表示：wav2vec 2.0【演进中】](../03-decision-specialties/k-speech-audio.md#k04) | — |
| K05 | [TTS、声码器与音频评估【成熟组件，生成前沿活跃】](../03-decision-specialties/k-speech-audio.md#k05) | — |

## L. 时间序列

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| L01 | [时间索引、窗口化与无泄漏切分【稳定】](../03-decision-specialties/l-time-series.md#l01) | — |
| L02 | [朴素、季节朴素与指数平滑基线【稳定】](../03-decision-specialties/l-time-series.md#l02) | — |
| L03 | [自回归、平稳性与滚动预测【稳定】](../03-decision-specialties/l-time-series.md#l03) | — |
| L04 | [深度时序：RNN、TCN 与 Transformer【成熟组件，选型演进中】](../03-decision-specialties/l-time-series.md#l04) | — |
| L05 | [概率预测、区间校准与异常检测【稳定原则，模型演进中】](../03-decision-specialties/l-time-series.md#l05) | — |

## M. 因果推断

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| M01 | [结构因果模型、干预与反事实【稳定理论】](../03-decision-specialties/m-causal-inference.md#m01) | — |
| M02 | [潜在结果、随机试验与 ATE【稳定理论】](../03-decision-specialties/m-causal-inference.md#m02) | — |
| M03 | [混杂、后门准则与调整【稳定理论】](../03-decision-specialties/m-causal-inference.md#m03) | — |
| M04 | [倾向得分、重加权与双重稳健【稳定方法】](../03-decision-specialties/m-causal-inference.md#m04) | — |
| M05 | [自然实验：工具变量与双重差分【稳定设计，假设强】](../03-decision-specialties/m-causal-inference.md#m05) | — |

## N. 推荐、搜索与检索

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| N01 | [候选、排序与离线检索指标【稳定】](../03-decision-specialties/n-recommendation-search-retrieval.md#n01) | — |
| N02 | [隐式反馈、时间切分与曝光偏差【稳定问题，纠偏演进中】](../03-decision-specialties/n-recommendation-search-retrieval.md#n02) | — |
| N03 | [协同过滤、矩阵分解与 BPR【稳定】](../03-decision-specialties/n-recommendation-search-retrieval.md#n03) | — |
| N04 | [双塔与稠密检索【稳定架构，训练技巧演进中】](../03-decision-specialties/n-recommendation-search-retrieval.md#n04) | [DPR：稠密段落检索](../../readings/papers/dpr.md)、[RAG：检索文档的概率边际化](../../readings/papers/rag.md) |
| N05 | [Learning to Rank 与重排【稳定基础】](../03-decision-specialties/n-recommendation-search-retrieval.md#n05) | — |
| N06 | [ANN、混合检索与线上实验【成熟系统，索引演进中】](../03-decision-specialties/n-recommendation-search-retrieval.md#n06) | [DPR：稠密段落检索](../../readings/papers/dpr.md) |

## O. LLM 后训练、RAG 与 Agent

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| O01 | [指令数据与监督微调 SFT【稳定流程】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o01) | [InstructGPT：人类反馈与策略优化](../../readings/papers/instructgpt.md)、[TRL：监督偏好与在线策略训练](../../readings/projects/trl.md) |
| O02 | [参数高效微调 LoRA【稳定基础，变体活跃】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o02) | [LoRA：低秩任务更新](../../readings/papers/lora.md)、[PEFT：参数高效适配的注入保存与合并](../../readings/projects/peft.md) |
| O03 | [偏好数据与奖励模型【稳定框架，标注科学活跃】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o03) | [InstructGPT：人类反馈与策略优化](../../readings/papers/instructgpt.md)、[DPO：从偏好直接优化策略](../../readings/papers/dpo.md)、[DeepSeekMath：数学训练与GRPO](../../readings/papers/deepseekmath.md)、[TRL：监督偏好与在线策略训练](../../readings/projects/trl.md) |
| O04 | [RLHF、KL 约束与 DPO【成熟主线，快速演进】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o04) | [InstructGPT：人类反馈与策略优化](../../readings/papers/instructgpt.md)、[DPO：从偏好直接优化策略](../../readings/papers/dpo.md)、[DeepSeekMath：数学训练与GRPO](../../readings/papers/deepseekmath.md)、[TRL：监督偏好与在线策略训练](../../readings/projects/trl.md) |
| O05 | [RAG：切分、索引与检索【稳定架构，配方演进中】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o05) | [DPR：稠密段落检索](../../readings/papers/dpr.md)、[RAG：检索文档的概率边际化](../../readings/papers/rag.md) |
| O06 | [Grounded RAG、引用与端到端评估【演进中】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o06) | [RAG：检索文档的概率边际化](../../readings/papers/rag.md) |
| O07 | [工具、资源与 Agent 协议【协议稳定化中】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o07) | [ReAct：推理行动与外部观察](../../readings/papers/react.md)、[LangGraph：有状态工具流程与恢复](../../readings/projects/langgraph.md) |
| O08 | [Agent 策略：ReAct、规划、记忆与多 Agent【研究前沿】](../04-systems-agents-robotics/o-llm-posttraining-rag-agents.md#o08) | [ReAct：推理行动与外部观察](../../readings/papers/react.md)、[LangGraph：有状态工具流程与恢复](../../readings/projects/langgraph.md) |

## P. MLOps、安全与评测

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| P01 | [环境、随机性与可复现运行【稳定工程原则】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p01) | — |
| P02 | [数据版本、血缘与数据契约【稳定工程原则】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p02) | — |
| P03 | [实验追踪、基线与持续测试【稳定工程原则】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p03) | — |
| P04 | [打包、推理接口与服务契约【稳定工程原则】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p04) | [PagedAttention：KV缓存分页与共享](../../readings/papers/pagedattention.md)、[vLLM：缓存调度与模型服务](../../readings/projects/vllm.md) |
| P05 | [导出、量化与性能剖析【成熟技术，硬件配方演进中】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p05) | [FlashAttention：分块精确注意力](../../readings/papers/flashattention.md)、[PagedAttention：KV缓存分页与共享](../../readings/papers/pagedattention.md)、[vLLM：缓存调度与模型服务](../../readings/projects/vllm.md) |
| P06 | [监控、漂移与反馈闭环【稳定原则，检测方法演进中】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p06) | — |
| P07 | [评测集、统计不确定性与回归决策【稳定原则】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p07) | — |
| P08 | [威胁建模、对抗输入与 LLM 工具安全【安全实践演进中】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p08) | [LangGraph：有状态工具流程与恢复](../../readings/projects/langgraph.md) |
| P09 | [治理、隐私、公平与模型卡【稳定框架，法规会变化】](../04-systems-agents-robotics/p-mlops-safety-evaluation.md#p09) | — |

## Q. 具身智能与机器人

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| Q01 | [机器人系统分层、坐标系与 SE(3)【稳定】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q01) | — |
| Q02 | [感知、标定与状态估计【稳定基础，学习感知演进中】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q02) | — |
| Q03 | [正运动学、逆运动学与可达性【稳定】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q03) | — |
| Q04 | [碰撞检测、运动规划与轨迹控制【稳定核心】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q04) | — |
| Q05 | [行为克隆、动作分块与 Diffusion Policy【演进中】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q05) | — |
| Q06 | [机器人 RL、仿真到现实与安全探索【演进中】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q06) | — |
| Q07 | [VLA、世界模型与分层技能【研究前沿】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q07) | — |
| Q08 | [具身 Agent、多机器人协同与安全执行【研究前沿】](../04-systems-agents-robotics/q-embodied-ai-robotics.md#q08) | — |

## R. 2024–2026 前沿技术地图

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| R01 | [稀疏 Mixture of Experts（MoE）与路由](../05-frontier/r-frontier-2024-2026.md#r01) | — |
| R02 | [RoPE、位置外推与长上下文](../05-frontier/r-frontier-2024-2026.md#r02) | — |
| R03 | [KV Cache、GQA、FlashAttention 与长序列系统](../05-frontier/r-frontier-2024-2026.md#r03) | — |
| R04 | [State Space Model 与 Mamba 路线](../05-frontier/r-frontier-2024-2026.md#r04) | — |
| R05 | [推理模型后训练：DPO、GRPO 与可验证奖励](../05-frontier/r-frontier-2024-2026.md#r05) | — |
| R06 | [Test-time Compute、Self-consistency 与 Verifier Search](../05-frontier/r-frontier-2024-2026.md#r06) | — |
| R07 | [统一多模态 Token、Early Fusion 与 Omni 模型](../05-frontier/r-frontier-2024-2026.md#r07) | — |
| R08 | [Diffusion Transformer、Flow Matching 与视频生成](../05-frontier/r-frontier-2024-2026.md#r08) | — |
| R09 | [世界模型、VLA 与具身基础模型](../05-frontier/r-frontier-2024-2026.md#r09) | — |
| R10 | [Agent 协议、状态机与可靠性评测](../05-frontier/r-frontier-2024-2026.md#r10) | — |
| R11 | [如何阅读“当前 GPT/基础模型”产品信息](../05-frontier/r-frontier-2024-2026.md#r11) | — |
| R12 | [合成数据、知识蒸馏与自举训练](../05-frontier/r-frontier-2024-2026.md#r12) | — |

## S. 综合项目与验收路线

| 编号 | 知识点 / 术语 | 延伸精读 |
|---|---|---|
| S01 | [两周基础诊断](../06-projects/s-projects-assessment.md#s01) | — |
| S02 | [经典 ML：表格预测](../06-projects/s-projects-assessment.md#s02) | — |
| S03 | [深度学习：可复用训练模板](../06-projects/s-projects-assessment.md#s03) | — |
| S04 | [CV：稠密预测与鲁棒性](../06-projects/s-projects-assessment.md#s04) | — |
| S05 | [NLP/LLM：从 TF-IDF 到 Mini-GPT](../06-projects/s-projects-assessment.md#s05) | — |
| S06 | [生成模型：二维分布实验室](../06-projects/s-projects-assessment.md#s06) | — |
| S07 | [VLM：图文检索与零样本分类](../06-projects/s-projects-assessment.md#s07) | — |
| S08 | [强化学习：GridWorld → 经典控制](../06-projects/s-projects-assessment.md#s08) | — |
| S09 | [RAG/Agent：可证伪的工具系统](../06-projects/s-projects-assessment.md#s09) | — |
| S10 | [综合研究项目](../06-projects/s-projects-assessment.md#s10) | — |
| S11 | [项目统一评分表](../06-projects/s-projects-assessment.md#s11) | — |
