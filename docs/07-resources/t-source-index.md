# T. 总索引与资料使用说明


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](../06-projects/s-projects-assessment.md)

---

## 知识单元

### T01 一手资料总入口

- 数学：[Mathematics for Machine Learning](https://mml-book.github.io/)——Ch2 线代、Ch4 矩阵分解、Ch5 微积分、Ch6 概率、Ch7 优化。
- 传统 ML：[CS229 公开历史材料](https://cs229.stanford.edu/materials.html-full)——当季材料可能需校内登录，因此使用该公开索引；实践可配 [ISLP Python](https://www.statlearning.com/)。
- 统一 DL 教材：[Dive into Deep Learning](https://d2l.ai/)——数学、代码和练习在同一页面。
- 经典 AI：[Berkeley CS188 在线教材](https://inst.eecs.berkeley.edu/~cs188/textbook/)——搜索、CSP、博弈、MDP、概率推断。
- CV：[CS231n 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html)及[作业](https://cs231n.stanford.edu/2025/assignments.html)。公开 slides/notes 可读，当季课堂视频不作为必要依赖。
- NLP/LLM：[CS224N 当前主页](https://web.stanford.edu/class/cs224n/)与 [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)。
- RL：[Sutton & Barto 第二版](http://incompleteideas.net/book/the-book-2nd.html)、[Berkeley CS185/285](https://rail.eecs.berkeley.edu/deeprlcourse/)和 [Gymnasium Tutorials](https://gymnasium.farama.org/tutorials/)。
- VLM：[Hugging Face Multimodal Unit](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/pre-intro)作为入门导航，结论回到各原论文。
- 评测与安全：[HELM](https://crfm.stanford.edu/helm/index.html)、[NIST Generative AI Profile](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf)、[OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf)。

### T02 资料可信度顺序

同一结论出现冲突时，按以下顺序处理：

1. 数学定义、正式标准或经过同行评审的原论文；
2. 作者官方技术报告、模型卡、官方代码与文档；
3. 大学课程原站和公开教材；
4. 高质量复现；
5. 博客、视频、二手总结；
6. 排行榜截图、社交媒体和营销演示。

产品价格、模型名称、API 限制和排行榜属于高漂移信息，必须重新打开官方页面核验；原理性定义不要从产品页推断。

### T03 这份百科的边界

“AI 的全部”不存在稳定终点。本讲义覆盖共同主干和主要现代分支，但测度论、随机微分方程、Bayesian nonparametrics、编译器/芯片设计、法律法规细则以及医疗/金融/生物等行业知识只能作为继续深挖方向。遇到新技术时，用下面五问把它挂回已有知识树：

1. 它改变的是数据、表示、目标函数、优化、推理还是系统接口？
2. 它依赖哪些已有单元？
3. 与最简单基线相比，真正新增了什么？
4. 论文在哪些数据、算力和指标上验证？哪些没有验证？
5. 能否用一个小实验复现其核心机制或证伪宣传？

只要能回答这五问，新论文就不会成为孤立名词。

### T04 课程、教材与本仓库的对应关系

本表是2026-10-03整理的学习入口：核对官方课程、目录和公开材料范围，不表示每门课所有视频或整本教材都已逐页读完。固定年份用于保持可定位性；课程主页更新时，优先保留所读历史版本。

| 领域与本地入口 | 主课程或开放讲义 | 教材及精读位置 | 公开程度与使用方式 |
|---|---|---|---|
| A 数学 | [MIT OCW 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | [Mathematics for Machine Learning](https://mml-book.github.io/)，Ch2–7：线代、几何、分解、微积分、概率、优化 | 官方公开课程；作者提供教材PDF，按先修选读 |
| B+ 经典AI | [CS188 Spring 2025](https://inst.eecs.berkeley.edu/~cs188/archive/sp25/) | [CS188在线教材](https://inst.eecs.berkeley.edu/~cs188/textbook/)，search/CSP/games/MDP | 公开讲义和教材；AIMA作者站不是整本免费版 |
| C 机器学习 | [CS229历史公开课程](https://see.stanford.edu/Course/CS229) | [ISLP](https://www.statlearning.com/)，回归、分类、重采样、正则化、树；书籍版本以作者页为准 | 公开视频与作者PDF；不要把校内当季录像当必需入口 |
| C+ 概率模型 | [CS228笔记](https://ermongroup.github.io/cs228-notes/) | [Probabilistic Machine Learning](https://probml.github.io/pml-book/book1.html)，概率、推断与潜变量章节 | 课程笔记和作者草稿；标明草稿与正式出版物差异 |
| D 深度学习 | [MIT 6.S191](https://introtodeeplearning.com/) | [D2L 1.0.3英文站](https://d2l.ai/)，Ch5 MLP、Ch7–8 CNN、Ch9–10 RNN、Ch11 Attention、Ch12优化；[Deep Learning](https://www.deeplearningbook.org/)，Ch6/8 | 公开材料，例程库版本可能变化 |
| E 视觉 | [CS231n 2025](https://cs231n.stanford.edu/2025/schedule.html) | [Szeliski第2版](https://szeliski.org/Book/)，图像形成、特征、对齐与三维 | 课件公开；当季录像和教材下载条件分别确认 |
| F 语言模型 | [CS224N Winter 2026](https://web.stanford.edu/class/cs224n/)、[CS336 Spring 2025归档](https://cs336.stanford.edu/spring2025/) | [SLP3草稿](https://web.stanford.edu/~jurafsky/slp3/)，语言模型、Transformer、预训练；CS336 assignments 1–5 | CS224N公开历史录像与当季校内录像区分；CS336完整作业需要额外算力 |
| G 生成模型 | [CS236](https://deepgenerativemodels.github.io/) | [VAE](../../readings/papers/vae.md)、[DDPM](../../readings/papers/ddpm.md)精读及原文 | 以公开课程和原论文解释目标，不用图片观感代替评估 |
| H 多模态 | [CMU 11-777 Fall 2023](https://multicomp.cs.cmu.edu/mmml-course/fall2023/) | representations/alignment/fusion/generation讲次；[CLIP](../../readings/papers/clip.md)、[LLaVA](../../readings/papers/llava.md) | 历史公开课件和部分录像；研究综述不称为正式教材 |
| I 强化学习 | [CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | [Sutton与Barto第二版](http://incompleteideas.net/book/the-book-2nd.html)，Ch2–6、9、13 | 公开课程及作者入口；特定页面若不可用应记录而非编造可访问性 |
| J 图学习 | [CS224W](https://web.stanford.edu/class/cs224w/) | [Hamilton GRL](https://www.cs.mcgill.ca/~wlh/grl_book/)，图嵌入与消息传递 | 作者公开的是出版前草稿 |
| K 语音 | [李宏毅DLHLP 2020](https://speech.ee.ntu.edu.tw/~hylee/dlhlp/2020-spring.php) | SLP3语音表征、ASR、TTS章节 | 中文课程公开；用较新教材补内容，不能称2020课为最新 |
| L 时间序列 | [FPP3开放教材](https://otexts.com/fpp3/) | Ch2时间序列图、Ch3分解、Ch5预测基础、Ch8 ETS、Ch9 ARIMA | 作者HTML教材；可将R例子的原理对应Python实践 |
| M 因果 | [What If作者公开PDF（2024）](https://www.hsph.harvard.edu/miguel-hernan/wp-content/uploads/sites/1268/2024/04/hernanrobins_WhatIf_26apr24.pdf) | Part I：因果定义、交换性、标准化、IPW | 作者公开教材；识别假设与计算估计分开 |
| N 检索 | [Stanford IR教材](https://nlp.stanford.edu/IR-book/) | Ch6评分、Ch8评测、Ch11概率检索；[DPR](../../readings/papers/dpr.md) | HTML和PDF公开；传统IR与神经检索分别学习 |
| O 后训练与Agent | [RLHF Book及课程](https://rlhfbook.com/course)、[Berkeley Agents 2024](https://rdi.berkeley.edu/llm-agents/f24) | SFT/reward modeling/DPO/policy gradients；Agent的工具、规划、评估讲次 | 作者开放书与官方课件；SLP3未完成的Agent章节不作完整依据 |
| P 系统与可信AI | [DLSys](https://dlsyscourse.org/) | [MLSysBook](https://mlsysbook.ai/)、[Fairness and ML](https://fairmlbook.org/) | 开放课程/教材；系统测量与公平性约束承担不同问题 |
| Q 机器人 | [MIT Underactuated](https://underactuated.csail.mit.edu/) | [Robotic Manipulation](https://manipulation.csail.mit.edu/)，运动学、感知、规划与控制 | 作者开放讲义；仿真与硬件运行不能由静态阅读替代 |

### T05 论文、源码和实验的证据层

- [论文精读库](../../readings/README.md)：登记确切论文和阅读版本，关键实验回到原文图表。
- [项目源码导读](../../readings/projects/README.md)：针对固定版本和实际源文件，区分原始实现、教学实现和工程框架。
- [完整实践项目](../../labs/README.md)：自建小数据的真实算法运行；预训练模型扩展另行验证。
- [系统串讲](../08-walkthroughs/README.md)：原创例子和推导帮助理解，不把手算数值或模拟结果当作论文复现成绩。

公开可读与允许任意转载不同。优先链接官方材料并以自己的话解释，保留上游代码许可证；引用实验数字时写明它是作者报告还是本地运行。本轮不进行哈希值校验。

### T06 大模型主线的精确课程定位

下表在2026-10-03核对官方目录。在线材料可能继续更新；阅读时以所列版本、日期和小节名称共同定位。课堂练习与本仓库的自建实验分别维护，本仓库不是课程作业的逐题答案。

| 资料与阅读版本 | 具体位置 | 本仓库用途 | 开放范围 / 核验日期 |
|---|---|---|---|
| [CS229 SEE历史公开版](https://see.stanford.edu/Course/CS229) | Lecture2：线性回归、批量与随机梯度；Lecture3：概率解释、欠拟合/过拟合、逻辑回归；Lecture4：Newton与GLM | 串讲1–2的似然、损失与优化先修 | 官方视频、讲义与transcript；2026-10-03 |
| [D2L英文网页1.0.3](https://d2l.ai/) | §2.5自动微分、§3.6泛化、§5.3前后向、§11.5多头注意力、§11.7 Transformer、§12.10 Adam | 数学→张量→代码的连接 | 网页正文、代码公开；不要混用不同语言版章节编号；2026-10-03 |
| [CS224N Winter2026课程表](https://web.stanford.edu/class/cs224n/#schedule) | Jan20 Transformer、Jan27预训练、Jan29后训练、Feb3高效适配、Feb5 Agent/工具/RAG、Feb10评测 | 对应串讲3–6及精读路径 | 课件与阅读列表公开；当季录像的Canvas入口可能需身份，历史公开视频另查；2026-10-03 |
| [CS336 Spring2025课程表](https://cs336.stanford.edu/spring2025/#schedule) | Lecture1分词、2资源核算、3架构、9/11 scaling、10推理、12评测、13/14数据、15–17后训练；作业1–5 | Mini-GPT、推理与后训练的课程依据 | 历史归档、代码讲义及作业公开；本机项目采用较小自建任务；2026-10-03 |
| [RLHF Book网页与配套课](https://rlhfbook.com/course) | instruction tuning、reward modeling、policy gradients、direct preference optimization；网页主页标示last built 2026-09-24 | 串讲5及InstructGPT/DPO/GRPO的先修 | 作者公开书与课程；与2026年纸书的勘误差异以主页说明为准；2026-10-03 |
| [Berkeley Agents Fall2024](https://rdi.berkeley.edu/llm-agents/f24) | Sept16 Agent概述/ReAct、Sept23框架与知识助手、Oct7复合系统、Nov25能力测量、Dec2可信Agent | 串讲6的工具、状态、评估和失败边界 | 课件与edited video公开；original recording的校内入口分开；2026-10-03 |
