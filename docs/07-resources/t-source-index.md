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
