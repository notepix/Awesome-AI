# 多模态：图文双塔与组合泛化

项目自行生成彩色几何图像及英文 caption，以真实梯度下降训练图像塔和文本塔，最后评估图搜文、文搜图。先读 [公共环境](../README.md)。

## CPU 运行链

```powershell
python -m labs.run multimodal prepare
python -m labs.run multimodal train
python -m labs.run multimodal eval
python -m labs.run multimodal demo --prompt "a red triangle"
# 可选导出当前检索向量
python -m labs.run multimodal index
```

默认 4 种颜色 × 3 种形状 × 每类 12 张，共 144 张 32×32 图像。位置与大小有随机扰动。红色三角形、绿色方形、蓝色圆形三种**完整组合**只进入测试集；其他组合各保留三张验证图像，其余用于训练。模型训练能见到所有基本颜色词与形状词。

因此这里检验的是**留出属性组合的泛化**，不声称模型认识任意自然图片或未学习过的新词。验证集保留给扩展实验；默认固定步骤训练，不用测试集挑选 checkpoint。

## 两个塔与对比损失

图像塔：两层卷积、池化和线性投影。文本塔：词嵌入平均后投影。两塔输出均做 L2 归一化，温度缩放后的内积进入双向对比损失。

同一 caption 对应多张图像；损失把批内所有相同 caption 都视作正例，避免把另一张同类图像误当负例。保存的未训练基线用于判断训练带来的变化，不预设训练必然让留出组合成绩提高。

## 双向 Recall@k

`eval` 在所有 12 个 caption 候选中检索测试图像的描述；文搜图只对测试集中存在相关图像的 caption 计算。相关性定义为**相同语义 caption**，不是某个任意文件的一对一配对。分别报告两个方向的 Recall@1、3、5，并保存候选数与测试图像数。

候选数很小时较大的 k 容易达到满分，必须优先结合 Recall@1 和随机/未训练基线解释结果。demo 返回最相近图像路径、真实 caption 和相似度，便于直接检查错误。

## 本地预训练 CLIP

此扩展评估预训练模型，不重新训练 CLIP。自行准备 `openai/clip-vit-base-patch32` 完整目录并安装可选依赖：

```powershell
python -m labs.run multimodal index --config labs/multimodal/local.json --model-path D:/models/clip-vit-base-patch32
python -m labs.run multimodal eval --config labs/multimodal/local.json --model-path D:/models/clip-vit-base-patch32
python -m labs.run multimodal demo --config labs/multimodal/local.json --model-path D:/models/clip-vit-base-patch32 --prompt "a red triangle"
```

扩展使用模型对应的图像预处理和 tokenizer。caption 与类别提示保持英文，中文只用于讲解；不能据此推断该版本的中文检索表现。CLIP 阶段读取本地权重，不访问模型仓库；缺模型明确失败。

## 文件、测试与继续实验

实现位于 [`multimodal.py`](../ai_learning/multimodal.py)。输出包含 images、数据划分、双塔 checkpoint、可选向量索引和阶段 JSON。公共测试验证 train/test 组合不重叠、基本词仍被训练见过、多正例损失和双向语义 Recall。

建议继续比较：随机图像划分与组合划分为何得出不同结论；不同 batch 中重复 caption 对普通 InfoNCE 有何影响；留出组合能力与预训练 CLIP 的零样本能力为什么不是同一件事。

## 回到讲义与精读

[生成与多模态讲义](../../docs/08-walkthroughs/07-generative-multimodal.md) · [CLIP 论文精读](../../readings/papers/clip.md) · [LLaVA 论文精读](../../readings/papers/llava.md) · [Transformers 模型接口精读](../../readings/projects/transformers.md)。
