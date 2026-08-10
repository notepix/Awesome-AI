# E. 计算机视觉


[🏠 仓库首页](../../README.md) · [📚 学习导航](../README.md) · [📖 完整单文件版](../../full/AI_Encyclopedia.md) · [← 上一章](../01-foundations/d-deep-learning.md) · [下一章 →](f-nlp-transformers-llms.md)

---

## 知识单元

### E01 图像表示、成像、颜色、采样与增强 `[核心·成熟]`

- **先修**：A02、A10、B01、B05、D06。
- **定义与解析**：数字图像是采样网格上的强度/颜色张量，常写成 `N×C×H×W`。成像把连续辐照度经光学、曝光、传感器响应、量化变为像素；颜色空间只是坐标系，不等于人的完整色觉。增强是在“标签应保持不变”的变换族上采样，隐含了任务不变性假设。
- **公式/机制**：采样率不足会混叠；缩小前应低通。标准化为 `x'=(x-μ)/σ`。经验风险加增强后为 `E_(x,y) E_(t~T) L(f(t(x)),y)`；若裁剪删掉目标或翻转改变语义，该假设失效。
- **资料定位**：[D2L 14.1 Image Augmentation，§14.1.1–14.1.3](https://d2l.ai/chapter_computer-vision/image-augmentation.html)；[Szeliski《Computer Vision》2e 官网，电子版 Ch.2 Image Formation、Ch.3 Image Processing](https://szeliski.org/Book/)。
- **最小代码（可运行，NumPy）**：

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
x = x[:, ::-1]                         # 水平翻转
x = x[::2, ::2].astype(np.float32)     # 仅示意下采样；真实缩小先低通
mean, std = x.mean((0, 1)), x.std((0, 1)).clip(1e-6)
z = (x - mean) / std
print(z.shape, np.round(z.mean((0, 1)), 5))
```

- **检测题/小实验**：先预测“文字识别、左右手分类、普通猫狗分类”中水平翻转分别是否保标签；再比较“直接隔点采样”和“2×2 均值后采样”的棋盘格混叠。
- **常见坑**：混用 `HWC/CHW`；训练/验证归一化统计不一致；把测试增强结果用于挑模型造成泄漏；认为所有几何增强都安全。

### E02 滤波、边缘、局部特征、多视几何与传统视觉 `[分支·成熟]`

- **先修**：A02、A04、A05、A16、B01、E01。
- **定义与解析**：滤波用局部算子抑噪或提取频率；边缘是强度快速变化，不天然等于物体边界。局部特征由检测器、描述子、匹配器组成；多视几何利用同一 3D 点在多相机中的投影约束恢复姿态/结构。
- **公式/机制**：离散相关 `y[i,j]=Σ_(u,v) k[u,v]x[i+u,j+v]`；Sobel 近似 `∇I`。针孔模型 `s p=K[R|t]P`；匹配点满足极线约束 `p'^T F p=0`，但纹理弱、重复纹理、退化运动会使估计不稳。
- **资料定位**：[Szeliski 2e 官网电子版，Ch.3 Image Processing、Ch.7 Feature Detection and Matching、Ch.11 Structure from Motion](https://szeliski.org/Book/)；官网说明 PDF 内含章节/公式超链接。
- **最小代码（可运行，NumPy Sobel）**：

```python
import numpy as np
x = np.arange(25, dtype=float).reshape(5, 5)
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
ky = kx.T
p = np.pad(x, 1, mode="edge")
gx = np.empty_like(x); gy = np.empty_like(x)
for i in range(5):
    for j in range(5):
        gx[i, j] = (p[i:i+3, j:j+3] * kx).sum()
        gy[i, j] = (p[i:i+3, j:j+3] * ky).sum()
print(np.hypot(gx, gy))
```

- **检测题/小实验**：为什么高斯平滑后再求导通常比先求导更稳？把图像整体加常数，Sobel 响应应如何变化？
- **常见坑**：把数学卷积与库中的互相关混为一谈；用像素距离直接匹配不同视角；RANSAC 内点多就断言几何正确；忽略相机标定和坐标系。

### E03 CNN、卷积、等变性与感受野 `[核心·成熟]`

- **先修**：A07、B04、D01–D06、E01–E02。
- **定义与解析**：CNN 用局部连接与参数共享编码平移结构。卷积层对平移近似**等变**，池化/全局汇聚才可能带来一定不变性；步幅、边界填充和下采样都会破坏严格等变。
- **公式/机制**：`Y[o,i,j]=b_o+Σ_cuv W[o,c,u,v]X[c,i+u,j+v]`，输出宽 `⌊(W+2P-D(K-1)-1)/S⌋+1`。连续 `L` 个步幅 1、`3×3` 层的理论感受野为 `1+2L`，有效感受野通常更集中。
- **资料定位**：[CS231n ConvNets，Convolutional Layer 的 Local Connectivity、Spatial Arrangement、Parameter Sharing](https://cs231n.github.io/convolutional-networks/)（同页给出输出尺寸公式与 NumPy 索引例）。
- **最小代码（可运行，PyTorch）**：

```python
import torch
torch.manual_seed(0)
conv = torch.nn.Conv2d(1, 2, 3, padding=1, bias=False)
x = torch.randn(1, 1, 8, 8)
y = conv(x)
xs = torch.roll(x, shifts=1, dims=3)
ys = conv(xs)
err = (torch.roll(y, 1, 3)[:, :, :, 1:-1] - ys[:, :, :, 1:-1]).abs()
print(y.shape, err.max().item())
```

- **检测题/小实验**：解释为何代码排除边界后误差更小；给出 `K=5,S=2,P=2,D=1,W=32` 的输出宽。
- **常见坑**：把通道数叫“空间深度”；默认卷积严格平移不变；只算理论感受野不查信息路径；忘记 dilation 对有效核大小的影响。

### E04 ResNet、EfficientNet、ConvNeXt 与现代骨干 `[核心·成熟]`

- **先修**：D04、D07、D10、D14、E03。
- **定义与解析**：骨干网络把图像变为分层特征。ResNet 学残差 `F(x)` 并走捷径；EfficientNet 联合缩放深度/宽度/分辨率；ConvNeXt 用现代训练与大核、分组卷积等设计重新审视纯 ConvNet。它们是设计族，不存在跨数据/预算永远最优者。
- **公式/机制**：残差块 `y=x+F(x)` 给梯度一条恒等路径；尺寸/通道改变时用投影捷径。复合缩放写作 `d=α^φ,w=β^φ,r=γ^φ`，受计算量约束约为 `αβ²γ²≈2`。
- **资料定位**：[ResNet，CVPR 2016 §3.1–3.3、Fig.2](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)；[EfficientNet，ICML 2019 §3 Compound Model Scaling](https://proceedings.mlr.press/v97/tan19a.html)；[ConvNeXt，CVPR 2022 §2–3](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html)。
- **最小代码（可运行，PyTorch）**：

```python
import torch
class Block(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Conv2d(c, c, 3, padding=1),
                                     torch.nn.ReLU(),
                                     torch.nn.Conv2d(c, c, 3, padding=1))
    def forward(self, x):
        return torch.relu(x + self.f(x))
torch.manual_seed(0)
x = torch.randn(2, 4, 8, 8, requires_grad=True)
y = Block(4)(x).mean(); y.backward()
print(y.item(), x.grad.norm().item())
```

- **检测题/小实验**：将 `x+self.f(x)` 改成 `self.f(x)`，在 20–50 层小网络中比较初始梯度范数；通道翻倍时捷径为什么不能直接相加？
- **常见坑**：把残差理解为“永不梯度消失”；只比参数量不比 FLOPs/延迟/分辨率；把论文训练配方带来的收益全归因于结构。

### E05 图像分类、定位、归因与可解释性 `[核心·较成熟]`

- **先修**：C03、C12、D03、D08、D11、D15、E03–E04。
- **定义与解析**：分类预测全图标签；定位还要给目标位置。归因方法回答“当前输出对哪些输入/特征敏感”，不是因果解释，也不证明模型用了人类认可的概念。解释质量应做忠实性、稳定性与随机化检查。
- **公式/机制**：多类损失 `L=-log softmax(z)_y`；Grad-CAM 对类别 `c` 取 `α_k^c=mean_(ij) ∂y^c/∂A^k_ij`，热图 `ReLU(Σ_k α_k^c A^k)`。热图分辨率受最后特征图限制。
- **资料定位**：[CS231n Transfer Learning，Fine-tuning ConvNets 与固定特征](https://cs231n.github.io/transfer-learning/)；[Grad-CAM，ICCV 2017 §3、Fig.2](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html)。
- **最小代码（可运行，输入梯度归因）**：

```python
import torch
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 3))
x = torch.randn(1, 1, 4, 4, requires_grad=True)
score = model(x)[0, 1]
score.backward()
attr = (x * x.grad).detach().abs()[0, 0]
print(attr, "top-pixel=", divmod(attr.argmax().item(), 4))
```

- **检测题/小实验**：随机重置模型权重后，若“解释图”几乎不变说明什么？比较预测概率与 logit 的梯度，饱和时哪个更可能接近零？
- **常见坑**：把显著图当分割掩码；只展示好看的个例；以解释的一致性代替正确性；用测试集反复挑增强/阈值。

### E06 目标检测与实例分割 `[分支·成熟]`

- **先修**：C12、D03、D08、E03–E05。
- **定义与解析**：检测输出类别、置信度和框；实例分割再为每个实例输出掩码。两阶段方法先提候选再分类回归，一阶段方法密集预测；端到端集合预测则用匹配消除手工锚框/NMS 依赖。
- **公式/机制**：框回归、分类和掩码损失组成 `L=L_cls+λ_box L_box+λ_mask L_mask`。IoU=`|A∩B|/|A∪B|`；NMS 按分数保留框并抑制高 IoU 重复框。AP 是整条 precision–recall 曲线的汇总，不是单一准确率。
- **资料定位**：[Faster R-CNN，NeurIPS 2015 §3 Region Proposal Networks、§4 unified training](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046f-Abstract.html)；[Mask R-CNN，ICCV 2017 §3、RoIAlign](https://openaccess.thecvf.com/content_ICCV_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)。
- **最小代码（可运行，NumPy NMS）**：

```python
import numpy as np
boxes = np.array([[0,0,2,2], [0.5,0.5,2.5,2.5], [3,3,4,4]], float)
scores = np.array([.9, .8, .7]); keep = []
order = scores.argsort()[::-1]
while order.size:
    i = order[0]; keep.append(i)
    xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
    yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
    xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
    yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
    inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
    area = lambda b: (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    iou = inter / (area(boxes[[i]])[0] + area(boxes[order[1:]]) - inter)
    order = order[1:][iou <= .5]
print(keep)
```

- **检测题/小实验**：调 NMS 阈值从 0.1 到 0.9，预测重复数和漏检趋势；同一 AP 下，小目标召回是否必然相同？
- **常见坑**：坐标端点是否含边导致 IoU 偏差；在不同 IoU/尺度定义间直接比 AP；数据增强后未同步变换框/掩码；类别不平衡只看总 loss。

### E07 语义分割、深度估计、光流与稠密预测 `[分支·成熟]`

- **先修**：A04、C03、D03、E02–E04。
- **定义与解析**：稠密预测为每个像素/位置输出类别、深度、法线或二维运动。编码器压缩语义，解码器恢复空间细节；跳连融合不同尺度。单目深度常只有相对尺度，光流是表观对应而非真实 3D 速度。
- **公式/机制**：语义分割常用逐像素交叉熵与 Dice/IoU；深度用 `L1`、尺度不变损失；经典光流亮度恒常线性化为 `I_x u+I_y v+I_t=0`，一条方程不能唯一解二维流（孔径问题）。
- **资料定位**：[FCN，CVPR 2015 §3 Fully Convolutional Networks、§4 segmentation architecture](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)；[RAFT，ECCV 2020 §3，correlation volumes 与 recurrent updates](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php)。
- **最小代码（可运行，像素交叉熵）**：

```python
import torch
torch.manual_seed(0)
logits = torch.randn(2, 4, 8, 8, requires_grad=True)
target = torch.randint(0, 4, (2, 8, 8))
target[:, :2] = 255                    # ignore 区域
loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=255)
loss.backward()
pred = logits.argmax(1)
print(loss.item(), pred.shape, logits.grad.norm().item())
```

- **检测题/小实验**：全预测背景时 pixel accuracy 可能很高而 mIoU 很低，构造一个 10 像素例子；单目深度整体乘 2 为何可能仍保持相对几何？
- **常见坑**：插值标签时使用双线性造成非法类别；忽略 void label；把 optical flow 当物体速度；训练/评测深度单位和尺度对齐不一致。

### E08 ViT、视觉自监督与视觉基础模型 `[核心·较成熟；基础模型持续演进]`

- **先修**：D09、D12–D14、E01、E03–E04。
- **定义与解析**：ViT 把图像切成 patch token 后用 Transformer；视觉自监督通过对比、掩码重建或教师—学生目标从无人工标签图像学习表征。“视觉基础模型”强调可迁移范围与规模，但并非任意任务零样本都可靠。
- **公式/机制**：patch 展平后 `z_0=[x_cls;x_p^1E;…]+E_pos`；自注意力为 `softmax(QK^T/√d)V`，计算随 token 数平方增长。对比损失拉近同图增强视图；掩码建模只重建被遮 patch；DINO 类方法匹配教师/学生分布并避免坍塌。
- **资料定位**：[ViT，ICLR 2021 §3、Eq.(1)–(4)](https://openreview.net/forum?id=YicbFdNTTy)；[DINOv2 §3 automatic data curation、§4 discriminative self-supervised pretraining](https://arxiv.org/abs/2304.07193) 与[官方代码](https://github.com/facebookresearch/dinov2)。
- **最小代码（可运行，patch token）**：

```python
import torch
torch.manual_seed(0)
x = torch.randn(2, 3, 16, 16)
patch = torch.nn.Conv2d(3, 12, kernel_size=4, stride=4)
t = patch(x).flatten(2).transpose(1, 2)  # [B, 16, 12]
cls = torch.zeros(2, 1, 12)
seq = torch.cat([cls, t], dim=1)
attn = torch.nn.MultiheadAttention(12, 3, batch_first=True)
y, w = attn(seq, seq, seq)
print(seq.shape, y.shape, w.shape)
```

- **检测题/小实验**：patch 从 `16×16` 改为 `8×8` 时 token 数与注意力矩阵元素数各变几倍？冻结骨干线性探测与全量微调分别测什么？
- **常见坑**：把 patch embedding 当语义分词；只报线性探测不报迁移设置；把数据规模效应归因于单个目标；将发布方称“foundation”当作跨域能力证据。

### E09 视频理解、3D 视觉、点云与 NeRF `[分支·较成熟；世界建模解释仍前沿]`

- **先修**：A04–A05、E02–E08、F07（学习时可后补）。
- **定义与解析**：视频模型还要建模时间与运动；点云是无序 3D 点集；多视 3D 依赖相机几何；NeRF 用坐标网络表示位置/方向到密度与颜色的连续辐射场，再可微体渲染。新视角合成好不等于恢复了唯一真实几何或物理规律。
- **公式/机制**：点集函数需对排列不变，如 `g({x_i})=γ(max_i h(x_i))`。NeRF 光线颜色 `C(r)=Σ_i T_i α_i c_i`，`α_i=1-exp(-σ_iδ_i)`、`T_i=Π_(j<i)(1-α_j)`；姿态误差/稀疏视角会产生伪影。
- **资料定位**：[PointNet，CVPR 2017 §3.2 symmetric function](https://openaccess.thecvf.com/content_cvpr_2017/html/Qi_PointNet_Deep_Learning_CVPR_2017_paper.html)；[NeRF，ECCV 2020 §4 volume rendering、§5 positional encoding](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1473_ECCV_2020_paper.php)；[I3D，CVPR 2017 §3 two-stream inflated 3D ConvNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)。
- **最小代码（可运行，体渲染离散合成）**：

```python
import torch
sigma = torch.tensor([.2, 1.0, .4])
delta = torch.tensor([.5, .5, 1.0])
rgb = torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
alpha = 1 - torch.exp(-sigma * delta)
survive = torch.cat([torch.ones(1), 1 - alpha[:-1]])
T = torch.cumprod(survive, dim=0)
weights = T * alpha
print(weights, (weights[:, None] * rgb).sum(0))
```

- **检测题/小实验**：随机打乱点顺序，`max_i h(x_i)` 是否变化？令所有 `σ→0` 或首个 `σ→∞`，渲染颜色各趋向什么？
- **常见坑**：把帧独立分类叫视频理解；混淆相机坐标/世界坐标；NeRF 密度当可直接测得的实体；由生成视频“看似合理”推断模型已学会物理。边界截至 **2026-08-11**。
