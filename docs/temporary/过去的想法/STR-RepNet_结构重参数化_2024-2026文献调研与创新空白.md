# STR-RepNet：轻量遥感二值变化检测中的结构重参数化——2024–2026 文献调研与创新空白

> **课题**：Lightweight Spatial-Temporal Structural Re-parameterization Network for Remote Sensing Change Detection  
> **Baseline**：HAM-CD（IEEE TGRS 2026）+ 当前 STR-RepNet GitHub 实现  
> **检索截止**：2026-09-19  
> **目标**：为论文 Related Work、方法命名、创新点边界、消融设计与“训练图→部署图”等价性论证提供可核验文献依据。  
> **检索原则**：2024–2026 为主体；RepVGG / ACNet / DBB / RepLKNet / OREPA / RepGhost / FastViT / MobileOne / RepOptimizer 等作为历史锚点。CCF-A 会议与 SCI 期刊分开标注；JSTARS、GRSL、ICASSP、CVPR Workshop 均不标成 CCF-A。预印本明确写明“Preprint”。
> **链接纪律**：优先给官方代码、CVF/AAAI/NeurIPS/Springer/IEEE/Elsevier 等出版社页面或 arXiv；若未核验到官方代码仓库，不以第三方复现仓库冒充“官方代码”，而只给论文/DOI 页面。

---

## 0. 先给结论：你的研究路线应该如何定位

### 0.1 最值得作为论文主创新的不是“把 RepConv 塞进 HAM-CD”，而是“**整解码器可组合结构重参数化**”

当前 HAM-CD 本身已经不是“完全没有重参数化”的干净起点。当前 STR-RepNet 仓库中的 `Spatial_Mamba.py::StateFusion` 已经明确实现：

- 训练期：3 路 `3×3 DWConv`，dilation = 1 / 3 / 5；
- 推理期：把三路权重嵌入并相加成一个 `11×11 depthwise kernel`；
- 推理调用单个 `F.conv2d(..., groups=dim)`。

因此，论文不能把“在 TAM 的 dilation 1/3/5 分支上首次使用结构重参数化”当作核心创新。这个点已经存在于 baseline 源码中。真正有空间的表述应升级为：

> **Decoder-wide / compositional structural re-parameterization**：不局限于某一个大核卷积分支，而是对 HAM decoder 中所有满足代数可折叠条件的线性子图做系统识别、训练期扩展和部署期收缩，并进一步研究相邻线性块的级联折叠，以及二时相拓扑与空间卷积拓扑的联合折叠。

这与 CD-RLKNet、LKMamba-CD 一类“在某模块里放一个 re-parameterized large-kernel block”有明显区分。

### 0.2 你的 Temporal Rep 是目前检索中最有潜力的“变化检测专属重参数化”切入点

设：

\[
Y_c = W_c * [X_1,X_2] + b_c,\quad
Y_s = W_s * (X_1+X_2)+b_s,\quad
Y_d = W_d * (X_2-X_1)+b_d
\]

将 \(W_c\) 按输入时相通道拆成 \(W_c=[W_{c1},W_{c2}]\)，则：

\[
Y =
(W_{c1}+W_s-W_d)*X_1 +
(W_{c2}+W_s+W_d)*X_2 +
(b_c+b_s+b_d)
\]

所以部署时可精确变为：

\[
W_{\text{deploy}}
=
[W_{c1}+W_s-W_d,\;
 W_{c2}+W_s+W_d]
\]

即对 `[X1, X2]` 的一个卷积。

**关键点**：必须是 signed difference `X2-X1`。若改成常见的 `abs(X2-X1)`，绝对值引入非线性，就不能做上述严格代数折叠。

在本次核验的 2024–2026 RSCD 文献中，大量工作做了 concat / difference / adaptive difference / multiscale difference / spatial-temporal fusion，但**未核验到一篇将 Concat + Sum + signed-Diff 三路变化拓扑严格代数坍缩成单个 temporal projection，并把“train/deploy exact equivalence”作为变化检测核心机制的已发表高水平工作**。这应写成“在本次检索范围内未发现”，不要写成绝对的“世界首次”。

### 0.3 编码器三分支路线的正确术语是“异构多教师蒸馏”，不是结构重参数化

若训练期为：

- 主学生：VMamba；
- 辅助教师：DINOv2 / SAM / CNN / Transformer 等；
- 教师只进入 feature/logit distillation loss；
- 推理时删除教师；

那么最准确的定位是：

> **heterogeneous multi-teacher knowledge distillation / teacher-ensemble distillation / cross-architecture distillation**

若多个可训练网络彼此互教，可叫：

> **deep mutual learning (DML)**

若所有模型共享一个 supernet 权重并抽取不同宽度/子网，才更接近：

> **once-for-all / slimmable network**

“ensemble collapse”不是这一类方法最规范、最稳定的主术语，建议不要作为论文主标签。

它与卷积结构重参数化的根本区别是：

| 维度 | 卷积级结构重参数化 | 异构教师蒸馏 |
|---|---|---|
| 训练→部署关系 | 参数代数变换 | 优化得到另一个函数 |
| 是否可证明逐点等价 | 可以，满足条件时 | 不可以 |
| 是否需要额外训练来完成“折叠” | 不需要 | 需要 KD 训练 |
| 部署前后主网络函数 | 理论上相同 | 一般不同 |
| 误差目标 | 可做到 `<1e-6` 数值误差 | 不存在严格等价要求 |
| 论文归类 | Structural Re-parameterization | Knowledge Distillation / Model Compression |

---

# 1. Baseline 与当前代码：哪些地方能折，哪些地方不能折

## 1.1 HAM-CD 原论文的关键结构

HAM-CD（Guanlin Li 等，IEEE TGRS 2026）采用：

- 共享权重 Siamese VMamba encoder；
- 4 级 HAM decoder；
- HAM = 并行 TAM + TAB，再 concat→conv 融合；
- TAM 中有 textual-aware SSM / TASF，多尺度 dilation 1/3/5；
- TAB 是 channel-transposed attention，Q/K/V 由 pointwise + depthwise projection 产生；
- ICSF 做跨级融合，包含卷积、DWConv 与 SE-style channel gate。

原文官方代码：<https://github.com/guanguanboy/HAM-CD>

当前课题仓库：<https://github.com/YuqiWang-code/STR-RepNet>

## 1.2 当前 STR-RepNet 代码层面的“可折叠地图”

| HAM-CD / 当前代码位置 | 当前算子 | 静态精确折叠？ | 建议 |
|---|---|---:|---|
| `Spatial_Mamba.StateFusion` | 3×3 DWConv, d=1/3/5，训练多支路 | **是，而且 baseline 已经做了** | 作为已有机制，不可再单独包装成新创新 |
| selective scan / S6 / TASF 中输入依赖递推 | 输入决定状态更新 | **否** | 部署保留；不要声称能卷积化精确折叠 |
| SpatialMamba 前置 DWConv + SiLU | Conv→SiLU | 跨 SiLU **否** | 只能在 SiLU 两侧分别做局部线性重参数化 |
| MDTA Q/K/V 的 `1×1 conv → 3×3 DWConv` | 线性串联 | 数学上可组成 3×3 dense conv | 需比较组合后 FLOPs/参数/latency，不能只追求“1 个算子” |
| MDTA `Softmax(KQ)` 与 `V·A` | 输入依赖乘法 | **否** | Attention core 保留 |
| FFN 内 DWConv | 线性空间算子附近有激活/门控 | 局部可，跨激活不可 | 用 RepDW 替换单个 DWConv 可行 |
| `fuse_layer_*` | 1×1 Conv + BN + ReLU | Conv+BN 可；跨 ReLU 不可 | 先 BN-fold，再判断前后线性块是否可合并 |
| ICSF `conv+BN` | 线性+BN | eval 时 **是** | 适合 BN folding / branch rep |
| ICSF DWConv+BN | 线性+BN | **是** | 可做多尺度/非对称/identity 分支再合并 |
| ICSF SE | GAP→MLP→Sigmoid→乘法 | **否** | 数据依赖门控必须保留，除非换成 ASR 类“attention-alike”机制 |
| `_upsample_add` | bilinear resize + add | 线性但不是普通同尺度 Conv2d | 不建议为了“单算子”强行合并 |
| residual add | 同输入同形状线性分支求和 | 条件满足时 **是** | 可做 RepVGG/DBB 风格 branch sum folding |

---

# 2. 子方向一：结构重参数化基础与 2024–2026 新进展

## 2.1 经典锚点：论文 Related Work 必须知道，但不是 2024–2026 主证据

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 核心机制 | 与 STR-RepNet 的映射 |
|---|---|---|---|---|---|
| **ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks** | ICCV 2019，CCF-A | Xiaohan Ding，清华大学 | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.html) | 3×3 + 1×3 + 3×1 训练分支，部署合并为 3×3 | 给 TAM/ICSF 的非对称空间分支提供最直接公式模板 |
| **RepVGG: Making VGG-Style ConvNets Great Again** | CVPR 2021，CCF-A | Xiaohan Ding，清华大学 BNRist | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html) / [Code](https://github.com/DingXiaoH/RepVGG) | 3×3、1×1、identity+BN 多分支 → 单 3×3 | 你所有“训练多支路→部署单支路”的基本范式 |
| **Diverse Branch Block (DBB)** | CVPR 2021，CCF-A | Xiaohan Ding，清华大学 | [CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.html) / [Code](https://github.com/DingXiaoH/DiverseBranchBlock) | 支持串联 conv、不同 kernel、avg pooling 等复杂支路的等价合并 | **最值得精读的“级联折叠”理论基础** |
| **RepLKNet** | CVPR 2022，CCF-A | Xiaohan Ding，清华/MEGVII 系合作 | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.html) / [Code](https://github.com/DingXiaoH/RepLKNet-pytorch) | 小核辅助大核训练，部署为大 DW kernel | TAM dilation branch、CD-RLKNet/LKMamba-CD 的直接祖先 |
| **OREPA: Online Convolutional Re-Parameterization** | CVPR 2022，CCF-A | Mu Hu，浙江大学（合作作者含 Alibaba Cloud / DAMO Academy） | [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Online_Convolutional_Re-Parameterization_CVPR_2022_paper.html) / [Code](https://github.com/JUGGHM/OREPA_CVPR2022) | 训练期在线压缩复杂 re-param block，显著降低训练开销 | 若整 decoder 全 Rep 后训练图太重，OREPA 是必要参考 |
| **RepGhost** | 2022，**arXiv 预印本，不是 CCF-A 正式论文** | Chengpeng Chen；官方 arXiv 摘要页未列单位 | [arXiv](https://arxiv.org/abs/2211.06088) / [Code](https://github.com/ChengpengChen/RepGhost) | 用 SRP 隐式 feature reuse，减少 concat 的硬件代价 | HAM concat→1×1 fusion 可从“FLOPs≠latency”角度借鉴 |
| **RepOptimizer** | ICLR 2023，顶会锚点 | Xiaohan Ding，清华背景 | [OpenReview](https://openreview.net/forum?id=B92TMCG_7rp) / [Code](https://github.com/DingXiaoH/RepOptimizers) | 把结构先验转移到梯度/优化器，训练和部署都保持简单网络 | 可作为“结构重参数化不是唯一 Rep 范式”的辨析文献 |
| **MobileOne** | CVPR 2023，CCF-A | Pavan Kumar A. Vasu，Apple | [CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.html) | 移动端 SRP，多支路训练、低 latency 部署 | 说明你的最终指标不能只报 Params/FLOPs，还应尽量报 FPS/latency |
| **FastViT** | ICCV 2023，CCF-A | Pavan Kumar A. Vasu，Apple | [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html) / [Code](https://github.com/apple/ml-fastvit) | RepMixer 用 SRP 消除 skip connection，优化 memory access | 支持“部署图算子拓扑本身”应成为贡献，而非只看参数量 |

## 2.2 2024–2026 代表工作

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 一句话机制 | 与你的映射 |
|---|---|---|---|---|---|
| **RepViT: Revisiting Mobile CNN From ViT Perspective** | CVPR 2024，**CCF-A** | Ao Wang，清华大学 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html) / [Code](https://github.com/THU-MIG/RepViT) | 从 ViT 设计原则重新构造移动 CNN，并利用 re-param token mixer/结构优化 | 你的论文“轻量 + 部署 latency”最重要通用对照之一 |
| **UniRepLKNet** | CVPR 2024，**CCF-A** | Xiaohan Ding，Tencent AI Lab | [IEEE/CVF](https://doi.org/10.1109/CVPR52733.2024.00527) / [Code](https://github.com/AILab-CVC/UniRepLKNet) | Dilated Reparam Block：多尺度 dilation 小核训练，部署为单个大核 | 与 TAM 的 dilation 1/3/5 **高度同源**；必须重点区分 |
| **RepAn: Enhanced Annealing through Re-parameterization** | CVPR 2024，**CCF-A** | Xiang Fei，厦门大学 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Fei_RepAn_Enhanced_Annealing_through_Re-parameterization_CVPR_2024_paper.html) / [Code](https://github.com/xfey/RepAn) | 在 annealing 周期中反复 Rep expansion→restoration→BP | 说明 Rep 还可作为训练动态，不必只做静态一次转换 |
| **Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities** | CVPR 2024，**CCF-A** | Yiyuan Zhang，CUHK MMLab | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Pathway_Improve_Transformers_with_Irrelevant_Data_from_Other_Modalities_CVPR_2024_paper.html) / [Code](https://github.com/AILab-CVC/M2PT) | Cross-Modal Re-parameterization，把辅助 Transformer 权重吸收到目标路径，不增推理成本 | 适合在 §2 中讨论“参数吸收式 Rep”和 KD 的边界 |
| **Stripe Observation Guided Inference Cost-free Attention Mechanism (ASR)** | ECCV 2024，**CCF-A** | Zhongzhan Huang，中山大学 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3451_ECCV_2024_paper.php) / [Code](https://github.com/zhongshsh/ASR) | 发现 channel attention 趋向常量向量，构造 inference-cost-free attention-alike SRP | **与你的 SE/attention 不可折叠问题最相关** |
| **RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization** | 2024，**arXiv Preprint** | Mingshu Zhao；官方 arXiv 摘要页未列单位 | [arXiv](https://arxiv.org/abs/2406.16004) / [Code](https://github.com/suous/RepNeXt) | 同时使用 serial + parallel SRP 获得多尺度移动 CNN | 与你的“串联式多重 Rep”概念很接近，虽非顶会正式发表 |
| **RepFC: Universal Structural Reparametrization Block...** | CVPRW 2025，**Workshop，非 CCF-A 主会论文** | 官方 TUM 页面可核验团队归属 | [TUM](https://portal.fis.tum.de/en/publications/repfc-universal-structural-reparametrization-block-for-high-perfo/) | 尝试更通用地选择需要展开的 CONV，而非全层无差别展开 | 对“整 decoder 全部 Rep 会不会训练成本失控”的反例/警示 |

### 本节最值得精读

1. **UniRepLKNet**：你 TAM 的 dilation 1/3/5 与其 Dilated Reparam Block 在机制层面最接近；Related Work 不处理好，审稿人很容易说“就是 UniRepLKNet 搬到 CD decoder”。
2. **DBB**：你要做“跨模块级联折叠”，DBB 对 sequential conv branch 的等价变换是基础理论。
3. **ASR**：它明确指出普通 attention 的 multiplicative/input-dependent 形式不能直接 SRP，这正好支持你“SE、Softmax attention 保留不可约”的方法边界。

---

# 3. 子方向二：两类“多分支训练→单分支部署”的机制辨析

## 3.1 A 类：卷积/线性算子级代数折叠

代表文献：RepVGG、DBB、RepLKNet、OREPA、FastViT、UniRepLKNet、RepNeXt、ASR（attention-alike 特例）。

判定标准：

\[
f_{\text{train}}(x;\theta)
=
f_{\text{deploy}}(x;T(\theta))
\]

其中 \(T\) 是显式参数变换，**不需要再训练**。在 eval 模式、同一数值精度下，应能做到接近机器误差的输出差。

你的论文应把“Structural Re-parameterization”严格限定在这一类。

## 3.2 B 类：模块/编码器级知识迁移后删除辅助分支

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 类型 | 对你的备选路线意义 |
|---|---|---|---|---|---|
| **One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation** | NeurIPS 2023，CCF-A 锚点 | Zhiwei Hao，北京理工大学 / Huawei Noah’s Ark Lab | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html) / [Code](https://github.com/Hao840/OFAKD) | Heterogeneous KD | CNN/Transformer/MLP 异构中间表示不宜直接 L2 对齐；要做 latent/logit alignment |
| **Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition** | AAAI 2025，**CCF-A** | Chuanguang Yang，中科院计算所 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32990) | Multi-teacher KD | 若你用 DINOv2/SAM/CNN 多教师，teacher 权重不能默认平均 |
| **Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation** | ICCV 2025，**CCF-A** | Jhe-Hao Lin，National Yang Ming Chiao Tung University | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_Perspective-Aware_Teaching_Adapting_Knowledge_for_Heterogeneous_Distillation_ICCV_2025_paper.html) / [Code](https://github.com/jimmylin0979/PAT) | Heterogeneous feature KD | 对异构 encoder 的“view mismatch”给出比直接 feature MSE 更合理的证据 |
| **Improving Accuracy and Calibration via Differentiated Deep Mutual Learning** | CVPR 2025，**CCF-A** | Han Liu；清华/合作团队（以论文作者页为准） | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Improving_Accuracy_and_Calibration_via_Differentiated_Deep_Mutual_Learning_CVPR_2025_paper.html) | DML | 只有多个网络共同训练、互相学习时才适合叫 DML |
| **ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models** | ICCV 2025，**CCF-A** | Guoyizhe Wei；论文作者页与 Rama Chellappa 合作 | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_ViT-Linearizer_Distilling_Quadratic_Knowledge_into_Linear-Time_Vision_Models_ICCV_2025_paper.html) | Cross-architecture KD | 直接展示 ViT→linear-time recurrent/Mamba-style student 的路线 |
| **MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling** | ICCV 2025，**CCF-A** | Yingyue Li，华中科技大学 HUST-VL | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_MaTVLM_Hybrid_Mamba-Transformer_for_Efficient_Vision-Language_Modeling_ICCV_2025_paper.html) / [Code](https://github.com/hustvl/MaTVLM) | Transformer→Mamba hybrid KD | 说明“Mamba 作 student”在高水平工作中是成立的，但仍是 KD |
| **Tiny-vGamba** | ICCV Workshops 2025，**非 CCF-A 主会** | Yunusa Haruna | [CVF Workshop](https://openaccess.thecvf.com/content/ICCV2025W/ECLR/html/Haruna_Tiny-vGamba_Distilling_Large_Vision-Language_Knowledge_from_CLIP_into_a_Lightweight_ICCVW_2025_paper.html) | CLIP→Mamba-like KD | 轻量 student + 冻结大教师的直接相邻证据 |
| **Image restoration model compression via Mamba-oriented heterogeneous KD (MHKD)** | Neural Networks 2026，SCI 期刊 | Sai Yang | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608026006209) | Transformer→Mamba KD | 进一步证明“异构教师→Mamba student”应归 KD 而非 Rep |

### 结论：备选路线怎么写才不会被审稿人挑术语

建议名称：

> **Heterogeneous Multi-Teacher Distillation for VMamba Change Encoder**

不建议：

> “Encoder Structural Re-parameterization”  
> “Multi-encoder re-parameterization”

除非辅助 encoder 的参数能通过显式代数映射 \(T(\theta)\) 无训练地并入 VMamba，并逐输入保持相同输出，否则不能叫 structural re-parameterization。

### 本节最值得精读

1. **PAT (ICCV 2025)**：异构 feature space 对齐最直接。
2. **MTKD-RL (AAAI 2025)**：多教师权重分配最直接。
3. **ViT-Linearizer (ICCV 2025)**：Mamba/线性时序 student 的强先例。

---

# 4. 子方向三：结构重参数化在遥感变化检测 / 分割 / 遥感视觉中的迁移

## 4.1 变化检测直接先例

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 核心方法 | 与你路线的关系 |
|---|---|---|---|---|---|
| **CD-RLKNet: Large kernel convolution application for land cover change detection of remote sensing images** | 2024，International Journal of Applied Earth Observation and Geoinformation，**SCI 期刊** | Junqing Huang，Macao Polytechnic University | [Publisher/Institution](https://research.mpu.edu.mo/en/publications/large-kernel-convolution-application-for-land-cover-change-detect) / [Code](https://github.com/juncyan/cdrlknet) | Re-parameterization large-kernel conv + spatial-temporal adaptive fusion + bitemporal integration | **最直接 prior art**。你不能只说“RSCD 首次引入大核重参数化” |
| **LKMamba-CD: Large Kernel State Space Model for Remote Sensing Change Detection** | 2026，PFG，SCI/专业期刊，非 CCF-A | Yi Yu；官方出版信息中机构需以 Springer 论文为准 | [DOI](https://doi.org/10.1007/s41064-026-00407-9) / [Code](https://github.com/kion-86/lkmamba) | re-parameterized large kernel + Mamba LKS，放入 BCIF/LRFA | **对你的“Mamba + 大核 Rep”构成更近的 2026 碰撞** |
| **ESIFCD** | IEEE Access 2025，SCI 期刊但非本课题主目标档次 | R. Ashraf | [DOI](https://doi.org/10.1109/ACCESS.2025.3612956) / [Code](https://github.com/russo-ashraf/ESIFCD) | RLKB re-parameterized large kernel + adaptive difference fusion | 说明“差异融合 + RLKB”组合已出现，不能仅靠模块并置形成创新 |

## 4.2 分割/通用视觉可迁移证据

| 工作 | 年份 / venue | 证据 | 对你的意义 |
|---|---|---|---|
| **UniRepLKNet** | CVPR 2024，CCF-A | 官方实现覆盖下游 dense prediction | 大核 DW re-param 可迁移到 segmentation/dense task，不只是分类 |
| **RepViT** | CVPR 2024，CCF-A | 官方 repo 提供 semantic segmentation pipeline | 轻量 Rep backbone 在 dense prediction 有成熟路径 |
| **OREPA** | CVPR 2022，CCF-A 锚点 | 论文明确验证 semantic segmentation | 说明 structural rep 不只适合分类 |
| **FastViT** | ICCV 2023，CCF-A 锚点 | 论文验证 segmentation | 支持“算子拓扑/内存访问”在 dense task 同样重要 |
| **RepCrack** | 2025，工程类期刊，**辅助证据，非核心高水平文献** | 结构重参数化用于 crack segmentation | 可作为“Rep-UNet/segmentation 类”补充，但不应放 Top-15 |
| **RA2M-UNet** | 2026，Biomedical Signal Processing and Control，**辅助证据** | Rep conv + attention + 2D SSM | 只作跨领域相邻方法，不宜作为主要 novelty 支撑 |

### 这一节对创新点的直接约束

**已经有人做过：**

- remote sensing CD + re-parameterized large kernel；
- Mamba + re-parameterized large kernel + CD；
- difference fusion + re-parameterized block。

**本次检索中仍明显稀缺：**

- 把 HAM decoder 视为一个“可折叠算子图”，系统覆盖 TAM projection / FFN / ICSF / fusion shell；
- 跨相邻模块的 compositional / cascade folding；
- 变化检测特有的 temporal topology algebraic folding；
- 部署图以“单路径、低算子数、严格误差测试”作为一等设计目标。

### 本节最值得精读

1. **CD-RLKNet (2024)**：论文 Related Work 里必须正面比较。
2. **LKMamba-CD (2026)**：与“HAM/Mamba + Rep 大核”最接近的碰撞风险。
3. **UniRepLKNet (CVPR 2024)**：机制上最接近 dilation-rep，而不是任务上最接近。

---

# 5. 子方向四：Mamba / SSM 轻量化与推理加速

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 核心方法 | 对 STR-RepNet 的映射 |
|---|---|---|---|---|---|
| **VMamba: Visual State Space Model** | NeurIPS 2024，**CCF-A** | Yue Liu，中国科学院大学（UCAS） | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/baa2da9ae4bfed26520bb61d259a3653-Abstract.html) / [Code](https://github.com/MzeroMiko/VMamba) | SS2D 四向 cross-scan，将 1D selective scan 扩展到 2D | HAM encoder 的直接基础；SS2D 是动态递推，不是静态卷积 |
| **Multi-Scale VMamba (MSVMamba)** | NeurIPS 2024，**CCF-A** | Yuheng Shi，City University of Hong Kong | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d69e771d9f274f7c624198ea74f5b98-Abstract.html) / [Code](https://github.com/YuHengsss/MSVMamba) | 原分辨率+下采样图多尺度扫描，减少 multi-scan 冗余 | 支持你把“扫描路径裁剪”当效率对照，而不是结构重参数化 |
| **EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba** | AAAI 2025，**CCF-A** | Xiaohuan Pei，The University of Sydney | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32690) / [Code](https://github.com/TerryPei/EfficientVMamba) | Atrous Selective Scan + conv branch，以 skip sampling 降成本 | 对“SSM 不能折，但能换更便宜扫描”最直接 |
| **MambaOut: Do We Really Need Mamba for Vision?** | CVPR 2025，**CCF-A** | Weihao Yu，National University of Singapore | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html) / [Code](https://github.com/yuweihao/MambaOut) | 移除 SSM token mixer，分析哪些视觉任务真的需要 Mamba | 给你一个重要消融：不是所有层都值得保留 SSM，但 dense task 更可能受益 |
| **Mamba-Adaptor** | CVPR 2025，**CCF-A** | Fei Xie，Shanghai Jiao Tong University | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Mamba-Adaptor_State_Space_Model_Adaptor_for_Visual_Recognition_CVPR_2025_paper.html) | Adaptor-T memory + Adaptor-S 多尺度 dilation conv | 与 HAM 的“SSM + local dilated conv”结构高度相关 |
| **MambaVision** | CVPR 2025，**CCF-A** | Ali Hatamizadeh，NVIDIA | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html) / [Code](https://github.com/NVlabs/MambaVision) | Mamba + Transformer hybrid，在后层引入 attention | 支持你“保留不可约的 attention/SSM，只重参数化线性壳层” |
| **PVMamba: Parallelizing Vision Mamba via Dynamic State Aggregation** | ICCV 2025，**CCF-A** | Fei Xie，Shanghai Jiao Tong University | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Xie_PVMamba_Parallelizing_Vision_Mamba_via_Dynamic_State_Aggregation_ICCV_2025_paper.html) / [Code](https://github.com/VISION-SJTU/PVMamba) | DSA 将顺序状态计算并行化，并使用动态局部采样 | 说明 SSM 加速更合理的方向是算法/并行化，不是强行卷积折叠 |
| **LightMamba** | 2025，**arXiv Preprint** | Renjie Wei | [arXiv](https://arxiv.org/abs/2502.15260) | 量化 + FPGA hardware co-design | 属于硬件侧加速，与你的模型级 Rep 正交 |
| **ChangeMamba** | IEEE TGRS 2024，**SCI 期刊** | Hongruixuan Chen，RIKEN / 合作团队 | [DOI](https://doi.org/10.1109/TGRS.2024.3417253) / [Code](https://github.com/ChenHongruixuan/ChangeMamba) | spatiotemporal state-space model for CD | CD 里的 Mamba 时空建模基线 |
| **ST-Mamba** | IEEE TGRS 2025，**SCI 期刊** | Jiaqi Zhao，China University of Mining and Technology | [DOI](https://doi.org/10.1109/TGRS.2025.3579617) | Spatio-temporal synergistic Mamba，抑制 pseudo-change | 你的 temporal rep 需要与其“时空联合建模”区分：你强调 exact fold，而非新 SSM |
| **SCAM** | IEEE JSTARS，2025 online / Vol.19 2026，**SCI 期刊；不是 CCF-A** | Junyi Zhang，Nanjing University of Aeronautics and Astronautics | [IEEE](https://ieeexplore.ieee.org/document/11282994/) | channel-adaptive scan + difference fusion + lightweight CNN decoder | 可作为 scan pruning/selection 对照，不能误标为 CCF-A |

### 对“SSM 与 CNN 等价性”的判断

不要把“线性 SSM 在 LTI 条件下可写成卷积核”直接推广到 HAM/VMamba 的 selective scan。

Mamba/VMamba 中关键参数（如离散化步长、B/C 等）由输入动态生成，且视觉版本还涉及 scan ordering / cross-scan。它不是一个固定的、输入无关的卷积核。因此：

> **selective scan 不属于你论文中可以做 exact static re-parameterization 的算子。**

可以研究近似替换、低秩、scan reduction、并行化，但那会变成另一条研究线，并且部署前后不会天然 `<1e-6` 等价。

### 本节最值得精读

1. **EfficientVMamba**：SSM 轻量化的最直接高水平路线。
2. **MSVMamba**：多方向扫描冗余的系统分析。
3. **PVMamba**：说明“动态 SSM 的部署优化”更像并行化/状态聚合，而非卷积权重折叠。

---

# 6. 子方向五：混合注意力 / SE / MDTA 的重参数化与部署折叠

## 6.1 最重要的结论

普通注意力不能因为“里面有卷积”就整体折叠。

### SE

\[
g(x)=\sigma(W_2\delta(W_1\operatorname{GAP}(x)))
\]

\[
y=x\odot g(x)
\]

因为 \(g(x)\) 随输入变化，无法变成固定卷积权重 \(W'\)。

### MDTA / Transposed Attention

即使：

\[
Q=W^Q_d W^Q_p x,\;
K=W^K_d W^K_p x,\;
V=W^V_d W^V_p x
\]

其中 projection 是线性卷积，attention core：

\[
A(x)=\operatorname{Softmax}(KQ/\alpha),\qquad
y=V A(x)
\]

仍是输入依赖的乘法，因此不能静态合并成固定卷积。

## 6.2 可用文献

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 与 HAM 的关系 |
|---|---|---|---|---|
| **ASR: Stripe Observation Guided Inference Cost-free Attention Mechanism** | ECCV 2024，**CCF-A** | Zhongzhan Huang，中山大学 | [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3451_ECCV_2024_paper.php) | 论文直接指出 attention 因 multiplicative + input-dependent 不能直接 SRP；是你“SE 不可折叠”最强引用 |
| **HAM-CD** | IEEE TGRS 2026，SCI | Guanlin Li，Xi’an University of Architecture and Technology | [DOI](https://doi.org/10.1109/TGRS.2026.3665418) / [Code](https://github.com/guanguanboy/HAM-CD) | MDTA/TAB、TAM、ICSF 的直接 baseline |
| **MambaVision** | CVPR 2025，CCF-A | Ali Hatamizadeh，NVIDIA | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html) | 高效网络并不要求把 attention 消失掉，可保留少量高价值非线性模块 |
| **RepAttn3D** | Neural Networks 2026（online 2025），SCI 期刊 | Xiusheng Lu，清华大学 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025011943) | 展示“attention re-parameterization”可以通过特定结构设计实现，但不是普通 SE/MDTA 的直接代数折叠 |
| **RepViT** | CVPR 2024，CCF-A | Ao Wang，清华大学 | [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html) | 说明轻量网络可把复杂 token mixing 简化，但不是把任意 attention 强行 fold |

## 6.3 对你最合理的设计方式：只重参数化 attention 的“线性壳层”

建议把 TAB/MDTA 分成：

1. **projection shell**：PWC / DWC / BN 等线性或可静态化算子；
2. **attention core**：reshape、normalization、Softmax、QK、AV；
3. **output projection shell**。

创新可写：

> **Re-parameterized Projection Shell for Transposed Attention**

而不要写：

> “Re-parameterized MDTA attention”

除非你真正替换了 attention core 并证明等价。

### 本节最值得精读

1. **ASR**：不可折叠边界的理论/经验依据。
2. **HAM-CD**：准确画出你的 projection/core 边界。
3. **RepAttn3D**：了解“attention rep”这个词在别人的论文里到底指什么，避免命名撞车。

---

# 7. 子方向六：整网络 / 串联式多重重参数化与跨模块折叠

## 7.1 最相关文献

| 工作 | 年份 / venue | 核心点 | 对你的具体启发 |
|---|---|---|---|
| **DBB** | CVPR 2021，CCF-A 锚点 | 支持 sequential conv branch、multi-scale、avg pool 等复杂分支等价变换 | 你的“相邻无非线性卷积跨模块合并”最重要数学参考 |
| **OREPA** | CVPR 2022，CCF-A 锚点 | 在线把复杂 training block 压成 single conv | 若 decoder-wide Rep 训练显存过高，可借 OREPA 降训练开销 |
| **RepLKNet** | CVPR 2022，CCF-A 锚点 | small-kernel branch→large DW kernel | 大核 DW + dilation folding 基础 |
| **FastViT** | ICCV 2023，CCF-A 锚点 | SRP 消除 skip connection，优化 memory access | “部署算子图更简单”可以单独成为设计目标 |
| **UniRepLKNet** | CVPR 2024，CCF-A | Dilated Reparam Block | 与 d=1/3/5 DWConv 合并最直接 |
| **RepNeXt** | arXiv 2024 | 同时强调 serial + parallel reparameterization | 与你“串联式多重 Rep”字面最接近，应主动引用 |
| **CD-RLKNet** | IJAEO 2024，SCI | 大核 Rep 引入 RSCD | 任务域直接 prior art |
| **LKMamba-CD** | PFG 2026，SCI/专业期刊 | Mamba + reparam large kernel | 任务+骨干双重 prior art |

## 7.2 你可以提出的“折叠代数库”

### A. Conv + BN

推理时 BN 使用 running statistics，可把：

\[
y=\gamma \frac{Wx+b-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

变为：

\[
W'=\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W
\]

\[
b'=\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}(b-\mu)+\beta
\]

这是后续所有级联折叠的第一步。

### B. 并行同形状卷积分支相加

\[
y=\sum_i W_i*x+b_i
\]

先把不同 kernel/dilation 嵌入共同最大 kernel 支撑域：

\[
W_{\text{deploy}}=\sum_i \operatorname{Pad/Embed}(W_i)
\]

\[
b_{\text{deploy}}=\sum_i b_i
\]

### C. 1×1 → k×k 的线性串联

若中间无 ReLU/GELU/SiLU/LN/Softmax：

\[
y=W_2*(W_1*x+b_1)+b_2
\]

可通过通道维 contraction 得到等效 \(k\times k\) kernel。

但必须检查：

- padding；
- bias；
- groups；
- dilation；
- 边界行为。

尤其连续 padded convolution 带 bias 时，边界处可能破坏“简单一个普通 Conv2d”形式的完全等价，需要显式推导/单测，而不能凭公式想当然。

### D. 非对称核

ACNet 的 `1×k`、`k×1` 可 zero-pad 到 `k×k` 后相加。

### E. dilation DWConv

一个 `3×3,d=5` 的有效 kernel 支撑为 11×11。把原 9 个权重放到间隔 5 的位置，其余置零即可。当前 baseline 已经这么做。

## 7.3 “Temporal-Spatial 联合折叠”最容易踩的复杂度陷阱

假设：

- temporal 1×1：`2C -> C`
- 后接 DW k×k：`C -> C`

原图参数约：

\[
2C^2+Ck^2
\]

若强行组成一个普通 `k×k Conv(2C→C)`，参数变成：

\[
2C^2k^2
\]

很可能**大幅上升**。

所以论文不能把“算子数从 2 个变 1 个”自动等同于“更轻量”。

### 更合理的准则

只有当联合折叠同时满足至少一项时才执行：

- 参数不增加；
- FLOPs 不增加；
- 实测 latency 降低且内存访问改善；
- 或最终 kernel 仍保持 group/depthwise/低秩结构。

否则，保留“Temporal Rep 单独折 + Spatial Rep 单独折”的两级部署图反而更优。

### 本节最值得精读

1. **DBB**：串联折叠。
2. **UniRepLKNet**：dilation→大核。
3. **FastViT**：算子图/latency 视角。

---

# 8. 子方向七：异构多教师蒸馏 / DML / OFA / Slimmable

## 8.1 代表工作

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | 核心方法 | 与你的备选映射 |
|---|---|---|---|---|---|
| **OFA-KD** | NeurIPS 2023，CCF-A 锚点 | Zhiwei Hao，北京理工大学 / Huawei Noah’s Ark Lab | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html) | 投影到 architecture-agnostic latent/logit space | VMamba 与 DINO/SAM/Conv teacher 不应直接盲做 raw feature MSE |
| **MTKD-RL** | AAAI 2025，CCF-A | Chuanguang Yang，中科院计算所 | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32990) | RL 动态分配多教师权重 | 3 个异构 encoder 教师的直接方法参考 |
| **PAT** | ICCV 2025，CCF-A | Jhe-Hao Lin，NYCU | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_Perspective-Aware_Teaching_Adapting_Knowledge_for_Heterogeneous_Distillation_ICCV_2025_paper.html) | student feedback prompt + region-aware attention 解决 view mismatch | 更适合跨架构 feature KD |
| **Diff-DML** | CVPR 2025，CCF-A | Han Liu | [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Improving_Accuracy_and_Calibration_via_Differentiated_Deep_Mutual_Learning_CVPR_2025_paper.html) | 保持 ensemble diversity 的 mutual learning | 若 3 个 encoder 都训练且互教，可引用 |
| **ViT-Linearizer** | ICCV 2025，CCF-A | Guoyizhe Wei | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_ViT-Linearizer_Distilling_Quadratic_Knowledge_into_Linear-Time_Vision_Models_ICCV_2025_paper.html) | ViT teacher→linear recurrent/Mamba-like student | Mamba student 的高质量先例 |
| **MaTVLM** | ICCV 2025，CCF-A | Yingyue Li，HUST-VL | [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Li_MaTVLM_Hybrid_Mamba-Transformer_for_Efficient_Vision-Language_Modeling_ICCV_2025_paper.html) | Transformer teacher→Mamba-2 hybrid | 进一步证明 distill-to-Mamba 可行 |
| **Tiny-vGamba** | ICCVW 2025，非主会 | Yunusa Haruna | [CVF](https://openaccess.thecvf.com/content/ICCV2025W/ECLR/html/Haruna_Tiny-vGamba_Distilling_Large_Vision-Language_Knowledge_from_CLIP_into_a_Lightweight_ICCVW_2025_paper.html) | frozen CLIP teacher→lightweight Mamba-like student | 与“冻结 DINO/SAM 辅助训练”非常相似 |
| **MHKD** | Neural Networks 2026，SCI | Sai Yang | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608026006209) | Transformer→Mamba heterogeneous KD | 术语定位证据 |

## 8.2 Once-for-All / Slimmable 为什么不是你的首选术语

**Once-for-All / Slimmable** 典型逻辑：

- 单个 supernet；
- 权重共享；
- 不同 width/depth/kernel 子网在同一权重空间训练；
- 部署时抽取某一子网。

你的备选：

- 多个完整异构 encoder；
- 参数不共享；
- 辅助分支输出作为监督信号；
- 最后只保留 VMamba。

所以它不是标准 once-for-all，也不是 slimmable。

## 8.3 是否已有“异构多编码器蒸馏→单 Mamba 变化检测网络”先例？

本次在 2024–2026 公开可核验高水平文献中：

- 已核验到 Transformer/ViT/CLIP → Mamba student 的跨任务先例；
- 已核验到多教师 KD；
- 已核验到遥感 CD 中 Mamba、SAM2 等独立路线；
- **未核验到一篇把 DINOv2/SAM/CNN 等异构多编码器共同作为教师，最终压到单个 VMamba Siamese encoder，并专门面向全监督二值 RSCD 的高水平已发表工作。**

因此这条路线可能有新颖性，但它会把论文主叙事从“结构重参数化”拉向“KD + 多教师训练”，建议只作为：

- 备选；
- 或独立训练策略附录；
- 或后续论文。

### 本节最值得精读

1. PAT。
2. MTKD-RL。
3. ViT-Linearizer。

---

# 9. 子方向八：二时相交互建模的可折叠设计

## 9.1 2024–2026 代表 RSCD 工作

| 工作 | 年份 / venue | 一作+机构 | 官方链接 | Temporal / Difference 机制 | 与你的映射 |
|---|---|---|---|---|---|
| **CD-RLKNet** | IJAEO 2024，SCI | Junqing Huang，Macao Polytechnic University | [Publisher](https://research.mpu.edu.mo/en/publications/large-kernel-convolution-application-for-land-cover-change-detect) | Spatial and Temporal Adaptive Fusion + bitemporal integration | 有“时空融合”，但不是你这种 exact topology folding |
| **MixCDNet** | IEEE TGRS 2024，SCI | Linlin Wang，哈尔滨工业大学（合作作者含 University of Trento） | [IEEE DOI](https://doi.org/10.1109/TGRS.2024.3438228) | CNN/Transformer local-global bidirectional mixing | 0.32M/低 FLOPs 提醒你 temporal rep 必须真的轻 |
| **RFANet** | ISPRS JPRS 2024，**权威 SCI 期刊** | Zhi-Hui You；Anhui University 系合作 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S092427162400251X) / [Code](https://github.com/Youzhihui/RFANet) | multi-level feature reinforcement + refined difference + lightweight decoder | 轻量 CD 的强竞争背景 |
| **Efficient Adjacent Feature Harmonizer Network** | IEEE TGRS 2024，SCI | Yikui Zhai，Wuyi University | [DOI](https://doi.org/10.1109/TGRS.2024.3502768) | 邻级特征 harmonization，强调 ultralightweight | 用于证明“复杂模块堆叠不是唯一解” |
| **M2M-LINet: From Macro to Micro** | IEEE TGRS 2025，SCI | Yetong Xu，Shaanxi University of Science and Technology | [DOI](https://doi.org/10.1109/TGRS.2025.3548562) | macro→micro interleaved lightweight CD | 你的竞争对象应同时看 accuracy 与 efficiency |
| **Difference-Aware Multiscale Feature Aggregation Network (DMFANet)** | IEEE TGRS 2025，SCI | Tao Zhan，Northwest A&F University | [DOI](https://doi.org/10.1109/TGRS.2025.3560977) | FMM + cross-domain difference enhancement + multiscale context | 说明 difference 建模已经很拥挤；你的 novelty 要落在“可折叠” |
| **ST-Mamba** | IEEE TGRS 2025，SCI | Jiaqi Zhao，China University of Mining and Technology | [DOI](https://doi.org/10.1109/TGRS.2025.3579617) | spatio-temporal synergistic Mamba | 与你的 temporal-spatial 叙事接近，但机制不是 algebraic folding |
| **Difference Enhancement and Interscale Interactive Fusion Mamba** | IEEE TGRS 2025，SCI | Weiwei Sun，Ningbo University | [DOI](https://doi.org/10.1109/TGRS.2025.3628639) | difference enhancement + interscale interactive fusion | 同样强调差分/跨尺度；你必须突出 deployment equivalence |
| **SCAM** | IEEE JSTARS 2025/2026，SCI，非 CCF-A | Junyi Zhang，NUAA | [IEEE](https://ieeexplore.ieee.org/document/11282994/) | difference fusion + adaptive scan | 可作为“动态差异融合 vs 静态可折叠 temporal rep”对照 |
| **HAM-CD** | IEEE TGRS 2026，SCI | Guanlin Li，Xi’an University of Architecture and Technology | [DOI](https://doi.org/10.1109/TGRS.2026.3665418) / [Code](https://github.com/guanguanboy/HAM-CD) | concat embedding + HAM global/local modeling + ICSF | 你的 baseline |

## 9.2 Temporal Rep 的严格条件

你的：

\[
Y=\operatorname{Conv}_c([X_1,X_2])
+\operatorname{Conv}_s(X_1+X_2)
+\operatorname{Conv}_d(X_2-X_1)
\]

可以精确合并，但必须满足：

1. 三个 branch 的输出 shape 相同；
2. stride/padding/dilation 对应一致，或能够映射到共同 kernel 支撑；
3. 不在 branch 内放 ReLU/GELU/SiLU；
4. BN 必须先在 eval 状态折进 conv；
5. branch weight 若使用门控，门控必须是训练后常数，而不能依赖当前输入；
6. `Diff` 必须是 signed linear difference，不是 `abs diff`；
7. A/B 的几何对齐/增强必须同步，否则数学等价虽然成立，CD 学习本身会失真。

## 9.3 为什么这比普通“difference fusion”更适合作为你的第二主创新

普通 CD 论文通常回答：

> “哪种融合更准？”

你的机制回答：

> “如何让训练时同时学习原始双时相、共同成分与方向性变化成分，但部署时恢复成与普通 concat projection 完全同复杂度的一条路径？”

这直接把 **CD domain prior** 和 **structural re-parameterization** 连接起来，论文叙事比“再加一个 difference module”更集中。

### 本节最值得精读

1. **DMFANet (TGRS 2025)**：看差异特征设计已经做到什么程度。
2. **ST-Mamba (TGRS 2025)**：看 spatial-temporal 联合建模怎么叙述。
3. **CD-RLKNet (2024)**：看 temporal fusion + Rep prior art 的交叉。

---

# 10. 全局 Top-15 优先精读清单

> 排序依据是“对 STR-RepNet 机制设计和审稿碰撞风险的直接程度”，不是给论文做学术价值排名。

| 优先级 | 文献 | 为什么先读 |
|---:|---|---|
| 1 | **HAM-CD, TGRS 2026** | Baseline，必须把 TAM/TAB/ICSF 的线性与非线性边界逐层画清 |
| 2 | **CD-RLKNet, IJAEO 2024** | 变化检测 + reparameterized large kernel 的直接 prior art |
| 3 | **UniRepLKNet, CVPR 2024** | dilation reparam 与 TAM StateFusion 机制高度接近 |
| 4 | **DBB, CVPR 2021** | 你“串联折叠 / 跨模块合并”的数学核心 |
| 5 | **ASR, ECCV 2024** | 回答“为什么 SE/attention 不能直接折” |
| 6 | **LKMamba-CD, 2026** | Mamba + reparam large kernel + CD，最危险的新碰撞 |
| 7 | **RepViT, CVPR 2024** | 轻量部署与移动 latency 的强参考 |
| 8 | **OREPA, CVPR 2022** | decoder-wide 多分支训练开销过高时的解决思路 |
| 9 | **EfficientVMamba, AAAI 2025** | SSM 侧轻量化边界 |
| 10 | **MSVMamba, NeurIPS 2024** | multi-scan 冗余分析 |
| 11 | **PVMamba, ICCV 2025** | 动态 SSM 加速的正确方向：并行化而非静态折叠 |
| 12 | **DMFANet, TGRS 2025** | 差异融合领域的强近邻 |
| 13 | **ST-Mamba, TGRS 2025** | spatial-temporal Mamba 近邻 |
| 14 | **PAT, ICCV 2025** | 备选异构多教师 feature KD |
| 15 | **ViT-Linearizer, ICCV 2025** | Mamba/linear student 的跨架构蒸馏先例 |

---

# 11. 我判断的 3 个“创新点空白”

## 空白 1：Decoder-Wide Compositional Re-parameterization

### 现有文献普遍做法

- 对一个 block 做 RepVGG / DBB；
- 对一个 large-kernel block 做 dilation rep；
- 在 CD 中把 reparam large kernel 当一个 feature extractor。

### 你可以做的不同点

把 HAM decoder 建模成有类型的算子图：

\[
G=(V_{\text{linear}},V_{\text{dynamic}},E)
\]

其中：

- `linear/foldable`：Conv、DWConv、BN(eval)、常数 scale、residual sum、concat→linear projection；
- `dynamic/non-foldable`：Selective Scan、SE、Softmax attention、activation、LayerNorm（一般不能跨输入静态吸收）。

设计一个固定的折叠顺序：

1. Branch normalization；
2. BN folding；
3. parallel kernel embedding；
4. local serial composition；
5. branch sum；
6. optional cross-block composition；
7. export deployment graph。

这样贡献就从“一个 RepBlock”提升到“**HAM decoder 的全图级结构重参数化原则**”。

### 可证伪假设

在部署 Params/FLOPs 不高于 baseline 的前提下：

- training multi-branch 能显著提升至少 2/4 数据集 F1/IoU；
- 4 数据集均无系统性退化；
- fold 前后 `max_abs_error < 1e-6`（FP32 eval，同输入）；
- 部署 graph 的算子数/latency 有明确下降。

### 失败判据

若收益只在单数据集单 seed <0.2 F1，或部署计算量上升，不能作为主创新。

---

## 空白 2：Change-Specific Temporal Algebraic Re-parameterization

### 核心新意

不是“新差分模块”，而是：

> 利用二时相线性拓扑本身的代数冗余，把训练期 Concat / Sum / signed-Diff 的互补归纳偏置压到部署期一个标准 projection。

### 为什么有论文价值

这不是把通用 RepVGG 搬进 CD，而是从 bitemporal data topology 推导出的任务专属 reparameterization。

### 最小消融

- Baseline concat；
- concat + sum（train multi-branch，deploy fold）；
- concat + diff；
- concat + sum + signed diff（Full Temporal Rep）；
- `abs diff` 只作为“不可精确 fold”的性能上界对照，不纳入 deploy-equivalent 主模型。

### 验证

随机权重时先做 algebra unit test，再训练，避免把“训练后结果差”与“公式写错”混在一起。

---

## 空白 3：受复杂度约束的 Temporal–Spatial Joint Folding

不建议把“全部强行合成 1 个卷积”本身作为目标。建议定义：

> **Cost-aware compositional folding**

仅当：

\[
\Delta Params \le 0,\quad \Delta FLOPs \le 0
\]

或实测 latency 明显改善时才执行跨层合并。

否则保持两个已经各自 reparameterized 的轻量算子。

这比单纯“cascade folding”更可信，因为它正面处理了 dense-kernel explosion。

---

# 12. 两条主线 + 一条备选的不可行陷阱

## 12.1 P0：会直接破坏“严格结构重参数化”定义

### 陷阱 A：折 selective scan

Selective SSM 是输入依赖的动态递推，不是固定 kernel。直接宣称“SSM 可折成 Conv”会被质疑。

### 陷阱 B：折 SE

SE 的 GAP→MLP→sigmoid 权重随样本变化，不能变成固定卷积参数。

### 陷阱 C：折 Softmax attention

Q/K/V projection 可局部处理，QK Softmax 和 AV 不行。

### 陷阱 D：把 abs difference 放进 Temporal Rep

`abs()` 非线性，三路无法精确线性合并。

### 陷阱 E：跨 ReLU/GELU/SiLU/LN 级联

\[
W_2\phi(W_1x)\neq W'x
\]

一般不存在固定 \(W'\)。

---

## 12.2 P1：数学上能合，但会让模型更重

### 陷阱 F：1×1 temporal mixing + DW k×k 全部合成 dense k×k

可能大幅增加 Params/FLOPs。必须先计算部署复杂度再决定是否合。

### 陷阱 G：只报 FLOPs，不报 latency

RepGhost、MobileOne、FastViT 都提示：concat、branch、memory access、kernel 实现会显著影响真实延迟。论文至少建议报告：

- Params；
- FLOPs；
- FPS / image latency；
- peak memory（可选）；
- train/deploy operator count（建议）。

---

## 12.3 P1：创新撞车

### 陷阱 H：“首次在遥感变化检测用结构重参数化”

CD-RLKNet 已经否定这个表述。

### 陷阱 I：“首次把 Mamba 与 reparameterized large kernel 用于 CD”

LKMamba-CD 已构成直接风险。

### 陷阱 J：“首次把 dilation 1/3/5 合成大核”

UniRepLKNet、RepLKNet 体系以及当前 baseline 自身都已经覆盖相关思想。

真正应该声称的，是**系统范围、组合层级、任务特有 temporal folding 与严格 deploy equivalence**。

---

## 12.4 P1：把 KD 误写成 Rep

编码器多教师路线如果没有显式代数等价变换，就应单独归入 KD。最安全的论文结构是：

- 主方法：Structural Re-parameterization；
- optional training strategy：Heterogeneous KD；
- 不把两者混成一个“统一 Rep 理论”。

否则 Related Work 和理论定义会变得很难自洽。

---

# 13. 推荐的论文 Related Work 组织方式

## 13.1 Structural Re-parameterization

写作链条：

**ACNet/RepVGG → DBB → RepLKNet/OREPA → MobileOne/FastViT → RepViT/UniRepLKNet → ASR**

强调：

- 早期：parallel branch；
- 中期：diverse/large-kernel/online；
- 近期：mobile deployment、large-kernel universal representation、attention-alike cost-free；
- 缺口：鲜有工作把 SRP 从“局部 block”推进到 binary CD decoder 的 temporal-spatial compositional graph。

## 13.2 Efficient Mamba for Dense Vision and Change Detection

写作链条：

**VMamba → MSVMamba/EfficientVMamba → Mamba-Adaptor/PVMamba/MambaVision → ChangeMamba/ST-Mamba/SCAM/HAM-CD**

强调：

- SSM 解决 global dependency 与线性复杂度；
- 但 scan redundancy/local modeling 仍需优化；
- 你的工作不改 encoder，不与这些工作争“新 SSM”，而是优化 HAM decoder 的可部署线性子图。

## 13.3 Lightweight Remote Sensing Change Detection and Temporal Fusion

重点：

**RFANet / MixCDNet / M2M-LINet / DMFANet / CD-RLKNet / HAM-CD**

最后自然引出：

> Existing methods mainly optimize feature interaction quality, whereas the train-time temporal topology and inference-time topology are usually identical. STR-RepNet instead decouples temporal representation learning from deployment topology, using algebraically equivalent structural re-parameterization to obtain richer temporal inductive biases during training without retaining multi-branch temporal operators at inference.

---

# 14. 建议的方法命名与贡献边界

## 14.1 主模块命名候选

### 候选 A：Decoder-wide Compositional Re-parameterization（DCR）

优点：突出“整解码器 + 可组合”。

### 候选 B：Spatial-Temporal Algebraic Re-parameterization（STAR）

优点：突出时空联合；但 STAR 很常见，检索撞名风险高。

### 候选 C：Temporal-Spatial Structural Re-parameterization（TSSR）

稳妥，但辨识度略低。

### 当前更建议

> **STR-RepNet：Spatial-Temporal Re-parameterization Network**

内部两个明确贡献：

- **DCR — Decoder-wide Compositional Re-parameterization**
- **TAR — Temporal Algebraic Re-parameterization**

这样论文不会把每个小 RepConv 都包装成一个模块。

---

# 15. 实验设计：让“创新”可证伪而不是靠调参解释

## 15.1 最小主实验

四数据集：

- CDD-CD-256
- LEVIR-CD-256
- SYSU-CD-256
- WHU-CD-256

统一：

- A/B/label + list；
- `gray >= 128`；
- 相同训练预算；
- 相同 seed；
- 相同 pretrained VMamba；
- encoder 不动。

## 15.2 核心消融顺序

| Exp | 变量 | 目的 |
|---|---|---|
| E0 | HAM-CD repo baseline | 干净基线 |
| E1 | 只做 local Spatial Rep | 验证通用卷积 rep 是否有收益 |
| E2 | 只做 Temporal Rep | 验证任务特有创新 |
| E3 | Decoder-wide Rep，不做 cross-block fold | 验证“覆盖范围” |
| E4 | + safe cascade fold | 验证级联折叠 |
| E5 | + cost-aware temporal-spatial joint fold | 验证联合折叠是否真的更快 |
| E6 | 去掉不可折叠 SE/SSM 的“错误折叠替代”对照 | 证明保留 dynamic operators 的合理性 |

## 15.3 每个实验必须报告

- Recall
- Precision
- OA
- F1
- IoU
- Kappa
- training params
- deployment params
- deployment FLOPs
- inference latency / FPS（强烈建议）
- deploy 前后主预测最大绝对误差

## 15.4 必做部署一致性测试

```text
model.eval()
x1, x2 = fixed_random_input

y_train_graph = model_before_convert(x1, x2)
model.switch_to_deploy()
y_deploy_graph = model_after_convert(x1, x2)

max_err = max(abs(y_train_graph - y_deploy_graph))
assert max_err < 1e-6
```

同时测试：

- CPU FP32；
- GPU FP32；
- 至少 10 组随机 shape/输入；
- 各 Rep block 单元测试；
- 整模型 end-to-end 测试。

---

# 16. 文献清单索引：按“主证据 / 辅助 / 经典锚点”分类

## 16.1 2024–2026 主证据

1. Wang A., et al. **RepViT: Revisiting Mobile CNN From ViT Perspective.** CVPR 2024.  
   <https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html>

2. Ding X., et al. **UniRepLKNet: A Universal Perception Large-Kernel ConvNet...** CVPR 2024.  
   <https://doi.org/10.1109/CVPR52733.2024.00527>

3. Fei X., et al. **RepAn: Enhanced Annealing through Re-parameterization.** CVPR 2024.  
   <https://openaccess.thecvf.com/content/CVPR2024/html/Fei_RepAn_Enhanced_Annealing_through_Re-parameterization_CVPR_2024_paper.html>

4. Zhang Y., et al. **Multimodal Pathway: Improve Transformers with Irrelevant Data from Other Modalities.** CVPR 2024.  
   <https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Pathway_Improve_Transformers_with_Irrelevant_Data_from_Other_Modalities_CVPR_2024_paper.html>

5. Huang Z., et al. **Stripe Observation Guided Inference Cost-free Attention Mechanism.** ECCV 2024.  
   <https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3451_ECCV_2024_paper.php>

6. Liu Y., et al. **VMamba: Visual State Space Model.** NeurIPS 2024.  
   <https://proceedings.neurips.cc/paper_files/paper/2024/hash/baa2da9ae4bfed26520bb61d259a3653-Abstract.html>

7. Shi Y., et al. **Multi-Scale VMamba: Hierarchy in Hierarchy Visual State Space Model.** NeurIPS 2024.  
   <https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d69e771d9f274f7c624198ea74f5b98-Abstract.html>

8. Pei X., et al. **EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba.** AAAI 2025.  
   <https://ojs.aaai.org/index.php/AAAI/article/view/32690>

9. Yu W., Wang X. **MambaOut: Do We Really Need Mamba for Vision?** CVPR 2025.  
   <https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html>

10. Xie F., et al. **Mamba-Adaptor: State Space Model Adaptor for Visual Recognition.** CVPR 2025.  
    <https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Mamba-Adaptor_State_Space_Model_Adaptor_for_Visual_Recognition_CVPR_2025_paper.html>

11. Hatamizadeh A., Kautz J. **MambaVision: A Hybrid Mamba-Transformer Vision Backbone.** CVPR 2025.  
    <https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html>

12. Xie F., et al. **PVMamba: Parallelizing Vision Mamba via Dynamic State Aggregation.** ICCV 2025.  
    <https://openaccess.thecvf.com/content/ICCV2025/html/Xie_PVMamba_Parallelizing_Vision_Mamba_via_Dynamic_State_Aggregation_ICCV_2025_paper.html>

13. Lin J.-H., et al. **Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation.** ICCV 2025.  
    <https://openaccess.thecvf.com/content/ICCV2025/html/Lin_Perspective-Aware_Teaching_Adapting_Knowledge_for_Heterogeneous_Distillation_ICCV_2025_paper.html>

14. Wei G., Chellappa R. **ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models.** ICCV 2025.  
    <https://openaccess.thecvf.com/content/ICCV2025/html/Wei_ViT-Linearizer_Distilling_Quadratic_Knowledge_into_Linear-Time_Vision_Models_ICCV_2025_paper.html>

15. Li Y., et al. **MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling.** ICCV 2025.  
    <https://openaccess.thecvf.com/content/ICCV2025/html/Li_MaTVLM_Hybrid_Mamba-Transformer_for_Efficient_Vision-Language_Modeling_ICCV_2025_paper.html>

16. Yang C., et al. **Multi-Teacher Knowledge Distillation with Reinforcement Learning for Visual Recognition.** AAAI 2025.  
    <https://ojs.aaai.org/index.php/AAAI/article/view/32990>

17. Liu H., et al. **Improving Accuracy and Calibration via Differentiated Deep Mutual Learning.** CVPR 2025.  
    <https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Improving_Accuracy_and_Calibration_via_Differentiated_Deep_Mutual_Learning_CVPR_2025_paper.html>

18. Huang J., et al. **Large kernel convolution application for land cover change detection of remote sensing images (CD-RLKNet).** IJAEO 2024.  
    <https://doi.org/10.1016/j.jag.2024.104077>

19. You Z.-H., et al. **Robust feature aggregation network for lightweight and effective remote sensing image change detection.** ISPRS JPRS 2024.  
    <https://doi.org/10.1016/j.isprsjprs.2024.06.013>

20. Wang L., Zhang J., Bruzzone L. **MixCDNet: A Lightweight Change Detection Network Mixing Features Across CNN and Transformer.** IEEE TGRS 2024.  
    <https://doi.org/10.1109/TGRS.2024.3438228>

21. Zhai Y., et al. **Efficient Adjacent Feature Harmonizer Network With UAV-CD+ Dataset for Remote Sensing Change Detection.** IEEE TGRS 2024.  
    <https://doi.org/10.1109/TGRS.2024.3502768>

22. Chen H., et al. **ChangeMamba: Remote Sensing Change Detection with Spatiotemporal State Space Model.** IEEE TGRS 2024.  
    <https://doi.org/10.1109/TGRS.2024.3417253>

23. Xu Y., et al. **From Macro to Micro: A Lightweight Interleaved Network for Remote Sensing Image Change Detection.** IEEE TGRS 2025.  
    <https://doi.org/10.1109/TGRS.2025.3548562>

24. Zhan T., et al. **Difference-Aware Multiscale Feature Aggregation Network for Building Change Detection.** IEEE TGRS 2025.  
    <https://doi.org/10.1109/TGRS.2025.3560977>

25. Zhao J., et al. **ST-Mamba: Spatio-Temporal Synergistic Model for Remote Sensing Change Detection.** IEEE TGRS 2025.  
    <https://doi.org/10.1109/TGRS.2025.3579617>

26. Sun W., et al. **Difference Enhancement and Interscale Interactive Fusion Mamba for Remote Sensing Image Change Detection.** IEEE TGRS 2025.  
    <https://doi.org/10.1109/TGRS.2025.3628639>

27. Zhang J., et al. **SCAM: Scan Channel Attention Mamba-Based Network for Remote Sensing Change Detection.** IEEE JSTARS, online 2025 / Vol.19 2026.  
    <https://ieeexplore.ieee.org/document/11282994/>

28. Li G., et al. **HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection.** IEEE TGRS 2026.  
    <https://doi.org/10.1109/TGRS.2026.3665418>

29. Yu Y., et al. **LKMamba-CD: Large Kernel State Space Model for Remote Sensing Change Detection.** PFG 2026.  
    <https://doi.org/10.1007/s41064-026-00407-9>

30. Yang S., et al. **Image restoration model compression via mamba-oriented heterogeneous knowledge distillation.** Neural Networks 2026.  
    <https://doi.org/10.1016/j.neunet.2026.109159>

## 16.2 预印本 / 辅助证据，引用时要明确状态

31. Zhao M., Luo Y., Ouyang Y. **RepNeXt.** arXiv 2024.  
    <https://arxiv.org/abs/2406.16004>

32. Wei R., et al. **LightMamba.** arXiv 2025.  
    <https://arxiv.org/abs/2502.15260>

33. Haruna Y., et al. **Tiny-vGamba.** ICCV Workshops 2025，非主会。  
    <https://openaccess.thecvf.com/content/ICCV2025W/ECLR/html/Haruna_Tiny-vGamba_Distilling_Large_Vision-Language_Knowledge_from_CLIP_into_a_Lightweight_ICCVW_2025_paper.html>

34. **RepFC.** CVPR Workshops 2025，非主会。  
    <https://portal.fis.tum.de/en/publications/repfc-universal-structural-reparametrization-block-for-high-perfo/>

35. **ESIFCD.** IEEE Access 2025，直接 prior-art 辅助。  
    <https://doi.org/10.1109/ACCESS.2025.3612956>

## 16.3 经典理论锚点

36. Ding X., et al. **ACNet.** ICCV 2019.  
    <https://openaccess.thecvf.com/content_ICCV_2019/html/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.html>

37. Ding X., et al. **RepVGG.** CVPR 2021.  
    <https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html>

38. Ding X., et al. **Diverse Branch Block.** CVPR 2021.  
    <https://openaccess.thecvf.com/content/CVPR2021/html/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.html>

39. Ding X., et al. **RepLKNet.** CVPR 2022.  
    <https://openaccess.thecvf.com/content/CVPR2022/html/Ding_Scaling_Up_Your_Kernels_to_31x31_Revisiting_Large_Kernel_Design_CVPR_2022_paper.html>

40. Hu M., et al. **OREPA.** CVPR 2022.  
    <https://openaccess.thecvf.com/content/CVPR2022/html/Hu_Online_Convolutional_Re-Parameterization_CVPR_2022_paper.html>

41. Chen C., et al. **RepGhost.** arXiv 2022（预印本）。  
    <https://arxiv.org/abs/2211.06088>

42. Ding X., et al. **Re-parameterizing Your Optimizers rather than Architectures.** ICLR 2023.  
    <https://openreview.net/forum?id=B92TMCG_7rp>

43. Vasu P. K. A., et al. **MobileOne.** CVPR 2023.  
    <https://openaccess.thecvf.com/content/CVPR2023/html/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.html>

44. Vasu P. K. A., et al. **FastViT.** ICCV 2023.  
    <https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html>

45. Hao Z., et al. **One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation.** NeurIPS 2023.  
    <https://proceedings.neurips.cc/paper_files/paper/2023/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html>

---

# 17. 最终路线建议

## 主线 1：保留，并升级表述

建议核心标题：

> **Decoder-wide Compositional Structural Re-parameterization for Hybrid Mamba Change Decoding**

重点不是“多少个 RepConv”，而是：

- foldability taxonomy；
- compositional conversion；
- operator-graph simplification；
- exact-equivalence verification；
- cost-aware folding。

## 主线 2：强烈保留，作为变化检测专属贡献

建议标题：

> **Temporal Algebraic Re-parameterization for Bi-temporal Change Representation**

这是最容易与通用视觉 SRP 区分的部分。

## 备选编码器多教师：暂不纳入主论文核心

除非主线 1+2 在四数据集上收益不足，否则不建议把 DINOv2/SAM 多教师再叠进来。原因不是它没价值，而是：

- 论文会同时讲 Rep、KD、Mamba、multi-teacher；
- contribution 边界会变散；
- 训练成本显著上升；
- 它不满足“train/deploy exact algebraic equivalence”的主叙事。

更适合作为第二阶段独立方向。

---

# 18. 立即执行顺序

1. **先冻结 baseline**：保存当前 HAM-CD 复现结果、Params/FLOPs、四数据集协议。
2. **写 foldability map**：逐层标注 `[exact fold] / [local fold only] / [non-foldable]`。
3. **先实现 Temporal Rep 单元测试**：不训练，随机 tensor 直接验证转换误差 `<1e-6`。
4. **实现一个最小 Spatial Rep block**：优先 ICSF 的 DWConv / FFN DWConv，避免先碰 SSM/attention core。
5. **跑 E0/E1/E2**：判断“通用 Spatial Rep”和“任务专属 Temporal Rep”哪个贡献更有效。
6. **再做 decoder-wide coverage**，不要一开始把所有模块同时改掉。
7. **最后做 safe cascade fold**：只有在无非线性、复杂度不升的区段启用。
8. **真实部署统计**：deploy params、FLOPs、latency、operator count、转换误差。
9. **只有主线效果不足时**再启用 heterogeneous multi-teacher KD。

---

# 19. 仍需补充的证据

后续若进入正式论文写作，建议再做三轮“窄检索”，而不是继续泛搜：

1. **IEEE Xplore / Scopus 精确查重**：`("change detection" AND "structural re-parameterization")`、`("bitemporal" AND "re-parameterization")`，确认 2026 新收录论文没有完全相同的 Temporal Rep。
2. **专门检索 serial/cascade structural reparameterization**：验证“cross-block compositional folding”是否已有更直接 2025–2026 正式工作。
3. **专门检索 attention reparameterization**：除 ASR/RepAttn3D 外，确认是否有 2026 CCF-A 对 channel attention 做 exact/static deployment folding 的新工作。

任何“首次”措辞，都应在论文投稿前再做一次以投稿日期为截止时间的检索。

---

## 一句话总判断

**这条课题仍有论文空间，但创新点不能停留在“把 HAM-CD 的若干卷积换成 RepConv”。当前最有辨识度、也最符合结构重参数化严格定义的路线，是“整解码器可组合折叠 + 变化检测专属 Temporal Algebraic Rep”，并明确把 SSM/SE/Softmax attention 留作不可约动态核；异构多编码器路线则应独立归为多教师知识蒸馏，而不是重参数化。**
