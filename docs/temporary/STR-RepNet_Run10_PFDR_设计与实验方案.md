# STR-RepNet Run10：PFDR 预融合扩域重参数化——设计与实验方案

> 项目：STR-RepNet（Lightweight Spatial-Temporal Structural Re-parameterization Network for Remote Sensing Change Detection）  
> 任务：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256 全监督遥感二值变化检测  
> 当前仓库审查版本：`main @ 7437133c37035a429549b9d802f3c1b5a79ed0d2`（2026-10-01）  
> 当前正式锚点：Run2 `full_last2`  
> 硬部署锚点：`Params <= 28,828,706`，`FLOPs <= 12.6062G`  
> 固定训练协议：seed=2333，300 epoch，batch=16，CE + 2×Lovász，test-as-val 挑 best  
> 正式结果纪律：只读取同一 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块

---

## 0. 先给结论：为什么 Run9 已“结束结构搜索”，现在仍值得重开一次

### 0.1 决策

**可以重开，但只建议重开一个与 Run3–Run9 的“参数化函数类”真正不同的方向：PFDR（Pre-Fusion Dilated Re-parameterization，预融合扩域重参数化）。**

Run9 在当时“论文收尾”的目标下结束结构搜索是合理的；现在目标已经改变为四数据集硬门槛，尤其 LEVIR `0.9144 -> >=0.9250`、SYSU `0.8345 -> >=0.8500`，因此旧的“停止搜索”不再是逻辑上的永久禁令。但重开必须满足一个更严格条件：**不能继续在相同 deploy operator 的同一函数类里堆训练分支。**

Run3/6/8/9 的共同负证据不是“分支没有学起来”，而是：

- Run3 Edge-Basis：空间结构分支可折叠，但四集净效应均非正；
- Run6 NSCR：LEVIR Recall `+0.41pp`、Precision `-0.52pp`，F1 反而下降；
- Run7 PBRU：branch 学到非零 phase structure，但主要体现为 Precision/置信度提升，rep 净增益只有 `+0.09pp`；
- Run8 MPCR：branch γ≈0.9，仍出现 Recall↓ / Precision↑，最终 d1 判别力不升；
- Run9 BiFTR：`||Δin||F=0.67`、`||Δout||F=1.21`、有效核变化约 13%、cross-term 5.65%，说明双侧乘性项确实被利用，但最终仍是 Recall `-0.05pp` / Precision `+0.17pp`、F1 仅 `+0.03pp`。

因此，继续设计“另一个 1×1 / 3×3 additive branch、另一个 channel basis、另一个 serial factor”没有充分依据。

### 0.2 PFDR 与前三类失败参数化的实质差异

PFDR 不把创新放在“同一个已有 deploy kernel 如何被训练得更花哨”，而是改变 **fine-scale lateral 在跨尺度融合前的 deploy spatial support**：

```text
Run2 当前 fine path：
TAR-stage1 t1
    -> fuse1( t1, up(d2) )       # 1×1 跨尺度混合
    -> block1.dw3                # 融合后才做局部空间建模
    -> block1.pw
    -> refine

Run10 PFDR：
TAR-stage1 t1
    -> PFDR-DW5(t1)              # 先在 fine lateral 上做 5×5 空间条件化
    -> fuse1( ..., up(d2) )
    -> 原 block1 / refine 不变
```

也就是说：**空间信息先被保真/扩域，再与高层语义混合。** 这直接针对 Run6 暴露的 small-object FN，而不是在最终 logits 或已经融合后的表示上做 calibration。

训练时 PFDR-DW5 用多分支（5×5 主分支 + 3×3 近邻 + dilated-3×3 远邻 + 1×1 中心）增强优化；部署时所有分支解析折叠为**一个 DW5×5**。因此：

- 训练图 richer；
- 部署图仍是单路径静态卷积；
- 不引入 attention / gate / teacher；
- 不改变 backbone；
- 不依赖 threshold/loss/sweep；
- C0 与 M1 部署图完全相同，可单独识别“deploy topology”与“structural reparameterization”贡献。

### 0.3 重要：两个现有约束冲突必须显式记录

**P0-A：whole-model `<1e-6` 与仓库现有实测不一致。** 项目长期约束写过“部署前后主预测误差 <1e-6”，但当前 Run2–Run9 的 FP32 whole-model fold 实测约为 `1e-5 ~ 1e-4`，Run9 约 `6.7e-5`，同时 argmax disagreement=0。Run9 文档也明确采用“FP64 解析组合 + 最后 FP32 cast + whole-model 记录实际误差”的口径。因此 Run10 不能偷偷把两种口径混为一谈：

1. 新增 PFDR 的 **FP64 解析代数必须达到 `<1e-10`，目标 `<1e-12`**；
2. PFDR block FP32 train/deploy 差异固定阈值，训练前写死；
3. whole-model 必须 `argmax disagreement=0` 并记录 max-abs-error；
4. 若最终论文仍坚持“whole-model `<1e-6`”为硬规则，则必须先对 **Run2 本身**做 P0 复核；Run2 若不满足，任何 Run10 结果都不能声称满足这一条，需先统一项目级等价性规范或重构既有 folding 数值路径。

**P0-B：服务器说明文件是旧项目快照。** 已上传的 RSML-3 说明中项目/env 仍是 `LS-Rep_BCD_RSML_3 / lsrep`，而本任务明确指定当前项目 `/home/yqwang/projects/STR-RepNet` 与环境 `strrep`。Run10 按本任务当前说明执行；旧文件仅继续沿用硬件、`/share_datasets/CD`、A/B/label/list 与 `gray>=128` 等通用数据合同。

---

# 1. 瓶颈归因复核

## 1.1 证据表

| 来源 | 直接事实 | 对瓶颈的含义 | 证据类型 |
|---|---|---|---|
| Run2 last2 | LEVIR 0.9144 / SYSU 0.8345 | 当前正式锚点；已有 encoder 部分适配 | 代码/日志事实 |
| Run5 D0 full-encoder | LEVIR 0.9168，仅 +0.24pp | encoder 额外适配有有限收益，但不足以解释 +1.06pp 缺口 | 代码/日志事实 |
| Run6 small-object profile | LEVIR small≤502px pixel recall=80.8%，24% small 完全漏检；medium/large≈92%/91% | 强烈指向 small-object / fine-detail information retention | 代码/诊断事实 |
| Run6 D0 full-encoder small | small recall 81.5%，完全漏检 23.7% | full encoder 也几乎修不动 small FN；主瓶颈不在 encoder capacity | 代码/诊断事实 |
| 320 测试尺度 | Precision↑、Recall↓ | 不是简单“输入分辨率不够” | 代码/诊断事实 |
| Run7 stage discriminability | head AUROC@64=0.9881 > full-LDA@64=0.9766 | 已训练 d1 的静态粗尺度线性空间没有明显可榨 headroom | 代码/诊断事实 |
| Run6/8/9 | 多种 rep 均出现 Recall↓/Precision↑ 或 Recall↑/Precision大降 | 既有参数化更多改变 calibration/aggressiveness，而非产生新的可分信息 | 跨轮证据支持推断 |
| 当前 DCR 数据流 | fine t1 先进入 1×1 `fuse1` 与 up(d2) 混合，空间 DW 在融合后 | small lateral detail 可能在 semantic mixing 前被稀释；这是目前未单独验证的位置 | 代码事实 + 待验证假设 |

## 1.2 P0 / P1 / P2 归类

### P0 正确性

当前没有证据表明 Run3–Run9 的失败由实现错误导致；九轮 smoke / fold / budget / argmax 反复通过。**Run10 的 P0 重点不是“修模型 bug”，而是先统一严格等价性口径**（见 §0.3）。此外必须保证新增 PFDR 不破坏 A/B/label 同步几何增强、label `gray>=128`、split list 读取合同。

### P1 方法瓶颈（最高优先级）

**P1-1：decoder fine-scale lateral 在跨尺度融合前的 small-object 空间保真不足。**

这是当前最强根因。理由：small object recall 明显低于 medium/large；full encoder 不修复；扩大测试输入也不修复；Run6 提高 lateral 贡献后能拉 Recall，却以大量 FP 为代价，说明“细节信号存在，但融合方式缺乏空间选择性”。

**P1-2：浅层细节与深层语义的 semantic gap。**

Run6 的 `Recall↑ / Precision↓↓` 是很典型的“把浅层细节放大了，但没有让它在空间上变得更判别”。因此下一步不该继续做纯 channel scale，而应在 `fuse1` 前先形成局部空间上下文。

### P1 次优先级

**编码器特征质量：低—中。** D0 full encoder只有 +0.24pp，且 small FN 几乎不变。可以作为诊断 upper bound，不再作为 Run10 主创新。

**时相交互建模：低。** TAR temporal aux 在 LEVIR 的贡献长期很小；swap/BOTR 均中性。继续做时序对称或 concat/sum/diff 变体缺乏先验。

**输出头上采样：低。** PBRU 的 topology 对 LEVIR只有 +0.12pp，且 WHU 为 -0.35pp；phase rep 主要抬 Precision。输出端太晚，难以恢复已丢失的小目标证据。

### P2 实验工程 / 论文协议

`test-as-val` 是当前项目明确固定的搜索协议，因此 Run10 不改变。但它不是最终论文最严格的无偏评估协议；方法冻结后需要 corrected split / multi-seed 公平性补充。**不能在 Run10 用协议调整、augmentation、loss、threshold 来“补”机制。**

---

# 2. 文献调研（2024–2026，高水平工作）

> 层级说明：CVPR / IJCAI 按 CCF-A 会议标注；IEEE TGRS 是权威 SCI 期刊，**不标成 CCF-A**。

## 2.1 候选方向 A：Large-support / Position-aware Structural Re-parameterization（首选）

### UniRepLKNet

- 题名：**UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition**
- 年份 / venue：2024，CVPR，**CCF-A**
- 一作 / 机构：Xiaohan Ding，Tencent AI Lab
- 官方论文：https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html
- 官方代码：https://github.com/AILab-CVC/UniRepLKNet
- 可借鉴点：大核能够在不堆深度的情况下获得更宽空间感受野；训练期 structural re-param 促进大核优化，部署保持简洁。
- 与本项目实质区别：UniRepLKNet 是通用 backbone 设计；本项目不能重构 VMamba encoder。PFDR 只作用在 **change decoder 的 fine lateral 融合边界**，而且由 LEVIR small-FN 诊断驱动。

### PBConv

- 题名：**Exploiting Position Information in Convolutional Kernels for Structural Re-parameterization**
- 年份 / venue：2025，IJCAI，**CCF-A**
- 一作 / 机构：Tianxiang Hao；Tsinghua University / Hangzhou Zhuoxi Institute of Brain and Intelligence
- 官方论文：https://www.ijcai.org/proceedings/2025/121
- 官方 PDF：https://www.ijcai.org/proceedings/2025/0121.pdf
- 可借鉴点：卷积核不同位置的重要性并不相同；多个小核可在训练后等价折叠回原核，并特别有利于 fine-grained low-level feature extraction。
- 与本项目实质区别：PBConv 使用启发式搜索候选结构；本项目**禁止事后结构 sweep**，所以 PFDR 不复制 PBConv 的 search，而采用一个预注册固定 branch set，并由 C0/M1 直接裁决。

### 为什么可能突破“置信度锐化”签名

Run8/9 改的是 channel/basis optimization；PFDR 改的是 **fuse 前可见的 spatial neighborhood**。若 small-object evidence 在单像素/小邻域中不够稳定，扩大 fine lateral 的空间支持可以产生新的局部可分证据，而不是只移动决策置信度。

---

## 2.2 候选方向 B：Frequency–Spatial Detail Preservation

### FSG-Net

- 题名：**FSG-Net: Frequency-Spatial Synergistic Gated Network for High-Resolution Remote Sensing Change Detection**
- 年份 / venue：2026，IEEE TGRS，**权威 SCI 期刊（非 CCF-A）**
- 一作 / 机构：Zhongxiang Xie，College of Land Science and Technology, China Agricultural University
- arXiv：https://arxiv.org/abs/2509.06482
- 官方代码：https://github.com/zxXie-Air/FSG-Net
- DOI：https://doi.org/10.1109/TGRS.2026.3666124
- 关键机制：frequency-domain discrepancy handling + spatial-temporal attention + semantics-driven shallow/deep gated fusion；论文明确把 shallow detail / deep semantic gap 视为边界与伪变化问题来源。
- 与本项目实质区别：FSG-Net 的 wavelet / attention / dynamic gate 在推理期仍存在，不能直接满足 STR-RepNet 的“单路径解析折叠 + 极低部署预算”铁律。

### 为什么可能突破锐化签名

它不是简单增加同层参数自由度，而是改变“什么信息被保留/融合”。但若直接移植会破坏部署约束，因此只借鉴“**先增强细节证据，再融合语义**”的机制动机，不照搬模块。

### 取舍

**不作为 Run10 主方案。** 若 PFDR 最终证明空间支持仍不足，再考虑可严格线性折叠的 fixed frequency basis；但 Run3 Sobel 已经给过“固定频率/边缘基不一定有效”的负证据，因此现在优先级低于 PFDR。

---

## 2.3 候选方向 C：Local–Global / Cross-stage Adaptive Fusion

### HAM-CD

- 题名：**HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection**
- 年份 / venue：2026，IEEE TGRS，**权威 SCI 期刊（非 CCF-A）**
- 一作 / 机构：Guanlin Li，Xi'an University of Architecture and Technology
- DOI：https://doi.org/10.1109/TGRS.2026.3665418
- 官方代码：https://github.com/guanguanboy/HAM-CD
- 关键机制：HAM 并行结合 local TAM 与 global TAB，ICSF 做 channel/spatial cross-stage fusion；TAM 还采用多尺度 dilated convolution。

### ELGC-Net

- 题名：**ELGC-Net: Efficient Local-Global Context Aggregation for Remote Sensing Change Detection**
- 年份 / venue：2024，IEEE TGRS，**权威 SCI 期刊（非 CCF-A）**
- 一作 / 机构：Mubashir Noman，MBZUAI（论文/官方代码作者联系信息）
- 出版信息：https://research.ibm.com/publications/elgc-net-efficient-local-global-context-aggregation-for-remote-sensing-change-detection
- 官方代码：https://github.com/techmn/elgcnet
- DOI：https://doi.org/10.1109/TGRS.2024.3362914
- 关键机制：depthwise local aggregation + pooled transpose global attention，强调低复杂度 local-global context。

### 为什么可能突破锐化签名

Adaptive fusion 可以在空间上选择“哪些 shallow details 值得通过”，理论上比 Run6 的无条件 native-scale增强更能避免 Precision 崩塌。

### 取舍

**不作为主方案。** attention / input-dependent gate 不能解析折叠为固定卷积；强行做 static approximation 会失去其核心机制。它们适合作为“为什么需要 pre-fusion spatial conditioning”的文献动机，而不是直接实现。

---

## 2.4 最终选择

### 主方案：PFDR（Pre-Fusion Dilated Re-parameterization，预融合扩域重参数化）

**唯一必要对照：C0_PlainPF-DW5。**

不再同时启动 frequency、gate、另一个 temporal rep。这样 Run10 保持唯一变量：

> 在相同 D*、相同 deploy `DW5 + 原 fuse1` 图下，M1 是否因为训练期 fixed multi-branch structural reparameterization，而优于 plain DW5 C0。

---

# 3. 主方案 PFDR 定义

## 3.1 插入位置

只作用于 `DCRDecoder.forward()` 的最细尺度 `t1`，且在 `fuse1` 之前：

```python
t1, t2, t3, t4 = feats
...
u2 = interpolate(d2, size=t1.shape[-2:])

# Run10
l1 = self.prefuse1(t1)        # PFDR/C0 only

d1 = self.block1(self.act(self.fuse1(l1, u2)))
d1 = self.refine(d1)
```

不改：

- VMamba encoder；
- TAR 四尺度 temporal algebra；
- fuse2/fuse3；
- block2/block3；
- output head；
- loss / optimizer / augmentation / threshold。

## 3.2 C0：Plain Pre-Fusion DW5

C0 的 deploy operator：

\[
Y=\operatorname{BN}_5(\operatorname{DWConv}_{5\times5}(X)) + \alpha X
\]

其中 `alpha` 与 identity kernel 可在 deploy 时合并；也可以直接采用与 M1 相同的 main DW5 + residual 实现，但 **aux branches 全关闭**。

C0 的目的不是追求最好数字，而是隔离两个效应：

1. `A0 -> C0`：pre-fusion 5×5 deploy topology / width-budget effect；
2. `C0 -> M1`：真正的 structural reparameterization effect。

## 3.3 M1：PFDR-DW5

训练期：

\[
Y= B_5(X)+E_3(B_3(X))+E_{d2}(B_{3,d=2}(X))+E_1(B_1(X))+\alpha X
\]

其中：

- `B5`：DW5×5 + BN，主分支；
- `B3`：DW3×3 + BN，近邻细节；
- `B3,d=2`：DW3×3 dilation=2 + BN，有效 receptive field 5×5；
- `B1`：DW1×1 + BN，中心/identity-like basis；
- auxiliary BN `gamma=0, beta=0`，保证 M1 auxiliary 在 epoch-0 为零；
- C0/M1 main branch 初始化、随机数顺序必须完全一致。

**不要再加 Sobel、Laplacian、DCT、wavelet branch。** Run10 只验证“fixed spatial-support reparam”，避免模块堆叠。

## 3.4 解析折叠

先用现有 `fold_conv_bn()` 在 FP64 得到：

\[
(\bar K_5,\bar b_5),
(\bar K_3,\bar b_3),
(\bar K_d,\bar b_d),
(\bar K_1,\bar b_1).
\]

定义 5×5 embedding：

### 3×3 中心嵌入

\[
E_3(K)[:,:,1:4,1:4]=K.
\]

### dilation=2 的 3×3 嵌入

\[
E_{d2}(K)[:,:,\{0,2,4\},\{0,2,4\}]=K.
\]

### 1×1 中心嵌入

\[
E_1(K)[:,:,2,2]=K.
\]

### identity depthwise kernel

\[
I_5[c,0,2,2]=1.
\]

最终：

\[
K_{eq}=\bar K_5+E_3(\bar K_3)+E_{d2}(\bar K_d)+E_1(\bar K_1)+\alpha I_5
\]

\[
b_{eq}=\bar b_5+\bar b_3+\bar b_d+\bar b_1.
\]

部署：

```text
PFDR multi-branch
        ↓ switch_to_deploy (FP64)
DWConv5×5(D -> D, groups=D, padding=2, bias=True)
```

auxiliary module、BN、alpha 参数全部删除。

## 3.5 部署等价性

PFDR 的所有训练分支都是线性 depthwise convolution + affine BN；dilation=2 的 3×3 在有限 5×5 support 上有精确稀疏嵌入，因此上述合并是**解析等价**，不是蒸馏、近似或数值拟合。

训练图：

```text
TAR t1
  ├─ DW5+BN ─────────────┐
  ├─ DW3+BN ─────────────┤
  ├─ DW3(d=2)+BN ────────┤ + alpha·identity
  └─ DW1+BN ─────────────┘
            ↓
          fuse1(t1', up(d2))
            ↓
       block1 -> refine -> head
```

部署图：

```text
TAR t1 -> single DW5 -> fuse1(t1', up(d2)) -> block1 -> refine -> head
```

## 3.6 与 Run3–Run9 的区别

| Run | 家族 | deploy 函数类是否实质扩展 | 失败签名 |
|---|---|---:|---|
| Run3 Edge-Basis | refine.dw 内 additive fixed basis | 否，仍原 DW3 | 微弱 recall / precision trade-off |
| Run6 NSCR | fuse 1×1 可交换 affine branch | 否，仍原 1×1 fusion | Recall↑ Precision↓↓ |
| Run7 PBRU | 输出 topology + phase rep | 是，但位置在最终 head | 主要 Precision/置信度锐化 |
| Run8 MPCR | dense PW 的 channel partition over-param | 否，仍原 PW1×1 | Recall↓ Precision↑ |
| Run9 BiFTR | encoder transition serial multiplicative | 否，仍原 downsample Conv | Recall↓ Precision↑ |
| **Run10 PFDR** | **fine lateral pre-fusion spatial-support rep** | **是：增加 5×5 spatial support，但用预算回收保持总预算** | 待验证 |

这就是重开 Run10 的核心科学理由。

---

# 4. 部署 Params / FLOPs 与预算回收

## 4.1 PFDR-DW5 自身开销（D=160 的估算）

部署一个 depthwise 5×5：

- weight：`25D = 4000`
- bias：`D = 160`
- 共约 `4160 params`
- 64×64 feature 上 MAC/FLOP 计数量级：

\[
64\times64\times160\times25=16,384,000\approx0.0164G.
\]

这本身很小，但由于 Run2 已经是硬预算上界，**不能直接加。**

## 4.2 确定性 budget search，不属于超参数 sweep

新增：

```text
analyse/search_run10_pfdr_budget.py
```

固定搜索：

```text
D = 160, 159, 158, ..., 152
```

对每个 D：

1. 构建 C0 deploy graph；
2. 构建 M1 后 switch_to_deploy graph；
3. 用现有 fvcore selective-scan handlers；
4. 输出 raw integer params / per-op FLOPs / unsupported ops；
5. 选择满足两条硬预算的**最大 D**。

预计 `D*=159` 或 `158`，但**文档不把估计当事实**，以机器审计为准。

### 预算前置 gate

- 若 `D* >= 158`：允许训练；
- 若必须降到 `D* < 158` 才能满足预算：**Run10 在训练前停止**，因为宽度损失过大，会让“空间支持收益”与“容量回收损失”混杂。

Run7 已给出 D=158 的 LEVIR 宽度控制仅约 `-0.04pp`，所以 `158–159` 是有课题内证据支持的可接受回收范围；但 Run10 仍必须使用自己的 C0。

## 4.3 训练图参数

训练期 M1 额外 branch 只有 O(D·k²) 规模，约几千参数；它们全部在 deploy 删除/折叠。论文分别报告：

```text
TOTAL-TRAIN-GRAPH-PARAMS
TRAINABLE-PARAMS
DEPLOY-PARAMS
DEPLOY-FLOPS
```

不能只报 deploy 参数而隐藏训练图规模。

---

# 5. 逐文件修改清单

## 5.1 `models/changedetection/models/reparam.py`

新增：

```python
class PFDRDW5(nn.Module):
    # mode: plain | rep
    # main: DW5 + BN
    # rep-only: DW3 + BN, DW3(d=2)+BN, DW1 + BN
    # optional alpha identity

    def forward(self, x): ...
    def get_equivalent_kernel_bias(self): ...
    def branch_stats(self): ...
    def switch_to_deploy(self): ...
```

新增纯函数：

```python
pad_3x3_to_5x5()
dilate_3x3_to_5x5_d2()
pad_1x1_to_5x5()
identity_dw_kernel(..., k=5)
```

要求：

- fold 组合全程 `float64`；
- 最后只 cast 一次；
- branch BN gamma/beta 的零初始化不要污染已有 RepDW3 行为；
- `switch_to_deploy()` 后 state_dict 不得再有 `dw3_aux / dwd2 / dw1 / bn_* / alpha` 等训练结构。

## 5.2 `models/changedetection/models/dcr_decoder.py`

新增构造参数：

```text
use_pfdr=False
pfdr_mode="rep"  # plain | rep
pfdr_scope="fine1"  # Run10 固定，仅 t1
```

构建：

```python
self.prefuse1 = PFDRDW5(dim, use_aux=(pfdr_mode == "rep"))
```

forward 只改一处：

```python
l1 = self.prefuse1(t1) if self.use_pfdr else t1
u2 = F.interpolate(...)
d1 = self.block1(self.act(self.fuse1(l1, u2)))
```

**不要改 block1 / refine 内部结构。**

`switch_to_deploy()` 沿用递归机制，确认 PFDR 被折叠一次且不会重复折叠。

## 5.3 `models/changedetection/models/STRRepNet.py`

新增：

```text
use_pfdr
pfdr_mode
```

透传给 `DCRDecoder`。

Run10 guard：

```text
use_botr=0
use_nscr=0
use_pbru=0
use_mpcr=0
use_biftr=0
head_mode=bilinear
encoder_train=last2
rep_mode=full
```

这样 Run10 只在 Run2 主干上增加 PFDR。

## 5.4 `models/changedetection/models/tar.py`

**方法结构不改。**

只允许必要的接口/诊断 hook 支持；不得加入新 temporal branch。

## 5.5 `models/changedetection/models/Mamba_backbone.py`

**不改方法。** Run9 BiFTR 保留历史实现，但 Run10 强 guard `use_biftr=0`。

## 5.6 `models/changedetection/script/train.py`

新增 CLI：

```text
--use_pfdr 0|1
--pfdr_mode plain|rep
```

新增 guard；新增 TEST 区块字段：

```text
[PFDR] 0|1
[PFDR-MODE] plain|rep
[PFDR-SCOPE] fine1_prefuse
[PFDR-DEPLOY-KERNEL] 5
[PFDR-BRANCH-NORM] main=... near3=... dilated3=... center1=...
```

branch norm 只用于解释，不参与 checkpoint selection。

保持：

- best 仍按 test F1；
- final 仍 deploy graph 重新评估；
- 正式结果仍只认最终完整 TEST RESULTS；
- 不改 CE/Lovász 权重；
- 不加 threshold tuning。

## 5.7 `models/changedetection/script/smoke_test.py`

新增 Run10 smoke：

```text
[ ] A0 原路径无 PFDR
[ ] C0/M1 的 prefuse1 输入/输出 shape = B×D×64×64
[ ] C0/M1 main branch 初始化逐位一致
[ ] M1 aux branch epoch-0 输出严格为 0
[ ] C0 vs M1 同随机种子 epoch-0 logits max_diff = 0
[ ] aux BN gamma grad finite 且非零（至少一个 branch）
[ ] 第二 optimizer step 后 aux conv grad 能出现（避免 gamma=0 导致首步 conv grad=0 被误判）
[ ] deploy 后仅剩一个 DW5
[ ] no aux BN / dilation branch in state_dict
[ ] C0/M1 deploy Params/FLOPs 相等
[ ] deploy 均不超过 Run2 anchor
[ ] argmax disagreement = 0
```

## 5.8 `models/changedetection/script/test_reparam_equivalence.py`

新增 T0/T1/T2，详见 §7。

## 5.9 `analyse/`

新增两个工具：

### `analyse/levir_prefusion_retention_profile.py`

只读 Run2 checkpoint，hook：

```text
TAR t1
fuse1 output (pre-act)
block1 output
refine output d1
```

复用 small≤502px instance 划分，输出：

```text
small / medium / large:
  centroid distance
  Fisher
  diag-LDA AUROC (fit first half / held-out second half)
  effective rank
  mean abs channel correlation
```

它只解释“fine lateral 在 fuse 前后是否损失 small-target separability”，**不用于挑超参**。

### `analyse/search_run10_pfdr_budget.py`

执行 D* 预算搜索与 raw count audit。

可选新增：

```text
analyse/levir_pfdr_profile.py
```

训练后比较 A0/C0/M1 的 same-stage profile，验证 M1 是否真的提升 small-object representation，而非只改变 logits calibration。

---

# 6. 实验设计

## 6.1 唯一变量声明

Run10 所有组固定：

```text
rep_mode=full
encoder_train=last2
use_residual=1
head_mode=bilinear
use_botr=0
use_nscr=0
use_pbru=0
use_mpcr=0
use_biftr=0
seed=2333
epochs=300
batch=16
lr=1e-4
encoder_lr_ratio=沿用 Run2
CE + 2×Lovasz
crop=256
temporal_swap_prob=0
```

变化只有：

```text
A0: no PFDR, D=160   （历史 Run2，不重跑）
C0: plain pre-fusion DW5, D=D*
M1: PFDR multi-branch train / single DW5 deploy, D=D*
```

**C0 与 M1 deploy graph完全相同。**

## 6.2 Phase -1：免费诊断，先于训练

运行：

```text
levir_prefusion_retention_profile.py
```

希望观察到：

```text
t1 对 small-change 的 held-out separability
    > fuse1/block1 后的 small-change separability
```

若成立，支持“融合前细节被稀释”的机制假设。

若不成立，不直接取消 Run10（因为端到端训练可能重塑 t1），但把先验从“较强”降为“探索性”，并在最终解释中如实写明。

## 6.3 Phase 0：代码与预算门槛

顺序严格固定：

```text
T0 algebra
-> T1 PFDR block
-> smoke
-> budget search D*
-> T2 whole model
-> LEVIR 2~5 batch dry run
```

任何 hard failure 都不进入 300 epoch。

## 6.4 Phase 1：LEVIR 决策实验

GPU0 两个 job：

```text
C0_PlainPF_DW5_Dstar
M1_PFDR_DW5_Dstar
```

A0 复用 Run2：

```text
F1=0.9144
IoU=0.8424
Precision=0.9260
Recall=0.9032
```

### 新硬目标下的预注册裁决

旧的 `0.9175` 在“寻找微增益”阶段合理；现在目标是 LEVIR≥0.925，因此 Run10 的 **机制 PASS 线提高到 0.9200**，但另设 TARGET-HIT 标记。

#### PASS（机制值得跨数据集验证）

同时满足：

```text
M1 F1        >= 0.9200
M1 IoU       >= 0.8520
M1 Precision >= 0.9230
M1 - C0 F1   >= +0.0015   （+0.15pp，rep-supported）
Deploy Params/FLOPs <= Run2 anchor
Argmax disagreement = 0
```

说明：`F1=0.9200` 对应 IoU 约 0.8519，取 0.8520 作为一致阈值。

#### TARGET-HIT（额外标志，不替代 PASS/WEAK/FAIL）

```text
LEVIR F1 >= 0.9250
```

只有 TARGET-HIT 才代表 LEVIR 的项目硬目标真正完成。

#### WEAK

任一情况：

```text
0.9159 <= M1 F1 < 0.9200
或 M1 F1>=0.9200 但 IoU / Precision 未达 PASS
或 +0.0005 <= (M1-C0) < +0.0015
```

结论：记录机制信号，但**不做 branch / dilation / scope / kernel size sweep**。

#### FAIL

任一情况：

```text
M1 F1 < 0.9159
或 Precision < 0.9220
或 M1-C0 < +0.0005
或预算/等价性/argmax 任一 hard gate 失败
```

### 归因优先级

最终必须同时计算：

```text
Topology effect = C0 - A0
Rep effect      = M1 - C0
System effect   = M1 - A0
```

如果 `C0 >> A0`、但 `M1≈C0`：

> 说明 5×5 pre-fusion topology 有用，但“结构重参数化”主张不成立；不能把 C0 的 topology gain 写成 rep 创新。

## 6.5 Phase 2：只有 LEVIR PASS 才扩展

优先顺序：

```text
SYSU -> WHU -> CDD
```

理由：SYSU 是另一个未达硬目标的数据集，缺口更大（+1.55pp），应先验证 PFDR 是否只对建筑 small-object 有效，还是对通用地表变化也有效。

### GPU 安排

在 GPU0 最多 4 job 的约束下：

第一批（若显存实测允许）：

```text
C0_SYSU
M1_SYSU
C0_WHU
M1_WHU
```

共 4 job。

第二批：

```text
C0_CDD
M1_CDD
```

若四 job 显存不足，按 `SYSU C0/M1 -> WHU C0/M1 -> CDD C0/M1` 两 job 一批，不为了省时间改变 batch size。

### 四集最终硬门槛

最终模型只有同时满足：

```text
CDD   >= 0.9800
WHU   >= 0.9500
LEVIR >= 0.9250
SYSU  >= 0.8500
```

才能说项目硬目标完成。

CDD/WHU 即便已有 Run2 达标，也不能允许 Run10 final model 掉到门槛以下。

---

# 7. 等价性与测试：T0 / T1 / T2

## T0：纯 FP64 kernel embedding algebra

随机生成 depthwise：

```text
K5
K3
K3_d2
K1
BN affine
x
```

比较：

```text
train multi-branch FP64 direct
vs
single K_eq DW5 FP64
```

固定要求：

```text
max_abs_error < 1e-10
目标 <1e-12
```

并单测：

```text
3x3 center embedding
3x3 dilation=2 sparse embedding
1x1 center embedding
identity center
```

## T1：PFDRDW5 block

对 plain / rep 两模式都测试。

不能只测 zero-init；必须把 auxiliary branch perturb 到训练后合理量级。

要求：

```text
FP64 equivalent-kernel algebra < 1e-10
FP32 block train-vs-deploy 阈值训练前固定（建议先沿仓库 2e-5 envelope）
branch-free deploy state_dict
shape exact
```

如果项目最终强制 whole-model `<1e-6`，该数值要求必须另做 P0 audit，不能事后改 T1 阈值来“通过”。

## T2：whole model

覆盖：

```text
A0 Run2-like
C0 plain DW5 D*
M1 PFDR D*
```

至少 5 组固定随机 pre/post pair，关闭 TF32，deterministic cuDNN。

记录：

```text
max_abs_error
mean_abs_error
argmax disagreement
train graph params
trainable params
deploy params
deploy FLOPs
unsupported ops
```

硬要求：

```text
argmax disagreement = 0
Params <= 28,828,706
FLOPs <= 12.6062G
unsupported selective-scan 已用项目 handler 正确计数
```

并单独打印 PFDR deploy module graph，确认只有一个 DW5。

---

# 8. 失败模式与预案（禁止事后改判据救机制）

## A. Recall↑、Precision稳定、F1 PASS

最理想。支持：

> pre-fusion larger spatial support 找回了 small-object evidence，同时没有像 NSCR 那样把背景细节一起放大。

下一步按 SYSU→WHU→CDD 扩展，并跑 `levir_pfdr_profile.py` 验证 small-object separability 是否真正上升。

## B. Recall↑、Precision明显↓

说明 PFDR 只是更强地传播浅层局部纹理，复现 Run6 的“更激进 decoder”失败模式。

**直接 WEAK/FAIL；不调 threshold，不加 gate，不换 dilation。**

## C. Recall↓、Precision↑

说明“置信度锐化”再次出现，即使 branch norm 很大也不算成功。

停止 PFDR。

## D. C0 PASS、M1≈C0

部署 5×5 topology 有效，但 structural rep 无独立贡献。

- 可把 C0 留作非创新性能对照；
- **不能将 PFDR 作为结构重参数化贡献。**

## E. M1-C0 正，但 M1 仍 WEAK

说明 structural rep 有方向性信号，但效应不足以服务当前 +1pp 级硬目标。

归档，不做 kernel=7 / dilation=3 / branch count sweep。

## F. LEVIR PASS，SYSU FAIL

说明机制偏向 building / small-object spatial detail，缺乏跨场景泛化。论文不能声称普适；且项目硬目标仍未完成。

## G. branch 全部学到非零，但 representation profile 不升

这会形成第四类“训练自由度被利用、判别信息未增加”的负证据。此时应停止 structural branch 搜索，转向重新审查 **deploy architecture function class / dataset protocol / baseline choice**，而不是继续堆 rep branches。

---

# 9. 训练脚本与目录

建议：

```text
train_scripts/TAR-DCR/Run10/
├── README.md
├── C0_PlainPF_DW5/
│   ├── train_LEVIR-CD-256.sh
│   ├── train_SYSU-CD-256.sh
│   ├── train_WHU-CD-256.sh
│   └── train_CDD-CD-256.sh
└── M1_PFDR_DW5/
    ├── train_LEVIR-CD-256.sh
    ├── train_SYSU-CD-256.sh
    ├── train_WHU-CD-256.sh
    └── train_CDD-CD-256.sh
```

checkpoint：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run10/<group>/<dataset>/
```

log：

```text
/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run10/<group>/<dataset>/train_log.txt
```

diagnostics：

```text
/home/yqwang/outputs/STR-RepNet/diagnostics/Run10_PFDR/
```

不同组禁止交叉 resume。

---

# 10. 恢复兼容性

## 10.1 老 checkpoint -> Run10

Run2 checkpoint 没有 `prefuse1.*`，因此不能直接 strict-load 到 C0/M1 后继续训练作为“增量微调”；Run10 应与前几轮一样从相同 pretrained VMamba + 固定 seed 构建完整模型后训练，保持实验协议一致。

若分析工具要读 A0：

```text
use_pfdr=0
strict load Run2
```

若读 C0/M1：

```text
use_pfdr=1
pfdr_mode=plain|rep
strict load 对应 checkpoint
```

不要用 `strict=False` 静默吞掉结构不匹配。

## 10.2 train checkpoint -> deploy

`switch_to_deploy()` 后：

```text
prefuse1 -> one DW5 Conv2d
```

部署 state_dict 与 train graph 不同，必须用项目现有 deploy 导出/测试流程，不用 deploy state_dict 反向 resume 训练。

---

# 11. 论文叙事（只有实验支持时才能使用）

若 M1 真正 PASS 且跨数据集成立，论文故事不应写成“我们又加了一个大核模块”，而应写为：

> **已有结构重参数化在遥感 BCD 中容易落入 optimization-only / confidence-sharpening：训练分支虽学到非零参数，但 deploy operator 的有效空间支持未改变，无法恢复 fine-scale missed changes。STR-RepNet 因此提出 fine-lateral pre-fusion receptive-field reparameterization：训练期以多尺度 depthwise basis 优化局部空间关系，部署折叠为单个 DW5，并通过宽度预算回收保持总 Params/FLOPs 不超过既有锚点。**

这条故事需要三类证据同时成立：

1. small-object / pre-fusion retention 诊断；
2. `M1-C0` 独立 rep 增益；
3. LEVIR + SYSU 至少两类场景一致正向，并保持 CDD/WHU 门槛。

缺任何一项，都要降低主张。

---

# 12. 立即执行顺序

```text
1. 记录当前 main commit = 7437133c37035a429549b9d802f3c1b5a79ed0d2
2. 冻结 Run9 FAIL / §22-A 旧裁决，不回改历史判据
3. 先跑 levir_prefusion_retention_profile.py（只读 Run2）
4. 在 reparam.py 实现 PFDRDW5 + 5×5 embedding
5. dcr_decoder.py 只在 t1->fuse1 前插入 prefuse1
6. STRRepNet.py / train.py 增加 use_pfdr / pfdr_mode 与互斥 guard
7. test_reparam_equivalence.py：T0 -> T1
8. smoke_test：C0/M1 epoch-0 equality、gradient、deploy graph clean
9. search_run10_pfdr_budget.py 搜 D*=最大可行 D；若 D*<158，停止
10. T2 whole-model + Params/FLOPs + argmax audit
11. LEVIR C0/M1 各 2~5 batch dry run
12. GPU0 启动 LEVIR C0 + M1 两 job
13. 300 epoch 后只读最后完整 TEST RESULTS
14. 计算 C0-A0 / M1-C0 / M1-A0，一次性 PASS/WEAK/FAIL
15. 仅 PASS -> SYSU C0/M1
16. SYSU 结果支持 -> WHU C0/M1 -> CDD C0/M1
17. 四集均结束后再判断是否达到 98/92.5/95/85 硬目标
18. final method 冻结后，再做 3-seed × 4 datasets 与论文级公平性补充
```

---

# 13. 仍需补充的证据

1. **Run2 deploy strict-equivalence P0 audit**：项目 `<1e-6` 规则与现有 whole-model `1e-5~1e-4` 记录需要统一；不能在论文里同时写两种口径。
2. **Run2 fine-lateral retention profile**：目前 small-object FN 已知，但还缺 `t1 -> fuse1 -> block1 -> refine` 的定位证据；这是 Run10 最值得先补的 cheap diagnostic。
3. **Run10 D* 机器预算**：本文只给出 DW5 解析估算；最终以 RSML-3 相同 fvcore handler 的 raw counts 为准。
4. **SYSU 的 object-size / morphology error profile**：LEVIR 有 small-object 证据，SYSU 目前只有总指标。若 LEVIR PASS 后 SYSU 不涨，需要先判断 SYSU 的缺口是否同源，不能把 building-specific 解释硬套到 SYSU。
5. **最终公平性**：方法冻结后补 3 seed；若要与 HAM-CD 做统计性比较，HAM-CD 也应尽量按同协议重跑，不能把单次 baseline 与 mean±std 混成“显著提升”。
6. **论文级 latency**：硬约束是 Params/FLOPs，但 PFDR 新增一个 DW5 kernel launch。若 M1 成功，建议补 RTX5090 batch=1 latency / throughput，证明 FLOPs 不增之外实际部署仍实时；这属于结果补充，不参与 Run10 裁决。

---

# 14. 最终一句话方案

> **Run10 不再给既有 1×1/3×3 operator 叠一个“更复杂的训练分支”，而在当前最强证据指向的 fine-scale 跨尺度融合入口，把 t1 在 fuse1 前先经过一个可部署的 DW5；训练期用 5×5 / 3×3 / dilated-3×3 / 1×1 多尺度 depthwise basis 做 structural reparameterization，部署解析折叠成单个 DW5。C0 与 M1 使用完全相同的 D* 和 deploy graph，M1-C0 单独证明 rep；D* 由机器预算搜索回收到 Run2 的 28.828706M / 12.6062G 内。LEVIR 机制 PASS 提高到 F1≥0.9200、IoU≥0.8520、Precision≥0.9230 且 M1-C0≥+0.15pp，同时另标记硬目标 F1≥0.9250；FAIL 后不 sweep 救机制。**

---

# 15. 可核验文献链接

- Ding X. et al. **UniRepLKNet**, CVPR 2024.  
  Paper: https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html  
  Code: https://github.com/AILab-CVC/UniRepLKNet

- Hao T. et al. **Exploiting Position Information in Convolutional Kernels for Structural Re-parameterization**, IJCAI 2025.  
  Publisher: https://www.ijcai.org/proceedings/2025/121  
  PDF: https://www.ijcai.org/proceedings/2025/0121.pdf

- Xie Z. et al. **FSG-Net: Frequency-Spatial Synergistic Gated Network for High-Resolution Remote Sensing Change Detection**, IEEE TGRS 2026.  
  arXiv: https://arxiv.org/abs/2509.06482  
  Code: https://github.com/zxXie-Air/FSG-Net  
  DOI: https://doi.org/10.1109/TGRS.2026.3666124

- Li G. et al. **HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection**, IEEE TGRS 2026.  
  DOI: https://doi.org/10.1109/TGRS.2026.3665418  
  Code: https://github.com/guanguanboy/HAM-CD

- Noman M. et al. **ELGC-Net: Efficient Local-Global Context Aggregation for Remote Sensing Change Detection**, IEEE TGRS 2024.  
  Publisher/author page: https://research.ibm.com/publications/elgc-net-efficient-local-global-context-aggregation-for-remote-sensing-change-detection  
  Code: https://github.com/techmn/elgcnet  
  DOI: https://doi.org/10.1109/TGRS.2024.3362914

---

## 附：证据纪律

本方案把以下四类内容严格区分：

- **代码/日志事实**：来自当前 GitHub README、Run9 README、当前 models/analyse 源码和既有 Run1–Run9 正式 TEST RESULTS；
- **证据支持推断**：decoder small-object / pre-fusion retention 为当前最可能瓶颈；
- **待验证假设**：PFDR 能在不引入大规模 FP 的前提下提高 small-object recall；
- **缺失信息**：Run2 `t1->fuse1` 定位 profile、Run10 D* 实测、SYSU object-size error profile、最终 multi-seed。

任何单 seed、单数据集、<0.3pp 的结果都不称为普适提升；Run10 若成功，也必须用跨数据集和 multi-seed 才能形成正式论文结论。
