# STR-RepNet Run8：Fine-stage Multi-Partition Channel Reparameterization（MPCR-Fine）设计与实验方案

> **硕士课题**：《轻量化遥感二值变化检测中的结构重参数化研究》  
> **仓库**：`YuqiWang-code/STR-RepNet`  
> **审查分支 / commit**：`main @ 21350d1fdfed063f51341269b903a0f97a504ad5`  
> **本轮目标**：在不再修改 TAR / edge / loss / threshold / 数据增强 / encoder 规模 / 输出上采样拓扑的前提下，验证 **fine-stage decoder channel representation 是否仍可通过训练期结构重参数化获得净增益**。  
> **Run2 部署硬预算**：`Params ≤ 28.828706M`，`FLOPs ≤ 12.6062G`。  
> **部署要求**：train multi-branch → deploy single-path；训练期独有结构必须解析折叠或删除；`argmax disagreement = 0`。  
> **训练协议**：seed=2333，300 epoch，batch=16，CE + 2×Lovász，当前阶段沿用 test-as-val。  
> **正式结果口径**：只认同一 `train_log.txt` 最后一个完整的 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块。

---

# 0. 结论先行

## 0.1 Run8 选方向 (a)，不把 Run7 的纯 PixelShuffle 头并入主干

本轮建议明确选择：

> **方向 (a)：refine / fine-stage feature representation。**

暂不采用：

> **方向 (b)：把 Run7 C1 纯 PixelShuffle 头并入主干。**

原因不是 PixelShuffle 在 LEVIR 上完全无效，而是它的跨数据集行为已经给出明确反证：

```text
LEVIR:
A0 0.9144
C0 D158 bilinear 0.9140
C1 PixelShuffle 0.9152
PixelShuffle topology effect = +0.12pp

WHU:
A0 0.9514
C0 D158 bilinear 0.9522
C1 PixelShuffle 0.9487
PixelShuffle topology effect = -0.35pp
```

这说明：

> **纯 PixelShuffle 是一个有价值的“架构观察”，但目前不足以成为新的统一主干。**

如果 Run8 直接以 C1 为主干，会把一个已经在 WHU 上表现出明显负效应的 topology 固化进后续实验；随后任何新 rep 模块的收益都要先弥补这部分架构退化，归因会变得更差。

因此 Run8 回到最干净的部署锚点：

```text
Run2 full_last2
D = 160
head = 2ch 1×1 + bilinear×4
```

然后只改：

> **最终 refine 的 channel-mixing 训练参数化。**

---

## 0.2 Run7 给 Run8 的真正结论不是“继续改 head”，而是“head 已经把现有 d1 榨干，需要改 d1 本身”

Run7 的 `analyse/levir_stage_discriminability.py` 已经给出：

```text
d1 full-cov LDA AUROC@64 = 0.9766
trained head AUROC@64    = 0.9881
```

在当前冻结 Run2 decoder 的静态分析里，训练好的 head 并不存在明显的“粗尺度线性 headroom”。

这意味着继续在同一个 `d1` 上堆更多静态线性读出，不是最有信息量的方向。

Run7 PBRU 的 phase-basis rep 在 LEVIR / WHU 都主要表现为：

```text
Precision ↑
```

但系统级没有达到稳定主方法标准。

因此 Run8 的科学问题应改成：

> **不是“同一个 d1 怎么再读得更聪明”，而是“能否通过可折叠的训练期 channel basis，让最终 refine 学到一个更适合二分类变化判别的 d1”。**

这正好对应 Run7 失败预案中：

> `refine / fine-stage feature representation`

这一方向。

---

# 1. Run8 候选方向比较

| 候选 | 改动位置 | deploy 增量 | 与已有证据关系 | 主要风险 | 决策 |
|---|---|---:|---|---|---|
| **A. MPCR-Fine：多分区通道重参数化** | `refine.pw` | **+0** | 直接改最终 d1 形成过程；避开已失败的 edge / output-head 主线 | 可能只是优化重参数化，增益不足 | **主方案** |
| B. 继续做 refine spatial basis / DCT / Laplacian | `refine.dw` | +0 | 属于 fine-stage，但与 Run3 Sobel Edge-Basis 同属 spatial-kernel basis | 容易重复已被否定的空间先验方向 | **否决** |
| C. PixelShuffle 作为新主干 + 新 rep | output head + refine | 需 D*=158 | LEVIR 正，但 WHU topology −0.35pp | 会把 dataset-specific topology 退化带入 Run8 | **只保留架构观察，不并入主线** |
| D. 扩大 / 堆叠 RepPW low-rank branch | `refine.pw` | +0 | 实现简单 | 当前 RepPW 已有 low-rank + diag + residual，容易变成“更多 branch” | **不选** |

Run8 不再做：

```text
TAR
BOTR
NSCR
Edge-Basis
IBAS
PBRU basis sweep
r sweep
loss
threshold
augmentation
full encoder
更大 dim
attention / Transformer
```

---

# 2. 主方案命名

## 2.1 名称

英文：

> **MPCR-Fine: Multi-Partition Channel Reparameterization for Fine-stage Change Decoding**

中文：

> **细粒度多分区通道重参数化**

简称：

> **MPCR-Fine**

核心贡献不是“发明 group conv / permutation”，而是：

> **把多个互补 channel partition 下的训练期 grouped mixing branch，解析折叠回原有 final dense PW1×1，从而只增强训练参数化，不改变部署图。**

如果后续实验成立，论文中应强调：

```text
training-time complementary channel subspaces
→ exact dense-kernel composition
→ original single-path fine-stage PW at deployment
```

而不是把 `group conv` 或 `channel shuffle` 本身包装成创新。

---

# 3. 为什么选择 `refine.pw`，而不是 `refine.dw`

当前 `DCRDecoder`：

```text
...
fuse1
→ block1
→ refine
    ├─ RepDW3
    ├─ SiLU
    ├─ RepPW1x1
    └─ SiLU
→ head
```

已有证据：

1. Run3 已经对 `refine.dw` 做过 Sobel-X/Y spatial basis，四数据集整体微降；
2. Run6 在高分辨率 cross-scale fusion 上增加 native-scale affine branch，Recall↑ 但 Precision↓；
3. Run7 表明最终 d1 的线性读出已相当充分；
4. 当前 `RepPW1x1` 虽然已有 dense main / low-rank serial / diagonal / residual，但尚未验证 **full-rank structured channel subspace 的互补训练路径**。

因此 Run8 只在：

```text
DCRDecoder.refine.pw
```

引入 MPCR。

不动：

```text
block1
block2
block3
fuse1/2/3
refine.dw
head
TAR
encoder
```

这样能保持实验变量最干净。

---

# 4. MPCR-Fine 的核心思想

设 `refine.pw` 输入：

\[
X\in\mathbb{R}^{B	imes D	imes H	imes W},
\quad D=160,\ H=W=64.
\]

原 Run2 `RepPW1x1` 的训练图记为：

\[
F_{\mathrm{core}}(X).
\]

Run8 增加两个训练期 grouped 1×1 branch：

\[
R_0(X),
\quad
R_1(X).
\]

其中两个 branch 参数量完全相同，但使用不同的 channel partition。

训练输出：

\[
Y_{\mathrm{train}}
=
F_{\mathrm{core}}(X)
+
R_0(X)
+
R_1(X).
\]

然后进入原来的 `SiLU`。

部署时：

\[
Y_{\mathrm{deploy}}
=
W_{\mathrm{eq}}*X+b_{\mathrm{eq}}
\]

仍然只是一层：

```text
single dense PW 1×1
```

与 Run2 `refine.pw` 的部署 operator 完全一致。

---

# 5. 两个固定 channel partition

## 5.1 固定设置

Run8 预注册：

```text
D = 160
groups = 4
group width = 40
branch count = 2
```

不做：

```text
groups sweep
branch-count sweep
partition sweep
```

## 5.2 Partition-0：连续分区

原 channel：

```text
0 ... 39
40 ... 79
80 ... 119
120 ... 159
```

定义：

\[
P_0 = I.
\]

grouped 1×1 在四个连续 channel block 内独立做 full-rank mixing。

## 5.3 Partition-1：交织分区

建议固定 permutation：

```python
perm = torch.arange(D).view(groups, D // groups).t().reshape(-1)
```

D=160, groups=4 时：

```text
0, 40, 80, 120,
1, 41, 81, 121,
2, 42, 82, 122,
...
```

再按 40-channel 一组做 grouped 1×1。

因此：

```text
Partition-0:
局部 contiguous channel mixing

Partition-1:
cross-partition interleaved channel mixing
```

两个 branch 的参数量、groups、kernel、BN 完全一致，唯一差别是固定 permutation。

---

# 6. MPCR 训练图

对 partition \(P_j\)：

\[
X_j=P_jX.
\]

grouped 1×1：

\[
G_j(X_j).
\]

独立 BN：

\[
\hat G_j(X_j)
=
BN_j(G_j(X_j)).
\]

再逆 permutation 回原 channel 顺序：

\[
R_j(X)
=
P_j^{-1}\hat G_j(P_jX).
\]

总训练图：

\[
Y_{\mathrm{train}}
=
F_{\mathrm{core}}(X)
+
P_0^{-1}BN_0(G_0(P_0X))
+
P_1^{-1}BN_1(G_1(P_1X)).
\]

其中：

\[
P_0=I.
\]

---

# 7. 部署折叠数学

## 7.1 原 core

现有 `RepPW1x1.get_equivalent_kernel_bias()` 已得到：

\[
W_0,\ b_0,
\]

其中：

\[
W_0\in\mathbb{R}^{D	imes D	imes1	imes1}.
\]

其内部已经解析折叠：

```text
main
+ low-rank serial
+ diagonal
+ alpha·I
```

并保持 FP64。

## 7.2 grouped branch + BN

第 \(j\) 个 grouped branch权重实际 shape：

\[
[D,\ D/g,\ 1,\ 1].
\]

先按现有 `fold_conv_bn()`：

\[
(ar G_j,ar b_j)
=
FoldBN(G_j,BN_j).
\]

将其嵌入 dense 1×1 matrix：

\[
\widetilde G_j\in\mathbb{R}^{D	imes D}.
\]

若：

\[
q=D/g,
\]

则输出 channel \(o\) 属于 group：

\[
k=\lfloor o/qfloor,
\]

只连接输入：

\[
[kq,(k+1)q).
\]

其余 dense matrix 位置补 0。

## 7.3 permutation folding

因为 permutation 是线性正交变换：

\[
P_j^{-1}=P_j^T.
\]

branch：

\[
R_j(X)
=
P_j^{-1}
(\widetilde G_jP_jX+ar b_j).
\]

所以：

\[
R_j(X)
=
(P_j^{-1}\widetilde G_jP_j)X
+
P_j^{-1}ar b_j.
\]

定义：

\[
W_j=P_j^{-1}\widetilde G_jP_j,
\]

\[
b_j=P_j^{-1}ar b_j.
\]

最终：

\[
W_{	ext{eq}}=W_{	ext{core}}+W_0^{branch}+W_1^{branch},
\]

\[
b_{	ext{eq}}=b_{	ext{core}}+b_0^{branch}+b_1^{branch}.
\]

部署：

\[
Y_{	ext{deploy}}
=
W_{	ext{eq}}X+b_{	ext{eq}}.
\]

所以两个 grouped + permuted branch 全部可以解析吸收到原有 dense PW1×1。

部署图不保留：

```text
group conv
BN
permute
inverse permute
branch add
```

---

# 8. FP64 组合规范

严格沿用 Run5–Run7 已建立的规范：

```text
所有等价 kernel / bias：
float64 compose
↓
所有 branch 在 FP64 中求和
↓
最终创建 deploy nn.Conv2d
↓
一次性 .float()
```

禁止：

```text
core 先 switch_to_deploy() → FP32
然后再和 MPCR branch 相加
```

正确方式：

```python
W_core, b_core = self.core.get_equivalent_kernel_bias()  # FP64
W0, b0 = fold_group_branch_fp64(...)
W1, b1 = fold_group_branch_fp64(...)

W_eq = W_core + W0 + W1
b_eq = b_core + b0 + b1

self.pw.weight.data = W_eq.float()
self.pw.bias.data   = b_eq.float()
```

---

# 9. 零初始化：epoch 0 必须与 Run2 逐位一致

两个 MPCR branch：

```text
group conv weight：正常 Kaiming 初始化
BN gamma = 0
BN beta  = 0
```

于是：

\[
BN_j(G_j(X))=0.
\]

因此：

\[
Y_{\mathrm{train}}^{epoch0}
=
F_{\mathrm{core}}(X).
\]

也就是：

> **Run8 M1 在 epoch 0 的 forward 输出必须与同一初始化的 Run2 core 逐位一致。**

验收：

```text
max_abs_diff == 0.0
```

不是 `<1e-6`。

---

# 10. 为什么 MPCR 与现有 RepPW1x1 有实质区别

现有：

\[
W_{RepPW}
=
W_{main}
+
W_2W_1
+
D_{diag}
+
lpha I.
\]

其中：

- `W2W1`：全局但 low-rank；
- `diag`：逐 channel；
- `I`：identity。

MPCR 增加：

\[
P_0^{-1}B_0P_0
+
P_1^{-1}B_1P_1,
\]

其中 \(B_j\) 是 block-diagonal、组内 full-rank matrix。

因此它提供的是：

> **structured full-rank local subspace basis + complementary partition connectivity**

而不是简单继续加 low-rank / diagonal / residual branch。

---

# 11. 部署 Params / FLOPs 证明

Run8 deploy：

```text
refine.pw:
Run2  = dense Conv2d(160→160,1,bias=True)
Run8  = dense Conv2d(160→160,1,bias=True)
```

完全相同。

因此理论上：

\[
\Delta Params_{deploy}=0,
\qquad
\Delta FLOPs_{deploy}=0.
\]

整网应保持：

```text
28.828706M
12.6062G
```

以本机当前 fvcore 实测值为准。

训练期每个 grouped branch：

\[
160	imes40=6400
\]

权重，BN trainable：

\[
2D=320.
\]

两个 branch：

\[
2(6400+320)=13440.
\]

所以：

```text
train graph +13,440 parameters
deploy +0 parameters
```

---

# 12. D* 预算搜索如何处理

Run8 主方案不需要像 PBRU 那样主动缩宽，因为 deploy topology 和 Run2 完全一致。

但仍建议新增：

```text
analyse/search_run8_budget.py
```

逻辑：

```text
anchor:
Run2, D=160, bilinear, MPCR=0

candidate:
Run8 deploy, D from 160 down
```

由于 MPCR 全折叠，理论上：

```text
D=160
```

第一项就应满足：

```text
Params_candidate == Params_anchor
FLOPs_candidate == FLOPs_anchor
```

因此：

> **Run8 的机器搜索结果应退化为 D*=160。**

如果机器搜索得到：

```text
D*=159 或更低
```

说明实现有 P0 错误，不能开训练。

Run7 的 D*=158 只服务于 PixelShuffle/PBRU deploy head 预算回收，Run8 不应沿用。

---

# 13. Run8 最小实验组

## 13.1 A0：历史锚点，不重跑

```text
A0_Run2_full_last2
D=160
bilinear
MPCR=0
```

LEVIR：

```text
Recall    0.9032
Precision 0.9260
F1        0.9144
IoU       0.8424
```

WHU / SYSU / CDD：

```text
0.9514 / 0.8345 / 0.9842
```

## 13.2 C0：Same-Partition Capacity Control

组名：

> `C0_MPCR_Same2`

配置：

```text
refine.pw:
core RepPW1x1
+
group branch 0: identity partition
+
group branch 1: identity partition
```

两个 branch：

```text
groups=4
同样参数量
同样 BN
同样零初始化
```

它回答：

> **如果只是“多两个 grouped branch”本身，是否就能涨点？**

## 13.3 M1：Complementary Multi-Partition MPCR

组名：

> `M1_MPCR_Multi2`

配置：

```text
refine.pw:
core RepPW1x1
+
branch 0: identity partition
+
branch 1: interleaved partition
```

C0 / M1：

```text
train-only params 完全相同
branch 数完全相同
groups 完全相同
deploy graph 完全相同
```

唯一变量：

> **branch 1 的 channel partition。**

---

# 14. Run8 不把 PixelShuffle 放进实验组

Run7 已经完整得到：

```text
C1 PixelShuffle D158
LEVIR 0.9152
WHU   0.9487
```

无需重跑。

Run8 README 可以在“历史观察”中引用它，但：

```text
Run8 C0 / M1 均使用 bilinear D160
```

这样不会把 width effect / PixelShuffle topology / fine-stage rep 重新混到一起。

---

# 15. 预注册假设

## H8-1：fine-stage channel basis 假设

如果最终 refine 的 channel mixing 是剩余瓶颈之一，则：

```text
M1_MPCR_Multi2
```

应优于 A0，并且优于相同训练参数量的：

```text
C0_MPCR_Same2
```

## H8-2：互补 partition，而非“branch 越多越好”

如果收益只是因为增加训练分支：

```text
C0 ≈ M1
```

则 MPCR 的 multi-partition 主张不成立。

只有：

```text
M1 > C0
```

才能支持 complementary channel partition 的结构设计本身有贡献。

---

# 16. LEVIR PASS / WEAK / FAIL

完全沿用既有强度，不事后移动。

锚点：

```text
F1        = 0.9144
IoU       = 0.8424
Precision = 0.9260
```

## PASS

同时：

```text
F1        >= 0.9175
IoU       >= 0.8475
Precision >= 0.9230
```

以及：

```text
Deploy Params <= 28.828706M
Deploy FLOPs  <= 12.6062G
argmax disagreement = 0
```

## WEAK

满足任一：

```text
0.9159 <= F1 < 0.9175
```

或 F1 达标但：

```text
IoU < 0.8475
```

或：

```text
Precision < 0.9230
```

处理：

```text
只记录
做完 C0/M1 归因
不扩 SYSU/CDD
```

## FAIL

任一：

```text
F1 < 0.9159
Precision < 0.9220
budget fail
argmax disagreement != 0
fold 实现不合格
```

处理：

> **立即停止 MPCR，不救机制。**

禁止：

```text
groups 2/4/8 sweep
换 permutation
加第三个 partition
gamma scale
loss
threshold
PixelShuffle 叠加
full encoder
NSCR
```

---

# 17. rep 归因判据

核心定义：

\[
\Delta_{partition}
=
F1(M1)-F1(C0).
\]

因为 C0 / M1：

```text
相同 branch 数
相同 groups
相同 train params
相同 deploy graph
```

只差 partition。

## rep-supported

```text
M1 - C0 >= +0.15pp F1
```

且：

```text
Precision(M1) 不比 C0 下降超过 0.15pp
```

## rep-weak

```text
+0.05pp <= M1-C0 < +0.15pp
```

只能写：

> 单 seed 下 multi-partition 有正向迹象。

## rep-not-supported

```text
M1-C0 < +0.05pp
```

包括负值。

说明：

> 增益若存在，主要来自 generic grouped over-parameterization，而不是 complementary partition。

---

# 18. LEVIR 后的继续 / 停止逻辑

## 情况 A：M1 PASS + rep-supported

进入 WHU。

## 情况 B：M1 PASS + rep-weak

只扩 WHU 作为跨数据集机制裁决。

如果 WHU 同时：

```text
ΔF1 vs A0 >= 0
且
M1-C0 >= +0.05pp
```

再进入：

```text
SYSU → CDD
```

## 情况 C：M1 PASS + rep-not-supported

不把 MPCR 作为主创新，不扩全四集。

## 情况 D：M1 WEAK

完成 LEVIR C0/M1 归因后停止，不跑 WHU。

## 情况 E：M1 FAIL

立即停止，不做任何 rescue。

---

# 19. WHU gate

若进入 WHU，锚点：

```text
A0 = 0.9514
```

继续 SYSU/CDD 必须同时满足：

```text
M1 WHU ΔF1 >= 0
M1-C0 >= +0.05pp
deploy cap 合格
argmax = 0
```

如果：

```text
M1 WHU < A0 - 0.15pp
```

判为明显 cross-dataset FAIL，停止。

如果：

```text
-0.15pp <= ΔF1 < 0
```

记录为 dataset-specific / 不稳健，同样不扩 SYSU/CDD。

---

# 20. 最终四数据集判据

只有通过前两级 gate 后才跑。

沿用已有标准：

```text
Macro F1 >= 92.25%
LEVIR     >= 91.75%
至少 3/4 数据集 ΔF1 >= 0
任一数据集退化 <= 0.15pp
```

同时：

```text
每个 dataset:
Deploy Params/FLOPs ≤ cap
argmax disagreement = 0
```

---

# 21. 失败后的预设决策

## 21.1 A0 < C0 ≈ M1

说明 generic grouped overparameterization 有效，但 multi-partition 没有独立价值。

处理：

```text
不把 MPCR-Multi 作为创新
不增加更多 partitions
```

## 21.2 C0 < A0，M1 ≈ A0

说明 complementary partition 只是抵消了不良 grouped branch，不构成净收益。

处理：FAIL。

## 21.3 M1 > C0，但 M1 仍 WEAK

说明 multi-partition 机制可能有效，但强度不足以支撑扩展。

处理：

```text
记录
停止
```

不通过 sweep 放大效应。

## 21.4 M1 Recall↑、Precision 明显↓

如果再次出现 Run6 型模式且：

```text
Precision < 0.9220
```

直接 FAIL。

不通过 threshold / precision-oriented loss / boundary loss 修复。

## 21.5 M1 LEVIR PASS、WHU 负

说明 fine-stage partition optimization 可能 LEVIR-specific。

停止全四数据集扩展。

---

# 22. 代码修改清单

## 22.1 `models/changedetection/models/reparam.py`

新增 helper：

```text
group1x1_to_dense_fp64()
make_interleaved_permutation()
invert_permutation()
```

新增：

> `MPCRPW1x1`

建议接口：

```python
class MPCRPW1x1(nn.Module):
    def __init__(
        self,
        channels,
        groups=4,
        mode="multi2",   # same2 | multi2
        use_aux=True,
        use_residual=True,
        deploy=False,
    ):
        ...
```

内部：

```text
core = RepPW1x1(...)

branch0 = Conv2d(D,D,1,groups=4,bias=False) + BN
branch1 = Conv2d(D,D,1,groups=4,bias=False) + BN

perm0 = identity
perm1 =
    same2  -> identity
    multi2 -> interleaved
```

必须实现：

```text
forward()
get_equivalent_kernel_bias()
branch_stats()
switch_to_deploy()
```

## 22.2 `models/changedetection/models/dcr_decoder.py`

`RepLocalBlock` 增加可选 PW 类型：

```python
RepLocalBlock(
    ...,
    use_mpcr=False,
    mpcr_mode="multi2",
    mpcr_groups=4,
)
```

仅 `self.refine` 开 MPCR。

`block1/2/3` 保持原 `RepPW1x1`。

## 22.3 `models/changedetection/models/STRRepNet.py`

新增：

```text
use_mpcr
mpcr_mode
mpcr_groups
```

Run8 默认：

```text
head_mode=bilinear
use_pbru=0
use_nscr=0
use_botr=0
decoder_dim=160
```

## 22.4 `models/changedetection/script/train.py`

新增 CLI：

```text
--use_mpcr 0|1
--mpcr_mode same2|multi2
--mpcr_groups 4
```

Run8 attribution guard：

```python
if args.use_mpcr:
    assert args.head_mode == "bilinear"
    assert args.decoder_dim == 160
    assert not args.use_pbru
    assert not args.use_nscr
    assert not args.use_botr
```

最终 TEST block 增加：

```text
[MPCR] 1
[MPCR-MODE] same2|multi2
[MPCR-GROUPS] 4
[MPCR-GAMMA-NORM] p0=... p1=...
```

branch norm 只用于机制记录，不挑 checkpoint。

## 22.5 `models/changedetection/script/smoke_test.py`

增加：

```text
[ ] MPCR class 只出现在 refine.pw
[ ] BN gamma=beta=0
[ ] epoch0 output 与 A0 exact equal
[ ] gamma gradients 非零
[ ] group conv 后续能收到非零 grad
[ ] deploy branch 全删除
[ ] deploy params exact equal anchor
[ ] deploy FLOPs exact equal anchor
[ ] argmax disagreement = 0
```

## 22.6 `models/changedetection/script/test_reparam_equivalence.py`

新增：

### T0：Permutation + dense embedding

FP64：

```text
group conv direct output
vs
dense embedded kernel output
```

要求 `<1e-12`。

再验证：

```text
P^-1 G(Px)
vs
(P^-1 G P)x
```

FP64 `<1e-12`。

### T1：MPCRPW1x1 block

使用非零 branch，验证：

```text
train graph vs deploy PW1x1
```

block-level `<2e-5`。

同时单独验证 `get_equivalent_kernel_bias()` FP64 algebra `<1e-12`。

### T2：whole model

Run8：

```text
bilinear
D160
MPCR multi2
```

记录：

```text
max_abs_error
argmax disagreement
```

沿用：

```text
whole-model < 2e-4
argmax disagreement = 0
```

不虚构 `<1e-6`。

---

# 23. 预算验证脚本

新增：

```text
analyse/search_run8_budget.py
```

不复用 `search_pbru_budget.py` 的 PixelShuffle candidate。

输出：

```text
anchor_params_raw
candidate_params_raw
anchor_flops
candidate_flops
delta_params
delta_flops
Dstar
```

预期：

```text
delta_params = 0
delta_flops  = 0
Dstar = 160
```

若不是：

> **P0 implementation bug，禁止训练。**

---

# 24. 建议新增 fine-stage 分析工具

新增：

```text
analyse/levir_fine_stage_profile.py
```

hook：

```text
block1 output
refine.dw output
refine.dw + SiLU
refine.pw output
refine final output
```

统计：

```text
changed / unchanged feature centroid distance
normalized Fisher separation
diagonal-LDA AUROC
channel covariance effective rank
mean absolute inter-channel correlation
```

目的：

> 验证 MPCR 是否真的改变 final channel representation，而不是只让分类阈值漂移。

这只是机制解释，不用于挑 checkpoint，也不作为新的可调判据。

---

# 25. 有效秩建议

对 covariance eigenvalues \(\lambda_i\)：

\[
p_i=rac{\lambda_i}{\sum_j\lambda_j}
\]

effective rank：

\[
R_{eff}
=
\exp\left(-\sum_i p_i\log(p_i+\epsilon)ight).
\]

只做机制解释，不预注册“rank 越高越好”。

重点看：

```text
M1 vs A0 / C0
```

是否出现 F1 改善同时 Fisher/AUROC 有一致方向变化。

---

# 26. 最小实验启动顺序

## Phase 0：证据冻结

记录：

```bash
git rev-parse HEAD
# 21350d1fdfed063f51341269b903a0f97a504ad5
```

冻结 Run7 结论，不再改判据。

## Phase 1：实现 MPCR

修改：

```text
reparam.py
dcr_decoder.py
STRRepNet.py
train.py
smoke_test.py
test_reparam_equivalence.py
```

新增：

```text
analyse/search_run8_budget.py
analyse/levir_fine_stage_profile.py
```

## Phase 2：解析等价性先行

先跑：

```text
permutation exact test
group->dense exact test
MPCR FP64 fold
MPCR FP32 block fold
whole-model fold
```

全部通过后再继续。

## Phase 3：预算机器验证

必须得到：

```text
D*=160
Deploy Params=Run2 exact
Deploy FLOPs=Run2 exact
```

## Phase 4：真实 LEVIR dry run

每组：

```text
2~5 batches
forward
CE
Lovasz
backward
optimizer.step
```

检查：

```text
loss finite
BN gamma grad != 0
branch conv 后续 grad finite
encoder last2 状态正确
stage1/2 frozen
```

## Phase 5：LEVIR 两组并行

建议：

```text
GPU0:
C0_MPCR_Same2 / LEVIR

GPU1:
M1_MPCR_Multi2 / LEVIR
```

A0 不重跑。

## Phase 6：一次性裁决

只读最后完整 TEST RESULTS，计算：

```text
M1-A0
C0-A0
M1-C0
```

按 §16 / §17 裁决。

---

# 27. 数据集扩展顺序

只有达到 gate 才扩：

```text
LEVIR
↓
WHU
↓
SYSU
↓
CDD
```

原因：

- LEVIR：当前主要 gap；
- WHU：同为建筑变化，最关键跨数据集 sanity；
- SYSU：更通用地表变化；
- CDD：高 ceiling，最后验证是否无害。

---

# 28. Run8 脚本目录

建议：

```text
train_scripts/TAR-DCR/Run8/
├─ README.md
├─ C0_MPCR_Same2/
│  ├─ train_LEVIR-CD-256.sh
│  ├─ train_WHU-CD-256.sh
│  ├─ train_SYSU-CD-256.sh
│  └─ train_CDD-CD-256.sh
└─ M1_MPCR_Multi2/
   ├─ train_LEVIR-CD-256.sh
   ├─ train_WHU-CD-256.sh
   ├─ train_SYSU-CD-256.sh
   └─ train_CDD-CD-256.sh
```

脚本可以预写全部四数据集，但未过 gate 的脚本禁止启动。

---

# 29. checkpoint / log 路径

checkpoint：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run8/<group>/<dataset>/
```

log：

```text
/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run8/<group>/<dataset>/train_log.txt
```

诊断：

```text
/home/yqwang/outputs/STR-RepNet/diagnostics/Run8_MPCR/
```

不同 group 禁止互相 resume。

---

# 30. README 记账格式

建议 Run8 README 固定记录：

| Group | D | Head | refine.pw train graph | Deploy refine.pw | Deploy Params | Deploy FLOPs | F1 |
|---|---:|---|---|---|---:|---:|---:|
| A0 | 160 | bilinear | Run2 RepPW | 160→160 PW | 28.828706M | 12.6062G | 0.9144 |
| C0 | 160 | bilinear | core + same partition×2 | 160→160 PW | machine | machine | TBD |
| M1 | 160 | bilinear | core + identity/interleaved | 160→160 PW | machine | machine | TBD |

另单列：

```text
Δsystem = M1 - A0
Δgeneric = C0 - A0
Δpartition = M1 - C0
```

---

# 31. P0 / P1 / P2 风险

## P0 正确性

### P0-1 permutation indexing 错误

必须测试：

```text
P^-1(Px) == x
```

exact。

### P0-2 grouped kernel dense embedding 错位

必须 FP64 direct-vs-dense 测试。

### P0-3 BN fold 后 bias permutation 漏做

权重和 bias 都必须做逆 permutation。

### P0-4 switch_to_deploy 提前 FP32

禁止 core 提前 deploy。

### P0-5 MPCR 泄漏到 block1/2/3

Run8 只允许：

```text
decoder.refine.pw
```

## P1 方法风险

### P1-1 dense core 已经足够强

MPCR 是高信息量但非必涨点实验。

### P1-2 channel ordering 没有天然语义

interleaved partition 是固定结构优化 basis，不应声称某组对应边界或语义。

### P1-3 可能只提供 generic overparameterization

这就是 C0 `same2` 必须存在的原因。

## P2 实验工程

- 保持现有环境；
- smoke 后才真数据；
- 真实数据先 dry run；
- 不删除旧 Run7；
- 不覆盖 Run2 checkpoint；
- Git commit 前检查 staged files；
- 不提交 `.pth` / outputs / cache。

---

# 32. Run8 成功后论文主线如何组织

如果 MPCR 真正成立，论文主线可整理为：

## Contribution 1：TAR

```text
temporal algebraic reparameterization
```

## Contribution 2：DCR

```text
decoder-wide local / channel / pair-fusion reparameterization
```

## Contribution 3：MPCR-Fine

```text
fine-stage complementary channel-partition reparameterization
→ train structured grouped paths
→ deploy original dense PW
```

Run7 PBRU 不强行升级为正式贡献，可作为：

> **输出头结构探索 / negative-to-weak evidence**

放在方法探索或消融讨论中。

这样论文不会变成 TAR + Edge + IBAS + NSCR + PBRU + MPCR 的模块堆叠。

---

# 33. 与 Run7 的关系

Run7 给 Run8 提供了三条证据：

1. **纯 output topology 对数据集敏感**：LEVIR 正、WHU 负，因此不适合直接升为主干；
2. **phase rep 两数据集都提高 Precision**：说明训练期结构重参数化仍可能改善判别置信度，但 PBRU 系统级收益不足；
3. **现有 d1 的 coarse linear readout 已经很强**：下一步更合理的是改变 d1 的形成过程，而不是继续更换读出。

所以 Run8 是：

> **从 output readout reparameterization，转向 fine-stage representation reparameterization。**

不是无关跳方向。

---

# 34. 不建议的 Run8 组合

不要做：

```text
MPCR + PixelShuffle
MPCR + PBRU
MPCR + NSCR
MPCR + BOTR
MPCR + full encoder
MPCR + Edge
```

第一轮必须只有 MPCR，否则即使涨点，也无法回答到底是哪一个结构起作用。

---

# 35. 仍需补充证据

Run8 开训前只缺：

1. MPCR FP64 解析折叠实际误差；
2. whole-model FP32 fold + argmax；
3. 机器预算验证 D*=160；
4. LEVIR fine-stage feature profile；
5. C0 vs M1 的同参数量结构归因。

其中 1–3 是启动训练前硬门槛。

---

# 36. 立即执行顺序

```text
1. 冻结 main@21350d1 与 Run7 结论
2. 在 reparam.py 实现 MPCRPW1x1
3. dcr_decoder.py 仅把 refine.pw 切成可选 MPCR
4. STRRepNet.py 增加 use_mpcr / mode / groups
5. train.py 增加参数与互斥 guard
6. test_reparam_equivalence.py 加 T0/T1/T2
7. smoke_test.py 加 epoch0 exact equality + gradient + branch deletion
8. 新增 search_run8_budget.py
9. 跑 FP64 algebra test
10. 跑 block fold
11. 跑 whole-model fold / argmax
12. 跑机器预算，必须 D*=160
13. LEVIR 2~5 batch dry run
14. GPU0 跑 C0_MPCR_Same2 / LEVIR
15. GPU1 跑 M1_MPCR_Multi2 / LEVIR
16. 按预注册 PASS/WEAK/FAIL + rep attribution 裁决
17. 仅满足 gate 后：WHU → SYSU → CDD
18. 更新 README / experiment_metrics / Run8 设计文档
```

---

# 37. 一句话执行版

> **Run8 不把 PixelShuffle 升为主干，回到 Run2 D160 + bilinear 的统一部署图，只在最终 `refine.pw` 做 MPCR-Fine：训练期用两个同参数量 grouped 1×1 分支，C0 两个都用连续 partition，M1 使用“连续 + 交织”互补 partition；两个 branch 均 BN γ=β=0 保证 epoch 0 与 Run2 逐位一致，部署时用 \(P^{-1}GP\) 解析吸收回原 dense PW1×1，FP64 组合后一次 FP32 cast，理论 deploy +0 Params/+0 FLOPs、机器预算搜索应得到 D*=160。LEVIR 仍按 F1≥0.9175 / 0.9159 体系裁决，M1−C0≥+0.15pp 才算 rep-supported；WEAK/FAIL 不 sweep、不救机制。**

---

# 38. 本次仓库审查依据

本次实际读取 `main@21350d1`：

```text
README.md
train_scripts/TAR-DCR/Run7/README.md

models/changedetection/models/
  STRRepNet.py
  dcr_decoder.py
  reparam.py

models/changedetection/script/
  train.py
  smoke_test.py
  test_reparam_equivalence.py

analyse/
  levir_stage_discriminability.py
  search_pbru_budget.py
```

Run7 当前正式归因：

```text
LEVIR:
A0 0.9144
C0 0.9140
C1 0.9152
M1 0.9161
topology +0.12pp
rep +0.09pp = rep-weak

WHU:
A0 0.9514
C0 0.9522
C1 0.9487
M1 0.9506
topology -0.35pp
rep +0.19pp = rep-supported
```

Run7 最终判定：

```text
M1/LEVIR = WEAK
不扩 SYSU/CDD
PBRU 不宣称稳定普适 rep 增益
```

Run8 由此进入：

> **fine-stage representation reparameterization。**
