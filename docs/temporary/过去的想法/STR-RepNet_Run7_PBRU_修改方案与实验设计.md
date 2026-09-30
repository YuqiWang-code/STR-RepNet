# STR-RepNet Run7：PBRU 修改方案与实验设计

> **课题**：轻量化遥感二值变化检测中的结构重参数化  
> **仓库**：`YuqiWang-code/STR-RepNet`，`main`  
> **本次审查 HEAD**：`6a746b381f05a7a50828c4841c91292cfb49ad62`（2026-09-28，`update code`）  
> **目标**：在部署预算不高于 Run2 锚点的前提下，验证最终输出头空间自由度是否是 LEVIR 小目标 FN 的关键瓶颈，并保持论文创新主线为**结构重参数化**。  
> **硬部署预算**：Deploy Params ≤ Run2 精确值（README 记账值 28.829M），Deploy FLOPs ≤ Run2 精确值（README 记账值 12.6062G），部署单路径，argmax disagreement = 0。  
> **训练协议**：seed=2333，300 epoch，batch=16，CE + 2×Lovász，当前阶段沿用 test-as-val。  
> **正式指标口径**：只认每个 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块。

---

# 0. 结论先行

## 0.1 Run7 建议转 PBRU，但必须修正 Run6 文档中的动机表述

**结论：同意 Run7 转向 PBRU（Phase-Basis Reparameterized Upsampling），但不是因为“更高输入分辨率有效”。恰恰相反，现有尺度诊断对“输入尺度瓶颈”给出了负证据。**

Run6 文档 §29 的原始情况 A 写的是：

> NSCR FAIL，且 small-component FN 集中、尺度诊断阳性 → 转 PBRU。

现在真实证据是：

- small：pixel recall **80.8%**，**24.0%** 小目标完全漏检；
- D0 full-encoder：small 仅 **81.5%**，漏检率 **23.7%**，说明继续放开 encoder 不能实质修复；
- 320 输入：F1 0.9189，但 Recall **0.8960 下降**、Precision **0.9429 上升**；
- 384 继续出现 Recall 下降；
- Run6 NSCR：Recall **+0.41pp**，Precision **−0.52pp**，说明简单增加 native-scale 线性路径只让 decoder 更激进，没有提高可靠性。

因此，**“高输入分辨率能救小目标”这个前提没有成立**。严格说，Run6 文档 §29 的“尺度诊断阳性”不能原样沿用。

但这并不否定 PBRU。Run7 应把科学问题改写为：

> **在输入固定为 256×256、encoder/TAR 不变的条件下，当前 `d1(160,64×64) → 2-channel coarse logits → fixed bilinear×4` 是否对最终变化图施加了过强的空间子空间约束，使小目标的相位级/亚像素级分类自由度不足？**

这才是 PBRU 的正确动机。

---

## 0.2 负性的输入尺度诊断，反而帮助把 PBRU 与“提高输入分辨率”彻底区分开

当前输出为：

```text
d1: [B,160,64,64]
  ↓
head 1×1: 160 → 2
  ↓
coarse logits: [B,2,64,64]
  ↓
fixed bilinear ×4
  ↓
[B,2,256,256]
```

最终 256×256 每个像素的两类 logits 并不是独立学习出来的，而是受一个 **64×64×2 coarse logit lattice** 和固定 bilinear 核约束。

PBRU 部署图改为：

```text
d1: [B,D*,64,64]
  ↓
head 1×1: D* → 2×4² = 32
  ↓
PixelShuffle(4)
  ↓
[B,2,256,256]
```

于是每个 64×64 cell 内的 16 个 phase 都拥有**独立的类别投影自由度**。这改变的是：

- 输出映射的空间参数化；
- 64×64 feature 到 256×256 logits 的相位自由度；
- 不是输入图像的采样尺度；
- 不是 encoder 的感受野；
- 也不是通过更大输入“塞更多像素”。

所以：

> **320/384 的 Recall 下降只否定“input-resolution rescue”；它不能否定“output-head spatial freedom”。**

更准确地说，负性尺度结果使 PBRU 的论证必须从“higher resolution”改成：

> **output phase freedom / learned upsampling freedom / fixed-interpolation bottleneck。**

---

## 0.3 但我不建议立刻开 300 epoch：先补一个零训练 stage-wise discriminability 诊断

当前 `analyse/levir_error_profile.py` 的实际代码只实现了：

1. connected-component size FN；
2. 192/256/320/384 scale sensitivity。

**它没有实现 Run6 设计文档中写过的 stage-wise discriminability。**

这正是 Run7 前最后一个高信息量、低成本证据缺口。

我的建议不是再做一轮长诊断，而是：

> **先补一个 1 次离线 hook 诊断，再立即实现 PBRU。**

时间上不应拖慢 Run7：

```text
补诊断脚本
→ Run2 checkpoint 跑一次 train-subset + test-subset feature probe
→ 无强反证则立即进入 PBRU
```

重点 hook：

```text
Encoder:
P1/Q1
P2/Q2
P3/Q3
P4/Q4

TAR:
t1/t2/t3/t4

Decoder:
fuse1 output
block1 output
refine output
```

建议输出两类指标：

- normalized Fisher separation；
- train-centroid / closed-form linear discriminant 在 test 上的 AUROC。

### 对 PBRU 的支持模式

如果：

```text
t1 / fuse1 / refine 仍有较好的 changed-vs-background separability
但最终 bilinear head 的 small-object recall 很差
```

则强支持：

> **信息仍在 feature 中，但最终输出映射的空间自由度不足。**

### 对 PBRU 的强反证

如果：

```text
t1 已弱
或
fuse1/refine 的 AUROC/Fisher 相对 t1 显著塌陷
```

那么 bottleneck 在更早的 decoder 表征阶段，单改输出头很可能救不回来。

**我的执行意见：先补这个诊断，但只作为 preflight，不再做任何超参搜索。**

---

# 1. 证据表：Run7 为什么值得验证 PBRU

## 1.1 代码/日志直接事实

当前 GitHub `main` 的关键事实：

- `STRRepNet.forward()`：
  - Siamese VMamba；
  - TAR；
  - DCR；
  - `head = nn.Conv2d(dim, 2, 1)`；
  - `F.interpolate(..., size=pre.shape[-2:], mode="bilinear", align_corners=False)`。
- 默认 `dim=160`。
- `dim` 同时被 TAR 输出宽度和 DCR 统一宽度使用。
- Run6 NSCR 只改 `fuse1/fuse2`，部署最终仍回到原来的单个 cross-scale 1×1。
- 当前重参数化组合采用 FP64，中间不 cast，最后一次写回 FP32。
- Run2 full_last2 deploy：
  - Params = **28.829M**
  - FLOPs = **12.6062G**
- Run6 4 个数据集 deploy 仍为同一预算，argmax disagreement = 0。

Run2 正式六指标：

| Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| CDD | 0.9849 | 0.9835 | 0.9961 | 0.9842 | 0.9689 | 0.9820 |
| LEVIR | 0.9032 | 0.9260 | 0.9914 | 0.9144 | 0.8424 | 0.9099 |
| SYSU | 0.7956 | 0.8775 | 0.9256 | 0.8345 | 0.7160 | 0.7866 |
| WHU | 0.9435 | 0.9595 | 0.9962 | 0.9514 | 0.9074 | 0.9494 |

Run6 正式六指标：

| Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| CDD | 0.9842 | 0.9836 | 0.9960 | 0.9839 | 0.9683 | 0.9816 |
| LEVIR | 0.9073 | 0.9208 | 0.9913 | 0.9140 | 0.8416 | 0.9094 |
| SYSU | 0.7901 | 0.8824 | 0.9257 | 0.8337 | 0.7148 | 0.7860 |
| WHU | 0.9426 | 0.9596 | 0.9961 | 0.9510 | 0.9066 | 0.9490 |

---

## 1.2 证据支持的推断

目前可以合理推断：

1. **LEVIR 的剩余问题不是简单 encoder 不适配。**  
   full encoder 只能把 LEVIR 从 0.9144 推到 0.9168，小目标 recall 也几乎不变。

2. **不是“输入像素不够”。**  
   320/384 test scale 没有提高 Recall。

3. **decoder 确实可以被推向更高 Recall，但容易用 Precision 换回来。**  
   Run6 就是直接证据。

4. **最终预测头是尚未被单独验证的结构瓶颈。**  
   前 6 轮主要改 TAR、局部 DCR、边界先验、cross-scale fusion；最终 `160→2 + bilinear` 基本没有被触碰。

---

## 1.3 待验证假设 H7

Run7 只验证一个主假设：

> **H7：LEVIR 小目标漏检的一部分来自最终 coarse-logit + fixed bilinear 输出约束。将固定插值改为 phase-specific learned projection，并通过训练期 phase-basis reparameterization 改善优化，可在不突破原 deploy budget 的情况下提高 F1/IoU，同时不再次出现 Run6 式 Precision 大幅下降。**

可证伪：

- 若预算匹配后的 PBRU 在 LEVIR 仍低于失败线；
- 或 Precision 再次跌破预注册底线；
- 或 plain PixelShuffle 已获得全部收益而 phase-basis rep 没有额外贡献；

则不能把 PBRU 的**结构重参数化机制**作为论文主创新。

---

# 2. Run7 主方案：PBRU

## 2.1 命名

**Phase-Basis Reparameterized Upsampling**

中文建议：

> **相位基重参数化上采样**

论文中建议缩写：

> **PBRU**

---

## 2.2 部署图

设 decoder 最终 feature：

\[
X\in\mathbb{R}^{D\times H\times W},
\quad H=W=64,\quad r=4,\quad C=2.
\]

部署时：

\[
Z = W_{\text{dep}} * X + b_{\text{dep}},
\]

其中：

\[
W_{\text{dep}}\in
\mathbb{R}^{(Cr^2)\times D\times1\times1}.
\]

然后：

\[
Y = \operatorname{PixelShuffle}_r(Z)
\in \mathbb{R}^{C\times rH\times rW}.
\]

对于本任务：

```text
C = 2
r = 4
Cr² = 32
```

部署图只有：

```text
refine feature
   ↓
single 1×1 projection: D* → 32
   ↓
PixelShuffle(4)
   ↓
2×256×256 logits
```

`PixelShuffle` 是固定 permutation/reshape：

- 0 参数；
- 不引入可学习动态分支；
- 不需要 teacher；
- 部署仍是单路径。

---

# 3. 为什么 PBRU 增加的是“输出相位自由度”

原 head：

\[
L_c(h,w) = W_c X(h,w)+b_c,\quad c\in\{0,1\}
\]

然后高分辨率像素由固定 bilinear 决定：

\[
Y_c(i,j)
=
\sum_{(h,w)\in\mathcal N(i,j)}
\lambda_{i,j,h,w}L_c(h,w).
\]

网络只能学习 coarse logits \(L\)，不能学习 \(\lambda\)。

PBRU 中，每个 low-resolution cell 直接产生 16 个 sub-pixel phase 的类别 logits：

\[
Z_{c,p}(h,w)
=
W_{c,p}X(h,w)+b_{c,p},
\quad p=0,\dots,15.
\]

PixelShuffle 仅将：

\[
Z_{c,p}(h,w)
\]

放到对应的：

\[
Y_c(4h+i,4w+j).
\]

因此网络可学习：

```text
同一个 64×64 feature cell 内
16 个空间 phase
各自不同的 change/background 分类边界。
```

这才是 PBRU 的机制核心。

---

# 4. 训练图：主 projection + coarse/phase basis 可折叠分支

我不建议一上来做很多可学习 spatial kernel，也不建议再引入 loss。

训练图保持纯线性、多分支、可解析合并：

```text
X
├─ Main: 1×1 D→32
├─ Coarse basis branch: 1×1 D→2 + BN → fixed phase expansion φ00
├─ Phase-X branch:      1×1 D→2 + BN → fixed phase expansion φ10
├─ Phase-Y branch:      1×1 D→2 + BN → fixed phase expansion φ01
└─ Phase-XY branch:     1×1 D→2 + BN → fixed phase expansion φ11
          ↓ sum
        Z_train
          ↓
      PixelShuffle(4)
          ↓
       logits 2×256²
```

不增加任何额外 loss。

---

## 4.1 固定 4×4 phase basis

令 phase 坐标：

\[
u_j = \frac{2j+1-r}{r},\quad
v_i = \frac{2i+1-r}{r},
\quad i,j=0,\dots,r-1.
\]

当 \(r=4\)：

```text
u,v = [-0.75, -0.25, 0.25, 0.75]
```

固定四个低阶 basis：

\[
\phi_{00}(i,j)=1
\]

\[
\phi_{10}(i,j)=u_j
\]

\[
\phi_{01}(i,j)=v_i
\]

\[
\phi_{11}(i,j)=u_jv_i
\]

含义：

- `φ00`：16 个 phase 共用的 coarse change evidence；
- `φ10`：水平 phase variation；
- `φ01`：垂直 phase variation；
- `φ11`：二维交互 variation。

这四个 basis 是固定常数，不训练、不增加 deploy 参数。

**Run7 不做 basis 数量 sweep。**

---

# 5. PBRU 严格折叠数学

## 5.1 主分支

主分支：

\[
Z_m = W_mX+b_m,
\]

\[
W_m\in\mathbb{R}^{Cr^2\times D\times1\times1}.
\]

---

## 5.2 第 k 个 phase-basis 分支

第 \(k\) 个 branch 先输出 2-channel：

\[
Q_k = \operatorname{BN}_k(V_k*X).
\]

在 eval/fold 时 BN 可写成：

\[
Q_k = \bar V_k*X+\bar b_k.
\]

其中：

\[
\bar V_k\in\mathbb{R}^{C\times D\times1\times1}.
\]

固定 phase expansion \(E_{\phi_k}\) 定义：

\[
[E_{\phi_k}(Q_k)]_{c r^2+p}
=
\phi_k[p]\cdot [Q_k]_c.
\]

因此：

\[
Z_{\text{train}}
=
Z_m+\sum_k E_{\phi_k}(Q_k).
\]

展开得：

\[
Z_{c,p}
=
W_{m,c,p}X+b_{m,c,p}
+
\sum_k
\phi_k[p](\bar V_{k,c}X+\bar b_{k,c}).
\]

整理：

\[
W_{\text{eq},c,p}
=
W_{m,c,p}
+
\sum_k\phi_k[p]\bar V_{k,c},
\]

\[
b_{\text{eq},c,p}
=
b_{m,c,p}
+
\sum_k\phi_k[p]\bar b_{k,c}.
\]

所以：

\[
Z_{\text{train}}
=
W_{\text{eq}}X+b_{\text{eq}}.
\]

之后 PixelShuffle 是固定置换 \(S_r\)：

\[
Y_{\text{train}}
=
S_r(Z_{\text{train}})
=
S_r(W_{\text{eq}}X+b_{\text{eq}})
=
Y_{\text{deploy}}.
\]

**结论：四个训练期 phase-basis branch 可以严格解析吸收到单个 `D→32` 1×1 projection。**

---

# 6. 零初始化策略

PBRU 的零初始化目标应该是：

> **M1_PBRU 与 plain PixelShuffle control 在 epoch 0 拥有同一个主预测。**

不是强求它与 Run2 bilinear head 初始输出完全一致——二者 deploy topology 本来就不同。

建议：

```python
main_proj: normal Kaiming init

for each phase branch:
    conv: Kaiming init
    BN.gamma = 0
    BN.beta  = 0
```

这样：

```text
epoch 0:
PBRU auxiliary output = exactly 0
M1_PBRU prediction = plain PixelShuffle prediction
```

随后：

- gamma 首先获得梯度；
- branch 再逐步展开；
- 可直接比较 `M1_PBRU` 与 `C1_PixelShufflePlain`。

建议训练结束日志记录：

```text
[PBRU-GAMMA-NORM] coarse=...
[PBRU-GAMMA-NORM] phase_x=...
[PBRU-GAMMA-NORM] phase_y=...
[PBRU-GAMMA-NORM] phase_xy=...
```

只用于机制解释，不用于挑 checkpoint。

---

# 7. 预算问题：为什么必须机器搜索

旧 head 参数：

\[
P_{\text{old}} = 2D + 2.
\]

PBRU deploy head：

\[
P_{\text{pbru}} = 2r^2D + 2r^2
=32D+32.
\]

在 \(D=160\) 时：

```text
old head = 322 params
PBRU head = 5152 params
Δ = +4830 params
```

head FLOPs 也会增加约 16 倍，但相对整网仍很小。

然而你的约束不是“增加很少也可以”，而是：

> **部署 Params/FLOPs 不能超过当前 Run2。**

所以必须从共享 width \(D\) 回收。

---

## 7.1 一个重要代码事实：当前 `dim` 不是 decoder-only

`STRRepNet.py` 当前：

```python
self.tar = MultiScaleTAR(..., dim=dim)
self.decoder = DCRDecoder(dim=dim, ...)
self.head = nn.Conv2d(dim, 2, 1)
```

因此把：

```text
D=160 → D*
```

会同时影响：

- TAR 的输出 width；
- DCR 的 width；
- final head 输入 width。

所以论文/README 不应写成：

> “只压 decoder channel”。

更准确应写：

> **budget-recovery shared representation width adjustment**  
> 或  
> **预算回收宽度调整**。

这也是为什么 Run7 必须设置 width control。

---

# 8. 结构改造与通道回收如何解耦归因

这是 Run7 最关键的实验设计。

## 8.1 A0：历史锚点，不重跑

```text
A0_Run2_D160_Bilinear
D=160
old head 160→2 + bilinear
F1=0.9144
IoU=0.8424
Precision=0.9260
```

直接复用正式 Run2 结果。

---

## 8.2 C0：预算回收 width control

```text
C0_Bilinear_Dstar
D=D*
old bilinear head
PBRU off
```

它只回答：

> 把统一 width 从 160 降到 D* 本身会损失多少？

定义：

\[
\Delta_{\text{width}}
=
F1(C0)-F1(A0).
\]

---

## 8.3 C1：plain learned-upsample control

```text
C1_PixelShufflePlain_Dstar
D=D*
head D*→32
PixelShuffle(4)
phase-basis branches OFF
```

它回答：

> 仅把 fixed bilinear 改成 phase-specific learned projection，有多少收益？

定义：

\[
\Delta_{\text{head}}
=
F1(C1)-F1(C0).
\]

---

## 8.4 M1：完整 PBRU

```text
M1_PBRU_Dstar
D=D*
head D*→32
PixelShuffle(4)
coarse + phase-x + phase-y + phase-xy branches ON
deploy fold → one D*→32 conv
```

结构重参数化本身的贡献：

\[
\Delta_{\text{rep}}
=
F1(M1)-F1(C1).
\]

最终预算中性系统收益：

\[
\Delta_{\text{net}}
=
F1(M1)-F1(A0).
\]

---

## 8.5 这四组能回答四个不同问题

| 比较 | 回答的问题 |
|---|---|
| C0 − A0 | 预算回收 width 代价 |
| C1 − C0 | learned phase upsampling / output freedom |
| M1 − C1 | **phase-basis structural reparameterization** 的额外贡献 |
| M1 − A0 | 最终 budget-neutral 系统净收益 |

因此不会出现：

> “PBRU 涨点了，但不知道是 PixelShuffle、channel shrink 还是 rep branch 导致的。”

---

# 9. 预算搜索脚本

新增：

```text
analyse/search_pbru_budget.py
```

## 9.1 不要用 28.829M 的四舍五入值做硬判断

脚本首先自动构造：

```text
Run2 deploy model:
D=160
bilinear head
```

记录：

```text
P_cap_raw = exact integer parameter count
F_cap_raw = exact fvcore FLOPs
```

然后搜索：

```python
for D in range(160, 127, -1):
    build PBRU deploy model(D)
    params = exact_numel(model)
    flops = fvcore_flops(model)

    feasible = (
        params <= P_cap_raw and
        flops <= F_cap_raw
    )
```

选：

```text
最大的 feasible D
```

即：

\[
D^*
=
\max\{D:
P_{\text{PBRU}}(D)\le P_{\text{Run2}},
F_{\text{PBRU}}(D)\le F_{\text{Run2}}\}.
\]

不手工指定 158/156/159。

---

## 9.2 搜索输出

同时生成：

```text
docs/temporary/Run7_PBRU_budget_search.json
docs/temporary/Run7_PBRU_budget_search.csv
```

至少字段：

```text
D
head_mode
deploy_params_raw
deploy_params_M
deploy_flops_G
unsupported_ops
params_margin
flops_margin
feasible
selected
git_head
```

README 只抄机器结果，不人工算。

---

# 10. 逐文件修改清单

## 10.1 `models/changedetection/models/reparam.py`

新增：

```text
PhaseBasisBranch
PBRUHead
build_phase_basis(r=4)
expand_phase_kernel(...)
```

建议接口：

```python
class PBRUHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=2,
        upscale=4,
        use_phase_rep=True,
        deploy=False,
    ):
        ...
```

核心成员：

```text
main_proj: Conv2d(D, 32, 1, bias=True)

train-only:
branch_coarse: Conv2d(D,2,1,bias=False)+BN
branch_px:     Conv2d(D,2,1,bias=False)+BN
branch_py:     Conv2d(D,2,1,bias=False)+BN
branch_pxy:    Conv2d(D,2,1,bias=False)+BN

fixed buffers:
phi00/phi10/phi01/phi11
```

必须实现：

```text
forward()
get_equivalent_kernel_bias()
branch_stats()
switch_to_deploy()
```

折叠组合全程 FP64，最后一次 cast FP32。

部署后必须删除：

```text
branch_*
bn_*
phase basis training objects
main train graph
```

仅保留：

```text
proj: Conv2d(D,32,1,bias=True)
PixelShuffle(4)
```

---

## 10.2 `models/changedetection/models/dcr_decoder.py`

主数据流不需要改。

只需要保证：

```text
decoder output仍是 [B,D,H/4,W/4]
```

并支持 `dim=D*`。

不要：

- 新增额外 decoder block；
- 改 fuse1/fuse2；
- 同时开 NSCR；
- 改 refine。

Run7 要保持输出头为唯一机制变量。

---

## 10.3 `models/changedetection/models/STRRepNet.py`

新增参数：

```python
dim=160
head_mode="bilinear"   # bilinear | pixelshuffle
use_pbru=False
pbru_upscale=4
```

建议逻辑：

```python
if head_mode == "bilinear":
    self.head = nn.Conv2d(dim, 2, 1)
elif head_mode == "pixelshuffle":
    self.head = PBRUHead(
        dim,
        num_classes=2,
        upscale=4,
        use_phase_rep=use_pbru,
    )
```

forward：

```python
x = self.decoder(feats)

if head_mode == "bilinear":
    logits = self.head(x)
    logits = F.interpolate(
        logits,
        size=pre.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
else:
    logits = self.head(x)
```

必须 assert：

```text
input 256 -> d1 64
pbru_upscale=4
final logits exactly 256
```

`switch_to_deploy()`：

```python
self.tar.switch_to_deploy()
self.decoder.switch_to_deploy()

if hasattr(self.head, "switch_to_deploy"):
    self.head.switch_to_deploy()
```

---

## 10.4 `models/changedetection/script/train.py`

新增 CLI：

```text
--decoder_dim
--head_mode bilinear|pixelshuffle
--use_pbru 0|1
--pbru_upscale 4
```

禁止 Run7 同时：

```text
use_nscr=1
use_botr=1
```

建议加显式配置检查：

```python
if args.use_pbru:
    assert args.head_mode == "pixelshuffle"
    assert not args.use_nscr
    assert not args.use_botr
```

最终 TEST block 新增：

```text
[DECODER-DIM] D*
[HEAD-MODE] pixelshuffle
[PBRU] 1
[PBRU-UPSCALE] 4
[PBRU-BASIS] bilinear4
```

以及 branch norm。

---

## 10.5 `models/changedetection/script/smoke_test.py`

新增 PBRU smoke：

1. build；
2. output shape；
3. aux branch zero-init；
4. branch gamma 有梯度；
5. local head folding；
6. whole-network folding；
7. deploy graph branch deletion；
8. PixelShuffle 后 256×256；
9. exact deploy Params/FLOPs；
10. 与 budget cap 对比。

必须检查：

```text
argmax_disagree == 0
```

---

## 10.6 `models/changedetection/script/test_reparam_equivalence.py`

新增四层验证：

### T0：phase expansion 代数测试，FP64

随机：

```text
X
Wmain
Vk
bk
phi
```

直接 train formula vs equivalent `W_eq/b_eq`：

```text
max abs < 1e-12
```

### T1：`PBRUHead` block test

```text
train graph vs deploy graph
```

建议 PBRU head 本身目标：

```text
max_abs_error < 1e-6
```

如果因一次 FP32 cast 稍高，则记录，不应超过当前 block-level envelope。

### T2：whole network

不要求虚构 `<1e-6`。

沿用当前诚实标准：

```text
record actual FP32 error
不超过现有 whole-model envelope
argmax_disagree = 0
```

### T3：branch deletion

deploy state_dict 不得存在：

```text
branch_coarse
branch_px
branch_py
branch_pxy
bn_*
```

---

## 10.7 `analyse/search_pbru_budget.py`

用途见 §9。

---

## 10.8 新增 stage diagnosis

建议新增：

```text
analyse/levir_stage_discriminability.py
```

不要继续把所有逻辑塞进已有 `levir_error_profile.py`，避免一个脚本职责过重。

---

# 11. Stage-wise discriminability 的建议实现

## 11.1 Encoder P/Q

对每一层：

\[
\Delta_i=|P_i-Q_i|.
\]

将 feature vector 作为像素样本。

---

## 11.2 TAR / decoder

直接使用：

```text
t1/t2/t3/t4
fuse1
block1
refine
```

对应像素 feature vector。

---

## 11.3 Fisher separation

设 change/background 两类 feature 均值：

\[
\mu_1,\mu_0.
\]

建议用维度归一化的 Fisher：

\[
J=
\frac{\|\mu_1-\mu_0\|_2^2/C}
{\operatorname{tr}(\Sigma_1+\Sigma_0)/C+\epsilon}.
\]

避免不同 channel 数层之间因维度不同而直接不可比。

---

## 11.4 AUROC

不要在 test 上拟合。

流程：

```text
固定 Run2 model
train split 抽样 feature
→ 估计 μ1, μ0 和 diagonal variance
→ 构造 closed-form diagonal LDA score
→ test split 只评估 AUROC
```

同时输出：

```text
all-change AUROC
small-component AUROC
```

small threshold 固定使用现有 train quantile：

```text
small <= 502 px
```

---

## 11.5 PBRU preflight 判读

### 支持

例如：

```text
t1 small AUROC 较高
fuse1/refine 没有明显崩掉
```

说明：

> feature 中仍保有小目标可判别信息，最终 output mapping 值得改。

### 强反证

若：

```text
refine small AUROC 接近 chance
或
相对 t1 出现明显大幅下降
```

则先不要开 PBRU 300 epoch。

这里不建议再发明一个精细阈值调参；只把它当“强反证检查”。

---

# 12. Run7 最小实验组

## 12.1 第一阶段只做 LEVIR

### A0：历史锚点

```text
A0_Run2_full_last2
D=160
bilinear
PBRU=0
```

**不重跑。**

正式锚点：

```text
Recall    0.9032
Precision 0.9260
F1        0.9144
IoU       0.8424
```

---

### C0：width control

```text
C0_Bilinear_Dstar
D=D*
head=bilinear
PBRU=0
encoder_train=last2
```

---

### C1：plain PixelShuffle

```text
C1_PixelShufflePlain_Dstar
D=D*
head=pixelshuffle
PBRU=0
encoder_train=last2
```

---

### M1：完整 PBRU

```text
M1_PBRU_Dstar
D=D*
head=pixelshuffle
PBRU=1
basis=bilinear4
encoder_train=last2
```

---

## 12.2 为减少无效算力的启动顺序

建议：

```text
Phase 0
budget search + stage discriminability

Phase 1
C0_Bilinear_Dstar / LEVIR
M1_PBRU_Dstar      / LEVIR
```

两卡可独立跑。

若 M1 明确 FAIL，且 C0 不差：

```text
停止 PBRU
C1 不必再跑
```

若 M1 PASS/WEAK：

```text
再跑 C1_PixelShufflePlain_Dstar
```

用于判断 phase-basis rep 的真实贡献。

---

# 13. 四数据集扩展顺序

只有同时满足：

```text
M1 对 A0 达到系统 PASS
且
M1 对 C1 有可辨识的 rep 增益
```

才扩。

顺序：

```text
LEVIR
→ WHU
→ SYSU
→ CDD
```

原因：

1. LEVIR：当前明确瓶颈集；
2. WHU：同为 building/small-target，可验证机制是否仅 LEVIR-specific；
3. SYSU：类别/场景更杂，检验泛化；
4. CDD：高 ceiling，最后跑。

---

# 14. checkpoint / log / script 路径

建议统一：

```text
train_scripts/TAR-DCR/Run7/
  C0_Bilinear_Dstar/
  C1_PixelShufflePlain_Dstar/
  M1_PBRU_Dstar/
```

checkpoint：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run7/<group>/<dataset>/
```

日志：

```text
/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run7/<group>/<dataset>/train_log.txt
```

诊断：

```text
/home/yqwang/outputs/STR-RepNet/diagnostics/Run7_PBRU/
```

预算搜索：

```text
/home/yqwang/outputs/STR-RepNet/diagnostics/Run7_PBRU/budget_search.json
```

本地归档：

```text
docs/temporary/Run7_PBRU_budget_search.json
docs/temporary/Run7_PBRU_budget_search.csv
```

跨组 checkpoint 不互相 resume。

---

# 15. README 预算记账格式

Run7 README 中单列：

| Group | D | Head | Phase Rep | Deploy Params | Deploy FLOPs | vs cap Params | vs cap FLOPs |
|---|---:|---|---:|---:|---:|---:|---:|
| A0 Run2 | 160 | 2ch+bilinear | 0 | exact cap | exact cap | 0 | 0 |
| C0 | D* | 2ch+bilinear | 0 | machine | machine | ≤0 | ≤0 |
| C1 | D* | 32ch+PS | 0 | machine | machine | ≤0 | ≤0 |
| M1 | D* | 32ch+PS | 1→fold | **与 C1 deploy 相同** | **与 C1 deploy 相同** | ≤0 | ≤0 |

注意：

> `M1` 的 train graph 参数可以增加；硬约束只针对 deploy。

---

# 16. 预注册判据

## 16.1 LEVIR 系统级 PASS/WEAK/FAIL

我建议**不移动 Run6 的门槛**，避免事后降低标准。

锚点：

```text
F1        = 0.9144
IoU       = 0.8424
Precision = 0.9260
```

### PASS

同时满足：

```text
F1        >= 0.9175
IoU       >= 0.8475
Precision >= 0.9230
```

且：

```text
Deploy Params <= Run2 exact cap
Deploy FLOPs  <= Run2 exact cap
argmax disagreement = 0
```

---

### WEAK

任一：

```text
0.9159 <= F1 < 0.9175
```

或：

```text
F1 达标
但 IoU < 0.8475
或 Precision < 0.9230
```

处理：

```text
记录
不做 basis sweep
不改 r
不改 loss
最多补 C1 做归因
不直接扩四数据集
```

---

### FAIL

任一：

```text
F1 < 0.9159
```

或：

```text
Precision < 0.9220
```

或部署预算/等价性失败。

处理：

> **停止 PBRU，不救机制。**

不做：

```text
r=2/r=8 sweep
basis 数量 sweep
gamma sweep
额外边界 loss
full encoder 叠加
NSCR+PBRU
```

---

# 17. “结构重参数化是否成立”的第二层判据

仅仅 M1 超过 A0 还不够。

因为论文主线是 structural re-parameterization。

必须看：

```text
M1_PBRU
vs
C1_PixelShufflePlain
```

建议预注册解释强度：

### Rep-supported

```text
M1 - C1 >= +0.15pp F1
```

且 Precision 不比 C1 下降超过 0.15pp。

### Rep-weak

```text
+0.05pp <= M1-C1 < +0.15pp
```

只能称：

> 单 seed 下有正向迹象。

不能称稳定/普适提升。

### Rep-not-supported

```text
M1-C1 < +0.05pp
```

即使 M1 对 A0 很好，也应解释为：

> 主要收益来自 learned PixelShuffle head，而不是 phase-basis training-time reparameterization。

这时 **PBRU 不能作为“结构重参数化核心贡献”强推**。

---

# 18. 四数据集扩展判据

沿用 Run6 强度：

```text
Macro F1 >= 92.25%
LEVIR     >= 91.75%
至少 3/4 数据集 ΔF1 >= 0
任一数据集退化 <= 0.15pp
```

同时每个数据集都要求：

```text
deploy cap 合格
argmax disagreement = 0
```

如果：

```text
LEVIR PASS
WHU 明显 FAIL
```

则优先判断为：

> LEVIR-specific output-head effect。

不要立刻跑 SYSU/CDD 做“凑平均”。

---

# 19. 失败后的预设决策

## 情况 A：M1 FAIL，C0 ≈ A0

说明：

> budget width recovery 没有明显伤害，但 PBRU 没有解决问题。

决策：

```text
停止 PBRU
不改 r/basis
回到 stage discriminability
```

下一方向更应考虑：

> refine/fine-stage feature representation，而不是 output upsampling。

---

## 情况 B：M1 FAIL，C0 也明显下降

说明：

> 为满足原 budget 所需的 width adjustment 已经破坏了表示能力。

这不是 PBRU 可以靠调 basis 救的问题。

决策：

```text
停止当前 PBRU 版本
```

若未来再做，只能设计**不同的预算回收位置**，不能把 D 再手工扫一圈。

---

## 情况 C：M1 PASS，但 C1 与 M1 几乎相同

说明：

> learned phase-specific head 有效，但 phase-basis structural rep 没有证据。

论文可以把 PixelShuffle head 作为架构观察，但不能把 PBRU rep 分支作为核心创新宣称。

---

## 情况 D：C1 一般，M1 明显更好

这是 Run7 最理想的科研结果：

```text
width penalty 可控
learned head 单独不足
phase-basis rep 带来额外增益
deploy 完全一致
```

这样 PBRU 才真正符合课题标题。

---

## 情况 E：Recall 再涨但 Precision 明显跌

若再次出现 Run6 型：

```text
Recall ↑
Precision ↓ > 0.4pp
F1 不过线
```

直接判断：

> 输出自由度只扩大了 positive support，没有提高判别可靠性。

**停止，不加 precision loss / boundary loss / threshold tuning。**

---

# 20. P0 / P1 / P2 风险分级

## P0：正确性风险

### P0-1 PixelShuffle channel ordering

必须按 PyTorch 的：

```text
channel = c*r² + phase
phase   = i*r + j
```

构造 basis expansion。

如果 channel order 写错：

- fold 仍可能数值“自洽”；
- 但 phase 的空间位置会错；
- 结果可能出现规则棋盘伪影。

必须单测一个 one-hot phase tensor。

### P0-2 预算 cap 使用 rounded 值

不能用：

```text
28.829M
12.6062G
```

字符串比较。

必须在同一脚本里重建 Run2 anchor 获取 raw cap。

### P0-3 `dim` 实际同时改变 TAR + DCR

README/论文要如实写，不要叫 decoder-only shrink。

### P0-4 恢复兼容性

旧 Run2 checkpoint：

```text
head.weight = [2,160,1,1]
```

与 PBRU：

```text
main_proj.weight = [32,D*,1,1]
```

不兼容。

因此：

- Run7 不允许误加载旧 `last.pth`；
- 每组必须独立 ckpt dir；
- resume 只允许同组同配置。

---

## P1：方法瓶颈风险

### P1-1 PBRU 缺少跨 cell spatial mixing

`1×1→PixelShuffle` 只做 phase-specific projection，不显式混合相邻 64×64 cells。

潜在风险：

- 4×4 block seam；
- checkerboard；
- Precision 下滑。

但不要在 Run7 第一版加 3×3 head，否则变量又变多。

### P1-2 小目标 FN 可能早在 refine 前就已丢失

这就是必须先补 stage-wise diagnosis 的原因。

### P1-3 结构收益可能全部来自 PixelShuffle

C1 是必要对照，不能省。

---

## P2：实验工程风险

- 两卡独立跑，不用 DDP；
- 保持 `num_workers` 当前稳定配置；
- 新脚本目录独立；
- 训练前先 smoke；
- 真数据先 2–5 batch dry run；
- `extract_metrics_to_excel.py` 继续只读最后完整 TEST block；
- Git 不提交 checkpoint / outputs / cache。

---

# 21. smoke / dry run / deploy 验收

## 21.1 单元 smoke

必须全部通过：

```text
[ ] phase basis shape = [4,16]
[ ] phase ordering one-hot test
[ ] branch output = 0 at init
[ ] branch gamma grad != 0
[ ] PBRU output = [B,2,256,256]
[ ] local FP64 algebra error < 1e-12
[ ] PBRU head fold error ideally < 1e-6
[ ] whole-model fold error不劣于现有量级
[ ] argmax disagreement = 0
[ ] deploy state_dict无 train-only branch
[ ] deploy params/flops <= raw anchor cap
```

---

## 21.2 真数据 dry run

LEVIR：

```text
2–5 batches
forward
CE
Lovasz
backward
optimizer.step
```

确认：

```text
loss finite
main_proj grad finite
phase gamma grad finite
encoder last2 训练状态正确
stage1/2 frozen
```

---

# 22. 建议的 Run7 启动顺序

## Step 1：冻结证据与 commit

记录：

```bash
git rev-parse HEAD
# 6a746b381f05a7a50828c4841c91292cfb49ad62
```

---

## Step 2：补 stage discriminability

新增：

```text
analyse/levir_stage_discriminability.py
```

只跑 Run2 checkpoint。

**不训练。**

---

## Step 3：实现 PBRUHead

只改：

```text
reparam.py
STRRepNet.py
train.py
smoke_test.py
test_reparam_equivalence.py
```

`dcr_decoder.py` 仅检查 `dim` 兼容，不增加新机制。

---

## Step 4：跑 budget search

得到唯一：

```text
D*
```

提交进 Run7 README。

---

## Step 5：完整单元测试

先：

```text
phase ordering
FP64 fold
PBRU block
whole-model
```

再 smoke。

---

## Step 6：LEVIR dry run

2–5 batch。

---

## Step 7：第一阶段训练

优先同时：

```text
GPU0: C0_Bilinear_Dstar / LEVIR
GPU1: M1_PBRU_Dstar      / LEVIR
```

---

## Step 8：一次性裁决

### M1 FAIL

停止。

### M1 WEAK/PASS

补：

```text
C1_PixelShufflePlain_Dstar / LEVIR
```

做结构归因。

---

## Step 9：只有结构归因成立才扩

```text
WHU
→ SYSU
→ CDD
```

---

# 23. 对 Run7 论文叙事的建议

如果 Run7 成功，论文不要写：

> “higher resolution improves small-object change detection”。

现有证据不支持。

建议写成：

> **The remaining false negatives are not alleviated by increasing the input test resolution or by further encoder adaptation, indicating that the bottleneck is unlikely to be input-side spatial sampling. We instead identify the fixed coarse-logit interpolation at the prediction head as a restrictive output parameterization. PBRU replaces the fixed bilinear mapping with phase-specific learned logits and introduces train-time low-order phase-basis branches that are analytically merged into a single sub-pixel projection for deployment.**

中文：

> **现有小目标漏检无法通过提高输入测试尺度或进一步 encoder 适配解决，说明瓶颈并非输入侧空间采样不足。我们进一步关注最终预测头的固定 coarse-logit 插值约束，通过相位特定的可学习输出投影释放全分辨率预测自由度，并利用训练期低阶相位基分支进行结构重参数化，部署时解析合并为单个 sub-pixel projection。**

这比“高分辨率头”更严谨。

---

# 24. 当前最关键的科研判断

我对 Run7 的建议可以压缩成一句话：

> **可以做 PBRU，但必须把它定义成“输出空间相位自由度”问题，而不是“输入尺度问题”；先用一次 stage-wise discriminability 排除 refine 之前已经丢信息的强反证，再以机器搜索得到唯一 D*，通过 A0/C0/C1/M1 四层归因把 width recovery、PixelShuffle topology 和 phase-basis structural reparameterization 三个变量拆开。**

Run7 真正值得追求的不是：

```text
PBRU 最终 F1 比 0.9144 高一点
```

而是完整证据链：

```text
small FN 集中
+ full encoder 不救
+ input scale 不救
+ refine feature仍可判别
+ learned phase head有效
+ phase-basis rep 在同一 deploy graph上再有增益
+ 部署预算不超过 Run2
+ fold后 argmax恒等
```

只有这条链成立，PBRU 才足以成为毕业论文「结构重参数化」主线中的真正贡献，而不是一次普通 head replacement。

---

# 25. 立即执行顺序

```text
1. 补 levir_stage_discriminability.py（零训练）
2. 实现 PBRUHead + phase basis folding
3. 修改 STRRepNet/train/smoke/equivalence
4. 新增 search_pbru_budget.py
5. 机器搜索唯一 D*
6. FP64 解析折叠测试
7. PBRU block + whole-model fold/argmax 测试
8. LEVIR 2–5 batch dry run
9. C0_Bilinear_Dstar + M1_PBRU_Dstar
10. M1 过线/弱阳性后补 C1_PixelShufflePlain_Dstar
11. 结构归因成立后 WHU → SYSU → CDD
12. 更新 README + Run7 设计文档 + metrics 表
```

---

# 26. 仍需补充证据

Run7 开跑前只缺三项：

1. **stage-wise discriminability**  
   当前仓库尚未实现；这是最值得补的零成本诊断。

2. **机器搜索得到的 D\***  
   不能手算、不能预设。

3. **PBRU 实际 FP32 fold error**  
   解析数学成立不等于 RTX 5090 的整个 FP32 graph 一定达到某个理想误差值；应继续采用：
   - FP64 algebraic equivalence；
   - block-level actual error；
   - whole-model actual error；
   - argmax disagreement=0；
   的四层证据。

---

# 27. 本次审查依据

GitHub `main`：

```text
README.md
models/changedetection/models/reparam.py
models/changedetection/models/dcr_decoder.py
models/changedetection/models/STRRepNet.py
models/changedetection/script/train.py
models/changedetection/script/smoke_test.py
models/changedetection/script/test_reparam_equivalence.py
docs/temporary/STR-RepNet_Run6_下一步改进方向与实验设计_NSCR-Fuse.md
docs/temporary/models_and_metrics_TAR-DCR_Run6.txt
analyse/levir_error_profile.py
train_scripts/TAR-DCR/Run6/README.md
```

审查时仓库最新：

```text
HEAD = 6a746b381f05a7a50828c4841c91292cfb49ad62
```

Baseline：

```text
HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection
IEEE TGRS, 2026
```

---

# 28. 最终决策

**Run7：做 PBRU。**

但执行方式不是：

```text
直接把 head 改成 PixelShuffle
然后找一个 D 跑四数据集
```

而是：

```text
stage preflight
    ↓
exact budget search → D*
    ↓
C0 width-control
    +
M1 full PBRU
    ↓
若 M1 有效，补 C1 plain PixelShuffle
    ↓
拆出：
width effect
head topology effect
structural-rep effect
    ↓
只有 structural-rep effect 成立
才扩 WHU → SYSU → CDD
```

这套 Run7 设计与前六轮最大的区别是：

> **不再只问“能不能涨点”，而是强制回答“涨点到底来自哪里，以及是否真的来自结构重参数化”。**
