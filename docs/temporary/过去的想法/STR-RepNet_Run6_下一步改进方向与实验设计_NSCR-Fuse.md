# STR-RepNet：Run2 回退后的下一步结构重参数化方向与 Run6 实验设计

> **课题**：轻量化遥感二值变化检测中的结构重参数化  
> **仓库**：`YuqiWang-code/STR-RepNet`  
> **审查对象**：GitHub `main` 最新状态（审查时 HEAD：`ab260efe63c24e76e279e437893f0b1b209e22a4`，2026-09-28，commit message=`update code`）  
> **任务**：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256，全监督二值变化检测，256×256，`A/B/label + list`，`gray >= 128`  
> **硬约束**：部署参数量/FLOPs 不高于 **28.829M / 12.6062G**；训练期新增结构必须可严格折叠或删除；不以 loss/训练技巧充当创新；seed=2333；当前阶段沿用 test-as-validation 协议  
> **结论日期**：2026-09-28

---

# 0. 结论先行

## 0.1 下一步主推：NSCR-Fuse —— Native-Scale Commutative Re-parameterized Fusion

回退到 Run2 `full_last2` 后，我最建议做的不是继续扩 TAR、继续做边界、继续加 decoder 模块，也不是先把 `encoder_train=full` 升格为正式主协议，而是：

> **在 DCR 的高分辨率跨尺度融合处，引入“原生尺度可交换重参数化分支”——NSCR-Fuse（Native-Scale Commutative Re-parameterized Fusion）。**

核心思想是：

- 训练期给 `fuse1 / fuse2` 的两个输入各增加一个**原生尺度 BN/仿射残差分支**：
  - 高分辨率 lateral 路：`BN_L(L)`；
  - 低分辨率 semantic 路：先在低分辨率做 `BN_H(H)`，再 bilinear upsample；
- 部署时**不保留任何新分支**；
- 利用“逐通道仿射变换与 bilinear interpolation 可交换”这一线性性质，把两个训练分支全部吸收到原有 `RepPairFuse1x1` 的 `[L, U(H)]` 两半输入权重中；
- 部署图仍然是：
  `bilinear upsample -> 原有单个 1×1 fuse -> 原有 DCR`
- **部署 Params/FLOPs 理论上完全不变**；
- 训练分支零初始化，不改变 Run2 初始主路径；
- 机制直接针对 Run5 预注册失败后已经写明的下一阶段：  
  **“高分辨率 / 小目标信息保真 + budget-neutral decoder channel allocation”**，但优先做其中更符合“结构重参数化”论文主线、变量更干净的那一半：**高分辨率跨尺度信息保真**。

我建议暂定中文名：

> **原生尺度可交换重参数化融合（NSCR-Fuse）**

英文：

> **Native-Scale Commutative Re-parameterized Fusion**

如果后续效果成立，可在论文中把它归入 DCR 的升级版本，而不是再造一个完全独立大模块。

---

## 0.2 为什么现在优先做它

当前结果已经给了很清楚的排除链：

1. **边界算子级先验**：Run3 Edge-Basis 四数据集整体无收益；
2. **边界监督级先验**：Run4 IBAS 只轻微抬 Precision，F1 不过线；
3. **时序顺序结构**：Run5 BOTR 与普通 swap 几乎完全重合；
4. **full encoder**：LEVIR 从 0.9144 到 0.9168，只有 +0.24pp，正向但未达到预注册的“主要瓶颈”线；
5. 当前 decoder 的最高分辨率仍只到 VMamba stage1，随后 `head(160->2)` 后直接 bilinear 回 256²。

所以剩余最值得验证的问题已经从：

> “TAR 是否不够复杂？”

转成：

> **“高分辨率 lateral detail 与上采样 semantic feature 在逐级融合时，是否被同一个融合投影过早混合、缺少独立的原生尺度统计/梯度路径？”**

这与现有证据的契合度明显高于再次扩 TAR/edge/loss。

---

## 0.3 `encoder_train=full` 的定位

**现在不要直接把 `full` 设成 Run6 主协议。**

原因不是它无效，而是：

- 它是当前 LEVIR 最好结果：**0.9168**；
- 但相对 Run2 `full_last2` 只有 **+0.24pp**；
- 如果 Run6 主结构同时切换到 `encoder_train=full`，则结构收益和训练协议收益混在一起；
- 论文中无法干净回答：“提升来自 NSCR，还是来自 full fine-tune？”

因此 Run6 第一阶段仍建议：

```text
encoder_train = last2
```

只改变 NSCR 这一项。

等 NSCR 本身成立后，再补：

```text
NSCR + full_encoder
```

看二者是否可叠加。

如果最终论文决定使用 full encoder，那么必须给四数据集重新建立**同协议 full-encoder anchor**，不能拿 Run2 `last2` 当最终公平对照。

---

# 1. 仓库现状审查与证据表

## 1.1 当前 main 代码事实

本次实际读取了：

```text
README.md

models/changedetection/models/
  STRRepNet.py
  tar.py
  reparam.py
  dcr_decoder.py
  Mamba_backbone.py

models/changedetection/script/
  train.py
  smoke_test.py
  test_reparam_equivalence.py

models/changedetection/datasets/
  make_data_loader.py

models/changedetection/configs/vssm1/
  vssm_tiny_224_0229flex.yaml

analyse/
  extract_metrics_to_excel.py
  models_to_txt.py

docs/temporary/
  STR-RepNet_Run5_回退Run2与BOTR最小消融实验方案.md
  过去的想法/STR-RepNet_下一步实验方向_代码审查与文献建议.md
  过去的想法/STR-RepNet_结构重参数化_2024-2026文献调研与创新空白.md
  models_and_metrics_TAR-DCR_Run5.txt

docs/参考文献/
  文献索引.md

train_scripts/TAR-DCR/Run2/
train_scripts/TAR-DCR/Run5/
```

### 当前主路径

```text
A ─┐
   ├─ Shared VMamba-Tiny ── P1,P2,P3,P4
B ─┘
   └─ Shared VMamba-Tiny ── Q1,Q2,Q3,Q4

每尺度：
(Pi,Qi)
   ↓
TemporalRep1x1
  train: concat + sum + signed-diff (+ optional BOTR)
  deploy: single 1×1
   ↓ SiLU
RepLocalBlock
  DW structural rep
  PW structural rep
   ↓
ti

t4
 ↓ up
fuse3(t3,u4) -> block3
 ↓ up
fuse2(t2,u3) -> block2
 ↓ up
fuse1(t1,u2) -> block1
 ↓
refine
 ↓
head 1×1: 160 -> 2
 ↓
bilinear to 256×256
```

### 关键代码事实

- `TemporalRep1x1` 已经是每分支独立 BN；
- sum / signed-diff 已经可吸收到 concat 1×1；
- BOTR reverse-concat 也能通过通道半区置换吸收；
- `RepDW3`：`3×3 + 1×3 + 3×1 + αI` -> 单 DW3×3；
- `RepPW1x1`：main + low-rank serial + diag + αI -> 单 PW1×1；
- `RepPairFuse1x1`：concat + sum + diff + `αL` -> 单 1×1；
- 所有 kernel/bias 组合已改为 FP64，中途不 cast；
- 最终写入 deploy Conv 时一次性转 FP32；
- `switch_to_deploy()` 只处理 TAR / DCR，不改变 encoder；
- final test 是在 deploy graph 上跑；
- `extract_metrics_to_excel.py` 已严格只读取最后一个完整 `TEST RESULTS` 区块。

这说明目前代码基础已经适合继续做**真正可证明折叠的结构设计**，不需要再重构框架。

---

## 1.2 正式指标证据

`models_and_metrics_TAR-DCR_Run5.txt` 中，Run2 `full_last2` 的最后 TEST 结果为：

| Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| CDD | 0.9849 | 0.9835 | 0.9961 | 0.9842 | 0.9689 | 0.9820 |
| LEVIR | 0.9032 | 0.9260 | 0.9914 | 0.9144 | 0.8424 | 0.9099 |
| SYSU | 0.7956 | 0.8775 | 0.9256 | 0.8345 | 0.7160 | 0.7866 |
| WHU | 0.9435 | 0.9595 | 0.9962 | 0.9514 | 0.9074 | 0.9494 |

部署复杂度统一：

```text
Deploy Params = 28.829 M
Deploy FLOPs = 12.6062 G
```

HAM-CD baseline：

```text
CDD   = 0.9879
LEVIR = 0.9211
WHU   = 0.9500
SYSU  = 0.8299
```

因此：

```text
HAM-CD Macro F1 = 92.2225%
Run2    Macro F1 = 92.1125%
Macro gap         = -0.1100 pp
```

而部署复杂度相对 HAM-CD：

```text
Params: 36.081M -> 28.829M，约 -20.10%
FLOPs : 16.2593G -> 12.6062G，约 -22.47%
```

这很重要：**当前不是一个“整体性能还差很多”的网络。**

真正主要待追的是：

```text
LEVIR: -0.67 pp
CDD  : -0.37 pp
```

而：

```text
WHU : +0.14 pp
SYSU: +0.46 pp
```

已经超过 baseline。

因此不应该再大改整体架构；最合理的是做针对性的、可证伪的 decoder 信息保真改造。

---

## 1.3 Run5 后已经被证伪的假设

### A. spatial boundary prior

已做两层：

```text
operator level: Sobel Edge-Basis
supervision level: IBAS
```

均未达到预注册判据。

**决策：停止边界方向。**

### B. temporal order bias

```text
C1 swap-only = 0.9143
M1 BOTR      = 0.9144
```

二者只差约 0.01pp。

**决策：停止 BOTR，不再做 gate / 更多 reverse branches。**

### C. encoder adaptation 是不是主瓶颈

```text
Run2 last2 = 0.9144
Run5 full  = 0.9168
ΔF1        = +0.24pp
```

说明 early encoder adaptation 有帮助，但没有达到你上一轮预注册的 +0.30pp“主要瓶颈”线。

**决策：full encoder 可以作为训练协议候选，但不是主创新。**

---

# 2. 候选方向比较

| 候选 | 部署开销 | 与已有证据关系 | 结构重参数化创新性 | 主要风险 | 建议 |
|---|---:|---|---|---|---|
| **A. NSCR-Fuse：原生尺度可交换重参数化融合** | **+0** | 直接承接 Run5 失败预案；针对 high-res/detail 与 cross-scale fusion | **高**：跨 bilinear 边界的 affine-commutative exact folding | 可能只是优化参数化，实际增益不足 | **主推 Run6** |
| B. PBRU：Phase-Basis Reparameterized Upsampling | 需预算回收 | 直接解决最终 1/4 -> 1× 输出的空间自由度不足 | 中高 | 必须改 deploy head，预算耦合、归因难 | **NSCR 失败且尺度诊断阳性后再做** |
| C. Budget-neutral 非均匀 decoder channel allocation | 总预算可持平 | 对高分辨率 stage 多给通道、coarse 少给通道 | 中低 | 容易像架构/宽度调参，论文贡献弱 | **作为对照或后备，不作为首个 Run6 主创新** |

---

# 3. LEVIR：先做零部署成本诊断，而不是先训练新模型

在写 Run6 模型代码之前，建议先建立一个 `analyse/levir_error_profile.py`。

它不改模型，不训练，只读取：

```text
GT
Run2 full_last2 checkpoint
Run5 D0_full_encoder checkpoint
可选：HAM-CD baseline prediction
```

目标不是“找一个漂亮故事”，而是回答：

> **LEVIR 剩余错误到底是不是小目标/尺度保真问题？如果是，错误发生在 encoder 特征层，还是 decoder fusion 之后？**

---

## 3.1 诊断 1：按 GT 连通域尺寸分层的 FN 分析 —— 优先级最高

### 不要用人工固定阈值

不要直接写：

```text
small < 64 pixels
medium 64~256
large > 256
```

这样很容易变成事后选阈值。

应该：

1. 用 **LEVIR train GT**；
2. 对所有 change connected components 做面积统计；
3. 用训练集分位数定义尺寸 bin，例如：
   - small：≤ Q33；
   - medium：Q33~Q67；
   - large：> Q67。
4. 阈值确定后冻结；
5. 再去 test 上统计。

### 输出至少包括

```text
bin
GT_component_count
GT_positive_pixels
TP_pixels
FN_pixels
pixel_recall
completely_missed_components
missed_component_rate
```

“完全漏检 component”的定义建议固定为：

```text
GT component 与预测正类区域交集 = 0
```

不要使用可调 IoU 阈值。

### 最关键比较

```text
Run2 full_last2
vs
Run5 D0 full_encoder
```

如果 full encoder 主要改善 large/medium，而 small FN 基本不动，则更支持 decoder/detail bottleneck。

如果 full encoder 对 small component 也明显改善，则说明特征适配仍占一部分，需要谨慎解释 NSCR。

---

## 3.2 诊断 2：单 checkpoint 测试尺度敏感性

这只能作为**离线诊断**，不能成为最终方法。

建议对同一个 Run2 checkpoint 依次：

```text
192
256
320
384
```

流程：

```text
A/B resize -> 模型 -> logits resize 回 256 -> argmax -> metric
```

不要：

```text
multi-scale average
TTA ensemble
挑最优尺度作为最终结果
```

只看模型对分辨率的敏感性。

### 支持“尺度问题”的证据

可预先定义一个诊断性阈值：

```text
若 320 或 384 相对 256：
Recall +>= 0.20pp
且 Precision 下降 <= 0.20pp
```

则说明高分辨率输入确实释放了当前模型没利用好的小目标/细节信息。

但即便这样，最终方法也仍必须回到 256 输入和 12.6062G 部署预算。

HAM-CD 自身的输入尺度消融并不是“越大越好”，因此这一步只用于定位，不应把“大分辨率推理”当解法。

---

## 3.3 诊断 3：stage-wise change discriminability

这是决定 NSCR 是否合理的最有科研价值诊断。

对：

```text
P1/Q1
P2/Q2
P3/Q3
P4/Q4

TAR 输出 t1/t2/t3/t4

以及：
fuse1 input/output
block1 output
refine output
```

挂 hook。

### 最简单、不额外训练的判别量

对 encoder 时相特征：

\[
D_i(x,y)=\|P_i(x,y)-Q_i(x,y)\|_2
\]

也可再补：

\[
D_i^{cos}=1-\cos(P_i,Q_i)
\]

把 GT 用 nearest downsample 到同尺度，然后统计：

```text
changed pixels 的 D 均值
unchanged pixels 的 D 均值
Fisher separation
AUROC(D, GT)
```

推荐 Fisher separation：

\[
J_i =
\frac{(\mu_c-\mu_u)^2}
{\sigma_c^2+\sigma_u^2+\epsilon}
\]

其中：

- `c` = changed；
- `u` = unchanged。

### 对 decoder feature

对于 `t1 -> fuse1 -> block1 -> refine`，可以用一个**无训练 probe**：

```text
每像素 channel L2 norm
或
PCA 第一主成分 / feature energy
```

更严谨的版本是：

- 从 train feature 抽样；
- 只训练一个很小的 logistic linear probe；
- 固定模型；
- 再在 test 评估 AUROC；
- probe 只用于诊断，不进入论文最终方法。

### 关键解释

如果看到：

```text
stage1 / t1 的 changed-vs-unchanged separation 较好
但经过 fuse1 / block1 后显著下降
```

则非常支持：

> **问题发生在 cross-scale decoder fusion，而不是 backbone 没看到小目标。**

这正是 NSCR 的目标位置。

---

## 3.4 诊断 4：A/B swap sensitivity 只做 sanity check

Run5 已经基本把时序顺序问题排除。

因此不建议再把它作为 Run6 主诊断。

若已有代码能很便宜地输出：

```text
pred(A,B)
pred(B,A)
pixel disagreement
F1_AB
F1_BA
```

可以留作附录/台账，但不要继续围绕它设计新结构。

---

## 3.5 诊断 5：cross-scale 路径激活/梯度比例

只需要在 Run2 模型上做短时 hook，记录：

```text
fuse1:
  RMS(L)
  RMS(U(H))
  grad_RMS(L)
  grad_RMS(U(H))

fuse2:
  同上
```

最好按训练早/中/晚三个 checkpoint 看。

目的：

- 如果 high-res lateral 一直明显弱于 coarse semantic 路径；
- 或 gradient 被强烈压制；

则 NSCR 的“独立原生尺度归一化 + 梯度路径”动机更扎实。

---

# 4. 主方案：NSCR-Fuse

## 4.1 现有融合

当前 `RepPairFuse1x1` 在部署时等价为：

\[
Y_0 =
W_L * L
+
W_H * \mathcal{U}(H)
+b
\]

其中：

- \(L\)：当前高分辨率 lateral feature；
- \(H\)：上一 decoder stage 的低分辨率 feature；
- \(\mathcal U\)：bilinear interpolation；
- \(W_L,W_H\)：原 deploy 1×1 的两个输入半区。

当前 train graph 内部虽然已有：

```text
concat
sum
diff
αL
```

但前提是 `L` 与 `U(H)` 已经先到了同一分辨率。

也就是说：

> 当前 DCR 的训练过参数化主要发生在**对齐之后**。

NSCR 要增加的是：

> **对齐之前、各自在原生尺度上的独立线性统计路径。**

---

## 4.2 训练图

建议只加在：

```text
fuse1
fuse2
```

暂时不加 `fuse3`。

记两个训练分支为：

\[
R_L(L)=BN_L(L)
\]

\[
R_H(H)=\mathcal U(BN_H(H))
\]

训练期输出：

\[
Y_{train}
=
F_{core}^{train}(L,\mathcal U(H))
+
R_L(L)
+
R_H(H)
\]

然后再经过现有：

```text
SiLU -> RepLocalBlock
```

### 为什么只做 fuse1 + fuse2

因为当前目标是：

- LEVIR 小建筑；
- high-resolution information preservation；
- 不是把“更多 branch”当目标本身。

`fuse3` 位于更低分辨率，更偏语义层。

如果第一版就三层全开：

- 失败时无法判断 high-resolution 分支是否有效；
- 成功时也难说明是哪里贡献；
- 更像无差别 branch stacking。

因此主方案预注册为：

```text
scope = high2
```

即：

```text
fuse1 + fuse2
```

---

# 5. NSCR 的严格可折叠数学

## 5.1 BN 在 eval 时是逐通道仿射

对任一通道：

\[
BN(x)=
\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

可写成：

\[
BN(x)=Ax+c
\]

其中：

\[
A=
\operatorname{diag}
\left(
\frac{\gamma}
{\sqrt{\sigma^2+\epsilon}}
\right)
\]

\[
c=
\beta-A\mu
\]

对 `1×1 conv` 来说，\(A\) 就是一个对角 kernel。

---

## 5.2 bilinear interpolation 与逐通道线性映射可交换

bilinear interpolation 对空间位置是固定线性组合，并且不混合 channel。

因此：

\[
\mathcal U(AH)=A\mathcal U(H)
\]

又因为 bilinear 对常数图保持常数：

\[
\mathcal U(c)=c
\]

所以：

\[
\mathcal U(BN_H(H))
=
A_H\mathcal U(H)+c_H
\]

这是本方案最关键的数学点。

注意：

> **这里不是把 bilinear interpolation 本身“折叠进卷积”。**

bilinear 仍然保留在 deploy graph。

我们只是把**位于 bilinear 之前的逐通道 affine branch**，等价地移动到 bilinear 之后，再吸收到原有 fuse 1×1 的 `H` 半区权重。

这与“强行把 resize+conv 合成一个 conv”完全不同。

---

## 5.3 与现有核心融合合并

现有核心部署算子：

\[
Y_0 =
W_L L + W_H\mathcal U(H)+b
\]

加上 NSCR：

\[
Y =
Y_0
+
A_LL+c_L
+
A_H\mathcal U(H)+c_H
\]

整理：

\[
Y =
(W_L+A_L)L
+
(W_H+A_H)\mathcal U(H)
+
(b+c_L+c_H)
\]

因此定义：

\[
W'_L=W_L+A_L
\]

\[
W'_H=W_H+A_H
\]

\[
b'=b+c_L+c_H
\]

最终：

\[
Y =
W'_L L
+
W'_H\mathcal U(H)
+b'
\]

部署仍然只是：

```python
U_H = F.interpolate(H, size=L.shape[-2:], mode="bilinear", align_corners=False)
Y = fused_conv(torch.cat([L, U_H], dim=1))
```

其中 `fused_conv` 仍然是：

```python
nn.Conv2d(2*C, C, 1, bias=True)
```

---

# 6. 部署复杂度

若通道数：

```text
C = 160
```

每个 NSCR fuse 训练期新增两个 BN：

```text
BN_L: gamma + beta = 2C
BN_H: gamma + beta = 2C
```

所以每 fuse 新增训练参数：

```text
4C = 640
```

`fuse1 + fuse2`：

```text
1280 parameters = 0.00128M
```

部署前全部吸收。

因此理论目标：

```text
Deploy Params:
28.829M -> 28.829M

Deploy FLOPs:
12.6062G -> 12.6062G
```

这两个值必须用实际 `fvcore` 再验一次，不能只根据理论宣称。

---

# 7. 零初始化策略

这是本方案必须做对的地方。

对两个新增 BN：

```python
nn.init.zeros_(bn.weight)  # gamma = 0
nn.init.zeros_(bn.bias)    # beta  = 0
```

于是初始化：

\[
BN_L(L)=0,\qquad BN_H(H)=0
\]

因此：

```text
NSCR 模型初始化主预测 = 原 Run2 初始化主预测
```

只新增一个“从零开始学习”的优化路径。

这样比：

```text
gamma=1
```

更干净，因为后者一开始会把 `L + U(H)` 又加一遍，初始函数已经改变。

### 一个需要记录的训练事实

gamma=0 时：

- 第一批反传仍能给 `gamma / beta` 梯度；
- BN 前输入本身的梯度一开始可能为 0；
- gamma 学离零后，该支路逐步参与。

这恰好符合“训练期渐进展开”的需求。

不要再额外加可学习 scalar gate，否则会引入第二个机制变量。

---

# 8. 与现有结构重参数化工作的实质区别

## 8.1 vs RepVGG / DBB

传统 RepVGG/DBB 主要是：

```text
同尺度
不同卷积/BN分支
-> kernel padding/composition
-> 单卷积
```

NSCR 的重点不是再造一个同尺度 RepConv，而是：

```text
不同 native resolution
  ↓
逐通道 affine
  ↓
利用 bilinear 的线性/可交换性
  ↓
跨 resize 边界搬移
  ↓
吸收到已有 cross-scale projection
```

---

## 8.2 vs UniRepLKNet

UniRepLKNet 的 Dilated Reparam 重点是：

```text
多 dilation 小核
-> 空间 kernel embedding
-> 单个大核
```

NSCR 不增加部署 kernel size，也不做 large-kernel rep。

核心是：

> **cross-scale affine commutation + existing projection absorption**

---

## 8.3 vs ASR

ASR 解决的是：

- 常规 attention input-dependent / multiplicative；
- 无法直接静态折叠；
- 通过 attention-alike reparameterization 实现 inference-cost-free enhancement。

NSCR 不模拟 attention，也不做输入动态 gate。

它始终保持：

```text
linear / affine / deterministic interpolation
```

因此可以给出直接解析折叠式。

---

## 8.4 vs CD-RLKNet / LKMamba-CD

已有变化检测中的 reparameterization 近邻已经包括：

```text
large-kernel reparam
Mamba + reparam large kernel
difference fusion + reparam block
```

所以本课题不能再把：

> “把一个 RepDW / RepLK 放到 CD”

作为创新。

NSCR 的区分点应写成：

> **针对变化检测 decoder 的 coarse semantic / fine detail 跨尺度交互，将训练期原生尺度独立统计分支通过插值线性可交换性精确吸收到既有部署融合算子，保持部署拓扑与预算不变。**

---

# 9. 可证伪假设

主方案不是“有理由就一定有效”，而是预先给出失败条件。

## H1：高分辨率 lateral detail 的独立训练路径有价值

若成立，预期：

```text
LEVIR Recall 上升
F1 / IoU 同时上升
Precision 不发生明显牺牲
```

若：

```text
Recall 上升
但 Precision 明显掉
```

则它很可能只是让 decoder 更激进，不是更好的 detail preservation。

---

## H2：价值主要来自 high-resolution cross-scale stages

因此首版仅：

```text
fuse1 + fuse2
```

如果主实验失败，不允许再立刻：

```text
fuse3 也开
BN 换 GN
再加 gate
gamma 改 1
branch weight sweep
```

去“救”机制。

---

## H3：full encoder 与 NSCR 是相对正交的

如果 NSCR 在 `last2` 成立，再跑：

```text
NSCR + full_encoder
```

如果还能从 0.9168 基础继续抬升，说明结构改造与 encoder adaptation 可以叠加。

如果加 full 后反而不增，则不能把两者拼成“最终最佳配置”强行包装。

---

# 10. 逐文件修改清单

## P0：`models/changedetection/models/reparam.py`

新增一个最小 primitive，例如：

```python
class NSCRPairFuse1x1(nn.Module):
    ...
```

不要重写整个 DCR。

### 需要的 helper

建议新增：

```python
def fold_identity_bn(bn, channels):
    ...
```

返回 FP64：

```text
A: C×C×1×1 diagonal kernel
c: C bias
```

### train graph

类内部持有：

```text
core = RepPairFuse1x1(...)
bn_l
bn_h
```

forward：

```python
up_h = F.interpolate(H, size=L.shape[-2:], mode="bilinear", align_corners=False)

if deploy:
    return fused(torch.cat([L, up_h], dim=1))

y = core(L, up_h)
y = y + bn_l(L)
y = y + F.interpolate(
    bn_h(H),
    size=L.shape[-2:],
    mode="bilinear",
    align_corners=False,
)
return y
```

### deploy fold

**重要：不要先调用 `core.switch_to_deploy()`。**

因为那样 core 会先 cast FP32，然后你再加 NSCR，破坏“组合全程 FP64、最后一次 cast”的 P0 约定。

正确顺序：

```text
1. core.get_equivalent_kernel_bias() -> FP64
2. fold bn_l -> FP64
3. fold bn_h -> FP64
4. 分割 core 的 L/H kernel
5. 加 diagonal A_L/A_H
6. bias 全部相加
7. cat
8. 最后一次 cast FP32
9. 创建 deploy Conv
10. 删除 core / bn_l / bn_h
```

---

## P1：`models/changedetection/models/dcr_decoder.py`

增加：

```python
use_nscr=False
nscr_scope="high2"
```

推荐：

```text
fuse1 = NSCRPairFuse1x1
fuse2 = NSCRPairFuse1x1
fuse3 = 原 RepPairFuse1x1
```

但为了使 NSCR 能看到 native-resolution `H`，需要把对应 upsample 移入 wrapper。

当前：

```python
u3 = interpolate(d3)
d2 = block2(act(fuse2(t2, u3)))
```

改成逻辑上：

```python
d2 = block2(act(fuse2(t2, d3)))
```

其中 `fuse2` 自己做：

```python
interpolate(d3 -> t2 size)
```

deploy 时仍然执行同一个 bilinear。

这只是代码归属变化，不是部署算子变化。

---

## P1：`models/changedetection/models/STRRepNet.py`

新增参数：

```python
use_nscr=False
nscr_scope="high2"
```

传给：

```python
DCRDecoder(...)
```

TAR 不修改。

BOTR 保留开关但 Run6 全部：

```text
use_botr = 0
```

---

## P1：`models/changedetection/script/train.py`

新增：

```text
--use_nscr 0|1
--nscr_scope high2
```

TEST RESULTS 内必须写：

```text
[NSCR] 1
[NSCR-SCOPE] high2
```

保持：

```text
[DEPLOY-PARAMS]
[DEPLOY-FLOPS]
[REPARAM-MAX-ABS-ERROR]
[REPARAM-ARGMAX-DISAGREE]
```

不改：

```text
CE
Lovasz weight
optimizer
epoch
batch
seed
```

---

## P1：`models/changedetection/script/test_reparam_equivalence.py`

新增一个**不同空间尺寸**的单元测试：

```text
L: [2,160,32,32]
H: [2,160,16,16]
```

测试：

```text
NSCRPairFuse1x1 train-eval graph
vs
deploy graph
```

还要单独测试数学交换：

\[
U(AH+c)
\quad vs\quad
AU(H)+c
\]

建议：

```text
FP64 CPU:
max_abs < 1e-12

FP32 CUDA:
记录实际误差，不强行声称 <1e-6
```

---

## P1：`models/changedetection/script/smoke_test.py`

新增：

1. `use_nscr=1` 能构建；
2. zero-init 后 NSCR residual output 约为 0；
3. BN gamma 有非零梯度；
4. `switch_to_deploy` 后：
   - 不再存在 `bn_l/bn_h/core` 训练分支；
5. deploy params 与 `use_nscr=0` 完全相等；
6. deploy FLOPs 与 `use_nscr=0` 完全相等；
7. argmax disagreement = 0。

---

## P2：`analyse/levir_error_profile.py`

新建，不改训练。

建议支持：

```bash
--checkpoint
--dataset_root
--train_list
--test_list
--output_dir
--feature_hooks
--test_scales 192 256 320 384
```

输出：

```text
component_bins.json
component_metrics.csv
scale_sensitivity.csv
feature_discriminability.csv
```

---

## P2：`analyse/extract_metrics_to_excel.py`

建议顺手增加：

```text
NSCR
NSCRScope
EncoderTrain
```

否则 Run6 后 Excel 无法直接筛结构组。

---

## P2：当前顺手修两个日志/CLI 小问题

这两项不影响方法，但建议 Run6 前处理：

### `train.py`

当前 header 仍写：

```text
STR-RepNet Clean TAR-DCR | train_scripts/TAR-DCR/Run1
```

parser description 仍有：

```text
HAM-CD baseline training (Run1)
```

已经过时。

### `smoke_test.py`

`STRRepNet` 支持：

```text
plain / tar / dcr / full
```

但 smoke parser 当前 choices 没有 `dcr`。

Run6 顺手补齐。

---

# 11. Run6 最小实验组

## 11.1 第一阶段：只跑 LEVIR

### A0：锚点，不重跑

```text
Run2/full_last2
encoder_train=last2
use_nscr=0
F1=0.9144
IoU=0.8424
Precision=0.9260
Recall=0.9032
```

**复用既有正式 TEST RESULTS。**

不要为了 Run6 再重跑 A0，因为：

- 同 seed；
- 同代码主路径；
- 同预算；
- 重跑只会增加一次随机样本，反而让锚点定义变复杂。

但必须在 Run6 README 明确写：

```text
A0 is a historical Run2 anchor and was not re-run.
```

---

## 11.2 M1：唯一主实验

```text
M1_NSCR_high2
```

配置：

```text
rep_mode=full
encoder_train=last2
encoder_lr_ratio=0.1
use_residual=1
use_botr=0
temporal_swap_prob=0.0

use_nscr=1
nscr_scope=high2

seed=2333
epochs=300
batch=16
lr=1e-4
weight_decay=5e-4
lovasz_weight=2.0
crop=256
```

**只改变 NSCR。**

---

## 11.3 M1 通过后，再补最小归因

不是 M1 之前就并行跑一堆。

### C1：Lateral only

```text
C1_NSCR_Lonly
```

训练期：

\[
F_{core}+BN_L(L)
\]

### C2：Semantic native-scale only

```text
C2_NSCR_Honly
```

训练期：

\[
F_{core}+U(BN_H(H))
\]

目的：

```text
证明收益是高分辨率 lateral
还是 coarse semantic 的独立尺度统计
还是二者互补
```

不需要：

```text
fuse1-only
fuse2-only
fuse3-only
gamma sweep
BN momentum sweep
```

除非论文最终真要做模块分析，否则首轮不要扩。

---

# 12. Run6 数据集启动顺序

建议：

```text
1. LEVIR
2. WHU
3. SYSU
4. CDD
```

理由：

### LEVIR

- 最大剩余 gap；
- 主诊断对象；
- 建筑小目标；
- 判断机制是否值得继续。

### WHU

- 同样偏 building change；
- Run2 已 +0.14pp；
- 可以验证 NSCR 是否能改善 LEVIR 而不破坏已经不错的 WHU。

### SYSU

- 场景更复杂；
- Run2 已 +0.46pp；
- 检验跨场景泛化，而不是只对建筑数据有效。

### CDD

- 已接近饱和；
- baseline 0.9879；
- 最容易出现 ceiling effect；
- 放最后最省实验成本。

---

# 13. checkpoint / log 目录

建议严格延续现有结构：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/
└── TAR-DCR/
    └── Run6/
        ├── M1_NSCR_high2/
        │   ├── LEVIR-CD-256/
        │   ├── WHU-CD-256/
        │   ├── SYSU-CD-256/
        │   └── CDD-CD-256/
        ├── C1_NSCR_Lonly/
        │   └── LEVIR-CD-256/
        ├── C2_NSCR_Honly/
        │   └── LEVIR-CD-256/
        └── M2_NSCR_fullenc/
            └── LEVIR-CD-256/
```

日志：

```text
/home/yqwang/outputs/STR-RepNet/
└── TAR-DCR/
    └── Run6/
        └── ...
```

脚本：

```text
train_scripts/TAR-DCR/Run6/
```

诊断输出：

```text
/home/yqwang/outputs/STR-RepNet/
└── diagnostics/
    └── Run6_LEVIR/
```

---

# 14. checkpoint 恢复规则

正式 Run6：

> **全部从同一个 VMamba pretrained 初始化开始。**

不要：

```text
Run2 best -> M1 继续训
D0 full -> M1 继续训
C1 -> C2 resume
```

这样会混入额外训练预算。

允许：

```text
M1_NSCR_high2 自己的 last.pth -> M1 恢复
```

即：

> 同组断点恢复可以，跨组 checkpoint 迁移不可以。

---

# 15. `encoder_train=full` 是否纳入正式协议

## 当前答案：第一阶段不纳入

Run6 主结论必须先在：

```text
last2
```

下成立。

### 为什么

Run5 D0 的结果：

```text
0.9144 -> 0.9168
```

已经证明 full fine-tune 是一个小幅正向训练策略。

如果从 M1 一开始就：

```text
NSCR + full
```

即使得到 0.9200，也不知道：

```text
+0.24pp 来自 full
剩余来自 NSCR
还是二者有强交互
```

---

## 15.1 什么时候跑 M2_NSCR_fullenc

只有 M1_NSCR_high2 先 PASS 后。

再跑：

```text
M2_NSCR_fullenc / LEVIR
```

对照：

```text
D0_full_encoder = 0.9168
```

此时可以回答：

> 在 full-encoder training protocol 下，NSCR 是否仍提供额外结构收益？

---

## 15.2 如果论文最终采用 full encoder

必须补：

```text
A0_fullenc / CDD
A0_fullenc / LEVIR
A0_fullenc / WHU
A0_fullenc / SYSU
```

然后所有最终主方法也用 full。

论文表格不要出现：

```text
baseline: last2
method: full
```

却把差值全部归因于结构。

---

# 16. Run6 预注册判据

LEVIR 锚点：

```text
F1        = 0.9144
IoU       = 0.8424
Precision = 0.9260
Recall    = 0.9032
```

---

## 16.1 PASS

建议沿用上一轮相同强度，不人为降低门槛：

```text
F1        >= 0.9175
IoU       >= 0.8475
Precision >= 0.9230
```

三者同时满足。

解释：

- F1 至少 +0.31pp；
- IoU 有同步增益；
- Precision 最多容忍约 -0.30pp；
- 防止靠单纯放宽 change prediction 来换 Recall。

---

## 16.2 WEAK / 中性

```text
0.9159 <= F1 < 0.9175
```

或者：

```text
F1 >= 0.9175
但 IoU / Precision 任一不满足
```

处理：

> 记录为中性，不扩四数据集，不做 stage/scope/gamma sweep。

如果确实需要理解机制，可以补一次 L-only/H-only，但不把它当“救模型”。

---

## 16.3 FAIL

```text
F1 < 0.9159
```

或：

```text
Precision < 0.9220
```

即相对锚点下降 >0.40pp。

即使 Recall 上升也判 FAIL。

处理：

> **停止 NSCR。**

不再：

```text
fuse3 也加
改 BN momentum
改 gamma init
加 sigmoid gate
加 attention
换 loss
```

---

# 17. 四数据集扩展判据

只有 LEVIR PASS 后才扩。

建议仍沿用上一轮全局标准：

```text
Macro F1 >= 92.25%
LEVIR    >= 91.75%
至少 3/4 数据集 ΔF1 >= 0
任何单数据集退化 <= 0.15pp
```

同时增加硬部署条件：

```text
Deploy Params <= 28.829M
Deploy FLOPs <= 12.6062G
Argmax disagreement = 0
```

如果：

```text
LEVIR PASS
但 WHU <-0.15pp
```

不要急着继续 SYSU/CDD。

先判定：

> 机制可能只适用于 LEVIR 特定分布，而不是 building-CD 普适 detail preservation。

这时主论文贡献需要降级。

---

# 18. 部署等价性：P0 必须继续单独建台账

## 18.1 结论

**是，继续作为独立 P0 台账处理。**

但不要再让它绑架方法迭代。

目前代码已经把真正能做到的部分做对了：

```text
kernel/bias algebraic composition: FP64
final deploy assignment: one-time FP32 cast
argmax disagreement: 0
```

whole-model logit 的约 `1e-5`：

```text
并不等于代数折叠公式错误
```

因为 train graph 和 deploy graph 的 FP32 加法/卷积执行顺序不同。

---

## 18.2 建议拆成三级验收

### Level A：代数参数等价

对所有 branch composition：

```text
FP64 kernel/bias error
```

目标建议：

```text
<= 1e-12
```

这是“公式正确”的证据。

---

### Level B：block output

随机 FP32：

```text
train-eval block
vs
deploy block
```

记录：

```text
max abs
mean abs
relative L2
```

当前合理量级：

```text
1e-6 ~ 1e-5
```

不要为了得到漂亮数字再偷偷转 FP64 inference，因为论文最终部署是 FP32。

---

### Level C：whole model

固定 seed、固定输入：

```text
max_abs_error
mean_abs_error
relative_L2
argmax_disagreement
```

最关键仍是：

```text
argmax disagreement = 0
```

最好再在真实 test batch 上抽若干批做：

```text
pixel argmax disagreement
```

而不仅只对一对随机图。

---

# 19. 论文里怎么诚实写 fold equivalence

如果最终 whole-model FP32 仍是 `1e-5`，不要写：

> “deployment output is strictly numerically equivalent with error <1e-6.”

建议写：

> **All foldable training branches are analytically fused into their corresponding deployment operators. The kernel/bias transformation is performed in FP64 before a single FP32 cast. Because the multi-branch and fused FP32 graphs have different floating-point accumulation orders, the observed maximum logit discrepancy is on the order of \(10^{-5}\), while the predicted labels show zero argmax disagreement in our equivalence tests.**

中文可写：

> 所有可折叠训练分支均通过解析参数变换吸收到对应部署算子中。折叠过程使用 FP64 完成参数组合并在最终部署权重写入时一次性转换为 FP32。由于多分支图与融合图的浮点累加顺序不同，FP32 下最大 logit 偏差约为 \(10^{-5}\)，但部署前后预测标签的 argmax 分歧率为 0。

这比硬宣称 `<1e-6` 更可信。

### 对当前“项目硬约束 <1e-6”的处理

内部表格中应明确：

```text
Analytical folding: PASS
FP32 <1e-6 whole-model: NOT MET
Argmax identity: PASS
```

不能把 `<1e-6` 改口说“已经满足”。

但科研判断上，这个问题应独立维护，不建议为了追一个浮点阈值继续改变网络设计。

---

# 20. NSCR 专属等价性测试

NSCR 需要比旧模块多一项：

```text
affine-interpolation commutation
```

## FP64 test

构造：

```text
H random float64
A random diagonal
c random
```

比较：

```python
lhs = interpolate(A*H + c)
rhs = A*interpolate(H) + c
```

目标：

```text
max_abs <= 1e-12
```

再测试完整：

```text
core train-eval + NSCR branches
vs
single deploy fuse
```

---

# 21. Smoke / dry run / deploy 验收

## 21.1 随机 smoke

至少检查：

```text
[1] build
[2] forward shape = (B,2,256,256)
[3] NSCR gamma gradient != None
[4] zero-init residual ≈ 0
[5] switch_to_deploy
[6] no NSCR train branch remains
[7] deploy params exact equal to A0
[8] deploy FLOPs exact equal to A0
[9] argmax disagreement = 0
```

---

## 21.2 真实数据 dry run

不要直接开始 300 epoch。

建议增加一个最小真实数据检查：

```text
读取 LEVIR train list
2~5 个 batch
真实 forward
CE+Lovasz
backward
optimizer.step
```

检查：

```text
loss finite
NSCR gamma 从 0 离开
encoder stage1/2 无 grad
encoder stage3/4 有 grad
label unique in {0,1}
A/B/label shape 正确
```

然后再删掉 dry-run checkpoint，**但不要覆盖正式目录**。

dry run 建议独立：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/_dryrun/Run6_NSCR/
```

---

# 22. Run6 日志新增字段

除了已有字段，建议写：

```text
[NSCR] 1
[NSCR-SCOPE] high2
```

最终还可以记录：

```text
[NSCR-GAMMA-NORM]
fuse1/l = ...
fuse1/h = ...
fuse2/l = ...
fuse2/h = ...
```

作用不是拿 norm 当成果，而是失败时区分：

1. branch 根本没有学起来；
2. branch 学起来但无效；
3. lateral 有效、semantic 无效；
4. 反之。

不要把 branch norm 当 metric 选 checkpoint。

---

# 23. 为什么当前不主推 PBRU

第二候选：

> **Phase-Basis Reparameterized Upsampling**

动机非常直接：

当前：

```text
d1 at ~1/4
-> head 160->2
-> bilinear x4
```

最终 full-resolution logits 的空间自由度主要受低分辨率 2-channel map 限制。

可以考虑：

```text
1×1: 160 -> 2*r^2
PixelShuffle(r=4)
```

训练时再用 coarse / phase basis 分支，部署折成单个 projection。

### 但它为什么不是 Run6 第一枪

因为：

- deploy head 参数和 FLOPs 会增加；
- 必须从 decoder 其他位置回收预算；
- 需要同时改 channel width；
- “高分辨率 head”与“结构重参数化 branch”两个变量耦合；
- 一次失败信息量较低；
- 一次成功也难归因。

如果 NSCR FAIL、而尺度诊断非常明确地显示：

```text
higher test scale -> small-object Recall 明显提升
```

再启动 PBRU 更合理。

---

# 24. 如果将来做 PBRU，预算必须机器搜索，不手算拍一个 D

不要预先写：

```text
D=158
D=156
```

建议写脚本：

```text
for D in candidate_dims:
    build deploy model
    measure params
    measure fvcore FLOPs

select largest D such that:
Params <= 28.829M
FLOPs <= 12.6062G
```

也就是说：

> 高分辨率预测头增加的预算，必须由 decoder channel reduction 精确回收。

这样才能叫：

> budget-neutral

而不是“大致差不多”。

---

# 25. 为什么非均匀 channel allocation 只做后备

一个可行方向是：

```text
coarse stage 更窄
fine stage 更宽
```

例如：

```text
stage4 < stage3 < stage2 <= stage1
```

总 Params/FLOPs 保持 <= 当前。

它可能确实改善 LEVIR。

但论文风险是：

- 更像 width redistribution；
- 需要搜索多个 channel；
- 很容易被审稿人归类为 architecture tuning；
- 和“结构重参数化”主线联系弱。

所以更适合作为：

```text
PBRU 的预算回收方法
或
NSCR 之后的效率对照
```

而不是 Run6 主创新。

---

# 26. HAM-CD 给当前方向的启示应怎样用

HAM-CD 的 decoder 不是简单 top-down concat。

其 ICSF 显式做：

```text
cross-stage feature fusion
channel-wise enhancement
spatial-wise enhancement
```

而且论文的 LEVIR 消融显示，ICSF 的 channel/spatial 子组件分别有独立贡献，组合后继续提升。

对 STR-RepNet 的正确借鉴不是：

```text
把 ICSF / SE / attention 搬回来
```

因为这会增加部署开销、破坏你的轻量主线。

更好的问题是：

> HAM-CD 为什么需要独立 channel/spatial cross-stage enhancement，而 STR-RepNet 能不能用**训练期可折叠的静态线性路径**获得一部分优化收益？

NSCR 正是在回答这个问题。

即：

```text
HAM-CD:
dynamic/input-dependent cross-stage enhancement
-> expensive but expressive

STR-RepNet NSCR:
train-only native-scale independent affine paths
-> exact fold into existing fuse
-> deploy +0
```

这个差异比“再加一个 RepConv”更像一条清楚的论文主线。

---

# 27. 文献边界与 novelty 风险

当前仓库已经覆盖的高相关文献中：

```text
RepVGG / DBB
UniRepLKNet
RepViT
ASR
CD-RLKNet
LKMamba-CD
ChangeMamba
DMFANet
ST-Mamba
DEIF-Mamba
HAM-CD
```

已经可以明确排除：

> “RSCD 首次结构重参数化”

> “Mamba-CD 首次 Rep large kernel”

> “首次差分融合 + Rep”

这些都不安全。

截至本次仓库文献索引与补充官方来源核验，尚未看到一个与你这里**完全同构**的高水平已发表机制：

> 在 progressive decoder 中，让 coarse/fine 两个分支分别在自己的 native resolution 上训练独立 channel-affine 路径，再利用 linear interpolation commutation 将它们严格吸收回已有 cross-scale 1×1 fusion，部署不增加任何 operator/parameter/FLOP。

但论文最终不能写：

> “the first in the world”

更安全的表述：

> “To the best of our knowledge, existing structural re-parameterization methods mainly focus on same-resolution convolutional branches or spatial kernel transformations. We instead exploit the commutativity between channel-wise affine transformations and linear interpolation to construct train-time native-scale branches for cross-scale change decoding, which are analytically absorbed into the original deployment fusion operator.”

正式投稿前仍建议再做一次专门 novelty search：

```text
structural re-parameterization
cross-scale fusion
interpolation commutation
decoder reparameterization
multi-resolution reparameterization
```

---

# 28. Run6 成功以后，论文主方法应该怎样组织

如果 M1 成立，我建议最终方法只保留三层逻辑：

## Contribution 1：TAR

```text
变化检测专属 temporal algebraic reparameterization
concat + sum + signed-diff
-> single temporal projection
```

## Contribution 2：DCR

```text
decoder-wide structural reparameterization
DW / PW / pair-fusion
-> simple deployment graph
```

## Contribution 3：NSCR-Fuse

```text
native-scale cross-scale training branches
+ interpolation commutation
-> absorbed into existing fusion operator
```

这样三个贡献是同一条数学主线：

> **temporal topology -> local decoder operator -> cross-scale decoder topology**

而不是：

```text
TAR
+ Edge
+ boundary loss
+ attention
+ KD
```

这种模块堆叠。

这会让毕业论文和投稿论文都更完整。

---

# 29. 如果 Run6 失败，预设决策

## 情况 A：NSCR FAIL，且 small-component FN 明显集中、尺度诊断阳性

结论：

> 问题确实偏高分辨率输出自由度，而不是 cross-scale affine optimization。

下一步转：

> **PBRU + exact budget recovery**

不要继续修 NSCR。

---

## 情况 B：NSCR FAIL，stage1/t1 discriminability 本身就低

结论：

> 浅层 encoder / temporal feature 本身没有形成足够变化判别性，decoder fusion 并不是主要丢失点。

这时：

- full encoder 已只有 +0.24pp；
- 不建议无限 fine-tune encoder；
- 可以重新检查 TAR stage1 的 change separability；
- 但 BOTR/swap 已停止，不能再回头叠时序顺序分支。

更可能应该进入：

> budget-neutral feature allocation / shallow-detail capacity

而不是边界、loss、BOTR。

---

## 情况 C：NSCR WEAK

严格按预注册：

```text
不扩四数据集
不 sweep
```

如果 branch stats 显示：

```text
L branch 明显学习
H branch 近 0
```

可补 L-only/H-only 作为机制解释，但不要用其结果继续搜超参。

---

## 情况 D：LEVIR PASS，WHU FAIL

说明：

> NSCR 可能是 LEVIR-specific optimization，不足以支持“building change 普适提升”。

如果退化 >0.15pp：

- 停止全四集扩展；
- 不把它定为论文主创新；
- 回到诊断。

---

## 情况 E：LEVIR / WHU / SYSU 均稳定，CDD 轻微 ceiling-effect

如果：

```text
前三个稳定正向
CDD -0.05~-0.10pp
Macro 仍明显提升
```

可继续讨论，但仍按你预注册的：

```text
任一数据集退化 <=0.15pp
至少 3/4 非负
```

执行。

不能把 CDD 的下降隐去。

---

# 30. 立即执行顺序

## Step 1：冻结当前锚点

记录：

```text
git rev-parse HEAD
# ab260efe63c24e76e279e437893f0b1b209e22a4
```

新建：

```text
run6-nscr
```

不要 reset Run3/4/5 历史。

---

## Step 2：先写诊断脚本，不改模型

实现：

```text
analyse/levir_error_profile.py
```

先跑：

```text
Run2 full_last2
Run5 D0_full_encoder
```

拿到：

```text
component size FN
scale sensitivity
stage-wise discriminability
cross-scale RMS/grad
```

---

## Step 3：如果诊断没有明显反证，再实现 NSCR

只改：

```text
reparam.py
dcr_decoder.py
STRRepNet.py
train.py
smoke_test.py
test_reparam_equivalence.py
```

不动：

```text
Mamba_backbone.py
tar.py
loss
dataset augmentation
```

---

## Step 4：先做解析折叠单测

优先过：

```text
FP64 affine/interpolation commutation
NSCR block fold
deploy branch deletion
```

再做 whole model。

---

## Step 5：随机 smoke

确认：

```text
shape
grad
params
FLOPs
fold
argmax
```

全部正常。

---

## Step 6：真实 LEVIR dry run

2~5 batch：

```text
forward
loss
backward
step
```

确认 gamma 学起来、梯度路径符合预期。

---

## Step 7：只启动 M1_NSCR_high2 / LEVIR

不要同时开：

```text
full encoder
PBRU
channel allocation
L-only
H-only
```

保持唯一变量。

---

## Step 8：按预注册判定

### PASS

```text
补 C1/C2
然后 WHU -> SYSU -> CDD
```

### WEAK

```text
记录
最多做机制归因
不扩四集
```

### FAIL

```text
停止 NSCR
按诊断决定是否转 PBRU
```

---

# 31. 我对当前课题路线的最终判断

从目前五轮结果看，STR-RepNet 已经完成了非常重要的一件事：

> **用约 20.1% 参数量下降、22.5% FLOPs 下降，把 HAM-CD 的四数据集 Macro F1 差距压缩到约 0.11pp，并在 WHU/SYSU 上超过 baseline。**

这意味着下一步最危险的做法不是“创新不够多”，而是：

> 为了追回 LEVIR 0.67pp，不断叠加彼此无关的模块，把一条已经很清楚的“结构重参数化”论文主线做散。

所以 Run6 应该比 Run3~Run5 更克制：

> **先定量证明误差发生在哪个尺度/阶段，再只做一个可以写出严格折叠公式、部署图完全不变、失败即停止的 cross-scale reparameterization。**

NSCR-Fuse 的优势不是“我能保证它一定涨点”，而是它满足现在最重要的四个条件：

1. **位置对**：直指 Run5 后剩余的 cross-scale/high-resolution 信息保真问题；
2. **主线对**：创新仍然是结构重参数化，不是 loss/蒸馏/attention；
3. **预算对**：部署 operator、Params、FLOPs 都可保持 Run2；
4. **实验信息量高**：无论成功失败，都能明确判断“native-scale cross-scale optimization”是否值得继续。

因此我的 Run6 主决策是：

```text
先诊断 LEVIR 尺寸/尺度/特征判别性
        ↓
M1_NSCR_high2 + last2
        ↓
按 F1 + IoU + Precision 预注册一次性裁决
        ↓
PASS -> 最小归因 + WHU/SYSU/CDD
FAIL -> 不救机制，若尺度诊断阳性再转 PBRU
```

---

# 32. 仍需补充/验证的证据

在真正提交 Run6 代码前，还缺四类本地实证：

1. **LEVIR train GT connected-component 面积分布**  
   当前仓库没有这份统计，不能直接断言“主要是 small object FN”。

2. **Run2 与 D0 的 per-component error profile**  
   只有全局 Recall/Precision/F1，无法知道 +0.24pp 来自哪一类目标。

3. **stage-wise feature discriminability**  
   当前没有证据证明 detail 是在 encoder、TAR 还是 fuse1/refine 丢失。

4. **NSCR 的 CUDA FP32 实际折叠误差**  
   数学上可折叠不等于 RTX 5090 的 whole-model FP32 一定 <1e-6，必须实测。

这四项都应该在 Run6 README 中明确区分为：

```text
代码/日志直接事实
证据支持推断
待验证假设
缺失信息
```

---

# 33. 参考依据

## 当前仓库

- `https://github.com/YuqiWang-code/STR-RepNet`
- `README.md`
- `docs/temporary/STR-RepNet_Run5_回退Run2与BOTR最小消融实验方案.md`
- `docs/temporary/models_and_metrics_TAR-DCR_Run5.txt`
- `docs/参考文献/文献索引.md`

## Baseline

- Li G, Han P, Wang W, et al. **HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection**. IEEE Transactions on Geoscience and Remote Sensing, 2026. DOI: `10.1109/TGRS.2026.3665418`.

## Structural Re-parameterization

- Wang A, Chen H, Lin Z, et al. **RepViT: Revisiting Mobile CNN From ViT Perspective**. CVPR 2024.
- Ding X, Zhang Y, Ge Y, et al. **UniRepLKNet: A Universal Perception Large-Kernel ConvNet...** CVPR 2024.
- Huang Z, Zhong S, Wen W, et al. **Stripe Observation Guided Inference Cost-free Attention Mechanism**. ECCV 2024.
- Ding X, Zhang X, Ma N, et al. **RepVGG: Making VGG-Style ConvNets Great Again**. CVPR 2021.
- Ding X, Zhang X, Han J, Ding G. **Diverse Branch Block**. CVPR 2021.
- Hu M, et al. **Online Convolutional Re-Parameterization (OREPA)**. CVPR 2022.

## RSCD near-neighbors

- ChangeMamba, IEEE TGRS 2024.
- CD-RLKNet, IJAEO 2024.
- DMFANet, IEEE TGRS 2025.
- ST-Mamba, IEEE TGRS 2025.
- DEIF-Mamba, IEEE TGRS 2025.
- LKMamba-CD, 2026.

---

## 一句话执行版

> **不要再改 TAR、边界或 loss。先给 LEVIR 做尺寸分层 FN + 尺度敏感性 + stage discriminability；若没有反证，就以 Run2 `full_last2` 为唯一锚点，在 `fuse1+fuse2` 上实现 NSCR-Fuse：训练期 native-scale BN 双分支，利用 bilinear 与逐通道 affine 的可交换性，部署吸收到原有 cross-scale 1×1；只跑 LEVIR，F1≥0.9175、IoU≥0.8475、Precision≥0.9230 才扩四数据集，低于 0.9159 直接停止，不救机制。**
