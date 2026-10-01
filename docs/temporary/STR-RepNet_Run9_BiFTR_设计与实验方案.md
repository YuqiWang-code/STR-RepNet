# STR-RepNet Run9：BiFTR 双侧冻训边界重参数化设计与实验方案

> **硕士课题**：《轻量化遥感二值变化检测中的结构重参数化研究》  
> **仓库**：`YuqiWang-code/STR-RepNet`  
> **本次审查版本**：`main @ a2130d5e1dc0fa4b8d9d7e683d87c67afba9f3e7`  
> **当前正式锚点**：Run2 `full_last2`，LEVIR F1=0.9144 / IoU=0.8424 / Precision=0.9260  
> **部署硬预算**：Params ≤ **28.828706M**，FLOPs ≤ **12.6062G**  
> **部署等价性规范**：解析折叠全程 FP64 → 最后一次 FP32 cast；最终 `argmax disagreement = 0`  
> **训练协议**：seed=2333，300 epoch，batch=16，CE + 2×Lovász，当前阶段继续 test-as-val  
> **正式结果纪律**：只认同一 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块

---

# 0. 三个方向性判断——先回答结论

## 0.1 判断一：结构性创新还要不要继续找？

### 结论

> **可以再找一次，但 Run9 应当是“最后一次正交结构搜索”；decoder 内部的 additive multi-branch folding 路线应当正式结束。**

我不建议再做任何：

```text
输出头新 basis
refine 新 branch
fuse1/fuse2 新 branch
group/permutation sweep
边界/空间先验 branch
PixelShuffle 变体
```

原因不是“理论上再也不可能涨”，而是你已经用 Run6/7/8 给出了相当强的**课题内负证据**：

```text
Run6  NSCR:
Recall +0.41pp
Precision -0.52pp
F1 0.9140 FAIL

Run7  PBRU:
LEVIR system +0.17pp，但 WEAK
phase-rep +0.09pp，rep-weak
WHU PixelShuffle topology -0.35pp

Run8  MPCR-Fine:
C0 0.9135
M1 0.9129
M1-C0 -0.06pp
rep-not-supported
d1 LDA 0.9372 < A0 0.9397
```

更重要的是三轮出现了共同签名：

> **零初始化、可折叠的额外 decoder 参数化会学习到非零结构，但主要表现为 Recall↓ / Precision↑ 的置信度锐化，而不是找回漏检。**

这说明失败不是“branch 没训练起来”：

- Run7 phase-basis γ 非零；
- Run8 MPCR γ≈0.9；
- 折叠和梯度路径均已验证；
- 但表征和最终 F1 没有改善。

因此，对**又一个 decoder additive branch** 的期望收益应评为：

> **低。**

这里不人为给“成功概率百分比”，因为单 seed 八轮实验不足以支持概率估计；但从已有实证看，要从 0.9144 跨到 PASS 线 0.9175，需要 **+0.31pp**，而最近 decoder 侧结构搜索没有展示出接近这一幅度且可跨数据集保持的净效应。

### Run9 的合理性

Run9 只有在同时满足以下条件时值得做：

1. **位置换掉**：离开 decoder；
2. **参数化家族换掉**：离开“若干并行零初始化 branch 最后线性相加”的模式；
3. **仍然严格 structural reparameterization**；
4. **部署图不增加任何 operator / 参数 / FLOPs**；
5. **一次 LEVIR 裁决，FAIL 就收尾。**

所以我的导师式决策是：

> **Run9 可以做，但只能做一个“encoder 侧、串行/乘性重参数化”的最后尝试。Run9 若不能 PASS，停止继续搜结构，进入论文收尾。**

---

## 0.2 判断二：若继续，最值得试的两个位置

### 候选 1 —— 首选：stage2→stage3 的 frozen→trainable encoder transition

**位置：**

```text
VMamba stage2
  ↓
layers[1].downsample
Conv2d(192→384, k=2, stride=2)
  ↓
Norm
  ↓
VMamba stage3
```

Run2 `encoder_train=last2` 的真实含义是：

```text
stage1 + stage2：冻结
stage3 + stage4：训练
```

因此：

> `layers[1].downsample (192→384)` 恰好位于**冻结区域 → 可训练区域的边界**，而该 transition 本身仍属于 frozen layer2。

这是目前一个非常干净、此前没有验证过的位置。

### 为什么它比 decoder 更值得试

已有 D0：

```text
full encoder LEVIR = 0.9168
vs
last2           = 0.9144
Δ = +0.24pp
```

它没有达到“encoder 是主要瓶颈”的强判据，也没有修复 small-object FN，但至少说明：

> **last2 之外的 encoder 适配不是完全无效。**

与其把整个 encoder 改成 full training，不如问一个更符合课题的问题：

> **能否只通过训练期结构重参数化，适配 frozen→trainable 的单个 transition，而部署仍恢复到原始的一个 2×2 stride-2 Conv？**

这既继承 D0 的正信号，又不把“full fine-tuning”包装成创新。

---

### 候选 2 —— 次选：stage3/stage4 内部 SS2D `out_proj` 的串行深线性重参数化

VMamba 的 SS2D 中存在：

```text
out_norm
→ out_proj: Linear(d_inner → d_model)
→ dropout
→ residual
```

可将训练期写成：

```text
out_proj
→ identity-initialized serial factor
```

部署时：

\[
W' = A W,\qquad b' = Ab+c
\]

重新并回原 `out_proj`。

优点：

- 不碰 decoder；
- 不改 TAR；
- 不增加部署结构；
- 属于**串行 deep-linear reparameterization**，不同于 Run3–8 的输出加支路。

但优先级低于候选 1，因为：

> stage3/4 的 `out_proj` 在 `last2` 下本来就已经可以训练，额外 serial factor 主要改变优化几何，而不是打开此前被冻结的适配位置。

所以 Run9 不建议同时做两个。

---

## 0.3 判断三：如果现在收尾，论文怎么组织？multi-seed 与 full encoder 怎么处理？

### 论文主线建议

如果 Run9 失败，我建议论文不再继续堆第十个模块，而是将贡献组织为：

#### Contribution 1：统一的 train-rich / deploy-simple structural reparameterization 框架

以：

```text
TAR + DCR + BN-FR
```

作为主方法。

核心不是说“所有 rep 分支都涨点”，而是：

> **将遥感二时相交互和 decoder 局部/通道/跨尺度融合统一写成训练期富结构、部署期单路径的解析折叠体系。**

#### Contribution 2：显著部署效率收益 + 可接受的精度折中

Run2 相对 HAM-CD：

```text
HAM-CD deploy:
36.081M
16.2593G

STR-RepNet:
28.828706M
12.6062G
```

约：

```text
Params -20%
FLOPs  -22%
```

而四数据集 Macro F1：

```text
HAM-CD ≈ 92.22
Run2   ≈ 92.11
```

只差约 0.11pp，同时：

```text
WHU / SYSU 超过 baseline
LEVIR / CDD 略低
```

这本身是一条清晰的：

> **accuracy-efficiency trade-off**

论文故事。

#### Contribution 3：严格解析部署与系统化负结果边界

Run3–Run8 不适合全部写成“六个创新”。

应整理成：

> **pre-registered mechanism exploration / design boundary study**

正文保留最有信息量的 2–3 个：

```text
NSCR：Recall↑ Precision↓
PBRU：输出 topology 有数据集依赖
MPCR：分支学到但 d1 判别性不升
```

其余放附录/补充材料。

可形成一个很有价值的结论：

> **并非增加任意可折叠训练自由度都会带来变化检测收益；当原始 deploy operator 的函数类已经足够且瓶颈不位于该层时，额外训练参数化可能只改变置信度而不增加漏检目标的可分性。**

这个结论比继续包装失败模块更可信。

### 是否需要四数据集 multi-seed？

#### 当前 Run9 搜索阶段：不需要

仍按当前：

```text
seed=2333
```

做机制筛选。

不能在搜索期用多 seed “救” WEAK/FAIL。

#### 方法冻结、准备论文正式结果后：建议需要

> **建议 3 seeds × 4 datasets，至少对最终主方法和其正式锚点做。**

因为你现在讨论的差异大量处在：

```text
0.05~0.30pp
```

量级。

如果最终论文要声称：

```text
+0.1 / +0.2pp
稳定提升
```

而没有 seed 波动，很难判断是不是训练噪声。

所以正式投稿阶段建议：

```text
seed = 2333 / S2 / S3
```

对：

```text
final STR-RepNet
vs
对应 clean anchor
```

在四数据集报告：

```text
mean ± std
```

**不需要把 Run3–Run8 全部补 multi-seed。**

这些已按单 seed 机制筛选归档即可。

### `encoder_train=full` 是否升格为正式协议？

当前不建议。

理由：

1. 目前只有 LEVIR D0：
   ```text
   0.9168
   +0.24pp
   ```
2. 它没有 4 数据集证据；
3. 它改变了：
   ```text
   trainable params
   training compute
   optimization scope
   ```
   虽然 deploy 不变；
4. 如果现在把 full encoder 升成主协议，Run2 A0 及 Run3–Run8 的比较基础都会发生变化。

因此正式定位：

> **D0 full-encoder 继续作为 encoder adaptation upper-bound / diagnostic。**

若论文方法最终冻结后你还想争取最终数字，可以补一个独立的：

```text
last2 vs full
4-dataset training-protocol ablation
```

如果 full 在 4 集上稳定正向，再把它作为：

> **training recipe**

而不是 structural innovation。

一旦正式改用 full：

> 所有最终主表中的 anchor 与 final model 必须在同一 encoder protocol 下重新跑，不能把 `full M1` 对 `last2 A0`。

---

# 1. Run9 主决策

Run9 选择候选 1：

> # **BiFTR — Bi-sided Frozen-to-Trainable Transition Reparameterization**
>
> 中文：**双侧冻训边界重参数化**

定位：

> 在 VMamba 的 frozen stage2 → trainable stage3 边界，用训练期前/后双侧 channel basis 变换重新参数化原有 stride-2 downsample Conv；部署时三者精确合并回原始单个 `Conv2d(192→384,k=2,s=2)`。

Run9 是：

> **最后一次结构搜索。**

预注册总规则：

```text
PASS  → 才考虑 WHU
WEAK  → 记录，停止
FAIL  → 结束结构搜索，进入论文收尾
```

不再有 Run10 “救机制”。

---

# 2. 为什么 BiFTR 不是 Run3–8 的换汤不换药

Run3–Run8 的主要训练参数化基本属于：

\[
y = F(x) + R_1(x)+R_2(x)+...
\]

部署：

\[
W_{eq}=W_F+\sum W_{R_i}
\]

本质是：

> **parallel additive over-parameterization**

BiFTR 改成：

\[
x'=(I+\Delta_{in})x
\]

\[
z=W_{ds}x'
\]

\[
y=(I+\Delta_{out})z
\]

所以 deploy kernel：

\[
W_{eq}=(I+\Delta_{out})W_{ds}(I+\Delta_{in})
\]

展开：

\[
W_{eq}=W_{ds}+\Delta_{out}W_{ds}+W_{ds}\Delta_{in}+\Delta_{out}W_{ds}\Delta_{in}
\]

关键区别是最后一项：

\[
\Delta_{out}W_{ds}\Delta_{in}
\]

这是**双侧乘性交互项**。

它不是：

```text
再加一个能折叠的 branch
```

而是：

> **通过输入 basis 与输出 basis 的联合可训练变化，对同一个 pretrained transition kernel 做双侧矩阵重参数化。**

这正是 Run9 与前几轮最需要保持的机制差异。

---

# 3. 选择 stage2→stage3 transition 的机制依据

256×256 输入下，VMamba 大致路径：

```text
patch embed
→ stage1
→ downsample 96→192
→ stage2
→ downsample 192→384   ← Run9
→ stage3
→ downsample 384→768
→ stage4
```

Run2 `last2`：

```text
stage1 frozen
stage2 frozen
stage3 trainable
stage4 trainable
```

所以：

```text
192→384 downsample
```

是唯一非常明确的：

> **frozen representation → trainable semantic representation**

边界。

Run9 不去：

- 重新训练整个 stage1/2；
- 增加 encoder depth；
- 增加 encoder channels；
- 修改 selective scan；
- 修改 TAR。

只给这个 frozen transition 一个**训练期可合并适配自由度**。

---

# 4. 模块定义

设原 downsample Conv：

\[
W\in\mathbb{R}^{C_o\times C_i\times2\times2}
\]

其中：

\[
C_i=192,\qquad C_o=384.
\]

原 bias：

\[
b\in\mathbb{R}^{384}.
\]

## 4.1 输入侧变换

定义：

\[
B=I_{192}+\Delta_{in}
\]

其中：

\[
\Delta_{in}\in\mathbb{R}^{192\times192}
\]

训练时：

\[
x_B=Bx.
\]

代码上可视为：

```text
1×1 192→192
identity + zero-init delta
```

## 4.2 原 transition Conv

\[
z=W*x_B+b.
\]

这里：

```text
kernel=2
stride=2
padding=0
```

保持 pretrained VMamba 的原始 operator。

## 4.3 输出侧变换

定义：

\[
A=I_{384}+\Delta_{out}
\]

其中：

\[
\Delta_{out}\in\mathbb{R}^{384\times384}.
\]

训练：

\[
y=Az.
\]

然后进入原：

```text
LayerNorm2d / norm
```

norm 结构、参数、位置完全不改。

---

# 5. C0 与 M1

## 5.1 A0：Run2 锚点，不重跑

```text
A0_Run2_full_last2
encoder_train=last2
BiFTR=0
D=160
bilinear head
```

LEVIR：

```text
Recall    0.9032
Precision 0.9260
F1        0.9144
IoU       0.8424
```

## 5.2 C0：Post-only Transition Reparameterization

组名：

> `C0_FTR_Post`

训练：

\[
B=I
\]

\[
A=I+\Delta_{out}
\]

即：

```text
stage2
→ original frozen downsample Conv
→ post serial basis A
→ Norm
→ stage3
```

它回答：

> **只在 frozen transition 输出端做 channel calibration，是否已经足够？**

## 5.3 M1：Bi-sided FTR

组名：

> `M1_BiFTR`

训练：

\[
B=I+\Delta_{in}
\]

\[
A=I+\Delta_{out}
\]

即：

```text
stage2
→ input serial basis B
→ original frozen downsample Conv
→ output serial basis A
→ Norm
→ stage3
```

它比 C0 多出的机制是：

> **输入 basis adaptation + 双侧乘性交互。**

核心归因：

\[
\Delta_{bi}=F1(M1)-F1(C0).
\]

---

# 6. 为什么 C0 不设计成“另一个 additive branch”

如果 C0 再使用：

```text
parallel conv branch
```

那么 Run9 又会退回之前已被大量验证过的 family。

因此 C0 也保持：

> **serial reparameterization**

只是从：

```text
post-only
```

到：

```text
pre + post
```

进行最小机制归因。

需要诚实说明：

> C0 与 M1 的训练期参数数目不完全相同，因此 `M1-C0` 证明的是“双侧参数化相对单侧参数化”的净贡献，而不是严格同 train-param 的容量消融。

部署图则完全相同。

---

# 7. epoch 0 初始化：与 Run2 逐位一致

这里不使用随机 A/B，而采用**零偏移参数化**：

\[
A=I+\Delta_{out},\qquad B=I+\Delta_{in}.
\]

初始化：

\[
\Delta_{out}=0,\qquad \Delta_{in}=0.
\]

代码建议：

```python
x = x + conv1x1_delta_in(x)   # delta weight = 0
z = base_downsample(x)
z = z + conv1x1_delta_out(z)  # delta weight = 0
```

注意：

> 虽然实现写成 residual 形式以保证初始化数值恒等，但它在结构上是**串行 basis transform**，部署折叠是 `A W B`，并非把两个独立输出 branch 直接加到最终预测上。

因为 delta conv 全 0：

```text
delta_in(x)  = exact 0
delta_out(z) = exact 0
```

所以 epoch 0：

```text
C0 == A0
M1 == A0
```

要求 smoke：

```text
max_abs_diff == 0.0
```

若不是 0：

> 禁止启动 Run9。

---

# 8. 梯度路径

零初始化不会造成参数永久不学习。

输入侧：

\[
x'=x+\Delta_{in}x
\]

即使：

\[
\Delta_{in}=0
\]

对参数的梯度仍可非零；输出侧同理。

smoke 必须验证：

```text
delta_in.weight.grad != 0  # M1
delta_out.weight.grad != 0 # C0/M1
```

---

# 9. 精确部署折叠公式

训练：

\[
y=A(W*(Bx)+b).
\]

由于 A/B 均为 1×1 channel transform，且 W 是 2×2 stride-2 Conv，可解析组合。

## 9.1 先折输入侧

设：

\[
B_{n,i}
\]

定义：

\[
W_B[m,i,p,q]=\sum_n W[m,n,p,q]B[n,i].
\]

若 C0：

\[
B=I,\qquad W_B=W.
\]

## 9.2 再折输出侧

\[
W_{eq}[o,i,p,q]=\sum_m A[o,m]W_B[m,i,p,q].
\]

合并：

\[
W_{eq}[o,i,p,q]=\sum_{m,n}A[o,m]W[m,n,p,q]B[n,i].
\]

即：

\[
\boxed{W_{eq}=AWB}
\]

## 9.3 bias

因为 input factor 不带 bias：

\[
b_{mid}=b.
\]

post factor 同样不带 bias，所以：

\[
\boxed{b_{eq}=Ab}
\]

如果原 downsample Conv 的 `bias=None`：

\[
b=0.
\]

---

# 10. FP64 组合规范

所有部署组合必须：

```python
W = base.weight.detach().double()
b = base.bias.detach().double() if base.bias is not None else zeros

A = I_out.double() + delta_out.double()

if mode == "bi":
    B = I_in.double() + delta_in.double()
else:
    B = I_in.double()

W_pre = einsum("mnxy,ni->mixy", W, B)
W_eq  = einsum("om,mixy->oixy", A, W_pre)
b_eq  = A @ b
```

最后：

```python
deploy_conv.weight.data = W_eq.float()
deploy_conv.bias.data   = b_eq.float()
```

严格禁止提前 FP32 compose；只允许最后一次 cast。

---

# 11. 训练图 / 部署图

## 11.1 A0

```text
stage2 feature
    ↓
Conv2×2 s2 192→384  [frozen]
    ↓
Norm
    ↓
stage3
```

## 11.2 C0 训练图

```text
stage2 feature
    ↓
Conv2×2 s2 192→384  [pretrained/frozen]
    ↓
(I + Δout)           [train-only]
    ↓
Norm
    ↓
stage3
```

## 11.3 M1 训练图

```text
stage2 feature
    ↓
(I + Δin)            [train-only]
    ↓
Conv2×2 s2 192→384   [pretrained/frozen]
    ↓
(I + Δout)           [train-only]
    ↓
Norm
    ↓
stage3
```

## 11.4 C0/M1 部署图

两者都变成：

```text
stage2 feature
    ↓
single Conv2×2 s2 192→384
    ↓
Norm
    ↓
stage3
```

与 Run2 operator graph 完全相同。

---

# 12. 部署 Params / FLOPs 增量证明

Run2 transition：

```text
Conv2d(192,384,k=2,s=2)
```

Run9 deploy transition：

```text
Conv2d(192,384,k=2,s=2)
```

因此：

\[
\Delta Params_{deploy}=0,
\qquad
\Delta FLOPs_{deploy}=0.
\]

理论整网仍：

```text
28.828706M
12.6062G
```

不需要 D*=158 一类通道回收。

---

# 13. 训练期额外参数

C0：

\[
384^2=147456
\]

个 `Δout` 权重。

M1 再加：

\[
192^2=36864.
\]

总：

\[
184320
\]

train-only weights。

无 bias。

这不会进入 deploy 参数量。

---

# 14. 为什么 Run9 不做 D* channel search

Run7 需要 D* 是因为 deploy head 从 2 channels 变成 32 channels，部署图确实变宽。

Run9：

```text
deploy operator type
kernel
Cin
Cout
stride
```

全部与 Run2 相同。

所以 Run9 应做的不是搜一个更小 D，而是：

> **机器预算 equality audit。**

新增脚本：

```text
analyse/search_biftr_budget.py
```

预期：

```text
A0:
Params 28.828706M
FLOPs 12.6062G

C0 deploy:
ΔParams = 0
ΔFLOPs = 0

M1 deploy:
ΔParams = 0
ΔFLOPs = 0
```

如果不为 0：

> P0 实现错误，禁止训练。

---

# 15. 为什么只改一个 transition

不要同时：

```text
stage1→2
stage2→3
stage3→4
```

否则一轮就又变成“全 encoder 多点插模块”。

首轮只选：

> **stage2→stage3 / `layers[1].downsample`**

因为这是 `last2` 的 frozen/trainable 边界。

如果 M1 FAIL：

> 不允许再用 stage/scope sweep 救机制。

---

# 16. 与 full-encoder D0 的关系

D0：

```text
full encoder F1=0.9168
```

是 Run9 的**机制先验**，不是新的比较锚点。

Run9 第一关仍按：

```text
A0=0.9144
PASS=0.9175
```

若 M1 ≈ 0.9165~0.9169：

> targeted frozen-boundary adaptation 大致复现 full encoder 的收益，但仍属 WEAK。

不能事后把 0.9168 改成 PASS。

若 M1 ≥ 0.9175，才说明：

> 串行结构重参数化在不放开整个 encoder 的情况下超过现有 encoder adaptation upper-bound observation。

---

# 17. LEVIR 预注册 PASS / WEAK / FAIL

沿用此前体系。

A0：

```text
F1        0.9144
IoU       0.8424
Precision 0.9260
```

## PASS

必须同时：

```text
F1        >= 0.9175
IoU       >= 0.8475
Precision >= 0.9230
```

并满足：

```text
Deploy Params <= 28.828706M
Deploy FLOPs  <= 12.6062G
argmax disagreement = 0
```

## WEAK

任一：

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

> **记录，停止 Run9 扩展。**

Run9 是最后结构搜索，因此 WEAK 不再进入“下一版 BiFTR”。

## FAIL

任一：

```text
F1 < 0.9159
Precision < 0.9220
budget fail
argmax disagreement != 0
fold correctness fail
```

处理：

> **结束结构搜索，进入论文收尾。**

禁止：

```text
换 stage
rank sweep
scope sweep
加 stage3→4
full encoder + BiFTR
PBRU + BiFTR
MPCR + BiFTR
loss
threshold
augmentation
```

---

# 18. Run9 结构归因判据

主归因：

\[
\Delta_{bi}=F1(M1)-F1(C0).
\]

## rep-supported

```text
M1 - C0 >= +0.15pp
```

且：

```text
Precision(M1) >= Precision(C0) - 0.15pp
```

## rep-weak

```text
+0.05pp <= M1-C0 < +0.15pp
```

只能写：

> 双侧 basis adaptation 存在单 seed 正向迹象。

## rep-not-supported

```text
M1-C0 < +0.05pp
```

包括负值。

解释：

> 单侧 post calibration 已包含主要效果，或额外 input-side multiplicative factor 没有有效贡献。

---

# 19. 继续 WHU 的双门槛

只有：

```text
M1 = PASS
```

且：

```text
M1-C0 >= +0.05pp
```

才允许 WHU。

其中：

```text
>= +0.15pp  → rep-supported
+0.05~0.15 → rep-weak，但允许用 WHU 做一次跨集裁决
```

---

# 20. WHU gate

WHU A0：

```text
0.9514
```

若启动 WHU，需要：

```text
M1 WHU >= 0.9514
M1-C0 >= +0.05pp
```

才允许：

```text
SYSU → CDD
```

如果：

```text
M1 WHU < 0.9499
```

即退化超过 0.15pp：

> 明显 FAIL，立即停止。

即使退化在 0~0.15pp：

> 也不扩 SYSU/CDD，因为 Run9 的目标是找可跨集支撑主张的结构，而不是再接受 dataset-specific 弱效应。

---

# 21. 最终四数据集扩展标准

若通过 LEVIR + WHU：

```text
WHU
→ SYSU
→ CDD
```

最终继续沿用：

```text
Macro F1 >= 92.25%
LEVIR     >= 91.75%
至少 3/4 数据集 ΔF1 >= 0
任一退化 <= 0.15pp
```

每集：

```text
budget合格
argmax=0
```

---

# 22. 失败预案——Run9 后不再救机制

## 情况 A：C0、M1 都 FAIL

结论：

> frozen/trainable transition adaptation 也无效。

决策：

> **结束结构搜索。**

## 情况 B：C0 WEAK，M1 WEAK，M1>C0

结论：

> serial encoder-boundary reparam 可能有效，但幅度不足。

决策：

> **作为弱结果记录，不做 scope/stage sweep。**

## 情况 C：C0 PASS，M1≈C0

结论：

> post-only calibration 有效，但“双侧 multiplicative reparameterization”主张没有独立贡献。

决策：

> 不把 BiFTR 升格为创新。

## 情况 D：M1 PASS，M1-C0 rep-supported

进入 WHU。

这是 Run9 唯一真正理想结果。

## 情况 E：M1 Recall↓、Precision↑ 再现

若再次出现该签名，即使 delta norm 很大，也说明：

> multiplicative encoder adaptation 最终仍主要改变置信度，而没有增加漏检目标的可分性。

严格按 F1 / Precision 判据裁决，不另调 threshold。

---

# 23. 代码修改点

## 23.1 `models/changedetection/models/Mamba_backbone.py`

建议把 Run9 wrapper 写在这里，不直接大改第三方 `classification/models/vmamba.py`。

新增：

```python
class BiFTRTransition(nn.Module):
    ...
```

职责：

```text
保存原 pretrained Conv2d
Δin（M1）
Δout（C0/M1）
forward
get_equivalent_kernel_bias
switch_to_deploy
stats
```

初始化时在 `Backbone_VSSM` 已加载 pretrained 后，对：

```text
encoder.layers[1].downsample
```

中的：

```text
Conv2d(192,384,k=2,s=2)
```

进行 wrapper。

必须用 module/shape assert 定位，不能静默假设 index。

例如同时 assert：

```text
in_channels == 192
out_channels == 384
kernel_size == (2,2)
stride == (2,2)
```

不满足即报错。

## 23.2 `models/changedetection/models/STRRepNet.py`

新增：

```text
use_biftr
biftr_mode = post | bi
biftr_transition = "stage2_to_stage3"
```

建议顺序：

```text
1. build/load pretrained encoder
2. install BiFTR wrapper if enabled
3. _setup_encoder_train()
4. 显式重新打开 Δin/Δout requires_grad
```

因为 `last2` freeze 会把 layer1 的 wrapper 一并冻结，必须显式恢复 train-only delta 参数。

base Conv 继续：

```text
requires_grad=False
```

## 23.3 `switch_to_deploy()`

增加：

```text
encoder BiFTR switch_to_deploy
```

然后：

```text
TAR deploy
DCR deploy
```

部署后 state_dict 不允许出现：

```text
delta_in
delta_out
biftr adapter
```

仅保留一个等价 downsample Conv。

## 23.4 `models/changedetection/script/train.py`

新增：

```text
--use_biftr 0|1
--biftr_mode post|bi
```

强 guard：

```text
encoder_train == last2
decoder_dim == 160
head_mode == bilinear
use_pbru == 0
use_mpcr == 0
use_nscr == 0
use_botr == 0
```

Run9 不允许旧模块叠加。

TEST RESULTS 新增：

```text
[BIFTR] 1
[BIFTR-MODE] post|bi
[BIFTR-TRANSITION] stage2_to_stage3
[BIFTR-DELTA-NORM] in=... out=...
[BIFTR-EFFECTIVE-UPDATE-NORM] ...
[BIFTR-CROSS-TERM-RATIO] ...
```

统计仅用于解释，不用于选 checkpoint。

## 23.5 `smoke_test.py`

新增 Run9 检查：

```text
[ ] wrapper 定位的是 192→384,k2,s2
[ ] base downsample frozen
[ ] C0 只有 Δout trainable
[ ] M1 Δin/Δout trainable
[ ] Δ 参数初始全 0
[ ] epoch0 C0 vs A0 max_diff = 0.0
[ ] epoch0 M1 vs A0 max_diff = 0.0
[ ] Δout grad != 0
[ ] Δin grad != 0（M1）
[ ] deploy 后 delta 参数全部消失
[ ] deploy params 与 A0 exact equal
[ ] deploy FLOPs 与 A0 exact equal
[ ] argmax disagreement = 0
```

## 23.6 `test_reparam_equivalence.py`

增加三层。

### T0：纯 FP64 代数

随机：

```text
W: Co×Ci×2×2
A: Co×Co
B: Ci×Ci
x
```

比较：

```text
A( W * (B x) )
```

与：

```text
W_eq * x
```

要求：

```text
max_abs_error < 1e-12
```

### T1：BiFTRTransition block

扰动：

```text
Δin / Δout
```

到训练后合理量级，不能只测全零初始化。

FP64 kernel composition：

```text
< 1e-12
```

FP32 train graph vs deploy conv：

```text
< 2e-5
```

或以训练前实测合理 envelope 固化，不能看结果后改。

### T2：whole model

测试：

```text
A0
C0 post
M1 bi
```

要求：

```text
whole-model max_abs_error < 2e-4
argmax disagreement = 0
```

论文继续记录实际误差，不声称整个网络 FP32 `<1e-6`。

---

# 24. 预算机器验证

新增：

```text
analyse/search_biftr_budget.py
```

严格用同一环境 / fvcore / selective scan handlers 重建：

```text
A0 deploy
C0 deploy
M1 deploy
```

输出 raw integer：

```text
params
flops
unsupported_ops
delta_params
delta_flops
```

硬要求：

```text
C0 delta_params = 0
M1 delta_params = 0
C0 delta_flops  = 0
M1 delta_flops  = 0
```

如果 fvcore 最后一位浮点打印有格式差异：

> 以原始 counts 和 operator graph 核验，不允许“大约相同”。

---

# 25. Run9 分析工具

新增：

```text
analyse/levir_biftr_profile.py
```

目标不是再寻找超参数，而是解释：

> BiFTR 是否真的改变 frozen→trainable transition 的可分性。

## 25.1 hook

建议：

```text
stage2 output / downsample input
transition output pre-Norm
transition output post-Norm
stage3 output
stage4 output
```

比较：

```text
A0
C0
M1
```

## 25.2 输出指标

沿用已有工具的成熟口径：

```text
centroid distance
normalized Fisher
diag-LDA AUROC
effective rank
mean |channel correlation|
```

另加 weight-space 统计：

```text
||Δin||F
||Δout||F

||W_eq - W0||F / ||W0||F

linear input-side contribution:
||W Δin||F

linear output-side contribution:
||Δout W||F

multiplicative cross term:
||Δout W Δin||F

cross-term ratio:
||Δout W Δin||F / ||W_eq-W||F
```

若 M1 PASS 且 cross-term 明显非零，才有证据支持：

> 双侧乘性耦合确实被利用。

如果 cross-term 接近 0：

> 即使 M1 涨点，也更像单侧 calibration。

---

# 26. 最小实验组

| 组 | 配置 | LEVIR | 作用 |
|---|---|---:|---|
| A0 | Run2 full_last2，不重跑 | 0.9144 | 正式锚点 |
| **C0_FTR_Post** | stage2→3 post-only serial reparam | TBD | 单侧适配控制 |
| **M1_BiFTR** | stage2→3 pre+post 双侧 reparam | TBD | Run9 主实验 |

公共条件：

```text
rep_mode=full
encoder_train=last2
decoder_dim=160
head_mode=bilinear

use_botr=0
use_nscr=0
use_pbru=0
use_mpcr=0

seed=2333
epochs=300
batch=16
lr=1e-4
lovasz=2.0
temporal_swap_prob=0
```

---

# 27. 为什么 A0 不重跑

继续沿用前八轮纪律。

A0 已经是同一协议正式结果：

```text
0.9144
```

Run9 不因为换到 encoder 位置就重新抽一次 A0 seed=2333 寻找更有利 baseline。

只有最终论文 multi-seed 阶段才统一补。

---

# 28. 启动顺序

## Step 1：冻结 Run8

记录：

```bash
git rev-parse HEAD
# a2130d5e1dc0fa4b8d9d7e683d87c67afba9f3e7
```

Run8：

```text
FAIL
rep-not-supported
```

不可回头改判据。

## Step 2：只实现 BiFTR

修改：

```text
Mamba_backbone.py
STRRepNet.py
train.py
smoke_test.py
test_reparam_equivalence.py
```

新增：

```text
analyse/search_biftr_budget.py
analyse/levir_biftr_profile.py
train_scripts/TAR-DCR/Run9/
```

不碰：

```text
tar.py
dcr_decoder.py 的方法结构
loss
dataset augmentation
```

如为了参数透传需要极少量入口改动，可以改文件接口，但不能新增 TAR/DCR 机制。

## Step 3：等价性测试先行

顺序：

```text
T0 FP64 A-W-B algebra
↓
T1 BiFTR block
↓
T2 whole model
```

全部过再继续。

## Step 4：预算 equality audit

必须：

```text
ΔParams=0
ΔFLOPs=0
```

否则停止修代码，不能训练。

## Step 5：真实 LEVIR dry run

C0 / M1：

```text
2~5 batch
forward
loss
backward
optimizer step
```

确认：

```text
delta grad finite
base frozen transition grad=None
stage3/4正常 trainable
stage1/2其余参数 frozen
loss finite
```

## Step 6：GPU0 启动 LEVIR

```text
C0_FTR_Post
M1_BiFTR
```

按显存决定并行。

不要提前启动：

```text
WHU
SYSU
CDD
```

Run9 严格按 LEVIR gate。

## Step 7：读取正式结果

只读：

```text
train_log.txt
最后一个完整
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

计算：

```text
C0-A0
M1-A0
M1-C0
```

按预注册一次性裁决。

## Step 8：只在 PASS 后跑 profile

机制 profile 不用于选 checkpoint。

对：

```text
A0/C0/M1
```

统一分析。

## Step 9：是否进入 WHU

只有：

```text
M1 PASS
且
M1-C0 >= +0.05pp
```

才启动 WHU。

---

# 29. 目录建议

```text
train_scripts/TAR-DCR/Run9/
├── README.md
├── C0_FTR_Post/
│   ├── train_LEVIR-CD-256.sh
│   ├── train_WHU-CD-256.sh
│   ├── train_SYSU-CD-256.sh
│   └── train_CDD-CD-256.sh
└── M1_BiFTR/
    ├── train_LEVIR-CD-256.sh
    ├── train_WHU-CD-256.sh
    ├── train_SYSU-CD-256.sh
    └── train_CDD-CD-256.sh
```

可以预写脚本，但未过 gate 的脚本不得启动。

---

# 30. checkpoint / log 路径

建议：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run9/<group>/<dataset>/

/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run9/<group>/<dataset>/train_log.txt
```

诊断：

```text
/home/yqwang/outputs/STR-RepNet/diagnostics/Run9_BiFTR/
```

不同组 checkpoint 禁止互相 resume。

---

# 31. P0 / P1 / P2 风险

## P0：正确性

### P0-1 包错 transition

一定用 shape/operator assert。

只允许：

```text
192→384
kernel2
stride2
```

### P0-2 freeze 顺序错误

最危险。

若先 wrapper，再 `_setup_encoder_train()`，`Δin/Δout` 会被整体冻结。

必须最后显式：

```text
delta.requires_grad_(True)
```

并 smoke 检查。

### P0-3 base Conv 被误解冻

Run9 假设是：

> frozen transition + train-only serial reparameterization

base 必须保持 frozen。

### P0-4 bias fold 错误

post matrix A 必须同时作用 W 和 bias。

### P0-5 deploy 后 adapter 残留

state_dict / module graph 必须确认无 delta conv / extra 1×1。

## P1：方法风险

### P1-1 D0 已表明 encoder 只提供 +0.24pp

这意味着 Run9 的先验并不强。

> **我预期 Run9 最可能的结果是 WEAK 或 FAIL，而不是大幅突破。**

因此把它定义成最后一次结构实验最合理。

### P1-2 full encoder 没修小目标 FN

所以不要把 BiFTR 动机写成：

> “专门解决小目标漏检”。

更严谨：

> **改善 frozen low-level representation 与 trainable high-level semantics 之间的 transition optimization / basis alignment。**

small-object 是否改善是结果，不是前提。

### P1-3 可能仍只做置信度 calibration

所以保留 Recall/Precision 与 transition profile。

## P2：工程

- 不改第三方 VMamba 主源码优先；
- 不提交 checkpoint/log/cache；
- Run9 训练前 commit；
- 每组独立目录；
- smoke → budget → dry run → 300 epoch；
- 服务器仍按现有 RSML-3 规范。

---

# 32. Run9 若成功，论文如何吸收

如果：

```text
M1 PASS
M1-C0 rep-supported
WHU 不退
```

则论文主线可形成：

### 贡献 1：TAR

时相代数重参数化。

### 贡献 2：DCR

decoder-wide compositional reparameterization。

### 贡献 3：BiFTR

> encoder frozen/trainable boundary 的双侧乘性结构重参数化。

三个位置形成：

```text
encoder transition
→ temporal bridge
→ decoder
```

但部署仍然没有额外 train-only operator。

这是比继续加 decoder 模块更完整的结构重参数化论文叙事。

---

# 33. Run9 若失败，立即切换收尾方案

若 M1：

```text
WEAK 或 FAIL
```

我的建议是：

> **不再设计 Run10。**

进入以下正式工作：

1. 清理主方法，只保留最终部署有效组件；
2. 统一重新计算 Params/FLOPs；
3. 补部署前后实际误差；
4. 画 train graph / deploy graph；
5. 组织 Run1–Run9 证据；
6. 补最终 multi-seed；
7. 做论文级公平比较；
8. 将失败模块放到 systematic exploration / supplementary；
9. 不把调参和失败 branch 继续包装成创新。

---

# 34. 正式 multi-seed 执行建议

在 Run9 裁决完成后再做。

如果 Run9 成功：

```text
A0
M1_BiFTR
```

在：

```text
CDD
LEVIR
SYSU
WHU
```

各 3 seed。

如果 Run9 失败、Run2 仍为最终版本：

> 至少对最终 STR-RepNet / clean anchor 做 3 seed 稳定性。

如果要做与 HAM-CD 的统计公平对比，最好也同协议重跑 HAM-CD 多 seed；若算力不允许，正文必须明确：

```text
HAM-CD 为单次复现实验
STR-RepNet 稳定性另报 mean±std
```

不能把两种统计口径混成显著性结论。

---

# 35. full encoder 的最终处理建议

保持：

```text
D0_full_encoder = diagnostic upper bound
```

直到完成 4 数据集独立验证。

论文可以报告：

> LEVIR 上 full encoder 从 0.9144 提升至 0.9168，说明增加训练期 encoder 自由度存在有限正收益，但未达到预注册的主瓶颈判据，且尚无跨数据集证据，因此主协议仍采用 last2。

这是足够严谨的。

如果论文收尾阶段补了 4 数据集 full：

### 全集稳定正向

可把 full 写为：

> training protocol improvement

但必须同步重跑最终主方法。

### 数据集不一致

继续保留 last2。

---

# 36. Run9 成功阈值再强调一次

Run9 不是：

```text
只要 0.9145 就继续
```

也不是：

```text
比 Run8 0.9129 高就算成功
```

唯一标准仍是：

```text
PASS:
F1 >= 0.9175
IoU >= 0.8475
Precision >= 0.9230

WEAK:
0.9159 <= F1 < 0.9175
或副指标不满足

FAIL:
F1 < 0.9159
或 Precision < 0.9220
```

并且 `M1-C0` 必须独立做 structural attribution。

---

# 37. 立即执行顺序

```text
1. 冻结 a2130d5 与 Run8 FAIL 结论
2. 不再修改 decoder 方法结构
3. 在 Mamba_backbone.py 实现 BiFTRTransition
4. 只包装 layers[1].downsample 的 192→384 stride-2 Conv
5. STRRepNet 增加 use_biftr / biftr_mode
6. 修正 freeze 顺序，只让 delta 参数额外 trainable
7. train.py 增加 CLI / 互斥 guard / TEST 标签
8. smoke_test 做 epoch0 exact equality + gradient + deploy graph clean
9. equivalence 做 T0 FP64 / T1 block / T2 whole model
10. search_biftr_budget.py 验证 ΔParams=ΔFLOPs=0
11. LEVIR C0/M1 2~5 batch dry run
12. GPU0 启动 C0_FTR_Post + M1_BiFTR
13. 300 epoch 后只读最终 TEST RESULTS
14. 一次性按 PASS/WEAK/FAIL + M1-C0 裁决
15. 只有 PASS 且 rep 至少 weak → WHU
16. WHU 也非负且 rep 至少 weak → SYSU → CDD
17. Run9 WEAK/FAIL → 不再 Run10，进入论文收尾
18. 方法冻结后补 3-seed × 4 datasets 正式稳定性
```

---

# 38. 一句话 Run9 方案

> **Run9 不再搜索 decoder 分支，而把最后一次结构尝试放到 `encoder_train=last2` 的 frozen stage2 → trainable stage3 边界：以 BiFTR 将原 `Conv2d(192→384,k=2,s=2)` 训练期重参数化为 `(I+Δout) W (I+Δin)`，C0 只训练 post factor，M1 训练 pre+post 双侧 factor；两个 Δ 全零初始化，epoch0 与 Run2 逐位一致，部署时 FP64 精确合并回原单个 2×2 stride-2 Conv，因此 Params/FLOPs 严格 +0。LEVIR 仍按 0.9175/0.9159 体系一次性裁决；Run9 若 WEAK/FAIL，不再设计 Run10，直接进入论文与 multi-seed 收尾。**

---

# 39. 本次仓库审查依据

本次按用户给定版本实际读取/核对：

```text
main @ a2130d5e1dc0fa4b8d9d7e683d87c67afba9f3e7

README.md

train_scripts/TAR-DCR/Run8/README.md

docs/temporary/
  STR-RepNet_Run8_MPCR-Fine_设计与实验方案.md

models/changedetection/models/
  STRRepNet.py
  Mamba_backbone.py
  tar.py
  dcr_decoder.py
  reparam.py

models/classification/models/
  vmamba.py

models/changedetection/script/
  train.py
  smoke_test.py
  test_reparam_equivalence.py

analyse/
  levir_fine_stage_profile.py
```

已核对的关键代码事实：

```text
Backbone_VSSM:
layer.blocks → layer.downsample
outnorm 独立输出多尺度 feature

VMamba downsample v2:
Conv2d(dim→out_dim,k=2,stride=2)
→ norm

Run2 last2:
layers[2], layers[3] + outnorm2/outnorm3 trainable
更浅 stage 保持 frozen
```

因此 `layers[1].downsample` 正好是当前协议下的 frozen→trainable transition，适合作为 Run9 的最后一个正交结构位置。
