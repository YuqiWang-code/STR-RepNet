# STR-RepNet：中间模块 + 解码器统一结构重参数化方案

> **方案名：TAR-DCR（Temporal Algebraic Re-parameterization + Decoder-wide Compositional Re-parameterization）**  
> **适用仓库**：`YuqiWang-code/STR-RepNet`  
> **冻结项**：VSSM-Tiny / VMamba Siamese Encoder 完全冻结，不改结构、不改预训练权重、不参与本轮创新设计。  
> **唯一目标**：在不提高部署 Params/FLOPs 的前提下，让训练图拥有更丰富的二时相与空间归纳偏置，并通过显式代数变换收缩为单路径部署图。  
> **结果纪律**：本文档是**方法设计与代码审查结论**，不是实验结果；任何 F1 提升都必须以同一 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块为准。

---

## 一句话总览判断

**唯一推荐 TAR-DCR：用四级 `Concat + Sum + signed-Diff` 的 Temporal Algebraic Rep Bridge 替换当前高开销 `concat→3×3 dense conv`，部署时每级严格坍缩为一个 `1×1` temporal projection；同时把当前 HAM decoder 的“实际可执行线性卷积子图”系统改造成可组合 Rep 图，而 SSM、SE、Softmax attention、LayerNorm、激活与门控全部保留为不可约动态算子——这是当前路线中最有机会同时做到“F1 增益 + 约 30.2M 部署参数 + 约 14.75G FLOPs + 严格 deploy-equivalence”的方案。**

> 不能诚实地在未跑实验前保证 CDD≈97 / LEVIR≈92 / SYSU≈84 / WHU≈95；本方案的目标是最大化达到这些目标的概率，并让失败能够被最小消融明确证伪。

---

# 1. 先给 ASCII：训练图与部署图

## 1.1 全局结构

### 训练图：TAR-DCR

```text
              ┌────────────────────────────────────────────────────┐
T1 image ───► │                                                    │
              │ Frozen weight-sharing VSSM-Tiny / VMamba Encoder  │
T2 image ───► │  F1(96)  F2(192)  F3(384)  F4(768)                │
              └──────────────┬─────────────────────────────────────┘
                             │ pre_i, post_i   (i=1..4)
                             ▼
              ┌──────────────────────────────────────────┐
              │ TAR Bridge @ each scale                  │
              │                                          │
              │ [pre,post] ─ Conv1×1+BN ─┐              │
              │ pre+post   ─ Conv1×1+BN ─┼─ SUM ─► Z_i  │
              │ post-pre   ─ Conv1×1+BN ─┘              │
              │             ↑ signed diff only           │
              └───────────────────┬──────────────────────┘
                                  │ Z_i : 128 ch
                                  ▼
              ┌─────────────────────────────────────────────────────┐
              │ RepHAM stage                                       │
              │                                                     │
              │  ┌──────────── TAM / SpatialMamba ───────────────┐ │
              │  │ RepCPE-DW3 → LN → SSM(dynamic)               │ │
              │  │              └─ StateFusion d=1/3/5          │ │
              │  │                 (existing SRP, exactness fix) │ │
              │  │ → residual → RepCPE-DW3 → LN → MLP           │ │
              │  └───────────────────────────────────────────────┘ │
              │                         │                           │
              │                         ├───────────────┐           │
              │                         │               │           │
              │  ┌──────────── MDTA / TAB ───────────┐ │           │
              │  │ LN → RepQKV-1×1 → RepDW3         │ │           │
              │  │    → normalize → QKᵀ → Softmax   │ │           │
              │  │    → AV → RepPW-1×1 → residual   │ │           │
              │  └───────────────────────────────────┘ │           │
              │                         │               │           │
              │                         └───────┬───────┘           │
              │                                 ▼                   │
              │ HAM algebraic fusion:                               │
              │ [TAM,TAB] / TAM+TAB / TAB-TAM                      │
              │       → 3×(1×1+BN) → SUM → ReLU                   │
              └───────────────────┬─────────────────────────────────┘
                                  │
                     previous stage ↑ bilinear + add
                                  │
                                  ▼
              ┌────────────────────────────────────────────┐
              │ RepICSF (Stage 2~4 对应当前 3 个 smooth)   │
              │ RepConv3×3 → ReLU                          │
              │ → RepDW3×3 → ReLU                          │
              │ → SE(GAP→1×1→ReLU→1×1→Sigmoid) [保留]     │
              │ → residual add → ReLU                      │
              └────────────────────────────────────────────┘
                                  │
                                  ▼
                            main_clf 1×1
                                  │
                             bilinear upsample
                                  ▼
                               CD logits
```

### 部署图：所有新增 Rep 分支消失

```text
pre_i ─┐
       ├─ CONCAT ─► single Conv1×1(2C_i→128, bias=True) ─► Z_i
post_i ┘

Z_i
 │
 ├─► TAM:
 │    single DW3×3(CPE, identity folded where safe)
 │    → LN
 │    → SSM / selective scan                    [不可折，保留]
 │       └─ single DW11×11 StateFusion          [已有 Rep，修正显式转换]
 │    → residual
 │    → single DW3×3(CPE)
 │    → LN → MLP                                [激活跨不过去]
 │
 ├─► MDTA:
 │    LN
 │    → single Conv1×1(C→3C)
 │    → single DW3×3(3C)
 │    → normalize / QKᵀ / Softmax / AV          [不可折，保留]
 │    → single Conv1×1(C→C)
 │    → residual
 │
 └─► single Conv1×1([TAM,TAB]:256→128)
      → ReLU
      → upsample + add
      → single Conv3×3
      → ReLU
      → single DW3×3
      → ReLU
      → SE                                       [不可折，保留]
      → residual + ReLU
      → next stage
```

**部署图中不存在新增的并行 temporal branch、1×3/3×1 branch、QKV 辅助 branch 或 HAM fusion branch。**

---

# 2. 证据表：仓库实际代码与论文结构必须先区分

下面按“代码直接事实 / 论文事实 / 推断 / 待验证”分开。

## 2.1 代码直接事实

| 位置 | 仓库当前事实 | 对方案的影响 |
|---|---|---|
| `README.md` | baseline = VSSM-Tiny + HAM-CD decoder；36.08M / 16.26G；目标四数据集 F1 为约 97/92/84/95 | 部署硬上限 |
| `MambaBCD.py` | Encoder 输出四级特征后直接送入 `ChangeDecoder`，最终 `128→2` 的 `1×1` 分类头 | Encoder 可完全冻结 |
| Decoder `dim_convert{4,3,2,1}1` | 当前不是 1×1，而是 `OverlapPatchEmbed(2C_i,128)`，内部是 **dense 3×3 conv** | TAR 可以在不增 FLOPs 情况下替换，并显著降复杂度 |
| Decoder forward | 四级均先 `torch.cat([pre,post],dim=1)`，不存在显式 sum/diff 分支 | Temporal Rep 有明确新增归纳偏置 |
| `Spatial_Mamba.StateFusion` | 训练期 3 路 3×3 DWConv，dilation=1/3/5；eval 时合成 11×11 DW kernel | 已有 SRP，不能当新创新 |
| `TransformerBlock.forward` | `x = x + attn(norm1(x))` 生效；`ffn(norm2(x))` **被注释** | 当前 executable graph 中 FFN 不工作，不能把“改 FFN”作为本轮有效变量 |
| Decoder Stage IV | `p12 = self.at_layer_21(p10)`；`at_layer_11` 被定义但没有调用 | `at_layer_11` 是死参数；Stage 3/4 共用同一个 MDTA 参数模块 |
| Decoder | `st_block_42/43, 32/33, 22/23, 12/13` 被注册但 forward 不调用 | 这些参数可在部署导出时删除，但只能算工程清理，不算创新 |
| `ICSFBlock` | 当前源码实际为单输入：`3×3 dense conv→DW3×3→SE→residual` | 与 HAM-CD 论文 Fig.3 的双输入 CFEM/SFEM 不是同一实现，设计必须以源码为准 |
| `train.py` | 每 epoch 用 `test_list` 做 `_evaluate()`，并用 test F1 选 best checkpoint | **P0 实验有效性风险：测试集参与模型选择** |

## 2.2 HAM-CD 论文事实

HAM-CD 论文给出的设计是：共享 VMamba encoder；四级 HAM decoder；HAM 由并行 TAM + TAB 组成；TAB 是 channel-transposed attention；ICSF 通过 channel/spatial enhancement 做跨级融合。论文的 TAM 使用 Spatial-Mamba/TASF 思路，TAB 的 Q/K/V 为 pointwise + depthwise projection，ICSF 包含 SE-style channel gate。

**因此本课题不应声称“首次把 dilation 1/3/5 合成大核”或“首次在 HAM 中使用结构重参数化”。**

## 2.3 当前最重要的 P0 问题：StateFusion 的“折叠”需要先变成真正可验证的显式转换

当前 `StateFusion.forward()` 依赖 `self.training`：

```text
training=True:
    每个 dilation branch 先 replicate padding，再 conv

training=False:
    构造 11×11 sparse merged kernel
    F.conv2d(..., padding=5)   # zero padding
```

训练分支使用 **replicate padding**，当前 eval merged branch 使用 **zero padding**。边界位置并非严格同一个算子；另外 `_merge_weight` 是 forward 中惰性缓存，不是显式 `switch_to_deploy()` 后的正式参数。

因此，如果不处理这一点，即使 TAR/RepConv 都写对，**整模型 `<1e-6` 仍可能被 inherited StateFusion 破坏**。

### 推荐处理

新增显式：

```python
StateFusion.switch_to_deploy()
```

并把 deployment state 变成真正的单个 `11×11 DWConv`。

为满足你规定的“FP32 eval fold 前后 `<1e-6`”，建议：

- `model.eval()` 时，`deploy=False` 仍走“可比的 multi-branch eval graph”；
- `switch_to_deploy()` 后才替换为单 11×11 DWConv；
- 训练/部署 padding 必须统一。
- 若坚持保留当前训练分支的 replicate 语义，则 deploy 使用：
  `F.pad(x, 5, mode="replicate") → DWConv11(padding=0)`；
- 若未来愿意重跑严格 baseline，可统一改成 zero padding，从训练开始就与 `padding=5` 的 deploy 完全一致。

**这属于 exactness 修复，不计论文创新。**

---

# 3. 唯一主方案：TAR-DCR

## 3.1 名称与论文贡献边界

建议论文内部只保留两个方法级名称：

1. **TAR — Temporal Algebraic Re-parameterization Bridge**
2. **DCR — Decoder-wide Compositional Re-parameterization**

Full model 可写作：

> **STR-RepNet with TAR-DCR**

不要再给每个 RepConv 起一个独立“创新模块名”；`RepDW / RepConv / RepQKV` 只是 DCR 的基础算子模板。

---

# 4. A：中间模块——TAR Bridge

## 4.1 为什么这里最值得做

当前每级输入：

```text
[pre_i, post_i] → dense Conv3×3(2C_i→128)
```

它把“时相混合”和“空间 3×3 建模”绑在同一个大 dense kernel 中。

但后续 HAM/TAM 已经拥有：

- CPE 3×3 depthwise；
- SSM 前置 3×3 depthwise；
- StateFusion 大感受野 DW；
- MDTA 的 DW3×3；
- 后三级 ICSF 的 dense 3×3 + DW3×3。

因此，encoder→decoder 桥接更需要的是**明确的时相归纳偏置**，而不是再花大量 FLOPs 做一次 2C→128 的 dense 3×3。

TAR 直接把这一层分解为“训练期丰富 temporal topology，部署期单一 temporal projection”。

---

## 4.2 四级输入尺寸

由 `PATCH_SIZE=4`、VSSM-Tiny `EMBED_DIM=96` 推导，256×256 输入下四级大致为：

| 级别 | 单时相通道 | 空间尺寸 | 当前 concat 输入 |
|---|---:|---:|---:|
| F1 | 96 | 64×64 | 192 |
| F2 | 192 | 32×32 | 384 |
| F3 | 384 | 16×16 | 768 |
| F4 | 768 | 8×8 | 1536 |

统一投影到 `D=128`。

---

## 4.3 训练图

对每一级 `i`：

\[
X_1 = F_i^{pre},\quad X_2 = F_i^{post}.
\]

三条分支：

\[
Y_c = \mathrm{BN}_c\left(W_c * [X_1,X_2]\right)
\]

\[
Y_s = \mathrm{BN}_s\left(W_s * (X_1+X_2)\right)
\]

\[
Y_d = \mathrm{BN}_d\left(W_d * (X_2-X_1)\right)
\]

最终：

\[
Z_i = Y_c + Y_s + Y_d.
\]

其中：

- `W_c`: `Conv1×1(2C_i→128, bias=False)`
- `W_s`: `Conv1×1(C_i→128, bias=False)`
- `W_d`: `Conv1×1(C_i→128, bias=False)`
- 三路相加后**不再插入无法跨越的非线性**；
- 必须使用 `post-pre` 的 **signed difference**。

### 为什么不用 `abs(post-pre)`

\[
|X_2-X_1|
\]

是输入依赖非线性，无法被吸收到固定 `1×1` kernel，因此不属于严格 Structural Re-parameterization。

如果想测试 abs-diff，只能作为“不可部署等价的性能上界诊断”，不能进入主模型。

---

# 5. TAR 的严格折叠公式

## 5.1 先折 BN

任一 Conv+BN 在 eval 下：

\[
\tilde W =
\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}W,
\]

\[
\tilde b =
\beta+
\frac{\gamma}{\sqrt{\sigma^2+\epsilon}}(b-\mu).
\]

当前 Conv 可以设 `bias=False`，则 \(b=0\)。

得到：

\[
(\tilde W_c,\tilde b_c),\quad
(\tilde W_s,\tilde b_s),\quad
(\tilde W_d,\tilde b_d).
\]

将 concat branch 的输入通道一分为二：

\[
\tilde W_c=[\tilde W_{c1},\tilde W_{c2}].
\]

则：

\[
Y =
(\tilde W_{c1}+\tilde W_s-\tilde W_d)X_1+
(\tilde W_{c2}+\tilde W_s+\tilde W_d)X_2+
(\tilde b_c+\tilde b_s+\tilde b_d).
\]

因此：

\[
W_{deploy}=
[
\tilde W_{c1}+\tilde W_s-\tilde W_d,\;
\tilde W_{c2}+\tilde W_s+\tilde W_d
],
\]

\[
b_{deploy}=
\tilde b_c+\tilde b_s+\tilde b_d.
\]

部署图：

```text
cat([pre, post]) → Conv1×1(2C_i→128, bias=True)
```

不需要再训练。

---

# 6. TAR 的参数量与 FLOPs

## 6.1 当前 baseline 四级 bridge

当前四个 `OverlapPatchEmbed` 都是 dense 3×3：

\[
\sum_i (2C_i)\times128\times9
=3,317,760
\]

约 **3.318M 参数**。

256×256 下四级 analytical MAC/FLOP 量约：

**1.699G**。

## 6.2 TAR 训练图

三分支权重总量：

\[
(2C_i+C_i+C_i)\times128=4C_i\times128.
\]

四级合计含 BN 可训练参数约：

**0.740M**。

也就是说，TAR 甚至在**训练参数量**上仍显著低于当前四个 dense 3×3 bridge。

## 6.3 TAR 部署图

四级单 1×1：

约 **0.369M 参数**。

对应 analytical FLOPs：

约 **0.189G**。

因此仅 bridge 一项相对当前实现大约：

- Params：`3.318M → 0.369M`
- FLOPs：`1.699G → 0.189G`
- 约节省 **2.95M 参数**
- 约节省 **1.51G FLOPs**

这是本方案“又轻又强”最关键的复杂度来源。

---

# 7. B：DCR——Decoder-wide Compositional Re-parameterization

DCR 不是“塞一个 RepConv”，而是先做**可折叠算子图审计**，然后只对 active linear subgraph 使用统一模板。

---

## 7.1 Foldability Map

| 当前位置 | 操作 | 处理 |
|---|---|---|
| TAR bridge | temporal concat/sum/signed-diff | **严格折成单 1×1** |
| SpatialMamba `cpe1/cpe2` | DW3×3 + residual sum | **RepDW + identity kernel fold** |
| SSM `conv2d` | DW3×3 → SiLU | **只折 SiLU 前 DW block** |
| StateFusion | d=1/3/5 | **baseline 已 Rep；只做 exactness repair** |
| selective scan | 输入依赖递推 | **不可折，保留** |
| MLP `Linear→act→Linear` | 中间有 GELU | **不可跨激活串联折叠** |
| MDTA qkv `1×1` | 线性 | **RepQKV → 单 1×1** |
| MDTA qkv DW3×3 | 线性 | **RepDW → 单 DW3×3** |
| Q/K normalize | 输入依赖 | **不可折** |
| QKᵀ / Softmax / AV | 输入依赖乘法 | **不可折** |
| MDTA project_out 1×1 | 线性 | **RepPW → 单 1×1** |
| HAM `cat→1×1+BN→ReLU` | ReLU 前线性融合 | **代数多分支 fusion → 单 1×1；ReLU 保留** |
| ICSF dense 3×3 + BN | 线性 + 静态 BN(eval) | **RepConv3 → 单 3×3** |
| ICSF DW3×3 + BN | 线性 + BN | **RepDW → 单 DW3×3** |
| ICSF SE | GAP→MLP→Sigmoid→乘法 | **不可折，保留** |
| upsample + add | resize + add | 不强行卷积化 |
| main classifier | 1×1 | 保留 |
| Transformer FFN | **当前 forward 未执行** | 本轮不启用，不作为创新 |

---

# 8. DCR 基础模板 1：RepDW3

用于：

- `SpatialMambaBlock.cpe1`
- `SpatialMambaBlock.cpe2`
- `StructureAwareSSM.conv2d`
- `Attention.qkv_dwconv`
- `ICSFBlock.dwconv`
- 若未来真正启用 GDFN，则可用于 `FeedForward.dwconv`

训练：

```text
x ─ DW3×3 + BN ─┐
x ─ DW1×3 + BN ─┼─ SUM ─► y
x ─ DW3×1 + BN ─┘
```

部署：

```text
x ─ single DW3×3(bias=True) ─► y
```

折叠步骤：

1. 每个 Conv+BN 先 fold；
2. `1×3` kernel pad 到 3×3 中心行；
3. `3×1` kernel pad 到 3×3 中心列；
4. 三个 kernel/bias 求和。

\[
K_{deploy}=K_{3\times3}+
\mathrm{pad}(K_{1\times3})+
\mathrm{pad}(K_{3\times1})
\]

\[
b_{deploy}=b_3+b_{13}+b_{31}.
\]

### CPE 的 residual 还能进一步折

当前：

\[
y=x+\mathrm{DWConv}(x).
\]

若中间没有激活，则：

\[
K'_{deploy}=K_{deploy}+I_\delta,
\]

其中 \(I_\delta\) 是 depthwise 3×3 中心为 1 的 identity kernel。

因此 CPE 的“conv + residual add”部署时可以直接变成**一个 DW3×3**。

这是 DCR 里真正有价值的“跨模块/残差级组合折叠”，而不是简单把一个 RepConv 替进去。

---

# 9. DCR 基础模板 2：RepConv3

仅用于当前实际执行的 `ICSFBlock.conv1`。

训练：

```text
x ─ Conv3×3 + BN ─┐
x ─ Conv1×1 + BN ─┤
x ─ Conv1×3 + BN ─┤─ SUM → ReLU
x ─ Conv3×1 + BN ─┘
```

部署：

```text
x ─ single Conv3×3(bias=True) → ReLU
```

这里的核心不是大核，而是**固定 3×3 deploy topology 下提升训练表示**。

这与 CD-RLKNet/LKMamba-CD 的 reparameterized large-kernel 路线有本质区别：本方案不靠部署大核扩大感受野，而是保持原 3×3 的部署成本。

---

# 10. DCR 基础模板 3：RepQKV / RepPW

## 10.1 为什么不把 `1×1 qkv → DW3×3` 直接合成 dense 3×3

数学上可以，但当前单个 MDTA：

- qkv 1×1：`128→384`，约 49k weight；
- qkv DW3×3：约 3.5k weight；
- 若合成 dense 3×3：`128×384×9 ≈ 442k`。

仅这一处就会从约 52.6k 线性权重膨胀到约 442k，约 **8.4×**。

所以：

> **“算子个数变少”不等于“更轻”。这一跨层 fold 默认否决。**

---

## 10.2 RepQKV 的轻量做法

保留主 dense 1×1：

\[
QKV_{main}=W_{qkv}x.
\]

训练期增加极轻的 channel-diagonal 辅助分支：

\[
Q_{aux}=s_q\odot x,\quad
K_{aux}=s_k\odot x,\quad
V_{aux}=s_v\odot x.
\]

\[
QKV=QKV_{main}+
\mathrm{Concat}(Q_{aux},K_{aux},V_{aux}).
\]

三个 \(s\in\mathbb R^C\) 只有 `3C` 参数。

部署时把 diagonal scale 写入 q/k/v 对应输出块的 1×1 kernel 对角位置：

\[
W_{qkv}^{deploy}
=
W_{qkv}+
\mathrm{BlockDiag}(D_q,D_k,D_v).
\]

部署仍为：

```text
single Conv1×1(C→3C)
```

然后 qkv 的 DW3×3 使用前面的 `RepDW3`。

`project_out(C→C)` 同理可做：

```text
main 1×1 + diagonal scale branch → single 1×1
```

---

# 11. HAM global-local fusion 也做代数重参数化

当前：

```text
p_tam, p_tab
  → cat(256ch)
  → Conv1×1(256→128)+BN
  → ReLU
```

DCR 把 ReLU 前的线性融合扩展成：

\[
Y_c=W_c[U,V]+b_c
\]

\[
Y_s=W_s(U+V)+b_s
\]

\[
Y_d=W_d(V-U)+b_d
\]

\[
Y=Y_c+Y_s+Y_d.
\]

这里：

- \(U\)=TAM output；
- \(V\)=TAB/MDTA output。

部署公式与 TAR 完全同构：

\[
W_{deploy}=
[
W_{cU}+W_s-W_d,\;
W_{cV}+W_s+W_d
].
\]

部署仍为一个 `Conv1×1(256→128)`。

这不是新的第三个贡献，而是说明 **DCR 是一个 decoder-wide algebraic rule，而不是若干互不相关的 RepConv**。

---

# 12. FFN：必须按“当前代码实际执行图”处理

你在任务里要求覆盖“门控 FFN 的 DWConv”，但当前仓库：

```python
class TransformerBlock:
    ...
    self.ffn = FeedForward(...)

def forward(self, x):
    x = x + self.attn(self.norm1(x))
    # x = x + self.ffn(self.norm2(x))
    return x
```

因此：

- FFN 参数被注册；
- 但 forward 根本不执行；
- 改 `FeedForward.dwconv` 不会改变 F1；
- 把 FFN 打开会新增真实 inference FLOPs，且不再是单一变量。

**主方案不启用 FFN。**

如果后续你决定先把 baseline 修正为“论文预期版 HAM”，并重新从 E0 开始跑，那么：

```text
project_in → RepDW3 → split → GELU(x1)*x2 → project_out
```

只能折 `RepDW3` 本身，**不能跨过 `GELU × gate`**。

---

# 13. ICSF：保留 SE，不做“伪折叠”

当前 executable `ICSFBlock`：

```text
Conv3×3+BN → ReLU
→ DW3×3+BN → ReLU
→ GAP → 1×1 → ReLU → 1×1 → Sigmoid
→ multiply
→ residual add → ReLU
```

DCR 只替换：

- `Conv3×3+BN` → RepConv3 → single Conv3×3
- `DW3×3+BN` → RepDW3 → single DW3×3

SE 保留。

### 为什么不把 SE 折成卷积

SE 的权重：

\[
g(x)=\sigma(W_2\delta(W_1\operatorname{GAP}(x)))
\]

依赖当前输入 \(x\)。

因此不存在固定 kernel \(K\) 使：

\[
x\odot g(x)=K*x
\]

对所有输入成立。

ASR（ECCV 2024）提供的是“attention-alike / cost-free attention”的另一条设计路线，而不是证明普通 SE 可被逐样本严格等价折叠。

在你当前 `<1e-6 exact equivalence` 硬约束下，**SE 不动是正确做法**。

---

# 14. StateFusion：保留已有创新，只做转换工程化

`Spatial_Mamba.StateFusion` 已经是：

```text
train: DW3 d=1 + DW3 d=3 + DW3 d=5
deploy: one DW11
```

这与 UniRepLKNet 的 Dilated Reparam Block 思想高度接近，并且是 baseline 自带。

本论文对它只能写：

> “We retain the existing structure-aware state fusion and integrate it into a decoder-wide re-parameterization framework.”

不能写：

> “We propose a novel dilated multi-branch-to-large-kernel re-parameterization.”

本轮应做的只是：

1. 从 `self.training` 触发改为显式 `deploy` flag；
2. 新增 `get_equivalent_kernel_bias()`；
3. 新增 `switch_to_deploy()`；
4. 统一 padding；
5. 删除惰性 `_merge_weight` 缓存；
6. 做单元测试。

---

# 15. 与现有工作的实质区别

## 15.1 对 CD-RLKNet

CD-RLKNet 已经把 re-parameterized large kernel 用于变化检测，并通过 spatial/temporal fusion + large-kernel module 建模变化。

本方案不主张“CD 首次结构重参数化”，区别是：

- CD-RLKNet：核心在 **large-kernel CD module**；
- TAR-DCR：核心在 **二时相拓扑的代数折叠 + 整个 HAM decoder 可折叠算子图 + exact deploy graph**；
- TAR deployment 是 1×1 temporal projection，不依赖大核；
- decoder spatial branch 维持原 3×3/DW3×3 部署成本。

## 15.2 对 LKMamba-CD

LKMamba-CD 2026 已把 **Mamba/state-space + large kernel + CD** 结合起来，因此“把 Mamba 和 reparameterized large kernel 用于 CD”已经高度撞车。

TAR-DCR 的避碰策略：

- 不改 encoder；
- 不提出新 SSM；
- 不把大核作为新贡献；
- 主创新落在 change-specific temporal algebra + decoder-wide graph conversion。

## 15.3 对 UniRepLKNet

UniRepLKNet 的 Dilated Reparam Block：

- 训练期非膨胀大核 + 多个 dilated small kernels；
- 部署期转为单一 non-dilated large kernel。

TAR-DCR：

- StateFusion 部分承认同源；
- 新贡献不在 dilation rep；
- Temporal Rep 处理的是**二时相输入拓扑**；
- DCR 处理的是 **HAM decoder heterogeneous operator graph**。

## 15.4 对 DBB

DBB 证明 sequence conv、多尺度 conv、pooling 等复杂 branch 可以做等价重参数化。

本方案借用其“线性算子组合”的理论基础，但任务专属之处是：

- temporal concat/sum/signed-diff 的二时相代数映射；
- HAM 的 TAM/TAB 双分支同构代数 fusion；
- cost-aware 规则阻止“可折但更重”的跨层合并。

## 15.5 对 ASR

ASR 明确指出普通 attention 由于 multiplicative + input-dependent，无法直接按普通 SRP 处理。

这正好支持本方案的边界：

- SE：保留；
- Softmax attention core：保留；
- 只 reparameterize QKV/project/local conv；
- 不把“attention 删除”包装成等价折叠。

---

# 16. 为什么只推荐这一套，而否掉其他候选

## 被否候选 1：全 decoder 改成 reparameterized large-kernel

**否决原因：**

- 与 CD-RLKNet / LKMamba-CD / UniRepLKNet 高度碰撞；
- baseline `StateFusion` 已经有 d=1/3/5→11×11；
- 再扩大 kernel 容易增加 deploy FLOPs/latency；
- 创新辨识度反而下降。

## 被否候选 2：把 MDTA `1×1 qkv + DW3×3` 联合折成 dense 3×3

**否决原因：**

- 数学可折，但单 MDTA 线性权重约从 52.6k 膨胀到 442k；
- “1 个算子”换来约 8.4× kernel 参数，违反 cost-aware folding；
- 可能让 5090 latency 也更差。

## 被否候选 3：用 ASR/静态 attention 替换现有 SE 或 Softmax

**否决原因：**

- 普通 SE/Softmax 不是固定线性映射；
- 替换会改变函数，而不是把已训练模块严格等价折叠；
- 会把论文主线从“exact structural rep”拉向“attention approximation / redesign”；
- 当前阶段收益风险高于 TAR。

---

# 17. 训练 / 部署参数量与 FLOPs 估算

## 17.1 baseline

仓库 README：

- **36.08M**
- **16.26G**
- 输入：双时相 `2×3×256×256`

## 17.2 仅 TAR 的 deploy estimate

bridge：

- `3.318M → 0.369M`
- `1.699G → 0.189G`

整网粗估：

\[
36.08 - 3.318 + 0.369
\approx 33.13M
\]

\[
16.26 - 1.699 + 0.189
\approx 14.75G
\]

即：

- **约 33.1M / 14.75G**，即使完全不清理死参数也已经低于 baseline。

## 17.3 Full TAR-DCR deploy

当前代码存在不参与 forward 的注册参数：

- 8 个 `st_block_*2/*3`；
- 整个 `at_layer_11`；
- 3 个真正被调用 MDTA 内仍注册但不执行的 FFN；
- `at_layer_11` 自身还包含一套死 FFN。

按源码结构估算，死参数约 **2.97M**。

部署导出时删除这些 unreachable modules（注意：**这是工程清理，不算论文创新**），再加 TAR：

\[
36.08 - 2.97 - 3.318 + 0.369
\approx 30.16M.
\]

因此 Full 的合理预期：

- **Deploy Params ≈ 30.2M**
- **Deploy FLOPs ≈ 14.75G 或略低**
- 相对 README baseline：
  - 参数下降约 **16%**
  - FLOPs 下降约 **9%**

DCR 的 Rep branches 在 deploy 时全部合回原 kernel shape，因此不增加上述部署复杂度。

## 17.4 Full training params 粗估

去掉死参数后：

- baseline reachable ≈ `36.08 - 2.97 = 33.11M`
- bridge 从 3.318M 换成 TAR train ≈0.740M
- DCR 额外 train-only branch 约 +0.5M 量级

所以 Full training params 粗估：

**≈31.0M–31.2M**

即训练参数也不一定高于当前 baseline 注册参数。

> 最终数字必须由改完后的 `measure_params()` 和同一 `fvcore` 代码实测，本文档数字只作为设计前预算。

---

# 18. Latency / FPS 必须怎么测

不能用“算子数减少”代替 latency。

在 RSML-3 / RTX 5090 上统一：

```text
input: pre/post = 1×3×256×256
dtype: FP32
model.eval()
torch.inference_mode()
warmup: 200 iterations
measure: 1000 iterations
torch.cuda.synchronize() before/after
CUDA Event timing
```

至少报告：

- batch=1 latency(ms)
- batch=1 FPS
- batch=16 throughput
- peak allocated memory（建议）
- train graph Params
- deploy graph Params
- deploy FLOPs
- fold error

### latency 失败判据

若 Full 的 batch=1 latency 比 baseline **慢 >5%**，即使 FLOPs 更低，也要检查：

- 11×11 DW kernel 实现；
- replicate pad；
- BN 是否真正 fold；
- deploy graph 是否仍残留 branch；
- 是否错误把 qkv+DW 合成 dense 3×3。

---

# 19. 可证伪假设

## H1：TAR 是主要 F1 来源

**假设：**

Concat branch 提供时相联合自由度；Sum branch 强化不变/shared content；signed-Diff branch 强化方向性变化；三者作为不同训练参数化能改善优化，但部署函数仍落回一个普通 1×1 temporal projection。

**可证伪：**

E1 相对 E0 在 LEVIR + SYSU 两个代表数据集都没有任何有效增益，或同时明显下降。

**失败判据：**

- 两个代表数据集 F1 均下降 ≥0.3；
- 或 Full 四数据集只有 ≤1 个数据集超过 E0。

---

## H2：DCR 的收益来自训练参数化，而不是部署加容量

**假设：**

RepDW/RepConv/RepQKV 为同一个 deploy operator 提供多个训练几何/通道参数化，使局部边界、细粒度建筑变化和复杂地物更易优化。

**可证伪：**

E2/E3 增加 train branch 后 F1 不升反降，而 deploy kernel 与 E1 等复杂度。

**失败判据：**

- E2 ≤ E1 且 LEVIR/SYSU 均下降 >0.2；
- 或出现明显训练不稳定、BN running stats 异常。

---

## H3：decoder-wide 比“单个 RepConv”更有意义

**假设：**

只有 bridge 或一个 ICSF RepConv 不足以改变整体优化；统一覆盖 active foldable graph 才能形成可观察增益。

**可证伪：**

E1 已获得全部收益，而 E2/E3 无增益，说明 decoder-wide 叙事没有实验支撑。

---

## H4：部署完全不以复杂度换精度

**硬判据：**

- Deploy Params >36.08M：失败
- Deploy FLOPs >16.26G：失败
- `max_abs_error >=1e-6`：失败
- 需要 fine-tune 才能恢复 deploy 输出：失败；这已经不是严格 SRP

---

# 20. 逐文件修改清单

## 20.1 新增：`models/changedetection/models/reparam_ops.py`

新增统一 SRP 工具：

```text
fuse_conv_bn(...)
pad_1x1_to_3x3(...)
pad_1x3_to_3x3(...)
pad_3x1_to_3x3(...)
identity_dw_kernel(...)

RepDW3
RepConv3
RepQKV1x1
RepPW1x1
TemporalRep1x1
AlgebraicFuse1x1

switch_model_to_deploy(model)
```

所有模块必须提供：

```python
get_equivalent_kernel_bias()
switch_to_deploy()
```

并在 deploy 后删除 train-only branches。

---

## 20.2 修改：`Spatial_Mamba.py`

### `StateFusion`

- 新增 `deploy=False`
- 不再用 `self.training` 决定结构
- 新增 `get_equivalent_kernel_bias()`
- 新增 `switch_to_deploy()`
- 修复 padding 一致性
- deploy 后为单个 11×11 DWConv
- 不把此改动计入创新

### `StructureAwareSSM.conv2d`

当前：

```python
nn.Conv2d(d_inner,d_inner,3,groups=d_inner)
```

改为：

```python
RepDW3(d_inner)
```

SiLU 保留在 block 外，不能跨越。

### `SpatialMambaBlock.cpe1/cpe2`

改为 `RepDW3(128)`。

`switch_to_deploy()` 时，在没有非线性阻断的 CPE residual 上把 identity delta 合入 kernel。

---

## 20.3 修改：`ChangeDecoder_spatialMamba_small_ICFKFusion_DualTransformer_MDTA_four_blocks_with_one_embeding_layer.py`

### 四个 `dim_convert*`

删除：

```text
OverlapPatchEmbed(2C,128) = dense Conv3×3
```

替换为：

```text
TemporalRep1x1(C,128)
```

### `Attention`

- `qkv` → `RepQKV1x1`
- `qkv_dwconv` → `RepDW3(3*dim)`
- `project_out` → `RepPW1x1`
- normalize / temperature / Softmax / matmul 全保留

### `fuse_layer_1~4`

从：

```text
Conv1×1(256→128)+BN+ReLU
```

变为：

```text
AlgebraicFuse1x1(128,128) + ReLU
```

训练吃 `(p_tam,p_tab)` 两个 tensor，而不是先 cat 后只做单 branch。

### `ICSFBlock`

- `conv1` → `RepConv3`
- `dwconv` → `RepDW3`
- SE 原样
- residual 原样
- ReLU 原样

### 死代码

新 decoder 建议**不再注册**：

- `st_block_42/43`
- `st_block_32/33`
- `st_block_22/23`
- `st_block_12/13`

但这项只记为：

> dead-parameter cleanup / export cleanup

不要放在论文 contribution。

### `at_layer_11` / `at_layer_21`

当前 Stage IV 调用 `at_layer_21`。

为了保持 E0 的实际 forward 语义，本轮主实验**不要顺手改成 `at_layer_11`**，否则会同时改变参数共享关系，破坏唯一变量。

有两个选择：

1. **主实验保持 Stage3/4 共享 MDTA**，新代码显式写成 `shared_at_layer_21`，避免“误以为是 bug”的隐性状态；
2. 以后若想修正为四级独立 MDTA，必须单独作为 baseline-correction 实验重新跑，不能混入 TAR-DCR。

本方案选择 **1**。

---

## 20.4 修改：`MambaBCD.py`

新增：

```python
def switch_to_deploy(self):
    self.decoder.switch_to_deploy()
```

可选：

```python
def export_deploy_state_dict(...)
```

Encoder 完全不碰。

---

## 20.5 建议新增测试文件

虽然你要求修改集中在 `models/changedetection/models/`，测试最好单独放：

```text
models/changedetection/script/test_reparam_equivalence.py
```

另可新增：

```text
models/changedetection/script/profile_deploy.py
```

---

# 21. Checkpoint 与恢复兼容性

训练 checkpoint 保存的是 train graph 参数，deploy checkpoint 保存的是 folded 参数，二者不能混着无标记读取。

建议 checkpoint metadata：

```python
{
    "model": ...,
    "optimizer": ...,
    "epoch": ...,
    "best_f1": ...,
    "best_epoch": ...,
    "model_variant": "tar_dcr",
    "deploy": False,
    "reparam_version": 1
}
```

正式测试流程：

```text
load best training checkpoint
→ model.eval()
→ 保存一份 before-fold logits
→ deepcopy
→ switch_to_deploy()
→ equivalence test
→ measure deploy params/FLOPs/latency
→ 用 deploy model 做最终 test
```

不要在训练中途把主模型永久 `switch_to_deploy()`，否则无法继续训练多分支参数。

---

# 22. 部署一致性测试

## 22.1 每个 Rep block 单元测试

固定随机种子：

```python
torch.manual_seed(2333)
module.eval()
x = torch.randn(...)
y0 = module(x)

m = deepcopy(module)
m.switch_to_deploy()
m.eval()
y1 = m(x)

err = (y0-y1).abs().max().item()
assert err < 1e-6
```

对：

- TemporalRep1x1
- AlgebraicFuse1x1
- RepDW3
- RepConv3
- RepQKV1x1
- RepPW1x1
- StateFusion

逐个测试。

---

## 22.2 整模型测试

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True

model.eval()

pre  = torch.randn(2,3,256,256, device=device)
post = torch.randn(2,3,256,256, device=device)

with torch.no_grad():
    y_before = model(pre,post)

deploy_model = deepcopy(model)
deploy_model.switch_to_deploy()
deploy_model.eval()

with torch.no_grad():
    y_after = deploy_model(pre,post)

max_err = (y_before-y_after).abs().max().item()
assert max_err < 1e-6
```

### 必测集合

- CPU FP32：10 组随机输入
- GPU FP32：10 组随机输入
- shape：至少
  - 1×3×256×256
  - 2×3×256×256
- 真实数据：从四个数据集各随机取 2 对
- logits 比较，不比较 argmax

### 更严格的层级阈值

建议：

- block-level `<1e-7` 尽量做到
- model-level `<1e-6`

若 GPU 因不同 convolution kernel accumulation order 导致误差超过 1e-6：

1. 先排除 TF32；
2. 检查 BN running stats；
3. 检查 padding；
4. 检查 kernel center / dilation mapping；
5. 检查是否在 fold 后重复加 identity；
6. 若仍无法过阈值，则该实现**不能进入主论文结果**。

---

# 23. Smoke / dry run

## Smoke 1：纯随机输入

目标：

- forward 正常；
- backward 正常；
- 所有 Rep branch 有 gradient；
- Encoder 参数 `requires_grad=False`；
- `switch_to_deploy()` 正常；
- fold error 过线。

## Smoke 2：真实数据 dry run

每数据集 8~16 个样本：

- A/B/label 同步；
- label `gray>=128`；
- 一次 forward/backward；
- loss finite；
- prediction shape 正确；
- 不创建/覆盖正式 checkpoint。

## Smoke 3：resume

- train graph checkpoint 能恢复 optimizer；
- deploy checkpoint 只用于 inference；
- `deploy=False` checkpoint 不允许直接加载到 `deploy=True` class，除非先构造 train graph再 convert。

---

# 24. 消融顺序

你的约束是单 seed=2333，不做重复，因此“稳定”只能解释为：

> 训练无异常、loss/metric 无明显崩坏、最终完整 TEST block 可复现读取。

不能把单 seed 叫“统计稳定”。

## 主消融链

| 实验 | 唯一变量 | 建议数据集 | 目的 |
|---|---|---|---|
| E0 | 当前 HAM-CD baseline | 已有四数据集 Run1 | 基线 |
| E1 | E0 + TAR Bridge | LEVIR + SYSU | 验证 change-specific temporal rep |
| E2 | E1 + Spatial-DCR：TAM/MDTA-DW/ICSF RepDW/RepConv | LEVIR + SYSU | 验证 decoder-wide spatial train parameterization |
| E3 | E2 + RepQKV/RepPW + algebraic HAM fusion | LEVIR + SYSU | 验证 linear/fusion graph coverage |
| E4 Full | E3 + 显式 safe cascade fold + deploy dead-code cleanup | 四数据集 | 最终模型；F1 应与 E3 train graph 对应，fold 只改变部署结构 |

### 为什么 LEVIR + SYSU 做中间消融

- LEVIR：建筑小目标、类别不平衡，对边界和局部细节敏感；
- SYSU：变化类型更广，对 temporal/spatial fusion 更敏感。

E4 再跑 CDD/LEVIR/SYSU/WHU 四个完整主结果，避免无必要地把每个小实验扩成 20+ 次长训练。

如果你坚持“每个 ablation 都四数据集”，可以照同一顺序全部扩展，但不会增加方法论信息量。

---

# 25. 成功阈值与失败解释

## Full 成功

必须同时：

1. **至少 2 个数据集 F1 > E0**
2. Deploy Params ≤36.08M
3. Deploy FLOPs ≤16.26G
4. whole-model FP32 eval `max_abs_error <1e-6`
5. 无 deploy fine-tune
6. 最好 batch1 latency 不慢于 baseline

## 更强成功

若：

- LEVIR 接近/达到 92
- SYSU 接近/达到 84
- WHU 接近/达到 95
- CDD 接近/达到 97

同时约 30M / 14.75G，则足以支持“轻量 + 精度”的论文主叙事。

## 若 E1 失败

说明：

- 当前 dense 3×3 bridge 的空间混合很重要；
- 单纯 1×1 temporal projection 丢失的空间先验超过 temporal branch 带来的收益。

下一步不是马上加重模块，而是做唯一一个对照：

```text
TAR 1×1 → single DW3×3
```

仍保持低 FLOPs，再验证“空间因子化”是否补回精度。

这应当是失败后的 Plan-B，而不是一开始就堆进 Full。

## 若 E2 失败

说明 generic spatial Rep 在当前 HAM 上没有贡献；保留 TAR，删除无收益的 RepDW/RepConv，不要为了“decoder-wide”叙事硬留。

## 若 E3 失败

说明 QKV/fusion 参数化过强或 BN/scale 破坏 attention 优化；回退到 E2。

## 若 only 1 dataset 提升

不能写“普适提升”；只能说单数据集现象，当前主方法证据不足。

---

# 26. 当前训练协议的 P0 实验问题

`train.py` 当前把 `test_list` 同时用于：

- 每 epoch evaluation；
- 选择 `best_f1` checkpoint；
- 训练完成后的“final test”。

这意味着测试集参与模型选择。

对于**方法探索内部实验**，如果你明确冻结协议并只和相同 E0 比较，可以暂时维持一致性；但对论文正式结果，这会造成 test leakage。

### 不建议现在偷偷改

因为你要求：

> 与 `train_scripts/baseline/Run1` 完全一致。

所以本轮方法筛选若改成 val，会同时改变 baseline protocol。

### 投稿前必须做的事

最终论文版应该：

```text
train → val 选 best
best checkpoint → test 一次
```

然后 E0 与 Full 同协议重跑。

这不改变 TAR-DCR 方法本身，但决定正式结果是否可信。

---

# 27. 论文里如何写“结构重参数化严格定义”

建议定义：

\[
f_{train-eval}(x;\theta)
=
f_{deploy}(x;T(\theta))
+\epsilon,
\]

其中：

- \(T\) 是显式参数变换；
- 不进行再次优化；
- 不使用 teacher；
- 不做 post-fold finetune；
- FP32 eval 下：
  \[
  \|\epsilon\|_\infty < 10^{-6}.
  \]

### 本方案属于

- **时相结构重参数化**：TAR
- **空间异构结构重参数化**：RepDW / RepConv
- **狭义重参数化**：单 block branch folding
- **多重重参数化**：多类 Rep operator 在 decoder 共存
- **串联/组合重参数化**：CPE residual identity fold、BN fold、safe algebraic composition
- **cost-aware compositional folding**：能合但更重的 qkv→dense3×3 明确拒绝

### 本方案不属于

- KD
- teacher collapse
- ensemble collapse
- pruning-as-reparameterization
- attention approximation
- “把 SSM 变成卷积”

---

# 28. 文献定位与设计依据

## 28.1 最直接的 prior art

1. **HAM-CD**, IEEE TGRS 2026  
   Baseline；HAM=TAM+TAB，ICSF 融合。  
   DOI: https://doi.org/10.1109/TGRS.2026.3665418

2. **CD-RLKNet**, IJAEO 2024  
   已在 remote sensing change detection 中使用 re-parameterized large kernel，因此不能声称“首次把 SRP 用于 CD”。  
   DOI: https://doi.org/10.1016/j.jag.2024.104077

3. **LKMamba-CD**, PFG 2026  
   Mamba/state-space + large kernel + CD 的直接碰撞项。  
   DOI: https://doi.org/10.1007/s41064-026-00407-9

4. **UniRepLKNet**, CVPR 2024, CCF-A  
   Dilated Reparam Block：多 dilated small kernels → single non-dilated large kernel。  
   https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html

5. **Diverse Branch Block (DBB)**, CVPR 2021, CCF-A  
   sequence / multi-scale / diverse branches 的代数 folding 基础。  
   https://openaccess.thecvf.com/content/CVPR2021/html/Ding_Diverse_Branch_Block_Building_a_Convolution_as_an_Inception-Like_Unit_CVPR_2021_paper.html

6. **ASR**, ECCV 2024, CCF-A  
   明确讨论 input-dependent multiplicative attention 为什么不能直接普通 SRP，并提出 attention-alike SRP。  
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3451_ECCV_2024_paper.php

## 28.2 方案与 prior art 的一句话边界

> **Existing CD re-parameterization works primarily modify spatial large-kernel blocks, whereas TAR-DCR treats bi-temporal fusion and the executable HAM decoder as a compositional foldable operator graph: temporal concat/sum/signed-difference branches are algebraically collapsed into a single 1×1 projection, while only static linear subgraphs inside TAM, MDTA and ICSF are re-parameterized and all input-dependent SSM/SE/Softmax operators remain intact.**

---

# 29. 立即执行顺序

1. **先不要跑四数据集。**
2. 新增 `reparam_ops.py`，先只实现 `TemporalRep1x1`。
3. 随机 tensor 做 TAR 单元测试，必须 `<1e-6`。
4. 把 `StateFusion` 改为显式 `switch_to_deploy()`，先解决 inherited exactness。
5. 跑整模型随机输入 fold test；此时只改 TAR + StateFusion conversion infrastructure。
6. 做真实数据 dry run。
7. 跑 **E1：LEVIR + SYSU**。
8. E1 不明显失败后，再实现 `RepDW3 / RepConv3`。
9. 跑 **E2：LEVIR + SYSU**。
10. 再实现 `RepQKV / RepPW / AlgebraicFuse1x1`。
11. 跑 **E3：LEVIR + SYSU**。
12. 只在 E3 通过后构造 deploy export，删除 dead modules，做 Params/FLOPs/latency/equivalence。
13. 跑 **E4 Full：CDD / LEVIR / SYSU / WHU 各一次，seed=2333**。
14. 最终只从每个 `train_log.txt` 最后一个完整 TEST block 汇总正式指标。

---

# 30. 仍需补充证据

当前还缺：

1. **Run1 四数据集最终 TEST block**：GitHub 当前检索不到 `=== TEST RESULTS ===`，因此还不能建立 E0 实测基线表。
2. **baseline batch1 latency/FPS**：README 只有 Params/FLOPs。
3. **StateFusion 当前 fold 的实测误差**：必须先跑，验证 padding 问题到底造成多大误差。
4. **Encoder 是否在实际训练脚本中真的冻结参数**：你的任务要求完全冻结，但当前 `train.py` 是 `optim.AdamW(self.model.parameters(),...)`；应检查是否有显式 `requires_grad=False`。如果没有，这是另一个 P0——“不改 backbone”与“冻结 backbone 不训练”不是一回事。
5. **预训练权重加载日志**：正式跑前确认 `vssm_tiny_0230_ckpt_epoch_262.pth` 没有关键层 unexpected/missing。
6. **最终 deploy Params/FLOPs**：本文估算要由改后代码同一 fvcore 复核。

---

# 31. 最终自查

| 硬约束 | TAR-DCR |
|---|---|
| Encoder 不改 | **满足** |
| Temporal train multi-branch / deploy single 1×1 | **满足** |
| signed diff，不用 abs | **满足** |
| SSM selective scan 不伪折叠 | **满足** |
| SE 不伪折叠 | **满足** |
| Softmax attention core 不伪折叠 | **满足** |
| Conv/BN/DW/constant scale/residual 可折 | **满足** |
| deploy Params ≤36.08M | **预计约30.2M，满足** |
| deploy FLOPs ≤16.26G | **预计约14.75G，满足** |
| 不靠 teacher / KD | **满足** |
| fold 无再训练 | **满足** |
| FP32 eval `<1e-6` | **设计上可实现，但必须以单元测试和整模型测试作为硬门槛** |
| 与 CD-RLKNet/LKMamba-CD 区分 | **有明确差异** |
| baseline StateFusion 不重复包装 | **满足** |
| FFN 当前未执行这一事实被处理 | **满足** |
| dead-code 参数不包装成创新 | **满足** |

---

# 最终结论

**本轮不要做“大核 Mamba 再升级”，也不要把 SE/Softmax 强行静态化。最值得实现的是 TAR-DCR：先用 Temporal Algebraic Rep 把四级二时相 bridge 从 3×3 dense concat embedding 改成 train-time Concat/Sum/signed-Diff 三拓扑、deploy-time 单 1×1；再用统一的 decoder-wide operator policy 覆盖 TAM 局部 DW、MDTA QKV/DW/project、HAM fusion、ICSF conv/DW，并只在没有输入依赖非线性的边界内组合折叠。按当前代码结构估算，Full deploy 可从 36.08M / 16.26G 降到约 30.2M / 14.75G；真正决定论文是否成立的，是 E1→E3 在 LEVIR/SYSU 上是否能证明训练参数化带来 F1 收益，以及整模型 fold 是否严格通过 `<1e-6`。**

