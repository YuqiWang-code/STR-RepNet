# STR-RepNet Run5：回退 Run2 后的改进方案与最小消融设计

> 审查基准：GitHub `YuqiWang-code/STR-RepNet` 当前 `main`（2026-09-27，HEAD=`fbc686fe4d1c9b38082238b0e4bb5a974015dfb4`）+ 用户提供的 Run2 完整代码/指标快照 `models_and_metrics_TAR-DCR_Run2(2).txt`。  
> 任务：CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256，全监督二值变化检测。  
> 硬约束：部署参数量/FLOPs不增加；训练期新增结构必须可折叠或可删除；部署图保持单路径；正式结果只认最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===`。

---

## 0. 结论先行

Run4 IBAS 应按预注册协议判定 **FAIL**，不做 λ sweep、不再扩大边界头，也不再沿边界先验方向加码。Run3（算子级边缘基）与 Run4（监督级边界正则）的结果可以保留为论文中的完整负/中性消融：当前证据不支持“LEVIR 的主要缺口来自边界表征不足”。

Run5 建议：

1. **方法基线回到 Run2 `full_last2`，但不要把 Git 历史 reset 回 Run2。**
2. 在 Run5 前先做两个零部署成本诊断：
   - `encoder_train=full`：测 early encoder adaptation 的上界；
   - A/B swap sensitivity：测当前 TAR 的时序顺序偏置。
3. Run5 主方法选择 **BOTR（Bi-Order Temporal Re-parameterization，双顺序时相结构重参数化）**：
   - 在 TAR 中增加训练期 `[Q,P]` reverse-concat 分支；
   - 通过输入通道置换与 BN folding，部署时严格折叠回原始单个 `1×1`；
   - 部署 Params/FLOPs 与 Run2 完全一致。
4. 最小实验组只保留：
   - A0：Run2 `full_last2` 现有锚点（不重跑）；
   - D0：`full_encoder` 上界诊断；
   - C1：`swap_only` 必要控制；
   - M1：`BOTR` 主实验。
5. 第一阶段只跑 LEVIR；M1 过预注册判据后再扩 WHU → SYSU → CDD。

---

# 1. 当前证据：Run2 仍是唯一合理锚点

Run2 `full_last2`：

| Dataset | Recall | Precision | OA | F1 | IoU | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| CDD | 0.9849 | 0.9835 | 0.9961 | **0.9842** | 0.9689 | 0.9820 |
| LEVIR | 0.9032 | 0.9260 | 0.9914 | **0.9144** | 0.8424 | 0.9099 |
| SYSU | 0.7956 | 0.8775 | 0.9256 | **0.8345** | 0.7160 | 0.7866 |
| WHU | 0.9435 | 0.9595 | 0.9962 | **0.9514** | 0.9074 | 0.9494 |

Run2 Macro F1：

\[
(0.9842+0.9144+0.8345+0.9514)/4=0.921125
\]

即 **92.11%**。

相对 HAM-CD baseline：

- CDD：98.42 vs 98.79，差 -0.37 pp；
- LEVIR：91.44 vs 92.11，差 -0.67 pp；
- SYSU：83.45 vs 82.99，高 +0.46 pp；
- WHU：95.14 vs 95.00，高 +0.14 pp。

所以当前真正需要追的是 **LEVIR，其次 CDD**；SYSU/WHU 不应为了追 LEVIR 而明显退化。

---

# 2. Run4 的正式处理：结论保留，方法退出

Run4 IBAS：

| 数据集 | Run4 | Run2 锚点 | ΔF1 |
|---|---:|---:|---:|
| LEVIR | 0.9145 | 0.9144 | +0.01 pp |
| CDD | 0.9839 | 0.9842 | -0.03 pp |
| WHU | 0.9508 | 0.9514 | -0.06 pp |
| SYSU | 0.8324 | 0.8345 | -0.21 pp |

LEVIR：

- F1 0.9145 < 0.9175；
- IoU 0.8424 < 0.8475；
- Precision 0.9269 ≥ 0.9240，但 Recall 0.9032 → 0.9023；
- F1 低于失败线 0.9159。

因此应写成：

> IBAS 达到了“抑制误检”的局部目标，但未改善整体召回，最终 F1/IoU 未达到预注册阈值；按协议停止，不进行 λ sweep。

Run3 + Run4 的科研价值仍可保留：

> 空间边界先验分别在“算子级”和“监督级”被验证，均未给出判据级净增益，因此下一轮不再围绕边界继续堆叠。

---

# 3. 回退建议：回退“活跃方法代码”，不回退 Git 历史

## 3.1 不建议

不要：

```bash
git reset --hard 65958b0
git push -f
```

原因：

- Run3/Run4 是有价值的负结果；
- README、Run3/Run4 快照、Excel、设计文档都应该保留；
- 后续论文的研究演进和失败消融需要可追溯。

## 3.2 推荐做法

先新建 Run5 分支：

```bash
cd /home/yqwang/projects/STR-RepNet
git status
git switch -c run5-botr
```

确认工作区干净后，只恢复 Run2 的**活跃方法主路径**：

```bash
git restore --source=65958b076b286a5361d98b1fb08dc01f9abfffb9 -- \
  models/changedetection/models/STRRepNet.py \
  models/changedetection/models/dcr_decoder.py \
  models/changedetection/models/reparam.py \
  models/changedetection/models/tar.py \
  models/changedetection/script/train.py
```

保留：

```text
train_scripts/TAR-DCR/Run3/
train_scripts/TAR-DCR/Run4/
docs/temporary/models_and_metrics_TAR-DCR_Run3.txt
docs/temporary/models_and_metrics_TAR-DCR_Run4.txt
docs/temporary/STR-RepNet_Run4_IBAS_训练期内侧边界辅助监督方案.md
README.md 中 Run3 / Run4 的历史结果
analyse/
docs/experiment_metrics.xlsx
```

`boundary_loss.py` 不必删除；Run5 不 import、不调用即可。

### 不要从 Run3 / Run4 checkpoint 接着训练

Run5 所有正式组都从相同 VMamba pretrained 权重重新开始；否则会混入额外训练预算。

---

# 4. Run5 前必须先修的 P0 / P2 问题

## 4.1 P0：Run2 的 FP64 folding 实现与注释不一致

Run2 `reparam.py` 注释声称：

> all kernel/bias composition done in FP64, cast to FP32 once at the end

但 `fold_conv_bn()` 实际在单个分支 fold 后就做了：

```python
W = (...double...).float()
b = (...double...).float()
```

因此多分支后续求和仍是 FP32。

应改成：

```python
def fold_conv_bn(weight, bias, bn):
    C = weight.shape[0]
    eps = bn.eps if bn.eps is not None else 1e-5

    gamma = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()

    std = torch.sqrt(var + eps)
    t = gamma / std

    bd = (
        bias.detach().double()
        if bias is not None
        else torch.zeros(
            C,
            dtype=torch.float64,
            device=weight.device,
        )
    )

    W = weight.detach().double() * t.view(C, 1, 1, 1)
    b = beta + (bd - mu) * t
    return W, b
```

同时保证：

- `W2 @ W1`
- diag 分支
- identity kernel
- BOTR reverse branch
- 所有 kernel/bias 加和

都保持 `float64`，直到最终写入 deploy Conv 时才 cast。

### 重要说明

这能减少 folding 数值误差，但不能先验保证 RTX 5090 上“多分支 FP32执行”与“单卷积 FP32执行”的 whole-model max error 必然 <1e-6，因为两张计算图的浮点累加顺序不同。

因此 Run5 必须同时记录：

1. kernel/bias algebraic error；
2. block output error；
3. whole-model output error；
4. argmax disagreement。

硬目标仍保留 `<1e-6`，但如果 GPU FP32 仍停留在 `1e-5`，论文不能直接写“严格数值等价 <1e-6”。

## 4.2 P2：指标表必须拆开 train / deploy params

历史 TXT 中 `Params(M)=29.570` 很容易抓到：

```text
[TOTAL-TRAIN-GRAPH-PARAMS]
```

而不是：

```text
[DEPLOY-PARAMS]
```

Run5 前把汇总字段改成：

```text
TrainGraphParams(M)
TrainableParams(M)
DeployParams(M)
DeployFLOPs(G)
ReparamErr
```

论文复杂度只使用 deploy 数值。

---

# 5. Run5 前置诊断：先确认瓶颈，而不是直接赌模块

新增：

```text
analyse/diagnose_run2_levir.py
```

读取 Run2 `full_last2` 最佳 checkpoint，只做评估，不改权重。

## 5.1 A/B swap sensitivity

分别评估：

\[
f(A,B),\qquad f(B,A)
\]

记录：

```text
F1_AB
F1_BA
Delta_F1_swap
pixel_disagreement_rate
mean_abs_logit_difference
```

建议 BOTR 强动机阈值：

```text
|F1_AB - F1_BA| >= 0.20 pp
或
pixel_disagreement_rate >= 0.5%
```

若低于该阈值，BOTR 仍可跑，但应降低优先级，并在论文中避免夸大“时序不对称是主要问题”。

## 5.2 小目标 / 小连通域 Recall 分析

不要先假设 LEVIR 一定是“小目标问题”。

对训练集 GT change mask 做连通域面积统计，用训练集面积分位数定义 small / medium / large，而不是拍脑袋写固定像素阈值。

测试集输出：

```text
small / medium / large component recall
missed component count
FN pixels by size bin
```

如果 FN 的确集中在 small bin，再为下一轮设计高分辨率保真模块。

---

# 6. 候选方向比较

| 候选 | 部署开销 | 与当前证据关系 | 创新性 | Run5 决策 |
|---|---:|---|---|---|
| `encoder_train=full` | +0 | 检验 stage1/2 适配上界 | 低，训练协议 | 做诊断 |
| `temporal_swap_prob=0.5` | +0 | 直接检验 A/B 顺序偏置 | 低~中 | 做必要对照 |
| size-aware / Tversky loss | +0 | 可能提高 Recall | 低，易沦为调 loss | 暂不做主线 |
| multi-scale inference / TTA | 实际 FLOPs↑ | 可诊断尺度敏感性 | 低 | 否决为主方法 |
| **BOTR** | **+0** | 当前 TAR 使用有序 concat + signed diff | **高，属结构重参数化** | **主方案** |

---

# 7. Run5 主方案：BOTR

## 7.1 名称

**BOTR — Bi-Order Temporal Re-parameterization**

中文：

**双顺序时相结构重参数化**

现阶段不要写“首次提出”，正式论文前再做系统文献查重。

## 7.2 Run2 TAR

对第 \(i\) 个尺度：

\[
P_i=E_i(A),\qquad Q_i=E_i(B)
\]

Run2：

\[
Y_i=
BN_c(W_c[P_i,Q_i])
+
BN_s(W_s(P_i+Q_i))
+
BN_d(W_d(Q_i-P_i))
\]

之后：

\[
T_i=\mathrm{RepLocalBlock}(\mathrm{SiLU}(Y_i))
\]

当前存在三个事实：

- concat 显式使用有序 `[P,Q]`；
- diff 使用 signed `Q-P`；
- Run2 正式脚本 `temporal_swap_prob=0.0`。

而 binary change label 对 A/B 交换语义不变。

## 7.3 BOTR 训练图

增加一个 reverse-concat 训练分支：

\[
Y_i=
BN_c(W_c[P_i,Q_i])
+
BN_r(W_r[Q_i,P_i])
+
BN_s(W_s(P_i+Q_i))
+
BN_d(W_d(Q_i-P_i))
\]

其中：

```text
Wc : Conv1x1(2Cin -> Cout)
Wr : Conv1x1(2Cin -> Cout)
Ws : Conv1x1(Cin -> Cout)
Wd : Conv1x1(Cin -> Cout)
```

每支独立 BN。

### 初始化

```python
nn.init.zeros_(self.proj_r.weight)
```

BN 使用默认 gamma=1, beta=0。

不要再额外给 reverse branch 加一个零初始化 gate，否则形成“双零初始化”，容易让分支学习过慢。

---

# 8. BOTR 的严格代数折叠

fold 后：

\[
\hat W_c,\hat b_c=\mathrm{Fold}(W_c,BN_c)
\]

\[
\hat W_r,\hat b_r=\mathrm{Fold}(W_r,BN_r)
\]

\[
\hat W_s,\hat b_s=\mathrm{Fold}(W_s,BN_s)
\]

\[
\hat W_d,\hat b_d=\mathrm{Fold}(W_d,BN_d)
\]

将：

\[
\hat W_c=[W_{cP}\mid W_{cQ}]
\]

reverse branch 的输入顺序是 `[Q,P]`：

\[
\hat W_r=[W_{rQ}\mid W_{rP}]
\]

转换回 deploy 输入 `[P,Q]`：

\[
W_P=
W_{cP}
+
W_{rP}
+
W_s
-
W_d
\]

\[
W_Q=
W_{cQ}
+
W_{rQ}
+
W_s
+
W_d
\]

\[
b=b_c+b_r+b_s+b_d
\]

最终：

\[
W_{deploy}=[W_P\mid W_Q]
\]

仍是：

```python
nn.Conv2d(
    2 * in_channels,
    out_channels,
    kernel_size=1,
    bias=True,
)
```

所以：

\[
Params_{deploy}^{BOTR}=Params_{deploy}^{Run2}
\]

\[
FLOPs_{deploy}^{BOTR}=FLOPs_{deploy}^{Run2}
\]

关键区别：

> BOTR 不是双向推理；`[Q,P]` 只存在训练态，部署时通过通道置换被吸收到同一个 `1×1`。

---

# 9. 与已有工作的区别

2024 IEEE TGRS 已有工作讨论 temporal-symmetric representations，说明“输入顺序一致性”本身不是无人研究的空白。

BOTR 不应把“temporal symmetry”本身当原创点，而应强调：

1. 双顺序只存在训练图；
2. 不需要 A/B 双向 inference；
3. 不保留额外 temporal branch；
4. reverse-order full projection 通过通道置换 + BN folding 吸收到单 `1×1`；
5. 与 TAR/DCR 统一为“训练期过参数化、部署期单路径”的结构重参数化框架。

---

# 10. 逐文件修改清单

## 10.1 `models/changedetection/models/tar.py`

### `TemporalRep1x1`

新增：

```python
def __init__(
    self,
    in_channels,
    out_channels,
    use_aux=True,
    use_reverse_aux=False,
    deploy=False,
):
```

训练态：

```python
if self.use_reverse_aux:
    self.proj_r = nn.Conv2d(
        2 * in_channels,
        out_channels,
        1,
        bias=False,
    )
    self.bn_r = nn.BatchNorm2d(out_channels)
    nn.init.zeros_(self.proj_r.weight)
```

forward：

```python
y = self.bn_c(
    self.proj_c(torch.cat([P, Q], dim=1))
)

if self.use_reverse_aux:
    y = y + self.bn_r(
        self.proj_r(torch.cat([Q, P], dim=1))
    )

if self.use_aux:
    y = y \
        + self.bn_s(self.proj_s(P + Q)) \
        + self.bn_d(self.proj_d(Q - P))
```

`get_equivalent_kernel_bias()`：

- fold `proj_r + bn_r`；
- 把 reverse branch 的两半通道交换回 `[P,Q]`；
- 与 concat/sum/diff 一起 FP64 累加。

`switch_to_deploy()` 删除：

```text
proj_r
bn_r
```

`branch_stats()` 增加：

```text
reverse_concat
```

## 10.2 `TARStage`

新增：

```python
use_reverse_aux=False
```

传给 `TemporalRep1x1`。

## 10.3 `MultiScaleTAR`

首轮 M1 建议四尺度统一开启 BOTR。

不要第一轮同时做：

```text
stage1-only
stage1+stage2-only
learnable scale gate
```

BOTR 通过后，再做浅层/全尺度消融。

## 10.4 `models/changedetection/models/STRRepNet.py`

新增：

```python
use_botr=False
```

传给：

```python
self.tar = MultiScaleTAR(
    ...,
    use_reverse_aux=use_botr,
)
```

DCR 不改。

## 10.5 `models/changedetection/script/train.py`

新增：

```python
parser.add_argument(
    "--use_botr",
    type=int,
    default=0,
)
```

构建模型时：

```python
use_botr=bool(self.args.use_botr)
```

最终 TEST RESULTS 建议新增：

```text
[BOTR] 0/1
[ENCODER-TRAIN] frozen/last2/full
[TEMPORAL-SWAP-PROB] 0.0/0.5
```

## 10.6 `smoke_test.py`

必须支持：

```text
--encoder_train
--use_botr
```

检查：

- BOTR forward；
- backward；
- `proj_r` gradient 非 None；
- last2 stage3/4 有梯度；
- stage1/2 frozen；
- deploy 后无 `proj_r/bn_r`；
- deploy Params 与 Run2 一致；
- deploy FLOPs 与 Run2 一致；
- fold error。

## 10.7 `test_reparam_equivalence.py`

增加：

```text
TemporalRep1x1(use_reverse_aux=True)
TARStage(use_reverse_aux=True)
MultiScaleTAR(use_reverse_aux=True)
STRRepNet(use_botr=True)
```

并统一 docstring 与实际 assert threshold，不能再出现“文档写 <1e-6、实际 assert <1e-4”。

---

# 11. `train_scripts/TAR-DCR/Run5` 最小实验组

建议目录：

```text
train_scripts/
└── TAR-DCR/
    └── Run5/
        ├── README.md
        ├── D0_full_encoder/
        │   └── train_LEVIR-CD-256.sh
        ├── C1_swap_only/
        │   └── train_LEVIR-CD-256.sh
        └── M1_BOTR/
            ├── train_LEVIR-CD-256.sh
            ├── train_WHU-CD-256.sh
            ├── train_SYSU-CD-256.sh
            └── train_CDD-CD-256.sh
```

后 3 个 M1 四数据集脚本可以先写好，但只有 LEVIR 过线后才启动。

---

# 12. A0：Run2 `full_last2` 锚点

不重新跑 300 epoch。

配置：

```text
rep_mode=full
encoder_train=last2
use_residual=1
temporal_swap_prob=0.0
use_botr=0
seed=2333
```

LEVIR：

```text
Recall    = 0.9032
Precision = 0.9260
F1        = 0.9144
IoU       = 0.8424
```

只有 rollback 后 smoke 无法复现原始结构 / 参数量时，才重跑 A0。

---

# 13. D0：full encoder upper bound

用途：

> 不是创新实验，只回答“LEVIR 剩余 0.67pp 是否主要来自 stage1/2 没有适配”。

配置：

```text
rep_mode=full
encoder_train=full
encoder_lr_ratio=0.1
use_residual=1
temporal_swap_prob=0.0
use_botr=0
seed=2333
epochs=300
```

只跑 LEVIR。

解释：

- ΔF1 ≥ +0.30 pp：early encoder adaptation 有明显作用；
- ΔF1 < +0.15 pp：后续不再把主要精力投入 encoder；
- +0.15~0.30 pp：中性。

即使成功，也不能把 `encoder_train=full` 包装成论文创新。

---

# 14. C1：swap-only 必要对照

配置：

```text
rep_mode=full
encoder_train=last2
use_residual=1
temporal_swap_prob=0.5
use_botr=0
seed=2333
epochs=300
```

只跑 LEVIR。

目的：

> 判断“普通 A/B 交换增强”是否已经足够解决顺序偏置。

这组对于解释 BOTR 是必要的。

---

# 15. M1：BOTR 主实验

配置：

```text
rep_mode=full
encoder_train=last2
use_residual=1
temporal_swap_prob=0.0
use_botr=1
seed=2333
epochs=300
batch_size=16
learning_rate=1e-4
weight_decay=5e-4
lovasz_weight=2.0
encoder_lr_ratio=0.1
```

第一阶段只跑 LEVIR。

不要首轮组合：

```text
BOTR + swap
BOTR + full encoder
BOTR + IBAS
BOTR + Edge-Basis
BOTR + new loss
```

否则无法归因。

---

# 16. Run5 预注册判据

Run2 LEVIR：

```text
F1        0.9144
Recall    0.9032
Precision 0.9260
IoU       0.8424
```

## M1 PASS

同时满足：

```text
F1        >= 0.9175
Recall    >= 0.9065
Precision >= 0.9230
IoU       >= 0.8475
```

## WEAK / 中性

```text
0.9159 <= F1 < 0.9175
```

只记录，不马上四数据集扩展，也不做 branch-weight sweep。

## FAIL

```text
F1 < 0.9159
```

或者：

```text
Recall 上升，但 Precision 下降 > 0.40 pp
```

则停止 BOTR，不再继续加 gate / 更多 reverse branches 来“救”。

---

# 17. C1 与 M1 的归因规则

如果：

```text
C1 >= M1 - 0.10 pp
```

说明 BOTR 的主要收益可能被普通 swap augmentation 解释，结构创新价值较弱。

如果：

```text
M1 - C1 >= 0.20 pp
```

且 M1 达到 PASS，则更支持：

> 双顺序训练结构本身提供了超出数据增强的优化收益。

---

# 18. M1 过线后的四数据集扩展

顺序：

```text
LEVIR
→ WHU
→ SYSU
→ CDD
```

四数据集最终接受标准建议：

```text
Macro F1 >= 92.25%
LEVIR F1  >= 91.75%
任一数据集退化不得 > 0.15 pp
至少 3/4 数据集非负增益
```

Run2 macro：

```text
92.1125%
```

如果 BOTR 只在 LEVIR 提升，而 SYSU/CDD 明显下降，只能描述为 dataset-specific effect，不能称普适提升。

---

# 19. 训练脚本关键参数模板

`M1_BOTR/train_LEVIR-CD-256.sh`：

```bash
--rep_mode full \
--use_residual 1 \
--use_botr 1 \
--encoder_train last2 \
--encoder_lr_ratio 0.1 \
--temporal_swap_prob 0.0 \
--learning_rate 1e-4 \
--weight_decay 5e-4 \
--lovasz_weight 2.0 \
--epochs 300 \
--batch_size 16 \
--seed 2333
```

checkpoint：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run5/M1_BOTR/LEVIR-CD-256/
```

log：

```text
/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run5/M1_BOTR/LEVIR-CD-256/train_log.txt
```

控制组：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run5/D0_full_encoder/LEVIR-CD-256/
/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run5/C1_swap_only/LEVIR-CD-256/
```

对应 outputs 使用同样目录结构。

三个实验绝不能共享 `last.pth`。

---

# 20. Checkpoint / resume 兼容性

Run2 checkpoint 不能作为 BOTR 正式训练的 resume 起点，因为 BOTR 新增：

```text
proj_r.*
bn_r.*
```

M1 应：

- 从同一 VMamba pretrained 权重开始；
- TAR/DCR 重新初始化；
- seed 2333；
- 训练预算 300 epoch 一致。

同一个 M1 目录中的 `last.pth` 可以正常断点续训。

不要在：

```text
M1_BOTR
C1_swap_only
D0_full_encoder
```

之间互相 resume。

---

# 21. Smoke / dry run / deploy 验收

## 21.1 随机 smoke

真实训练前必须检查：

```text
[1] model build
[2] forward
[3] backward
[4] BOTR reverse branch grad != None
[5] last2 stage3/4 grad != None
[6] stage1/2 requires_grad=False
[7] switch_to_deploy
[8] deploy graph 无 proj_r / bn_r
[9] deploy params 与 Run2 一致
[10] deploy FLOPs 与 Run2 一致
[11] fold error 记录
```

## 21.2 真实 LEVIR dry run

使用独立路径：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/dry_run/Run5_BOTR_LEVIR/
/home/yqwang/outputs/STR-RepNet/dry_run/Run5_BOTR_LEVIR/
```

跑 1~2 epoch 检查：

- label 仍按 `gray>=128`；
- A/B/label 几何增强同步；
- loss 无 NaN；
- `proj_r.weight` 从 0 开始后 norm 增长；
- checkpoint 可 resume；
- deploy 后 argmax 与 train graph 一致。

---

# 22. Run5 日志新增诊断

建议每 20 或 50 epoch 记录：

```text
[BOTR-STATS]
stage1 reverse_concat_norm=...
stage2 reverse_concat_norm=...
stage3 reverse_concat_norm=...
stage4 reverse_concat_norm=...
stage1 concat_norm=...
...
```

训练结束记录：

```text
[BOTR-RATIO]
stage1 ||Wr|| / ||Wc|| = ...
...
```

这样 BOTR 若失败，可以区分：

1. reverse branch 根本没学起来；
2. reverse branch 学起来了，但任务不需要；
3. 某些尺度有效、某些尺度干扰。

---

# 23. 为什么本轮不先做“小目标 loss”

Run4 已证明：

> 更多局部监督并不自动转化为 F1 提升。

若现在直接上：

```text
Tversky
Focal
small-object weighted CE
component-aware loss
```

很容易变成“换 loss + 调权重”的迭代，偏离结构重参数化主线。

正确顺序：

1. 先做 size-stratified error profile；
2. 若 small FN 确实占主要部分；
3. 再设计可折叠的高分辨率结构机制；
4. 不先靠 loss 猜。

---

# 24. 为什么不把 multi-scale inference 作为 Run5

多尺度测试 / TTA 会增加实际推理 FLOPs 与 latency。

它可以作为离线诊断：

> 如果更高测试尺度显著改善 LEVIR Recall，说明模型存在尺度/小目标敏感性。

但不能作为论文最终“零部署增量”方案，也不应该进入 Run5 主结果。

---

# 25. Run5 失败后的预设决策

## 情况 A：D0 full encoder 明显提升，BOTR 不提升

结论：

> 剩余瓶颈更偏 encoder domain adaptation。

后续可把 full fine-tune 当训练协议，但不能包装成创新。

## 情况 B：C1 swap-only 提升，M1 不超过 C1

结论：

> 顺序问题存在，但普通数据增强足够。

停止 BOTR。

## 情况 C：M1 明显超过 C1

这是最有价值的结果：

> 双顺序结构过参数化比普通 temporal swap 更有效。

再扩四数据集。

## 情况 D：D0 / C1 / M1 都无明显提升

说明：

- boundary 基本排除；
- temporal order 基本排除；
- full encoder 也不是主要上界。

下一轮再进入：

> **高分辨率 / 小目标信息保真 + budget-neutral decoder channel allocation**

而不是继续给 TAR、edge、loss 叠模块。

---

# 26. 文献定位

Run5 只需要两类文献支撑动机：

### Temporal symmetry

2024 IEEE TGRS 已有工作明确讨论 bitemporal change detection 的 temporal-symmetric representations，因此“输入顺序一致性”是成立的问题背景。

BOTR 的创新点不能写成“首次发现 temporal symmetry”，而应写成：

> 双顺序 full projection 只用于训练，通过代数通道置换与结构重参数化折叠到单个 temporal `1×1`，不增加部署路径。

### Structural re-parameterization

可用 2024 CVPR 的 RepViT 等现代轻量网络作为结构重参数化/轻量部署背景，但 BOTR 不是照搬其 block，而是把 re-parameterization 引入二时相顺序建模。

---

# 27. Git 提交前检查

实现完成后：

```bash
git status
git diff --stat
git diff --cached
```

确认不提交：

```text
*.pth
*.pt
outputs/
dataset/
cache/
__pycache__/
token / secret
```

在提交 Run5 代码前，必须先完成：

```text
smoke
real-data dry run
deploy equivalence
```

---

# 28. 立即执行顺序

1. 保留当前 `main=fbc686f`；新建 `run5-botr` 分支。
2. 只恢复 Run2 的活跃方法主路径到 `65958b0`；保留 Run3/Run4 历史结果。
3. 修正真正的 FP64 folding；拆开 train/deploy params 统计。
4. 跑 rollback smoke，确认 Run2 `full_last2` 结构与 deploy Params/FLOPs。
5. 新增 `diagnose_run2_levir.py`：
   - A/B swap sensitivity；
   - size-stratified component recall。
6. 实现 BOTR，只改 temporal rep，不碰 DCR、不碰 loss、不碰 boundary。
7. 跑 BOTR block/whole-model folding smoke。
8. LEVIR 真实数据 1~2 epoch dry run。
9. 启动三组 LEVIR：
   - D0 `full_encoder`
   - C1 `swap_only`
   - M1 `BOTR`
10. 按预注册判据判断，不做临时 sweep。
11. M1 通过后才扩 WHU → SYSU → CDD。
12. 全部完成后再更新 README / Excel / Run5 snapshot / `_SUMMARY.md`。

---

# 29. 仍需补充证据

当前可以确认：

- Run2 `full_last2` 仍是四数据集最稳主锚点；
- Run3 Edge-Basis 应停止；
- Run4 IBAS 按预注册协议应停止；
- 当前 TAR 有有序 concat 与 signed diff；
- 数据 loader 已支持 temporal swap，但 Run2 正式实验未启用；
- BOTR 可在代数上折叠回原始单 `1×1` temporal projection。

仍未确认：

1. Run2 LEVIR 的 A/B swap disagreement 到底多大；
2. LEVIR 的 FN 是否主要集中在小连通域；
3. `encoder_train=full` 的真实上界；
4. 修正 FP64 composition 后，RTX 5090 FP32 whole-model fold error 是否能达到 `<1e-6`；
5. BOTR reverse branch 是否在训练中形成有效非零贡献。

因此 Run5 的核心不是“再赌一个模块”，而是用 **D0 / C1 / M1** 三个最小实验，把：

\[
\text{encoder适配}
\quad/\quad
\text{数据级时序对称}
\quad/\quad
\text{结构级时序重参数化}
\]

三个假设拆开验证。

若 BOTR 成功，论文主线会比 Run3/Run4 更统一：

\[
\text{TAR/BOTR temporal re-param}
+
\text{DCR decoder-wide re-param}
\rightarrow
\text{single-path deployment}
\]

若 BOTR 失败，也能干净排除“边界 + 时序顺序”两个方向，再进入有诊断证据支持的高分辨率小目标设计。
