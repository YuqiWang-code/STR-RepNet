# STR-RepNet Clean TAR-DCR：`models/` 全量重构实施方案
## 只保留 VMamba-Tiny Siamese Encoder；中间桥接与 Decoder 全部重新设计

> **目标版本**：Clean TAR-DCR  
> **保留**：VMamba / VSSM-Tiny Siamese Encoder，以及 `vssm_tiny_0230_ckpt_epoch_262.pth` 预训练权重加载方式  
> **删除**：HAM-CD 原 Decoder、Spatial-Mamba Decoder、MDTA、ICSF、旧 `MambaBCD.py` 等 baseline 解码代码  
> **新建**：完全独立的 TAR 中间桥接 + DCR Decoder  
> **原则**：所有新增可学习线性层都按结构重参数化设计；训练图多分支，部署图逐模块收缩为单路径；不声称激活/插值等非线性可折叠  
> **硬门槛**：FP32 eval fold 前后主 logits `max_abs_error < 1e-6`

---

# 0. 这次与上一版最大的区别

这一次**不再“在 HAM-CD decoder 上打补丁”**。

最终 working tree 中，`models/changedetection/models/` 不再保留：

```text
HAM Decoder
Spatial_Mamba Decoder
MDTA
ICSF
旧 MambaBCD wrapper
旧 decoder utils
```

而是只留下：

```text
VMamba-Tiny Encoder
+
全新的 TAR Bridge
+
全新的 DCR Top-down Decoder
+
全新的 STRRepNet wrapper
```

也就是说，最终模型是：

```text
A/B
  ↓
Shared VMamba-Tiny Encoder
  ↓
F1/F2/F3/F4
  ↓
TAR Multi-scale Temporal Bridge
  ↓
DCR Top-down Decoder
  ↓
1×1 Change Head
  ↓
Binary Change Logits
```

这才是一个干净的 **STR-RepNet / TAR-DCR**，而不是 HAM-CD decoder 的改名版本。

---

# 1. 先给最终网络结构

## 1.1 Encoder：唯一保留的旧模块

保留：

```text
Backbone_VSSM
```

输入：

```text
pre image  : B×3×256×256
post image : B×3×256×256
```

权重共享 Siamese：

```text
pre  ─┐
      ├── Shared VMamba-Tiny
post ─┘
```

四级输出按当前 VSSM-Tiny：

```text
F1: C=96   H=W=64
F2: C=192  H=W=32
F3: C=384  H=W=16
F4: C=768  H=W=8
```

默认统一 decoder width：

```text
D = 160
```

为什么选 160：

- 128 更轻，但在完全移除 HAM 的 SSM + MDTA 后容量略保守；
- 192 仍然很轻，但高分辨率点卷积计算继续增加；
- 160 是本方案唯一推荐值，在“恢复 decoder 表达能力”和“显著低于 baseline 复杂度”之间更平衡。

不要同时开 128/160/192 三档调参。

---

# 2. 全局训练图

```text
PRE  ─────► Shared Frozen VMamba-Tiny ─────► P1  P2  P3  P4
                                              │   │   │   │
POST ─────► Shared Frozen VMamba-Tiny ─────► Q1  Q2  Q3  Q4
                                              │   │   │   │
                                              ▼   ▼   ▼   ▼
                      ┌───────────────────────────────────────────┐
                      │         Multi-scale TAR Bridge            │
                      │                                           │
                      │  each level i:                            │
                      │                                           │
                      │  [Pi,Qi] ── 1×1 main ───────┐            │
                      │  Pi+Qi   ── 1×1 sum  ───────┼─ SUM       │
                      │  Qi-Pi   ── 1×1 diff ───────┘            │
                      │                  │                        │
                      │                BN + SiLU                  │
                      │                  │                        │
                      │          RepLocalBlock ×1                 │
                      │                  │                        │
                      │             Ti : 160 ch                   │
                      └──────┬────────┬────────┬────────┬─────────┘
                             T1       T2       T3       T4
                              │        │        │        │
                              │        │        │        ▼
                              │        │        │      D4=T4
                              │        │        │        │
                              │        │        │        │ ↑ bilinear
                              │        │        ▼        │
                              │        │    PairFuse(T3,↑D4)
                              │        │        │
                              │        │   RepLocalBlock
                              │        │        │
                              │        │        D3
                              │        │        │
                              │        │        │ ↑ bilinear
                              │        ▼        │
                              │    PairFuse(T2,↑D3)
                              │        │
                              │   RepLocalBlock
                              │        │
                              │        D2
                              │        │
                              │        │ ↑ bilinear
                              ▼        │
                          PairFuse(T1,↑D2)
                              │
                         RepLocalBlock
                              │
                              D1
                              │
                         RepLocalBlock
                         (final refine)
                              │
                          Conv1×1 160→2
                              │
                       Bilinear ×4 to 256
                              │
                         Change logits
```

---

# 3. 全局部署图

所有训练分支折叠后：

```text
P1,Q1 ─► single TAR Conv1×1 ─► SiLU ─► DW3 ─► SiLU ─► PW1 ─► SiLU ─► T1
P2,Q2 ─► single TAR Conv1×1 ─► SiLU ─► DW3 ─► SiLU ─► PW1 ─► SiLU ─► T2
P3,Q3 ─► single TAR Conv1×1 ─► SiLU ─► DW3 ─► SiLU ─► PW1 ─► SiLU ─► T3
P4,Q4 ─► single TAR Conv1×1 ─► SiLU ─► DW3 ─► SiLU ─► PW1 ─► SiLU ─► T4

T4 = D4

T3, up(D4)
    └─► single PairFuse Conv1×1
        └─► SiLU
            └─► single DW3
                └─► SiLU
                    └─► single PW1
                        └─► SiLU = D3

T2, up(D3) → 同结构 = D2
T1, up(D2) → 同结构 = D1

D1
 → single DW3
 → SiLU
 → single PW1
 → SiLU
 → 1×1 classifier
 → upsample
```

部署 decoder 中**没有**：

```text
MDTA
Softmax attention
SE
Spatial-Mamba
selective scan
large-kernel decoder branch
FFN gate
```

全局建模由已经保留的 VMamba Encoder 完成。

Decoder 只做：

```text
变化提取
局部细化
跨尺度融合
```

这比“Encoder 已经是 VMamba、Decoder 再堆一套 Mamba+Attention”更干净。

---

# 4. Clean TAR-DCR 的核心设计逻辑

本方案只保留两个方法级贡献：

## 4.1 TAR：Temporal Algebraic Re-parameterization

负责：

```text
encoder → decoder
```

的二时相变化表达。

训练：

```text
Concat
Sum
signed-Diff
```

三路。

部署：

```text
single 1×1
```

这是 change detection 专属的 temporal structural re-parameterization。

## 4.2 DCR：Decoder-wide Compositional Re-parameterization

整个 decoder 不再有“普通 Conv + 某处偶尔一个 RepConv”。

相反：

```text
每一个局部空间 mixer
每一个 channel mixer
每一个跨尺度 fusion
```

都使用可折叠训练参数化。

最终 deploy：

```text
固定 Conv/DWConv/1×1 单路径拓扑
```

---

# 5. 最终 `models/changedetection/models/` 目录

重构完成后建议只留下：

```text
models/changedetection/models/
├── __init__.py
├── Mamba_backbone.py
├── reparam.py
├── tar.py
├── dcr_decoder.py
└── STRRepNet.py
```

一共 6 个文件。

---

# 6. 哪些旧文件删除

在新模型 smoke / dry-run 全部通过后，从 working tree 删除：

```text
models/changedetection/models/
├── MambaBCD.py
├── Spatial_Mamba.py
├── ChangeDecoder_spatialMamba_small_ICFKFusion_DualTransformer_MDTA_four_blocks_with_one_embeding_layer.py
└── utils.py
```

删除原因：

### `MambaBCD.py`

它是 HAM-CD baseline wrapper。

新模型使用：

```text
STRRepNet.py
```

### `Spatial_Mamba.py`

它只服务旧 HAM decoder。

新 decoder 不再使用 decoder-side Mamba。

### 超长 `ChangeDecoder...py`

完全删除。

不再保留：

```text
HAM
TAM
TAB
MDTA
ICSF
OverlapPatchEmbed
旧 TransformerBlock
旧 FeedForward
旧死参数
```

### `utils.py`

旧 decoder-side Spatial-Mamba selective-scan FLOPs helper。

新 decoder 无 selective scan。

但**删除前必须先 grep**：

```bash
grep -R "changedetection.models.utils" \
    /home/yqwang/projects/STR-RepNet/models \
    /home/yqwang/projects/STR-RepNet/train_scripts
```

把旧 `train.py` / `smoke_test.py` 中相关 import 一并改掉后再删。

---

# 7. 哪些文件绝对不删

保留：

```text
models/changedetection/models/Mamba_backbone.py
models/classification/
models/kernels/selective_scan/
models/changedetection/configs/
models/changedetection/datasets/
models/changedetection/utils_func/
```

尤其：

```text
classification/models/vmamba.py
```

仍然是 Encoder 主体。

---

# 8. 删除 baseline decoder 前先做 Git 归档

“working tree 不保留 baseline decoder”不等于“把科研证据永久抹掉”。

在删除前：

```bash
cd /home/yqwang/projects/STR-RepNet

git status
git add .
git commit -m "archive HAM-CD baseline before TAR-DCR refactor"
git tag baseline-hamcd-run1
git push origin main
git push origin baseline-hamcd-run1
```

之后再做 clean refactor。

这样：

```text
main
```

可以是干净 STR-RepNet。

旧 HAM-CD baseline 可以随时从 tag 恢复。

不要删除：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/baseline/Run1/
/home/yqwang/outputs/STR-RepNet/baseline/Run1/
```

---

# 9. `reparam.py`：整个 TAR-DCR 的代数基础

这个文件包含：

```text
fuse_conv_bn()
pad_1x1_to_3x3()
pad_1x3_to_3x3()
pad_3x1_to_3x3()
identity_dw_kernel()

RepDW3
RepPW1x1
RepPairFuse1x1

switch_module_to_deploy()
```

---

# 10. 统一规则：所有 Rep 模块都用共享 BN

为了：

```text
训练稳定
+
折叠公式简单
+
FP32 <1e-6 更容易通过
```

不要给每条辅助 branch 各加一套 BN。

统一写成：

```text
Linear branch A ─┐
Linear branch B ─┼─ SUM → one shared BN → activation
Linear branch C ─┘
```

部署：

```text
先把 A/B/C kernel 合起来
→ 再 fold shared BN
→ single Conv(bias=True)
→ activation
```

这样非常干净。

---

# 11. `RepDW3`

## 11.1 训练图

```text
                ┌─ DW3×3 ───────┐
x ──────────────┼─ DW1×3 ───────┤
                ├─ DW3×1 ───────┼─ SUM → BN → SiLU
                └─ Identity ─────┘
```

其中 Identity 是可选的。

用于 `RepLocalBlock` 时：

```text
include_identity=True
```

这样部署时 residual 直接折到 DW3 中心。

## 11.2 部署图

```text
x → single DW3×3(bias=True) → SiLU
```

## 11.3 kernel

```text
Kdeploy =
    K3x3
  + pad(K1x3)
  + pad(K3x1)
  + Iδ
```

其中：

```text
Iδ[:,0,1,1] = 1
```

然后 fold shared BN。

---

# 12. `RepDW3` 初始化

主分支：

```text
DW3×3：正常 Kaiming
```

辅助：

```text
DW1×3：0
DW3×1：0
```

Identity：

```text
固定，不是参数
```

这样 epoch 0 的行为接近：

```text
x + DW3(x)
```

不会多分支随机输出同时放大。

---

# 13. `RepPW1x1`

仅仅：

```text
main 1×1 + diagonal scale
```

还不够强。

Clean DCR 推荐加入一个**无激活的低秩串联分支**：

```text
main Conv1×1(C→C) ─────────────────────┐
                                       │
Conv1×1(C→r) → Conv1×1(r→C) ──────────┼─ SUM → BN → SiLU
                                       │
channel diagonal scale ────────────────┘
```

其中：

```text
C = 160
r = C/4 = 40
```

注意：

```text
C→r→C
```

中间**不能有激活、BN、LayerNorm**。

否则不能合并为单 1×1。

---

# 14. `RepPW1x1` 折叠

主：

\[
W_m \in \mathbb{R}^{C\times C}
\]

低秩：

\[
W_1 \in \mathbb{R}^{r\times C}
\]

\[
W_2 \in \mathbb{R}^{C\times r}
\]

组合：

\[
W_{serial}=W_2W_1
\]

对角分支：

\[
D=\mathrm{diag}(s)
\]

最终：

\[
W_{deploy}=W_m+W_2W_1+D
\]

然后 shared BN fold。

部署：

```text
single Conv1×1(C→C)
→ SiLU
```

---

# 15. `RepPW1x1` 初始化

```text
main：正常 Kaiming
serial W1：正常 Kaiming
serial W2：全 0
diag scale：全 0
```

为什么不是 W1/W2 都 0：

如果两层都 0：

```text
梯度传播会很差
```

使用：

```text
W1 random
W2 zero
```

可让 branch 初始输出仍为 0，但 W2 能立即获得梯度。

---

# 16. `RepPairFuse1x1`

用于 Decoder 跨尺度融合：

输入：

```text
L = 当前尺度 TAR feature
H = 上一级 decoder feature 上采样结果
```

两者：

```text
B×160×H×W
```

训练：

```text
[L,H] ─ Conv1×1(320→160) ───────┐
L+H   ─ Conv1×1(160→160) ───────┼─ SUM → BN → SiLU
H-L   ─ Conv1×1(160→160) ───────┘
```

部署：

```text
cat([L,H])
→ single Conv1×1(320→160,bias=True)
→ SiLU
```

它的代数形式和 TAR 相同，但语义不同：

```text
TAR：时间维融合
PairFuse：跨尺度语义/细节融合
```

这两个都属于 DCR 的统一 algebraic folding 框架。

---

# 17. `RepPairFuse1x1` 初始化

```text
concat branch：Kaiming
sum branch：0
diff branch：0
BN weight=1 / bias=0
```

这样初始行为接近普通 concat fusion。

---

# 18. `tar.py`

只包含：

```text
TemporalRep1x1
TARStage
MultiScaleTAR
```

---

# 19. `TemporalRep1x1`

输入：

```text
P: B×Ci×H×W
Q: B×Ci×H×W
```

训练：

```text
[P,Q] → Conv1×1 ─┐
P+Q   → Conv1×1 ─┼→ SUM → BN → SiLU
Q-P   → Conv1×1 ─┘
```

必须是：

```python
Q - P
```

不能：

```python
abs(Q - P)
```

---

# 20. TAR 折叠公式

主 concat kernel：

\[
W_c=[W_{cP},W_{cQ}]
\]

sum branch：

\[
W_s(P+Q)
\]

signed diff：

\[
W_d(Q-P)
\]

所以：

\[
W_P=W_{cP}+W_s-W_d
\]

\[
W_Q=W_{cQ}+W_s+W_d
\]

最终：

\[
W_{deploy}=[W_P,W_Q]
\]

然后 fold shared BN。

部署：

```text
cat(P,Q)
→ one Conv1×1(2Ci→160)
→ SiLU
```

---

# 21. TAR 初始化

```text
concat branch：Kaiming
sum：0
diff：0
```

这点非常重要。

不要三路都随机初始化。

---

# 22. `TARStage`

每一级不只是做 1×1 temporal projection。

设计：

```text
TemporalRep1x1
→ RepLocalBlock
```

所以：

```text
Pi,Qi
→ change-specific temporal projection
→ local spatial refinement
→ Ti
```

---

# 23. `RepLocalBlock`

统一结构：

```text
x
→ RepDW3(identity=True)
→ shared BN folded in deploy
→ SiLU
→ RepPW1x1
→ shared BN folded in deploy
→ SiLU
```

部署：

```text
x
→ single DW3
→ SiLU
→ single PW1
→ SiLU
```

没有 deploy residual branch。

严格意义上是单路径。

---

# 24. 为什么不再放 decoder-side Mamba

Encoder 已经是：

```text
VMamba-Tiny
```

它负责：

```text
长距离依赖
全局空间建模
hierarchical representation
```

如果 decoder 再放：

```text
Spatial Mamba
MDTA
SE
```

会有三个问题：

1. 论文方法焦点继续被 HAM-CD 牵着走；
2. deployment graph 里长期保留一堆不可折动态算子；
3. 难以证明“性能来自结构重参数化”。

Clean TAR-DCR 的 decoder 专门解决：

```text
temporal change extraction
local reconstruction
multi-scale detail fusion
```

这条叙事更清楚。

---

# 25. `MultiScaleTAR`

```python
class MultiScaleTAR(nn.Module):
    def __init__(
        self,
        encoder_dims=(96,192,384,768),
        dim=160,
        rep=True,
    ):
```

内部：

```text
stage1: 96  →160
stage2: 192 →160
stage3: 384 →160
stage4: 768 →160
```

forward：

```python
t1 = self.stage1(pre1, post1)
t2 = self.stage2(pre2, post2)
t3 = self.stage3(pre3, post3)
t4 = self.stage4(pre4, post4)

return [t1,t2,t3,t4]
```

---

# 26. `dcr_decoder.py`

只包含：

```text
RepLocalBlock
DCRDecoder
```

以及可选：

```text
PlainLocalBlock
```

但更推荐同一类通过 `rep_mode` 控制辅助分支。

---

# 27. DCR Decoder

输入：

```text
T1: 160×64×64
T2: 160×32×32
T3: 160×16×16
T4: 160×8×8
```

---

# 28. Deep stage

```text
D4 = T4
```

T4 在 TARStage 中已经经过一个 RepLocalBlock。

不再单独放：

```text
attention
Mamba
large kernel
```

---

# 29. Stage 3

```text
U4 = bilinear(D4 → T3 spatial size)

F3 = RepPairFuse1x1(
    T3,
    U4
)

D3 = RepLocalBlock(F3)
```

---

# 30. Stage 2

```text
U3 = bilinear(D3 → T2 size)
F2 = RepPairFuse1x1(T2,U3)
D2 = RepLocalBlock(F2)
```

---

# 31. Stage 1

```text
U2 = bilinear(D2 → T1 size)
F1 = RepPairFuse1x1(T1,U2)
D1 = RepLocalBlock(F1)
```

---

# 32. Final refine

再加一个：

```text
D1
→ RepLocalBlock
→ refined D1
```

这是整个 decoder 里唯一额外的高分辨率 refinement。

不要加：

```text
EdgeGate
SE
CBAM
MDTA
deep supervision
```

第一轮保持 TAR-DCR 主线纯净。

---

# 33. `STRRepNet.py`

这是最终唯一模型入口。

最终：

```python
class STRRepNet(nn.Module):
```

不再叫：

```text
STMambaBCD
HAMCD
```

---

# 34. `STRRepNet.__init__`

结构：

```python
self.encoder = Backbone_VSSM(
    out_indices=(0,1,2,3),
    pretrained=pretrained,
    ...
)

self.tar = MultiScaleTAR(
    encoder_dims=self.encoder.dims,
    dim=160,
    ...
)

self.decoder = DCRDecoder(
    dim=160,
    ...
)

self.head = nn.Conv2d(
    160,
    2,
    kernel_size=1
)
```

---

# 35. Encoder 冻结

按本课题最终约束：

```python
for p in self.encoder.parameters():
    p.requires_grad_(False)
```

更重要的是：

```python
self.encoder.eval()
```

仅仅 `requires_grad=False` 还不够。

因为 `model.train()` 可能重新打开：

```text
DropPath
Dropout
```

所以覆盖：

```python
def train(self, mode=True):
    super().train(mode)

    # Frozen feature extractor:
    self.encoder.eval()

    return self
```

这样训练 decoder 时 Encoder 始终是固定特征提取器。

---

# 36. 关于你当前 HAM-CD baseline 与 Frozen Encoder 的公平性

必须明确：

当前仓库旧 `train.py` 是：

```text
optimizer(model.parameters())
```

旧 `MambaBCD.py` 没有冻结 Encoder。

所以当前：

```text
WHU 0.9500
LEVIR 0.9211
CDD 0.9879
SYSU ~0.83
```

属于**旧 HAM-CD trainable-encoder protocol**。

而 Clean TAR-DCR 按本课题最终定义：

```text
Frozen Encoder
```

这两者可以作为工程参考，但不能作为论文中的严格同协议因果比较。

因此：

### 当前工程目标

仍然可以定成：

```text
Clean TAR-DCR Full 尽量超过这四个当前数字
```

### 论文正式 baseline

方法确定后需要再补：

```text
HAM-CD Frozen-Encoder E0
vs
TAR-DCR Frozen-Encoder Full
```

否则审稿时协议不一致。

---

# 37. `STRRepNet.forward`

```python
def forward(self, pre, post):
    with torch.no_grad():
        pre_feats = self.encoder(pre)
        post_feats = self.encoder(post)

    temporal_feats = self.tar(
        pre_feats,
        post_feats
    )

    x = self.decoder(temporal_feats)

    logits = self.head(x)

    logits = F.interpolate(
        logits,
        size=pre.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    return logits
```

使用 `torch.no_grad()` 能进一步减少 frozen encoder 训练内存。

---

# 38. `switch_to_deploy`

```python
@torch.no_grad()
def switch_to_deploy(self):
    self.tar.switch_to_deploy()
    self.decoder.switch_to_deploy()
    return self
```

Encoder 无需转换。

Head 无需转换。

---

# 39. `__init__.py`

最终：

```python
from .STRRepNet import STRRepNet

__all__ = ["STRRepNet"]
```

不要再 export 旧 HAM 类。

---

# 40. Clean models 最终依赖关系

```text
STRRepNet.py
 ├── Mamba_backbone.py
 ├── tar.py
 │    └── reparam.py
 └── dcr_decoder.py
      └── reparam.py
```

不存在：

```text
STRRepNet
→ old decoder
→ Spatial_Mamba
→ MDTA
→ ICSF
```

---

# 41. 三种 ablation mode 必须直接内置进新模型

为了不保留 baseline decoder，但仍能做最小消融：

```text
mode="plain"
mode="tar"
mode="full"
```

---

# 42. `plain`

**同一个最终 deploy topology**，但关闭所有 auxiliary rep branches。

也就是：

```text
Temporal:
only concat main 1×1

RepDW:
only main DW3

RepPW:
only main PW1

PairFuse:
only concat main 1×1
```

仍然：

```text
TAR-style new clean architecture
```

但不使用 structural expansion。

它是：

> **Clean Single-Path Counterpart**

这是比拿 HAM-CD 当唯一消融基线更严谨的内部对照。

---

# 43. `tar`

开启：

```text
Temporal concat + sum + signed-diff
```

但：

```text
RepDW auxiliary off
RepPW serial/diag auxiliary off
PairFuse sum/diff auxiliary off
```

因此：

```text
plain → tar
```

唯一新增变量就是 **Temporal Rep**。

---

# 44. `full`

全部打开：

```text
Temporal TAR
RepDW
RepPW
RepPairFuse
```

因此：

```text
tar → full
```

唯一新增变量就是 **Decoder-wide Rep expansion**。

---

# 45. 最小消融为什么用这三个版本

最小因果链：

```text
A0 Plain
   ↓ + Temporal Rep
A1 TAR
   ↓ + Decoder-wide Rep
A2 Full
```

这样能分别回答：

```text
TAR 是否有效？
DCR 是否在 TAR 上继续有效？
```

不需要第一轮再跑十几个 `w/o`。

---

# 46. 复杂度预算：Clean TAR-DCR

统一：

```text
D=160
low-rank r=40
```

---

# 47. TAR 部署参数

四级：

```text
2Ci → 160
```

deploy 总量约：

```text
0.461M
```

---

# 48. Cross-scale PairFuse 部署参数

3 个：

```text
320 →160
```

共约：

```text
0.154M
```

---

# 49. RepLocalBlock 部署参数

每个：

```text
DW3:
160×3×3

PW1:
160×160
```

约：

```text
27.3k / block
```

总计 8 个左右：

```text
约 0.218M
```

---

# 50. 中间+Decoder+Head 总 deploy 参数

解析估算：

```text
≈ 0.835M
```

注意这是：

```text
不包含 VMamba encoder
```

所以整网：

```text
Total deploy Params
=
Frozen VMamba-Tiny encoder
+
约 0.835M
```

最终必须用代码实测，不要把 0.835M 当整网参数。

---

# 51. 中间+Decoder trainable 参数

由于训练期 auxiliary branches 存在：

```text
≈ 1.56M
```

仍然非常轻。

Encoder 冻结后：

```text
Trainable Params ≈ 1.56M
```

加 classifier 已包含在这个量级估算里。

---

# 52. 中间+Decoder deploy FLOPs

按：

```text
256 input
F1=64
F2=32
F3=16
F4=8
D=160
```

解析估算约：

```text
≈ 0.92G
```

不含共享 VMamba Encoder 的双时相 FLOPs。

所以整模型 deploy FLOPs 必须实测：

```text
2× Encoder
+
~0.92G middle/decoder
```

预计会显著低于当前 HAM-CD 的：

```text
16.26G
```

但论文中只写实测值。

---

# 53. 这版为什么比上一版更“Clean”

上一版：

```text
HAM Decoder
+
TAR
+
在旧 TAM/MDTA/ICSF 上做 Rep
```

本版：

```text
VMamba Encoder
+
TAR
+
DCR
```

没有 legacy HAM component。

论文 Contribution 也更干净：

```text
Contribution 1:
Temporal Algebraic Re-parameterization

Contribution 2:
Decoder-wide Compositional Re-parameterization
```

而不是：

```text
HAM
Mamba
MDTA
ICSF
RepConv
Temporal Rep
...
```

一起堆。

---

# 54. 不可折算子现在有哪些

Clean Decoder 中几乎只剩：

```text
SiLU
bilinear interpolation
```

这两个本来就不需要声称可折。

Encoder 中仍有：

```text
VMamba selective scan
LayerNorm
SiLU / activation
```

Encoder 不属于本轮结构重参数化范围。

因此论文的 exactness 边界非常容易解释。

---

# 55. 为什么不加 SE

SE：

```text
GAP
→ MLP
→ Sigmoid
→ input-dependent gate
```

不能严格代数折叠。

Clean TAR-DCR 完全不需要它。

---

# 56. 为什么不加 Softmax Attention

同理：

```text
QK
→ Softmax
→ AV
```

输入依赖。

既然 Encoder 已有 VMamba 全局建模，没必要在 decoder 再留一个不可折 attention core。

---

# 57. 为什么不加大核 Rep

当前研究已经知道：

```text
CD-RLKNet
LKMamba-CD
UniRepLKNet
```

都使“大核 reparam”很容易撞车。

Clean DCR 选择：

```text
3×3 target kernel
+
1×3 / 3×1 training branches
```

核心不是“大感受野”，而是：

```text
同 deploy cost
更丰富训练参数化
```

---

# 58. 必须修改 `train.py`

旧：

```python
from changedetection.models.MambaBCD import STMambaBCD
```

改：

```python
from changedetection.models.STRRepNet import STRRepNet
```

---

# 59. `_build_model`

改成：

```python
model = STRRepNet(
    pretrained=self.args.pretrained_weight_path,
    rep_mode=self.args.rep_mode,
    patch_size=...,
    ...
)
```

新增：

```python
--rep_mode plain
--rep_mode tar
--rep_mode full
```

---

# 60. Optimizer 只传 trainable 参数

必须改：

```python
trainable_params = [
    p for p in self.model.parameters()
    if p.requires_grad
]

self.optimizer = optim.AdamW(
    trainable_params,
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
```

这样 optimizer 不保存 frozen VMamba state。

---

# 61. 日志必须同时报告两种参数量

```text
[TOTAL-PARAMS]
[TRAINABLE-PARAMS]
```

因为 Encoder frozen。

例如：

```python
total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)
```

---

# 62. FLOPs helper 修改

旧 `train.py` 还注册：

```text
changedetection.models.utils.selective_scan_state_flop_jit
```

那是旧 Spatial-Mamba decoder 用的。

Clean Decoder 删除后：

```text
只保留 classification.models.vmamba selective_scan FLOPs helper
```

即只处理 Encoder VMamba。

删除：

```python
from changedetection.models.utils import selective_scan_state_flop_jit
```

和相应：

```text
prim::PythonOp.SelectiveScanStateFn
```

如果 Encoder 的具体 op key 实测仍需要，按 `classification.models.vmamba` 当前实现保留。

---

# 63. `smoke_test.py` 修改

旧：

```text
STMambaBCD
```

改：

```text
STRRepNet
```

Smoke 输出至少检查：

```text
pre/post shape
4-level encoder shape
decoder output
logits shape = (2,2,256,256)
total params
trainable params
FLOPs
```

---

# 64. 新增 `test_reparam_equivalence.py`

必须测试：

```text
TemporalRep1x1
RepDW3
RepPW1x1
RepPairFuse1x1
TARStage
MultiScaleTAR
RepLocalBlock
DCRDecoder
STRRepNet whole model
```

---

# 65. Block-level equivalence

流程：

```python
m.eval()

y0 = m(...)

m2 = copy.deepcopy(m)
m2.switch_to_deploy()
m2.eval()

y1 = m2(...)

err = (y0-y1).abs().max()

assert err < 1e-6
```

建议 block：

```text
<1e-7
```

whole model：

```text
<1e-6
```

---

# 66. GPU equivalence 前关闭 TF32

```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
```

避免数值路径差异污染判断。

---

# 67. 整模型测试输入

至少：

```text
10 组 random:
1×3×256×256

10 组 random:
2×3×256×256
```

再从：

```text
CDD
LEVIR
SYSU
WHU
```

各抽：

```text
2 pair
```

比较 logits。

---

# 68. `switch_to_deploy()` 后必须检查结构

写一个检查器：

```text
assert no Temporal auxiliary branch
assert no RepDW 1x3 branch
assert no RepDW 3x1 branch
assert no RepPW serial branch
assert no RepPW diag parameter
assert no PairFuse sum/diff branch
```

部署 graph 只能剩：

```text
Conv
DWConv
SiLU
Upsample
Head
Encoder
```

---

# 69. checkpoint

新 Clean STR-RepNet checkpoint：

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_f1": ...,
    "best_epoch": ...,
    "rep_mode": args.rep_mode,
    "deploy": False,
    "decoder_dim": 160,
    "reparam_version": 1
}
```

不要加载旧 HAM decoder checkpoint。

唯一初始化来源：

```text
VMamba-Tiny pretrained encoder
+
new decoder random init
```

---

# 70. `test_best()` 必须对 deploy graph 测试

顺序：

```text
load best train graph
→ eval
→ reference logits
→ deepcopy
→ switch_to_deploy
→ equivalence
→ measure deploy Params
→ measure deploy FLOPs
→ latency
→ final test metrics
```

最终 test：

```text
必须用 deploy model
```

---

# 71. 最终日志格式

```text
=== TEST RESULTS ===
[MODEL] STR-RepNet Clean TAR-DCR
[REP-MODE] full
[TOTAL-TRAIN-GRAPH-PARAMS] ...
[TRAINABLE-PARAMS] ...
[DEPLOY-PARAMS] ...
[DEPLOY-FLOPS] ...
[REPARAM-MAX-ABS-ERROR] ...
[LATENCY-B1-MS] ...
[FPS-B1] ...

Recall=...
Precision=...
OA=...
F1=...
IoU=...
Kappa=...
=== END TEST RESULTS ===
```

---

# 72. 最小消融：现在应该怎么跑

Clean refactor 后，不能只做：

```text
HAM-CD
vs
Full
```

因为二者 decoder 拓扑已经完全不同。

最小内部消融必须是：

```text
A0 Plain
A1 TAR
A2 Full
```

---

# 73. A0：Plain Single-Path

训练期：

```text
TAR:
只用 concat main

RepDW:
只用 3×3 main

RepPW:
只用 main 1×1

PairFuse:
只用 concat main
```

最终 deploy graph 与 Full 的 kernel shapes 完全相同。

因此它回答：

> 如果没有 training-time expansion，这个轻量 decoder 本身能做到多少？

---

# 74. A1：TAR

A0 +：

```text
Temporal sum
Temporal signed-diff
```

其他辅助 branch 仍关闭。

回答：

> Temporal Rep 有没有额外训练收益？

---

# 75. A2：Full TAR-DCR

A1 +：

```text
DW1×3
DW3×1
PW serial low-rank
PW diagonal
PairFuse sum
PairFuse signed-diff
```

回答：

> Decoder-wide Rep 是否继续有收益？

---

# 76. 最小新增长训练：6 次

## 第 1～3 次：只在 SYSU

SYSU 当前是：

```text
最弱
最有提升空间
最适合先筛结构
```

跑：

```text
SYSU Plain
SYSU TAR
SYSU Full
```

如果：

```text
Plain < TAR < Full
```

或至少：

```text
Full > Plain
且 TAR 对 Full 有正贡献
```

就继续。

---

# 77. 第 4～6 次：Full 补三个数据集

只跑：

```text
LEVIR Full
WHU Full
CDD Full
```

这样总共：

```text
6 个新 300-epoch runs
```

就可以得到：

### 内部消融

```text
SYSU:
Plain
TAR
Full
```

### 四数据集 Full

```text
SYSU
LEVIR
WHU
CDD
```

这是**绝对最小版本**。

---

# 78. 论文版更稳的补充

如果 Full 成功，投稿前再只补 2 次：

```text
LEVIR Plain
LEVIR TAR
```

这样模块有效性就有：

```text
SYSU
LEVIR
```

两个不同场景支持。

总共 8 次，而不是一开始就跑十几次。

---

# 79. SYSU 最小消融表

| Exp | Temporal Rep | Spatial/Channel DCR | Cross-scale DCR | Deploy Topology | F1 |
|---|---:|---:|---:|---|---:|
| A0 Plain | ✗ | ✗ | ✗ | identical kernel shapes |  |
| A1 TAR | ✓ | ✗ | ✗ | identical kernel shapes |  |
| A2 Full | ✓ | ✓ | ✓ | identical kernel shapes |  |

最关键的是：

> 三个版本的 **deploy kernel shapes 基本相同**。

区别主要发生在训练图。

这正是 Structural Re-parameterization 最漂亮的消融。

---

# 80. 与当前 HAM-CD 四个 F1 怎么比较

工程目标：

```text
WHU  > 0.9500
LEVIR > 0.9211
CDD   > 0.9879
SYSU  > 最终 baseline TEST F1
```

不能保证全部超过。

真正成功判据：

```text
Full 至少在 2/4 dataset 超过当前 HAM-CD reference
+
deploy Params/FLOPs 显著低
+
fold error <1e-6
```

尤其：

```text
SYSU
LEVIR
WHU
```

比 CDD 更有判断力。

CDD=0.9879 已接近饱和。

---

# 81. 第一阶段 Go / No-Go

## TAR 有效

建议内部门槛：

```text
A1 >= A0 + 0.15 F1 point
```

只是工程筛选门槛，不叫统计显著。

## DCR 有效

```text
A2 >= A1 + 0.15 F1 point
```

或：

```text
A2 明显 > A0
同时不牺牲 deploy cost
```

---

# 82. 如果 Plain 就远低于 HAM-CD

例如 SYSU：

```text
Plain 比 HAM 低 >1.0 F1 point
```

说明：

```text
decoder 压得过轻
```

不要先怪 TAR。

第一调整只允许：

```text
D=160 → D=192
```

不要同时堆新模块。

因为 192 的中间+Decoder 仍然很轻。

---

# 83. 如果 TAR 下降

说明：

```text
signed temporal parameterization
没有给当前 clean decoder 带来优化收益
```

先检查：

```text
sum branch norm
diff branch norm
```

如果几乎为 0：

```text
branch 没学起来
```

如果很大且指标下降：

```text
temporal asymmetry / optimization 可能有问题
```

此时不要改成 abs-diff，因为 abs 无法 exact fold。

---

# 84. 如果 Full 下降

分两类：

### TAR 好，Full 差

说明：

```text
decoder-wide Rep expansion 过强
```

后续第二阶段只做：

```text
Full without RepPW serial
Full without cross-scale sum/diff
```

不要改整个模型。

### Plain/TAR/Full 都差

说明 clean decoder 本体能力不足。

优先：

```text
D=192
```

而不是重新塞 MDTA/Mamba。

---

# 85. 如果 Full 成功

再做细消融：

```text
Full - Temporal diff branch
Full - RepPW serial branch
Full - PairFuse diff branch
```

先 SYSU 单数据集。

这些是论文第二轮，不是现在。

---

# 86. 不要加的东西

第一版 Clean TAR-DCR 禁止加入：

```text
SE
CBAM
MDTA
Transformer
decoder-side Mamba
large-kernel 11×11
edge head
deep supervision
KD
teacher
contrastive loss
new loss
new augmentation
```

否则“clean”又会变成模块堆叠。

---

# 87. loss 不变

保持：

```text
CE + 2.0×Lovász-Softmax
```

---

# 88. 数据协议不变

```text
A/B/label
gray>=128
crop=256
```

A/B/label 同步几何增强。

---

# 89. 训练预算

保持：

```text
seed=2333
epochs=300
batch=16
test_batch=16
AdamW
lr=1e-4
weight_decay=5e-4
```

注意：

Frozen encoder 后 optimizer 只包含 trainable decoder。

---

# 90. 当前 test leakage 问题

当前旧 baseline：

```text
test set 每 epoch 选 best
```

Clean TAR-DCR 如果完全照旧，可以做**内部工程比较**。

但投稿正式结果不能这样。

正式论文必须：

```text
train
→ val select
→ test once
```

并给：

```text
HAM Frozen E0
Clean TAR-DCR Frozen Full
```

都重跑。

---

# 91. `train_scripts` 建议

Clean 后：

```text
train_scripts/
├── baseline/Run1/       # 可以保留脚本作为历史，不属于 models
└── clean_tardcr/
    ├── A0_Plain/
    │   └── train_SYSU-CD-256.sh
    ├── A1_TAR/
    │   └── train_SYSU-CD-256.sh
    └── A2_Full/
        ├── train_SYSU-CD-256.sh
        ├── train_LEVIR-CD-256.sh
        ├── train_WHU-CD-256.sh
        └── train_CDD-CD-256.sh
```

---

# 92. checkpoint 目录

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/clean_tardcr/
├── A0_Plain/SYSU-CD-256/
├── A1_TAR/SYSU-CD-256/
└── A2_Full/
    ├── SYSU-CD-256/
    ├── LEVIR-CD-256/
    ├── WHU-CD-256/
    └── CDD-CD-256/
```

---

# 93. log 目录

```text
/home/yqwang/outputs/STR-RepNet/clean_tardcr/
```

同样分：

```text
A0_Plain
A1_TAR
A2_Full
```

---

# 94. Smoke 顺序

在任何 300 epoch 前：

## 1. import

```bash
python -m py_compile \
changedetection/models/reparam.py \
changedetection/models/tar.py \
changedetection/models/dcr_decoder.py \
changedetection/models/STRRepNet.py
```

## 2. block fold

```text
全部 <1e-6
```

## 3. whole model random

```text
B=2
256×256
logits=(2,2,256,256)
```

## 4. backward

确认：

```text
encoder grad = None
decoder grad finite
```

## 5. real data dry-run

SYSU：

```text
8~16 pair
5 iterations
```

## 6. deploy equivalence

整模型：

```text
<1e-6
```

通过后才开长训练。

---

# 95. Encoder 冻结验证

Smoke 必须打印：

```python
encoder_trainable = sum(
    p.numel()
    for p in model.encoder.parameters()
    if p.requires_grad
)

assert encoder_trainable == 0
```

并且：

```python
assert model.encoder.training is False
```

即使：

```python
model.train()
```

之后仍必须为 False。

---

# 96. pretrained 权重加载验证

当前 `Mamba_backbone.py` 用：

```text
strict=False
```

所以必须人工检查：

```text
missing_keys
unexpected_keys
```

不能只看到：

```text
Successfully load ckpt
```

就认为全部加载正确。

Smoke 建议将关键 missing/unexpected 输出写日志。

---

# 97. 代码删除执行顺序

不要一上来 `rm`。

顺序：

```text
1. Git tag baseline
2. 新建 4 个 Clean 文件
3. 修改 train/smoke import
4. py_compile
5. smoke
6. whole-model equivalence
7. grep legacy imports
8. 删除 old decoder files
9. 再 py_compile
10. 再 smoke
```

---

# 98. Legacy import 检查

删除前：

```bash
grep -R \
"MambaBCD\|Spatial_Mamba\|ChangeDecoder_spatialMamba\|changedetection.models.utils" \
/home/yqwang/projects/STR-RepNet/models \
/home/yqwang/projects/STR-RepNet/train_scripts
```

最终除了：

```text
Git history / docs 描述
```

运行代码不应出现旧 decoder import。

---

# 99. 最终 `models` 树必须长这样

```text
models/changedetection/models/
├── __init__.py
├── Mamba_backbone.py
├── reparam.py
├── tar.py
├── dcr_decoder.py
└── STRRepNet.py
```

这是本轮 clean refactor 的验收条件之一。

---

# 100. 各文件职责严格单一

## `Mamba_backbone.py`

```text
VMamba Encoder only
```

## `reparam.py`

```text
代数 folding primitives
```

## `tar.py`

```text
bi-temporal bridge only
```

## `dcr_decoder.py`

```text
multi-scale decode only
```

## `STRRepNet.py`

```text
top-level network
```

不要重新造一个 2000 行 decoder 文件。

---

# 101. 训练图/部署图一致性定义

必须满足：

\[
f_{\text{train-eval}}(x;\theta)
=
f_{\text{deploy}}(x;T(\theta))
+\epsilon
\]

其中：

```text
T(θ) = 显式 kernel/bias 变换
```

不允许：

```text
post-fold fine-tune
KD
teacher
calibration training
重新优化
```

并要求：

\[
\|\epsilon\|_\infty < 10^{-6}.
\]

---

# 102. 本方案哪些部分属于严格 SRP

```text
Temporal concat/sum/signed-diff
→ single 1×1

DW3 / 1×3 / 3×1 / identity
→ single DW3

main PW / serial low-rank PW / diagonal
→ single PW1

cross-scale concat/sum/signed-diff
→ single fusion 1×1

BN
→ fold into preceding Conv
```

全部是显式代数参数变换。

---

# 103. 哪些部分不是 SRP

```text
SiLU
bilinear interpolation
Frozen VMamba Encoder dynamics
```

它们部署时原样保留。

这不影响 TAR-DCR 结构重参数化定义。

---

# 104. 最小方法图

最终论文方法图可以非常清楚：

```text
             Shared VMamba-Tiny Encoder
                  /             \
                T1               T2
                  \             /
                   Multi-scale TAR
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       F1/TAR         F2/TAR        F3/TAR       F4/TAR
          │             │             │             │
          └────── top-down DCR Decoder ────────────┘
                        │
                     CD Head
```

而每个模块只有两张图：

```text
train graph
deploy graph
```

无需再解释 HAM 的 TAM/TAB/ICSF。

---

# 105. 为什么这套更适合作为硕士论文主方法

它形成了非常完整的逻辑闭环：

```text
问题：
二时相变化特征需要丰富训练表达，
但部署端不能承担复杂融合与多分支成本。

方法：
时间维用 TAR；
空间/通道/跨尺度用 DCR。

训练：
异构多分支。

部署：
统一单路径。

理论：
显式参数变换。

验证：
<1e-6。

效率：
decoder <1G 级。

任务：
四个二值遥感 CD dataset。
```

比“HAM-CD + 很多 RepConv”更容易形成独立方法身份。

---

# 106. 最终立即执行顺序

```text
① 等 SYSU baseline 完成，保存最后 TEST block

② Git commit + tag 当前 HAM-CD baseline

③ 新建：
   reparam.py
   tar.py
   dcr_decoder.py
   STRRepNet.py

④ 改：
   models/changedetection/models/__init__.py
   train.py
   smoke_test.py

⑤ 实现 rep_mode：
   plain
   tar
   full

⑥ 跑 block-level fold test

⑦ 跑 whole-model <1e-6

⑧ 跑 real-data dry-run

⑨ grep 旧 import

⑩ 删除：
   MambaBCD.py
   Spatial_Mamba.py
   old ChangeDecoder...
   utils.py

⑪ 再 smoke

⑫ SYSU Plain

⑬ SYSU TAR

⑭ SYSU Full

⑮ 如果 Full 有效：
   LEVIR Full
   WHU Full
   CDD Full

⑯ 汇总最后 TEST block
```

---

# 107. 最终验收表

| 项目 | 要求 |
|---|---|
| Encoder | VMamba-Tiny，加载原预训练 |
| Encoder 结构 | 不改 |
| Encoder 参数 | Frozen |
| 旧 HAM Decoder | **删除** |
| Spatial_Mamba decoder | **删除** |
| MDTA | **删除** |
| ICSF | **删除** |
| TAR | 全新 |
| Decoder | 全新 DCR |
| Decoder SSM | 无 |
| Decoder Attention | 无 |
| Decoder SE | 无 |
| Train multi-branch | 有 |
| Deploy branch | 全部折叠 |
| Deploy middle+decoder Params | 约 0.835M，待实测 |
| Deploy middle+decoder FLOPs | 约 0.92G，待实测 |
| Fold error | `<1e-6` |
| Loss | CE + 2×Lovász |
| Seed | 2333 |
| Epoch | 300 |
| Minimum ablation | Plain → TAR → Full |
| Full datasets | CDD/LEVIR/SYSU/WHU |

---

# 108. 最终结论

这次重构后，**STR-RepNet 不再是 HAM-CD decoder 的修改版**。

它应当成为：

> **Frozen VMamba-Tiny Siamese Encoder + Multi-scale Temporal Algebraic Re-parameterization Bridge + Decoder-wide Compositional Re-parameterization Decoder**

中间模块与 Decoder 全部重新设计，并且所有新增可学习线性算子都围绕同一个原则：

```text
训练时展开
部署时代数吸收
```

最终部署 Decoder 不保留：

```text
Mamba
Attention
SE
large-kernel branch
HAM legacy modules
```

只保留：

```text
1×1 temporal projection
DW3 local refinement
1×1 channel mixing
1×1 cross-scale fusion
SiLU
bilinear upsample
classifier
```

这会让论文的方法身份、复杂度、等价性验证和消融逻辑都比此前版本更干净。

**第一轮不要再加其它模块。先用 `Plain → TAR → Full` 在 SYSU 做三次最小因果消融。如果 Full 成立，再直接补 LEVIR / WHU / CDD Full。这样只需要 6 个新长训练，就能先判断 Clean TAR-DCR 是否值得继续。**
