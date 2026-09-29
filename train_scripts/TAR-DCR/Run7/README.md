# TAR-DCR Run7 — PBRU（Phase-Basis 重参数化输出头）

## 本轮动机

Run6 NSCR FAIL 后诊断：LEVIR 小目标（≤502px）召回 80.8% / 24% 完全漏检（Run2），
320 尺度 Recall↓ Precision↑。瓶颈定位在**解码器最终输出头**（固定的 2ch 粗 logits +
bilinear×4），而非输入分辨率 / encoder。Run7 用结构重参数化给输出头空间与相位自由度。

## 改动

- `PBRUHead`（`models/changedetection/models/reparam.py`）：
  - 部署图：单个 `Conv2d(D → C·r², 1)` + `PixelShuffle(r)`，全分辨率输出，**无插值**；
  - 训练图：主投影 + 4 个零初始化（BN γ=β=0）相位基分支 `{1, u, v, u·v}`（粗/水平/垂直/对角），
    训练期相位展开 `z += Σ_k E_φk(BN_k(branch_k(x)))`，epoch 0 辅助输出**恒等于 0**；
  - 折叠：FP64 解析吸收（`W_eq[c·r²+p] += φ_k[p]·V̄_k[c]`），部署时一次 `.float()` 转换；
    `φ` 为注册的非持久 buffer（4×r² 固定常数），部署 +0 Params/FLOPs。
- `STRRepNet`：`head_mode ∈ {bilinear, pixelshuffle}`、`use_pbru`、`pbru_upscale=4`。
- 训练脚本：`--decoder_dim / --head_mode / --use_pbru / --pbru_upscale`，配置互斥断言
  （use_pbru ⇒ pixelshuffle 且 botr=nscr=0），TEST RESULTS 增加
  `[HEAD-MODE]/[PBRU]/[DECODER-DIM]/[PBRU-GAMMA-NORM]` 标签。
- 详细设计：`docs/temporary/STR-RepNet_Run7_PBRU_修改方案与实验设计.md`。

## Phase 0 结果（预算搜索 + 阶段判别力预检）

- **机器预算搜索 D\***（`analyse/search_pbru_budget.py`，fvcore 实测 batch1@256）：
  锚点 = Run2 部署图（bilinear, D=160）= 28.828706 M / 12.6062 G。
  逐 D 下降搜索：**D\* = 158**（PBRU 部署 28.818618 M / 12.6055 G，双预算均 ≤ 锚点；
  D=159 FLOPs 超标）。Run7 全部实验 `decoder_dim=158`。
- **阶段判别力预检**（`analyse/levir_stage_discriminability.py`，Run2 LEVIR 锚点 ckpt，
  200 对 test，前 100 拟合 / 后 100 held-out，64×64 原生尺度 AUROC）：
  | 指标 | held-out | in-sample |
  |---|---|---|
  | d1 全协方差 LDA（ridge）AUROC@64 | 0.9766 | 0.9858 |
  | d1 对角 LDA AUROC@64 | 0.9397 | 0.9605 |
  | 训练头 AUROC@64 | 0.9881 | 0.9860 |
  | 训练头 AUROC@256（bilinear 部署） | 0.9933 | 0.9950 |
  结论：冻结 Run2 解码器下，d1 的**粗尺度线性可分信息已被训练头全部榨取**（头 0.9881 >
  全 LDA 上界 0.9766，headroom = −0.0115），不存在静态粗尺度线性空间。PBRU 的假设因此是
  **训练期端到端学习的亚单元相位自由度**（粗 64×64 细胞无法区分 4×4 子像素），静态预检
  不能裁决 —— 由 C0/C1/M1 对照裁决。

## 最小实验组（第一阶段只跑 LEVIR；GPU0）

| 组 | 配置（均 D=158, last2） | 状态 |
|---|---|---|
| A0 | Run2 full_last2 锚点（F1=0.9144 / IoU=0.8424 / Precision=0.9260，不重跑） | 复用 |
| C0_Bilinear_D158 | bilinear 头 + D=158（宽度控制，PBRU=0） | **训练中** |
| M1_PBRU_D158 | pixelshuffle 头 + 相位基分支（PBRU=1） | **训练中** |
| C1_PixelShuffle_D158 | 纯 PixelShuffle 头（PBRU=0，phase-rep 归因） | M1 PASS/WEAK 后 |

- 公共：`rep_mode=full / use_residual=1 / use_botr=0 / use_nscr=0 / temporal_swap_prob=0.0 /
  seed=2333 / 300 epoch / batch 16 / lr 1e-4 / lovasz 2.0`。
- C0/C1/M1 的 WHU/SYSU/CDD 脚本已预写；**仅当 M1/LEVIR 系统 PASS 且 M1−C1 有可辨识
  rep 增益后**，按 WHU→SYSU→CDD 顺序启动。

## 预注册判据（LEVIR）

- **M1 PASS**：F1 ≥ 0.9175 且 IoU ≥ 0.8475 且 Precision ≥ 0.9230，且 Deploy ≤
  28.828706 M / 12.6062 G、argmax 分歧 = 0。
- **M1 WEAK**：0.9159 ≤ F1 < 0.9175，或 F1 达标但 IoU/Precision 任一不满足 → 只记录，
  最多补 C1 归因，不扩四数据集。
- **M1 FAIL**：F1 < 0.9159 或 Precision < 0.9220 或预算/等价性失败 → **停止 PBRU，不救机制**
  （不做 r/basis sweep、不加 loss、不叠 full encoder / NSCR）。
- **rep 归因**（M1 vs C1）：+0.15pp+ = rep-supported；+0.05~0.15pp = rep-weak（只能称单 seed
  正向迹象）；< +0.05pp = rep-not-supported（收益主要来自 learned PixelShuffle 头）。
- 四数据集扩展标准：Macro F1 ≥ 92.25%、LEVIR ≥ 91.75%、至少 3/4 数据集 ΔF1 ≥ 0、
  任一退化 ≤ 0.15pp；每数据集 deploy cap 合格 + argmax 分歧 = 0。

## 等价性验证（Run7 全套，GPU0 已通过）

- 冒烟：PBRU 零初始化（4 分支 γ=β=0）、epoch-0 输出与 use_pbru=0 **逐位一致**、γ 梯度非零、
  部署图无 branch/bn/main_proj 残留、部署 Params/FLOPs 与 use_pbru=0 完全一致、
  fold 9.92e-05（<2e-4）、argmax=0。
- 重参数化等价性：T0 相位基常数 + PixelShuffle one-hot 通道序 + c-major 相位展开（精确）；
  T1 PBRUHead 折叠（FP32 1.43e-06 < 2e-5，非平凡权重）+ FP64 折叠代数（2.22e-15）；
  T2 全模型（bilinear/BOTR/NSCR/pixelshuffle/PBRU 五图，均 < 2e-4，argmax 全 0，
  PBRU 全模型 1.297e-04）。
- 说明：T1/T2 的头扰动用**训练后量级**（conv std 0.05 / γ std 0.5）；std=1 扰动会把
  合法 ~5e-5 的 TAR/DCR 折叠噪声经 160 通道 1×1 放大 ~100 倍（2.8e-3），不具代表性。
