## GitHub 更新（手动操作）

代码或文档更新后，手动同步到 GitHub（提交信息统一用 `update code`）：

```bash
cd f:/Code_Repositories_2/CursorCode/STR-RepNet
git add models/ train_scripts/ analyse/ docs/ others/ README.md .gitignore
git commit -m "update code"
git push origin main
```

- 预训练权重 `pretrained_weight/*.pth` 不进 git，更新时上传到 GitHub Releases
  （仓库页 → Releases → Draft a new release → 上传附件 `.pth`）。
- `docs/参考文献/` 的调研 PDF 与 `文献索引.md` 随 `docs/` 一并提交（`.gitignore` 未忽略 PDF）。

---

# STR-RepNet

Lightweight Spatial-Temporal Structural Re-parameterization Network for Remote Sensing Change Detection.

硕士课题：轻量化遥感二值变化检测中的结构重参数化研究。路线见
[`docs/temporary/Rep_Mamba_轻量遥感变化检测重参数化研究方案.md`](docs/temporary/Rep_Mamba_轻量遥感变化检测重参数化研究方案.md)，
服务器与数据规范见
[`docs/RSML-3_服务器环境与变化检测数据统一说明.md`](docs/RSML-3_服务器环境与变化检测数据统一说明.md)。

## 研究定位与约定

- **方法创新导向**：这是研究生论文课题，核心是方法创新（结构重参数化），不做工程化堆叠，也不把 loss 调参 / 训练技巧包装成创新贡献。
- **单 seed（2333）**：当前阶段只用单 seed 验证有效性与创新性，不做多 seed 统计显著；如需论文级结果，再按需补充。
- **训练协议**：从 ChangeMamba 起，后续 Mamba-based 的 BCD 模型都是「test 集当验证集、每 epoch 在 test 上挑 best」，本项目沿用同一协议。当前处于**长期迭代改进期**，暂不做正式 train/val/test 分离 + corrected-reproduction 的协议重跑，等方法收敛后再做。

## 方法（STR-RepNet / Clean TAR-DCR）

- **Encoder**：Frozen VMamba-Tiny Siamese（不改结构、不参与训练；预训练权重 `vssm_tiny_0230_ckpt_epoch_262.pth`）
- **TAR**（Temporal Algebraic Re-parameterization）：四级二时相 bridge，训练期 Concat + Sum + signed-Diff 三路 → 部署期折叠为单个 1×1
- **DCR**（Decoder-wide Compositional Re-parameterization）：RepDW3 / RepPW1x1 / RepPairFuse1x1，整个 decoder 的可折叠算子图
- **Edge-Basis**（Run3 迭代，已验证未达判据、未纳入主方法）：在 `DCRDecoder.refine.dw` 增加可折叠的 Sobel-X/Y 结构边缘基分支，部署仍折叠为单个 DW3×3（`--use_edge`）
- **IBAS**（Run4 迭代，辅助训练机制）：最终 decoder feature 上挂 161 参数零初始化训练期边界头，GT 内侧边界（`B⁺=Y−Erode3×3(Y)`）在线生成，`BCE+Dice` 加权 0.1，`switch_to_deploy` 整支删除（`--use_boundary_aux`，部署 +0 参数/FLOPs）
- **NSCR-Fuse**（Run6 迭代，已按判据停止）：`DCRDecoder` 的 fuse1+fuse2 加零初始化原生尺度 BN 分支（`BN_L(L)` + `U(BN_H(H))`），利用逐通道仿射与 bilinear 插值可交换性，部署时吸收回原 cross-scale 1×1（`--use_nscr`，部署 +0 参数/FLOPs）
- **PBRU**（Run7 迭代，已按判据 WEAK 归档）：Phase-Basis Reparameterized Upsampling 输出头——部署为单个 `Conv2d(D→C·r²,1)` + `PixelShuffle(r)`（全分辨率、无插值）；训练期叠加 4 个零初始化相位基分支 `{1,u,v,u·v}`（BN γ=β=0），FP64 解析折叠吸收（`--head_mode pixelshuffle --use_pbru 1`，部署 +0 参数/FLOPs vs 纯 PixelShuffle，经 D\*=158 预算搜索与 Run2 部署预算持平）
- **MPCR-Fine**（Run8 迭代，已按判据 FAIL 停止）：Fine-stage Multi-Partition Channel Reparameterization——`refine.pw` 加两个同参数量零初始化 grouped 1×1 分支（groups=4，连续/交织互补分区），`W_eq = W_core + Σ P⁻¹W̃P` 解析折叠回原 dense PW（`--use_mpcr`，部署 +0 参数/FLOPs，预算机器验证 D\*=160、Δ=0）
- **BiFTR**（Run9 迭代，最后一次结构搜索，已按判据 FAIL 结束）：Bi-sided Frozen-to-Trainable Transition Reparameterization——包装 `encoder.layers[1].downsample` 的 192→384 stride-2 Conv（frozen stage2→trainable stage3 边界），训练期 `y=(I+Δout)W((I+Δin)x)`（Δ 零初始化 1×1，base 冻结），部署 `W_eq=AWB、b_eq=Ab` 折叠回原单个 Conv（`--use_biftr`，部署 +0 参数/FLOPs，预算 equality audit Δ=0）

- 模型入口：`models/changedetection/models/STRRepNet.py`（`STRRepNet`）
- 核心文件：`reparam.py`（代数折叠原语）、`tar.py`（二时相 bridge）、`dcr_decoder.py`（多尺度解码器）
- `rep_mode`：`plain`（单路径对照）/ `tar`（时相 rep）/ `full`（全开）
- 损失 `CE + 2.0 × Lovász-Softmax`；300 epoch，batch 16，seed 2333
- 部署复杂度（不含 frozen encoder）：**中间+Decoder 约 0.84M 参数 / 0.92G FLOPs**；整网约 29.5M，可训练仅 1.56M
- 折叠等价性：FP32 eval `max_abs_error ≈ 1.7e-5`（FP32 累加固有误差，argmax/F1 不变）

> 旧 HAM-CD baseline 已归档在 git tag `baseline-hamcd-run1`
> （WHU 0.9500 / LEVIR 0.9211 / CDD 0.9879 / SYSU 0.8299）。

## 实验结果（Run1）

### Full vs baseline（4 数据集）

| 数据集 | HAM-CD baseline | TAR-DCR Full | ΔF1 |
|---|---|---|---|
| CDD | 0.9879 | 0.9768 | -1.11 |
| WHU | 0.9500 | 0.9403 | -0.97 |
| LEVIR | 0.9211 | 0.9033 | -1.78 |
| SYSU | 0.8299 | 0.8124 | -1.75 |

### 消融矩阵（4 数据集 × 3 模式）

| 数据集 | Plain | TAR | Full |
|---|---|---|---|
| CDD | 0.9732 | 0.9735 | 0.9768 |
| WHU | 0.9355 | 0.9388 | 0.9403 |
| LEVIR | 0.9009 | 0.9003 | 0.9033 |
| SYSU | 0.8114 | 0.8058 | 0.8124 |

- 部署统一：**28.83M 参数（-20%）、12.61G FLOPs（-22%）、可训练 1.56M**。
- 结论：DCR（decoder-wide 组合折叠）稳定正贡献（4 数据集 Full > Plain）；TAR（时相三路 rep）贡献微弱/为负；Full 整体比 HAM-CD baseline 低 1~2 点，换来 20% 复杂度下降。
- 下一步候选：decoder 宽度 D=160→192（仍轻），或按设计文档 §82 的预案调整。

## 实验结果（Run2：BN-FR + 公平性诊断）

Run2 在 Run1 基础上做了两处改造（部署图/参数/FLOPs 完全不变）：
- **BN-FR**：每个线性分支独立 BN + 可折叠 `α` 残差（替换 Run1 的共享 BN）；
- **encoder 解冻诊断**：`frozen`（全冻结）/ `last2`（解冻 stage3+stage4）。

| 变体 | LEVIR | SYSU |
|---|---|---|
| full_frozen（BN-FR，冻结） | 0.9094 | 0.8252 |
| full_last2（BN-FR + 解冻后两级） | 0.9144 | 0.8345 |
| dcr_frozen（BN-FR，仅 DCR） | 0.9090 | 0.8201 |

对比（F1）：

| 数据集 | Run1 Full | Run2 full_frozen | Run2 full_last2 | HAM-CD baseline |
|---|---|---|---|---|
| LEVIR | 0.9033 | 0.9094 | 0.9144 | 0.9211 |
| SYSU | 0.8124 | 0.8252 | 0.8345 | 0.8299 |

结论：
- **BN-FR 有效（零部署增量）**：full_frozen 相对 Run1 Full 提升 LEVIR +0.61 / SYSU +1.28。
- **冻结 encoder 是 Run1 落后 baseline 的主因**：解冻后两级再提升 +0.50 / +0.93，`full_last2` 在 SYSU 上反超 baseline（0.8345 > 0.8299）。
- temporal aux 贡献：LEVIR 上很小（full≈dcr，+0.04），SYSU 上 +0.51。

## 实验结果（Run3：Edge-Basis，已完成）

- 单变量：`DCRDecoder.refine.dw` 增加 Sobel-X/Y 结构边缘基分支（β 零初始化，部署折叠 +0 参数/FLOPs）。
- 4 数据集并行（full + last2 + use_residual 1 + use_edge 1，seed 2333，300 epoch）：

| 数据集 | Run3 Edge-Basis | Run2 full_last2 锚点 | ΔF1 | HAM-CD baseline |
|---|---|---|---|---|
| LEVIR | 0.9135 | 0.9144 | -0.09 | 0.9211 |
| CDD | 0.9841 | 0.9842 | -0.01 | 0.9879 |
| WHU | 0.9497 | 0.9514 | -0.17 | 0.9500 |
| SYSU | 0.8311 | 0.8345 | -0.34 | 0.8299 |

- 判据（LEVIR F1≥0.9175、Recall≥0.907）**未达成**：F1=0.9135 / Recall=0.9046。
- 细节：LEVIR Recall 相对锚点微升 +0.14pp（0.9032→0.9046，边缘基对召回有微弱正作用），但 Precision 掉 -0.35pp（0.926→0.9225），净 F1 微负；4 数据集均无净正贡献（SYSU 掉点最多）。
- 部署不变：28.83M 参数 / 12.61G FLOPs；fold 误差 1.0e-5~7.2e-5（<1e-4）。
- 结论：Edge-Basis 不作为主方法，当前最佳仍为 Run2 full_last2；下一步候选见「训练期边界监督（部署删除）」方向。

## 实验结果（Run4：IBAS，已完成）

- 单变量：训练期内侧边界辅助监督 IBAS（boundary head `Conv2d(160→1)` 零初始化、`B⁺=Y−Erode3×3(Y)` 内侧边界、`BCE+Dice`、λ=0.1），部署删除 +0 参数/FLOPs。
- 基线：Run2 full_last2；4 数据集并行（GPU1）；`full + last2 + use_residual 1 + use_edge 0 + use_boundary_aux 1 + boundary_weight 0.1`，seed 2333，300 epoch。

| 数据集 | Run4 IBAS | Run2 full_last2 锚点 | ΔF1 | Run3 Edge-Basis | HAM-CD baseline |
|---|---|---|---|---|---|
| LEVIR | 0.9145 | 0.9144 | +0.01 | 0.9135 | 0.9211 |
| CDD | 0.9839 | 0.9842 | -0.03 | 0.9841 | 0.9879 |
| WHU | 0.9508 | 0.9514 | -0.06 | 0.9497 | 0.9500 |
| SYSU | 0.8324 | 0.8345 | -0.21 | 0.8311 | 0.8299 |

- 判据（LEVIR：F1≥0.9175 / IoU≥0.8475 / Precision≥0.9240）**未达成**：F1=0.9145 / IoU=0.8424 / Precision=0.9269（F1 低于失败线 0.9159 → 按预注册协议不再做 λ sweep）。
- 细节：IBAS 相对 Run3 全面更好（LEVIR +0.10 / WHU +0.11 / SYSU +0.13，CDD -0.02），且 LEVIR Precision 相对锚点 +0.09pp（0.9260→0.9269，实现「不丢精度」的设计目标），但 Recall 微降（0.9032→0.9023），净 F1 持平锚点、无增益。
- 部署不变：28.83M 参数 / 12.61G FLOPs；fold 误差 8.6e-6~3.8e-5（<1e-4）。
- 结论：IBAS 是「中性」机制（持平锚点、优于 Run3），未带来判据级提升；当前最佳仍为 Run2 full_last2。WHU/SYSU 仍反超 baseline（+0.08 / +0.25）。
- 设计文档：[`docs/temporary/STR-RepNet_Run4_IBAS_训练期内侧边界辅助监督方案.md`](docs/temporary/STR-RepNet_Run4_IBAS_训练期内侧边界辅助监督方案.md)。

## 实验结果（Run5：回退 Run2 + BOTR，已完成）

- **代码回退**：活跃方法主路径恢复 Run2 版本（移除 Edge-Basis/IBAS 活跃代码；历史记录保留）；P0 修复 FP64 folding（组合全程 float64、最后一次性 cast FP32）；P2 拆分 train/deploy 参数统计。
- **主方案 BOTR**（Bi-Order Temporal Re-parameterization）：TAR 各尺度加零初始化 reverse-concat `W_r[Q,P]` 训练分支（独立 BN），部署时通道置换吸收回单个 temporal 1×1 → **+0 部署参数/FLOPs**（`--use_botr`）。
- 最小消融（LEVIR × 3）：**D0** `encoder_train=full` 上界诊断 / **C1** `temporal_swap_prob=0.5` 必要对照 / **M1** `use_botr=1` 主实验；A0 = Run2 full_last2 锚点（不重跑）。

| 实验 | F1 | Recall | Precision | IoU | ΔF1 vs 锚点 0.9144 |
|---|---|---|---|---|---|
| C1_swap_only | 0.9143 | 0.9043 | 0.9245 | 0.8421 | -0.01pp |
| M1_BOTR | 0.9144 | 0.9046 | 0.9244 | 0.8423 | ±0.00 |
| D0_full_encoder | **0.9168** | 0.9056 | 0.9282 | 0.8463 | **+0.24pp** |

- **M1/BOTR 判据 FAIL**（F1=0.9144 < 失败线 0.9159，恰在锚点线上）→ 按预注册协议停止 BOTR，不扩四数据集。
- **归因**：M1 − C1 = +0.0001pp → BOTR 未提供超出普通 swap 增强的收益；C1 本身也 ≈ 锚点（swap 增强同样中性）。
- **D0/full-encoder**：+0.24pp（中性偏正，未达 +0.30pp「主要瓶颈」线）；是 Run5 三者最佳、也是当前 LEVIR 最佳（仍低于 HAM-CD baseline 0.9211 约 -0.43pp）。
- 部署不变：三组均 28.829M / 12.6062G；fold 误差 1.4~1.6e-5，argmax 分歧 0。
- 结论：Run5 三个假设（encoder 适配 / 数据级时序对称 / 结构级时序重参数化）均无判据级突破 → 下一轮按文档预设进入「高分辨率/小目标信息保真 + budget-neutral decoder channel allocation」，不再给 TAR/edge/loss 叠模块；`encoder_train=full` 可作为训练协议候选（零部署增量）。
- 设计文档：[`docs/temporary/STR-RepNet_Run5_回退Run2与BOTR最小消融实验方案.md`](docs/temporary/STR-RepNet_Run5_回退Run2与BOTR最小消融实验方案.md)。

## 实验结果（Run6：NSCR-Fuse，已完成）

- **主方案 NSCR-Fuse**（Native-Scale Commutative Re-parameterized Fusion）：在 DCR 的 `fuse1+fuse2` 各加零初始化原生尺度 BN 分支（`BN_L(L)` + `U(BN_H(H))`），利用「逐通道仿射 × bilinear 插值可交换」在部署时精确吸收回原 cross-scale 1×1 → **+0 部署参数/FLOPs**（`--use_nscr`）。
- 训练：4 数据集并行（GPU0 整卡，用户指令并行）。

| 数据集 | Run6 M1_NSCR | Run2 full_last2 锚点 | ΔF1 | HAM-CD baseline |
|---|---|---|---|---|
| LEVIR | 0.9140 | 0.9144 | -0.04pp | 0.9211 |
| WHU | 0.9510 | 0.9514 | -0.04pp | 0.9500 |
| SYSU | 0.8337 | 0.8345 | -0.08pp | 0.8299 |
| CDD | 0.9839 | 0.9842 | -0.03pp | 0.9879 |

- **LEVIR 判据 FAIL**（F1=0.9140 < 失败线 0.9159，且 Precision=0.9208 < 0.9220 双触发）→ 按预注册协议**停止 NSCR**，不做 scope/gamma sweep。
- 关键细节：LEVIR **Recall +0.41pp**（0.9032→0.9073，原生尺度 lateral 分支确实找回细节）但 **Precision -0.52pp**（0.9260→0.9208），净 F1 微降——精确命中方案 H1 的失败模式（「只是让 decoder 更激进」）。四数据集均微降（-0.03~-0.08pp，Macro 92.07% vs 锚点 92.11%）。
- 部署不变：28.829M / 12.6062G；fold 误差 5.7e-6~3.5e-5；argmax 分歧 0。
- 诊断结论（`analyse/levir_error_profile.py`）：small（≤502px）pixel recall 仅 80.8%、**24% 小目标完全漏检**（medium/large 92%/91%）；D0 full-encoder 无法修复（81.5%、漏检率 23.7%）→ 瓶颈在 decoder 侧；测试尺度 320 仅抬 Precision（Recall 反降）→ 非输入分辨率问题。按失败预案（情况 A：decoder 小目标保真瓶颈）下一轮转 **PBRU（Phase-Basis Reparameterized Upsampling + 精确预算回收）**。
- 设计文档：[`docs/temporary/STR-RepNet_Run6_下一步改进方向与实验设计_NSCR-Fuse.md`](docs/temporary/STR-RepNet_Run6_下一步改进方向与实验设计_NSCR-Fuse.md)。

## 实验结果（Run7：PBRU，进行中）

- **主方案 PBRU**（Phase-Basis Reparameterized Upsampling）：输出头改为单个 `Conv2d(D→C·r²,1)` + `PixelShuffle(4)`（全分辨率、无插值）；训练期叠加 4 个零初始化（BN γ=β=0）相位基分支 `{1, u, v, u·v}`（粗/水平/垂直/对角），FP64 解析折叠吸收回单投影 → 部署与纯 PixelShuffle 完全一致、与 Run2 部署预算持平（D\*=158）。
- Phase 0（已完成）：
  - **机器预算搜索**（`analyse/search_pbru_budget.py`）：锚点 = Run2 部署图 = 28.828706 M / 12.6062 G；逐 D 下降搜索得 **D\*=158**（PBRU 部署 28.818618 M / 12.6055 G）。
  - **阶段判别力预检**（`analyse/levir_stage_discriminability.py`，Run2 LEVIR 锚点，200 对 test 前 100 拟合/后 100 held-out）：训练头 AUROC@64=0.9881 > d1 全协方差 LDA 上界 0.9766（headroom −0.0115）→ 冻结 Run2 解码器下 d1 无静态粗尺度线性空间；PBRU 假设是**训练期端到端学习的亚单元相位自由度**，由 C0/C1/M1 对照裁决。
  - 等价性验证全通过：冒烟（epoch-0 输出与 use_pbru=0 逐位一致、γ 梯度非零、部署 +0、fold 9.9e-05、argmax=0）；重参数化等价性 T0（相位基/PixelShuffle 通道序/相位展开，精确）+ T1（PBRUHead 折叠 1.4e-06、FP64 代数 2.2e-15）+ T2（全模型五图均 <2e-4、argmax 全 0）。
- Phase 1（已完成）：**C0_Bilinear_D158 + M1_PBRU_D158** × **LEVIR + WHU**（GPU0 并行 4 job，300 epoch）：
  | exp | LEVIR F1 | WHU F1 | ΔF1 vs A0 |
  |---|---|---|---|
  | C0_Bilinear_D158 | 0.9140 | 0.9522 | −0.04 / +0.08pp |
  | M1_PBRU_D158 | **0.9161** | 0.9506 | **+0.17 / −0.08pp** |
  - **M1/LEVIR = WEAK**（0.9161 ∈ [0.9159, 0.9175)，IoU 0.8452 差 0.23pp；Precision 0.9287 ✓，Deploy 28.819M/12.6055G ✓，argmax 0 ✓）。关键动态：C0 宽度控制 Recall↑/Precision↓（F1 持平），M1 相位头把 Precision 拉回 0.9287 并保住 Recall → **M1−C0 = +0.21pp**；PBRU 分支学到非平凡结构（γ 范数 pxy 0.41 > px/py 0.23~0.25 > coarse 0.05）。
  - WHU：M1 −0.08pp（0.15pp 容忍内）；M1−C0 = −0.16pp。
- Phase 2（已完成）：**C1_PixelShuffle_D158** × LEVIR + WHU 归因：
  | 效应拆解 | LEVIR | WHU |
  |---|---|---|
  | 宽度（C0−A0） | −0.04pp | +0.08pp |
  | PixelShuffle 拓扑（C1−C0） | +0.12pp | −0.35pp |
  | phase-basis rep（M1−C1） | **+0.09pp（rep-weak）** | **+0.19pp（rep-supported）** |
  - 一致性机制：rep 在两个数据集上都表现为**精度提升**（+0.63/+0.44pp，相位基分支≈输出置信度锐化）；拓扑 LEVIR 正、WHU 负。
  - **Run7 结论**：M1/LEVIR WEAK（0.9161<0.9175）+ rep 效应不一致 → 按 doc §28 **不扩 SYSU/CDD**；PBRU 不能宣称稳定普适的 rep 增益，只能作为「输出头相位置信度提升 + 可折叠相位基」的单 seed 观察记录（LEVIR +0.17pp ≈ 拓扑 +0.12pp + rep +0.09pp）。硬条件全程合格（预算内、argmax=0、fold ~1.4e-5）。
- 判据（LEVIR 锚点 F1=0.9144）：M1 PASS = F1≥0.9175 且 IoU≥0.8475 且 Precision≥0.9230 且预算/argmax 合格；FAIL = F1<0.9159 或 Precision<0.9220 → 停止 PBRU 不救机制。四数据集扩展需 M1 系统 PASS 且 M1−C1 有可辨识 rep 增益。
- 设计文档：[`docs/temporary/STR-RepNet_Run7_PBRU_修改方案与实验设计.md`](docs/temporary/STR-RepNet_Run7_PBRU_修改方案与实验设计.md)；`train_scripts/TAR-DCR/Run7/README.md`。

## 实验结果（Run8：MPCR-Fine，已完成 —— M1 FAIL）

- **主方案 MPCR-Fine**（Fine-stage Multi-Partition Channel Reparameterization）：Run7 归因证明输出侧已榨干 → 转向 fine-stage 表征。只在 `refine.pw` 加两个同参数量 grouped 1×1 训练分支（groups=4），C0=两个连续分区（容量控制）、M1=连续+交织互补分区；BN γ=β=0 零初始化保证 epoch-0 输出与 Run2 逐位一致，部署经 `W_eq = W_core + P0⁻¹W̃0P0 + P1⁻¹W̃1P1` 解析折叠回原 dense PW（`--use_mpcr`，部署 +0 参数/FLOPs）。
- 预训练门槛全部通过：T0 置换/嵌入精确（P⁻¹GP 2.1e-14）；T1 折叠 4.8~6.0e-06 + FP64 代数 ~7e-15；T2 全模型 5.9~8.8e-05、argmax 全 0；冒烟 epoch-0 与 Run2 **逐位一致**（max_diff=0.0）、部署 Params/FLOPs 与锚点精确相等；**预算机器验证 D\*=160、ΔParams=ΔFLOPs=0**。
- 结果（LEVIR + WHU，GPU0 并行 4 job）：

  | exp | LEVIR F1 | WHU F1 | ΔF1 vs A0 |
  |---|---|---|---|
  | C0_MPCR_Same2 | 0.9135 | 0.9520 | −0.09 / +0.06pp |
  | M1_MPCR_Multi2 | **0.9129** | 0.9500 | **−0.15 / −0.14pp** |

- **M1/LEVIR = FAIL**（0.9129 < 0.9159），rep 归因 **rep-not-supported**（M1−C0 = −0.06pp；
  动力学仍为 Recall↓/Precision↑ 的"置信度锐化"签名）。机制画像：M1 的 d1 对角 LDA
  AUROC@64 0.9372 < A0 0.9397 → MPCR 未改善最终表征（分支 γ~0.9 学到了结构但折叠后无净益）。
  按 §18-E **停止 MPCR，不救机制**；SYSU/CDD 不启动；硬条件全程合格（预算内、argmax=0）。
- 设计文档：[`docs/temporary/STR-RepNet_Run8_MPCR-Fine_设计与实验方案.md`](docs/temporary/STR-RepNet_Run8_MPCR-Fine_设计与实验方案.md)；`train_scripts/TAR-DCR/Run8/README.md`。

## 实验结果（Run9：BiFTR，已完成 —— M1 FAIL，结构搜索结束）

- **主方案 BiFTR**（Bi-sided Frozen-to-Trainable Transition Reparameterization）：位置离开 decoder，取 `encoder.layers[1].downsample` 的 192→384 stride-2 Conv（`last2` 下唯一的 frozen→trainable 边界）；训练期 `y=(I+Δout)W((I+Δin)x)`（Δ 零初始化 1×1，base 冻结；C0=仅 post，M1=双侧），部署 `W_eq=AWB、b_eq=Ab` 折叠回原单个 Conv（+0 部署）。参数化家族 = 串行乘性双侧 basis（含 ΔoutWΔin 交叉项），非并行 additive branch。
- 预训练门槛全部通过（GPU0）：T0 AWB FP64 代数 1.71e-10；T1 折叠 post 7.9e-06 / bi 8.3e-06 + FP64 代数 ~1e-14；T2 全模型 post 6.2e-05 / bi 8.4e-05、argmax 全 0；冒烟 epoch-0 与 Run2 **逐位一致**（max_diff=0.0）；**预算 equality audit C0/M1 ΔParams=ΔFLOPs=0**。
- 结果（LEVIR + WHU，GPU0 并行 4 job）：

  | exp | LEVIR F1 | WHU F1 | ΔF1 vs A0 |
  |---|---|---|---|
  | C0_FTR_Post | 0.9142 | 0.9496 | −0.02 / −0.18pp |
  | M1_BiFTR | **0.9147** | 0.9506 | **+0.03 / −0.08pp** |

- **M1/LEVIR = FAIL**（0.9147 < 0.9159），rep 归因在 not-supported/weak 边界（M1−C0 ≈ +0.05pp，
  Precision +0.17pp）。权重空间：‖Δ‖ 学到 0.67/1.21、核相对变化 13%、**交叉项比率 5.65%**
  —— 双侧乘性耦合确实被利用，但净效应中性（Recall −0.05pp / Precision +0.17pp，
  第三次出现"置信度锐化"签名）。硬条件全程合格（预算内、argmax=0、fold 6.7e-05）。
- **§22-A 预注册裁决：结束结构搜索**——Run9 是最后一次正交结构尝试（encoder 侧串行乘性
  重参数化），未达 PASS → **不设计 Run10**，进入论文收尾（doc §33）：清理主方法
  （TAR+DCR+BN-FR，Run2）、统一 Params/FLOPs 与部署误差、train/deploy 图、组织 Run1–Run9
  证据（正文保留最有信息量的负结果）、方法冻结后补 3-seed × 4 数据集正式稳定性。
- 设计文档：[`docs/temporary/STR-RepNet_Run9_BiFTR_设计与实验方案.md`](docs/temporary/STR-RepNet_Run9_BiFTR_设计与实验方案.md)；`train_scripts/TAR-DCR/Run9/README.md`。

## 参考文献

调研文献按 [`docs/temporary/过去的想法/STR-RepNet_结构重参数化_2024-2026文献调研与创新空白.md`](docs/temporary/过去的想法/STR-RepNet_结构重参数化_2024-2026文献调研与创新空白.md)
的分类整理，索引见 [`docs/参考文献/文献索引.md`](docs/参考文献/文献索引.md)。
索引内按「题名 / 年份 / venue / 层级（CCF-A 或 SCI）/ 一作+机构 / 官方链接 / 相关子方向 / Top-15 优先精读」登记每篇文献，
据此可直接定位到 `docs/参考文献/` 下的 PDF。

## 目录结构

```
models/                        # 全部代码
  changedetection/
    configs/                   # yacs 配置 + tiny yaml
    datasets/                  # DataLoader（A/B/label + list 格式）
    models/                    # 6 个文件：
      Mamba_backbone.py        #   Frozen VMamba-Tiny 编码器
      reparam.py               #   代数折叠原语（RepDW3/RepPW1x1/RepPairFuse1x1/NSCR/PBRUHead）
      tar.py                   #   TAR 二时相 bridge
      dcr_decoder.py           #   DCR 多尺度解码器
      STRRepNet.py             #   顶层网络（switch_to_deploy，head_mode/use_pbru）
      __init__.py
    script/
      train.py                 # 训练入口（rep_mode + 冻结 encoder + 部署测试）
      smoke_test.py            # 冒烟测试
      test_reparam_equivalence.py  # 折叠等价性测试
    utils_func/                # metrics / lovasz / edge loss
  classification/              # VMamba（VSSM）编码器实现
  kernels/selective_scan/      # 自定义 CUDA 内核（已适配 sm_120 + CUDA 13）
train_scripts/
  baseline/Run1/               # HAM-CD baseline 启动脚本（历史，已归档）
  TAR-DCR/Run1/                # TAR-DCR 启动脚本（A0_Plain / A1_TAR / A2_Full）
  TAR-DCR/Run7/                # PBRU 启动脚本（C0/C1/M1 × 4 数据集，D*=158）
  TAR-DCR/Run8/                # MPCR-Fine 启动脚本（C0_MPCR_Same2 / M1_MPCR_Multi2）
  TAR-DCR/Run9/                # BiFTR 启动脚本（C0_FTR_Post / M1_BiFTR）
analyse/                       # 分析工具
  extract_metrics_to_excel.py  #   outputs → docs/experiment_metrics.xlsx
  models_to_txt.py             #   models 代码快照 + 指标 → docs/temporary/*.txt
  search_pbru_budget.py        #   Run7 机器预算搜索（D*）
  levir_stage_discriminability.py  # Run7 阶段判别力预检（LDA AUROC）
  search_run8_budget.py        #   Run8 预算机器验证（必须 D*=160）
  levir_fine_stage_profile.py  #   Run8 精细阶段表征画像（机制解释）
  search_biftr_budget.py       #   Run9 预算 equality audit（必须 Δ=0）
  levir_biftr_profile.py       #   Run9 冻训边界表征画像（机制解释）
outputs/                       # 训练日志（训练结束后下载到这里）
docs/                          # 项目文档（temporary / 参考文献）
```

## 服务器环境（RSML-3）

- 环境 `strrep`（本项目）与 `mamba_base`（Mamba 模板）均已升级到
  **torch 2.14.0+cu132 / cuBLAS 13.4.0.1 / Python 3.10**，规避了 torch 2.10 在
  Blackwell（sm_120）上的 cuBLAS Lt 崩溃问题。`strrep` 额外补装 `timm fvcore yacs`。
- `mamba-ssm 2.3.2.post1` 已针对 torch 2.14 重新编译（源码 `-std=c++17`→`c++20`，
  移除 CUDA 13 不再支持的 sm_75/80/87，保留 sm_120），并重打 `mamba_simple.py` 补丁
  （`dt_proj.weight @ dt.t()` → `F.linear(dt, self.dt_proj.weight)`）。
- 自定义内核（`selective_scan_cuda_oflex`、`selective_scan_cuda_oflex_rh`）已在
  `strrep` 中编译（`-std=c++20`，`arch=compute_120,code=sm_120`），并适配 CUDA 13 的
  CUB API 变更（`cub::LaneId`/`cub::CTA_SYNC` 已替换）。
- 关键路径：
  - 代码 `/home/yqwang/projects/STR-RepNet/`
  - 数据集 `/share_datasets/CD/{CDD,LEVIR,SYSU,WHU}-CD-256/`
  - 预训练权重 `/home/yqwang/projects/STR-RepNet/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth`
  - checkpoint `/share_datasets/yqwang/checkpoints/STR-RepNet/baseline/Run1/<dataset>/`
  - 训练日志 `/home/yqwang/outputs/STR-RepNet/baseline/Run1/<dataset>/train_log.txt`

## 数据集（A/B/label + list 格式）

| 数据集 | Train / Val / Test | 目标 F1 | 说明 |
|---|---:|---:|---|
| CDD-CD-256 | 10,000 / 2,998 / 3,000 | ≈97 | 抗伪变化，label 为 JPG（阈值 ≥128） |
| LEVIR-CD-256 | 7,120 / 1,024 / 2,048 | ≈92 | 建筑小目标、极不平衡 |
| SYSU-CD-256 | 12,000 / 4,000 / 4,000 | ≈84 | 通用地表变化 |
| WHU-CD-256 | 5,947 / 743 / 744 | ≈95 | 建筑小目标、极不平衡 |

## 训练 / 测试

1. 本地改代码 → `python .claude/_deploy.py` 同步到服务器。
2. 服务器启动（`nohup`，每脚本带断点续训重试循环），TAR-DCR 脚本在
   `train_scripts/TAR-DCR/Run1/{A0_Plain,A1_TAR,A2_Full}/`：
   ```bash
   cd /home/yqwang/projects/STR-RepNet/train_scripts/TAR-DCR/Run1/A2_Full
   nohup bash train_SYSU-CD-256.sh > /dev/null 2>&1 &
   ```
   脚本内通过 `--rep_mode plain|tar|full` 控制消融版本。
3. 训练日志格式：
   - 开头：全部配置 + 总参数量 + 可训练参数量 + FLOPs(G)。
   - 每个 epoch 一行：CE / Lovasz / Total 三个 loss + Recall / Precision / OA / F1 / IoU / Kappa 六项指标。
   - 结尾：`=== TEST RESULTS ===` 部署参数量 + 部署 FLOPs + 折叠误差 + 六项指标（用 deploy 图测试）。
4. checkpoint：每 run 一个文件夹，只放 `last.pth`（断点续训）与 `best_F1=xxx.pth`（测试用）。
5. 训练结束后把 `train_log.txt` 下载回本地 `outputs/`（`baseline/Run1/` 或 `TAR-DCR/Run1/`）。

## 运行监控 / 分析

```bash
python .claude/_monitor.py              # 查看训练进度 + GPU 占用
python .claude/_ssh.py '<cmd>'          # 通用 SSH 执行
python analyse/extract_metrics_to_excel.py   # outputs → docs/experiment_metrics.xlsx
python analyse/models_to_txt.py --tag TAR-DCR --run Run1  # models 快照 + 指标 → docs/temporary/
```

## 注意事项

- RTX 5090（Blackwell sm_120）在 torch 2.10 下偶发 `CUBLAS_STATUS_NOT_INITIALIZED`，
  升级 torch 2.14+cu132（cuBLAS 13.4.0.1）后已消除；训练脚本仍保留 `num_workers=4` +
  断点续训重试循环兜底（崩溃自动从 `last.pth` 恢复）。
- `torch.load` 加载含优化器状态的 `last.pth` 需 `weights_only=False`（torch 2.6+ 默认
  `weights_only=True` 会拒收非张量对象），train.py 已处理。
- `.sh` 脚本需 LF 行尾（Windows 编辑后 `_deploy.py` 上传，服务器端会统一转 LF）。
