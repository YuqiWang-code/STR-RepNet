# TAR-DCR Run5 — 回退 Run2 + BOTR 最小消融

## 本轮改动（代码）

- 回退活跃方法主路径到 Run2（`65958b0`）：`STRRepNet/dcr_decoder/reparam/tar/train` 恢复 Run2 版本，
  移除 Run3 Edge-Basis 与 Run4 IBAS 的活跃代码（历史脚本/文档/快照保留；`boundary_loss.py` 保留但不 import）。
- **P0 修复**：`fold_conv_bn` 返回 float64，所有分支组合（含 BOTR 通道置换）全程 float64，只在写入 deploy Conv 时 cast FP32。
- **BOTR**（主方案）：`TemporalRep1x1` 增加零初始化 reverse-concat 分支 `W_r[Q,P]`（独立 BN）；
  部署时通道置换回 `[P,Q]` 吸收进同一个 temporal 1×1 → **部署 Params/FLOPs 与 Run2 完全一致（+0）**。
- **P2 修复**：TEST RESULTS 新增 `[BOTR]/[ENCODER-TRAIN]/[TEMPORAL-SWAP-PROB]/[REPARAM-ARGMAX-DISAGREE]`；
  Excel/快照拆分为 TrainGraphParams/TrainableParams/DeployParams/DeployFLOPs 四列。
- 详细设计：`docs/temporary/STR-RepNet_Run5_回退Run2与BOTR最小消融实验方案.md`。

## 最小实验组（第一阶段只跑 LEVIR × 3，GPU1 并行）

| 组 | 配置差异 | 目的 |
|---|---|---|
| A0 | Run2 full_last2 已有锚点（不重跑）：F1=0.9144 / Recall=0.9032 / Precision=0.9260 / IoU=0.8424 | 基线 |
| **D0_full_encoder** | `encoder_train=full`（+lr_ratio 0.1） | early encoder adaptation 上界诊断 |
| **C1_swap_only** | `temporal_swap_prob=0.5` | 普通 A/B 交换增强是否足够 |
| **M1_BOTR** | `use_botr=1`（swap 0.0） | 双顺序结构重参数化主实验 |

- 公共配置：`rep_mode=full / use_residual=1 / seed=2333 / 300 epoch / batch 16 / lr 1e-4 / lovasz 2.0`。
- M1 的 WHU/SYSU/CDD 脚本已预写，**仅当 M1/LEVIR 通过判据后才启动**。

## 预注册判据（LEVIR，锚点 F1=0.9144 / Recall=0.9032 / Precision=0.9260 / IoU=0.8424）

- **M1 PASS**：F1 ≥ 0.9175 且 Recall ≥ 0.9065 且 Precision ≥ 0.9230 且 IoU ≥ 0.8475。
- **WEAK**：0.9159 ≤ F1 < 0.9175 → 只记录，不扩四数据集、不做 sweep。
- **FAIL**：F1 < 0.9159，或 Recall 升但 Precision 掉 >0.40pp → 停止 BOTR。
- **归因**：若 C1 ≥ M1 − 0.10pp，则 BOTR 收益可被普通 swap 解释；若 M1 − C1 ≥ 0.20pp 且 M1 PASS，支持结构级收益。
- **D0 解读**：ΔF1 ≥ +0.30pp 说明 stage1/2 未适配是主要瓶颈；< +0.15pp 则不再投入 encoder。
- M1 过线后的四数据集标准：Macro F1 ≥ 92.25%、LEVIR ≥ 91.75%、任一数据集退化 ≤0.15pp、至少 3/4 非负。
