# TAR-DCR Run4 — IBAS（训练期内侧边界辅助监督，单变量）

## 本轮改动

- 在 `STRRepNet` 最终 decoder feature `d1` 上挂一个训练期边界头 `Conv2d(160→1,1)`（161 参数，零初始化）。
- GT 边界在线生成：`B+ = Y − Erode3×3(Y)`（变化对象**内侧一圈**；valid 也腐蚀 3×3 排除 255 padding）。
- 边界 loss：`BCEWithLogits + Dice`，总 loss `CE + 2·Lovász + 0.1·L_b`（λ 固定 0.1，不 sweep）。
- 部署时 `switch_to_deploy()` **删除整个 boundary head**：部署参数/FLOPs 与 Run2 完全一致（+0）。
- 其余全部不动（TAR / DCR / reparam / encoder / 主 loss / 主 head）。
- 详细设计见 `docs/temporary/STR-RepNet_Run4_IBAS_训练期内侧边界辅助监督方案.md`。

## 实验

- 数据集：4 个并行（GPU1）——LEVIR（主验证）+ CDD / WHU / SYSU（防掉点对照）。
- 配置：`rep_mode=full`、`encoder_train=last2`、`encoder_lr_ratio=0.1`、`use_residual=1`、
  `use_edge=0`、`use_boundary_aux=1`、`boundary_weight=0.1`、`seed=2333`、300 epoch、batch 16。
- 基线（不重跑）：Run2 `full_last2` —— CDD 0.9842 / LEVIR 0.9144 / WHU 0.9514 / SYSU 0.8345。

## 锚点（Run2 full_last2 / LEVIR）

- F1=0.9144，IoU=0.8424，Precision=0.9260，Recall=0.9032

## 成功判据（LEVIR 主判据，提前锁死）

- 成功：F1 ≥ 0.9175（+0.31pp）且 IoU ≥ 0.8475 且 Precision ≥ 0.9240。
- 失败：F1 < 0.9159（<+0.15pp）→ 不做 λ sweep，否决机制；
  或 F1 微涨但 Precision/Recall 以 >0.3pp 代价交换 → 机制不可靠。
- 四数据集兜底：不允许再次出现 Run3 式「普遍无收益」；WHU 必须不跌，否则视为数据集特化。
