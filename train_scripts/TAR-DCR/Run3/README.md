# TAR-DCR Run3 — Edge-Basis RepDW3（单变量：refine.dw）

## 本轮改动

- 只改 `DCRDecoder.refine.dw`（RepDW3），增加可折叠的 Sobel-X / Sobel-Y 结构边缘基分支（`use_edge`）。
- 训练图：`DW3×3 + DW1×3 + DW3×1 + αx + βx·(Kx*x) + βy·(Ky*x)`，`β` 零初始化。
- 部署图：完全折叠回单个 DW3×3，部署参数 / FLOPs 不变（+0）。
- 其余全部不动（TAR / encoder / loss / block1~3）。

## 实验

- 数据集：LEVIR-CD-256（当前最弱，Recall / IoU 缺口最大）
- 配置：`rep_mode=full`、`encoder_train=last2`、`use_residual=1`、`use_edge=1`、`seed=2333`、300 epoch、batch 16

## 锚点（Run2 full_last2 / LEVIR）

- F1=0.9144，Recall=0.9032，Precision=0.9260，IoU=0.8424

## 成功判据

- F1 ≥ 0.9175（至少 +0.31 pp）
- Recall ≥ 0.907（至少 +0.4 pp）
- 若 F1 涨幅 <0.15 pp，或 Precision 明显掉而 Recall 才换上去，则停止扩展该机制。
