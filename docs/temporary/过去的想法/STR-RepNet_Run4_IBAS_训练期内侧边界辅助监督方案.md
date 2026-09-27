# STR-RepNet Run4：Training-only Inner-Boundary Auxiliary Supervision（IBAS）方案

> 记录日期：2026-09-25。
> 上游决策来源：Run3 Edge-Basis 失败后的下一步实验方向讨论（见
> `docs/temporary/过去的想法/STR-RepNet_下一步实验方向_代码审查与文献建议.md` 的 P1-4/Phase 4 路线）。
> 本文档只记录**本轮（Run4）**的修改与实验协议；Run4 结果出来后按惯例回填 README。

---

## 0. 一句话结论

**方法基线回到 Run2 `full_last2`；Run4 只做一个单变量——训练期内侧语义边界辅助监督（IBAS），部署期完全删除（+0 参数/FLOPs）。**

## 1. 背景：Run3 Edge-Basis 为什么终止

Run3 在 `DCRDecoder.refine.dw`（RepDW3）加了可折叠 Sobel-X/Y 固定边缘基分支（`β` 零初始化，部署折叠为单个 DW3×3）：

| 数据集 | Run2 full_last2 | Run3 Edge-Basis | ΔF1 |
|---|---:|---:|---:|
| CDD | 0.9842 | 0.9841 | -0.01 |
| LEVIR | 0.9144 | 0.9135 | -0.09 |
| WHU | 0.9514 | 0.9497 | -0.17 |
| SYSU | 0.8345 | 0.8311 | -0.34 |

- 四数据集**没有一个净提升**（Macro F1 92.11 → 91.96，约 -0.15pp）。
- LEVIR 细节：Recall 0.9032→0.9046（+0.14），但 Precision 0.9260→0.9225（-0.35）。
  → 增强局部边缘响应确实多找回了一点变化像素，但**同时激活了更多非变化边缘**。

机制解释（终止理由）：
1. RepDW3 本来就有一个自由 3×3 DW kernel，`βxKx+βyKy` 不扩大部署函数空间，只改变训练参数化；四数据集证据表明 **Sobel 参数化没有产生有利的优化偏置**。
2. Sobel 作用对象是 `d1∈R^{160×64×64}` 的潜在语义特征，其高频响应 ≠ 语义变化边界（屋顶纹理/阴影/配准微差都会响应）。
3. 继续试 Sobel+Laplacian / 8 方向 / 逐层加基，属于「模块继续堆、理由越来越弱」。

**决定**：不再迭代固定边缘卷积基；Run3 作为负结果保留在 git 历史与代码中（`use_edge` 开关保留，不再启用）。

## 2. Run4 机制：IBAS（训练期内侧边界辅助监督）

本质区别：**Run3 让算子更「会看边缘」；Run4 让 GT 告诉网络「哪些边缘才是真变化边界」。**

### 数据流（训练图）

```text
                    ┌─ Main Head 1×1(160→2) ─ Upsample ─ change logits ─ 主 loss（不变）
                    │
DCR final feature ──┤
       d1 (160×64×64)
                    └─ Boundary Head 1×1(160→1) ─ Upsample ─ boundary logits ─ 边界 loss
                                                         ↑
                                                   GT 内侧边界（在线生成）
```

### 部署图（不变）

```text
DCR final feature ─ Main Head 1×1 ─ Upsample ─ change map
```

Boundary Head 整个删除 → **Params_deploy / FLOPs_deploy 与 Run2 完全相同**。

### 关键设计点

1. **边界头极小且零初始化**：只有 `Conv2d(160,1,1)` = 161 个训练参数，`weight/bias` 零初始化（延续本项目 aux branch zero-init 惯例，避免第一步给共享 `d1` 随机大梯度）。不用 Conv3×3+BN+ReLU、不用 attention。
2. **内侧边界（不是两侧 morphological boundary）**：`B+ = Y − Erode3×3(Y)`。只取变化对象**内部一圈**，不强化目标外侧背景边缘（直接针对 Run3 Precision 掉点的问题）。
3. **GT 边界在线生成**：`label`（A/B/label 协议，gray≥128）→ 0/1 mask → 在线 erosion。不建 `label_edge/` 目录、不离线 Canny（CDD 的 JPG 压缩纹理不会产生假 Canny 边缘）。
4. **Ignore=255 处理**：先 `valid = label != 255`，再把 valid 做 3×3 erosion 作为边界监督有效区，crop padding 的 255/0 交界不会被当成目标边界。
5. **Loss 固定**：`L = CE + 2·Lovász + λ_b·(BCEWithLogits + Dice)`，第一轮锁死 **λ_b = 0.1**，不做 0.05/0.1/0.2 sweep。
6. **主 loss 一字不动**；`TAR / DCR / reparam / encoder / head` 结构一律不动。
7. **不使用现有 `edge_loss.py`**（Gaussian/Laplacian pyramid + Charbonnier 是图像复原式高频残差比较，与「语义变化边界」不是一回事；Run3 已证明泛化高频偏置无益）。

### 部署删除的验收（比 Run3 更强）

- `switch_to_deploy()` 中删除 boundary head；
- 单独测「只删 boundary head、不 fold TAR/DCR」前后主 logits → 要求 **max|Δ| = 0**（bit-exact），证明 aux 分支从未进入主推理路径；
- TAR/DCR fold 误差沿用原阈值（<1e-4 工程验收；P0 `<1e-6` 论文硬约束单独记账，本轮不混变量）。

## 3. 代码修改清单（本轮）

| 文件 | 修改 |
|---|---|
| `models/changedetection/utils_func/boundary_loss.py` | **新增**：inner-boundary target（`Y−Erode3×3(Y)`，valid 腐蚀）+ BCE/Dice |
| `models/changedetection/models/STRRepNet.py` | 新增训练期 `boundary_head=Conv2d(dim,1,1)`（零初始化）；`forward(..., return_aux=False)`；`remove_boundary_aux()`；`switch_to_deploy()` 删除 boundary head |
| `models/changedetection/script/train.py` | `--use_boundary_aux`、`--boundary_weight`（默认 0.1）；训练循环接入边界 loss；epoch 行追加 `Boundary=` 字段（仅开启时） |
| `models/changedetection/script/smoke_test.py` | 新增 boundary/encoder 检查（形状、aux-remove bit-exact、梯度流、stage1/2 无梯度） |
| `dcr_decoder.py` / `tar.py` / `reparam.py` / encoder | **不改** |

## 4. 实验协议（Run4）

- **基线**：Run2 `full_last2`（已有四数据集结果，不重跑）——CDD 0.9842 / LEVIR 0.9144 / WHU 0.9514 / SYSU 0.8345。
- **配置**（与 Run2 full_last2 完全一致 + 单变量）：
  `rep_mode=full / encoder_train=last2 / encoder_lr_ratio=0.1 / use_residual=1 / use_edge=0 / use_boundary_aux=1 / boundary_weight=0.1 / seed=2333 / 300 epoch / batch 16 / lr 1e-4 / lovasz 2.0 / temporal_swap_prob=0`。
- **数据集**：4 个并行（GPU1）——LEVIR（主验证）+ CDD/WHU/SYSU（防掉点对照），同 Run3 模式。
- **checkpoint/log 路径**：`TAR-DCR/Run4/<dataset>/`（全新目录，绝不从 Run3 续训）。
- 启动脚本：`train_scripts/TAR-DCR/Run4/run_gpu1_{levir,cdd,whu,sysu}.sh`。

## 5. 判据（提前锁死，LEVIR 主判据）

Run2 full_last2 / LEVIR 锚点：F1=0.9144，IoU=0.8424，Precision=0.9260，Recall=0.9032。

- **成功**：F1 ≥ 0.9175（+0.31pp）且 IoU ≥ 0.8475（+0.51pp）且 Precision ≥ 0.9240（允许最多 -0.20pp）。
- **失败**：F1 < 0.9159（<+0.15pp）→ 不再做 λ sweep，机制否决；
  或 F1 微涨但 Precision/Recall 任一项以 >0.3pp 的代价交换 → 机制不可靠。
- **扩展**：LEVIR 过判据后才评估扩跑结论；WHU（第二个建筑数据集）必须不跌，否则视为数据集特化。
- **四数据集兜底**：不允许再次出现 Run3 式「四数据集普遍无收益」。

## 6. 论文定位（不包装成新 loss 创新）

- 主创新仍是 **TAR + DCR**（结构重参数化：训练强、部署轻）。
- IBAS 定位为**辅助训练机制**：「结构重参数化负责训练强/部署轻，boundary supervision 负责弥补轻量 decoder 高分辨率边界表征不足，部署删除」。
- 边界建模并非空白（EdgeRefNet TGRS 2026、EOCLNet TGRS 2025 均保留 inference-time edge path）；我们的差异点是 **training-only semantic boundary regularization + 零开销可折叠部署**，不声称「首次使用边界」。

## 7. 本轮不做（单独记账）

- **P0**：严格 fold `<1e-6`（当前 1e-5~1e-4 工程阈值）——与 IBAS 分轮处理，不混变量。
- **P2**：Excel/TXT 的 `Params(M)` 语义（当前抓 TRAIN-GRAPH-PARAMS 而非部署参数）——下轮拆分三列。
- λ sweep、boundary head 加容量、多尺度 boundary deep supervision（方案 C）——第一轮不做。
