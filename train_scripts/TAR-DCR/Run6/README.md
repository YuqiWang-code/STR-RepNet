# TAR-DCR Run6 — NSCR-Fuse（原生尺度可交换重参数化融合）

## 本轮改动

- 在 DCR 的 `fuse1 + fuse2`（high-resolution cross-scale fusion）各加两个**零初始化原生尺度 BN 分支**：
  - lateral 路：`BN_L(L)`（L 在其原生高分辨率上）；
  - semantic 路：`U(BN_H(H))`（H 先在其原生低分辨率做 BN，再 bilinear upsample）。
- 部署时利用「逐通道仿射 × bilinear 插值可交换」`U(AH+c)=A·U(H)+c`，把两个分支精确吸收回原有 `RepPairFuse1x1` 的 L/H 两半输入权重 → **部署 Params/FLOPs 与 Run2 完全一致（+0）**。
- 训练分支零初始化（gamma=beta=0），初始主预测与 Run2 完全相同，逐步展开。
- `fuse3` 不改；TAR / encoder / loss / 增广 不动。
- 详细设计：`docs/temporary/STR-RepNet_Run6_下一步改进方向与实验设计_NSCR-Fuse.md`。

## 最小实验组（第一阶段只跑 LEVIR；GPU0 整卡可用）

| 组 | 配置 | 状态 |
|---|---|---|
| A0 | Run2 full_last2 锚点（F1=0.9144 / IoU=0.8424 / Precision=0.9260，不重跑） | 复用 |
| **M1_NSCR_high2** | `use_nscr=1, nscr_scope=high2, last2` | **启动** |
| C1_NSCR_Lonly | `nscr_scope=lonly`（仅 lateral 分支） | M1 PASS 后 |
| C2_NSCR_Honly | `nscr_scope=honly`（仅 semantic-native 分支） | M1 PASS 后 |
| M2_NSCR_fullenc | `nscr_scope=high2, encoder_train=full`（对照 D0=0.9168） | M1 PASS 后 |

- 公共：`rep_mode=full / use_residual=1 / use_botr=0 / temporal_swap_prob=0.0 / seed=2333 / 300 epoch / batch 16 / lr 1e-4 / lovasz 2.0`。
- M1 的 WHU/SYSU/CDD 脚本已预写，**仅当 M1/LEVIR 通过判据后**按 WHU→SYSU→CDD 顺序启动。
- 跨组 checkpoint 不互用；同组 `last.pth` 可断点续训。

## 预注册判据（LEVIR，锚点 F1=0.9144 / IoU=0.8424 / Precision=0.9260）

- **PASS**：F1 ≥ 0.9175 且 IoU ≥ 0.8475 且 Precision ≥ 0.9230。
- **WEAK**：0.9159 ≤ F1 < 0.9175，或 F1 达标但 IoU/Precision 任一不满足 → 只记录，不扩四数据集。
- **FAIL**：F1 < 0.9159 或 Precision < 0.9220（掉 >0.40pp）→ 停止 NSCR，不救机制；若尺度诊断阳性再转 PBRU。
- 四数据集扩展标准（M1 过线后）：Macro F1 ≥ 92.25%、LEVIR ≥ 91.75%、至少 3/4 数据集 ΔF1 ≥ 0、任一退化 ≤ 0.15pp；硬部署条件：Deploy ≤ 28.829M / 12.6062G、argmax 分歧 = 0。

## 诊断（已实现，零训练）

`analyse/levir_error_profile.py`：对 Run2 / Run5-D0 checkpoint 输出尺寸分层 FN（训练集分位数分 bin）+ 测试尺度敏感性（192/256/320/384），输出到 `/home/yqwang/outputs/STR-RepNet/diagnostics/Run6_LEVIR/`。
