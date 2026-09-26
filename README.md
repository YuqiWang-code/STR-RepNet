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

## 实验结果（Run4：IBAS，进行中）

- 单变量：训练期内侧边界辅助监督 IBAS（boundary head `Conv2d(160→1)` 零初始化、`B⁺=Y−Erode3×3(Y)` 内侧边界、`BCE+Dice`、λ=0.1），部署删除 +0 参数/FLOPs。
- 基线：Run2 full_last2（CDD 0.9842 / LEVIR 0.9144 / WHU 0.9514 / SYSU 0.8345）；4 数据集并行（GPU1）。
- 配置：`full + last2 + use_residual 1 + use_edge 0 + use_boundary_aux 1 + boundary_weight 0.1`，seed 2333，300 epoch。
- 判据（LEVIR 主判据）：F1≥0.9175 / IoU≥0.8475 / Precision≥0.9240；失败线 F1<0.9159；另 3 数据集防掉点。
- 设计文档：[`docs/temporary/STR-RepNet_Run4_IBAS_训练期内侧边界辅助监督方案.md`](docs/temporary/STR-RepNet_Run4_IBAS_训练期内侧边界辅助监督方案.md)。

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
      reparam.py               #   代数折叠原语（RepDW3/RepPW1x1/RepPairFuse1x1）
      tar.py                   #   TAR 二时相 bridge
      dcr_decoder.py           #   DCR 多尺度解码器
      STRRepNet.py             #   顶层网络（switch_to_deploy）
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
analyse/                       # 分析工具
  extract_metrics_to_excel.py  #   outputs → docs/experiment_metrics.xlsx
  models_to_txt.py             #   models 代码快照 + 指标 → docs/temporary/*.txt
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
