## GitHub 更新（手动操作）

代码或文档更新后，手动同步到 GitHub（提交信息统一用 `update code`）：

```bash
cd f:/Code_Repositories_2/CursorCode/STR-RepNet
git add models/ train_scripts/ docs/ README.md .gitignore
git commit -m "update code"
git push origin main
```

- 预训练权重 `pretrained_weight/*.pth` 不进 git，更新时上传到 GitHub Releases
  （仓库页 → Releases → Draft a new release → 上传附件 `.pth`）。

---

# STR-RepNet

Lightweight Spatial-Temporal Structural Re-parameterization Network for Remote Sensing Change Detection.

硕士课题：轻量化遥感二值变化检测中的结构重参数化研究。路线见
[`docs/temporary/Rep_Mamba_轻量遥感变化检测重参数化研究方案.md`](docs/temporary/Rep_Mamba_轻量遥感变化检测重参数化研究方案.md)，
服务器与数据规范见
[`docs/RSML-3_服务器环境与变化检测数据统一说明.md`](docs/RSML-3_服务器环境与变化检测数据统一说明.md)。

## Baseline

[HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection](docs/参考文献/baseline/HAM-CD_Hybrid_Attention_Mamba_for_Remote_Sensing_Change_Detection.pdf)。

选用的基线结构 = **VSSM-Tiny 编码器**（VMamba，预训练权重 `vssm_tiny_0230_ckpt_epoch_262.pth`）
+ **HAM-CD 混合注意力 Mamba 解码器**（`SpatialMambaBlock` SSM + MDTA 自注意力 + ICSF 通道注意力）。

- 模型定义：`models/changedetection/models/MambaBCD.py`（`STMambaBCD`）
- 配置：`models/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml`
  （`EMBED_DIM 96 / DEPTHS [2,2,4,2] / SSM_FORWARDTYPE v3noz`）
- 损失：`CE + 2.0 × Lovász-Softmax`
- 优化器：`AdamW(lr=1e-4, weight_decay=5e-4)`，300 epoch，batch 16
- **参数量 36.08 M，FLOPs 16.26 G**（双时相 2×3×256×256，eval 模式，fvcore 实测）

## 目录结构

```
models/                    # 全部代码（已从 HAM-CD 源码整理、适配）
  changedetection/         # 变化检测主代码
    configs/               # yacs 配置 + tiny yaml
    datasets/              # DataLoader（A/B/label + list 格式）
    models/                # 编码器 + 解码器 + 模型
    script/train.py        # 训练入口（日志 + 参数量/FLOPs + last/best 保存）
    script/smoke_test.py   # 冒烟测试
    utils_func/            # metrics / lovasz / edge loss
  classification/          # VMamba（VSSM）编码器实现
  kernels/selective_scan/  # 自定义 CUDA 内核（已适配 sm_120 + CUDA 13）
train_scripts/baseline/Run1/  # 4 个数据集的服务器启动脚本
outputs/baseline/Run1/        # 训练日志（训练结束后下载到这里，每数据集一个文件夹）
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
2. 服务器启动（`nohup`，每脚本带断点续训重试循环）：
   ```bash
   cd /home/yqwang/projects/STR-RepNet/train_scripts/baseline/Run1
   nohup bash train_WHU-CD-256.sh > /dev/null 2>&1 &
   ```
3. 训练日志格式：
   - 开头：全部配置 + 参数量(M) + FLOPs(G)。
   - 每个 epoch 一行：CE / Lovasz / Total 三个 loss + Recall / Precision / OA / F1 / IoU / Kappa 六项指标。
   - 结尾：`=== TEST RESULTS ===` 推理参数量 + FLOPs + 六项指标。
4. checkpoint：每数据集一个文件夹，只放 `last.pth`（断点续训）与
   `best_F1=xxx.pth`（测试用）。
5. 训练结束后只把 `train_log.txt` 下载回本地 `outputs/baseline/Run1/<dataset>/`。

## 运行监控

```bash
python .claude/_monitor.py   # 查看 4 个数据集进度 + GPU 占用
python .claude/_ssh.py '<cmd>'  # 通用 SSH 执行
```

## 注意事项

- RTX 5090（Blackwell sm_120）在 torch 2.10 下偶发 `CUBLAS_STATUS_NOT_INITIALIZED`，
  升级 torch 2.14+cu132（cuBLAS 13.4.0.1）后已消除；训练脚本仍保留 `num_workers=4` +
  断点续训重试循环兜底（崩溃自动从 `last.pth` 恢复）。
- `torch.load` 加载含优化器状态的 `last.pth` 需 `weights_only=False`（torch 2.6+ 默认
  `weights_only=True` 会拒收非张量对象），train.py 已处理。
- `.sh` 脚本需 LF 行尾（Windows 编辑后 `_deploy.py` 上传，服务器端会统一转 LF）。
