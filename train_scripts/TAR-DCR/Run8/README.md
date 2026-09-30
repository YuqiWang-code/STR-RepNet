# TAR-DCR Run8 — MPCR-Fine（细粒度多分区通道重参数化）

## 本轮动机

Run7 归因结论：PixelShuffle 拓扑数据集敏感（LEVIR +0.12pp / WHU −0.35pp），phase-basis rep
两数据集都表现为精度提升但系统级 WEAK；阶段判别力预检显示冻结 Run2 解码器的 d1 粗尺度线性
信息已被训练头榨干（head AUROC@64 0.9881 > 全协方差 LDA 0.9766）。→ 下一步不是继续换输出
读出，而是**改变 d1 的形成过程**：refine / fine-stage feature representation。

## 方案（MPCR-Fine）

- 位置：**仅** `DCRDecoder.refine.pw`（block1/2/3、refine.dw、fuse、TAR、head 全不动）。
- 训练图：`Y = RepPW1x1(X) + P0⁻¹BN0(G0(P0X)) + P1⁻¹BN1(G1(P1X))`
  - G0/G1：grouped 1×1（D=160, groups=4, 组宽 40，Kaiming 初始化，bias=False）+ 独立 BN
    （**γ=β=0 零初始化** → epoch 0 输出与 Run2 逐位一致）；
  - P0 = identity；P1 = identity（C0/same2）或交织置换 `arange(160).view(4,40).t().reshape(-1)`
    （M1/multi2，跨分区交织）。
- 部署：`W_eq = W_core + P0⁻¹W̃0P0 + P1⁻¹W̃1P1`（FP64 组合、一次 FP32 cast）→ 仍为单个
  dense PW 1×1，与 Run2 `refine.pw` 部署算子完全一致，**部署 +0 Params/+0 FLOPs**。
- 训练期增量：2×(6400+320) = **+13,440 参数**（仅训练期）。
- 与现有 RepPW 的实质区别：新增的是 structured full-rank local subspace basis +
  complementary partition connectivity（组内 full-rank、跨分区互补连接），不是再叠
  low-rank/diag/residual。
- 设计文档：`docs/temporary/STR-RepNet_Run8_MPCR-Fine_设计与实验方案.md`。

## 预训练门槛（已全部通过，GPU0）

| 门槛 | 结果 |
|---|---|
| T0 置换/嵌入精确性 | P⁻¹(Px)==x 精确；grouped→dense 嵌入 0.0；P⁻¹G(Px)==(P⁻¹GP)x 2.1e-14 |
| T1 MPCRPW1x1 折叠（same2/multi2） | FP32 4.8e-06/6.0e-06（<2e-5）；FP64 代数 7.1e-15/8.0e-15 |
| T2 全模型折叠 | same2 5.9e-05 / multi2 8.8e-05（<2e-4），argmax 全 0 |
| 冒烟：epoch-0 与 Run2 逐位一致 | max_diff = **0.0** |
| 冒烟：梯度 | BN γ 梯度 @init 非零；group-conv 权重在 γ 离开零后收到非零梯度（零初始化分支固有动力学，RepVGG 式） |
| 冒烟：部署 | 无 g0/g1/bn0/bn1/core 残留；Params/FLOPs 与锚点**精确相等**（28.829M / 25.2126G@batch2） |
| **预算机器验证** | **D\*=160，ΔParams=0，ΔFLOPs=0**（28.828706M / 12.6062G 与锚点逐位一致） |

## 最小实验组（第一阶段只跑 LEVIR）

| 组 | 配置 | 状态 |
|---|---|---|
| A0 | Run2 full_last2 锚点（F1=0.9144 / IoU=0.8424 / Precision=0.9260，不重跑） | 复用 |
| C0_MPCR_Same2 | refine.pw = core + 2×identity 分区 grouped 分支（容量控制） | 启动 |
| M1_MPCR_Multi2 | refine.pw = core + identity + 交织互补分区 | 启动 |

- 公共：`rep_mode=full / D=160 / bilinear / last2 / use_residual=1 / botr=nscr=pbru=0 /
  seed=2333 / 300 epoch / batch 16 / lr 1e-4 / lovasz 2.0`；C0/M1 训练期参数、分支数、groups、
  部署图**完全相同**，唯一变量 = branch1 的 channel partition。
- **GPU0 并行（用户指令）**：C0+M1 的 LEVIR 与 WHU 共 4 job 并行（~28G/32.6G）。注意 WHU 属于
  容量驱动启动、先于 §18 gate；WHU 结果照常记录，但仅在 LEVIR 过 gate（18-A/B）后才参与扩展裁决。
- SYSU/CDD 脚本已预写，**仅过 gate 后**按 SYSU→CDD 启动（§18/§19）。
- 说明：设计文档 §26 建议 C0 跑 GPU0、M1 跑 GPU1；按本机实际分配（GPU1 属其他课题），
  **全部跑 GPU0**。

## 预注册判据（LEVIR，锚点 F1=0.9144 / IoU=0.8424 / Precision=0.9260）

- **PASS**：F1 ≥ 0.9175 且 IoU ≥ 0.8475 且 Precision ≥ 0.9230，且 Deploy ≤ 28.828706M /
  12.6062G、argmax=0。
- **WEAK**：0.9159 ≤ F1 < 0.9175，或 F1 达标但 IoU/Precision 任一不满足。
- **FAIL**：F1 < 0.9159 或 Precision < 0.9220 或预算/等价性失败 → 停止，**不救机制**
  （禁 groups/permutation sweep、禁加第三个分区、禁 loss/threshold/PixelShuffle/full-encoder 叠加）。
- **rep 归因（M1−C0）**：≥ +0.15pp 且 Precision 降 ≤0.15pp = rep-supported；
  +0.05~0.15pp = rep-weak（单 seed 迹象）；< +0.05pp = rep-not-supported。
- **继续/停止（§18）**：A=PASS+rep-supported→WHU；B=PASS+rep-weak→WHU 作跨数据集 gate
  （WHU ΔF1≥0 且 M1−C0≥+0.05pp 才进 SYSU/CDD）；C=PASS+rep-not-supported→不扩；
  D=WEAK→归因后停止，不跑 WHU；E=FAIL→立即停。

## 机制分析工具

`analyse/levir_fine_stage_profile.py`：hook block1 / refine.dw(±SiLU) / refine.pw / refine
最终输出五级特征，统计 centroid 距离、Fisher、对角 LDA AUROC、协方差有效秩（熵）、
平均通道间相关，验证 MPCR 是否真正改变最终通道表征（仅机制解释，不挑 checkpoint，不设新判据）。
