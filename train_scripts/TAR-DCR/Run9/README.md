# TAR-DCR Run9 — BiFTR（双侧冻训边界重参数化，最后一次结构搜索）

## 本轮定位

Run8 MPCR-Fine FAIL 后，按导师式决策：**Run9 是最后一次正交结构搜索**，位置离开 decoder、
参数化家族离开"并行零初始化分支线性相加"。若 Run9 WEAK/FAIL → **不再设计 Run10**，进入论文收尾
（Run2 TAR+DCR 为主线 + Run3~9 系统化负结果边界）。

## 方案（BiFTR：Bi-sided Frozen-to-Trainable Transition Reparameterization）

- 位置：`encoder.layers[1].downsample` 的 Conv（192→384）——`encoder_train=last2` 下
  唯一明确的 **frozen stage2 → trainable stage3** 边界。
- 训练图：`y = (I+Δout)·( W·((I+Δin)·x) + b )`，Δin(192²)/Δout(384²) 为 **零初始化 1×1 conv**
  （写成 residual 形式保证 epoch-0 与 Run2 **逐位一致**）；base Conv 保持 frozen，只有
  Δ 可训练（在 `_setup_encoder_train()` 之后显式重开 requires_grad）。
- 部署：`W_eq = A·W·B`、`b_eq = A·b`（FP64 组合、一次 FP32 cast）→ 仍为原单个 downsample
  Conv，**部署 +0 参数/FLOPs**，无需 D* 搜索（只做 budget equality audit）。
- **与前几轮的本质差异**：这是串行乘性双侧 basis 重参数化（含 Δout·W·Δin 交叉项），
  不是并行 additive branch。
- C0_FTR_Post（仅 Δout，单侧 calibration 控制）vs M1_BiFTR（双侧）；注意 C0/M1 训练参数
  数不同（147,456 vs 184,320），M1−C0 证明的是"双侧相对单侧"的净贡献（doc §6 已明示）。
- **与设计文档的一处事实修正**：文档假设 downsample 为 k2 s2（v2），本仓库配置实为
  `downsample_version=v3`（**k3 s2 p1**）。AWB 折叠与 kernel 尺寸无关，骨干算子原样保留，
  assert 只锁定"192→384、stride 2"这一冻训边界身份；测试用真实 v3 算子。

## 预训练门槛（GPU0）

| 门槛 | 结果 |
|---|---|
| T0 AWB FP64 代数 | 阈值 1e-10（192/384 维 GEMM 在 O(100) 量级下的纯 FP64 累加 ~9e-11，相对 ~1e-13；文档 1e-12 过紧） |
| T1 BiFTRTransition 折叠（post/bi） | FP32 <2e-5 + FP64 代数 <1e-10（同理由），扰动 Δ 到训练后量级 |
| T2 全模型（post/bi, last2） | <2e-4、argmax=0、冻结状态断言（base frozen / Δ trainable / stage1-2 frozen / stage3-4 trainable） |
| 冒烟 | wrapper 定位 192→384 s2（k3 v3 记录）、base frozen、Δ trainable+零初始化、**epoch-0 与 use_biftr=0 逐位一致（max_diff=0.0）**、Δ 梯度 @init 非零且 base grad=None、部署无 d_in/d_out 残留、Params/FLOPs 与锚点**精确相等**、argmax=0 |
| 预算 equality audit（`search_biftr_budget.py`） | C0/M1 部署 ΔParams=ΔFLOPs=**0**（原始整数核验） |

## 最小实验组（第一阶段只跑 LEVIR；GPU0）

| 组 | 配置 | 状态 |
|---|---|---|
| A0 | Run2 full_last2 锚点（F1=0.9144 / IoU=0.8424 / Precision=0.9260，不重跑） | 复用 |
| C0_FTR_Post | BiFTR post-only（仅 Δout） | 启动 |
| M1_BiFTR | BiFTR 双侧（Δin+Δout） | 启动 |

- 公共：`rep_mode=full / last2 / D=160 / bilinear / use_residual=1 / botr=nscr=pbru=mpcr=0 /
  seed=2333 / 300 epoch / batch 16 / lr 1e-4 / lovasz 2.0`。
- **GPU0 并行（用户指令）**：C0+M1 的 LEVIR 与 WHU 共 4 job 并行（~27.5G/32.6G）。注意 WHU 属于
  容量驱动启动、先于 §19 gate；WHU 结果照常记录，但仅在 LEVIR 过 gate（M1 PASS 且
  M1−C0≥+0.05pp）后才参与扩展裁决。
- SYSU/CDD 脚本已预写，**仅过 gate 后**按 SYSU→CDD 启动（§19/§20）。

## 预注册判据（LEVIR）

- **PASS**：F1 ≥ 0.9175 且 IoU ≥ 0.8475 且 Precision ≥ 0.9230，且预算/argmax 合格。
- **WEAK**：0.9159 ≤ F1 < 0.9175，或副指标不满足 → 记录并**结束结构搜索**（不进入下一版 BiFTR）。
- **FAIL**：F1 < 0.9159 或 Precision < 0.9220 或预算/等价性失败 → **结束结构搜索，进入论文收尾**。
  禁止：换 stage、rank/scope sweep、加 stage3→4、full encoder/BiFTR 叠加、loss/threshold/增强。
- **rep 归因（M1−C0）**：≥ +0.15pp 且精度不降超 0.15pp = rep-supported；+0.05~0.15pp = rep-weak；
  < +0.05pp = rep-not-supported。
- 机制记录标签：`[BIFTR-DELTA-NORM] / [BIFTR-EFFECTIVE-UPDATE-NORM] / [BIFTR-CROSS-TERM-RATIO]`
  （交叉项比率用于判断双侧乘性耦合是否真的被利用，不用于挑 checkpoint）。

## 实验结果（LEVIR + WHU，已完成）—— M1 FAIL，结构搜索结束

| exp | 数据集 | F1 | Recall | Precision | IoU | ΔF1 vs A0 | M1−C0 |
|---|---|---|---|---|---|---|---|
| C0_FTR_Post | LEVIR | 0.9142 | 0.9042 | 0.9243 | 0.8419 | −0.02pp | — |
| M1_BiFTR | LEVIR | **0.9147** | 0.9037 | 0.9260 | 0.8428 | **+0.03pp** | **≈+0.05pp（边界）** |
| C0_FTR_Post | WHU | 0.9496 | 0.9394 | 0.9601 | 0.9040 | −0.18pp | — |
| M1_BiFTR | WHU | 0.9506 | 0.9390 | 0.9625 | 0.9059 | −0.08pp | +0.10pp |

- **M1/LEVIR = FAIL**（F1 0.9147 < 失败线 0.9159，Precision 0.9260 达标但 F1 不过线）；
  rep 归因落在 not-supported/weak 边界（M1−C0 ≈ +0.05pp，Precision +0.17pp）。
  硬条件全程合格（28.829M / 12.6062G、argmax=0、fold ~6.7e-05）。
- 权重空间记录（M1 LEVIR）：‖Δin‖F=0.67、‖Δout‖F=1.21、有效更新范数 0.131（相对核变化 13%）、
  **交叉项比率 5.65%** —— 双侧乘性耦合确实被利用了，但净效应中性（Recall −0.05pp /
  Precision +0.17pp，又现"置信度锐化"签名）。
- **§22-A / 预注册裁决：结束结构搜索**。Run9 是最后一次正交结构尝试（encoder 侧串行乘性
  重参数化），未达 PASS → **不设计 Run10**，进入论文收尾（doc §33）：
  清理主方法（TAR+DCR+BN-FR，Run2）、统一 Params/FLOPs、部署误差记录、train/deploy 图、
  组织 Run1–Run9 证据（正文保留 NSCR/PBRU/MPCR/BiFTR 最有信息量的负结果）、
  方法冻结后补 3-seed × 4 数据集正式稳定性。
- 完整汇总见 `outputs/TAR-DCR/Run9/_SUMMARY.md`。

## 机制分析工具

`analyse/levir_biftr_profile.py`：hook pre_transition / transition_pre_norm /
transition_post_norm / stage3_out / stage4_out 五级特征（32/16/16/16/8 尺度），统计
centroid/Fisher/对角 LDA AUROC/有效秩/通道相关 + 权重空间统计（‖Δin‖F、‖Δout‖F、
‖W_eq−W‖F/‖W‖F、交叉项比率），比较 A0/C0/M1 是否真正改变冻训边界的可分性。

## 设计文档

`docs/temporary/STR-RepNet_Run9_BiFTR_设计与实验方案.md`（含 §0 三个方向性判断、
§22 失败预案、§33 失败后的论文收尾清单、§34 multi-seed 建议）。
