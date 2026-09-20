# 轻量遥感二值变化检测中的结构重参数化研究方案

> 更新：2026-09-19。Baseline 已定为 HAM-CD；研究重心从「重新设计 decoder」收敛为
> 「对 HAM-CD 的中间模块与解码器做结构重参数化改造」，并保留一条备选路线：
> 「加载预训练权重前提下对编码器做多分支推理→单分支折叠」。

## 1. 研究目标

- 任务：全监督遥感双时相二值变化检测，四个公共数据集
  CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256（A/B/label + list，label 阈值 gray≥128）。
- 目标：推理参数量与 FLOPs 保持极低、推理实时友好；
  **训练期**用多分支/异构结构增强表达，**部署期**折叠为单路径（结构重参数化），
  且部署前后主预测误差 < 1e-6。
- 硬约束：创新集中在 **decoder / 时相融合 / 中间模块 / 残差桥接 / 边界监督**，不重构 backbone；
  encoder 保持成熟预训练网络（VMamba/VSSM-Tiny）。

## 2. Baseline：HAM-CD

论文：*HAM-CD: Hybrid Attention Mamba for Remote Sensing Change Detection*，IEEE TGRS 2026（已录用发表）。
代码 `models/changedetection/`，模型入口 `models/changedetection/models/MambaBCD.py`（`STMambaBCD`）。

### 2.1 基线结构

- **Encoder**：权重共享 Siamese，VMamba（VSSM-Tiny，预训练 `vssm_tiny_0230_ckpt_epoch_262.pth`），
  4 级下采样，输出 4 级多尺度特征 `{F1..F4}`（`EMBED_DIM 96 / DEPTHS [2,2,4,2] / SSM_FORWARDTYPE v3noz`）。
- **Decoder**：4 级 HAM 解码器，每级 = HAM block（+ 第 2–4 级含 ICSF 跨级融合）→ ×2 上采样。
- **参数量 36.08 M，FLOPs 16.26 G**（双时相 2×3×256×256，eval 模式，fvcore 实测）。
- 损失：`CE + 2.0 × Lovász-Softmax`；优化器 AdamW(lr=1e-4, wd=5e-4)，300 epoch，batch 16。

### 2.2 关键模块（= 重参数化改造对象）

| 模块 | 作用 | 内部结构（对应代码） |
|---|---|---|
| HAM block | 全局+局部空间-时相建模 | 并行 TAM + TAB，`fuse_layer`（1×1 conv）融合 |
| TAM（Textual-aware Mamba） | 局部像素依赖 + SSM 长程 | 3×3 DWConv（替代 1D 因果卷积）+ TASF 三路膨胀卷积（dilation 1/3/5）+ S6 选择性扫描 |
| TAB / MDTA（`Attention`） | 全局依赖（通道维转置注意力） | `qkv` 1×1 conv + `qkv_dwconv` 3×3 DWConv（无偏置）+ `project_out` 1×1 + 跨通道协方差注意力 |
| ICSF（`ICSFBlock`/`D_ICSFBlock`） | 跨级时相特征融合 | 3×3 Conv+BN+ReLU → 3×3 DWConv+BN；CFEM=SE 式通道注意力（GAP→1×1→ReLU→1×1→Sigmoid）；SFEM=3×3 DWConv；残差 |
| FeedForward（门控 FFN） | 通道建模 | `project_in` 1×1 → 门控 3×3 DWConv → `project_out` 1×1 |
| SS2D（编码器） | 多方向交叉扫描 | 4 方向扫描 + 选择性扫描（输入依赖） |

### 2.3 为什么不选其它 baseline

- 早期 SCAM / ST-Mamba / ChangeMamba 均已在 Mamba-CD 方向上密集被做，创新空间有限；
  HAM-CD 是 2026 TGRS 最新 SOTA，且其 decoder/中间模块大量使用**卷积型多分支结构**，
  天然适合结构重参数化改造，同时编码器带官方预训练权重，满足「轻量 + 强 + 可发表 + 可实现」。

## 3. 研究路线：对哪些模块做结构重参数化

### 3.1 主攻方向 A：中间模块重参数化

针对 HAM block 内部的卷积型多分支做「训练多分支 → 部署单路径」折叠：

- **TAM 局部分支**：3×3 DWConv + 三路膨胀卷积（dilation 1/3/5）是典型多分支卷积，
  可按 ACNet / DBB 膨胀卷积折叠思路合并为单个等效大核；SSM 选择性扫描本身是输入依赖递归，**不可折叠**，部署保留轻量 SSM。
- **TAB / MDTA**：`qkv` 1×1 + `qkv_dwconv` 3×3 是 RepVGG 式「1×1+3×3→3×3」结构，可直接折叠；
  `project_out` 1×1 与后续融合层可进一步合并。
- **FFN**：门控 3×3 DWConv 可与相邻 1×1 折叠。
- 目标：训练期多分支增强表达，部署期每个 HAM block 折叠为「1×1 Conv → DW RepKernel → 轻量 SSM → 1×1 Conv」。

### 3.2 主攻方向 B：解码器 / 跨级融合重参数化

- **ICSF**：`3×3 Conv+BN+ReLU` 与 `3×3 DWConv+BN` 可 BN-fold + 多分支卷积折叠为单个 3×3；
  CFEM 的 SE 通道注意力是**空间全局 + 数据依赖**的门控，不能静态折叠为卷积，
  部署时保留为极低成本的通道尺度（或与后续 1×1 合并）。
- **HAM 并行分支融合**：`fuse_layer`（concat → 1×1）与 TAM/TAB 输出卷积可折叠；
  训练期 TAM、TAB 并行（参考论文 Table V：并行优于串行），部署期融合为单路径。
- **时相拓扑重参数化（Temporal Rep）**：训练期 Concat / Sum / Diff 三路
  `Y = Conv([X1,X2]) + Conv(X1+X2) + Conv(X2−X1)`，部署期由卷积线性性折叠为单个 1×1：
  `W_deploy = [Wc + Ws − Wd, Wc + Ws + Wd]`。

### 3.3 备选方向 C：编码器多分支推理 → 单分支折叠（加载预训练权重）

- 目标：在**加载官方预训练权重**的前提下，把编码器推理时的多方向交叉扫描（SS2D 4 方向）/
  多分支结构折叠/简化为更少分支或单分支，且不重训 backbone、不破坏预训练表示。
- 难点（需调研确认）：SS2D 的选择性扫描参数是输入依赖的（data-dependent），
  与 RepVGG 式「静态卷积可加性折叠」不同，多方向扫描结果不能直接相加等价，
  需调研是否存在「扫描方向剪枝 / 子方向蒸馏 / 秩约减 / 卷积化近似」等 2024–2026 相关工作。

## 4. 重参数化技术储备（待调研补全）

- **经典**：RepVGG（3×3+1×1+Identity→3×3）、ACNet（非对称核）、
  DBB（Diverse Branch Block，多尺度/多核/多拓扑折叠）、RepLKNet（大核 DWConv 重参数化）、
  OREPA（在线重参数化，缩减训练成本）、RepGhost / RepViT（轻量 backbone）。
- **用于本课题**：膨胀卷积多分支折叠（TAM 三路 dilation）、BN-fold + 卷积合并（ICSF/ResBlock）、
  时相拓扑线性折叠（Temporal Rep）、大核重参数化（RepLK 风格，服务建筑边界/大感受野）。
- **已有遥感 CD 先例**：CD-RLKNet（大核重参数化用于变化检测）等，需系统调研 2024–2026 后继工作。

## 5. 关键难点与可证伪判据

1. **可折叠性边界**：卷积型多分支可折叠；SSM 递归、SE/通道注意力（全局数据依赖门控）不可静态折叠，
   只能保留为部署期极低成本组件。任何「把不可折叠结构强行折叠」的声称都要证伪。
2. **部署一致性**：重参数化折叠前后主预测误差必须 < 1e-6（已列为硬约束）。
3. **预训练权重兼容**：encoder 折叠不得破坏 VSSM-Tiny 预训练表示（备选方向 C 的核心判据）。
4. **不堆模块**：每项改动要有机制动机 + 与既有工作实质区别 + 最小消融 + 失败判据，
   不把调参包装成创新；单数据集单种子微小差异不称普适提升。

## 6. 预期贡献（草案，随调研收敛）

1. 提出双时相拓扑重参数化（Temporal Rep：Concat/Sum/Diff 三路 → 单 1×1）。
2. 提出面向 Mamba 混合注意力解码器的多分支卷积重参数化（Rep-HAM：TAM/TAB/ICSF/FFN 折叠）。
3. 提出「训练复杂、部署单路径」的轻量变化检测网络，保持极低部署参数量/FLOPs。
4. （备选）预训练权重下编码器多方向扫描的推理期分支合并。

## 7. 实验设计

- **Baseline**：HAM-CD 原模型（4 数据集，Run1 协议）。
- **Ablation（唯一变量）**：① baseline → ② +Temporal Rep → ③ +Spatial/多分支折叠（中间模块）
  → ④ +ICSF 融合折叠 → ⑤ +Boundary/边界监督 → ⑥ Full。
- 每项报告：Rec/Prec/OA/F1/IoU/Kappa + 训练参数量 vs 部署参数量 + FLOPs + 折叠前后误差。
- 成功阈值：在 ≥2 个数据集、多 seed 上稳定超过 baseline，且部署 FLOPs/参数不增反降。

## 8. 下一步（当前）

对 §3 三条路线的相关文献做**详尽调研**（范围见 `ChatGPT_Project_Settings.md` §30：
2024–2026，CCF-A 会议 + IEEE TGRS/ISPRS JPRS/JSTARS/TIP 等权威期刊，可核验引用），
据此收敛唯一主方案与对照，再进入设计→实现→实验循环。
