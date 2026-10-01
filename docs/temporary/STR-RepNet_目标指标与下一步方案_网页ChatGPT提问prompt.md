# 网页 ChatGPT 提问 Prompt：目标指标攻坚——下一步修改方案与实验设计

> 用法：把下面代码块中的全部内容复制粘贴到网页 ChatGPT（该项目已设置
> `docs/ChatGPT_Project_Settings.md` 项目指令，含文献调研准则）。
> 它会按项目指令做调研与设计，最后以 `.md` 文件形式返回完整方案，可直接下载存入 `docs/temporary/`。

```text
GitHub 仓库：https://github.com/YuqiWang-code/STR-RepNet
请先浏览仓库 README.md（重点读「研究定位与约定」与「实验结果 Run1–Run9」各章）、
train_scripts/TAR-DCR/Run9/README.md 以及 docs/temporary/STR-RepNet_Run9_BiFTR_设计与实验方案.md
（§22-A 结构搜索结束裁决与 §33/§34 收尾清单）。若无法打开仓库，以下摘要已含关键事实，可直接基于摘要工作。

# 一、项目背景摘要

STR-RepNet：轻量化遥感二值变化检测 + 结构重参数化（硕士课题）。
冻结 VMamba-Tiny 孪生编码器（last2 协议）+ TAR 二时相 bridge + DCR 多尺度解码器，
训练期多分支结构全部可解析折叠为单路径部署。
部署硬预算（Run2 机器实测锚点）：Params ≤ 28.828706M、FLOPs ≤ 12.6062G（比 HAM-CD baseline −20%/−22%）；
折叠全程 FP64、部署后 argmax disagreement = 0（实测验证）。

九轮结果（test F1；seed 2333、300 epoch、batch 16、CE+2×Lovász、test 集当验证集挑 best）：

| 轮次 | 结构 | LEVIR F1 | 裁决 |
|---|---|---|---|
| Run1 | TAR+DCR 全开 | 0.9033 | 低于 baseline，但部署省 20% 算力 |
| Run2 | BN-FR + encoder last2 解冻 | 0.9144 | **正式锚点**；WHU 0.9514 / SYSU 0.8345 / CDD 0.9842 |
| Run3 | Edge-Basis（Sobel 可折叠分支） | 0.9135 | FAIL（四集全微降） |
| Run4 | IBAS（训练期边界头，部署删除） | 0.9145 | FAIL（中性持平） |
| Run5 | BOTR（时序逆序分支） | 0.9144 | FAIL（恰锚点）；D0 full-encoder 0.9168（诊断上界） |
| Run6 | NSCR-Fuse（原生尺度 BN 分支） | 0.9140 | FAIL（Recall+0.41pp、Precision−0.52pp） |
| Run7 | PBRU（PixelShuffle 头 + 相位基，D*=158） | 0.9161 | WEAK；归因拓扑+0.12pp、rep+0.09pp |
| Run8 | MPCR-Fine（refine.pw 双 grouped 分区，Δ=0） | 0.9129 | FAIL；rep-not-supported |
| Run9 | BiFTR（encoder 冻训边界串行乘性 AWB，Δ=0） | 0.9147 | FAIL；按预注册 §22-A 裁决结束结构搜索 |

关键诊断证据（跨轮，来自仓库）：
1. 三种参数化家族（additive BN 分支 / phase basis / serial multiplicative basis）都学到非零结构，
   但净效应中性或「置信度锐化」（Recall↓/Precision↑）——这是三轮重复出现的失败签名。
2. Run6 小目标诊断：LEVIR small（≤502px）pixel recall 仅 80.8%、24% 小目标完全漏检
   （medium/large 为 92%/91%）；D0 full-encoder 无法修复（81.5%、漏检率 23.7%）→ 瓶颈在 decoder 侧；
   测试尺度 320 只抬 Precision、Recall 反降 → 不是输入分辨率问题。
3. Run7 阶段判别力预检：冻结 Run2 解码器下 d1 粗尺度线性信息已被训练头榨干
   （head AUROC@64 0.9881 > 全协方差 LDA 0.9766，headroom −0.0115）。
4. 硬条件（预算/argmax/fold 误差/epoch-0 逐位一致）九轮全程合格——失败是方法问题，不是实现问题。

# 二、硬性目标（必须全部达成，不可放宽）

四数据集 test F1 目标（在现有部署预算与折叠铁律约束下）：

| 数据集 | Run2 锚点 F1 | 目标 F1 | 缺口 |
|---|---|---|---|
| CDD | 0.9842 | ≥98 | 已达 ✓（须多 seed 保持） |
| WHU | 0.9514 | ≥95 | 已达 ✓（须多 seed 保持） |
| LEVIR | 0.9144 | **≥92.5** | **+1.06** |
| SYSU | 0.8345 | **≥85** | **+1.55** |

HAM-CD baseline 参考：WHU 0.9500 / LEVIR 0.9211 / CDD 0.9879 / SYSU 0.8299。
注意：LEVIR 目标 92.5 高于 baseline 92.11；SYSU 需在已反超 baseline 的基础上再提 1.55 点。
→ 主攻 LEVIR 与 SYSU。

除指标外还必须满足（铁律）：
1. 创新必须是结构重参数化本身；禁止把 loss 调参、数据增广、threshold tuning、多 seed 包装成创新。
2. 训练期独有结构必须严格可折叠或可删除；折叠全程 FP64、末次 cast FP32；部署 argmax disagreement = 0。
3. 部署 Params/FLOPs 不允许超过 Run2 锚点；新增结构必须机器预算验证。
4. 论文需同时具备：创新性、故事性、可解释性、轻量化。
5. 每轮实验预注册判据（PASS/WEAK/FAIL + 明确数值线），事后不改判据、不 sweep 救机制。
6. 训练协议固定：seed 2333、300 epoch、batch 16、CE+2×Lovász、test 当验证集选 best；GPU0 最多 4 job 并行。

# 三、任务要求

请给出「下一步修改方案 + 完整实验设计」，按以下结构：

1. **瓶颈归因复核**：基于上述证据，判断 LEVIR +1.06 / SYSU +1.55 缺口最可能的根因，
   区分编码器特征质量、时相交互建模、decoder 小目标保真、输出头上采样、训练协议等来源，
   给出每项的证据与优先级（P0/P1/P2）。
2. **文献调研**（严格遵守项目指令中的调研准则）：只检索 2024–2026 年高水平工作——
   CCF-A 会议（CVPR/ICCV/ECCV/AAAI/NeurIPS 等）与权威期刊（IEEE TGRS、ISPRS JPRS、JSTARS、IEEE TIP 等），
   明确标注层级，只引用可核验的论文主页/出版社/arXiv/官方 GitHub，不得虚构引用。
   给出 2–4 个候选方向，每个方向列代表性工作（题名/年份/venue/一作+机构/官方链接/与本项目的实质区别、
   为什么可能突破「置信度锐化」签名），然后**只推荐一个主方案 + 必要对照**，并说明取舍理由。
3. **主方案定义**：数学形式（含折叠公式）、训练图/部署图数据流、部署等价性论证、
   部署 Params/FLOPs 增量估计（必须 ≈0 或给出预算回收方案）、与 Run3–Run9 失败参数化家族的区别。
4. **逐文件修改清单**：对应仓库 models/（STRRepNet.py / reparam.py / tar.py / dcr_decoder.py /
   Mamba_backbone.py / script/train.py / script/smoke_test.py / script/test_reparam_equivalence.py）与 analyse/。
5. **实验设计**：唯一变量声明、C0 对照 + M1 主实验消融、LEVIR 先行判据（PASS/WEAK/FAIL 的精确
   F1/IoU/Precision 数值线；历史风格参考：锚点 0.9144，失败线 0.9159，PASS 线 0.9175）、
   判据通过后扩展到 WHU/SYSU/CDD 的顺序与 GPU 预算、失败预案（按失败模式分支解释，不得改判据救机制）。
6. **测试与预算验证**：smoke / 等价性测试（T0/T1/T2）/ 预算脚本的具体要求。
7. **立即执行顺序** 与 **仍需补充的证据**。

重要前提说明：Run9 曾按预注册裁决「结束结构搜索」，但当时目标是论文收尾；现在硬目标（LEVIR ≥92.5 /
SYSU ≥85）未达成，结构方向可以重开——请在方案开头明确论证「为什么这次的结构/参数化方向有实质理由
突破此前三轮置信度锐化签名」，若你认为应先做非结构方向的诊断性实验（如协议/分辨率/预训练诊断），
也请给出该诊断的设计与它如何服务最终目标。

# 四、交付格式

最后，请把完整方案整理为一个 markdown 文档，并直接以 .md 文件形式发给我（可下载的附件），
文件名形如：STR-RepNet_Run10_<方法名>_设计与实验方案.md。
```

---

## 附：本次更新说明（同步给 ChatGPT 或自查用）

- README「研究定位与约定」新增目标指标硬门槛一条；数据集表「目标 F1」列改为 CDD ≥98 / LEVIR ≥92.5 /
  SYSU ≥85 / WHU ≥95。
- 当前 Run2 锚点与目标的差距：CDD ✓ / WHU ✓ / LEVIR −1.06 / SYSU −1.55。
