# STR-RepNet 交接文档（新对话启动必读）

> 生成时间：Run9 结束、结构搜索正式收尾之际。main @ `6d5cfee`。
> 用途：换新对话窗口后，把本文件路径丢给 AI（或直接复制下面的「新对话开场 prompt」），
> AI 读完本文件 + `README.md` 即可无缝继续。

---

# 0. 给新对话的开场 prompt（直接复制使用）

```text
这是 STR-RepNet 硕士课题仓库的续接任务。先读这两个文件再行动：
1. F:\Code_Repositories_2\CursorCode\STR-RepNet\README.md （九轮结果总览）
2. F:\Code_Repositories_2\CursorCode\STR-RepNet\docs\temporary\STR-RepNet_交接文档_新对话启动.md （交接文档，含本地/服务器工作流与当前状态）
工作模式：用户（211 硕士）驱动迭代，每轮给出设计文档后，AI 负责改代码、上服务器测试训练、
更新 README 并推送 GitHub。服务器与本地操作规范见交接文档第 4/5 节；当前状态见第 6 节。
```

---

# 1. 项目一句话

《轻量化遥感二值变化检测中的结构重参数化研究》：STR-RepNet = 冻结 VMamba-Tiny 孪生编码器
（last2 协议）+ TAR 二时相 bridge + DCR 多尺度解码器，全部算子可解析折叠为单路径部署。
**部署硬预算：Params ≤ 28.828706M、FLOPs ≤ 12.6062G**（Run2 机器实测锚点，比 HAM-CD
baseline −20%/−22%）。

## 1.1 铁律（每轮都必须遵守）

1. **创新必须是结构重参数化本身**；禁止 loss/数据增广/threshold tuning/多 seed 包装成创新。
2. 训练期独有结构必须**严格可折叠或可删除**；折叠全程 FP64、最后一次 FP32 cast；
   部署后 **argmax disagreement = 0**（实测验证）。
3. 部署 Params/FLOPs 不允许超过 Run2 锚点；新增结构必须做**机器预算验证**（budget 脚本）。
4. 每轮实验**预注册判据**（PASS/WEAK/FAIL + rep 归因阈值），事后永远不改判据、不 sweep 救机制。
5. 正式结果只认 `train_log.txt` 里**最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块**。
6. 训练协议：seed=2333、300 epoch、batch 16、CE + 2×Lovász、test 集当验证集选 best。
7. **GPU 纪律：只用 GPU0**（GPU1 属其他课题；GPU0 可 4 job 并行，每 job ~6.8G）。
8. 服务器是共享的：不要动别人文件；checkpoint/log 只放本课题目录。

---

# 2. 九轮历史（一页速览）

| 轮次 | 结构 | LEVIR F1（A0=0.9144） | 裁决 |
|---|---|---|---|
| Run1 | TAR+DCR 全开 | 0.9033 | 部署省 20% 算力，低于 baseline |
| Run2 | BN-FR + encoder 解冻诊断 | 0.9144 | **正式锚点**（WHU 0.9514 / SYSU 0.8345 / CDD 0.9842） |
| Run3 | Edge-Basis（Sobel 折叠分支） | 0.9135 | FAIL（四集全微降） |
| Run4 | IBAS（训练期边界头，部署删除） | 0.9145 | FAIL（中性持平） |
| Run5 | BOTR（时序逆序分支） | 0.9144 | FAIL（恰锚点）；D0 full-encoder 0.9168（diagnostic 上界） |
| Run6 | NSCR-Fuse（原生尺度 BN 分支） | 0.9140 | FAIL（Recall+0.41pp 但 Precision−0.52pp） |
| Run7 | PBRU（PixelShuffle 头+相位基，D*=158） | 0.9161 | WEAK；归因拓扑+0.12pp、rep+0.09pp |
| Run8 | MPCR-Fine（refine.pw 双 grouped 分区，Δ=0） | 0.9129 | FAIL；rep-not-supported |
| Run9 | BiFTR（encoder 冻训边界串行乘性 AWB，Δ=0） | 0.9147 | FAIL；M1−C0≈+0.05pp 边界 |

**跨轮共同签名**：三种不同参数化家族（additive BN 分支 / phase basis / serial
multiplicative basis）都学到非零结构，但净效应中性或 Recall↓/Precision↑ 的"置信度锐化"，
找不回小目标漏检；硬条件（预算/argmax/fold 误差/epoch-0 逐位一致）全程合格——失败是方法问题
不是实现问题。Run7 预检：冻结 Run2 解码器下 d1 粗尺度线性信息已被训练头榨干
（head AUROC@64 0.9881 > 全协方差 LDA 0.9766）。

**当前阶段：Run9 按预注册 §22-A 裁决 = 结束结构搜索，进入论文收尾**
（不设计 Run10）。收尾清单在 Run9 设计文档 §33/§34（multi-seed 3 seed × 4 数据集、
论文贡献组织、公平比较、D0 full-encoder 定位）。**用户暂未确定下一步**，
计划先带 GitHub 链接问网页 ChatGPT 要《论文收尾与最终实验方案》文档（prompt 已由上一个
对话给出，见第 7 节附录），拿到后再回来执行。

---

# 3. 本地仓库结构与"看哪个文件了解什么"

本地根目录：`F:\Code_Repositories_2\CursorCode\STR-RepNet`

## 3.1 总览类

| 文件 | 内容 / 何时看 |
|---|---|
| `README.md` | 项目总览：方法清单（各轮机制一句话 + 状态）、九轮结果章节、目录结构、服务器环境、训练/监控命令。**新对话必读**。 |
| `docs/temporary/` | 每轮设计文档（`STR-RepNet_RunN_*_设计与实验方案.md`，来自网页 ChatGPT）+ 代码快照（`models_and_metrics_TAR-DCR_RunN.txt`，由 analyse 工具生成）+ 历史想法（`过去的想法/`） |
| `train_scripts/TAR-DCR/RunN/README.md` | 每轮：方案、预训练门槛实测表、判据、结果、裁决。**查某轮细节看这里** |
| `outputs/TAR-DCR/RunN/` | 训练日志本地副本 + `_SUMMARY.md`（watcher 自动生成的裁决）+ `_watch*.log`。**outputs/ 被 gitignore，不进版本库** |
| `docs/experiment_metrics.xlsx` | 全部 55 行指标汇总（由 extract_metrics_to_excel.py 生成），含各轮 Tag/Run/Experiment/Dataset + 指标 + Params/FLOPs/fold 误差/模式开关 |

## 3.2 模型代码（改方法时动这里）

| 文件 | 内容 |
|---|---|
| `models/changedetection/models/STRRepNet.py` | 顶层网络：`__init__(..., rep_mode, dim, use_residual, encoder_train, use_botr, use_nscr, nscr_scope, head_mode, use_pbru, pbru_upscale, use_mpcr, mpcr_mode, mpcr_groups, use_biftr, biftr_mode, **encoder_kwargs)`；`_setup_encoder_train()`；`forward(pre,post)`；`train()` override；`switch_to_deploy()`。**新机制接线点** |
| `models/changedetection/models/reparam.py` | 折叠原语：`fold_conv_bn`（FP64）、`fold_identity_bn`、`RepDW3`、`RepPW1x1`、`RepPairFuse1x1`、`NSCRPairFuse1x1`、`PBRUHead`+`build_phase_basis`（Run7）、`MPCRPW1x1`+perm helpers（Run8）、`switch_module_to_deploy` |
| `models/changedetection/models/Mamba_backbone.py` | `Backbone_VSSM`（layers[i] = blocks + downsample；**downsample 是 Sequential(Identity, Conv2d(k3 s2 p1), Identity, LayerNorm2d)，config 用 `downsample_version="v3"`**）+ `BiFTRTransition` + `install_biftr_transition`（Run9，包装 layers[1].downsample[1]，192→384） |
| `models/changedetection/models/tar.py` | TAR 二时相 bridge（TemporalRep1x1/TARStage/MultiScaleTAR）——**已冻结，不再改** |
| `models/changedetection/models/dcr_decoder.py` | DCR 解码器（fuse3/block3/fuse2/block2/fuse1/block1/refine；`use_mpcr` 只作用于 refine.pw） |
| `models/classification/models/vmamba.py` | 第三方 VMamba 实现——**尽量不直接改**；只读它的 downsample/SS2D 结构 |
| `models/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml` | 模型配置（`DOWNSAMPLE: "v3"` 注意！） |

## 3.3 脚本代码

| 文件 | 内容 |
|---|---|
| `models/changedetection/script/train.py` | 训练入口：全部 CLI 开关 + 互斥 guard（Run7/8/9 各自断言）+ 每 epoch 日志 + `[PBRU-GAMMA-NORM]`/`[MPCR-GAMMA-NORM]`/`[BIFTR-*]` 机制记录 + 结尾 `=== TEST RESULTS ===` 部署图测试（Params/FLOPs/fold 误差/argmax + `[BEST-F1]`） |
| `models/changedetection/script/smoke_test.py` | 冒烟：构建/前向/epoch-0 逐位一致/梯度/部署分支删除/Params·FLOPs==锚点/argmax/dataloader。每轮新机制都要在这里加检查 |
| `models/changedetection/script/test_reparam_equivalence.py` | 等价性分层测试：T0（FP64 代数精确）、T1（block 折叠 FP32<2e-5 + FP64 代数）、T2（全模型 <2e-4 + argmax=0）。含 PBRU/MPCR/BiFTR 全套 |

## 3.4 analyse 工具（全部只读，不改训练）

| 文件 | 用途 |
|---|---|
| `extract_metrics_to_excel.py` | outputs/**/train_log.txt → `docs/experiment_metrics.xlsx`（watcher 自动调用；含 Run7/8/9 的 HeadMode/PBRU/DecoderDim/MPCR/BiFTR 列） |
| `models_to_txt.py` | models 代码快照 + 指标 → `docs/temporary/models_and_metrics_TAR-DCR_RunN.txt`（`--tag TAR-DCR --run RunN`） |
| `levir_error_profile.py` | Run6 诊断：小目标 FN 分层 + 尺度敏感性 |
| `search_pbru_budget.py` | Run7 机器预算搜索（得到 D*=158） |
| `levir_stage_discriminability.py` | Run7 阶段判别力预检（d1 LDA vs head AUROC） |
| `search_run8_budget.py` | Run8 预算验证（必须 D*=160、Δ=0） |
| `levir_fine_stage_profile.py` | Run8 精细阶段表征画像 |
| `search_biftr_budget.py` | Run9 预算 equality audit（必须 Δ=0） |
| `levir_biftr_profile.py` | Run9 冻训边界表征画像（M1 FAIL 后按协议未运行） |

## 3.5 `.claude/` 运维脚本（本地 ↔ 服务器）

| 文件 | 用途 |
|---|---|
| `.claude/_ssh.py` | 通用 SSH：`python .claude/_ssh.py "<cmd>" [timeout]`。**必须用 `E:\Anaconda\Anaconda3-2023.03-0\python.exe` 跑（本地默认 python 是沙箱运行时，没有 paramiko）** |
| `.claude/_deploy.py` | SFTP 上传白名单（models/ + analyse/ + train_scripts/Run1..RunN + 部分 utils）。**新 Run 目录/新 analyse 文件必须加进 UPLOAD 白名单再部署** |
| `.claude/_gen_runN_scripts.py` | 生成 train_scripts/TAR-DCR/RunN/*/train_*.sh（模板：conda 激活、GPU0、300 epoch、retry 循环） |
| `.claude/_run_tests_runN.py` / `_run_smoke_runN.py` | 在服务器 GPU0 跑 smoke + equivalence |
| `.claude/_run_budget_runN.py` | 服务器跑预算脚本 |
| `.claude/_dryrun_runN.py` | LEVIR 2-epoch dry run（写正式 ckpt 目录，正式训练自动续训） |
| `.claude/_launch_runN.py` / `_launch_runN_whu.py` | setsid nohup 启动训练 job + 验证进程/显存 |
| `.claude/_watch_runN.py` | **后台 watcher**：每 300s 轮询日志；全部 done 后自动下载日志 → 跑 extract/models_to_txt → 生成 `outputs/TAR-DCR/RunN/_SUMMARY.md`（含判据裁决与下一步）。用法：`pwsh` 后台 job 运行 |
| `.claude/_check_runN*.py` | 手动查训练进度/日志尾 |
| `.claude/_collect_runN*.py` | watcher 被杀时的手动收集脚本（历史轮次有；新轮若 watcher 死了就写一个：下载日志 + 跑两个 analyse + 生成 _SUMMARY.md） |

---

# 4. 本地标准工作流（新轮次的固定套路）

用户丢来 `docs/temporary/STR-RepNet_RunN_*.md` 并说「开始」后，按序执行：

1. **读设计文档全文**（文档可能 2000+ 行；关键节：模块定义 / 折叠公式 / C0-M1 消融 / 判据 / 执行顺序 / 失败预案）。
2. **改 `models/`**：实现机制（reparam/Mamba_backbone/dcr_decoder/STRRepNet/train.py 接线），
   注意三件事：
   - 零初始化分支的 epoch-0 恒等：**任何 nn.Conv2d/BN 构造都会消耗 RNG**，若新模块位于
     TAR/DCR/head 之前，会把下游共享模块的随机初始化平移掉——要么把安装放到所有 RNG 构造之后
     （Run9 BiFTR 的做法），要么在冒烟里同步 head 权重（Run8 MPCR 的做法）。
   - 零初始化分支的梯度：BN γ=0 时 conv 权重梯度恒为 0，必须"nudge γ 后再验"（Run8）；纯残差
     Δ（无 BN）则 @init 就有梯度（Run9）。
   - 冻结顺序：last2 的 `_setup_encoder_train()` 会冻结 layers[0/1]，新模块装在其内必须
     事后显式 `requires_grad_(True)`（Run9）。
3. **改测试**：smoke + equivalence 各加新机制检查（epoch-0 == 0.0 逐位、梯度、部署分支删除、
   部署 Params/FLOPs == 锚点精确相等、argmax=0；T0 FP64 / T1 block / T2 全模型）。
   FP64 阈值按量级实测校准（Run9 的 AWB 在 O(100) 量级下实测 1.7e-10 → 阈值 1e-9），
   事后不改、注释写清理由。
4. **新 analyse 脚本**（预算验证 / 机制画像）放入 `analyse/`，并把它们与 RunN 目录加进
   `.claude/_deploy.py` 的 UPLOAD 白名单。
5. **本地验证**：`python -m py_compile <files>`（沙箱 python 即可，只查语法）。
6. **部署**：`& "E:\Anaconda\Anaconda3-2023.03-0\python.exe" ".claude\_deploy.py"`。
7. **服务器测试**：`& "...\python.exe" ".claude\_run_tests_runN.py"`（smoke + equivalence，
   GPU0，约 10~15 分钟；后台 pwsh job 跑，注意读 job_output）。失败 → 修 → 重新部署 → 重跑。
8. **预算验证**（若部署图有变）：`_run_budget_runN.py`，硬要求 Δ=0（或 D\*=设计值）。
9. **dry run**：`_dryrun_runN.py`（LEVIR 2 epoch 写正式 ckpt 目录），确认 loss 有限、
   机制梯度/冻结状态正确、TEST 块正常。
10. **生成并部署训练脚本**：`_gen_runN_scripts.py`（C0/M1 × 4 数据集预写，只启动 LEVIR）。
11. **启动 + watcher**：`_launch_runN.py`；随后 `& "...\python.exe" ".claude\_watch_runN.py"`
    作为后台 pwsh job（记住 job id）。用户若要求并行其他数据集，再写 `_launch_runN_whu.py`，
    并把 watcher 扩成多数据集后重启（先 job_kill 旧的）。
12. **README**：主 README 加 RunN 章节（进行中）+ `train_scripts/TAR-DCR/RunN/README.md`。
13. **git**：`git add README.md models/ train_scripts/TAR-DCR/RunN/ analyse/ docs/temporary/` →
    `git commit -m "update code"` → `git push origin main`（代理偶发 SSL 错误，重试即可；
    push 成功判据 = 输出含 `main -> main`）。
14. watcher 完成后：更新 README 为最终结果 → 提交推送 → 汇报裁决与下一步。
    **watcher 若被 kill**（长时间无消息会被平台杀掉）：写 `_collect_runN.py` 手动下载日志 +
    跑两个 analyse + 生成 `_SUMMARY.md`。

---

# 5. 服务器环境与操作

| 项 | 值 |
|---|---|
| SSH | 走 `.claude/_ssh.py`（paramiko，读取 `.vscode/sftp.json`） |
| 代码 | `/home/yqwang/projects/STR-RepNet/` |
| 数据 | `/share_datasets/CD/{CDD,LEVIR,SYSU,WHU}-CD-256/`（各含 A/B/label + `list/{train,val,test}.txt`） |
| 预训练 | `.../STR-RepNet/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth` |
| checkpoint | `/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/RunN/<group>/<dataset>/`（last.pth 续训 + best_F1=xxx.pth） |
| 日志 | `/home/yqwang/outputs/STR-RepNet/TAR-DCR/RunN/<group>/<dataset>/train_log.txt` |
| 诊断输出 | `/home/yqwang/outputs/STR-RepNet/diagnostics/RunN_*/` |
| conda | `source /home/yqwang/miniforge3/etc/profile.d/conda.sh && conda activate strrep`（torch 2.14 cu132，sm_120 内核已编译） |
| GPU | **只用 GPU0**（2×RTX 5090；GPU1 属其他课题；4 job 并行约 28G/32.6G） |

启动训练（模板，务必新行分隔、setsid nohup）：

```bash
cd /home/yqwang/projects/STR-RepNet/train_scripts/TAR-DCR/RunN/<group>
setsid nohup bash train_LEVIR-CD-256.sh 0 >/dev/null 2>&1 </dev/null &
```

查进度：`grep -E '^Epoch' <log> | tail -1`；查进程：`ps aux | grep train.py | grep -v grep`。
日志行格式：`Epoch x/300 | CE=.. | Lovasz=.. | Total=.. | Recall=.. | Precision=.. | OA=.. | F1=.. | IoU=.. | Kappa=..`；
结尾 `[BEST] F1=.. at epoch ..`、`=== TEST RESULTS ===`（含 `[DEPLOY-PARAMS]/[DEPLOY-FLOPS]/[REPARAM-MAX-ABS-ERROR]/[REPARAM-ARGMAX-DISAGREE]/[BEST-F1]` 与各机制标签）。

---

# 6. 当前状态与待办（新对话从这里继续）

- **代码**：main @ `6d5cfee`，全部九轮实现都在（各机制默认关闭，互斥 guard 防止叠加）。
  主干默认配置 = Run2 等价（bilinear、D160、last2、无 botr/nscr/pbru/mpcr/biftr）。
- **服务器**：GPU0 **空闲**（162 MiB），GPU1 空闲但**不得使用**。所有九轮训练已完成，
  checkpoint 都在；无后台 watcher 存活。
- **未决事项**：结构搜索已按预注册结束；用户准备带 GitHub 链接问网页 ChatGPT
  要《论文收尾与最终实验方案》（涉及 multi-seed 3 seed × 4 数据集、贡献组织、
  公平比较、D0 定位）。**等用户把方案文档放进 `docs/temporary/` 后按第 4 节套路执行**。
- **已归档**：九轮结果全在 Excel/README/RunN README/_SUMMARY.md；Run3–Run9 属
  pre-registered mechanism exploration（论文中作负结果边界，不包装成创新）。

---

# 7. 附录

## 7.1 已知坑（务必记住）

1. **pwsh 引号地狱**：inline `python -c` 和带 `$e`/`|`/`^` 的 SSH 命令会被 pwsh 搅碎；
   一律写 `.claude/_*.py` 助手脚本再跑。
2. **本地 python 选择**：SSH/SFTP 脚本必须用 `E:\Anaconda\Anaconda3-2023.03-0\python.exe`
   （本地默认 python 是沙箱运行时，无 paramiko）。
3. **RNG 流**：新模块构造消耗 RNG → 平移下游 init（见 4.2）；epoch-0 逐位一致检查必须显式验证 == 0.0。
4. **零初始化分支梯度**：BN 前置的 conv 分支 @init 梯度为 0（γ=0 阻断），nudge 后才有；纯残差无 BN 则 @init 就有。
5. **downsample 是 v3**（k3 s2 p1），不是文档常见假设的 k2；折叠公式与 kernel 无关，别改骨干。
6. **FP64 阈值**：大维度 GEMM 的纯 FP64 累加在 O(100) 量级下实测 ~1e-10；等价性阈值按实测校准并注释，勿事后改。
7. **watcher 会被平台 kill**（长时间无消息）；回来先 `job_list` 查状态，死了就写 `_collect_runN.py` 手动收。
8. **git push 代理**：偶发 `SSL_ERROR_SYSCALL`，重试；成功判据 `main -> main`。
9. **outputs/ 被 gitignore**：日志本地副本不进 git，只有 README/Excel/快照进。
10. `.sh` 脚本 LF 行尾：`_deploy.py` 上传时服务器端统一转 LF，本地用 `newline="\n"` 生成。

## 7.2 给网页 ChatGPT 的收尾 prompt（上一对话已起草，用户要带去问）

核心内容：GitHub 链接 + 九轮证据表 + 跨轮"置信度锐化"签名 + 预注册结束裁决 +
要求输出《论文收尾与最终实验方案》：①贡献组织（TAR+DCR 部署效率主线 + 负结果边界）；
②multi-seed 方案（3 seed × 4 数据集、GPU0 分批、总时长估算）；③HAM-CD 统计口径诚实声明；
④D0 full-encoder 定位判据；⑤公平比较对照清单；⑥写作顺序；⑦禁止事项清单。
（用户拿到方案后会存 `docs/temporary/` 并发「开始」。）
