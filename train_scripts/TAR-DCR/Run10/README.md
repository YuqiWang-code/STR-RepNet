# Run10 PFDR（Pre-Fusion Dilated Re-parameterization）

> 设计文档：[`docs/temporary/STR-RepNet_Run10_PFDR_设计与实验方案.md`](../../../docs/temporary/STR-RepNet_Run10_PFDR_设计与实验方案.md)
> 状态：**进行中**（预训练门槛阶段）

## 方案一句话

在 DCR decoder 的最细尺度 `t1` 进入 `fuse1` 跨尺度混合**之前**，插入一个可部署的
depthwise 5×5 空间条件化模块（PFDR-DW5）；训练期用 `DW5 + DW3 + DW3(d=2) + DW1 + α·I`
多尺度 depthwise basis 做结构重参数化（aux 全部零初始化），部署解析折叠为**单个 DW5×5**
（FP64 组合、一次 FP32 cast）。预算通过 D* 解码器宽度搜索回收到 Run2 锚点内
（28.828706M / 12.6062G）。

## 消融

| 组 | 含义 | deploy graph |
|---|---|---|
| A0 | Run2 full_last2（历史锚点，不重跑） | 无 prefuse |
| C0_PlainPF_DW5 | plain pre-fusion DW5（拓扑/宽度对照，`--pfdr_mode plain`） | DW5 + fuse1 |
| M1_PFDR_DW5 | 多分支训练 / 单 DW5 部署（主实验，`--pfdr_mode rep`） | DW5 + fuse1（与 C0 完全相同） |

归因：Topology = C0−A0；Rep = M1−C0；System = M1−A0。只有 Rep 独立正贡献才能把
PFDR 写成结构重参数化创新。

## 预训练门槛（Phase 0，全部通过 ✅）

- [x] T0 FP64 kernel embedding 代数 2.1e-14（<1e-10，目标 <1e-12）+ 嵌入单测精确
- [x] T1 PFDRDW5 plain 1.4e-06 / rep 2.9e-06 折叠 FP32（<2e-5，非平凡扰动）+ FP64 代数 ~4e-15
- [x] smoke：C0/M1 main-init 一致 + epoch-0 logits 逐位一致（max_diff=0.0）、
      aux γ 梯度非零、γ nudge 后 aux conv 梯度出现、部署仅剩单个 DW5、argmax=0、
      C0/M1 deploy Params/FLOPs 相同
- [x] 预算搜索 **D\*=158**（28.817956M / 12.6028G ≤ 锚点；D=159 FLOPs 超 +0.0065G；
      C0/M1 deploy 精确相等；gate D\*≥158 通过）
- [x] smoke@D\* 锚点预算检查（deploy ≤ Run2 锚点）✅
- [x] T2 全模型（plain 9.9e-05 / rep 5.8e-05 <2e-4）、argmax=0；历史回归全过
- [x] LEVIR C0/M1 各 2-epoch dry run ✅（loss 有限：Total 0.44→0.31；M1 aux γ 2 epoch 已学到非零：
      near3=4.7e-2 / dilated3=5.8e-2 / center1=3.2e-2；TEST 块正常；正式训练从 last.pth 自动续训）
- [x] LEVIR C0/M1 300-epoch 正式训练已启动（GPU0 两 job，watcher 监控中）

## Phase -1 保留度诊断（已完成，先验降级）

冻结 Run2 解码器下，t1 的小/中/大目标 held-out diag-LDA AUROC（0.862/0.880/0.874）
**低于**融合后 fuse1（0.952/0.969/0.936）→ prefusion_retention_supported = False（全部 bin）。
按 doc §6.2 不取消 Run10，先验从「较强」降为「探索性」；bins 复现 Run6
（small ≤502px / medium ≤868px）。JSON：
`outputs/diagnostics/Run10_PFDR/phase1_retention/prefusion_retention_profile.json`。

## 预注册判据（LEVIR，锚点 F1=0.9144）

- **PASS**：M1 F1≥0.9200 且 IoU≥0.8520 且 Precision≥0.9230 且 M1−C0≥+0.15pp 且预算/argmax 合格
- **TARGET-HIT**（额外标志）：LEVIR F1≥0.9250（项目硬目标）
- **WEAK**：0.9159≤F1<0.9200，或 F1≥0.9200 但 IoU/Precision 不足，或 +0.05pp≤M1−C0<+0.15pp
- **FAIL**：F1<0.9159 或 Precision<0.9220 或 M1−C0<+0.05pp 或硬门槛失败
- WEAK/FAIL 一律**不做** branch/dilation/scope/kernel sweep（doc §8）。
- 扩展顺序：PASS → SYSU C0/M1 → WHU C0/M1 → CDD C0/M1（doc §6.5）。

## 结果

（训练结束后由 watcher 生成 `outputs/TAR-DCR/Run10/_SUMMARY.md`，此处补最终表与裁决。）

## 目录

```
train_scripts/TAR-DCR/Run10/
├── C0_PlainPF_DW5/   train_{LEVIR,SYSU,WHU,CDD}-CD-256.sh
└── M1_PFDR_DW5/      train_{LEVIR,SYSU,WHU,CDD}-CD-256.sh
```

- checkpoint：`/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run10/<group>/<dataset>/`
- 日志：`/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run10/<group>/<dataset>/train_log.txt`
- 诊断：`/home/yqwang/outputs/STR-RepNet/diagnostics/Run10_PFDR/`
