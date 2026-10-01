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

## 预训练门槛（Phase 0，全绿才允许 300 epoch）

- [ ] T0 FP64 kernel embedding 代数 <1e-10（目标 <1e-12）+ 嵌入单测精确
- [ ] T1 PFDRDW5 plain/rep 折叠 FP32 <2e-5（非平凡扰动）+ FP64 代数 <1e-10
- [ ] smoke：C0/M1 main-init 一致 + epoch-0 logits 逐位一致（max_diff=0.0）、
      aux γ 梯度非零、γ nudge 后 aux conv 梯度出现、部署仅剩单个 DW5、argmax=0
- [ ] 预算搜索 D*（必须 ≥158；C0/M1 deploy Params/FLOPs 精确相等）
- [ ] T2 全模型（plain/rep）<2e-4、argmax=0
- [ ] LEVIR C0/M1 各 2-epoch dry run（loss 有限、PFDR 梯度/折叠正常）

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
