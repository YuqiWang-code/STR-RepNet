# STR-RepNet 下一步实验方向、代码审查与参考实现建议

> **审查对象**：`YuqiWang-code/STR-RepNet` 当前 `main`、归档 tag `baseline-hamcd-run1`、仓库 `README.md`、`models/` 代码、`docs/参考文献/文献索引.md`，以及 HAM-CD（IEEE TGRS 2026）论文。  
> **审查日期**：2026-09-22  
> **任务边界**：全监督遥感二值变化检测；CDD-CD-256 / LEVIR-CD-256 / SYSU-CD-256 / WHU-CD-256；A/B/label + list；`gray >= 128`；256×256。  
> **硬约束**：不增加**部署** Params/FLOPs；训练期增强必须严格结构重参数化或可删除；部署前后主预测误差 `<1e-6`；不改造 backbone 结构。

---

# 0. 结论先行

## 0.1 下一步不是 A（D=160→192），而是 **“先修协议与可比性 → 做 B 的定量诊断 → 再做零部署成本的 TAR/DCR 重参数化改造”**

我的优先级是：

1. **P0：先修实验协议和重参数化验收。**
   - 当前 `train.py` 每个 epoch 都在 **test 集**上评估并按 test F1 选 best checkpoint，随后又在同一 test 集做 final test。这是测试集泄漏，当前 Run1 只能作为内部探索结果，不适合作为论文主表最终结果。
   - `README.md` 报告当前整网折叠 `max_abs_error ≈ 1.7e-5`；`test_reparam_equivalence.py` 与 `smoke_test.py` 实际阈值也是 `1e-4`，与本项目硬约束 `<1e-6` 冲突。**在 `<1e-6` 达成前，不应把“严格等价结构重参数化”作为已满足的论文事实。**
   - 当前 `main` 的 `train_scripts/baseline/Run1/*.sh` 仍调用当前 `changedetection/script/train.py`，而当前 `train.py` 已经构建 `STRRepNet`，所以这些脚本从 `main` 直接运行时**不会重新训练 HAM-CD baseline**。HAM-CD 真正代码在 tag `baseline-hamcd-run1`。
   - 归档 HAM-CD baseline 的优化器直接接收 `self.model.parameters()`；当前 STR-RepNet 则显式冻结 encoder 并在 `forward` 中 `torch.no_grad()`。因此“HAM-CD vs STR-RepNet”目前同时混入了**decoder 结构差异 + encoder 是否微调差异**。这很可能是 1–2 个 F1 点差距的重要来源之一，必须拆开。

2. **P1：先做 B，但不是只看“sum/diff 的 BN”。**
   - 当前 TAR 的 sum/diff 分支**没有各自的 BN**，只有所有分支求和后的一个共享 BN；因此“检查 sum/diff 分支 norm 是否学起来”在现代码里并不存在这个对象。
   - 应检查：`proj_s/proj_d` 权重范数、输出 RMS、梯度范数、与 main concat 分支的相对贡献、A/B swap 敏感性、各尺度贡献。
   - 更关键的是：当前 `Concat + Sum + signed-Diff` 在数学上最终仍只是对 `[P,Q]` 的一个线性 `1×1`；在**共享 BN**下，它主要改变优化参数化，并没有形成真正独立的训练期统计路径。这与 TAR 在 LEVIR/SYSU 上弱甚至负的现象一致。

3. **P1：首选结构改进不是加宽，而是“分支独立归一化 + 可折叠残差”的 Rep primitive 升级。**
   - 将当前“多分支 → 求和 → 单个 BN”改为 **“每个线性分支各自 BN → 求和”**，部署时先各自 fold BN，再把 kernel/bias 相加，仍折叠为同一个 `1×1` 或 `DW3×3`。
   - 将当前 `RepDW3(include_identity=True)` 中“identity 混在共享 BN 之前”的形式，改为**可折叠 clean residual**：`BN(F(x)) + αx`，再激活；部署时把 `αI` 加到卷积中心权重。这样训练期有真正的恒等梯度通路，部署仍是一个卷积，没有额外参数/FLOPs。
   - 对 `RepPairFuse1x1` 可把当前尺度 `L` 的 identity 残差折叠进 deploy `1×1` 的 `L` 半边通道权重。
   - 对 TAR，则把 symmetric（sum）/antisymmetric（signed-diff）做成**独立 BN 的训练分支**，并加入 50% A/B 交换增强作为训练协议对照。部署仍是单个 concat `1×1`。

4. **A：D=192 暂不做主线。**
   - 它会增加部署 Params/FLOPs，直接违反“**不提高部署 Params/FLOPs**”这个当前目标。
   - 若只想做“容量上界诊断”，可以临时跑一次，但必须明确它不是最终候选，也不能用于证明零开销创新。

5. **最值得先做的公平性诊断：保持 D=160 和部署图完全不变，只解冻 encoder 后两级。**
   - 这是**零部署 Params/FLOPs 增量**，只增加训练期可训练参数与显存。
   - 如果 LEVIR/SYSU 明显追回差距，说明主要瓶颈不是 DCR 本身，而是“冻结 ImageNet/VMamba 表征无法适配 CD”。
   - 如果几乎不提升，再集中火力改 decoder/TAR，结论会更扎实。

---

# 1. 证据等级说明

本文统一使用四类标记：

- **【代码事实】**：当前 GitHub `main` 或 `baseline-hamcd-run1` 中可以直接确认。
- **【论文事实】**：HAM-CD 或文献索引所列论文可直接支持。
- **【证据支持的推断】**：由代码结构 + Run1 现象共同支持，但尚未通过专项实验验证。
- **【待验证假设】**：下一轮实验需要证伪/证实。

另有一个数据访问限制：

> GitHub 连接器可以确认仓库存在 `docs/experiment_metrics.xlsx`，但该文件是二进制 XLSX，当前 GitHub 文本读取接口不能直接解析其内容。因此本文的 Run1 精确 F1 数值来自你在本次任务中给出的表格；我没有把“无法直接打开的 XLSX”伪装成已独立读取。论文最终表仍应以每个 `train_log.txt` 最后一个完整 `=== TEST RESULTS === ... === END TEST RESULTS ===` 区块为权威来源。

---

# 2. 当前 Run1 应怎样解读

你给出的内部 Run1 F1：

| 数据集 | HAM-CD | Plain | TAR | Full |
|---|---:|---:|---:|---:|
| CDD | 0.9879 | 0.9732 | 0.9735 | 0.9768 |
| WHU | 0.9500 | 0.9355 | 0.9388 | 0.9403 |
| LEVIR | 0.9211 | 0.9009 | 0.9003 | 0.9033 |
| SYSU | 0.8299 | 0.8114 | 0.8058 | 0.8124 |

### 2.1 与 HAM-CD 的绝对差距

| 数据集 | Full − HAM-CD | F1 百分点差距 |
|---|---:|---:|
| CDD | -0.0111 | -1.11 |
| WHU | -0.0097 | -0.97 |
| LEVIR | -0.0178 | -1.78 |
| SYSU | -0.0175 | -1.75 |

**【推断】**真正最需要解决的是 LEVIR/SYSU，而不仅是 CDD/WHU。前两者的差距接近 1.8 个 F1 点，仅靠 `D=160→192` 这种容量扩张未必能解释机制，也不满足当前零部署增量约束。

### 2.2 一个重要的消融解释纠正

当前 `rep_mode` 定义是：

- `plain`：TAR 无 temporal aux；decoder 无 DCR aux。
- `tar`：TAR 有 temporal aux；decoder 无 DCR aux。
- `full`：TAR 有 temporal aux；decoder 有 DCR aux。

所以：

- `TAR - Plain` 可以近似看 temporal rep 的边际作用；
- **`Full - TAR` 才是“在 TAR 已开启条件下，DCR aux 的边际作用”**；
- `Full - Plain` 同时含 TAR、DCR 和二者交互，不能被单独解释为 DCR。

计算得：

| 数据集 | TAR − Plain | Full − TAR（DCR 条件边际） |
|---|---:|---:|
| CDD | +0.0003 | +0.0033 |
| WHU | +0.0033 | +0.0015 |
| LEVIR | -0.0006 | +0.0030 |
| SYSU | -0.0056 | +0.0066 |

因此更严谨的结论是：

- **【代码事实+结果事实】TAR 单独在 LEVIR/SYSU 为负。**
- **【结果支持】DCR 在“已经打开 TAR”的条件下，4 个数据集都是正贡献，约 +0.15～+0.66 F1 百分点。**
- **【缺失信息】没有 `DCR-only`（temporal plain + decoder rep）这一格，因此还不能证明 DCR 在没有 TAR 时也稳定正贡献。**

> **必须补一个 `rep_mode=dcr`。** 这是下一轮最小、最有信息量的消融，不需要重新设计网络。

---

# 3. 当前代码的完整关键数据流

当前 `main` 的主路径可概括为：

```text
A ─┐
   ├─ Frozen shared VMamba-Tiny ─ {A1,A2,A3,A4}
B ─┘
   └─ Frozen shared VMamba-Tiny ─ {B1,B2,B3,B4}

每尺度:
[Ai,Bi]
   │
   ├─ TemporalRep1x1
   │    train: concat + sum + signed-diff -> shared BN
   │    deploy: single 1x1(2Ci -> D)
   │
   ├─ SiLU
   └─ RepLocalBlock
        RepDW3 -> SiLU -> RepPW1x1 -> SiLU
        ↓
      ti

t4
 │ bilinear ↑
 ├─ RepPairFuse1x1(t3, up(t4)) -> SiLU -> RepLocalBlock -> d3
 │ bilinear ↑
 ├─ RepPairFuse1x1(t2, up(d3)) -> SiLU -> RepLocalBlock -> d2
 │ bilinear ↑
 ├─ RepPairFuse1x1(t1, up(d2)) -> SiLU -> RepLocalBlock -> d1
 └─ RepLocalBlock(refine)
       │
       └─ 1x1 head -> bilinear ×4 -> 2-class logits
```

**【代码事实】**当前模型没有 decoder Mamba、Attention、SE；全局建模完全依赖 frozen VMamba encoder，decoder 本身只做局部空间重整与 top-down 融合。

---

# 4. P0 正确性 / 实验协议问题

## P0-1：test 集被用于每 epoch 选 best

当前 `train.py`：

```python
test_loader = self._make_loader(self.args.test_list, ...)
...
# Per-epoch validation on the test set (select best by F1).
rec, pre_, oa, f1, iou, kc = self._evaluate(test_loader)
...
if f1 > self.best_f1:
    ...
```

最终 `test_best()` 又加载这个由 test F1 选出的模型，在同一个 test 上报告。

### 影响

这是**测试集参与模型选择**。即使最后日志有完整 `=== TEST RESULTS ===`，这个结果仍带选择偏差。

### 必改

加入：

```text
--train_list .../list/train.txt
--val_list   .../list/val.txt
--test_list  .../list/test.txt
```

训练期间只在 val 上选 `best_F1`；训练结束后只对 test 做一次 final evaluation。

### 论文处理

- 现有 Run1：保留为“内部探索/方向筛选”。
- 最终论文：HAM-CD baseline 与最终 STR-RepNet 至少需要在**相同 val-selection 协议**下重跑。

---

## P0-2：严格重参数化阈值写的是 `<1e-6`，代码验收却是 `<1e-4`

`test_reparam_equivalence.py`：

- docstring 写“must be `<1e-6`”；
- `check_fold(..., tol=1e-4)`；
- whole model `ok = err < 1e-4`。

`smoke_test.py` 同样：

```python
assert err < 1e-4
```

README 还报告：

```text
max_abs_error ≈ 1.7e-5
```

### 结论

**【代码事实】当前仓库尚未达到项目自己定义的 `<1e-6` 硬门槛。**

### 必改

1. 所有测试统一阈值 `1e-6`；
2. TF32 关闭；
3. branch fold、kernel compose、bias compose 全部先在 FP64 计算，最后一次性 cast FP32；
4. 新增：
   - primitive-level；
   - TAR stage-level；
   - decoder-level；
   - whole-model logits；
   - 至少 10 组随机输入；
5. 任一超过 `1e-6`，full training 暂停。

> 不建议为了“过测试”把比较本身改成 FP64 推理；最终部署图与训练图都应在目标 FP32 推理条件下验证。

---

## P0-3：当前 main 下的 baseline shell 脚本已经不能复现 HAM-CD

当前：

```text
train_scripts/baseline/Run1/*.sh
```

调用：

```text
models/changedetection/script/train.py
```

但 `main` 中这个 `train.py` 现在构建的是 `STRRepNet`。

真正 HAM-CD baseline 代码在：

```text
git tag: baseline-hamcd-run1
commit: a90813164fa82df36ca709e4aae508aac5529241
```

### 必改

推荐二选一：

- **方案 1（推荐）**：保留 tag，不在 main 伪装可直接复现；把 `train_scripts/baseline/README.md` 写清楚“必须 checkout tag”。
- **方案 2**：在 `baseline_legacy/` 存只读代码快照，但避免和主模型 import 路径混淆。

---

## P0-4：HAM baseline 与 STR-RepNet encoder 训练条件不一致

归档 tag 的 HAM-CD `train.py`：

```python
self.optimizer = optim.AdamW(self.model.parameters(), ...)
```

且 `STMambaBCD.forward()` 直接运行 encoder，没有冻结。

当前 STR-RepNet：

```python
for p in self.encoder.parameters():
    p.requires_grad_(False)
...
with torch.no_grad():
    pre_feats = self.encoder(pre)
    post_feats = self.encoder(post)
```

并覆盖 `train()` 强制：

```python
self.encoder.eval()
```

### 结论

**【代码事实】HAM baseline 是 encoder 可训练，当前 STR-RepNet 是 encoder 完全冻结。**

所以当前 1–2 点差距不能全部归因于“轻量 decoder 不如 HAM”。

### 下一步必须补的公平诊断

部署图完全不变，仅改变训练策略：

```text
E-F0: Full-D160, encoder frozen（当前）
E-F1: Full-D160, 只解冻 stage3+stage4 + outnorm
E-F2: Full-D160, 全 encoder fine-tune（仅作为上界诊断）
```

先在 LEVIR/SYSU 做。

---

## P0-5：归档 HAM decoder 里有一个明确的 stage attention 调用异常

归档 HAM decoder 初始化了：

```python
self.at_layer_11 = TransformerBlock(...)
```

但 Stage IV forward 写的是：

```python
p12 = self.at_layer_21(p10)
```

而不是：

```python
p12 = self.at_layer_11(p10)
```

### 影响

- `at_layer_11` 变成死参数；
- `at_layer_21` 被 Stage III 和 Stage IV 共享；
- 参数统计仍会把未使用的 `at_layer_11` 算进去；
- 历史 baseline 结果对应的是这个实际执行图，而不一定是论文意图图。

### 处理原则

不要偷偷修改后还沿用旧 HAM 数值。论文最终 baseline 应明确选择：

1. **历史代码复现版**：保持 bug，结果与旧 Run1 一致；
2. **修正版 HAM**：修复后重新完整训练，并明确标记 corrected reproduction。

更推荐第 2 种用于最终公平比较。

---

# 5. P1 方法瓶颈审查

## P1-1：当前 TAR 的三路在共享 BN 下过度“代数冗余”

当前：

\[
y=W_c[P,Q] + W_s(P+Q)+W_d(Q-P),
\]

等价于：

\[
y=(W_{c,P}+W_s-W_d)P + (W_{c,Q}+W_s+W_d)Q.
\]

也就是说，最终仍然只是一个任意的 `[P,Q] → 1×1` 线性映射。

### 当前结构的问题不是“不能折叠”，而是“训练分支缺少真正异构性”

所有三路先相加，再走同一个 BN：

```python
y = proj_c(cat)
y += proj_s(P+Q)
y += proj_d(Q-P)
return bn(y)
```

因此：

- 每个 temporal basis 没有自己的 batch statistics / affine scaling；
- sum/diff 没有独立 normalization；
- zero-init 后的辅助分支容易沦为对 main kernel 的另一种参数分解；
- `signed-diff` 是方向敏感的，而二值 CD 标签通常应对 A/B 交换保持语义不变。

**【证据支持的推断】**这非常符合 TAR 在 LEVIR、尤其 SYSU 上不稳定的现象。

---

## P1-2：当前 DCR 的“identity”不是 clean residual

`RepDW3(include_identity=True)`：

```python
y = dw3(x) + aux(x) + x
return bn(y)
```

它是：

\[
BN(F(x)+x)
\]

而不是更典型的：

\[
BN(F(x)) + x.
\]

这两者训练动力学不同。前者的 identity 也被 BN 的 batch statistics、γ/β 一起缩放/平移。

### 为什么值得改

HAM-CD 的 ICSF 以及其 decoder 都大量使用残差式信息保留；HAM 论文还专门强调减少逐阶段信息损失。当前 DCR top-down 每一级都：

```text
fuse -> SiLU -> RepLocalBlock -> upsample
```

没有 clean same-scale identity 直通路径。

### 但不能直接加普通 block skip

如果在整个：

```text
DW -> SiLU -> PW -> SiLU
```

外面直接 `+x`，这个 skip 无法跨过中间非线性折叠进单个 DW/PW，会留下部署分支。

### 正确做法：**每个线性算子内部做 foldable residual**

例如 DW：

\[
z = BN(F_{DW}(x)) + \alpha x,\quad y=SiLU(z)
\]

部署：

\[
W' = W_{BN(F)}+\alpha I,\quad b'=b_{BN(F)}.
\]

仍然是：

```text
single DW3 -> SiLU
```

PW 同理。

Cross-scale fuse：

\[
z = F([L,H]) + \alpha L,
\]

把 `αI` 直接加到 deploy `1×1(2C→C)` 的前 C 个输入通道上。

---

## P1-3：RepPW1x1 的低秩支路第一个 step 存在梯度延迟

当前初始化：

```python
pw1: Kaiming
pw2: zeros
```

branch：

```python
pw2(pw1(x))
```

训练初始时：

- `pw2` 能收到梯度；
- 因为 `pw2=0`，传回 `pw1` 的梯度初始为 0。

这不是正确性 bug，但会让低秩两层中的前半段至少在最初更新滞后。

### 建议

作为小型对照可把：

```python
pw2.weight ~ Normal(0, 1e-3)
```

而不是全零，使 branch 初值仍近零但两层从第一个 batch 就有梯度。

**不建议把它单独包装成论文创新。**

---

## P1-4：decoder 的最高分辨率只到 1/4，再直接 bilinear 到 256²

VMamba stage1 通常是 1/4 尺度。当前：

```python
logits = self.head(d1)
logits = F.interpolate(logits, input_size, bilinear)
```

### 推断

这可能限制：

- LEVIR / WHU 小建筑；
- 细边界；
- SYSU 中碎片化变化。

HAM-CD 论文的定性与 limitation 也都反复强调精细边界。

### 但当前不建议加高分辨率 learnable decoder stage

因为那会增加部署 FLOPs。

更符合硬约束的方案是：

- 在**训练期**对主 logits 施加 boundary-aware auxiliary objective；
- 或训练期 auxiliary edge head，部署删除；
- 不改变主推理图。

这应作为第二阶段补充，而不是第一主线。

---

## P1-5：激活函数本身不是当前主要错误

当前 decoder 统一 `SiLU`。

### 判断

- **不是明显错误。**
- HAM/ICSF 多用 ReLU；VMamba 内部有 SiLU/GELU，但不存在“因此 decoder 必须用某一种激活”的硬结论。
- 结构重参数化只要求：需要折叠的多条**线性**路径在激活之前完成合并。当前这一点是对的。
- 不要把 ReLU/GELU/SiLU sweep 当成创新。

### 一个小问题

当前 Kaiming 初始化写：

```python
nonlinearity="relu"
```

实际后面是 SiLU。PyTorch Kaiming 没有标准 `silu` gain，这种写法常见但并不完全匹配。优先级远低于 encoder freeze、TAR branch design、test leakage。

---

## P1-6：BN 不是“缺失”，而是“位置和粒度不理想”

当前每个 Rep primitive 都有一个共享 BN，所以不能说“模型缺 BN”。

更准确地说：

> **缺的是 branch-specific BN，而不是缺 BN。**

结构重参数化经典工作（RepVGG / DBB / UniRepLKNet）的重要训练优势之一，就是让不同训练分支拥有独立的归一化/仿射统计，然后在 eval/deploy 时逐支折叠。

---

# 6. P2 实验工程与代码质量问题

## P2-1：`lovasz_loss.py` 使用 `is` 比较字符串

当前：

```python
if (classes is 'present' and fg.sum() == 0):
```

应改：

```python
if (classes == 'present' and fg.sum() == 0):
```

`is` 比较对象身份，不应作为字符串值判断。

---

## P2-2：metrics 对零分母没有保护

`Precision / Recall / F1 / IoU` 中多个分式没有 epsilon。

极端 batch/测试集下可能产生 NaN。虽然四个完整 test 集大概率不会触发，但发布代码应修。

---

## P2-3：随机性没有完整固定

`train.py` 只显式：

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

而 augmentation 大量使用 Python `random`。

建议：

```python
random.seed(seed)
torch.cuda.manual_seed_all(seed)
```

并为 DataLoader 提供 deterministic `generator` / `worker_init_fn`。

---

## P2-4：`random_rot()` 永远旋转，不存在 0°

当前：

```python
k = random.randrange(3) + 1
```

即只会 90°/180°/270°。

如果原意是均匀四方向，应：

```python
k = random.randrange(4)
```

这不是一定导致性能下降，但必须明确训练协议。

---

## P2-5：`models/README.md` 已过时

当前它仍然是原 HAM-CD README，包含：

- `MambaBCD_Small`
- 原始 HAM-CD 命令
- T1/T2/GT 目录描述

与当前 STR-RepNet 主实现不一致。应更新或明确标记为 upstream archive。

---

# 7. 首选改进机制：Branch-Normalized Foldable Residual Re-parameterization

这里先用工作名 **BN-FR Rep**，不要急着把名字写进论文标题。先验证机制。

## 7.1 核心目标

在**完全不改变 deploy operator graph、deploy Params、deploy FLOPs**的前提下，提升训练期的：

- 分支异构性；
- 梯度通路；
- temporal basis 的可学习性；
- 局部与跨尺度信息保留。

---

## 7.2 RepDW3-v2

### 训练图

```text
             DW3x3 -> BN3 ----\
             DW1x3 -> BN13 ----+
x ---------- DW3x1 -> BN31 ----+--> sum --> + αx --> SiLU
             (optional)
```

也可以把 identity 自己做 BN，但首版建议 clean `αx`，减少不必要分支。

### 部署图

```text
x -> single DW3x3 -> SiLU
```

### 折叠

每支：

\[
(W_i,b_i)=Fold(Conv_i,BN_i)
\]

非对称 kernel pad 到 3×3，然后：

\[
W_{eq}=\sum_i Pad(W_i)+\alpha I,\quad
b_{eq}=\sum_i b_i.
\]

---

## 7.3 RepPW1x1-v2

### 训练图

```text
main 1x1 --------> BNm --\
C->r->C ---------> BNlr --+--> sum --> + αx --> SiLU
diag/channel scale -------/
```

低秩支路内部**禁止插入激活**，否则不能精确 compose 成一个 1×1。

### 部署图

```text
single 1x1 -> SiLU
```

---

## 7.4 RepPairFuse1x1-v2

### 训练图

```text
cat[L,H] -> 1x1 -> BNc ----\
L+H      -> 1x1 -> BNs -----+
H-L      -> 1x1 -> BNd -----+--> sum --> + αL --> SiLU
```

### 部署图

```text
cat[L,H] -> single 1x1 -> SiLU
```

`αL` 折入 deploy kernel 的 `L` 通道半边。

---

## 7.5 TemporalRep1x1-v2（最关键）

### 训练图

```text
cat[P,Q] -> 1x1 -> BNc ----\
P+Q      -> 1x1 -> BNs -----+--> sum --> SiLU
Q-P      -> 1x1 -> BNd -----/
```

### 部署图

```text
cat[P,Q] -> single 1x1 -> SiLU
```

### 与当前 TAR 的实质区别

当前：

```text
(main + sum + diff) -> shared BN
```

v2：

```text
BN(main) + BN(sum) + BN(diff)
```

训练态 BN 使用各自 batch statistics，因此三条路径不再只是“一个共享 BN 前的线性重写”；部署 eval 时它们又都变成仿射算子，可以完全折叠。

这才更接近 DBB/RepVGG/UniRepLKNet 的结构重参数化训练逻辑。

---

# 8. TAR 专项诊断：先做什么、记录什么

不要先完整重训很多版本。先给当前 checkpoint 加 diagnostic hook。

## 8.1 每尺度记录

对 `stage1..4`：

```text
||W_c||
||W_s||
||W_d||
||grad(W_c)||
||grad(W_s)||
||grad(W_d)||
RMS(main_output)
RMS(sum_output)
RMS(diff_output)
cos(main,sum)
cos(main,diff)
cos(sum,diff)
```

每 10 epoch 写一次汇总。

### 判据

- 若 `W_s/W_d` 长期接近 0，说明 aux branch 未充分参与；
- 若输出 RMS 明显比 main 小一个数量级以上，也说明 branch 弱；
- 若 `diff` 在 SYSU 梯度大但最终贡献负，可能是方向敏感/噪声放大；
- 若 stage1/2 有效、stage3/4 无效，可尝试只在浅层启用 temporal aux，而不是四级全开。

---

## 8.2 A/B swap 诊断

在 eval：

```python
logit_ab = model(A, B)
logit_ba = model(B, A)
```

记录：

```text
mean_abs(logit_ab - logit_ba)
argmax disagreement ratio
F1(A,B)
F1(B,A)
```

对于二值“是否发生变化”，A/B 交换理论上不应大幅改变语义结果。

### 训练期最小修复

50% 概率交换 A/B：

```text
(A,B,label) -> (B,A,label)
```

无额外部署开销，也不需要第二次 forward。

---

# 9. 为什么 HAM-CD 比当前模型强：不是简单“因为有 Mamba/Attention”

HAM-CD 论文对 LEVIR 的主要消融显示：

- HAM 与 ICSF 都有明显正贡献；
- ICSF 内的 spatial/channel enhancement 有互补作用；
- parallel TAM/TAB 优于 serial；
- concat fusion 优于 direct summation；
- 4-stage decoder 优于更浅或更深配置。

当前 STR-RepNet 为了部署轻量化，删除了 HAM/TAB/SE/Mamba，这没问题；问题是同时也删掉了几类**训练和数据流上的能力**：

1. HAM 有明确的 global/local 双路互补；
2. ICSF 有 cross-stage residual + spatial/channel harmonization；
3. HAM decoder 的残差结构比当前 DCR 更强；
4. 当前 encoder 又被冻结，导致 decoder 接收到的特征不能适配 CD；
5. 当前 TAR 的 sum/diff 只是共享 BN 前的线性重参数化，训练异构性不足。

因此下一步不应“把 HAM 模块再塞回来”，而是：

> **把 HAM/ICSF 的“信息保留与异构训练路径”转译成可完全折叠的线性训练结构。**

这正是 STR-RepNet 应有的研究主线。

---

# 10. 文献里最值得借鉴的思想

## 10.1 DBB（CVPR 2021）——最直接的代码参考

官方仓库：

`https://github.com/DingXiaoH/DiverseBranchBlock`

最值得看：

- `diversebranchblock.py`
- `dbb_transforms.py`
- BN + branch kernel 的 equivalent conversion；
- serial conv branch 的 kernel composition；
- identity / avg / multi-scale branch 的融合方式。

### 对 STR-RepNet 的启发

不是复制 DBB block，而是把它的**“各分支先独立归一化，再等价变换并相加”**原则迁移到：

- temporal sum/diff；
- cross-scale pair fusion；
- DW local refinement。

---

## 10.2 UniRepLKNet（CVPR 2024）——验证“异构空间分支 + 独立 BN + 单核部署”的现代做法

官方仓库：

`https://github.com/AILab-CVC/UniRepLKNet`

论文的 Dilated Reparam Block 使用多条 dilated small-kernel + BN 分支，部署合并为一个 non-dilated large kernel。

### 对本项目的启发

- 不建议直接上 13×13 大核，因为会改变 deploy kernel/FLOPs；
- 重点参考其：
  - per-branch BN；
  - sparse/dilated kernel 到 dense equivalent kernel 的实现；
  - FP32 equivalence verification。

---

## 10.3 OREPA（CVPR 2022）——如果训练期分支越来越多，用在线重参数化控制训练成本

官方仓库：

`https://github.com/JUGGHM/OREPA_CVPR2022`

### 对本项目的启发

若未来 BN-FR 分支太多导致训练显存/速度明显变差，可研究在线聚合权重，而不是继续堆显式分支。

**当前先不要把 OREPA 直接并入主方法。**

---

## 10.4 RepVGG（CVPR 2021）——clean identity / BN folding 的基准实现

官方仓库：

`https://github.com/DingXiaoH/RepVGG`

重点用作：

- branch BN；
- identity branch；
- `switch_to_deploy()`；
- deploy consistency test 的工程参考。

---

## 10.5 ASR（ECCV 2024）——“推理零成本注意力”值得保存，但不是下一步首选

官方仓库：

`https://github.com/zhongshsh/ASR`

ASR 的核心动机是解决 attention 输入依赖、乘法结构难以直接折叠的问题，并提出 inference cost-free attention-like SRP。

### 对本项目的意义

它证明“训练期非普通卷积增强、推理零成本”仍有空间。

### 为什么暂不首发

- 当前最大证据已经指向 TAR 弱、encoder freeze、残差/BN 参数化；
- 直接上 attention-like SRP 会扩大变量空间；
- 先把纯代数重参数化做到强、稳、可解释，再考虑 ASR 类机制。

---

## 10.6 CD-RLKNet（IJAEO 2024）——遥感 CD + reparam large kernel 的直接 prior art

论文页面给出的官方代码：

`https://github.com/juncyan/cdrlknet.git`

### 应重点看

- temporal/spatial adaptive fusion；
- bi-temporal feature integration；
- large-kernel reparam 在 CD 中如何表述创新边界。

### 对本项目的提醒

论文里不能把“遥感 CD 使用结构重参数化”本身写成首创；创新必须更具体落到：

- temporal algebraic branch；
- decoder-wide composition；
- strict deploy-equivalent cross-scale fusion；
- 或训练期 symmetry / branch-statistics design。

---

## 10.7 ChangeMamba（TGRS 2024）——看 residual fusion，不是拿 Mamba 回来

官方仓库：

`https://github.com/ChenHongruixuan/ChangeMamba`

重点看其 cross-temporal / cross-stage residual fusion 的数据流。

### 对 STR-RepNet 的借鉴

把“保留当前尺度信息”的思想改写成**可折叠 residual**，而不是引入 SSM。

---

## 10.8 HAM-CD（TGRS 2026）——保留 upstream 作为 baseline 对照

官方仓库：

`https://github.com/guanguanboy/HAM-CD`

当前 STR-RepNet 源自 HAM-CD 体系，但建议仍单独保留 upstream clone，方便：

- diff decoder；
- 核对 ICSF；
- 核对论文与代码实现偏差；
- 最终复现 baseline。

---

# 11. 建议现在下载到 `others/` 的代码

## 11.1 第一优先级：现在就下载

```bash
cd /home/yqwang/projects/STR-RepNet
mkdir -p others
cd others

git clone https://github.com/DingXiaoH/DiverseBranchBlock.git
git clone https://github.com/AILab-CVC/UniRepLKNet.git
git clone https://github.com/JUGGHM/OREPA_CVPR2022.git
git clone https://github.com/DingXiaoH/RepVGG.git
git clone https://github.com/ChenHongruixuan/ChangeMamba.git
git clone https://github.com/guanguanboy/HAM-CD.git HAM-CD-upstream
git clone https://github.com/juncyan/cdrlknet.git CD-RLKNet
git clone https://github.com/zhongshsh/ASR.git
```

### 目的

| Repo | 主要参考点 | 是否直接复制模块 |
|---|---|---|
| DiverseBranchBlock | branch BN / kernel transform / serial compose | 否，只参考数学与实现 |
| UniRepLKNet | modern dilated reparam / independent BN | 否 |
| OREPA | online rep / 训练成本 | 暂不 |
| RepVGG | identity + BN folding | 否 |
| ChangeMamba | CD residual / temporal fusion | 否 |
| HAM-CD-upstream | baseline / ICSF / HAM | 仅核对 |
| CD-RLKNet | CD + reparam prior art | 否 |
| ASR | inference-cost-free attention 思路 | 后续 |

> `others/` 应加入 `.gitignore`，避免把多个外部仓库整体提交进 STR-RepNet。

---

## 11.2 第二优先级：不必现在下载

- RepViT
- FastViT
- MobileOne
- PVMamba
- MambaVision
- PAT
- ViT-Linearizer
- KD 类仓库

原因：当前最紧迫的问题不是 backbone/蒸馏/latency framework，而是 TAR/DCR 的训练参数化和公平实验协议。

---

## 11.3 暂不建议下载未核验官方仓库的论文代码

文献索引中的：

- LKMamba-CD
- EAFH-Net
- DMFANet
- ST-Mamba
- DEIF-Mamba

论文 PDF 可以继续精读，但如果找不到**论文作者/出版社明确链接的官方仓库**，不要随便把同名第三方 GitHub 当官方参考代码。

---

# 12. 最小实验设计

## Phase 0：代码修复，不做方法结论

### P0-A：评估协议

修改：

```text
models/changedetection/script/train.py
train_scripts/TAR-DCR/...
train_scripts/baseline/...
```

要求：

- train → optimization
- val → checkpoint selection
- test → final once
- final block保持：
  `=== TEST RESULTS === ... === END TEST RESULTS ===`

### P0-B：严格部署等价

修改：

```text
models/changedetection/script/test_reparam_equivalence.py
models/changedetection/script/smoke_test.py
```

硬门槛：

```text
max_abs_error < 1e-6
```

### P0-C：小代码修复

- Lovász `is` → `==`
- metrics 零分母保护
- `random.seed`
- DataLoader worker seed
- 明确 rotation 是否包括 0°
- 更新 `models/README.md`

**成功判据**：smoke + dry-run + deploy equivalence 全过，且现有模型在不改数学图时 F1 不应发生异常漂移。  
**失败判据**：任何 fold >1e-6、val/test 路径混用、resume 后结果协议不一致。

---

# 13. Phase 1：先定位“冻结 encoder”是否是主要瓶颈

先只跑 **LEVIR + SYSU**，因为它们对当前方法最敏感。

## E1-0 当前控制组

```text
Full / D=160 / frozen encoder
```

## E1-1 首选公平诊断

```text
Full / D=160 / unfreeze encoder stage3+stage4
```

- backbone 结构不改；
- deploy Params/FLOPs 不变；
- optimizer 给 encoder 后两级较小 LR，例如 decoder `1e-4`，encoder `1e-5`；
- 仍 300 epoch，或先 100~120 epoch screening 后再完整 300。

## E1-2 仅作为上界

```text
Full / D=160 / full encoder fine-tune
```

### 成功判据

若 E1-1 相对 E1-0：

- LEVIR `+≥0.5` F1 百分点；
- SYSU `+≥0.5`；
- 且训练稳定；

则“冻结 encoder”是实质瓶颈。

若能追回当前与 HAM 差距的 **≥50%**，建议最终模型允许 partial fine-tune，因为部署开销没有增加。

### 失败判据

- 两数据集均 `<+0.2`；
- 或一个明显提升、另一个下降 `>0.3`；
- 则不再把 encoder 解冻作为主要路线。

---

# 14. Phase 2：补齐 DCR-only，修正当前消融矩阵

新增：

```text
rep_mode = dcr
```

逻辑：

```python
use_temporal_aux = rep_mode in ("tar", "full")
use_dcr_aux = rep_mode in ("dcr", "full")
```

得到真正 2×2：

| temporal rep | decoder rep | 名称 |
|---|---|---|
| off | off | Plain |
| on | off | TAR |
| off | on | DCR |
| on | on | Full |

### 价值

可以直接测：

\[
\Delta DCR|TAR=off = DCR-Plain
\]

与：

\[
\Delta DCR|TAR=on = Full-TAR.
\]

只有这样才能声称 DCR 是否“独立稳定有效”。

---

# 15. Phase 3：主方法候选——BN-FR Rep

不要同时上 5 个新模块。先做两个必要版本：

## E3-0

当前 `Full-D160`。

## E3-1：Branch BN only

只改：

- TemporalRep1x1
- RepDW3
- RepPW1x1
- RepPairFuse1x1

从 shared BN 改 per-branch BN，deploy 完全相同。

## E3-2：Branch BN + Foldable Residual（主候选）

在 E3-1 基础上加入每个线性 op 内部的 `αI` foldable residual。

### 初筛数据集

LEVIR + SYSU。

### 初筛成功判据

相对 E3-0：

- 两数据集都 `+≥0.30` F1 百分点；
- 任一不允许下降超过 `0.10`；
- deploy Params/FLOPs 与 E3-0 相同；
- whole-model fold `<1e-6`。

满足后再跑 CDD/WHU。

### 失败判据

- LEVIR/SYSU 平均提升 `<0.20`；
- 或 SYSU 继续显著负；
- 或需要保留部署分支才能提升；
- 或 fold 误差无法压到 `<1e-6`。

失败则停止该路线，不继续堆模块。

---

# 16. Phase 4：如果边界仍是主要误差，再加“训练期边界监督”

只在 BN-FR 已证明有效后考虑。

可选：

```text
L = CE + 2.0*Lovasz + λ_edge*L_edge
```

其中 `L_edge` 直接从主 logits / GT 的边界构造，不引入推理头。

### 目标

重点看：

- LEVIR / WHU Recall；
- boundary FN；
- 小建筑漏检。

### 成功判据

- LEVIR 或 WHU `+≥0.25` F1；
- SYSU/CDD 不下降超过 `0.10`；
- deploy 图完全不变。

它是辅助训练策略，不应冒充结构重参数化主创新。

---

# 17. 不建议现在做的事情

## 17.1 不建议 D=192 直接作为下一步主实验

原因：

- 增加 deploy Params/FLOPs；
- 不能解释 TAR 为什么负；
- 即使提升，也容易被审稿人解释为“更多容量”。

如果一定要做，只把它放在：

```text
capacity upper bound
```

不进入最终主方法。

---

## 17.2 不建议把 HAM/TAB/SE/Mamba 塞回 decoder

这会：

- 破坏“训练复杂、部署单路径”的核心叙事；
- 增加部署算子；
- 让论文回到 HAM-CD 的混合注意力路线。

---

## 17.3 不建议先做 KD

PAT / ViT-Linearizer / MHKD 等适合第二篇或后续增强，但现在引入 teacher 会模糊 STR 的主创新。

---

# 18. 逐文件修改清单

## `models/changedetection/models/reparam.py`

### 必改

- 新增 `fold_conv_bn_fp64()`；
- 支持 per-branch BN；
- 新增 identity kernel merge；
- RepPairFuse 支持 `αL` fold；
- 所有 kernel/bias 在 FP64 中合并后一次 cast。

### 新增诊断 API

```python
branch_stats()
get_equivalent_kernel_bias()
```

---

## `models/changedetection/models/tar.py`

### 必改

- `TemporalRep1x1V2`：
  - concat branch BN
  - sum branch BN
  - diff branch BN
- 提供 branch diagnostic；
- 保证 deploy 仍是一个 `Conv2d(2Ci,D,1,bias=True)`。

---

## `models/changedetection/models/dcr_decoder.py`

### 必改

- `RepLocalBlock` 使用 v2 primitives；
- 不加无法折叠的 outer residual；
- cross-scale fuse 改 foldable same-scale residual。

---

## `models/changedetection/models/STRRepNet.py`

### 必改

新增：

```text
rep_mode = plain | tar | dcr | full | bnfr
encoder_train = frozen | last2 | full
```

不要复制多套模型类。

---

## `models/changedetection/script/train.py`

### P0 必改

- `--val_list`
- val 选 best
- test final once
- random seed 完整
- param groups 支持 encoder LR multiplier
- 最终日志同时写：
  - Recall
  - Precision
  - OA
  - F1
  - IoU
  - Kappa
  - train graph params
  - trainable params
  - deploy params
  - deploy FLOPs
  - reparam max abs error

---

## `models/changedetection/script/test_reparam_equivalence.py`

阈值改为：

```text
1e-6
```

并增加多随机输入 / 多尺度 / v2 block。

---

## `models/changedetection/script/smoke_test.py`

同样 `<1e-6`。

---

## `models/changedetection/datasets/make_data_loader.py`

加入可选：

```text
temporal_swap_prob=0.5
```

A/B 交换必须保持 label 不变。

---

## `models/changedetection/utils_func/lovasz_loss.py`

`is` → `==`。

---

## `models/changedetection/utils_func/metrics.py`

所有二分类分母加安全 epsilon。

---

# 19. smoke / dry run / deployment 验证

## 19.1 本地/服务器 smoke

顺序：

```text
1. build model
2. one forward
3. one backward
4. branch stats
5. switch_to_deploy
6. compare logits
7. measure deploy Params/FLOPs
```

硬门槛：

```text
max_abs_error < 1e-6
```

---

## 19.2 真实数据 dry run

每个数据集只取 2~4 batch：

- label unique 只能是 `{0,1}`（padding 情况另含 255）；
- A/B/label 几何增强同步；
- temporal swap 后 label 不变；
- loss finite；
- branch grad 非零；
- encoder freeze/last2 状态正确。

---

# 20. 日志与 checkpoint 建议

建议新一轮：

```text
/share_datasets/yqwang/checkpoints/STR-RepNet/
  TAR-DCR/
    Run2_protocol_fix/
    Run3_fair_encoder/
    Run4_BNFR/
```

日志：

```text
/home/yqwang/outputs/STR-RepNet/
  TAR-DCR/
    Run2_protocol_fix/
    Run3_fair_encoder/
    Run4_BNFR/
```

每 run 保持：

```text
last.pth
best_val_F1=xxxx.pth
train_log.txt
```

最终 `train_log.txt` 必须只有**一次最终 test block**：

```text
=== TEST RESULTS ===
...
=== END TEST RESULTS ===
```

---

# 21. 最终论文级成功标准

你目前希望“必须超过 HAM-CD”。这个目标可以保留，但要换成**公平协议后的目标**：

1. HAM-CD corrected reproduction 与 STR-RepNet：
   - 相同 train/val/test；
   - 相同 seed；
   - 相同 epoch；
   - 相同 augmentation；
   - 相同 loss；
   - 相同 pretrained encoder 初始化；
   - 都用 val 选 best；
   - test 只测一次。

2. 最终候选至少 3 seeds。

3. 不能仅报告 F1：
   - Recall / Precision / OA / F1 / IoU / Kappa；
   - training graph Params；
   - trainable Params；
   - deploy Params；
   - deploy FLOPs；
   - deploy equivalence error。

4. 论文主张建议分两层：
   - **机制主张**：BN-FR / temporal basis rep 在零部署增量下提升。
   - **系统主张**：在低于 HAM-CD 的 deploy Params/FLOPs 下达到或超过其准确率。

5. 若只在 1 个数据集或 1 个 seed 超过，不写“普适提升”。

---

# 22. 立即执行顺序

1. **修 `train.py` 的 val/test 泄漏。**
2. **把所有 reparam 测试阈值改为 `<1e-6`，先解决 README 中约 `1.7e-5` 的现状。**
3. **修 `lovasz_loss.py`、metrics、seed、README。**
4. **新增 `rep_mode=dcr`，补完整 2×2 消融。**
5. **给当前 TAR 加 branch norm / output RMS / grad norm / A↔B swap diagnostic。**
6. **在 LEVIR + SYSU 跑 encoder `frozen vs last2-unfreeze` 的公平诊断。**
7. **实现 per-branch BN。**
8. **实现 foldable clean residual。**
9. **只在 LEVIR + SYSU 初筛 BN-FR；过阈值后再跑四数据集。**
10. **最终才决定是否需要训练期 boundary supervision。**
11. **D=192 暂不进入主线。**

---

# 23. 仍需补充证据

为了下一次直接进入代码修改与 Run2/Run3 启动，还需要：

1. 四个历史 HAM baseline `train_log.txt` 的最后完整 test block；
2. 四个 Plain/TAR/Full 的最后完整 test block；
3. 当前 `experiment_metrics.xlsx` 若方便，可直接上传到对话，而不是只放 GitHub 二进制；
4. 当前 28.83M / 12.61G 对应的完整日志，因为 README 仍写约 29.5M，二者需要按“最终 test block > README”统一；
5. 若准备修正 HAM baseline，需决定是否修复 `at_layer_21(p10)` → `at_layer_11(p10)` 后重新跑；
6. 最好提供 LEVIR/SYSU 的定性 FP/FN 图，用于判断下一步是 temporal confusion、边界漏检还是背景伪变化主导。

---

# 24. 最终判断

**当前 Clean TAR-DCR 不是“结构重参数化方向失败”，但当前版本还不能证明它已经找到正确的 temporal reparameterization。**

最明确的证据是：

- DCR 在 `Full-TAR` 条件边际下四数据集均为正；
- TAR 在 LEVIR/SYSU 为负；
- TAR 当前数学上高度冗余，且没有 per-branch BN；
- decoder 缺少可保留 identity 的 clean residual；
- STR encoder 完全冻结，而 HAM baseline encoder 可训练；
- 当前实验还存在 test 选 best 的协议问题；
- 严格 `<1e-6` deploy equivalence 尚未达标。

因此，**下一轮最优策略不是“把网络做宽”，而是把实验变公平、把重参数化做得更“真”、把 temporal branch 变得有独立训练统计，同时保持部署图原封不动。**

如果 **partial encoder fine-tune + BN-FR** 能把 LEVIR/SYSU 分别追回约 1.5～2.0 个 F1 点，而部署仍保持当前 D=160 的单路径图，那么这条路线就具备真正的论文价值：不是用更大的推理网络追精度，而是用更强的训练图优化同一个轻量部署图。

---

# 参考链接（已核验的官方/论文来源）

- STR-RepNet：`https://github.com/YuqiWang-code/STR-RepNet`
- HAM-CD：`https://github.com/guanguanboy/HAM-CD`
- DBB：`https://github.com/DingXiaoH/DiverseBranchBlock`
- RepVGG：`https://github.com/DingXiaoH/RepVGG`
- OREPA：`https://github.com/JUGGHM/OREPA_CVPR2022`
- UniRepLKNet：`https://github.com/AILab-CVC/UniRepLKNet`
- ASR：`https://github.com/zhongshsh/ASR`
- ChangeMamba：`https://github.com/ChenHongruixuan/ChangeMamba`
- CD-RLKNet：`https://github.com/juncyan/cdrlknet`
- UniRepLKNet CVPR 2024：`https://openaccess.thecvf.com/content/CVPR2024/html/Ding_UniRepLKNet_A_Universal_Perception_Large-Kernel_ConvNet_for_Audio_Video_Point_CVPR_2024_paper.html`
- CD-RLKNet / IJAEO 2024 DOI：`10.1016/j.jag.2024.104077`
- HAM-CD / IEEE TGRS 2026 DOI：`10.1109/TGRS.2026.3665418`
