# 轻量化遥感二值变化检测中的重参数化研究方案调研与最终方向

## 1. 研究目标

任务：全监督遥感双时相二值变化检测（LEVIR-CD-256、SYSU-CD-256、WHU-CD-256、CDD-CD-256）。

目标：

-   参数量保持极低；
-   推理保持实时友好；
-   通过训练阶段多分支增强表达能力；
-   通过结构重参数化在部署阶段折叠为单路径。

当前实验平台： - RTX 5090 ×2； - 四个数据集统一采用 A/B/label 格式； -
mask 阈值统一为 gray \>=128。

数据规范来自项目环境文档：四个数据集分别为
CDD、LEVIR、SYSU、WHU，并已经完成 split 审计；DataLoader 使用 A/B/label
和 list 文件规范。fileciteturn0file1L40-L57

------------------------------------------------------------------------

# 2. Baseline 选择

## 2.1 不建议继续使用纯 MobileNetV2 baseline

原因：

MobileNetV2 + 简单 decoder 已经大量使用，创新空间主要集中在 decoder。

如果目标是 2025-2026 论文级工作，需要一个更新、更强的 encoder-decoder
框架。

------------------------------------------------------------------------

# 2.2 推荐 Baseline

## Baseline-A：SCAM（2025 JSTARS）

论文：

Scan Channel Attention Mamba-Based Network for Remote Sensing Change
Detection

优势：

1.  2025 IEEE JSTARS；
2.  使用 Mamba 建模长程依赖；
3.  在 LEVIR-CD、WHU-CD、SYSU-CD 等公开数据集验证；
4.  包含轻量 CNN decoder。

SCAM 报告在： - LEVIR-CD F1=91.01% - WHU-CD F1=93.14% - SYSU-CD
F1=83.60%

其结构适合作为改造对象： - encoder 保留 Mamba/CNN 特征提取； - decoder
替换为 Rep-Mamba Hybrid Decoder。

------------------------------------------------------------------------

## Baseline-B：ST-Mamba

2025 TGRS：

Spatio-Temporal Mamba for Remote Sensing Change Detection

特点：

-   专门处理双时相关系；
-   强调时空联合建模；
-   适合加入 temporal re-parameterization。

缺点：

Mamba模块较重，不适合极低参数目标。

------------------------------------------------------------------------

## Baseline-C：轻量 CNN baseline

如果最终目标 \<5M 参数：

推荐：

MobileNetV3 / EfficientNet-Lite / LWGANet 作为 encoder。

原因：

遥感变化检测中 decoder 和 fusion 往往比 backbone 更决定性能。

------------------------------------------------------------------------

# 最终推荐

采用：

## SCAM-lite + Rep-Temporal-Spatial Decoder

理由：

-   新；
-   有Mamba全局建模；
-   decoder存在重构空间；
-   不重复 ChangeMamba；
-   可以形成"轻量Mamba + 结构重参数化"的独立贡献。

------------------------------------------------------------------------

# 3. 重参数化技术调研

## 3.1 RepVGG思想

核心：

训练：

多分支：

3×3 Conv + 1×1 Conv + Identity

推理：

融合：

单个3×3 Conv。

意义：

解决训练表达能力和部署效率之间矛盾。

------------------------------------------------------------------------

# 3.2 DBB

Diverse Branch Block

CVPR 2021。

虽然早于要求年份，但是2024-2026相关工作大量继承其思想。

核心：

训练：

-   大核；
-   小核；
-   非对称卷积；
-   pooling；

推理：

融合成一个卷积。

迁移：

适合变化检测 decoder。

------------------------------------------------------------------------

# 3.3 RepLKNet思想

核心：

大核卷积重参数化。

训练：

多个小卷积辅助学习。

部署：

大核DWConv。

遥感变化检测优势：

-   建筑变化需要大感受野；
-   道路、建筑边缘需要方向信息。

已有变化检测工作 CD-RLKNet 将大核重参数化用于遥感变化检测。

------------------------------------------------------------------------

# 3.4 Rep-Mamba Hybrid Block

建议作为本文核心。

设计：

训练阶段：

                 Feature
                    |
          -----------------------
          |          |          |
       RepConv   Diff Branch   Mamba
          |          |          |
          -----------------------
                    |
              Fusion

三个分支：

## Branch A：Local RepConv

作用：

-   边界；
-   小目标；
-   高频纹理。

## Branch C：Difference Branch

输入：

\|F1-F2\|

作用：

直接强化变化区域。

## Branch B：Lightweight Mamba

作用：

-   长程一致性；
-   大范围变化。

部署：

A+C：

折叠。

B：

保留一个轻量SSM。

最终：

    Input

     ↓

    1×1 Conv

     ↓

    DW RepKernel

     ↓

    Light Mamba

     ↓

    Output

------------------------------------------------------------------------

# 4. 时空结构重参数化方案

参考已有设计：

## Temporal Rep

输入：

X1,X2

训练：

三个信息流：

1.  Concat

Yc = Conv(\[X1,X2\])

2.  Sum

Ys = Conv(X1+X2)

3.  Diff

Yd = Conv(X2-X1)

融合：

Y=Yc+Ys+Yd

部署：

折叠：

Wdeploy=

Wc+\[Ws-Wd, Ws+Wd\]

因此：

三路时相拓扑

↓

一个1×1 Conv。

------------------------------------------------------------------------

## Spatial Rep

训练：

    DW5×5
    DW3×3
    Dilated DW3×3
    DW1×5
    DW5×1
    Identity

部署：

统一融合：

DW5×5

优势：

同时获得：

-   大尺度上下文；
-   局部边界；
-   方向结构。

------------------------------------------------------------------------

# 5. 不建议直接使用 RepViT

RepViT（CVPR 2024）：

优势：

-   CNN+ViT思想；
-   移动端友好。

但是：

对于变化检测：

问题：

1.  backbone已经不是瓶颈；
2.  双时相交互需要专门设计；
3.  替换encoder创新风险大。

因此：

不作为第一选择。

------------------------------------------------------------------------

# 6. 最终唯一研究方向

## 名称：

Rep-Mamba Spatio-Temporal Re-parameterized Decoder for Lightweight
Remote Sensing Change Detection

中文：

轻量遥感变化检测的时空重参数化Mamba解码网络

------------------------------------------------------------------------

# 7. 网络结构

## Encoder

采用：

SCAM-lite / MobileNetV3 / EfficientNet-lite

输出：

E1-E5。

------------------------------------------------------------------------

## Decoder核心

每一级decoder加入：

## STR-Mamba Block

Spatial-Temporal Re-parameterized Mamba Block

训练：

            Bi-temporal Features

                  |
           Temporal Rep
        /      |        \
    Concat   Sum      Diff

                  |
           Spatial Rep
     /  /  /  /  /  /
    5x5 3x3 Dil AC Id

                  |

          Lightweight Mamba

                  |

            Decoder Output

部署：

    Concat

     |

    1×1 Conv

     |

    DW5×5

     |

    Mamba

     |

    1×1 Conv

------------------------------------------------------------------------

# 8. Boundary增强

训练阶段：

使用GT生成boundary：

Boundary = Dilate(mask)-Erode(mask)

增加：

Boundary Auxiliary Loss

Loss:

L=

BCE

-   

Dice

-   

λ Boundary Loss

作用：

提高：

-   建筑边缘；
-   小变化；
-   细长目标。

------------------------------------------------------------------------

# 9. 实验设计

必须包含：

## Baseline

原模型。

## Ablation

1.  

baseline

2.  

+Temporal Rep

3.  

+Spatial Rep

4.  

+Mamba

5.  

+Boundary supervision

6.  

Full

------------------------------------------------------------------------

# 10. 预期贡献

贡献1：

提出双时相拓扑重参数化。

贡献2：

提出空间异构卷积重参数化decoder。

贡献3：

提出Rep-Mamba Hybrid Block。

贡献4：

实现训练复杂、部署简单的轻量变化检测网络。

------------------------------------------------------------------------

# 11. 最终建议

不要重构 backbone。

重点：

Encoder保持成熟预训练网络。

创新全部集中：

-   decoder；
-   temporal fusion；
-   residual bridge；
-   boundary branch。

最终路线：

SCAM-lite encoder

-   

Temporal Rep

-   

Spatial RepLK

-   

Lightweight Mamba

-   

Boundary supervision

这是当前最符合： "轻量化 + SOTA + 可发表 + 可实现" 的方向。
