"""DCR decoder: RepLocalBlock + top-down multi-scale DCRDecoder.

The decoder contains no SSM / attention / SE. Global modeling is done by the
frozen VMamba encoder; this decoder only does temporal change extraction,
local refinement and cross-scale fusion, all as foldable Rep operators.
"""
import torch.nn as nn
import torch.nn.functional as F

from changedetection.models.reparam import (
    RepDW3,
    RepPW1x1,
    RepPairFuse1x1,
    switch_module_to_deploy,
)


class RepLocalBlock(nn.Module):
    """Local refinement block: DW3 (identity) -> SiLU -> PW1 -> SiLU."""

    def __init__(self, dim, use_aux=True, deploy=False):
        super().__init__()
        self.dim = dim
        self.dw = RepDW3(dim, include_identity=True, use_aux=use_aux, deploy=deploy)
        self.pw = RepPW1x1(dim, use_aux=use_aux, deploy=deploy)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.dw(x))
        x = self.act(self.pw(x))
        return x

    def switch_to_deploy(self):
        self.dw.switch_to_deploy()
        self.pw.switch_to_deploy()
        return self


class DCRDecoder(nn.Module):
    """Top-down multi-scale decoder over TAR features [t1,t2,t3,t4] (160ch)."""

    def __init__(self, dim=160, use_aux=True, deploy=False):
        super().__init__()
        self.dim = dim
        self.use_aux = use_aux
        self.act = nn.SiLU()

        self.fuse3 = RepPairFuse1x1(dim, use_aux=use_aux, deploy=deploy)
        self.block3 = RepLocalBlock(dim, use_aux=use_aux, deploy=deploy)

        self.fuse2 = RepPairFuse1x1(dim, use_aux=use_aux, deploy=deploy)
        self.block2 = RepLocalBlock(dim, use_aux=use_aux, deploy=deploy)

        self.fuse1 = RepPairFuse1x1(dim, use_aux=use_aux, deploy=deploy)
        self.block1 = RepLocalBlock(dim, use_aux=use_aux, deploy=deploy)

        self.refine = RepLocalBlock(dim, use_aux=use_aux, deploy=deploy)

    def forward(self, feats):
        t1, t2, t3, t4 = feats

        d4 = t4

        u4 = F.interpolate(d4, size=t3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.block3(self.act(self.fuse3(t3, u4)))

        u3 = F.interpolate(d3, size=t2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.block2(self.act(self.fuse2(t2, u3)))

        u2 = F.interpolate(d2, size=t1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.block1(self.act(self.fuse1(t1, u2)))

        d1 = self.refine(d1)
        return d1

    def switch_to_deploy(self):
        switch_module_to_deploy(self)
        return self
