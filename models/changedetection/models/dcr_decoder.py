"""DCR decoder: RepLocalBlock + top-down multi-scale DCRDecoder (v2 BN-FR primitives).

Run6 NSCR-Fuse: fuse1/fuse2 become NSCRPairFuse1x1 (native-scale zero-init BN
branches, deploy-absorbed); the upsample of the semantic input moves inside the
fuse wrapper so bn_h sees the NATIVE low resolution. fuse3 unchanged.
"""
import torch.nn as nn
import torch.nn.functional as F

from changedetection.models.reparam import (
    RepDW3,
    RepPW1x1,
    RepPairFuse1x1,
    NSCRPairFuse1x1,
    switch_module_to_deploy,
)

NSCR_MODES = {
    "none": "none",
    "high2": "both",
    "lonly": "lonly",
    "honly": "honly",
}


class RepLocalBlock(nn.Module):
    """Local refinement block: DW3 (residual) -> SiLU -> PW1 (residual) -> SiLU."""

    def __init__(self, dim, use_aux=True, use_residual=True, deploy=False):
        super().__init__()
        self.dim = dim
        self.dw = RepDW3(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        self.pw = RepPW1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
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
    """Top-down multi-scale decoder over TAR features [t1,t2,t3,t4] (dim ch).

    nscr_scope in {"none", "high2", "lonly", "honly"}: applies the NSCR native-scale
    branches to fuse1+fuse2 (high2 = both L and H branches).
    """

    def __init__(self, dim=160, use_aux=True, use_residual=True, nscr_scope="none", deploy=False):
        super().__init__()
        self.dim = dim
        self.use_aux = use_aux
        self.nscr_scope = nscr_scope
        self.act = nn.SiLU()
        mode = NSCR_MODES.get(nscr_scope, "none")

        self.fuse3 = RepPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        self.block3 = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)

        # nscr_scope="none" keeps the EXACT Run2 structure (state_dict keys and
        # forward identical), so old checkpoints stay loadable.
        if mode == "none":
            self.fuse2 = RepPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
            self.fuse1 = RepPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        else:
            self.fuse2 = NSCRPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual,
                                         nscr_mode=mode, deploy=deploy)
            self.fuse1 = NSCRPairFuse1x1(dim, use_aux=use_aux, use_residual=use_residual,
                                         nscr_mode=mode, deploy=deploy)
        self.block2 = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        self.block1 = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)

        self.refine = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)

    def forward(self, feats):
        t1, t2, t3, t4 = feats

        d4 = t4

        u4 = F.interpolate(d4, size=t3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.block3(self.act(self.fuse3(t3, u4)))

        if self.nscr_scope == "none":
            # Run2-identical path: upsample outside the fuse
            u3 = F.interpolate(d3, size=t2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.block2(self.act(self.fuse2(t2, u3)))
            u2 = F.interpolate(d2, size=t1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.block1(self.act(self.fuse1(t1, u2)))
        else:
            # NSCR path: fuse receives the semantic input at its NATIVE resolution;
            # the upsample happens inside the (NSCR) fuse so bn_h sees native H.
            d2 = self.block2(self.act(self.fuse2(t2, d3)))
            d1 = self.block1(self.act(self.fuse1(t1, d2)))

        d1 = self.refine(d1)
        return d1

    def switch_to_deploy(self):
        switch_module_to_deploy(self)
        return self
