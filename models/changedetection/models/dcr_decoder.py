"""DCR decoder: RepLocalBlock + top-down multi-scale DCRDecoder (v2 BN-FR primitives).

Run6 NSCR-Fuse: fuse1/fuse2 become NSCRPairFuse1x1 (native-scale zero-init BN
branches, deploy-absorbed); the upsample of the semantic input moves inside the
fuse wrapper so bn_h sees the NATIVE low resolution. fuse3 unchanged.

Run8 MPCR-Fine: refine.pw can become MPCRPW1x1 (two zero-init grouped-1x1
training branches with complementary channel partitions, absorbed into the
original dense PW at deploy). block1/2/3, refine.dw, fuses stay unchanged.

Run10 PFDR: t1 passes through PFDRDW5 (pre-fusion dilated re-parameterization,
single deploy DW5) BEFORE fuse1. The PFDR module itself is attached by
STRRepNet AFTER all other RNG-consuming constructions (Run9-style) so that the
C0/M1 epoch-0 outputs stay bitwise identical; here we only hold the flags.
"""
import torch.nn as nn
import torch.nn.functional as F

from changedetection.models.reparam import (
    RepDW3,
    RepPW1x1,
    RepPairFuse1x1,
    NSCRPairFuse1x1,
    MPCRPW1x1,
    PFDRDW5,
    switch_module_to_deploy,
)

NSCR_MODES = {
    "none": "none",
    "high2": "both",
    "lonly": "lonly",
    "honly": "honly",
}


class RepLocalBlock(nn.Module):
    """Local refinement block: DW3 (residual) -> SiLU -> PW1 (residual) -> SiLU.

    use_mpcr: PW becomes MPCRPW1x1 (Run8; refine.pw only) instead of RepPW1x1.
    """

    def __init__(self, dim, use_aux=True, use_residual=True, deploy=False,
                 use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4):
        super().__init__()
        self.dim = dim
        self.dw = RepDW3(dim, use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        if use_mpcr:
            self.pw = MPCRPW1x1(dim, groups=mpcr_groups, mode=mpcr_mode,
                                use_aux=use_aux, use_residual=use_residual, deploy=deploy)
        else:
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

    def __init__(self, dim=160, use_aux=True, use_residual=True, nscr_scope="none",
                 deploy=False, use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4,
                 use_pfdr=False, pfdr_mode="rep", pfdr_scope="fine1"):
        super().__init__()
        self.dim = dim
        self.use_aux = use_aux
        self.nscr_scope = nscr_scope
        self.use_mpcr = use_mpcr
        self.act = nn.SiLU()
        mode = NSCR_MODES.get(nscr_scope, "none")

        # Run10 PFDR flags; the PFDRDW5 module itself is attached by STRRepNet
        # AFTER all other RNG-consuming constructions (see STRRepNet).
        self.use_pfdr = use_pfdr
        self.pfdr_mode = pfdr_mode
        self.pfdr_scope = pfdr_scope
        self.prefuse1 = None

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

        # Run8 MPCR-Fine: only the final refine block gets the MPCR pointwise.
        self.refine = RepLocalBlock(dim, use_aux=use_aux, use_residual=use_residual,
                                    deploy=deploy, use_mpcr=use_mpcr,
                                    mpcr_mode=mpcr_mode, mpcr_groups=mpcr_groups)

    def forward(self, feats):
        t1, t2, t3, t4 = feats

        # Run10 PFDR: spatial conditioning of the fine lateral BEFORE cross-scale
        # semantic mixing (doc §3.1; pfdr_scope="fine1" only, before fuse1).
        if self.use_pfdr:
            assert self.prefuse1 is not None, "PFDR module not attached (STRRepNet)"
            t1 = self.prefuse1(t1)

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
