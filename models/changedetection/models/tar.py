"""TAR: Temporal Algebraic Re-parameterization bridge (v2: per-branch BN).

encoder -> decoder bi-temporal bridge: each scale projects (pre, post) with
Concat + Sum + signed-Diff branches, each with its own BatchNorm, collapsing to
a single 1x1 temporal projection at deploy.
"""
import torch
import torch.nn as nn

from changedetection.models.reparam import fold_conv_bn, switch_module_to_deploy
from changedetection.models.dcr_decoder import RepLocalBlock


class TemporalRep1x1(nn.Module):
    """Bi-temporal 1x1 re-parameterization: concat + sum + signed-diff (each BN) -> single 1x1.

    diff branch uses Q - P (signed difference, NOT abs).
    """

    def __init__(self, in_channels, out_channels, use_aux=True, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_aux = use_aux
        self.deploy = deploy
        if deploy:
            self.proj = nn.Conv2d(2 * in_channels, out_channels, 1, bias=True)
        else:
            self.proj_c = nn.Conv2d(2 * in_channels, out_channels, 1, bias=False)
            self.bn_c = nn.BatchNorm2d(out_channels)
            nn.init.kaiming_normal_(self.proj_c.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.proj_s = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                self.bn_s = nn.BatchNorm2d(out_channels)
                self.proj_d = nn.Conv2d(in_channels, out_channels, 1, bias=False)
                self.bn_d = nn.BatchNorm2d(out_channels)
                nn.init.zeros_(self.proj_s.weight)
                nn.init.zeros_(self.proj_d.weight)

    def forward(self, P, Q):
        if self.deploy:
            return self.proj(torch.cat([P, Q], dim=1))
        y = self.bn_c(self.proj_c(torch.cat([P, Q], dim=1)))
        if self.use_aux:
            y = y + self.bn_s(self.proj_s(P + Q)) + self.bn_d(self.proj_d(Q - P))
        return y

    def get_equivalent_kernel_bias(self):
        C = self.in_channels
        W_c, b_c = fold_conv_bn(self.proj_c.weight, None, self.bn_c)
        W_cP = W_c[:, :C]
        W_cQ = W_c[:, C:]
        if self.use_aux:
            W_s, b_s = fold_conv_bn(self.proj_s.weight, None, self.bn_s)
            W_d, b_d = fold_conv_bn(self.proj_d.weight, None, self.bn_d)
            W_P = W_cP + W_s - W_d
            W_Q = W_cQ + W_s + W_d
            b = b_c + b_s + b_d
        else:
            W_P, W_Q = W_cP, W_cQ
            b = b_c
        W = torch.cat([W_P, W_Q], dim=1)
        return W, b

    def branch_stats(self):
        s = {"concat": self.proj_c.weight.norm().item()}
        if self.use_aux:
            s["sum"] = self.proj_s.weight.norm().item()
            s["diff"] = self.proj_d.weight.norm().item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.proj = nn.Conv2d(2 * self.in_channels, self.out_channels, 1, bias=True)
        self.proj.weight.data = kernel
        self.proj.bias.data = bias
        self.deploy = True
        for name in ("proj_s", "proj_d", "bn_c", "bn_s", "bn_d", "proj_c"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class TARStage(nn.Module):
    """One scale: TemporalRep1x1 -> SiLU -> RepLocalBlock."""

    def __init__(self, in_channels, dim, use_temporal_aux=True, use_dcr_aux=True,
                 use_residual=True, deploy=False):
        super().__init__()
        self.temporal = TemporalRep1x1(in_channels, dim, use_aux=use_temporal_aux, deploy=deploy)
        self.block = RepLocalBlock(dim, use_aux=use_dcr_aux, use_residual=use_residual, deploy=deploy)
        self.act = nn.SiLU()

    def forward(self, P, Q):
        x = self.act(self.temporal(P, Q))
        return self.block(x)

    def switch_to_deploy(self):
        self.temporal.switch_to_deploy()
        self.block.switch_to_deploy()
        return self


class MultiScaleTAR(nn.Module):
    """Four-scale TAR bridge: encoder_dims (96,192,384,768) -> dim."""

    def __init__(self, encoder_dims=(96, 192, 384, 768), dim=160,
                 use_temporal_aux=True, use_dcr_aux=True, use_residual=True, deploy=False):
        super().__init__()
        self.stage1 = TARStage(encoder_dims[0], dim, use_temporal_aux, use_dcr_aux, use_residual, deploy)
        self.stage2 = TARStage(encoder_dims[1], dim, use_temporal_aux, use_dcr_aux, use_residual, deploy)
        self.stage3 = TARStage(encoder_dims[2], dim, use_temporal_aux, use_dcr_aux, use_residual, deploy)
        self.stage4 = TARStage(encoder_dims[3], dim, use_temporal_aux, use_dcr_aux, use_residual, deploy)

    def forward(self, pre_feats, post_feats):
        t1 = self.stage1(pre_feats[0], post_feats[0])
        t2 = self.stage2(pre_feats[1], post_feats[1])
        t3 = self.stage3(pre_feats[2], post_feats[2])
        t4 = self.stage4(pre_feats[3], post_feats[3])
        return [t1, t2, t3, t4]

    def switch_to_deploy(self):
        switch_module_to_deploy(self)
        return self
