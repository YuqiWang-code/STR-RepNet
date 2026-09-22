"""Structural re-parameterization primitives for TAR-DCR.

Every Rep module below obeys the same contract:
  - train graph = multiple linear branches + one shared BN
  - deploy graph = a single Conv/DWConv with the shared BN folded in
  - get_equivalent_kernel_bias() returns the folded (kernel, bias)
  - switch_to_deploy() replaces the branches by the folded conv and drops them

All branches share a single BatchNorm (see implementation doc §10), so the fold
is: combine kernels -> fold the shared BN -> single bias=True conv.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_bn_into_kernel(kernel, bn, bias=None):
    """Fold a BatchNorm2d into a preceding conv kernel.

    kernel: (C_out, C_in, kH, kW).  Returns (folded_kernel, folded_bias).
    Computed in float64 internally to minimize FP32 accumulation error.
    """
    C = kernel.shape[0]
    if bias is None:
        bias = torch.zeros(C, device=kernel.device, dtype=torch.float64)
    eps = bn.eps if bn.eps is not None else 1e-5
    gamma = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    std = torch.sqrt(var + eps)
    t = gamma / std
    kd = kernel.double()
    bd = bias.double()
    folded_kernel = (kd * t.view(C, 1, 1, 1)).float()
    folded_bias = (beta + (bd - mu) * t).float()
    return folded_kernel, folded_bias


def pad_1x3_to_3x3(w):
    """(C, 1, 1, 3) -> (C, 1, 3, 3), centered on the middle row."""
    return F.pad(w, (0, 0, 1, 1))


def pad_3x1_to_3x3(w):
    """(C, 1, 3, 1) -> (C, 1, 3, 3), centered on the middle column."""
    return F.pad(w, (1, 1, 0, 0))


def identity_dw_kernel(channels, k=3, device=None, dtype=None):
    """Depthwise identity kernel: (channels, 1, k, k) with 1 at the center."""
    kernel = torch.zeros(channels, 1, k, k, device=device, dtype=dtype)
    kernel[:, 0, k // 2, k // 2] = 1.0
    return kernel


class RepDW3(nn.Module):
    """Depthwise Rep 3x3: DW3x3 + DW1x3 + DW3x1 (+ identity) -> single DW3x3."""

    def __init__(self, channels, include_identity=False, use_aux=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.include_identity = include_identity
        self.use_aux = use_aux
        self.deploy = deploy
        if deploy:
            self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True)
        else:
            self.dw3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            self.bn = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.dw3.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.dw13 = nn.Conv2d(channels, channels, (1, 3), 1, (0, 1), groups=channels, bias=False)
                self.dw31 = nn.Conv2d(channels, channels, (3, 1), 1, (1, 0), groups=channels, bias=False)
                nn.init.zeros_(self.dw13.weight)
                nn.init.zeros_(self.dw31.weight)

    def forward(self, x):
        if self.deploy:
            return self.dw(x)
        y = self.dw3(x)
        if self.use_aux:
            y = y + self.dw13(x) + self.dw31(x)
        if self.include_identity:
            y = y + x
        return self.bn(y)

    def get_equivalent_kernel_bias(self):
        k = self.dw3.weight.double()  # (C, 1, 3, 3)
        if self.use_aux:
            k = k + pad_1x3_to_3x3(self.dw13.weight).double() + pad_3x1_to_3x3(self.dw31.weight).double()
        if self.include_identity:
            k = k + identity_dw_kernel(self.channels, 3, device=k.device, dtype=torch.float64)
        return fold_bn_into_kernel(k, self.bn)

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dw = nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels, bias=True)
        self.dw.weight.data = kernel
        self.dw.bias.data = bias
        self.deploy = True
        if hasattr(self, "dw13"):
            del self.dw13
        if hasattr(self, "dw31"):
            del self.dw31
        del self.dw3, self.bn
        return self


class RepPW1x1(nn.Module):
    """Pointwise Rep 1x1: main + serial low-rank (C->r->C) + diagonal scale -> single 1x1."""

    def __init__(self, channels, r=None, use_aux=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.r = r if r is not None else max(1, channels // 4)
        self.use_aux = use_aux
        self.deploy = deploy
        if deploy:
            self.pw = nn.Conv2d(channels, channels, 1, bias=True)
        else:
            self.pw_main = nn.Conv2d(channels, channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.pw_main.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.pw1 = nn.Conv2d(channels, self.r, 1, bias=False)
                self.pw2 = nn.Conv2d(self.r, channels, 1, bias=False)
                self.diag = nn.Parameter(torch.zeros(channels))
                nn.init.kaiming_normal_(self.pw1.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(self.pw2.weight)

    def forward(self, x):
        if self.deploy:
            return self.pw(x)
        y = self.pw_main(x)
        if self.use_aux:
            y = y + self.pw2(self.pw1(x)) + self.diag.view(1, -1, 1, 1) * x
        return self.bn(y)

    def get_equivalent_kernel_bias(self):
        C = self.channels
        W = self.pw_main.weight.double()  # (C, C, 1, 1)
        if self.use_aux:
            W2 = self.pw2.weight[:, :, 0, 0].double()  # (C, r)
            W1 = self.pw1.weight[:, :, 0, 0].double()  # (r, C)
            W_serial = (W2 @ W1).view(C, C, 1, 1)
            D = torch.diag(self.diag).double().view(C, C, 1, 1)
            W = W + W_serial + D
        return fold_bn_into_kernel(W, self.bn)

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.pw = nn.Conv2d(self.channels, self.channels, 1, bias=True)
        self.pw.weight.data = kernel
        self.pw.bias.data = bias
        self.deploy = True
        if hasattr(self, "pw1"):
            del self.pw1
        if hasattr(self, "pw2"):
            del self.pw2
        if hasattr(self, "diag"):
            del self.diag
        del self.pw_main, self.bn
        return self


class RepPairFuse1x1(nn.Module):
    """Cross-scale fusion Rep 1x1: concat + sum + signed-diff -> single 1x1.

    forward(L, H): L = current-scale feature, H = upsampled coarser feature.
    Deploy: cat([L, H]) -> single Conv1x1(2C -> C) (+ SiLU applied by caller).
    """

    def __init__(self, channels, use_aux=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.use_aux = use_aux
        self.deploy = deploy
        if deploy:
            self.fuse = nn.Conv2d(2 * channels, channels, 1, bias=True)
        else:
            self.fuse_c = nn.Conv2d(2 * channels, channels, 1, bias=False)
            self.bn = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.fuse_c.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.fuse_s = nn.Conv2d(channels, channels, 1, bias=False)
                self.fuse_d = nn.Conv2d(channels, channels, 1, bias=False)
                nn.init.zeros_(self.fuse_s.weight)
                nn.init.zeros_(self.fuse_d.weight)

    def forward(self, L, H):
        if self.deploy:
            return self.fuse(torch.cat([L, H], dim=1))
        y = self.fuse_c(torch.cat([L, H], dim=1))
        if self.use_aux:
            y = y + self.fuse_s(L + H) + self.fuse_d(H - L)
        return self.bn(y)

    def get_equivalent_kernel_bias(self):
        C = self.channels
        W_c = self.fuse_c.weight.double()  # (C, 2C, 1, 1)
        W_cL = W_c[:, :C]
        W_cH = W_c[:, C:]
        if self.use_aux:
            W_s = self.fuse_s.weight.double()
            W_d = self.fuse_d.weight.double()
            W_L = W_cL + W_s - W_d
            W_H = W_cH + W_s + W_d
        else:
            W_L, W_H = W_cL, W_cH
        W = torch.cat([W_L, W_H], dim=1)
        return fold_bn_into_kernel(W, self.bn)

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fuse = nn.Conv2d(2 * self.channels, self.channels, 1, bias=True)
        self.fuse.weight.data = kernel
        self.fuse.bias.data = bias
        self.deploy = True
        if hasattr(self, "fuse_s"):
            del self.fuse_s
        if hasattr(self, "fuse_d"):
            del self.fuse_d
        del self.fuse_c, self.bn
        return self


def switch_module_to_deploy(module):
    """Recursively call switch_to_deploy() on every child that defines it."""
    for m in module.modules():
        if m is not module and hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
    return module
