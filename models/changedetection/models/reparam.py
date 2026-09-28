"""Structural re-parameterization primitives (BN-FR: branch-normalized foldable residual).

v2 changes vs Run1:
  - per-branch BN (each linear branch has its own BatchNorm) instead of one shared BN;
  - foldable clean residual `+ alpha*x` (or `+ alpha*L` for cross-scale fuse), where alpha
    is a learnable scalar absorbed into the deploy kernel centre / diagonal;
  - all kernel/bias composition done in FP64, cast to FP32 once at the end.

Deploy operator graph / Params / FLOPs are unchanged vs Run1: each block still folds to a
single Conv/DWConv (bias=True).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fold_conv_bn(weight, bias, bn):
    """Fold one Conv (weight, bias) + BatchNorm into (W', b').

    Returns float64 tensors: ALL branch composition must stay in float64 and
    cast to FP32 once, at the final deploy-conv assignment (P0 fix, Run5).
    """
    C = weight.shape[0]
    eps = bn.eps if bn.eps is not None else 1e-5
    gamma = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    std = torch.sqrt(var + eps)
    t = gamma / std
    bd = bias.detach().double() if bias is not None else torch.zeros(C, dtype=torch.float64, device=weight.device)
    W = weight.detach().double() * t.view(C, 1, 1, 1)
    b = beta + (bd - mu) * t
    return W, b


def pad_1x3_to_3x3(w):
    return F.pad(w, (0, 0, 1, 1))


def pad_3x1_to_3x3(w):
    return F.pad(w, (1, 1, 0, 0))


def identity_dw_kernel(channels, k=3, device=None, dtype=torch.float32):
    kernel = torch.zeros(channels, 1, k, k, device=device, dtype=dtype)
    kernel[:, 0, k // 2, k // 2] = 1.0
    return kernel


def identity_1x1_kernel(channels, device=None, dtype=torch.float32):
    return torch.eye(channels, device=device, dtype=dtype).view(channels, channels, 1, 1)


class RepDW3(nn.Module):
    """Depthwise Rep 3x3: DW3x3 + DW1x3 + DW3x1 (each with own BN) + alpha*x -> single DW3x3."""

    def __init__(self, channels, use_aux=True, use_residual=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.use_aux = use_aux
        self.use_residual = use_residual
        self.deploy = deploy
        if deploy:
            self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True)
        else:
            self.dw3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
            self.bn3 = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.dw3.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.dw13 = nn.Conv2d(channels, channels, (1, 3), 1, (0, 1), groups=channels, bias=False)
                self.bn13 = nn.BatchNorm2d(channels)
                self.dw31 = nn.Conv2d(channels, channels, (3, 1), 1, (1, 0), groups=channels, bias=False)
                self.bn31 = nn.BatchNorm2d(channels)
                nn.init.zeros_(self.dw13.weight)
                nn.init.zeros_(self.dw31.weight)
            if self.use_residual:
                self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.deploy:
            return self.dw(x)
        y = self.bn3(self.dw3(x))
        if self.use_aux:
            y = y + self.bn13(self.dw13(x)) + self.bn31(self.dw31(x))
        if self.use_residual:
            y = y + self.alpha * x
        return y

    def get_equivalent_kernel_bias(self):
        W3, b3 = fold_conv_bn(self.dw3.weight, None, self.bn3)
        k = W3
        b = b3
        if self.use_aux:
            W13, b13 = fold_conv_bn(self.dw13.weight, None, self.bn13)
            W31, b31 = fold_conv_bn(self.dw31.weight, None, self.bn31)
            k = k + pad_1x3_to_3x3(W13) + pad_3x1_to_3x3(W31)
            b = b + b13 + b31
        if self.use_residual:
            k = k + self.alpha.detach().double() * identity_dw_kernel(
                self.channels, 3, device=k.device, dtype=torch.float64)
        return k, b

    def branch_stats(self):
        s = {"dw3": self.dw3.weight.norm().item()}
        if self.use_aux:
            s["dw13"] = self.dw13.weight.norm().item()
            s["dw31"] = self.dw31.weight.norm().item()
        if self.use_residual:
            s["alpha"] = self.alpha.item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dw = nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels, bias=True)
        self.dw.weight.data = kernel.float()
        self.dw.bias.data = bias.float()
        self.deploy = True
        for name in ("dw13", "dw31", "bn3", "bn13", "bn31", "dw3", "alpha"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class RepPW1x1(nn.Module):
    """Pointwise Rep 1x1: main + serial low-rank + diag (each BN) + alpha*x -> single 1x1."""

    def __init__(self, channels, r=None, use_aux=True, use_residual=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.r = r if r is not None else max(1, channels // 4)
        self.use_aux = use_aux
        self.use_residual = use_residual
        self.deploy = deploy
        if deploy:
            self.pw = nn.Conv2d(channels, channels, 1, bias=True)
        else:
            self.pw_main = nn.Conv2d(channels, channels, 1, bias=False)
            self.bn_main = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.pw_main.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.pw1 = nn.Conv2d(channels, self.r, 1, bias=False)
                self.pw2 = nn.Conv2d(self.r, channels, 1, bias=False)
                self.bn_lr = nn.BatchNorm2d(channels)
                self.diag = nn.Parameter(torch.zeros(channels))
                nn.init.kaiming_normal_(self.pw1.weight, mode="fan_out", nonlinearity="relu")
                nn.init.normal_(self.pw2.weight, mean=0.0, std=1e-3)
            if self.use_residual:
                self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.deploy:
            return self.pw(x)
        y = self.bn_main(self.pw_main(x))
        if self.use_aux:
            y = y + self.bn_lr(self.pw2(self.pw1(x))) + self.diag.view(1, -1, 1, 1) * x
        if self.use_residual:
            y = y + self.alpha * x
        return y

    def get_equivalent_kernel_bias(self):
        C = self.channels
        W, b = fold_conv_bn(self.pw_main.weight, None, self.bn_main)
        if self.use_aux:
            W2 = self.pw2.weight[:, :, 0, 0].double()  # (C, r)
            W1 = self.pw1.weight[:, :, 0, 0].double()  # (r, C)
            W_serial = (W2 @ W1).view(C, C, 1, 1)      # float64 compose
            W_lr, b_lr = fold_conv_bn(W_serial, None, self.bn_lr)
            D = torch.diag(self.diag.detach().double()).view(C, C, 1, 1)
            W = W + W_lr + D
            b = b + b_lr
        if self.use_residual:
            W = W + self.alpha.detach().double() * identity_1x1_kernel(
                C, device=W.device, dtype=torch.float64)
        return W, b

    def branch_stats(self):
        s = {"main": self.pw_main.weight.norm().item()}
        if self.use_aux:
            s["pw1"] = self.pw1.weight.norm().item()
            s["pw2"] = self.pw2.weight.norm().item()
        if self.use_residual:
            s["alpha"] = self.alpha.item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.pw = nn.Conv2d(self.channels, self.channels, 1, bias=True)
        self.pw.weight.data = kernel.float()
        self.pw.bias.data = bias.float()
        self.deploy = True
        for name in ("pw1", "pw2", "diag", "bn_main", "bn_lr", "pw_main", "alpha"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class RepPairFuse1x1(nn.Module):
    """Cross-scale fusion Rep 1x1: concat + sum + diff (each BN) + alpha*L -> single 1x1."""

    def __init__(self, channels, use_aux=True, use_residual=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.use_aux = use_aux
        self.use_residual = use_residual
        self.deploy = deploy
        if deploy:
            self.fuse = nn.Conv2d(2 * channels, channels, 1, bias=True)
        else:
            self.fuse_c = nn.Conv2d(2 * channels, channels, 1, bias=False)
            self.bn_c = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.fuse_c.weight, mode="fan_out", nonlinearity="relu")
            if self.use_aux:
                self.fuse_s = nn.Conv2d(channels, channels, 1, bias=False)
                self.bn_s = nn.BatchNorm2d(channels)
                self.fuse_d = nn.Conv2d(channels, channels, 1, bias=False)
                self.bn_d = nn.BatchNorm2d(channels)
                nn.init.zeros_(self.fuse_s.weight)
                nn.init.zeros_(self.fuse_d.weight)
            if self.use_residual:
                self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, L, H):
        if self.deploy:
            return self.fuse(torch.cat([L, H], dim=1))
        y = self.bn_c(self.fuse_c(torch.cat([L, H], dim=1)))
        if self.use_aux:
            y = y + self.bn_s(self.fuse_s(L + H)) + self.bn_d(self.fuse_d(H - L))
        if self.use_residual:
            y = y + self.alpha * L
        return y

    def get_equivalent_kernel_bias(self):
        C = self.channels
        W_c, b_c = fold_conv_bn(self.fuse_c.weight, None, self.bn_c)
        W_cL = W_c[:, :C]
        W_cH = W_c[:, C:]
        if self.use_aux:
            W_s, b_s = fold_conv_bn(self.fuse_s.weight, None, self.bn_s)
            W_d, b_d = fold_conv_bn(self.fuse_d.weight, None, self.bn_d)
            W_L = W_cL + W_s - W_d
            W_H = W_cH + W_s + W_d
            b = b_c + b_s + b_d
        else:
            W_L, W_H = W_cL, W_cH
            b = b_c
        if self.use_residual:
            W_L = W_L + self.alpha.detach().double() * identity_1x1_kernel(
                C, device=W_L.device, dtype=torch.float64)
        W = torch.cat([W_L, W_H], dim=1)
        return W, b

    def branch_stats(self):
        s = {"concat": self.fuse_c.weight.norm().item()}
        if self.use_aux:
            s["sum"] = self.fuse_s.weight.norm().item()
            s["diff"] = self.fuse_d.weight.norm().item()
        if self.use_residual:
            s["alpha"] = self.alpha.item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fuse = nn.Conv2d(2 * self.channels, self.channels, 1, bias=True)
        self.fuse.weight.data = kernel.float()
        self.fuse.bias.data = bias.float()
        self.deploy = True
        for name in ("fuse_s", "fuse_d", "bn_c", "bn_s", "bn_d", "fuse_c", "alpha"):
            if hasattr(self, name):
                delattr(self, name)
        return self


def switch_module_to_deploy(module):
    for m in module.modules():
        if m is not module and hasattr(m, "switch_to_deploy"):
            m.switch_to_deploy()
    return module


def fold_identity_bn(bn, channels):
    """Fold an eval-mode BN on an identity path into (A, c), both FP64.

    A: (channels, channels, 1, 1) diagonal 1x1 kernel (gamma/sqrt(var+eps));
    c: (channels,) bias (beta - A*mu).
    Used by NSCR-Fuse (Run6).
    """
    eps = bn.eps if bn.eps is not None else 1e-5
    gamma = bn.weight.detach().double()
    beta = bn.bias.detach().double()
    mu = bn.running_mean.detach().double()
    var = bn.running_var.detach().double()
    t = gamma / torch.sqrt(var + eps)
    A = torch.diag(t).view(channels, channels, 1, 1)
    c = beta - t * mu
    return A, c


class NSCRPairFuse1x1(nn.Module):
    """Run6 NSCR-Fuse: native-scale commutative re-parameterized cross-scale fusion.

    Wraps a RepPairFuse1x1 core. Training adds zero-init native-scale BN branches:
        y = core(L, U(H)) + BN_L(L) + U(BN_H(H))
    where U is bilinear interpolation (H is received at its NATIVE low resolution
    and upsampled inside this module). At deploy, the affine-interpolation
    commutation U(A H + c) = A U(H) + c absorbs both branches into the core
    1x1's L/H input halves, so the deploy graph stays one bilinear + one 1x1
    (+0 params/FLOPs). All composition stays FP64; one FP32 cast at the end.

    nscr_mode: "none" | "both" | "lonly" | "honly".
    """

    def __init__(self, channels, use_aux=True, use_residual=True, nscr_mode="none", deploy=False):
        super().__init__()
        self.channels = channels
        self.use_aux = use_aux
        self.use_residual = use_residual
        self.nscr_mode = nscr_mode
        self.deploy = deploy
        if deploy:
            self.fused = nn.Conv2d(2 * channels, channels, 1, bias=True)
        else:
            self.core = RepPairFuse1x1(channels, use_aux=use_aux, use_residual=use_residual, deploy=False)
            if nscr_mode in ("both", "lonly"):
                self.bn_l = nn.BatchNorm2d(channels)
                nn.init.zeros_(self.bn_l.weight)
                nn.init.zeros_(self.bn_l.bias)
            if nscr_mode in ("both", "honly"):
                self.bn_h = nn.BatchNorm2d(channels)
                nn.init.zeros_(self.bn_h.weight)
                nn.init.zeros_(self.bn_h.bias)

    def forward(self, L, H):
        up_h = F.interpolate(H, size=L.shape[-2:], mode="bilinear", align_corners=False)
        if self.deploy:
            return self.fused(torch.cat([L, up_h], dim=1))
        y = self.core(L, up_h)
        if self.nscr_mode in ("both", "lonly"):
            y = y + self.bn_l(L)
        if self.nscr_mode in ("both", "honly"):
            y = y + F.interpolate(self.bn_h(H), size=L.shape[-2:], mode="bilinear", align_corners=False)
        return y

    def get_equivalent_kernel_bias(self):
        # FP64 composition. NOTE: never call core.switch_to_deploy() here — that
        # would cast FP32 early and break the P0 one-time-cast rule.
        C = self.channels
        W_core, b_core = self.core.get_equivalent_kernel_bias()  # (C, 2C, 1, 1) FP64
        W_L = W_core[:, :C]
        W_H = W_core[:, C:]
        b = b_core
        if self.nscr_mode in ("both", "lonly"):
            A_l, c_l = fold_identity_bn(self.bn_l, C)
            W_L = W_L + A_l
            b = b + c_l
        if self.nscr_mode in ("both", "honly"):
            A_h, c_h = fold_identity_bn(self.bn_h, C)
            W_H = W_H + A_h
            b = b + c_h
        return torch.cat([W_L, W_H], dim=1), b

    def branch_stats(self):
        s = {"core": self.core.branch_stats()}
        if self.nscr_mode in ("both", "lonly"):
            s["nscr_l_gamma"] = self.bn_l.weight.norm().item()
        if self.nscr_mode in ("both", "honly"):
            s["nscr_h_gamma"] = self.bn_h.weight.norm().item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused = nn.Conv2d(2 * self.channels, self.channels, 1, bias=True)
        self.fused.weight.data = kernel.float()
        self.fused.bias.data = bias.float()
        self.deploy = True
        for name in ("core", "bn_l", "bn_h"):
            if hasattr(self, name):
                delattr(self, name)
        return self
