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


def build_phase_basis(upscale=4):
    """Fixed low-order 4x4 phase basis (Run7 PBRU): shape (4, r^2), float32.

    Row k is phi_k[p] for p = i*r + j (PyTorch PixelShuffle channel order,
    i = output row phase, j = output col phase):
      phi00[p] = 1
      phi10[p] = u_j = (2j+1-r)/r     (horizontal phase)
      phi01[p] = v_i = (2i+1-r)/r     (vertical phase)
      phi11[p] = u_j * v_i
    Fixed constants: no trainable params, no deploy cost.
    """
    r = int(upscale)
    i = torch.arange(r).view(r, 1).expand(r, r).reshape(-1).float()
    j = torch.arange(r).view(1, r).expand(r, r).reshape(-1).float()
    u = (2.0 * j + 1.0 - r) / r
    v = (2.0 * i + 1.0 - r) / r
    basis = torch.stack([torch.ones_like(u), u, v, u * v], dim=0)  # (4, r^2)
    return basis


class PBRUHead(nn.Module):
    """Run7 PBRU: Phase-Basis Reparameterized Upsampling head.

    Deploy graph:
        X (B,D,H,W) -> single 1x1 (D -> C*r^2) -> PixelShuffle(r) -> (B,C,rH,rW)

    Training graph adds four zero-init (BN gamma=beta=0) phase-basis branches:
        y = main_proj(X) + sum_k E_phi_k( BN_k( branch_k(X) ) )
    where E_phi_k expands the 2-channel branch output to the C*r^2 phase channels
    by the fixed basis phi_k. At deploy the branches are analytically absorbed
    into the single 1x1 (FP64 composition, one FP32 cast); epoch-0 aux output is
    exactly 0, so M1_PBRU starts from the plain PixelShuffle prediction.
    """

    def __init__(self, in_channels, num_classes=2, upscale=4, use_phase_rep=True, deploy=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upscale = int(upscale)
        self.use_phase_rep = use_phase_rep
        self.deploy = deploy
        self.out_channels = num_classes * self.upscale * self.upscale
        if deploy:
            self.proj = nn.Conv2d(in_channels, self.out_channels, 1, bias=True)
        else:
            self.main_proj = nn.Conv2d(in_channels, self.out_channels, 1, bias=True)
            if self.use_phase_rep:
                self.branch_coarse = nn.Conv2d(in_channels, num_classes, 1, bias=False)
                self.bn_coarse = nn.BatchNorm2d(num_classes)
                self.branch_px = nn.Conv2d(in_channels, num_classes, 1, bias=False)
                self.bn_px = nn.BatchNorm2d(num_classes)
                self.branch_py = nn.Conv2d(in_channels, num_classes, 1, bias=False)
                self.bn_py = nn.BatchNorm2d(num_classes)
                self.branch_pxy = nn.Conv2d(in_channels, num_classes, 1, bias=False)
                self.bn_pxy = nn.BatchNorm2d(num_classes)
                for bn in (self.bn_coarse, self.bn_px, self.bn_py, self.bn_pxy):
                    nn.init.zeros_(bn.weight)
                    nn.init.zeros_(bn.bias)
            self.register_buffer("phi", build_phase_basis(self.upscale), persistent=False)

    def forward(self, x):
        if self.deploy:
            return F.pixel_shuffle(self.proj(x), self.upscale)
        z = self.main_proj(x)
        if self.use_phase_rep:
            r2 = self.upscale * self.upscale
            for branch, bn, k in ((self.branch_coarse, self.bn_coarse, 0),
                                  (self.branch_px, self.bn_px, 1),
                                  (self.branch_py, self.bn_py, 2),
                                  (self.branch_pxy, self.bn_pxy, 3)):
                q = bn(branch(x))                                   # (B,C,H,W)
                q = q.unsqueeze(2) * self.phi[k].view(1, 1, r2, 1, 1)  # (B,C,r2,H,W)
                q = q.reshape(q.shape[0], self.out_channels, q.shape[3], q.shape[4])
                z = z + q
        return F.pixel_shuffle(z, self.upscale)

    def get_equivalent_kernel_bias(self):
        """FP64 fold of main + phase-basis branches into (W, b) of shape (C*r^2, D, 1, 1)."""
        C = self.num_classes
        r2 = self.upscale * self.upscale
        W = self.main_proj.weight.detach().double()                 # (C*r2, D, 1, 1)
        b = self.main_proj.bias.detach().double()                   # (C*r2,)
        if self.use_phase_rep:
            phi = self.phi.double()                                 # (4, r2)
            for branch, bn, k in ((self.branch_coarse, self.bn_coarse, 0),
                                  (self.branch_px, self.bn_px, 1),
                                  (self.branch_py, self.bn_py, 2),
                                  (self.branch_pxy, self.bn_pxy, 3)):
                V, bv = fold_conv_bn(branch.weight, None, bn)       # V:(C,D,1,1) bv:(C,)
                # channel n = c*r2 + p (c-major, matches forward reshape):
                Vt = (phi[k].view(1, r2, 1, 1, 1) * V.view(C, 1, V.shape[1], V.shape[2], V.shape[3]))
                W = W + Vt.reshape(C * r2, V.shape[1], V.shape[2], V.shape[3])
                b = b + (phi[k].view(1, r2) * bv.view(C, 1)).reshape(-1)
        return W, b

    def branch_stats(self):
        s = {}
        if self.use_phase_rep:
            s["pbru_coarse_gamma"] = self.bn_coarse.weight.norm().item()
            s["pbru_px_gamma"] = self.bn_px.weight.norm().item()
            s["pbru_py_gamma"] = self.bn_py.weight.norm().item()
            s["pbru_pxy_gamma"] = self.bn_pxy.weight.norm().item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.proj = nn.Conv2d(self.in_channels, self.out_channels, 1, bias=True)
        self.proj.weight.data = kernel.float()
        self.proj.bias.data = bias.float()
        self.deploy = True
        for name in ("main_proj", "branch_coarse", "bn_coarse", "branch_px", "bn_px",
                     "branch_py", "bn_py", "branch_pxy", "bn_pxy"):
            if hasattr(self, name):
                delattr(self, name)
        return self


# -----------------------------------------------------------------------------
# Run8 MPCR-Fine: Multi-Partition Channel Reparameterization (refine.pw only)
# -----------------------------------------------------------------------------
def make_interleaved_permutation(channels, groups):
    """Interleaved partition: perm = arange(D).view(groups, D//groups).t().reshape(-1).

    D=160, g=4 -> [0,40,80,120, 1,41,81,121, ...]  (cross-partition interleaved).
    """
    return torch.arange(channels).view(groups, channels // groups).t().reshape(-1)


def invert_permutation(p):
    inv = torch.empty_like(p)
    inv[p] = torch.arange(len(p), device=p.device)
    return inv


def group1x1_to_dense_fp64(w_group, groups):
    """Embed a grouped-1x1 folded weight (D, D//g, 1, 1) into a dense FP64 matrix
    (D, D, 1, 1): output channel o belongs to group k = o//(D//g) and connects only
    input channels [k*q, (k+1)*q); all other entries are exactly 0."""
    w = w_group.detach().double()
    D = w.shape[0]
    q = w.shape[1]
    assert D % groups == 0 and q == D // groups
    dense = torch.zeros(D, D, 1, 1, dtype=torch.float64, device=w.device)
    k = torch.arange(D, device=w.device) // q            # group id per output channel
    jj = torch.arange(q, device=w.device)
    abs_in = (k.view(D, 1) * q + jj.view(1, q)).reshape(-1)  # (D*q,) input channel ids
    rows = torch.arange(D, device=w.device).view(D, 1).expand(D, q).reshape(-1)
    dense[rows, abs_in, 0, 0] = w.reshape(-1)
    return dense


class MPCRPW1x1(nn.Module):
    """Run8 MPCR-Fine pointwise block (refine.pw only).

    Train graph:
        Y = RepPW1x1(X) + P0^-1 BN0( G0(P0 X) ) + P1^-1 BN1( G1(P1 X) )
    where G0/G1 are grouped 1x1 convs (same params/groups), P0 = identity and
    P1 = identity (mode="same2") or the interleaved permutation (mode="multi2").
    Both branches are zero-init (BN gamma=beta=0) -> epoch-0 output is EXACTLY
    the Run2 core output. At deploy everything is absorbed into the single dense
    1x1 via W_j = P_j^-1 W~_j P_j (FP64 composition, one FP32 cast).

    Deploy graph == Run2 refine.pw: dense Conv2d(D -> D, 1, bias=True), +0 cost.
    """

    def __init__(self, channels, groups=4, mode="multi2", use_aux=True,
                 use_residual=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mode = mode if mode in ("same2", "multi2") else "multi2"
        self.use_aux = use_aux
        self.use_residual = use_residual
        self.deploy = deploy
        assert channels % groups == 0
        if deploy:
            self.pw = nn.Conv2d(channels, channels, 1, bias=True)
        else:
            self.core = RepPW1x1(channels, use_aux=use_aux, use_residual=use_residual)
            self.g0 = nn.Conv2d(channels, channels, 1, groups=groups, bias=False)
            self.bn0 = nn.BatchNorm2d(channels)
            self.g1 = nn.Conv2d(channels, channels, 1, groups=groups, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.g0.weight, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.g1.weight, mode="fan_out", nonlinearity="relu")
            for bn in (self.bn0, self.bn1):
                nn.init.zeros_(bn.weight)
                nn.init.zeros_(bn.bias)
            p0 = torch.arange(channels)
            p1 = p0 if self.mode == "same2" else make_interleaved_permutation(channels, groups)
            self.register_buffer("perm0", p0, persistent=False)
            self.register_buffer("perm1", p1, persistent=False)
            self.register_buffer("inv0", invert_permutation(p0), persistent=False)
            self.register_buffer("inv1", invert_permutation(p1), persistent=False)

    def forward(self, x):
        if self.deploy:
            return self.pw(x)
        y = self.core(x)
        y = y + self.bn0(self.g0(x[:, self.perm0]))[:, self.inv0]
        y = y + self.bn1(self.g1(x[:, self.perm1]))[:, self.inv1]
        return y

    def get_equivalent_kernel_bias(self):
        """FP64: W_eq = W_core + P0^-1 W~0 P0 + P1^-1 W~1 P1 (same for b)."""
        W, b = self.core.get_equivalent_kernel_bias()   # (D, D, 1, 1), (D,) FP64
        for g, bn, inv in ((self.g0, self.bn0, self.inv0),
                           (self.g1, self.bn1, self.inv1)):
            Wg, bg = fold_conv_bn(g.weight, None, bn)   # (D, D//g, 1, 1) FP64
            Wd = group1x1_to_dense_fp64(Wg, self.groups)
            W = W + Wd[inv][:, inv]
            b = b + bg[inv]
        return W, b

    def branch_stats(self):
        s = {"p0_gamma": self.bn0.weight.norm().item(), "p1_gamma": self.bn1.weight.norm().item(),
             "g0_w": self.g0.weight.norm().item(), "g1_w": self.g1.weight.norm().item()}
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.pw = nn.Conv2d(self.channels, self.channels, 1, bias=True)
        self.pw.weight.data = kernel.float()
        self.pw.bias.data = bias.float()
        self.deploy = True
        for name in ("core", "g0", "g1", "bn0", "bn1"):
            if hasattr(self, name):
                delattr(self, name)
        return self


# -----------------------------------------------------------------------------
# Run10 PFDR: Pre-Fusion Dilated Re-parameterization (fine lateral, before fuse1)
# -----------------------------------------------------------------------------
def embed_3x3_center_5x5(w):
    """(D,1,3,3) -> (D,1,5,5): E_3(K)[:,:,1:4,1:4] = K."""
    return F.pad(w, (1, 1, 1, 1))


def embed_3x3_d2_5x5(w):
    """(D,1,3,3) -> (D,1,5,5): E_d2(K)[:,:,{0,2,4},{0,2,4}] = K (dilation-2 sparse)."""
    out = torch.zeros(w.shape[0], 1, 5, 5, dtype=w.dtype, device=w.device)
    out[:, :, 0::2, 0::2] = w
    return out


def embed_1x1_center_5x5(w):
    """(D,1,1,1) -> (D,1,5,5): E_1(K)[:,:,2,2] = K."""
    out = torch.zeros(w.shape[0], 1, 5, 5, dtype=w.dtype, device=w.device)
    out[:, :, 2, 2] = w
    return out


class PFDRDW5(nn.Module):
    """Run10 PFDR: Pre-Fusion Dilated Re-parameterization depthwise 5x5 block.

    Train graph (mode="rep"):
        Y = BN5(DW5x5(X)) + BN3(DW3x3(X)) + BNd2(DW3x3(d=2)(X)) + BN1(DW1x1(X)) + alpha*X
    aux convs are zero-init and aux BNs have gamma=beta=0, so the epoch-0 aux
    output is EXACTLY 0 and M1 starts from the plain main-branch prediction.
    Deploy graph (both modes): single DW5x5 (groups=D, padding=2, bias=True);
    all branches are composed in FP64 and cast to FP32 exactly once.

    mode="plain" (C0): only the main DW5+BN (+alpha residual) — deploy graph
    is IDENTICAL to mode="rep" (M1), so M1-C0 isolates the rep effect.

    NOTE (RNG): nn.Conv2d construction consumes RNG even for immediately-zeroed
    weights, so this module MUST be attached AFTER all other RNG-consuming
    constructions (Run9-style) for the C0/M1 epoch-0 bitwise identity to hold.
    """

    def __init__(self, channels, mode="rep", use_alpha=True, deploy=False):
        super().__init__()
        self.channels = channels
        self.mode = mode if mode in ("plain", "rep") else "rep"
        self.use_alpha = use_alpha
        self.deploy = deploy
        if deploy:
            self.dw = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=True)
        else:
            self.dw5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False)
            self.bn5 = nn.BatchNorm2d(channels)
            nn.init.kaiming_normal_(self.dw5.weight, mode="fan_out", nonlinearity="relu")
            if self.mode == "rep":
                self.dw3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
                self.bn3 = nn.BatchNorm2d(channels)
                self.dwd2 = nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, groups=channels, bias=False)
                self.bnd2 = nn.BatchNorm2d(channels)
                self.dw1 = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False)
                self.bn1 = nn.BatchNorm2d(channels)
                for c in (self.dw3, self.dwd2, self.dw1):
                    nn.init.zeros_(c.weight)
                for bn in (self.bn3, self.bnd2, self.bn1):
                    nn.init.zeros_(bn.weight)
                    nn.init.zeros_(bn.bias)
            if self.use_alpha:
                self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.deploy:
            return self.dw(x)
        y = self.bn5(self.dw5(x))
        if self.mode == "rep":
            y = y + self.bn3(self.dw3(x)) + self.bnd2(self.dwd2(x)) + self.bn1(self.dw1(x))
        if self.use_alpha:
            y = y + self.alpha * x
        return y

    def get_equivalent_kernel_bias(self):
        W5, b5 = fold_conv_bn(self.dw5.weight, None, self.bn5)
        K = W5
        b = b5
        if self.mode == "rep":
            W3, b3 = fold_conv_bn(self.dw3.weight, None, self.bn3)
            Wd, bd = fold_conv_bn(self.dwd2.weight, None, self.bnd2)
            W1, b1 = fold_conv_bn(self.dw1.weight, None, self.bn1)
            K = K + embed_3x3_center_5x5(W3) + embed_3x3_d2_5x5(Wd) + embed_1x1_center_5x5(W1)
            b = b + b3 + bd + b1
        if self.use_alpha:
            K = K + self.alpha.detach().double() * identity_dw_kernel(
                self.channels, 5, device=K.device, dtype=torch.float64)
        return K, b

    def branch_stats(self):
        s = {"main_gamma": self.bn5.weight.norm().item()}
        if self.mode == "rep":
            s["near3_gamma"] = self.bn3.weight.norm().item()
            s["dilated3_gamma"] = self.bnd2.weight.norm().item()
            s["center1_gamma"] = self.bn1.weight.norm().item()
        if self.use_alpha:
            s["alpha"] = self.alpha.item()
        return s

    def switch_to_deploy(self):
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dw = nn.Conv2d(self.channels, self.channels, 5, 1, 2, groups=self.channels, bias=True)
        self.dw.weight.data = kernel.float()
        self.dw.bias.data = bias.float()
        self.deploy = True
        for name in ("dw3", "bn3", "dwd2", "bnd2", "dw1", "bn1", "dw5", "bn5", "alpha"):
            if hasattr(self, name):
                delattr(self, name)
        return self
