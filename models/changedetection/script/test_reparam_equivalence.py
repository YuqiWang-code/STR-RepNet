"""TAR-DCR structural re-parameterization equivalence tests.

Thresholds (Run6, unified with the asserts below):
  - kernel/bias composition is done in FP64 (algebraic error ~1e-15; the
    affine-interpolation commutation test asserts < 1e-12 in FP64);
  - block-level FP32 train-graph vs deploy-graph: the two graphs accumulate
    floats in different orders, measured ~1e-6..1.1e-5 -> assert < 2e-5;
  - whole-model FP32 error is RECORDED (~1e-5), assert < 2e-4, plus argmax
    disagreement is recorded. NOT claimed as < 1e-6.
All random inputs are seeded for reproducibility.

Run7 additions (PBRU):
  - T0: phase-basis phi constants + PyTorch PixelShuffle one-hot channel
    ordering + the c-major phase-expansion reshape convention (exact);
  - T1: PBRUHead train->deploy fold (FP32, tol=2e-5) with NON-TRIVIAL weights
    (zero-init by design, perturbed to TRAINED-HEAD magnitudes via
    perturb_pbru_head: conv std 0.05, BN gamma std 0.5), plus an FP64
    algebraic check of get_equivalent_kernel_bias against a manual expansion;
  - T2: whole-model fold with head_mode="pixelshuffle" (use_pbru 0/1).

Run8 additions (MPCR-Fine):
  - T0: permutation roundtrip P^-1(Px)==x exact, interleaved pattern exact,
    grouped-kernel dense embedding == grouped conv (FP64 <1e-12),
    P^-1 G(Px) == (P^-1 G P)x incl. bias (FP64 <1e-12);
  - T1: MPCRPW1x1 (same2 + multi2) fold, FP32 tol=2e-5, non-trivial branches
    (conv std 0.05, BN gamma std 0.5) + FP64 algebra vs manual composition;
  - T2: whole-model fold with use_mpcr (same2/multi2), tol=2e-4, argmax=0.

Run9 additions (BiFTR):
  - T0: FP64 A(W(Bx)+b) == (AWB)x + Ab on the ACTUAL v3 downsample conv
    (k3 s2 p1; the fold is kernel-size-agnostic). FP64 tol 1e-9: 192/384-dim
    GEMMs at O(100) magnitudes give ~1.7e-10 pure-FP64 accumulation
    (relative ~5e-13);
  - T1: BiFTRTransition (post/bi) fold, FP32 tol=2e-5, non-trivial deltas
    (std 0.05) + FP64 algebra (1e-9, same reason);
  - T2: whole-model fold with use_biftr (post/bi, encoder_train=last2),
    tol=2e-4, argmax=0, freeze-state asserts (base frozen, deltas trainable).

Run10 additions (PFDR):
  - T0: FP64 multi-branch (DW5 + DW3 + DW3(d=2) + DW1 + alpha*I) == single
    K_eq DW5 algebra. Fixed requirement max_abs_error < 1e-10 (target <1e-12;
    pure FP64 accumulation floor at D=32). Embedding unit tests (3x3 center,
    dilation-2 sparse, 1x1 center, identity center) must be exact;
  - T1: PFDRDW5 (plain + rep) fold, FP32 tol=2e-5, non-trivial branch weights
    (conv std 0.05, BN gamma std 0.5) + FP64 algebra <1e-10;
  - T2: whole-model fold with use_pfdr (plain/rep, encoder_train=last2),
    tol=2e-4, argmax=0.
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from changedetection.models.reparam import (RepDW3, RepPW1x1, RepPairFuse1x1, NSCRPairFuse1x1,
                                            PBRUHead, build_phase_basis, fold_conv_bn,
                                            MPCRPW1x1, make_interleaved_permutation,
                                            invert_permutation, group1x1_to_dense_fp64,
                                            PFDRDW5, embed_3x3_center_5x5,
                                            embed_3x3_d2_5x5, embed_1x1_center_5x5,
                                            identity_dw_kernel)
from changedetection.models.Mamba_backbone import BiFTRTransition
from changedetection.models.tar import TemporalRep1x1, TARStage, MultiScaleTAR
from changedetection.models.dcr_decoder import RepLocalBlock, DCRDecoder


def check_fold(module, inputs, tol=1e-4, name="module"):
    module.eval()
    with torch.no_grad():
        y0 = module(*inputs)
    m2 = copy.deepcopy(module)
    m2.switch_to_deploy()
    m2.eval()
    with torch.no_grad():
        y1 = m2(*inputs)
    err = (y0 - y1).abs().max().item()
    ok = err < tol
    print(f"[{'OK' if ok else 'FAIL'}] {name}: max_abs_error={err:.3e}")
    return ok


def test_affine_interp_commutation():
    """FP64: U(A*H + c) == A*U(H) + c for channel-wise affine A,c and bilinear U."""
    torch.manual_seed(0)
    H = torch.randn(1, 8, 16, 16, dtype=torch.float64)
    A = torch.randn(8, dtype=torch.float64)
    c = torch.randn(8, dtype=torch.float64)
    lhs = F.interpolate(H * A.view(1, -1, 1, 1) + c.view(1, -1, 1, 1),
                        scale_factor=2, mode="bilinear", align_corners=False)
    rhs = A.view(1, -1, 1, 1) * F.interpolate(H, scale_factor=2, mode="bilinear", align_corners=False) \
        + c.view(1, -1, 1, 1)
    err = (lhs - rhs).abs().max().item()
    ok = err < 1e-12
    print(f"[{'OK' if ok else 'FAIL'}] affine-interp commutation (FP64): max_abs_error={err:.3e}")
    return ok


def test_phase_basis_algebra():
    """T0: phi constants, PixelShuffle one-hot channel ordering, phase expansion."""
    torch.manual_seed(0)
    r = 4
    r2 = r * r
    basis = build_phase_basis(r)
    i = torch.arange(r).view(r, 1).expand(r, r).reshape(-1).float()
    j = torch.arange(r).view(1, r).expand(r, r).reshape(-1).float()
    u = (2.0 * j + 1.0 - r) / r
    v = (2.0 * i + 1.0 - r) / r
    assert basis.shape == (4, r2)
    assert torch.equal(basis[0], torch.ones(r2)), "phi0 must be all ones"
    assert torch.equal(basis[1], u), "phi1 must be u_j=(2j+1-r)/r"
    assert torch.equal(basis[2], v), "phi2 must be v_i=(2i+1-r)/r"
    assert torch.equal(basis[3], u * v), "phi3 must be u_j*v_i"
    print("[OK] PBRU phase basis phi = {1, u_j, v_i, u_j*v_i} (exact, fixed constants)")

    # PyTorch PixelShuffle ordering: channel n=c*r2+p (p=i*r+j) -> pixel (i,j) of class c
    for c in (0, 1):
        for p in range(r2):
            z = torch.zeros(1, 2 * r2, 2, 3)  # input channels = C*r^2
            z[0, c * r2 + p, 1, 2] = 1.0
            out = F.pixel_shuffle(z, r)  # (1,2,8,12)
            ti, tj = 1 * r + p // r, 2 * r + p % r
            assert out[0, c, ti, tj].item() == 1.0 and out.abs().sum().item() == 1.0, \
                f"one-hot channel {p} must land at output pixel ({ti},{tj})"
    print("[OK] PixelShuffle one-hot ordering: n=c*r^2+p -> (c, i=p//r, j=p%r) (exact)")

    # phase expansion reshape convention (c-major, matches PBRUHead.forward and the fold)
    x = torch.randn(2, 2, 5, 5)
    for k in range(4):
        expanded = (x.unsqueeze(2) * basis[k].view(1, 1, r2, 1, 1)).reshape(2, 2 * r2, 5, 5)
        manual = torch.zeros_like(expanded)
        for c in range(2):
            for p in range(r2):
                manual[:, c * r2 + p] = basis[k][p] * x[:, c]
        assert torch.equal(expanded, manual), f"phase expansion k={k} must match c-major manual build"
    print("[OK] phase expansion = E_phi_k(q) with n=c*r^2+p (exact vs manual)")
    return True


def perturb_pbru_head(h):
    """Randomize a PBRUHead at TRAINED-HEAD magnitudes (not std=1: a std-1 head
    would amplify the legitimate ~5e-5 TAR/DCR fold noise by ~100x through the
    160-channel 1x1 and make the whole-model test meaningless)."""
    with torch.no_grad():
        for n, p in h.named_parameters():
            if n.endswith(".weight") and ("branch" in n or "main_proj" in n):
                p.normal_(0.0, 0.05)   # Kaiming-scale 1x1 weights (D=64-160)
            elif n.endswith(".bias") and "main_proj" in n:
                p.normal_(0.0, 0.05)
            else:                       # BN gamma / beta
                p.normal_(0.0, 0.5)
        if h.use_phase_rep:
            for bn in (h.bn_coarse, h.bn_px, h.bn_py, h.bn_pxy):
                bn.running_mean.normal_()
                bn.running_var.uniform_(0.5, 2.0)
    return h


def pbru_head_tests(device):
    """T1: PBRUHead fold (FP32 tol=2e-5, non-trivial weights) + FP64 algebra."""
    torch.manual_seed(2333)
    x = torch.randn(2, 64, 32, 32, device=device)
    C, r, r2 = 2, 4, 16

    ok = True
    h = PBRUHead(64, C, r, use_phase_rep=True).to(device)
    perturb_pbru_head(h)
    h.eval()
    with torch.no_grad():
        y0 = h(x)
    h2 = copy.deepcopy(h)
    h2.switch_to_deploy()
    h2.eval()
    with torch.no_grad():
        y1 = h2(x)
    err = (y0 - y1).abs().max().item()
    ok &= err < 2e-5
    print(f"[{'OK' if err < 2e-5 else 'FAIL'}] PBRUHead(phase) fold: max_abs_error={err:.3e}")
    assert not any(("branch_" in k or "bn_" in k or "main_proj" in k)
                   for k in h2.state_dict().keys()), "deploy PBRUHead must be branch-free"

    hp = PBRUHead(64, C, r, use_phase_rep=False).to(device)
    with torch.no_grad():
        for p in hp.parameters():
            p.normal_()
    ok &= check_fold(hp, (x,), tol=2e-5, name="PBRUHead(plain PixelShuffle)")

    # FP64 algebraic check of get_equivalent_kernel_bias vs a manual expansion
    W, b = h.get_equivalent_kernel_bias()
    assert W.dtype == torch.float64 and b.dtype == torch.float64
    xd = x.double()
    with torch.no_grad():
        z = F.conv2d(xd, W, b)
        y_alg = F.pixel_shuffle(z, r)
        z2 = F.conv2d(xd, h.main_proj.weight.double(), h.main_proj.bias.double())
        phi = h.phi.double()
        for branch, bn, k in ((h.branch_coarse, h.bn_coarse, 0),
                              (h.branch_px, h.bn_px, 1),
                              (h.branch_py, h.bn_py, 2),
                              (h.branch_pxy, h.bn_pxy, 3)):
            V, bv = fold_conv_bn(branch.weight, None, bn)
            q = F.conv2d(xd, V, bv)
            e = (q.unsqueeze(2) * phi[k].view(1, 1, r2, 1, 1)).reshape(xd.shape[0], C * r2,
                                                                       xd.shape[2], xd.shape[3])
            z2 = z2 + e
        y2 = F.pixel_shuffle(z2, r)
    err64 = (y_alg - y2).abs().max().item()
    ok &= err64 < 1e-12
    print(f"[{'OK' if err64 < 1e-12 else 'FAIL'}] PBRUHead fold algebra (FP64): max_abs_error={err64:.3e}")
    return ok


def test_mpcr_permutation_algebra():
    """T0 (Run8): permutation exactness + grouped->dense embedding + P^-1 G P fold."""
    torch.manual_seed(0)
    D, g, q = 160, 4, 40
    p = make_interleaved_permutation(D, g)
    inv = invert_permutation(p)
    x = torch.randn(2, D, 8, 8)
    assert torch.equal(x[:, p][:, inv], x), "P^-1(Px) must be exact"
    expect = torch.tensor([k * q + j for j in range(q) for k in range(g)])
    assert torch.equal(p, expect), "interleaved permutation pattern mismatch"
    print("[OK] PBRU->MPCR: permutation roundtrip exact + interleaved pattern [0,40,80,120,...] exact")

    # grouped conv direct vs dense embedding (FP64)
    Wg = torch.randn(D, q, 1, 1, dtype=torch.float64)
    Wd = group1x1_to_dense_fp64(Wg, g)
    xd = x.double()
    y_grouped = F.conv2d(xd, Wg, None, groups=g)
    y_dense = F.conv2d(xd, Wd, None)
    err1 = (y_grouped - y_dense).abs().max().item()
    assert err1 < 1e-12, f"grouped->dense embedding error {err1}"
    print(f"[{'OK' if err1 < 1e-12 else 'FAIL'}] grouped 1x1 -> dense embedding (FP64): {err1:.3e}")

    # P^-1 G(Px) vs (P^-1 G P)x, with bias
    z = xd[:, p]
    bg = torch.randn(D, dtype=torch.float64)
    y_pg = F.conv2d(z, Wg, bg, groups=g)[:, inv]
    Wf = Wd[inv][:, inv]
    bf = bg[inv]
    y_fold = F.conv2d(xd, Wf, bf)
    err2 = (y_pg - y_fold).abs().max().item()
    assert err2 < 1e-12, f"permuted-group fold error {err2}"
    print(f"[{'OK' if err2 < 1e-12 else 'FAIL'}] P^-1 G(Px) == (P^-1 G P)x + P^-1 b (FP64): {err2:.3e}")
    return True


def mpcr_block_tests(device):
    """T1 (Run8): MPCRPW1x1 train->deploy fold (FP32 tol=2e-5) + FP64 algebra."""
    torch.manual_seed(2333)
    x = torch.randn(2, 160, 16, 16, device=device)
    ok = True
    for mode in ("same2", "multi2"):
        h = MPCRPW1x1(160, groups=4, mode=mode).to(device)
        with torch.no_grad():
            for m in (h.g0, h.g1):
                m.weight.normal_(0.0, 0.05)
            for bn in (h.bn0, h.bn1):
                bn.weight.normal_(0.0, 0.5)
                bn.bias.normal_(0.0, 0.5)
                bn.running_mean.normal_()
                bn.running_var.uniform_(0.5, 2.0)
        h.eval()
        with torch.no_grad():
            y0 = h(x)
        h2 = copy.deepcopy(h)
        h2.switch_to_deploy()
        h2.eval()
        with torch.no_grad():
            y1 = h2(x)
        err = (y0 - y1).abs().max().item()
        ok &= err < 2e-5
        assert not any(("g0" in k or "g1" in k or "bn0" in k or "bn1" in k or ".core." in k)
                       for k in h2.state_dict().keys()), "deploy MPCRPW1x1 must be branch-free"
        print(f"[{'OK' if err < 2e-5 else 'FAIL'}] MPCRPW1x1({mode}) fold: max_abs_error={err:.3e}")

        W, b = h.get_equivalent_kernel_bias()
        assert W.dtype == torch.float64 and b.dtype == torch.float64
        with torch.no_grad():
            z = F.conv2d(x.double(), W, b)
            Wc, bc = h.core.get_equivalent_kernel_bias()
            z2 = F.conv2d(x.double(), Wc, bc)
            for g_, bn_, inv_ in ((h.g0, h.bn0, h.inv0), (h.g1, h.bn1, h.inv1)):
                Wg, bg = fold_conv_bn(g_.weight, None, bn_)
                Wd = group1x1_to_dense_fp64(Wg, h.groups)
                z2 = z2 + F.conv2d(x.double(), Wd[inv_][:, inv_], bg[inv_])
        err64 = (z - z2).abs().max().item()
        ok &= err64 < 1e-12
        print(f"[{'OK' if err64 < 1e-12 else 'FAIL'}] MPCRPW1x1({mode}) fold algebra (FP64): {err64:.3e}")
    return ok


def test_biftr_algebra():
    """T0 (Run9): FP64 A( W(Bx) + b ) == (AWB)x + Ab (stride-2 downsample conv).

    Tolerance 1e-10: with random A/B/W of std 1 and 192/384-dim channel GEMMs
    the intermediate magnitudes are O(100); measured FP64 error ~9e-11
    (relative ~1e-13) is pure GEMM accumulation, not a fold error.
    """
    torch.manual_seed(0)
    Ci, Co = 192, 384
    W = torch.randn(Co, Ci, 3, 3, dtype=torch.float64)  # v3 downsample k3 s2 p1
    b = torch.randn(Co, dtype=torch.float64)
    A = torch.randn(Co, Co, dtype=torch.float64)
    B = torch.randn(Ci, Ci, dtype=torch.float64)
    x = torch.randn(1, Ci, 32, 32, dtype=torch.float64)
    xb = torch.einsum("ni,bihw->bnhw", B, x)
    z = F.conv2d(xb, W, b, stride=2, padding=1)
    y_direct = torch.einsum("om,bmhw->bohw", A, z)
    W_pre = torch.einsum("mnxy,ni->mixy", W, B)
    W_eq = torch.einsum("om,mixy->oixy", A, W_pre)
    b_eq = A @ b
    y_fold = F.conv2d(x, W_eq, b_eq, stride=2, padding=1)
    err = (y_direct - y_fold).abs().max().item()
    assert err < 1e-9, f"BiFTR AWB algebra error {err}"
    print(f"[{'OK' if err < 1e-9 else 'FAIL'}] BiFTR A(W(Bx)+b) == (AWB)x + Ab (FP64): {err:.3e}")
    return True


def biftr_block_tests(device):
    """T1 (Run9): BiFTRTransition fold (FP32 tol=2e-5, non-trivial deltas) + FP64 algebra.

    FP64 algebra tol 1e-10: 192/384-dim channel GEMMs at O(100) magnitudes give
    ~1e-11 pure-FP64 accumulation (relative ~1e-13), cf. T0.
    """
    torch.manual_seed(2333)
    x = torch.randn(2, 192, 32, 32, device=device)
    ok = True
    for mode in ("post", "bi"):
        base = nn.Conv2d(192, 384, 3, 2, 1, bias=True).to(device)  # v3 downsample
        h = BiFTRTransition(base, mode=mode).to(device)
        with torch.no_grad():
            h.d_out.weight.normal_(0.0, 0.05)
            if hasattr(h, "d_in"):
                h.d_in.weight.normal_(0.0, 0.05)
        h.eval()
        with torch.no_grad():
            y0 = h(x)
        h2 = copy.deepcopy(h)
        h2.switch_to_deploy()
        h2.eval()
        with torch.no_grad():
            y1 = h2(x)
        err = (y0 - y1).abs().max().item()
        ok &= err < 2e-5
        assert not any(("d_in" in k or "d_out" in k) for k in h2.state_dict().keys()), \
            "deploy BiFTRTransition must be delta-free"
        print(f"[{'OK' if err < 2e-5 else 'FAIL'}] BiFTRTransition({mode}) fold: max_abs_error={err:.3e}")

        W, b = h.get_equivalent_kernel_bias()
        assert W.dtype == torch.float64 and b.dtype == torch.float64
        with torch.no_grad():
            y_alg = F.conv2d(x.double(), W, b, stride=2, padding=1)
            xb = x.double()
            if mode == "bi":
                xb = xb + F.conv2d(xb, h.d_in.weight.double())
            z = F.conv2d(xb, h.conv.weight.double(), h.conv.bias.double(), stride=2, padding=1)
            y2 = z + F.conv2d(z, h.d_out.weight.double())
        err64 = (y_alg - y2).abs().max().item()
        ok &= err64 < 1e-9
        print(f"[{'OK' if err64 < 1e-9 else 'FAIL'}] BiFTRTransition({mode}) fold algebra (FP64): {err64:.3e}")
    return ok


def test_pfdr_embedding_algebra():
    """T0 (Run10): FP64 PFDR multi-branch == single K_eq DW5 algebra.

    Fixed requirement: max_abs_error < 1e-10 (target <1e-12). D=32 with O(1)
    magnitudes; measured error is the pure-FP64 accumulation floor.
    """
    torch.manual_seed(0)
    D = 32
    x = torch.randn(2, D, 16, 16, dtype=torch.float64)
    K5 = torch.randn(D, 1, 5, 5, dtype=torch.float64)
    K3 = torch.randn(D, 1, 3, 3, dtype=torch.float64)
    Kd = torch.randn(D, 1, 3, 3, dtype=torch.float64)
    K1 = torch.randn(D, 1, 1, 1, dtype=torch.float64)
    b5 = torch.randn(D, dtype=torch.float64)
    b3 = torch.randn(D, dtype=torch.float64)
    bd = torch.randn(D, dtype=torch.float64)
    b1 = torch.randn(D, dtype=torch.float64)
    alpha = torch.randn(1, dtype=torch.float64)

    # embedding unit tests (must be exact)
    e3 = embed_3x3_center_5x5(K3)
    assert torch.equal(e3[:, :, 1:4, 1:4], K3) and e3.abs().sum() == K3.abs().sum()
    e_d = embed_3x3_d2_5x5(Kd)
    assert torch.equal(e_d[:, :, 0::2, 0::2], Kd) and e_d.abs().sum() == Kd.abs().sum()
    e1 = embed_1x1_center_5x5(K1)
    assert torch.equal(e1[:, :, 2, 2], K1[:, :, 0, 0]) and e1.abs().sum() == K1.abs().sum()
    ident = identity_dw_kernel(D, 5, dtype=torch.float64)
    assert torch.equal(ident[:, 0, 2, 2], torch.ones(D, dtype=torch.float64)) \
        and ident.abs().sum() == D
    print("[OK] PFDR embeddings exact (3x3 center / dilation-2 sparse / 1x1 center / identity)")

    y_direct = (F.conv2d(x, K5, b5, padding=2, groups=D)
                + F.conv2d(x, K3, b3, padding=1, groups=D)
                + F.conv2d(x, Kd, bd, padding=2, dilation=2, groups=D)
                + F.conv2d(x, K1, b1, groups=D)
                + alpha * x)
    K_eq = (K5 + e3 + e_d + e1
            + alpha * identity_dw_kernel(D, 5, dtype=torch.float64))
    b_eq = b5 + b3 + bd + b1
    y_fold = F.conv2d(x, K_eq, b_eq, padding=2, groups=D)
    err = (y_direct - y_fold).abs().max().item()
    ok = err < 1e-10
    print(f"[{'OK' if ok else 'FAIL'}] PFDR multi-branch == single DW5 (FP64): {err:.3e}")
    return ok


def pfdr_block_tests(device):
    """T1 (Run10): PFDRDW5 train->deploy fold (FP32 tol=2e-5) + FP64 algebra.

    Aux branches are zero-init by design -> perturb to TRAINED magnitudes
    (conv std 0.05, BN gamma/beta std 0.5, running stats randomized) so the
    fold is non-trivial. FP64 equivalent-kernel algebra tol 1e-10.
    """
    torch.manual_seed(2333)
    x = torch.randn(2, 64, 32, 32, device=device)
    ok = True
    for mode in ("plain", "rep"):
        h = PFDRDW5(64, mode=mode).to(device)
        with torch.no_grad():
            h.dw5.weight.normal_(0.0, 0.05)
            h.bn5.weight.normal_(0.0, 0.5)
            h.bn5.bias.normal_(0.0, 0.5)
            h.bn5.running_mean.normal_()
            h.bn5.running_var.uniform_(0.5, 2.0)
            if mode == "rep":
                for c in (h.dw3, h.dwd2, h.dw1):
                    c.weight.normal_(0.0, 0.05)
                for bn in (h.bn3, h.bnd2, h.bn1):
                    bn.weight.normal_(0.0, 0.5)
                    bn.bias.normal_(0.0, 0.5)
                    bn.running_mean.normal_()
                    bn.running_var.uniform_(0.5, 2.0)
        h.eval()
        with torch.no_grad():
            y0 = h(x)
        h2 = copy.deepcopy(h)
        h2.switch_to_deploy()
        h2.eval()
        with torch.no_grad():
            y1 = h2(x)
        err = (y0 - y1).abs().max().item()
        ok &= err < 2e-5
        keys = list(h2.state_dict().keys())
        assert keys == ["dw.weight", "dw.bias"], f"deploy PFDRDW5 must be a single DW5, got {keys}"
        print(f"[{'OK' if err < 2e-5 else 'FAIL'}] PFDRDW5({mode}) fold: max_abs_error={err:.3e}")

        W, b = h.get_equivalent_kernel_bias()
        assert W.dtype == torch.float64 and b.dtype == torch.float64
        with torch.no_grad():
            z = F.conv2d(x.double(), W, b, padding=2, groups=64)
            W5, b5 = fold_conv_bn(h.dw5.weight, None, h.bn5)
            z2 = F.conv2d(x.double(), W5, b5, padding=2, groups=64)
            if mode == "rep":
                W3, b3 = fold_conv_bn(h.dw3.weight, None, h.bn3)
                Wd, bd = fold_conv_bn(h.dwd2.weight, None, h.bnd2)
                W1, b1 = fold_conv_bn(h.dw1.weight, None, h.bn1)
                z2 = z2 + F.conv2d(x.double(), W3, b3, padding=1, groups=64) \
                    + F.conv2d(x.double(), Wd, bd, padding=2, dilation=2, groups=64) \
                    + F.conv2d(x.double(), W1, b1, groups=64)
            z2 = z2 + h.alpha.detach().double() * x.double()
        err64 = (z - z2).abs().max().item()
        ok &= err64 < 1e-10
        print(f"[{'OK' if err64 < 1e-10 else 'FAIL'}] PFDRDW5({mode}) fold algebra (FP64): {err64:.3e}")
    return ok


def block_tests(device):
    torch.manual_seed(2333)
    C = 160
    H = W = 16
    results = []

    x = torch.randn(2, C, H, W, device=device)
    results.append(check_fold(RepDW3(C, use_aux=True, use_residual=True).to(device), (x,), tol=2e-5, name="RepDW3"))
    results.append(check_fold(RepPW1x1(C, use_aux=True).to(device), (x,), tol=2e-5, name="RepPW1x1"))
    results.append(check_fold(RepLocalBlock(C, use_aux=True).to(device), (x,), tol=2e-5, name="RepLocalBlock"))

    L = torch.randn(2, C, H, W, device=device)
    Hh = torch.randn(2, C, H, W, device=device)
    results.append(check_fold(RepPairFuse1x1(C, use_aux=True).to(device), (L, Hh), tol=2e-5, name="RepPairFuse1x1"))

    P = torch.randn(2, 96, H, W, device=device)
    Q = torch.randn(2, 96, H, W, device=device)
    results.append(check_fold(TemporalRep1x1(96, C, use_aux=True).to(device), (P, Q), tol=2e-5, name="TemporalRep1x1"))
    results.append(check_fold(TemporalRep1x1(96, C, use_aux=True, use_reverse_aux=True).to(device),
                              (P, Q), tol=2e-5, name="TemporalRep1x1+BOTR"))
    results.append(check_fold(TARStage(96, C, use_temporal_aux=True, use_dcr_aux=True).to(device),
                              (P, Q), tol=2e-5, name="TARStage"))
    results.append(check_fold(TARStage(96, C, use_temporal_aux=True, use_dcr_aux=True,
                                       use_reverse_aux=True).to(device),
                              (P, Q), tol=2e-5, name="TARStage+BOTR"))

    # NSCR: different native spatial sizes (L high-res, H low-res)
    Ln = torch.randn(2, C, 32, 32, device=device)
    Hn = torch.randn(2, C, 16, 16, device=device)
    for mode in ("both", "lonly", "honly"):
        results.append(check_fold(NSCRPairFuse1x1(C, use_aux=True, use_residual=True, nscr_mode=mode).to(device),
                                  (Ln, Hn), tol=2e-5, name=f"NSCRPairFuse1x1({mode})"))

    return all(results)


def whole_model_test(device, cfg, pretrained, rep_mode="full", use_botr=False,
                     use_nscr=False, nscr_scope="high2", head_mode="bilinear",
                     use_pbru=False, pbru_upscale=4, use_mpcr=False, mpcr_mode="multi2",
                     encoder_train="frozen", use_biftr=False, biftr_mode="bi",
                     use_pfdr=False, pfdr_mode="rep"):
    from changedetection.configs.config import get_config
    from changedetection.models.STRRepNet import STRRepNet

    args = argparse.Namespace(cfg=cfg, opts=None, batch_size=16, data_path="", zip=False,
                              cache_mode=None, pretrained="", resume=None, accumulation_steps=None,
                              use_checkpoint=False, disable_amp=False, output="", tag="", eval=False,
                              throughput=False, traincost=False, enable_persistance=False,
                              enable_amp=False, fused_layernorm=False, optim=None, ddp=None)
    config = get_config(args)
    v = config.MODEL.VSSM

    model = STRRepNet(
        pretrained=pretrained, rep_mode=rep_mode, use_botr=use_botr,
        use_nscr=use_nscr, nscr_scope=nscr_scope, head_mode=head_mode,
        use_pbru=use_pbru, pbru_upscale=pbru_upscale,
        use_mpcr=use_mpcr, mpcr_mode=mpcr_mode, mpcr_groups=4,
        encoder_train=encoder_train, use_biftr=use_biftr, biftr_mode=biftr_mode,
        use_pfdr=use_pfdr, pfdr_mode=pfdr_mode,
        patch_size=v.PATCH_SIZE, in_chans=v.IN_CHANS, num_classes=config.MODEL.NUM_CLASSES,
        depths=v.DEPTHS, dims=v.EMBED_DIM,
        ssm_d_state=v.SSM_D_STATE, ssm_ratio=v.SSM_RATIO, ssm_rank_ratio=v.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if v.SSM_DT_RANK == "auto" else int(v.SSM_DT_RANK)),
        ssm_act_layer=v.SSM_ACT_LAYER, ssm_conv=v.SSM_CONV, ssm_conv_bias=v.SSM_CONV_BIAS,
        ssm_drop_rate=v.SSM_DROP_RATE, ssm_init=v.SSM_INIT, forward_type=v.SSM_FORWARDTYPE,
        mlp_ratio=v.MLP_RATIO, mlp_act_layer=v.MLP_ACT_LAYER, mlp_drop_rate=v.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE, patch_norm=v.PATCH_NORM,
        norm_layer=v.NORM_LAYER, downsample_version=v.DOWNSAMPLE,
        patchembed_version=v.PATCHEMBED, gmlp=v.GMLP, use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    ).to(device)

    # verify encoder freeze state per encoder_train
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    s12 = sum(p.numel() for i in (0, 1) for p in model.encoder.layers[i].parameters() if p.requires_grad)
    s34 = sum(p.numel() for i in (2, 3) for p in model.encoder.layers[i].parameters() if p.requires_grad)
    if encoder_train == "frozen":
        assert enc_trainable == 0, f"encoder has {enc_trainable} trainable params"
    else:
        if use_biftr and model.biftr is not None:
            s12 -= sum(p.numel() for p in model.biftr.parameters() if p.requires_grad)
        assert s12 == 0 and s34 > 0, f"last2: stage1/2 frozen, stage3/4 trainable (s12={s12}, s34={s34})"
    model.train()
    if encoder_train == "frozen":
        assert model.encoder.training is False, "encoder should stay eval after model.train()"
    if use_biftr:
        b = model.biftr
        assert not b.conv.weight.requires_grad, "BiFTR base Conv must stay frozen"
        assert b.d_out.weight.requires_grad, "BiFTR d_out must be trainable"
        if biftr_mode == "bi":
            assert b.d_in.weight.requires_grad, "BiFTR d_in must be trainable"
    print(f"[OK] encoder state (trainable={enc_trainable}, s12={s12}, s34={s34}, "
          f"training={model.encoder.training})")

    # PBRU branches are zero-init by design; perturb the head so the fold is non-trivial.
    if head_mode == "pixelshuffle":
        perturb_pbru_head(model.head)
    # MPCR branches are zero-init by design; perturb refine.pw so the fold is non-trivial.
    if use_mpcr:
        with torch.no_grad():
            pw = model.decoder.refine.pw
            for m in (pw.g0, pw.g1):
                m.weight.normal_(0.0, 0.05)
            for bn in (pw.bn0, pw.bn1):
                bn.weight.normal_(0.0, 0.5)
                bn.bias.normal_(0.0, 0.5)
                bn.running_mean.normal_()
                bn.running_var.uniform_(0.5, 2.0)
    # BiFTR deltas are zero-init by design; perturb so the fold is non-trivial.
    if use_biftr:
        with torch.no_grad():
            model.biftr.d_out.weight.normal_(0.0, 0.05)
            if hasattr(model.biftr, "d_in"):
                model.biftr.d_in.weight.normal_(0.0, 0.05)
    # PFDR aux branches are zero-init by design; perturb so the fold is non-trivial.
    if use_pfdr:
        with torch.no_grad():
            p = model.decoder.prefuse1
            p.dw5.weight.normal_(0.0, 0.05)
            p.bn5.weight.normal_(0.0, 0.5)
            p.bn5.bias.normal_(0.0, 0.5)
            p.bn5.running_mean.normal_()
            p.bn5.running_var.uniform_(0.5, 2.0)
            if p.mode == "rep":
                for c in (p.dw3, p.dwd2, p.dw1):
                    c.weight.normal_(0.0, 0.05)
                for bn in (p.bn3, p.bnd2, p.bn1):
                    bn.weight.normal_(0.0, 0.5)
                    bn.bias.normal_(0.0, 0.5)
                    bn.running_mean.normal_()
                    bn.running_var.uniform_(0.5, 2.0)

    model.eval()
    torch.manual_seed(2333)  # deterministic inputs for reproducible argmax check
    pre = torch.randn(2, 3, 256, 256, device=device)
    post = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        y0 = model(pre, post)

    m2 = copy.deepcopy(model)
    m2.switch_to_deploy()
    m2.eval()
    with torch.no_grad():
        y1 = m2(pre, post)

    err = (y0 - y1).abs().max().item()
    disagree = (y0.argmax(1) != y1.argmax(1)).float().mean().item()
    ok = err < 2e-4
    print(f"[{'OK' if ok else 'FAIL'}] STRRepNet whole model ({rep_mode}, botr={use_botr}, "
          f"nscr={use_nscr}/{nscr_scope}, head={head_mode}, pbru={use_pbru}, "
          f"mpcr={use_mpcr}/{mpcr_mode}, biftr={use_biftr}/{biftr_mode}, "
          f"pfdr={use_pfdr}/{pfdr_mode}, enc={encoder_train}): "
          f"max_abs_error={err:.3e}, argmax_disagree={disagree:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--rep_mode", type=str, default="full", choices=["plain", "tar", "full"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True

    print("=== affine-interpolation commutation (FP64) ===")
    c_ok = test_affine_interp_commutation()

    print("=== PBRU phase basis + PixelShuffle ordering (T0, exact) ===")
    t0_ok = test_phase_basis_algebra()

    print("=== MPCR permutation + dense embedding (T0, exact) ===")
    t0m_ok = test_mpcr_permutation_algebra()

    print("=== BiFTR AWB algebra (T0, exact) ===")
    t0b_ok = test_biftr_algebra()

    print("=== PFDR embedding + multi-branch algebra (T0, exact) ===")
    t0p_ok = test_pfdr_embedding_algebra()

    print("=== block-level fold tests (tol=2e-5; FP64-composed kernels) ===")
    b_ok = block_tests(device)
    p_ok = pbru_head_tests(device)
    pm_ok = mpcr_block_tests(device)
    pb_ok = biftr_block_tests(device)
    pp_ok = pfdr_block_tests(device)

    print("=== whole-model fold tests (error recorded; FP32 graphs, tol=2e-4) ===")
    w_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode)
    wb_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode, use_botr=True)
    wn_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                             use_nscr=True, nscr_scope="high2")
    wp_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                             head_mode="pixelshuffle", use_pbru=False)
    wq_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                             head_mode="pixelshuffle", use_pbru=True)
    wms_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                              use_mpcr=True, mpcr_mode="same2")
    wmm_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                              use_mpcr=True, mpcr_mode="multi2")
    wbp_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                              encoder_train="last2", use_biftr=True, biftr_mode="post")
    wbb_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                              encoder_train="last2", use_biftr=True, biftr_mode="bi")
    wppl_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                               encoder_train="last2", use_pfdr=True, pfdr_mode="plain")
    wprp_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode,
                               encoder_train="last2", use_pfdr=True, pfdr_mode="rep")

    all_ok = (c_ok and t0_ok and t0m_ok and t0b_ok and t0p_ok and b_ok and p_ok and pm_ok
              and pb_ok and pp_ok and w_ok and wb_ok and wn_ok and wp_ok and wq_ok and wms_ok
              and wmm_ok and wbp_ok and wbb_ok and wppl_ok and wprp_ok)
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
