"""Smoke test for STR-RepNet Clean TAR-DCR (build + forward + params/FLOPs + deploy + dataloader).

Run6 additions: --use_nscr checks (zero-init residual, gamma gradient, deploy
branch deletion, deploy params/FLOPs identical to use_nscr=0, argmax=0).
Run7 additions: --head_mode/--use_pbru checks (zero-init phase basis = exactly
the plain PixelShuffle prediction at init, gamma gradient, deploy fold clean,
deploy params/FLOPs identical to use_pbru=0).
Run8 additions: --use_mpcr checks (MPCR only on refine.pw, BN zero-init,
epoch-0 output EXACTLY equal to use_mpcr=0, group-conv/gamma gradients non-zero,
deploy branch deletion, deploy Params/FLOPs EXACTLY equal to the anchor).
Run10 additions: --use_pfdr checks (PFDRDW5 only on decoder t1 before fuse1,
aux conv/BN zero-init, C0/M1 main-branch init identical + epoch-0 logits
bitwise equal, aux BN gamma grads non-zero + aux conv grads after gamma nudge,
deploy = single DW5, C0/M1 deploy Params/FLOPs identical, optional anchor
budget check at D* via --check_anchor).
"""
import copy
import os
import sys
import argparse

import numpy as np
import torch

_MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list


def build_model(args, config):
    v = config.MODEL.VSSM
    return STRRepNet(
        pretrained=args.pretrained_weight_path, rep_mode=args.rep_mode,
        dim=args.decoder_dim, use_residual=bool(args.use_residual),
        encoder_train=args.encoder_train,
        use_botr=bool(args.use_botr), use_nscr=bool(args.use_nscr),
        nscr_scope=args.nscr_scope, head_mode=args.head_mode,
        use_pbru=bool(args.use_pbru), pbru_upscale=args.pbru_upscale,
        use_mpcr=bool(args.use_mpcr), mpcr_mode=args.mpcr_mode, mpcr_groups=args.mpcr_groups,
        use_biftr=bool(args.use_biftr), biftr_mode=args.biftr_mode,
        use_pfdr=bool(args.use_pfdr), pfdr_mode=args.pfdr_mode,
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
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, required=True)
    ap.add_argument('--opts', default=None, nargs='+')
    ap.add_argument('--pretrained_weight_path', type=str, required=True)
    ap.add_argument('--dataset_root', type=str, required=True)
    ap.add_argument('--test_list', type=str, required=True)
    ap.add_argument('--rep_mode', type=str, default='full', choices=['plain', 'tar', 'dcr', 'full'])
    ap.add_argument('--encoder_train', type=str, default='frozen', choices=['frozen', 'last2', 'full'])
    ap.add_argument('--use_residual', type=int, default=1)
    ap.add_argument('--use_botr', type=int, default=0)
    ap.add_argument('--use_nscr', type=int, default=0)
    ap.add_argument('--nscr_scope', type=str, default='high2', choices=['high2', 'lonly', 'honly'])
    ap.add_argument('--decoder_dim', type=int, default=160)
    ap.add_argument('--head_mode', type=str, default='bilinear', choices=['bilinear', 'pixelshuffle'])
    ap.add_argument('--use_pbru', type=int, default=0)
    ap.add_argument('--pbru_upscale', type=int, default=4)
    ap.add_argument('--use_mpcr', type=int, default=0)
    ap.add_argument('--mpcr_mode', type=str, default='multi2', choices=['same2', 'multi2'])
    ap.add_argument('--mpcr_groups', type=int, default=4)
    ap.add_argument('--use_biftr', type=int, default=0)
    ap.add_argument('--biftr_mode', type=str, default='bi', choices=['post', 'bi'])
    ap.add_argument('--use_pfdr', type=int, default=0)
    ap.add_argument('--pfdr_mode', type=str, default='rep', choices=['plain', 'rep'])
    ap.add_argument('--check_anchor', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    config = get_config(args)

    print("[1/6] building model ...")
    model = build_model(args, config).cuda()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params = {total/1e6:.3f} M, trainable = {trainable/1e6:.3f} M")

    if args.use_nscr:
        for name in ("fuse1", "fuse2"):
            m = getattr(model.decoder, name)
            assert m.__class__.__name__ == "NSCRPairFuse1x1", f"{name} should be NSCRPairFuse1x1"
            for bn in ("bn_l", "bn_h"):
                if hasattr(m, bn):
                    b = getattr(m, bn)
                    assert b.weight.abs().sum().item() == 0.0 and b.bias.abs().sum().item() == 0.0
        print("  NSCR zero-init OK (bn_l/bn_h weight=bias=0; residual output is exactly 0 at init)")

    if args.use_pbru:
        assert model.head.__class__.__name__ == "PBRUHead", "head should be PBRUHead"
        for name in ("bn_coarse", "bn_px", "bn_py", "bn_pxy"):
            b = getattr(model.head, name)
            assert b.weight.abs().sum().item() == 0.0 and b.bias.abs().sum().item() == 0.0
        print("  PBRU zero-init OK (4 branch BN weight=bias=0)")

    if args.use_mpcr:
        # MPCR must appear ONLY on refine.pw
        assert model.decoder.refine.pw.__class__.__name__ == "MPCRPW1x1", \
            "refine.pw should be MPCRPW1x1"
        for name in ("block1", "block2", "block3"):
            assert getattr(model.decoder, name).pw.__class__.__name__ == "RepPW1x1", \
                f"{name}.pw must stay RepPW1x1"
        assert model.decoder.refine.dw.__class__.__name__ == "RepDW3", "refine.dw untouched"
        pw = model.decoder.refine.pw
        for bn in ("bn0", "bn1"):
            b = getattr(pw, bn)
            assert b.weight.abs().sum().item() == 0.0 and b.bias.abs().sum().item() == 0.0
        print("  MPCR zero-init OK (bn0/bn1 weight=bias=0; MPCR only on refine.pw)")

    if args.use_biftr:
        # wrapper located exactly at layers[1].downsample[1], 192->384 k2 s2
        b = model.biftr
        assert b is not None and b.__class__.__name__ == "BiFTRTransition"
        assert model.encoder.layers[1].downsample[1] is b, "wrapper must replace downsample[1]"
        c = b.conv
        assert c.in_channels == 192 and c.out_channels == 384
        assert c.stride == (2, 2)
        print(f"  BiFTR target conv: {c.in_channels}->{c.out_channels}, k={c.kernel_size}, s={c.stride}")
        # freeze states: base frozen, deltas trainable
        assert not c.weight.requires_grad, "base downsample Conv must stay frozen"
        assert b.d_out.weight.requires_grad, "d_out must be trainable"
        if args.biftr_mode == "bi":
            assert b.d_in.weight.requires_grad, "d_in must be trainable"
        else:
            assert not hasattr(b, "d_in"), "post mode must not have d_in"
        # zero-init
        assert b.d_out.weight.abs().sum().item() == 0.0
        if hasattr(b, "d_in"):
            assert b.d_in.weight.abs().sum().item() == 0.0
        print("  BiFTR OK (192->384 stride-2 wrapped; base frozen; deltas trainable + zero-init)")

    if args.use_pfdr:
        print("[1b/6] PFDR structure ...")
        p = model.decoder.prefuse1
        assert p is not None and p.__class__.__name__ == "PFDRDW5", "decoder.prefuse1 must be PFDRDW5"
        assert p.mode == args.pfdr_mode, f"prefuse1 mode {p.mode} != {args.pfdr_mode}"
        assert model.decoder.pfdr_scope == "fine1"
        assert model.decoder.use_pfdr is True
        if args.pfdr_mode == "rep":
            for name in ("dw3", "dwd2", "dw1"):
                assert getattr(p, name).weight.abs().sum().item() == 0.0, f"{name} must be zero-init"
            for name in ("bn3", "bnd2", "bn1"):
                b = getattr(p, name)
                assert b.weight.abs().sum().item() == 0.0 and b.bias.abs().sum().item() == 0.0
        print("  PFDR zero-init OK (aux conv weights=0, aux BN gamma=beta=0)")

    # encoder train mode per encoder_train
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    model.train()
    if args.encoder_train == 'frozen':
        assert enc_trainable == 0 and model.encoder.training is False
        print(f"  encoder frozen OK (trainable={enc_trainable}, training={model.encoder.training})")
    elif args.encoder_train == 'last2':
        s12 = sum(p.numel() for i in (0, 1) for p in model.encoder.layers[i].parameters() if p.requires_grad)
        s34 = sum(p.numel() for i in (2, 3) for p in model.encoder.layers[i].parameters() if p.requires_grad)
        if args.use_biftr and model.biftr is not None:
            # BiFTR deltas live in layers[1] and are train-only; exclude them from
            # the frozen-stage accounting.
            s12 -= sum(p.numel() for p in model.biftr.parameters() if p.requires_grad)
        assert s12 == 0 and s34 > 0
        print(f"  encoder last2 OK (stage1+2 trainable={s12} excl. biftr deltas, stage3+4 trainable={s34})")

    print("[2/6] forward pass ...")
    model.eval()
    torch.manual_seed(2333)  # deterministic inputs for reproducible fold/argmax checks
    with torch.no_grad():
        pre = torch.randn(2, 3, 256, 256).cuda()
        post = torch.randn(2, 3, 256, 256).cuda()
        out = model(pre, post)
    print(f"  output shape = {tuple(out.shape)} (expect (2, 2, 256, 256))")

    if args.use_nscr:
        print("[2b/6] NSCR gamma gradient flow ...")
        m_grad = copy.deepcopy(model)
        m_grad.train()
        out_g = m_grad(pre, post)
        loss = torch.nn.functional.cross_entropy(out_g, torch.randint(0, 2, (2, 256, 256)).cuda())
        loss.backward()
        for name in ("fuse1", "fuse2"):
            m = getattr(m_grad.decoder, name)
            for bn in ("bn_l", "bn_h"):
                if hasattr(m, bn):
                    g = getattr(m, bn).weight.grad
                    assert g is not None and g.abs().sum().item() > 0, f"{name}.{bn} grad must be non-zero"
        print("  NSCR gamma grads non-zero OK")

    if args.use_pbru:
        print("[2c/6] PBRU epoch-0 identity vs use_pbru=0 ...")
        torch.manual_seed(2333)
        m_a = build_model(args, config).cuda()
        torch.manual_seed(2333)
        args_p0 = argparse.Namespace(**{**vars(args), "use_pbru": 0})
        m_b = build_model(args_p0, config).cuda()
        m_a.eval()
        m_b.eval()
        with torch.no_grad():
            oa = m_a(pre, post)
            ob = m_b(pre, post)
        d = (oa - ob).abs().max().item()
        assert d == 0.0, f"PBRU init must equal plain PixelShuffle exactly, got {d}"
        print(f"  PBRU epoch-0 output == plain PixelShuffle output exactly (max_diff={d})")

        print("[2d/6] PBRU gamma gradient flow ...")
        m_grad = copy.deepcopy(model)
        m_grad.train()
        out_g = m_grad(pre, post)
        loss = torch.nn.functional.cross_entropy(out_g, torch.randint(0, 2, (2, 256, 256)).cuda())
        loss.backward()
        for name in ("bn_coarse", "bn_px", "bn_py", "bn_pxy"):
            g = getattr(m_grad.head, name).weight.grad
            assert g is not None and g.abs().sum().item() > 0, f"head.{name} grad must be non-zero"
        print("  PBRU gamma grads non-zero OK")

    if args.use_mpcr:
        print("[2e/6] MPCR epoch-0 identity vs use_mpcr=0 ...")
        torch.manual_seed(2333)
        m_a = build_model(args, config).cuda()
        torch.manual_seed(2333)
        args_m0 = argparse.Namespace(**{**vars(args), "use_mpcr": 0})
        m_b = build_model(args_m0, config).cuda()
        # The g0/g1 Kaiming init inside MPCRPW1x1 consumes RNG BEFORE the head is
        # built, so the two models' HEAD inits differ. The head is not part of the
        # MPCR change: sync it so the epoch-0 check isolates the branches only.
        m_b.head.load_state_dict(m_a.head.state_dict())
        m_a.eval()
        m_b.eval()
        with torch.no_grad():
            oa = m_a(pre, post)
            ob = m_b(pre, post)
        d = (oa - ob).abs().max().item()
        assert d == 0.0, f"MPCR epoch-0 output must equal the Run2 core EXACTLY, got {d}"
        print(f"  MPCR epoch-0 output == Run2 core output exactly (max_diff={d})")

        print("[2f/6] MPCR gradient flow ...")
        m_grad = copy.deepcopy(model)
        m_grad.train()
        out_g = m_grad(pre, post)
        loss = torch.nn.functional.cross_entropy(out_g, torch.randint(0, 2, (2, 256, 256)).cuda())
        loss.backward()
        pw = m_grad.decoder.refine.pw
        for name in ("bn0", "bn1"):
            g = getattr(pw, name).weight.grad
            assert g is not None and g.abs().sum().item() > 0, f"{name} gamma grad must be non-zero"
        # Zero-init branch dynamics: with gamma=0 the grouped conv weights get NO
        # gradient at step 0 (dL/dw = dL/dy * gamma/sigma * x = 0). They start
        # receiving gradients once gamma departs from zero -> verify with a nudge.
        with torch.no_grad():
            pw.bn0.weight.add_(0.01)
            pw.bn1.weight.add_(0.01)
        m_grad.zero_grad()
        out_g2 = m_grad(pre, post)
        loss2 = torch.nn.functional.cross_entropy(out_g2, torch.randint(0, 2, (2, 256, 256)).cuda())
        loss2.backward()
        for name in ("g0", "g1"):
            g = getattr(pw, name).weight.grad
            assert g is not None and g.abs().sum().item() > 0, f"{name} weight grad must be non-zero after gamma nudge"
        print("  MPCR gamma grads non-zero @init + group-conv grads non-zero after gamma nudge: OK")

    if args.use_biftr:
        print("[2g/6] BiFTR epoch-0 identity vs use_biftr=0 ...")
        torch.manual_seed(2333)
        m_a = build_model(args, config).cuda()
        torch.manual_seed(2333)
        args_b0 = argparse.Namespace(**{**vars(args), "use_biftr": 0})
        m_b = build_model(args_b0, config).cuda()
        # zero-init deltas consume no RNG, so all shared weights match exactly.
        m_a.eval()
        m_b.eval()
        with torch.no_grad():
            oa = m_a(pre, post)
            ob = m_b(pre, post)
        d = (oa - ob).abs().max().item()
        assert d == 0.0, f"BiFTR epoch-0 output must equal Run2 EXACTLY, got {d}"
        print(f"  BiFTR epoch-0 output == Run2 output exactly (max_diff={d})")

        print("[2h/6] BiFTR gradient flow ...")
        m_grad = copy.deepcopy(model)
        m_grad.train()
        out_g = m_grad(pre, post)
        loss = torch.nn.functional.cross_entropy(out_g, torch.randint(0, 2, (2, 256, 256)).cuda())
        loss.backward()
        b = m_grad.biftr
        assert b.conv.weight.grad is None, "base Conv must receive no grad"
        g = b.d_out.weight.grad
        assert g is not None and g.abs().sum().item() > 0, "d_out grad must be non-zero"
        if hasattr(b, "d_in"):
            g = b.d_in.weight.grad
            assert g is not None and g.abs().sum().item() > 0, "d_in grad must be non-zero"
        print("  BiFTR grads OK (deltas non-zero @init, base conv grad=None)")

    if args.use_pfdr:
        print("[2i/6] PFDR C0/M1 epoch-0 identity + shared-init identity ...")
        torch.manual_seed(2333)
        m_rep = build_model(args, config).cuda()
        torch.manual_seed(2333)
        args_plain = argparse.Namespace(**{**vars(args), "pfdr_mode": "plain"})
        m_plain = build_model(args_plain, config).cuda()
        # main branch init identical
        pr = dict(m_rep.decoder.prefuse1.named_parameters())
        pl = dict(m_plain.decoder.prefuse1.named_parameters())
        for n in ("dw5.weight", "bn5.weight", "bn5.bias", "alpha"):
            a = pr[n].detach()
            b = pl[n].detach()
            assert torch.equal(a, b), f"PFDR main-branch init differs: {n}"
        # every non-PFDR shared weight identical (encoder/tar/decoder/head)
        sd_rep = m_rep.state_dict()
        sd_pl = m_plain.state_dict()
        for k in sd_rep:
            if "prefuse1" in k:
                continue
            assert torch.equal(sd_rep[k], sd_pl[k]), f"shared weight differs: {k}"
        m_rep.eval()
        m_plain.eval()
        with torch.no_grad():
            o_rep = m_rep(pre, post)
            o_pl = m_plain(pre, post)
        d = (o_rep - o_pl).abs().max().item()
        assert d == 0.0, f"PFDR M1 epoch-0 output must equal C0 EXACTLY, got {d}"
        print(f"  PFDR C0/M1 main-init identical + epoch-0 logits bitwise equal (max_diff={d})")

        if args.pfdr_mode == "rep":
            print("[2j/6] PFDR gradient flow ...")
            m_grad = copy.deepcopy(model)
            m_grad.train()
            out_g = m_grad(pre, post)
            loss = torch.nn.functional.cross_entropy(out_g, torch.randint(0, 2, (2, 256, 256)).cuda())
            loss.backward()
            p = m_grad.decoder.prefuse1
            for name in ("bn3", "bnd2", "bn1"):
                g = getattr(p, name).weight.grad
                assert g is not None and g.abs().sum().item() > 0, f"{name} gamma grad must be non-zero"
            # zero-init branch dynamics: gamma=0 blocks aux conv grads at step 0;
            # they must appear after a gamma nudge (Run8-style check).
            with torch.no_grad():
                for name in ("bn3", "bnd2", "bn1"):
                    getattr(p, name).weight.add_(0.01)
            m_grad.zero_grad()
            out_g2 = m_grad(pre, post)
            loss2 = torch.nn.functional.cross_entropy(out_g2, torch.randint(0, 2, (2, 256, 256)).cuda())
            loss2.backward()
            for name in ("dw3", "dwd2", "dw1"):
                g = getattr(p, name).weight.grad
                assert g is not None and g.abs().sum().item() > 0, f"{name} weight grad must appear after gamma nudge"
            print("  PFDR gamma grads non-zero @init + aux conv grads non-zero after gamma nudge: OK")

    print("[3/6] deploy fold ...")
    deploy = copy.deepcopy(model)
    deploy.switch_to_deploy()
    deploy.eval()
    sd_keys = set(deploy.state_dict().keys())
    has_nscr = any(("bn_l" in k or "bn_h" in k or ".core." in k) for k in sd_keys)
    assert not has_nscr, "deploy graph must not contain bn_l/bn_h/core"
    has_pbru = any(("branch_" in k or "bn_coarse" in k or "bn_px" in k or "bn_py" in k
                    or "bn_pxy" in k or "main_proj" in k) for k in sd_keys)
    assert not has_pbru, "deploy graph must not contain branch_/bn_/main_proj"
    has_mpcr = any(("g0" in k or "g1" in k or "bn0" in k or "bn1" in k
                    or ".core." in k) for k in sd_keys)
    assert not has_mpcr, "deploy graph must not contain g0/g1/bn0/bn1/core"
    has_biftr = any(("d_in" in k or "d_out" in k) for k in sd_keys)
    assert not has_biftr, "deploy graph must not contain d_in/d_out"
    has_pfdr = any(("dw3" in k or "dwd2" in k or "dw1" in k or "bn3" in k or "bnd2" in k
                    or "bn1" in k or "dw5" in k or "bn5" in k) for k in sd_keys)
    assert not has_pfdr, "deploy graph must not contain PFDR train branches"
    if args.use_pfdr:
        keys = sorted(k for k in sd_keys if "prefuse1" in k)
        assert keys == ["decoder.prefuse1.dw.bias", "decoder.prefuse1.dw.weight"], \
            f"deploy prefuse1 must be exactly one DW5, got {keys}"
        assert deploy.decoder.prefuse1.deploy, "PFDR module must be in deploy mode"
    if args.use_biftr:
        # deploy graph must be exactly one downsample Conv (wrapper folded)
        assert deploy.biftr.deploy, "BiFTR wrapper must be in deploy mode"
    with torch.no_grad():
        out_d = deploy(pre, post)
    err = (out - out_d).abs().max().item()
    disagree = (out.argmax(1) != out_d.argmax(1)).float().mean().item()
    deploy_params = sum(p.numel() for p in deploy.parameters())
    print(f"  fold max_abs_error = {err:.3e}, argmax_disagree = {disagree:.3e}, "
          f"deploy params = {deploy_params/1e6:.3f} M")
    print(f"  deploy graph clean (no bn_l/bn_h/core): {not has_nscr}")
    assert err < 2e-4, f"fold error too large: {err}"

    if args.use_nscr:
        # deploy params / FLOPs must be IDENTICAL to the use_nscr=0 model
        args0 = argparse.Namespace(**{**vars(args), "use_nscr": 0})
        model0 = build_model(args0, config).cuda()
        deploy0 = copy.deepcopy(model0)
        deploy0.switch_to_deploy()
        deploy0.eval()
        p0 = sum(p.numel() for p in deploy0.parameters())
        assert p0 == deploy_params, f"deploy params differ: nscr={deploy_params} vs 0={p0}"
        from fvcore.nn import flop_count
        from classification.models.vmamba import selective_scan_flop_jit
        supported = {
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        }
        with torch.no_grad():
            c_n, _ = flop_count(deploy, (pre, post), supported_ops=supported)
            c_0, _ = flop_count(deploy0, (pre, post), supported_ops=supported)
        f_n, f_0 = sum(c_n.values()), sum(c_0.values())
        print(f"  deploy FLOPs nscr={f_n:.4f} vs 0={f_0:.4f} (batch2)")
        assert abs(f_n - f_0) < 0.01, "deploy FLOPs differ"
        print(f"  deploy Params/FLOPs identical to use_nscr=0: OK ({deploy_params/1e6:.3f} M)")
    elif args.use_pbru:
        # deploy params / FLOPs must be IDENTICAL to the use_pbru=0 (plain PixelShuffle) model
        args0 = argparse.Namespace(**{**vars(args), "use_pbru": 0})
        model0 = build_model(args0, config).cuda()
        deploy0 = copy.deepcopy(model0)
        deploy0.switch_to_deploy()
        deploy0.eval()
        p0 = sum(p.numel() for p in deploy0.parameters())
        assert p0 == deploy_params, f"deploy params differ: pbru={deploy_params} vs 0={p0}"
        from fvcore.nn import flop_count
        from classification.models.vmamba import selective_scan_flop_jit
        supported = {
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        }
        with torch.no_grad():
            c_n, _ = flop_count(deploy, (pre, post), supported_ops=supported)
            c_0, _ = flop_count(deploy0, (pre, post), supported_ops=supported)
        f_n, f_0 = sum(c_n.values()), sum(c_0.values())
        print(f"  deploy FLOPs pbru={f_n:.4f} vs 0={f_0:.4f} (batch2)")
        assert abs(f_n - f_0) < 0.01, "deploy FLOPs differ"
        print(f"  deploy Params/FLOPs identical to use_pbru=0: OK ({deploy_params/1e6:.3f} M)")
    elif args.use_mpcr:
        # deploy params / FLOPs must be EXACTLY the Run2 anchor (use_mpcr=0, D=160)
        args0 = argparse.Namespace(**{**vars(args), "use_mpcr": 0})
        model0 = build_model(args0, config).cuda()
        deploy0 = copy.deepcopy(model0)
        deploy0.switch_to_deploy()
        deploy0.eval()
        p0 = sum(p.numel() for p in deploy0.parameters())
        assert p0 == deploy_params, f"deploy params differ: mpcr={deploy_params} vs anchor={p0}"
        from fvcore.nn import flop_count
        from classification.models.vmamba import selective_scan_flop_jit
        supported = {
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        }
        with torch.no_grad():
            c_n, _ = flop_count(deploy, (pre, post), supported_ops=supported)
            c_0, _ = flop_count(deploy0, (pre, post), supported_ops=supported)
        f_n, f_0 = sum(c_n.values()), sum(c_0.values())
        print(f"  deploy FLOPs mpcr={f_n:.4f} vs anchor={f_0:.4f} (batch2)")
        assert f_n == f_0, "deploy FLOPs must be EXACTLY the Run2 anchor"
        print(f"  deploy Params/FLOPs EXACTLY equal to the Run2 anchor: OK ({deploy_params/1e6:.3f} M)")
    elif args.use_biftr:
        # deploy params / FLOPs must be EXACTLY the Run2 anchor (use_biftr=0, last2, D160)
        args0 = argparse.Namespace(**{**vars(args), "use_biftr": 0})
        model0 = build_model(args0, config).cuda()
        deploy0 = copy.deepcopy(model0)
        deploy0.switch_to_deploy()
        deploy0.eval()
        p0 = sum(p.numel() for p in deploy0.parameters())
        assert p0 == deploy_params, f"deploy params differ: biftr={deploy_params} vs anchor={p0}"
        from fvcore.nn import flop_count
        from classification.models.vmamba import selective_scan_flop_jit
        supported = {
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        }
        with torch.no_grad():
            c_n, _ = flop_count(deploy, (pre, post), supported_ops=supported)
            c_0, _ = flop_count(deploy0, (pre, post), supported_ops=supported)
        f_n, f_0 = sum(c_n.values()), sum(c_0.values())
        print(f"  deploy FLOPs biftr={f_n:.4f} vs anchor={f_0:.4f} (batch2)")
        assert f_n == f_0, "deploy FLOPs must be EXACTLY the Run2 anchor"
        print(f"  deploy Params/FLOPs EXACTLY equal to the Run2 anchor: OK ({deploy_params/1e6:.3f} M)")
    elif args.use_pfdr:
        # C0/M1 deploy Params/FLOPs must be IDENTICAL (same single DW5 topology)
        args_p0 = argparse.Namespace(**{**vars(args), "pfdr_mode": "plain"})
        model0 = build_model(args_p0, config).cuda()
        deploy0 = copy.deepcopy(model0)
        deploy0.switch_to_deploy()
        deploy0.eval()
        p0 = sum(p.numel() for p in deploy0.parameters())
        assert p0 == deploy_params, f"deploy params differ: rep={deploy_params} vs plain={p0}"
        from fvcore.nn import flop_count
        from classification.models.vmamba import selective_scan_flop_jit
        supported = {
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        }
        with torch.no_grad():
            c_n, _ = flop_count(deploy, (pre, post), supported_ops=supported)
            c_0, _ = flop_count(deploy0, (pre, post), supported_ops=supported)
        f_n, f_0 = sum(c_n.values()), sum(c_0.values())
        print(f"  deploy FLOPs rep={f_n:.4f} vs plain={f_0:.4f} (batch2)")
        assert f_n == f_0, "deploy FLOPs must be identical between PFDR rep/plain"
        print(f"  C0/M1 deploy Params/FLOPs identical: OK ({deploy_params/1e6:.3f} M)")
        if args.check_anchor:
            # deploy budget vs the Run2 anchor (use_pfdr=0, D=160): must be <=.
            # Only run at D=D* (the budget script finds D*; at D=160 PFDR adds
            # the DW5 cost and legitimately exceeds the anchor).
            args_a = argparse.Namespace(**{**vars(args), "use_pfdr": 0, "decoder_dim": 160,
                                           "pfdr_mode": "plain"})
            model_a = build_model(args_a, config).cuda()
            deploy_a = copy.deepcopy(model_a)
            deploy_a.switch_to_deploy()
            deploy_a.eval()
            pa = sum(p.numel() for p in deploy_a.parameters())
            with torch.no_grad():
                c_a, _ = flop_count(deploy_a, (pre, post), supported_ops=supported)
            f_a = sum(c_a.values())
            assert deploy_params <= pa, f"deploy params exceed anchor: {deploy_params} > {pa}"
            assert f_n <= f_a, f"deploy FLOPs exceed anchor: {f_n} > {f_a}"
            print(f"  deploy <= Run2 anchor (D=160): params {deploy_params/1e6:.3f} <= {pa/1e6:.3f} M, "
                  f"FLOPs {f_n:.4f} <= {f_a:.4f} G: OK")
    else:
        print(f"  deploy params = {deploy_params/1e6:.3f} M")

    print("[4/6] dataloader ...")
    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type='test')
    a, b, l, name = ds[0]
    print(f"  sample: A {a.shape} {a.dtype}, B {b.shape}, label {l.shape} {l.dtype}, name {name}")
    print("  label unique:", np.unique(l))

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
