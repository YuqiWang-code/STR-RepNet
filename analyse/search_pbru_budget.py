"""Run7 PBRU budget search: machine-search the largest decoder width D* such that
the PBRU (PixelShuffle-head) deploy graph stays within the Run2 D=160 bilinear
deploy budget — BOTH Params and FLOPs, both measured fresh on this machine.

    anchor     = Run2 deploy graph (head_mode=bilinear, D=160)
    candidates = PBRU deploy graph (head_mode=pixelshuffle, D in {160,...,48})
    D*         = max D with Params <= anchor_params AND FLOPs <= anchor_flops

Measurement protocol (identical to train.py / smoke_test.py):
    fvcore flop_count, batch 1, 256x256, mamba selective-scan handlers,
    sum(counts.values()) — PixelShuffle is a zero-FLOP permutation.

Usage:
    python analyse/search_pbru_budget.py --cfg <vssm yaml> \
        --pretrained_weight_path <vssm pth> [--gpu 0]
"""
import argparse
import copy
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet


def build_deploy(args, config, dim, head_mode, use_pbru):
    v = config.MODEL.VSSM
    model = STRRepNet(
        pretrained=args.pretrained_weight_path, rep_mode="full", dim=dim,
        use_residual=True, encoder_train="frozen",
        use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode=head_mode, use_pbru=use_pbru, pbru_upscale=4,
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
    ).cuda()
    model.switch_to_deploy()
    model.eval()
    return model


def measure(model, size=256):
    from fvcore.nn import flop_count
    from classification.models.vmamba import selective_scan_flop_jit

    supported_ops = {
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
    }
    params = sum(p.numel() for p in model.parameters())
    pre = torch.randn(1, 3, size, size).cuda()
    post = torch.randn(1, 3, size, size).cuda()
    with torch.no_grad():
        counts, unsupported = flop_count(model, (pre, post), supported_ops=supported_ops)
    flops = sum(counts.values())
    return params, flops, len(unsupported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--pretrained_weight_path", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    for attr, val in [("opts", None), ("batch_size", 16), ("data_path", ""), ("zip", False),
                      ("cache_mode", None), ("pretrained", ""), ("resume", None),
                      ("accumulation_steps", None), ("use_checkpoint", False),
                      ("disable_amp", False), ("output", ""), ("tag", "")]:
        if not hasattr(args, attr):
            setattr(args, attr, val)

    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    config = get_config(args)

    print("[ANCHOR] Run2 deploy graph (head_mode=bilinear, D=160)")
    anchor = build_deploy(args, config, 160, "bilinear", False)
    ap_, af_, au_ = measure(anchor)
    print(f"  anchor Params = {ap_/1e6:.6f} M, FLOPs = {af_:.4f} G (unsupported_ops={au_})")
    del anchor
    torch.cuda.empty_cache()

    print("[SEARCH] PBRU deploy graph (head_mode=pixelshuffle), D from 160 down to 48")
    dstar = None
    for D in range(160, 47, -1):
        m = build_deploy(args, config, D, "pixelshuffle", True)
        p, f, u = measure(m)
        fit = (p <= ap_) and (f <= af_)
        print(f"  D={D:3d}: Params={p/1e6:.6f} M (d={p-ap_:+.0f}), "
              f"FLOPs={f:.4f} G (d={f-af_:+.4f}) -> {'FIT' if fit else 'over'}")
        del m
        torch.cuda.empty_cache()
        if fit:
            dstar = D
            break

    if dstar is None:
        print("[D-STAR] NOT FOUND in [160, 48]")
        sys.exit(1)
    print(f"[D-STAR] {dstar}  (Params <= {ap_/1e6:.6f} M AND FLOPs <= {af_:.4f} G)")
    print("[D-STAR] USE decoder_dim", dstar)


if __name__ == "__main__":
    main()
