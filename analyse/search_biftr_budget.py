"""Run9 BiFTR budget equality audit: C0/M1 deploy must equal the Run2 anchor EXACTLY.

    anchor = Run2 deploy graph (last2, D=160, bilinear, BiFTR=0)
    C0     = BiFTR post deploy graph
    M1     = BiFTR bi   deploy graph

Hard requirement: delta_params == 0 and delta_flops == 0 for both C0 and M1
(raw integer counts; "approximately equal" is not acceptable). Any non-zero
delta is a P0 implementation bug and training must NOT start.

Usage:
    python analyse/search_biftr_budget.py --cfg <vssm yaml> \
        --pretrained_weight_path <vssm pth> [--gpu 0]
"""
import argparse
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet


def build_deploy(args, config, use_biftr, biftr_mode):
    v = config.MODEL.VSSM
    model = STRRepNet(
        pretrained=args.pretrained_weight_path, rep_mode="full", dim=160,
        use_residual=True, encoder_train="last2",
        use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode="bilinear", use_pbru=False, pbru_upscale=4,
        use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4,
        use_biftr=use_biftr, biftr_mode=biftr_mode,
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

    print("[ANCHOR] Run2 deploy graph (last2, D=160, bilinear, BiFTR=0)")
    anchor = build_deploy(args, config, use_biftr=False, biftr_mode="bi")
    ap_, af_, au_ = measure(anchor)
    print(f"  anchor Params = {ap_} ({ap_/1e6:.6f} M), FLOPs = {af_:.4f} G (unsupported={au_})")
    del anchor
    torch.cuda.empty_cache()

    ok = True
    for tag, use_biftr, mode in [("C0_FTR_Post", True, "post"), ("M1_BiFTR", True, "bi")]:
        m = build_deploy(args, config, use_biftr=use_biftr, biftr_mode=mode)
        p, f, u = measure(m)
        dp, df = p - ap_, f - af_
        print(f"[{tag}] Params = {p} (d={dp:+d}), FLOPs = {f:.4f} G (d={df:+.4f}), unsupported={u}")
        if dp != 0 or df != 0:
            print(f"[P0-BUG] {tag} deploy differs from anchor -> training FORBIDDEN")
            ok = False
        del m
        torch.cuda.empty_cache()

    if ok:
        print("[OK] Run9 budget equality audit passed: C0/M1 deploy == Run2 anchor EXACTLY")
        sys.exit(0)
    sys.exit(2)


if __name__ == "__main__":
    main()
