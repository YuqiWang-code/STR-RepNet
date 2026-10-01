"""Run10 PFDR budget search: find the largest decoder width D* such that the
C0/M1 deploy graph stays within the Run2 anchor budget.

    anchor     = Run2 deploy graph (use_pfdr=0, bilinear, D=160)
    candidates = PFDR deploy graph (use_pfdr=1, pfdr_mode in {plain,rep}), D=160..152

PFDR deploy adds ONE depthwise 5x5 (D*25 + D params) on the 64x64 fine lateral,
so the budget must be reclaimed by lowering the decoder width D. Per the
pre-registered doc (Run10 §4.2):
  - D* = largest D with deploy Params <= anchor Params AND deploy FLOPs <= anchor FLOPs;
  - pre-training gate: if D* < 158 the run is STOPPED (width loss would confound
    the spatial-support gain), exit code 2;
  - the C0 (plain) and M1 (rep) deploy graphs must have IDENTICAL Params/FLOPs
    at D* (same single DW5 topology), verified here as a P0 check.

Usage:
    python analyse/search_run10_pfdr_budget.py --cfg <vssm yaml> \
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


def build_deploy(args, config, dim, use_pfdr, pfdr_mode="plain"):
    v = config.MODEL.VSSM
    model = STRRepNet(
        pretrained=args.pretrained_weight_path, rep_mode="full", dim=dim,
        use_residual=True, encoder_train="last2",
        use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode="bilinear", use_pbru=False, pbru_upscale=4,
        use_mpcr=False, mpcr_mode="multi2", mpcr_groups=4,
        use_biftr=False, biftr_mode="bi",
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

    print("[ANCHOR] Run2 deploy graph (use_pfdr=0, bilinear, D=160, last2)")
    anchor = build_deploy(args, config, 160, use_pfdr=False)
    ap_, af_, au_ = measure(anchor)
    print(f"  anchor Params = {ap_/1e6:.6f} M, FLOPs = {af_:.4f} G (unsupported_ops={au_})")
    del anchor
    torch.cuda.empty_cache()

    print("[CANDIDATES] PFDR deploy graph (pfdr_mode=plain), D from 160 down to 152")
    dstar = None
    for D in range(160, 151, -1):
        m = build_deploy(args, config, D, use_pfdr=True, pfdr_mode="plain")
        p, f, u = measure(m)
        dp, df = p - ap_, f - af_
        fit = (p <= ap_) and (f <= af_)
        print(f"  D={D:3d}: Params={p/1e6:.6f} M (d={dp:+.0f}), FLOPs={f:.4f} G (d={df:+.4f}) "
              f"-> {'FIT' if fit else 'over'}")
        del m
        torch.cuda.empty_cache()
        if fit:
            dstar = D
            break

    if dstar is None:
        print("[D-STAR] NOT FOUND in [160, 152] -> PFDR cannot fit the anchor budget; "
              "training FORBIDDEN")
        sys.exit(1)

    # P0: C0 (plain) and M1 (rep) deploy graphs must be IDENTICAL at D*.
    print(f"[P0] C0/M1 deploy equality audit at D*={dstar}")
    mc = build_deploy(args, config, dstar, use_pfdr=True, pfdr_mode="plain")
    pc, fc, uc = measure(mc)
    mm = build_deploy(args, config, dstar, use_pfdr=True, pfdr_mode="rep")
    pm, fm, um = measure(mm)
    print(f"  plain: Params={pc/1e6:.6f} M, FLOPs={fc:.4f} G")
    print(f"  rep  : Params={pm/1e6:.6f} M, FLOPs={fm:.4f} G")
    if pc != pm or fc != fm:
        print("[P0-BUG] C0/M1 deploy graphs differ -> PFDR rep does not fully fold; "
              "training FORBIDDEN")
        sys.exit(3)
    print("[P0-OK] C0/M1 deploy Params/FLOPs identical")

    # Pre-registered gate: D* >= 158 required (doc §4.2).
    if dstar < 158:
        print(f"[GATE-FAIL] D*={dstar} < 158: width loss would confound the spatial-support "
              "gain -> Run10 STOPPED before training (doc §4.2)")
        sys.exit(2)

    print(f"[D-STAR] {dstar}  (deploy <= Run2 anchor; C0 == M1 exactly)")
    print("[OK] Run10 budget verification passed")


if __name__ == "__main__":
    main()
