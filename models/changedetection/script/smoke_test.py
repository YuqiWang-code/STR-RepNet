"""Smoke test for the HAM-CD baseline pipeline (model build + forward + params/FLOPs + dataloader)."""
import os
import sys
import argparse

import numpy as np
import torch

_MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from changedetection.configs.config import get_config
from changedetection.models.MambaBCD import STMambaBCD
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list


def build_model(args, config):
    v = config.MODEL.VSSM
    return STMambaBCD(
        pretrained=args.pretrained_weight_path,
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
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    config = get_config(args)
    print("[1/4] building model ...")
    model = build_model(args, config).cuda()
    params = sum(p.numel() for p in model.parameters())
    print(f"  params = {params/1e6:.3f} M")

    print("[2/4] forward pass ...")
    model.eval()
    with torch.no_grad():
        pre = torch.randn(2, 3, 256, 256).cuda()
        post = torch.randn(2, 3, 256, 256).cuda()
        out = model(pre, post)
    print(f"  output shape = {tuple(out.shape)} (expect (2, 2, 256, 256))")

    print("[3/4] FLOPs ...")
    from fvcore.nn import flop_count
    from classification.models.vmamba import selective_scan_flop_jit
    from changedetection.models.utils import selective_scan_state_flop_jit
    supported = {
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanStateFn": selective_scan_state_flop_jit,
    }
    with torch.no_grad():
        counts, unsup = flop_count(model, (pre, post), supported_ops=supported)
    total = sum(counts.values())
    print(f"  FLOPs = {total:.4f} G  (unsupported={len(unsup)})")

    print("[4/4] dataloader ...")
    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type='test')
    a, b, l, name = ds[0]
    print(f"  sample: A {a.shape} {a.dtype}, B {b.shape}, label {l.shape} {l.dtype}, name {name}")
    print("  label unique:", np.unique(l))

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
