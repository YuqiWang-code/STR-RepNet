"""TAR-DCR structural re-parameterization equivalence tests.

Checks that switch_to_deploy() does not change the FP32 output of every Rep
block and the whole model (max_abs_error must be < 1e-6).
"""
import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F

_MODELS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MODELS_ROOT not in sys.path:
    sys.path.insert(0, _MODELS_ROOT)

from changedetection.models.reparam import RepDW3, RepPW1x1, RepPairFuse1x1
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


def block_tests(device):
    torch.manual_seed(2333)
    C = 160
    H = W = 16
    results = []

    x = torch.randn(2, C, H, W, device=device)
    results.append(check_fold(RepDW3(C, use_aux=True, use_residual=True).to(device), (x,), name="RepDW3"))
    results.append(check_fold(RepPW1x1(C, use_aux=True).to(device), (x,), name="RepPW1x1"))
    results.append(check_fold(RepLocalBlock(C, use_aux=True).to(device), (x,), name="RepLocalBlock"))

    L = torch.randn(2, C, H, W, device=device)
    Hh = torch.randn(2, C, H, W, device=device)
    results.append(check_fold(RepPairFuse1x1(C, use_aux=True).to(device), (L, Hh), name="RepPairFuse1x1"))

    P = torch.randn(2, 96, H, W, device=device)
    Q = torch.randn(2, 96, H, W, device=device)
    results.append(check_fold(TemporalRep1x1(96, C, use_aux=True).to(device), (P, Q), name="TemporalRep1x1"))
    results.append(check_fold(TARStage(96, C, use_temporal_aux=True, use_dcr_aux=True).to(device), (P, Q), name="TARStage"))

    return all(results)


def whole_model_test(device, cfg, pretrained, rep_mode="full"):
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
        pretrained=pretrained, rep_mode=rep_mode,
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

    # verify encoder is frozen
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert enc_trainable == 0, f"encoder has {enc_trainable} trainable params"
    model.train()
    assert model.encoder.training is False, "encoder should stay eval after model.train()"
    print(f"[OK] encoder frozen (trainable={enc_trainable}, training={model.encoder.training})")

    model.eval()
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
    ok = err < 1e-4
    print(f"[{'OK' if ok else 'FAIL'}] STRRepNet whole model ({rep_mode}): max_abs_error={err:.3e}")
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

    print("=== block-level fold tests ===")
    b_ok = block_tests(device)

    print("=== whole-model fold test ===")
    w_ok = whole_model_test(device, args.cfg, args.pretrained_weight_path, args.rep_mode)

    print(f"\n{'ALL PASSED' if (b_ok and w_ok) else 'SOME FAILED'}")
    sys.exit(0 if (b_ok and w_ok) else 1)


if __name__ == "__main__":
    main()
