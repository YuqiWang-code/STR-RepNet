"""Smoke test for STR-RepNet Clean TAR-DCR (build + forward + params/FLOPs + deploy + dataloader).

Run5 additions: --encoder_train / --use_botr checks (reverse-branch grad flow,
deploy graph without proj_r/bn_r, deploy params identical with/without BOTR).
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
        use_residual=bool(args.use_residual), encoder_train=args.encoder_train,
        use_botr=bool(args.use_botr),
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
    ap.add_argument('--rep_mode', type=str, default='full', choices=['plain', 'tar', 'full'])
    ap.add_argument('--encoder_train', type=str, default='frozen', choices=['frozen', 'last2', 'full'])
    ap.add_argument('--use_residual', type=int, default=1)
    ap.add_argument('--use_botr', type=int, default=0)
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
    if args.use_botr:
        assert all(hasattr(m, "proj_r") for m in model.tar.modules()
                   if m.__class__.__name__ == "TemporalRep1x1")

    # encoder train mode per encoder_train
    enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    model.train()
    if args.encoder_train == 'frozen':
        assert enc_trainable == 0 and model.encoder.training is False
        print(f"  encoder frozen OK (trainable={enc_trainable}, training={model.encoder.training})")
    elif args.encoder_train == 'last2':
        s12 = sum(p.numel() for i in (0, 1) for p in model.encoder.layers[i].parameters() if p.requires_grad)
        s34 = sum(p.numel() for i in (2, 3) for p in model.encoder.layers[i].parameters() if p.requires_grad)
        assert s12 == 0 and s34 > 0
        print(f"  encoder last2 OK (stage1+2 trainable={s12}, stage3+4 trainable={s34})")

    print("[2/6] forward pass ...")
    model.eval()
    with torch.no_grad():
        pre = torch.randn(2, 3, 256, 256).cuda()
        post = torch.randn(2, 3, 256, 256).cuda()
        out = model(pre, post)
    print(f"  output shape = {tuple(out.shape)} (expect (2, 2, 256, 256))")

    if args.use_botr:
        print("[2b/6] BOTR reverse-branch gradient flow ...")
        m_grad = copy.deepcopy(model)
        m_grad.train()
        out_g = m_grad(pre, post)
        loss = torch.nn.functional.cross_entropy(out_g, torch.randint(0, 2, (2, 256, 256)).cuda())
        loss.backward()
        proj_r_grads = [m.proj_r.weight.grad for m in m_grad.tar.modules()
                        if m.__class__.__name__ == "TemporalRep1x1"]
        assert all(g is not None for g in proj_r_grads), "proj_r grad must be non-None"
        gr = sum(g.abs().sum().item() for g in proj_r_grads)
        print(f"  proj_r grad sum = {gr:.3e} (expect > 0)")
        assert gr > 0
        if args.encoder_train == 'last2':
            s12_grad = sum(1 for i in (0, 1) for p in m_grad.encoder.layers[i].parameters()
                           if p.grad is not None)
            s34_grad = sum(p.grad.abs().sum().item()
                           for i in (2, 3) for p in m_grad.encoder.layers[i].parameters()
                           if p.grad is not None)
            assert s12_grad == 0 and s34_grad > 0
            print(f"  stage1+2 grads={s12_grad} (expect 0), stage3+4|grad|={s34_grad:.3e} (expect > 0)")

    print("[3/6] deploy fold ...")
    deploy = copy.deepcopy(model)
    deploy.switch_to_deploy()
    deploy.eval()
    has_r = any(hasattr(m, "proj_r") or hasattr(m, "bn_r") for m in deploy.modules())
    assert not has_r, "deploy graph must not contain proj_r / bn_r"
    with torch.no_grad():
        out_d = deploy(pre, post)
    err = (out - out_d).abs().max().item()
    deploy_params = sum(p.numel() for p in deploy.parameters())
    print(f"  fold max_abs_error = {err:.3e}, deploy params = {deploy_params/1e6:.3f} M")
    print(f"  deploy graph clean (no proj_r/bn_r): {not has_r}")
    assert err < 2e-4, f"fold error too large: {err}"

    print("[4/6] FLOPs ...")
    from fvcore.nn import flop_count
    from classification.models.vmamba import selective_scan_flop_jit
    supported = {
        "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
        "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
    }
    with torch.no_grad():
        counts, unsup = flop_count(deploy, (pre, post), supported_ops=supported)
    total_flops = sum(counts.values())
    print(f"  deploy FLOPs = {total_flops:.4f} G  (unsupported={len(unsup)})")

    print("[5/6] dataloader ...")
    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type='test')
    a, b, l, name = ds[0]
    print(f"  sample: A {a.shape} {a.dtype}, B {b.shape}, label {l.shape} {l.dtype}, name {name}")
    print("  label unique:", np.unique(l))

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
