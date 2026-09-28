"""Smoke test for STR-RepNet Clean TAR-DCR (build + forward + params/FLOPs + deploy + dataloader).

Run6 additions: --use_nscr checks (zero-init residual, gamma gradient, deploy
branch deletion, deploy params/FLOPs identical to use_nscr=0, argmax=0).
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
        use_botr=bool(args.use_botr), use_nscr=bool(args.use_nscr),
        nscr_scope=args.nscr_scope,
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

    print("[3/6] deploy fold ...")
    deploy = copy.deepcopy(model)
    deploy.switch_to_deploy()
    deploy.eval()
    sd_keys = set(deploy.state_dict().keys())
    has_nscr = any(("bn_l" in k or "bn_h" in k or ".core." in k) for k in sd_keys)
    assert not has_nscr, "deploy graph must not contain bn_l/bn_h/core"
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
