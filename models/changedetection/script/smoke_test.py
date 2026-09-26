"""Smoke test for STR-RepNet Clean TAR-DCR (build + forward + params/FLOPs + deploy + dataloader).

Run4 additions: --encoder_train / --use_boundary_aux checks (shapes, aux-remove
bit-exactness, gradient flow, stage1/2 frozen).
"""
import copy
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F

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
        use_residual=bool(args.use_residual), use_edge=bool(args.use_edge),
        encoder_train=args.encoder_train, use_boundary_aux=bool(args.use_boundary_aux),
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
    ap.add_argument('--use_edge', type=int, default=0)
    ap.add_argument('--use_boundary_aux', type=int, default=0)
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
    if args.use_boundary_aux:
        assert model.boundary_head is not None
        bh = sum(p.numel() for p in model.boundary_head.parameters())
        print(f"  boundary head params = {bh} (expect 161)")

    # verify encoder train mode per encoder_train
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

    if args.use_boundary_aux:
        print("[2b/6] boundary aux forward ...")
        with torch.no_grad():
            out_b, boundary = model(pre, post, return_aux=True)
        print(f"  boundary shape = {tuple(boundary.shape)} (expect (2, 1, 256, 256))")
        assert tuple(boundary.shape) == (2, 1, 256, 256)
        assert (out - out_b).abs().max().item() == 0.0

        print("[2c/6] boundary aux removal (bit-exact main path) ...")
        model_nb = copy.deepcopy(model)
        model_nb.remove_boundary_aux()
        with torch.no_grad():
            out_nb = model_nb(pre, post)
        err_rm = (out - out_nb).abs().max().item()
        print(f"  max_abs_error(main, remove-aux) = {err_rm} (expect 0.0)")
        assert err_rm == 0.0

        print("[2d/6] boundary aux gradient flow ...")
        # run the gradient test on a deepcopy so the main model's BN running stats
        # stay pristine for the [3/6] fold comparison.
        m_grad = copy.deepcopy(model)
        m_grad.train()
        out_b, boundary = m_grad(pre, post, return_aux=True)
        target = (torch.rand(2, 1, 256, 256).cuda() > 0.5).float()
        loss = F.binary_cross_entropy_with_logits(boundary, target)
        loss.backward()
        assert m_grad.boundary_head.weight.grad is not None
        assert m_grad.boundary_head.weight.grad.abs().sum().item() > 0
        # zero-init head: step-1 gradient on the shared feature d1 is 0 BY DESIGN
        # (head learns first). Emulate one SGD step on the head, then the 2nd
        # step must propagate gradients into decoder and stage3+4 encoder.
        with torch.no_grad():
            m_grad.boundary_head.weight -= 0.1 * m_grad.boundary_head.weight.grad
            m_grad.boundary_head.bias -= 0.1 * m_grad.boundary_head.bias.grad
        m_grad.zero_grad()
        out_b, boundary = m_grad(pre, post, return_aux=True)
        F.binary_cross_entropy_with_logits(boundary, target).backward()
        dec_grad = sum(p.grad.abs().sum().item() for p in m_grad.decoder.parameters()
                       if p.grad is not None)
        s34_grad = sum(p.grad.abs().sum().item()
                       for i in (2, 3) for p in m_grad.encoder.layers[i].parameters()
                       if p.grad is not None)
        s12_grad = sum(1 for i in (0, 1) for p in m_grad.encoder.layers[i].parameters()
                       if p.grad is not None)
        print(f"  boundary weight grad OK; after 1 head step: decoder|grad|={dec_grad:.3e}, "
              f"stage3+4|grad|={s34_grad:.3e}, stage1+2 grads={s12_grad} (expect 0)")
        assert dec_grad > 0 and s34_grad > 0
        if args.encoder_train == 'last2':
            assert s12_grad == 0

    print("[3/6] deploy fold ...")
    deploy = copy.deepcopy(model)
    deploy.switch_to_deploy()
    deploy.eval()
    assert getattr(deploy, "boundary_head", None) is None
    with torch.no_grad():
        out_d = deploy(pre, post)
    err = (out - out_d).abs().max().item()
    deploy_params = sum(p.numel() for p in deploy.parameters())
    print(f"  fold max_abs_error = {err:.3e}, deploy params = {deploy_params/1e6:.3f} M")
    assert err < 1e-4, f"fold error too large: {err}"

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
