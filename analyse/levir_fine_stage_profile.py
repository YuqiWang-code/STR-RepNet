"""Run8 MPCR-Fine stage profile (mechanism-only, read-only; no checkpoint selection).

Hooks five fine-stage points:
    block1 output (d1 pre-refine)
    refine.dw output (pre-SiLU)   + its SiLU
    refine.pw output (pre-SiLU)
    refine final output (decoder d1)
and reports per stage: changed/unchanged centroid distance (raw + normalized),
per-channel Fisher (top-10), diagonal-LDA AUROC (fit first half / held-out second
half), channel-covariance effective rank (entropy) and mean |inter-channel corr|.

Purpose: check whether MPCR changes the final channel representation itself
(separation / diversity), not just the classification threshold. Run on the Run2
anchor ckpt (baseline) and on the M1 ckpt after training.

Usage:
    python analyse/levir_fine_stage_profile.py --cfg <vssm yaml> \
        --checkpoint <pth> --dataset_root /share_datasets/CD/LEVIR-CD-256 \
        --test_list .../list/test.txt --output_dir <dir> \
        [--use_mpcr 0] [--mpcr_mode multi2] [--max_samples 200] [--gpu 0]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = os.path.join(_ROOT, "models")
if _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)

from changedetection.configs.config import get_config
from changedetection.models.STRRepNet import STRRepNet
from changedetection.datasets.make_data_loader import ChangeDetectionDataset, read_list
from levir_stage_discriminability import roc_auc, gt64_from_label

STAGES = [
    ("block1", ["block1"], "none"),
    ("refine_dw", ["refine", "dw"], "none"),
    ("refine_dw_silu", ["refine", "dw"], "silu"),
    ("refine_pw", ["refine", "pw"], "none"),
    ("refine_out", [], "none"),  # decoder output (post final SiLU)
]


def build_model(args, config):
    v = config.MODEL.VSSM
    return STRRepNet(
        pretrained=None, rep_mode="full", dim=160, use_residual=True,
        encoder_train="last2", use_botr=False, use_nscr=False, nscr_scope="high2",
        head_mode="bilinear", use_pbru=False, pbru_upscale=4,
        use_mpcr=bool(args.use_mpcr), mpcr_mode=args.mpcr_mode, mpcr_groups=4,
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
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--test_list", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--use_mpcr", type=int, default=0)
    ap.add_argument("--mpcr_mode", type=str, default="multi2", choices=["same2", "multi2"])
    ap.add_argument("--max_samples", type=int, default=200)
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

    model = build_model(args, config).cuda()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()
    print("loaded", args.checkpoint)

    captures = {}
    hooks = []
    for key, path, transform in STAGES:
        mod = model.decoder
        for part in path:
            mod = getattr(mod, part)
        def make_hook(k, t):
            def h(m, i, o):
                v = o.detach()
                if t == "silu":
                    v = F.silu(v)
                captures[k] = v
            return h
        hooks.append(mod.register_forward_hook(make_hook(key, transform)))

    ds = ChangeDetectionDataset(args.dataset_root, read_list(args.test_list), 256, type="test")
    n = min(len(ds), args.max_samples)
    n_fit = n // 2

    D = 160
    def forward_sample(i):
        a, b, label, _ = ds[i]
        a = torch.from_numpy(a).unsqueeze(0).cuda().float()
        b = torch.from_numpy(b).unsqueeze(0).cuda().float()
        with torch.no_grad():
            model(a, b)
        chg, valid = gt64_from_label(label)
        return {k: captures[k] for k, _, _ in STAGES}, chg, valid

    # fit stats per stage
    fits = {k: {"s0": torch.zeros(D, device="cuda", dtype=torch.float64),
                "s1": torch.zeros(D, device="cuda", dtype=torch.float64),
                "q0": torch.zeros(D, device="cuda", dtype=torch.float64),
                "q1": torch.zeros(D, device="cuda", dtype=torch.float64),
                "G": torch.zeros(D, D, device="cuda", dtype=torch.float64),
                "n0": 0, "n1": 0} for k, _, _ in STAGES}
    for i in range(n_fit):
        outs, chg, valid = forward_sample(i)
        y = chg[valid]
        m = (y == 1)
        for k, _, _ in STAGES:
            f = outs[k][0].permute(1, 2, 0).reshape(-1, D)[valid.reshape(-1)].double()
            st = fits[k]
            st["n1"] += int(m.sum().item())
            st["n0"] += int((~m).sum().item())
            st["s1"] += f[m].sum(0)
            st["s0"] += f[~m].sum(0)
            st["q1"] += (f[m] ** 2).sum(0)
            st["q0"] += (f[~m] ** 2).sum(0)
            st["G"] += f.T @ f

    # scoring (held-out second half + in-sample first half)
    def score_split(lo, hi):
        acc = {k: {"s": [], "y": []} for k, _, _ in STAGES}
        for i in range(lo, hi):
            outs, chg, valid = forward_sample(i)
            y = chg[valid].bool()
            for k, _, _ in STAGES:
                f = outs[k][0].permute(1, 2, 0).reshape(-1, D)[valid.reshape(-1)]
                acc[k]["s"].append((f.double() @ w_lda[k]))
                acc[k]["y"].append(y)
        return {k: roc_auc(torch.cat(acc[k]["s"]), torch.cat(acc[k]["y"])) for k, _, _ in STAGES}

    report = {}
    w_lda = {}
    for k, _, _ in STAGES:
        st = fits[k]
        mu0 = st["s0"] / st["n0"]
        mu1 = st["s1"] / st["n1"]
        var0 = st["q0"] / st["n0"] - mu0 ** 2
        var1 = st["q1"] / st["n1"] - mu1 ** 2
        den = var0 + var1 + 1e-6
        w_lda[k] = (mu1 - mu0) / den
        fisher = ((mu1 - mu0) ** 2) / den
        topk = torch.topk(fisher, 10)
        # pooled covariance over all valid pixels (fit half)
        tot = st["n0"] + st["n1"]
        mu_all = (st["s0"] + st["s1"]) / tot
        cov = st["G"] / tot - torch.outer(mu_all, mu_all)
        ev = torch.linalg.eigvalsh(cov)
        ev = torch.clamp(ev, min=1e-12)
        p = ev / ev.sum()
        ent = -(p * torch.log(p + 1e-12)).sum()
        eff_rank = float(torch.exp(ent).item())
        dg = torch.sqrt(torch.diag(cov) + 1e-12)
        corr = cov / torch.outer(dg, dg)
        off = corr[~torch.eye(D, dtype=torch.bool, device=cov.device)]
        mean_abs_corr = float(off.abs().mean().item())
        report[k] = {
            "centroid_l2": float((mu1 - mu0).norm().item()),
            "centroid_l2_norm": float(((mu1 - mu0).norm() / (mu1.norm() + mu0.norm() + 1e-12)).item()),
            "fisher_mean": float(fisher.mean().item()),
            "fisher_top10": [{"ch": int(c), "fisher": float(v)} for c, v in zip(topk.indices, topk.values)],
            "eff_rank": round(eff_rank, 2),
            "mean_abs_corr": round(mean_abs_corr, 4),
        }

    auc_ho = score_split(n_fit, n)
    auc_is = score_split(0, n_fit)
    for k, _, _ in STAGES:
        report[k]["diag_lda_auc_ho"] = round(auc_ho[k], 4)
        report[k]["diag_lda_auc_is"] = round(auc_is[k], 4)

    for h_ in hooks:
        h_.remove()

    out = {"checkpoint": args.checkpoint, "n_pairs": n, "n_fit": n_fit,
           "use_mpcr": args.use_mpcr, "mpcr_mode": args.mpcr_mode, "stages": report}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fine_stage_profile.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
